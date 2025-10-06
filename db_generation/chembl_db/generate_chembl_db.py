#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end ChEMBL → CCD → UniProt/Pfam pipeline (CLI) **with structured logging**.

This generates a database of chembl ligands with their interactions, includes corresponding PDB IDs and associated information about binding targets. 
It performs:

1) Query a local ChEMBL SQLite database for bioactivity rows (ligand, SMILES,
   InChIKey, protein UniProt, pChEMBL, comment) using an explicit SQL query.
2) Execute the CCD downloader/parser script to obtain a ChemComp map CSV
   (columns including `chemcomp_id` and `inchikey`).
3) Left-join the ChEMBL results with the CCD map on `inchikey` to add `pdb_id`
   (value taken from CCD `chemcomp_id`).
4) Resolve Pfam IDs per protein using the UniProt REST API via `uniprot_pfam`
   helper functions, and left-join them on `protein` → `uniprot` to bring
   `pfam_ids`.
5) Write the final table to CSV.

Example
-------
python pipeline_chembl_cli.py \
  --chembl-sqlite /path/to/chembl_35.db \
  --ccd-script ./download_parse_ccd.py \
  --workdir temp \
  --ccd-csv temp/ccd.csv \
  --out-csv chembl_db.csv \
  --pfam-workers 24 --pfam-timeout 45 --pfam-retries 3 \
  --log-dir logs --log-level INFO --no-console

Notes
-----
- Requires: pandas, requests, tqdm, and an importable module `uniprot_pfam`
  exposing `get_pfam_for_uniprot_ids` and `mapping_to_dataframe` (as provided
  earlier in this project).
- The CCD step *executes* an external script (kept decoupled). If you already
  have the CCD CSV, pass `--skip-ccd`.
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Dict, Set, Optional

import argparse
import pandas as pd

from logger import setup_logging
import db_generation.chembl_db.uniprot_pfam as up
from db_generation.chembl_db.download_parse_ccd import run_ccd_download_parse

SQL_QUERY = """
SELECT DISTINCT
    md.chembl_id                               AS ligand_id,
    cs.canonical_smiles                        AS smiles,
    UPPER(cs.standard_inchi_key)               AS inchikey,
    csq.accession                              AS protein,
    act.pchembl_value                          AS pchembl,
    act.activity_comment                       AS comment
FROM activities              AS act
JOIN assays                  AS a    ON act.assay_id = a.assay_id
JOIN molecule_dictionary     AS md   ON act.molregno = md.molregno
JOIN compound_structures     AS cs   ON md.molregno = cs.molregno
JOIN target_dictionary       AS td   ON a.tid = td.tid
JOIN target_components       AS tc   ON td.tid = tc.tid
JOIN component_sequences     AS csq  ON tc.component_id = csq.component_id
WHERE a.assay_type = 'B'
  AND td.target_type = 'SINGLE PROTEIN'
  AND (
        act.pchembl_value IS NOT NULL
        OR (
            act.pchembl_value IS NULL
            AND act.activity_comment IN ('Active', 'active', 'inactive', 'Not Active')
        )
      )
  AND cs.canonical_smiles IS NOT NULL
  AND cs.standard_inchi_key IS NOT NULL;
"""


def run_cmd_stream(cmd: list[str], logger: logging.Logger, cwd: Optional[Path] = None) -> int:
    """
    Run a subprocess and stream stdout/stderr to the logger in real time.

    Parameters
    ----------
    cmd : list of str
        Command and arguments. Do not pass a single shell string.
    logger : logging.Logger
        Logger to write the live output.
    cwd : Optional[pathlib.Path], optional
        Working directory for the child process, by default None.

    Returns
    -------
    int
        Process return code (0 means success).
    """
    logger.info("$ %s", " ".join(map(str, cmd)))
    if cwd:
        logger.info("  (cwd: %s)", cwd)

    start = time.time()
    with subprocess.Popen(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(cwd) if cwd else None,
    ) as p:
        assert p.stdout is not None
        for line in p.stdout:
            logger.info(line.rstrip("\n"))
        rc = p.wait()
    logger.info("=> Return code: %s (%.1f s)", rc, time.time() - start)
    return rc


# ========================== I/O helpers ==========================

def load_chembl_results(db_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Run the predefined SQL query against a ChEMBL SQLite database.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the ChEMBL SQLite database (e.g., chembl_35.db).
    logger : logging.Logger
        Logger for status and counts.

    Returns
    -------
    pandas.DataFrame
        Table with columns: ligand_id, smiles, inchikey, protein, pchembl, comment.
    """
    logger.info("[1/5] Querying ChEMBL SQLite…")
    with sqlite3.connect(str(db_path)) as con:
        df = pd.read_sql_query(SQL_QUERY, con)
    if "inchikey" in df.columns:
        df["inchikey"] = df["inchikey"].astype(str).str.strip().str.upper()
    
    comment_mapping = {
    "active": "Active",
    "Active": "Active",
    "inactive": "Inactive",
    "Not Active": "Inactive"
    }

    df["comment"] = df["comment"].replace(comment_mapping)

    logger.info("    rows: %s", f"{len(df):,}")
    return df

def run_ccd_script(
    workdir: Path,
    ccd_csv: Path,
    logger: logging.Logger,
    *,
    out_parquet: Optional[Path] = None,
    max_records: Optional[int] = None,
    batch_size: int = 10000,
) -> int:
    """
    Run the CCD downloader/parser pipeline as an importable function (no subprocess).

    If the target CSV (or Parquet) already exists, they are removed before rebuilding,
    ensuring a clean and non-duplicated output.

    Parameters
    ----------
    workdir : pathlib.Path
        Working directory used by the CCD step.
    ccd_csv : pathlib.Path
        Target CSV path for CCD output (chemcomp_id, inchikey).
    logger : logging.Logger
        Logger for progress and status messages.
    out_parquet : pathlib.Path or None, optional
        Optional Parquet output path.
    max_records : int or None, optional
        Limit parsing to N molecules (debug).
    batch_size : int, default=10000
        Batch size for incremental writes.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    from db_generation.chembl_db.download_parse_ccd import run_ccd_download_parse

    try:
        workdir.mkdir(parents=True, exist_ok=True)
        ccd_csv.parent.mkdir(parents=True, exist_ok=True)

        # Always delete old outputs to avoid duplication
        if ccd_csv.exists():
            try:
                ccd_csv.unlink()
                logger.info("Removed existing CCD CSV before regeneration: %s", ccd_csv)
            except Exception as e:
                logger.warning("Could not remove existing CCD CSV (%s): %s", ccd_csv, e)
        if out_parquet and out_parquet.exists():
            try:
                out_parquet.unlink()
                logger.info("Removed existing CCD Parquet before regeneration: %s", out_parquet)
            except Exception as e:
                logger.warning("Could not remove existing CCD Parquet (%s): %s", out_parquet, e)

        # Run the importable step
        logger.info("[2/5] Running CCD downloader/parser (imported)…")
        run_ccd_download_parse(
            workdir=workdir,
            out_csv=ccd_csv,
            out_parquet=out_parquet,
            parse=True,
            keep_gz=False,
            max_records=max_records,
            batch_size=batch_size,
        )

        logger.info("CCD parsing completed successfully → %s", ccd_csv)
        return 0

    except Exception as e:
        logger.exception("CCD step failed: %s", e)
        return 1

def read_ccd_map(ccd_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read the CCD CSV map (produced by the CCD step), normalize InChIKeys,
    and drop duplicate (inchikey, chemcomp_id) pairs if any.

    Parameters
    ----------
    ccd_csv : pathlib.Path
        Path to the CSV file with CCD data (including `chemcomp_id` and `inchikey`).
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    pandas.DataFrame
        DataFrame with CCD data. `inchikey` is normalized to uppercase/stripped
        and duplicate pairs are removed.
    """
    logger.info("[3/5] Reading CCD CSV… (%s)", ccd_csv)
    df = pd.read_csv(ccd_csv)

    if "inchikey" in df.columns:
        df["inchikey"] = df["inchikey"].astype(str).str.strip().str.upper()

    logger.info("    CCD rows before dedup: %s", f"{len(df):,}")

    # Deduplicate defensively
    if {"inchikey", "chemcomp_id"}.issubset(df.columns):
        before = len(df)
        df = df.drop_duplicates(subset=["inchikey", "chemcomp_id"])
        after = len(df)
        if after < before:
            logger.info("    Deduplicated CCD: %s rows removed", f"{before - after:,}")
    else:
        logger.warning("    CCD data missing 'inchikey' or 'chemcomp_id' for dedup")

    logger.info("    CCD rows after clean: %s", f"{len(df):,}")
    return df
# ========================== Merge helpers ==========================

def add_pdb_id_by_inchikey(results: pd.DataFrame, ccd_map: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Left-join `results` with CCD map on `inchikey` to add a `pdb_id` column.

    Parameters
    ----------
    results : pandas.DataFrame
        Table containing at least `inchikey`.
    ccd_map : pandas.DataFrame
        Table containing `inchikey` and `chemcomp_id` (from CCD CSV).
    logger : logging.Logger
        Logger for status and match counts.

    Returns
    -------
    pandas.DataFrame
        A copy of `results` with a new `pdb_id` column (from CCD `chemcomp_id`).
        Rows without a match remain with NaN in `pdb_id`.
    """
    if "inchikey" not in results.columns:
        raise ValueError("`results` is missing 'inchikey'.")
    if not {"inchikey", "chemcomp_id"}.issubset(ccd_map.columns):
        raise ValueError("`ccd_map` must contain 'inchikey' and 'chemcomp_id'.")

    res = results.copy()
    ccd = ccd_map[["inchikey", "chemcomp_id"]].copy()
    ccd["inchikey"] = ccd["inchikey"].astype(str).str.strip().str.upper()
    res["inchikey"] = res["inchikey"].astype(str).str.strip().str.upper()

    out = res.merge(
        ccd.rename(columns={"chemcomp_id": "pdb_id"}),
        on="inchikey",
        how="left",
    )
    matched = int(out["pdb_id"].notna().sum())
    logger.info("    after CCD join: %s matched / %s total", f"{matched:,}", f"{len(out):,}")
    return out


def add_pfam_ids_by_uniprot(results: pd.DataFrame, *, max_workers: int, timeout: int, retries: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Resolve Pfam IDs per UniProt accession and join them to `results`.

    Parameters
    ----------
    results : pandas.DataFrame
        Table containing at least the column `protein` (UniProt accession).
    max_workers : int
        Number of threads to use for UniProt API calls (1 = sequential).
    timeout : int
        Per-request timeout in seconds for the UniProt REST API.
    retries : int
        Max retry attempts on transient errors.
    logger : logging.Logger
        Logger for progress and summary.

    Returns
    -------
    pandas.DataFrame
        Copy of `results` with an additional column `pfam_ids` (semicolon-joined).
    """
    if "protein" not in results.columns:
        raise ValueError("`results` is missing 'protein'.")

    logger.info("[4/5] Resolving Pfam IDs via UniProt…")
    proteins = list(pd.Series(results["protein"], dtype=str).dropna().unique())
    logger.info("    unique proteins to query: %s", f"{len(proteins):,}")

    # Wrap tqdm progress via callback to feed logs without prints
    def on_progress(processed: int, total: int, elapsed: float, eta: Optional[float]):
        # Log every ~5%% progress or on completion
        if processed == total or processed % max(1, total // 20) == 0:
            if eta is not None:
                logger.info("    progress: %d/%d (elapsed %.1fs, eta %.1fs)", processed, total, elapsed, eta)
            else:
                logger.info("    progress: %d/%d (elapsed %.1fs)", processed, total, elapsed)

    mapping: Dict[str, Set[str]] = up.get_pfam_for_uniprot_ids(
        proteins,
        max_workers=max_workers,
        timeout=timeout,
        max_retries=retries,
        show_progress=False,              # suppress tqdm visual
        progress_callback=on_progress,    # push progress to logs
    )
    mapping_df = up.mapping_to_dataframe(mapping)

    out = results.copy().merge(
        mapping_df[["uniprot", "pfam_ids"]],
        left_on="protein",
        right_on="uniprot",
        how="left",
        validate="m:1",
    ).drop(columns=["uniprot"])

    ordered_cols = [
    "ligand_id",
    "pdb_id",
    "smiles",
    "inchikey",
    "protein",
    "pfam_ids",
    "pchembl",
    "comment",
]
    out = out[ordered_cols]
    return out

def run_chembl_db_pipeline(
    *,
    chembl_sqlite: str | Path,
    ccd_script: str | Path = "download_parse_ccd.py",
    workdir: str | Path = "temp",
    ccd_csv: str | Path = "temp/ccd.csv",
    out_csv: str | Path = "chembl_db.csv",
    skip_ccd: bool = False,
    drop_missing_pdb: bool = False,
    pfam_workers: int = 24,
    pfam_timeout: int = 45,
    pfam_retries: int = 3,
    log_dir: str | Path = "logs",
    log_level: str = "INFO",
    no_console: bool = False,
) -> int:
    """
    Run the complete ChEMBL→CCD→Pfam pipeline with structured logging.

    This function can be imported and executed programmatically from another script,
    or used via CLI through `main()` for testing. It sequentially performs:
    1) Querying a local ChEMBL SQLite database.
    2) Running or skipping the CCD parsing step.
    3) Merging PDB IDs by InChIKey.
    4) Resolving Pfam IDs via UniProt REST API.
    5) Exporting the final merged dataset to CSV.

    Parameters
    ----------
    chembl_sqlite : str or pathlib.Path
        Path to the local ChEMBL SQLite database (e.g., "chembl_33.db").
    ccd_script : str or pathlib.Path, default="download_parse_ccd.py"
        Path to the script that downloads and parses the PDB CCD data.
    workdir : str or pathlib.Path, default="temp"
        Working directory used for temporary CCD data files.
    ccd_csv : str or pathlib.Path, default="temp/ccd.csv"
        Path to the output CSV generated by the CCD parsing step.
    out_csv : str or pathlib.Path, default="chembl_db.csv"
        Path for the final merged CSV file to be written.
    skip_ccd : bool, default=False
        If True, skip running the CCD parsing script (requires an existing CSV).
    drop_missing_pdb : bool, default=False
        If True, drop rows without `pdb_id` after merging with CCD data.
    pfam_workers : int, default=24
        Number of threads for Pfam resolution via UniProt API (1 = sequential).
    pfam_timeout : int, default=45
        Per-request timeout (seconds) for UniProt REST API calls.
    pfam_retries : int, default=3
        Maximum number of retries for failed UniProt requests.
    log_dir : str or pathlib.Path, default="logs"
        Directory where log files will be stored.
    log_level : str, default="INFO"
        Logging verbosity level ("DEBUG", "INFO", "WARNING", or "ERROR").
    no_console : bool, default=False
        If True, disable console output (log only to file).

    Returns
    -------
    int
        0 if all steps completed successfully; non-zero if any step failed.

    Notes
    -----
    The pipeline assumes that helper functions are available in the same module:
    `setup_logging`, `load_chembl_results`, `run_ccd_script`,
    `read_ccd_map`, `add_pdb_id_by_inchikey`, and `add_pfam_ids_by_uniprot`.
    """
    # Paths
    chembl_db = Path(chembl_sqlite)
    workdir = Path(workdir)
    ccd_csv = Path(ccd_csv)
    out_csv = Path(out_csv)
    log_dir = Path(log_dir)

    # Logging
    log_filename = log_dir / "chembl_pipeline.log"
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("chembl_pipeline", log_filename)

    logger.info("[1/5] Loading ChEMBL results from SQLite: %s", chembl_db)
    results = load_chembl_results(chembl_db, logger)

    # 2) Run CCD 
    if not skip_ccd:
        logger.info("[2/5] Running CCD downloader/parser")
        workdir.mkdir(parents=True, exist_ok=True)
        rc = run_ccd_script(
            workdir=workdir,
            ccd_csv=ccd_csv,
            logger=logger,
            # opcionales:
            out_parquet=None,
            max_records=None,
            batch_size=10000,
        )
        if rc != 0:
            logger.error("CCD script failed with return code %s", rc)
            return rc
    else:
        logger.info("[2/5] Skipping CCD run (using existing CSV): %s", ccd_csv)

    # 3) Merge PDB data
    logger.info("[3/5] Reading CCD map and merging with ChEMBL data")
    ccd_map = read_ccd_map(ccd_csv, logger)
    results = add_pdb_id_by_inchikey(results, ccd_map, logger)

    if drop_missing_pdb:
        before = len(results)
        results = results.dropna(subset=["pdb_id"]).reset_index(drop=True)
        logger.info("Dropped rows without pdb_id: %s", f"{before - len(results):,}")

    # 4) Pfam resolution
    logger.info("[4/5] Resolving Pfam IDs via UniProt (workers=%d, timeout=%ds, retries=%d)",
                pfam_workers, pfam_timeout, pfam_retries)
    results = add_pfam_ids_by_uniprot(
        results,
        max_workers=pfam_workers,
        timeout=pfam_timeout,
        retries=pfam_retries,
        logger=logger,
    )

    # 5) Export
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    logger.info("[5/5] Wrote final CSV: %s", out_csv)
    logger.info("Pipeline finished successfully.")
    return 0
# ========================== CLI (Temporal for testing) ==========================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments used by `main`.
    """
    ap = argparse.ArgumentParser(description="""Run ChEMBL→CCD→Pfam pipeline (logged) and export CSV. This pipeline requires a local ChEMBL SQLite database (downloadable from:
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/)""")
    ap.add_argument("--chembl-sqlite", required=True, help="Path to chembl_XX.db (SQLite)")
    ap.add_argument("--ccd-script", default="download_parse_ccd.py", help="Path to download_parse_ccd.py (default: download_parse_ccd.py)")
    ap.add_argument("--workdir", default="temp", help="Working directory for CCD step (default: temp)")
    ap.add_argument("--ccd-csv", default="temp/ccd.csv", help="CCD CSV output (default: temp/ccd.csv)")
    ap.add_argument("--out-csv", default="chembl_db.csv", help="Final CSV output (default: chembl_db.csv)")
    ap.add_argument("--skip-ccd", action="store_true", help="Skip running the CCD script (assumes --ccd-csv exists)")
    ap.add_argument("--drop-missing-pdb", action="store_true", help="Drop rows with NaN in pdb_id after CCD merge")
    ap.add_argument("--pfam-workers", type=int, default=24, help="Threads for UniProt Pfam step (default: 24)")
    ap.add_argument("--pfam-timeout", type=int, default=45, help="Per-request timeout seconds (default: 45)")
    ap.add_argument("--pfam-retries", type=int, default=3, help="Max retries for UniProt calls (default: 3)")
    ap.add_argument("--log-dir", default="logs", help="Directory to store logs (default: logs)")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--no-console", action="store_true", help="Do not echo logs to console (file only)")
    return ap.parse_args()


def main() -> int:
    """
    Temporal for testing the pipeline.
    Run the complete pipeline according to CLI arguments with structured logging.

    Returns
    -------
    int
        Zero on success; non-zero if any step fails.
    """
    args = parse_args()
    return run_chembl_db_pipeline(**vars(args))


if __name__ == "__main__":
    main()
