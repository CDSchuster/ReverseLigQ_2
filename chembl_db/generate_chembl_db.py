#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end ChEMBL → CCD → UniProt/Pfam pipeline (CLI) **with structured logging**.

This script reproduces the Jupyter workflow as a single, reusable command-line
program and **logs** every step to a rotating log file (and optionally the
console). It performs:

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

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Dict, Set, Optional

import pandas as pd

# Local dependency: your translated UniProt→Pfam helper
import uniprot_pfam as up


SQL_QUERY = """
SELECT DISTINCT
    md.chembl_id                               AS ligand_id,
    cs.canonical_smiles                        AS smiles,
    UPPER(cs.standard_inchi_key)               AS inchikey,
    csq.accession                              AS protein,
    act.pchembl_value                          AS pchembl,
    act.activity_comment                       AS comment
FROM activities               AS act
JOIN assays                   AS a    ON act.assay_id = a.assay_id
JOIN target_dictionary        AS td   ON a.tid = td.tid
JOIN target_components        AS tc   ON td.tid = tc.tid
JOIN component_sequences      AS csq  ON tc.component_id = csq.component_id
JOIN compound_records         AS cr   ON act.record_id = cr.record_id
JOIN molecule_dictionary      AS md   ON cr.molregno = md.molregno
JOIN compound_structures      AS cs   ON md.molregno = cs.molregno
WHERE a.assay_type = 'B'
  AND td.target_type = 'SINGLE PROTEIN'
  AND act.pchembl_value IS NOT NULL
  AND cs.canonical_smiles IS NOT NULL
  AND cs.standard_inchi_key IS NOT NULL;
"""


# ========================== Logging ==========================

def setup_logger(log_dir: Path, level: str = "INFO", console: bool = True) -> logging.Logger:
    """
    Configure logging to a rotating file (and optional console).

    Parameters
    ----------
    log_dir : pathlib.Path
        Directory to store log files. A timestamped file will be created and
        a copy named ``latest.log`` is maintained for convenience.
    level : str, optional
        Logging level name (e.g., "DEBUG", "INFO", "WARNING"), by default "INFO".
    console : bool, optional
        If True, also log to stdout, by default True.

    Returns
    -------
    logging.Logger
        Configured logger instance named "pipeline".
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"run_{ts}.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    # Header context
    logger.info("=== ChEMBL→CCD→Pfam Pipeline ===")
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s | %s", platform.system(), platform.version())
    logger.info("CWD: %s", os.getcwd())
    logger.info("Log file: %s", log_path)

    # Maintain latest.log (copy for portability)
    try:
        shutil.copyfile(log_path, log_dir / "latest.log")
    except Exception:
        pass

    return logger


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
    logger.info("    rows: %s", f"{len(df):,}")
    return df


def run_ccd_script(python_exe: str, ccd_script: Path, *, workdir: Path, ccd_csv: Path, logger: logging.Logger) -> int:
    """
    Execute the CCD downloader/parser script to produce the CCD CSV mapping.

    Parameters
    ----------
    python_exe : str
        Python interpreter to invoke for the child script (e.g., sys.executable).
    ccd_script : pathlib.Path
        Path to the CCD script (download_parse_ccd.py).
    workdir : pathlib.Path
        Working directory used by the CCD script (contains SDF and outputs).
    ccd_csv : pathlib.Path
        Path where the CCD CSV will be written (e.g., temp/ccd.csv).
    logger : logging.Logger
        Logger used to stream the child process output.

    Returns
    -------
    int
        Subprocess return code (0 indicates success).
    """
    logger.info("[2/5] Running CCD downloader/parser…")
    workdir.mkdir(parents=True, exist_ok=True)
    ccd_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(ccd_script),
        "--workdir", str(workdir),
        "--out-csv", str(ccd_csv),
    ]
    return run_cmd_stream(cmd, logger)


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

    return out


# ========================== CLI ==========================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments used by `main`.
    """
    ap = argparse.ArgumentParser(description="Run ChEMBL→CCD→Pfam pipeline (logged) and export CSV.")
    ap.add_argument("--chembl-sqlite", required=True, help="Path to chembl_XX.db (SQLite)")
    ap.add_argument("--ccd-script", default="download_parse_ccd.py", help="Path to download_parse_ccd.py")
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
    Run the complete pipeline according to CLI arguments with structured logging.

    Returns
    -------
    int
        Zero on success; non-zero if any step fails.
    """
    args = parse_args()

    logger = setup_logger(Path(args.log_dir), level=args.log_level, console=not args.no_console)

    chembl_db = Path(args.chembl_sqlite)
    ccd_script = Path(args.ccd_script)
    workdir = Path(args.workdir)
    ccd_csv = Path(args.ccd_csv)
    out_csv = Path(args.out_csv)

    # 1) Query ChEMBL SQLite
    results = load_chembl_results(chembl_db, logger)

    # 2) Run CCD script (unless skipped)
    if not args.skip_ccd:
        rc = run_ccd_script(sys.executable, ccd_script, workdir=workdir, ccd_csv=ccd_csv, logger=logger)
        if rc != 0:
            logger.error("CCD script failed with return code %s", rc)
            return rc
    else:
        logger.info("[2/5] Skipping CCD run (using existing CSV)")

    # 3) Read CCD CSV and merge pdb_id by InChIKey
    ccd_map = read_ccd_map(ccd_csv, logger)
    results = add_pdb_id_by_inchikey(results, ccd_map, logger)
    if args.drop_missing_pdb:
        before = len(results)
        results = results.dropna(subset=["pdb_id"]).reset_index(drop=True)
        logger.info("    dropped rows without pdb_id: %s", f"{before - len(results):,}")

    # 4) Resolve Pfam IDs per protein and merge
    results = add_pfam_ids_by_uniprot(
        results,
        max_workers=args.pfam_workers,
        timeout=args.pfam_timeout,
        retries=args.pfam_retries,
        logger=logger,
    )

    # 5) Write final CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    logger.info("[5/5] Wrote final CSV: %s", out_csv)

    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
