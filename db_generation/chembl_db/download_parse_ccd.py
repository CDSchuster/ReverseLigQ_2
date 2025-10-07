#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download and parse the PDB Chemical Component Dictionary (CCD) into a table.

- Source: https://files.wwpdb.org/pub/pdb/data/monomers/components-pub.sdf.gz
- Resumes partial downloads using HTTP Range requests
- Decompresses to .sdf
- Streams and parses with RDKit (SDMolSupplier)
- Exports to Parquet and/or CSV

Output columns:
  chemcomp_id | inchikey

"""

from __future__ import annotations

import gzip
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import logging
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors  # noqa: F401 (kept for compatibility)

DEFAULT_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components-pub.sdf.gz"
DEFAULT_WORKDIR = "ccd_data"
SMILES_KEYS = [
    "SMILES", "smiles", "canonical_smiles", "cansmi", "CACTVS_TAUTO_PARENT_SMILES",
    "cactvs_tauto_parent_smiles", "ccd_smiles"
]
NAME_KEYS = ["name", "pdbx_synonyms", "pdbx_formal_charge", "pdbx_type"]
FORMULA_KEYS = ["formula", "chem_comp.formula", "pdbx_formula"]
log = logging.getLogger("generateDB_log")

# ------------------------- HTTP utils -------------------------
def make_session() -> requests.Session:
    """
    Create a `requests.Session` with retry logic suitable for large file downloads.

    Returns
    -------
    requests.Session
        A configured session with exponential backoff and a custom User-Agent.
    """
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "ccd-downloader/1.0"})
    return s


def download_with_resume(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> Path:
    """
    Download a file with HTTP Range resume support.

    Parameters
    ----------
    url : str
        The URL to download.
    dst : pathlib.Path
        Destination path for the gzipped file.
    chunk_size : int, optional
        Chunk size in bytes for streaming download, by default 1 MiB.

    Returns
    -------
    pathlib.Path
        The path to the completed ``.gz`` file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    sess = make_session()
    temp = dst.with_suffix(dst.suffix + ".part")

    # (Optional) probe remote size
    head = sess.head(url, timeout=30)
    total_size = None
    if head.ok and "Content-Length" in head.headers:
        try:
            total_size = int(head.headers["Content-Length"])  # type: ignore[assignment]
        except (TypeError, ValueError, KeyError):
            total_size = None

    pos = 0
    if temp.exists():
        pos = temp.stat().st_size

    headers = {}
    if pos > 0:
        headers["Range"] = f"bytes={pos}-"

    with sess.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if pos > 0 else "wb"
        with open(temp, mode) as f, tqdm(
            total=total_size if total_size else None,
            initial=pos,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dst.name}"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    temp.rename(dst)
    return dst


# ------------------------- Gzip utils -------------------------
def gunzip_file(src_gz: Path, dst_sdf: Path) -> Path:
    """
    Decompress a ``.gz`` file into a ``.sdf`` file using streaming I/O.

    Parameters
    ----------
    src_gz : pathlib.Path
        Path to the source ``.gz`` file.
    dst_sdf : pathlib.Path
        Destination path for the decompressed ``.sdf`` file.

    Returns
    -------
    pathlib.Path
        The path to the written ``.sdf`` file.
    """
    dst_sdf.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_gz, "rb") as fin, open(dst_sdf, "wb") as fout, tqdm(
        unit="B", unit_scale=True, desc=f"Decompressing {src_gz.name}"
    ) as pbar:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            fout.write(chunk)
            pbar.update(len(chunk))
    return dst_sdf


# ------------------------- Parsing CCD (SDF) -------------------------

def _get_first_prop(mol, keys: List[str]) -> Optional[str]:
    """
    Return the first non-empty molecule property among a set of keys.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule.
    keys : list of str
        Candidate property names to try in order.

    Returns
    -------
    Optional[str]
        The first non-empty property value found, otherwise ``None``.
    """
    v = None
    for k in keys:
        if mol.HasProp(k):
            v = mol.GetProp(k).strip()
    return v


def parse_ccd_sdf_to_table(
    sdf_path: Path,
    out_parquet: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    max_records: Optional[int] = None,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Parse the CCD SDF into a DataFrame with selected columns, writing in batches.

    This function streams records using ``SDMolSupplier`` to avoid loading the
    entire SDF file into memory. Optionally writes incremental Parquet/CSV files.

    Parameters
    ----------
    sdf_path : pathlib.Path
        Path to the ``.sdf`` file to parse.
    out_parquet : Optional[pathlib.Path], optional
        Path to the Parquet output; if provided, results are incrementally written.
    out_csv : Optional[pathlib.Path], optional
        Path to the CSV output; if provided, results are incrementally written.
    max_records : Optional[int], optional
        If set, stop after processing this many molecules (useful for debugging).
    batch_size : int, optional
        Batch size for incremental writes (default: 10000).

    Returns
    -------
    pandas.DataFrame

    """
    # RDKit is assumed available (hard dependency).
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    rows: List[dict] = []
    total = 0
    t0 = time.time()

    # Determine an upper limit for the loop:
    # - If max_records is provided, use it.
    # - If not and the supplier supports len(), use len(suppl).
    # - Otherwise, fall back to "infinite" (loop will stop at StopIteration).
    if max_records is None:
        try:
            limit = len(suppl)  # may raise TypeError for some suppliers
        except TypeError:
            limit = float("inf")
    else:
        limit = max_records

    # Configure a progress bar that knows the total only when 'limit' is finite.
    total_hint = None if limit == float("inf") else limit

    with tqdm(total=total_hint, desc="Parsing CCD (SDF)") as pbar:
        while total < limit:
            # Pull next molecule; stop on natural end of stream.
            try:
                mol = next(suppl)
            except StopIteration:
                break

            if mol is None:
                # Skip invalid/failed parses without incrementing counters.
                continue

            # Component ID (usually in _Name)
            chemcomp_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            # InChIKey (stable key for joins)
            inchi_key = mol.GetProp("InChIKey") if mol.HasProp("InChIKey") else None

            # If no InChIKey, try to generate one (best-effort, light sanitization).
            if inchi_key is None:
                try:
                    m2 = Chem.Mol(mol)
                    Chem.SanitizeMol(
                        m2,
                        Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION,
                        catchErrors=True,
                    )
                    inchi = Chem.MolToInchi(m2)  # requires RDKit built with InChI support
                    inchi_key = Chem.InchiToInchiKey(inchi) if inchi else None
                except (TypeError, ValueError, KeyError):
                    # Keep going if InChI generation fails for this record.
                    pass

            rows.append(
                {
                    "chemcomp_id": (chemcomp_id or "").strip() or None,
                    "inchikey": (inchi_key or "").strip().upper() or None,
                }
            )

            total += 1
            pbar.update(1)

            # Flush in batches to constrain memory footprint.
            if len(rows) >= batch_size:
                df_batch = pd.DataFrame(rows).dropna(subset=["chemcomp_id"]).drop_duplicates()
                _append_or_write(df_batch, out_parquet, out_csv, mode="a")
                rows = []

    # Flush remaining rows (final batch).
    if rows:
        df_batch = pd.DataFrame(rows).dropna(subset=["chemcomp_id"]).drop_duplicates()
        _append_or_write(df_batch, out_parquet, out_csv, mode="a")

    # If not writing to disk, return the full in-memory table.
    df = pd.DataFrame(rows).dropna(subset=["chemcomp_id"]).drop_duplicates()
    return df



def _append_or_write(
    df: pd.DataFrame,
    out_parquet: Optional[Path],
    out_csv: Optional[Path],
    mode: str = "a",
) -> None:
    """
    Write a batch to Parquet and/or CSV, creating or appending as needed.

    Parameters
    ----------
    df : pandas.DataFrame
        The batch to write.
    out_parquet : Optional[pathlib.Path]
        Parquet destination. If the file exists and ``mode == 'a'``, the content is
        concatenated in memory and deduplicated before writing.
    out_csv : Optional[pathlib.Path]
        CSV destination. Appends if the file exists; otherwise writes a new file
        with header.
    mode : str, optional
        Append mode flag (``'a'`` to append), by default ``'a'``.
    """
    if out_parquet:
        if out_parquet.exists() and mode == "a":
            old = pd.read_parquet(out_parquet)
            pd.concat([old, df], ignore_index=True).drop_duplicates().to_parquet(out_parquet, index=False)
        else:
            df.to_parquet(out_parquet, index=False)

    if out_csv:
        header = not out_csv.exists() or mode != "a"
        df.to_csv(out_csv, mode="a" if out_csv.exists() else "w", index=False, header=header)


def run_ccd_download_parse(
    *,
    url: str = DEFAULT_URL,
    workdir: str | Path = DEFAULT_WORKDIR,
    keep_gz: bool = False,
    parse: bool = True,
    out_parquet: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
    max_records: Optional[int] = None,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Download the PDB Chemical Component Dictionary (CCD), decompress it to SDF,
    and optionally parse it into a table (InChIKey + chemcomp_id), exporting to Parquet/CSV.

    Parameters
    ----------
    url : str, default=DEFAULT_URL
        Source URL for the gzipped CCD SDF (`components-pub.sdf.gz`).
    workdir : str or pathlib.Path, default="ccd_data"
        Working directory where the `.gz` and `.sdf` files will be stored.
    keep_gz : bool, default=False
        If True, keep the downloaded `.gz` file after decompression.
    parse : bool, default=True
        If True, parse the SDF into a DataFrame (requires RDKit). If False, only download/decompress.
    out_parquet : str or pathlib.Path or None, default=None
        If provided, write the parsed table to this Parquet file (incremental batches).
    out_csv : str or pathlib.Path or None, default=None
        If provided, write the parsed table to this CSV file (incremental batches).
    max_records : int or None, default=None
        If set, stop parsing after this many molecules (useful for debugging).
    batch_size : int, default=10000
        Batch size for incremental writes during parsing.

    Returns
    -------
    pandas.DataFrame
        If `parse=False`, returns an empty DataFrame (side-effect is the downloaded `.gz` and `.sdf`).
        If `parse=True`, returns the **full table** in memory.

    Notes
    -----
    - RDKit is a hard dependency for parsing (only needed when `parse=True`).
    - Files written:
        * ``{workdir}/components-pub.sdf.gz`` (downloaded)
        * ``{workdir}/components-pub.sdf`` (decompressed)
        * optional Parquet/CSV (if `out_parquet`/`out_csv` are set)
    """
    workdir = Path(workdir)
    gz_path = workdir / "components-pub.sdf.gz"
    sdf_path = workdir / "components-pub.sdf"

    # 1) Download (resumable)
    log.info("==> Downloading CCD (SDF.GZ)")
    download_with_resume(url, gz_path)

    # 2) Decompress
    if not sdf_path.exists():
        log.info("==> Decompressing")
        gunzip_file(gz_path, sdf_path)
    else:
        log.warning(f"[skip] {sdf_path} already exists")

    if not keep_gz:
        try:
            gz_path.unlink(missing_ok=True)
        except Exception:
            pass

    # 3) Optional parse & export
    if not parse:
        log.info("[OK] Download ready (parse=False).")
        return pd.DataFrame()

    log.info("==> Parsing SDF into table")
    pq = Path(out_parquet) if out_parquet else None
    cs = Path(out_csv) if out_csv else None

    df_preview = parse_ccd_sdf_to_table(
        sdf_path,
        out_parquet=pq,
        out_csv=cs,
        max_records=max_records,
        batch_size=batch_size,
    )

    return df_preview
