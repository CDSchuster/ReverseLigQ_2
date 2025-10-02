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

Usage:
  python download_parse_ccd.py \
    --out-parquet ccd.parquet \
    --out-csv ccd.csv

"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors  # noqa: F401 (kept for compatibility)
    RDKit_OK = True
except Exception:
    RDKit_OK = False


DEFAULT_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components-pub.sdf.gz"


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
        except Exception:
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
        while True:
            chunk = fin.read(1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            pbar.update(len(chunk))
    return dst_sdf


# ------------------------- Parsing CCD (SDF) -------------------------
SMILES_KEYS = [
    "SMILES", "smiles", "canonical_smiles", "cansmi", "CACTVS_TAUTO_PARENT_SMILES",
    "cactvs_tauto_parent_smiles", "ccd_smiles"
]
NAME_KEYS = ["name", "pdbx_synonyms", "pdbx_formal_charge", "pdbx_type"]
FORMULA_KEYS = ["formula", "chem_comp.formula", "pdbx_formula"]


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
    for k in keys:
        if mol.HasProp(k):
            v = mol.GetProp(k).strip()
            if v:
                return v
    return None


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
        Path to the Parquet output; if provided, results are incrementally written,
        by default ``None``.
    out_csv : Optional[pathlib.Path], optional
        Path to the CSV output; if provided, results are incrementally written,
        by default ``None``.
    max_records : Optional[int], optional
        If set, stop after processing this many molecules (useful for debugging),
        by default ``None``.
    batch_size : int, optional
        Batch size for incremental writes, by default ``10000``.

    Returns
    -------
    pandas.DataFrame
        If writing to disk, returns a small preview DataFrame (head of the written
        file). If not writing, returns the full in-memory table.

    Raises
    ------
    RuntimeError
        If RDKit is not available in the environment.
    """
    if not RDKit_OK:
        raise RuntimeError("RDKit is not available. Install with: pip install rdkit-pypi")

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    rows: List[dict] = []
    total = 0
    t0 = time.time()

    for mol in tqdm(suppl, desc="Parsing CCD (SDF)"):
        if mol is None:
            continue

        # Component ID (usually in _Name)
        chemcomp_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        # InChIKey (stable key for joins)
        inchi_key = mol.GetProp("InChIKey") if mol.HasProp("InChIKey") else None

        # If no InChIKey, try to generate (best-effort, light sanitization)
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
            except Exception:
                pass

        rows.append(
            {
                "chemcomp_id": (chemcomp_id or "").strip() or None,
                "inchikey": (inchi_key or "").strip().upper() or None,
            }
        )
        total += 1

        if max_records and total >= max_records:
            break

        # Flush in batches to constrain memory
        if len(rows) >= batch_size:
            df_batch = pd.DataFrame(rows).dropna(subset=["chemcomp_id"]).drop_duplicates()
            _append_or_write(df_batch, out_parquet, out_csv, mode="a")
            rows = []

    # Final batch
    if rows:
        df_batch = pd.DataFrame(rows).dropna(subset=["chemcomp_id"]).drop_duplicates()
        _append_or_write(df_batch, out_parquet, out_csv, mode="a")

    # If writing to disk by batches, return a brief preview to avoid reloading everything
    if out_parquet or out_csv:
        elapsed = time.time() - t0
        print(f"[OK] Processed ~{total} molecules in {elapsed:.1f}s")
        if out_parquet and out_parquet.exists():
            return pd.read_parquet(out_parquet).head(10)
        if out_csv and out_csv.exists():
            return pd.read_csv(out_csv, nrows=10)
        return pd.DataFrame()

    # If not writing to disk, return the full DataFrame in memory
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


# ------------------------- Main CLI -------------------------

def main() -> None:
    """
    Command-line entry point for downloading and parsing the CCD.

    Steps
    -----
    1. Download the gzipped SDF (resumable)
    2. Decompress to ``.sdf``
    3. Optionally parse and export to Parquet/CSV

    Notes
    -----
    RDKit must be available to parse the SDF. If you only want to download and
    decompress, use ``--only-download``.
    """
    ap = argparse.ArgumentParser(description="Download and parse the CCD (SDF) into a table.")
    ap.add_argument("--url", default=DEFAULT_URL, help="URL for components-pub.sdf.gz")
    ap.add_argument("--workdir", default="ccd_data", help="Working directory")
    ap.add_argument("--keep-gz", action="store_true", help="Keep the .gz after decompressing")
    ap.add_argument("--only-download", action="store_true", help="Only download and decompress; do not parse")
    ap.add_argument("--out-parquet", type=str, default=None, help="Output Parquet path (e.g., ccd.parquet)")
    ap.add_argument("--out-csv", type=str, default=None, help="Output CSV path (e.g., ccd.csv)")
    ap.add_argument("--max-records", type=int, default=None, help="Limit parsing to N molecules (debug)")
    ap.add_argument("--batch-size", type=int, default=10000, help="Batch size for incremental writes")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    gz_path = workdir / "components-pub.sdf.gz"
    sdf_path = workdir / "components-pub.sdf"

    print("==> Downloading CCD (SDF.GZ)")
    download_with_resume(args.url, gz_path)

    if not sdf_path.exists():
        print("==> Decompressing")
        gunzip_file(gz_path, sdf_path)
    else:
        print(f"[skip] {sdf_path} already exists")

    if not args.keep_gz:
        try:
            gz_path.unlink(missing_ok=True)
        except Exception:
            pass

    if args.only_download:
        print("[OK] Download ready.")
        return

    if not RDKit_OK:
        print("ERROR: RDKit is not installed. Install with: pip install rdkit-pypi", file=sys.stderr)
        sys.exit(1)

    print("==> Parsing SDF into table")
    out_parquet = Path(args.out_parquet) if args.out_parquet else None
    out_csv = Path(args.out_csv) if args.out_csv else None

    df_preview = parse_ccd_sdf_to_table(
        sdf_path,
        out_parquet=out_parquet,
        out_csv=out_csv,
        max_records=args.max_records,
        batch_size=args.batch_size,
    )

    if out_parquet or out_csv:
        print("[OK] Exported.")
        if not df_preview.empty:
            print(df_preview.head(5).to_string(index=False))
    else:
        # If no outputs were requested, show a preview in stdout
        print(df_preview.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
