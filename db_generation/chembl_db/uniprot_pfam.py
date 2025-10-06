# -*- coding: utf-8 -*-
"""
Minimal and robust: given a list of UniProt accessions, returns only Pfam IDs (PFxxxxx).
No positions, no batching, no streaming. Uses /uniprotkb/{ACC}.json per entry.
Now with optional parallelism (threads) and lightweight retries.

API
---
get_pfam_for_uniprot_ids(ids, timeout=30, max_workers=16, max_retries=3) -> Dict[str, Set[str]]
mapping_to_dataframe(mapping) -> pandas.DataFrame   (optional)
"""

from __future__ import annotations

from typing import Dict, Iterable, Callable, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

# Usa la variante "auto" para que funcione bien en Jupyter y en terminal
try:
    from tqdm.auto import tqdm
except Exception:  # fallback mínimo
    def tqdm(iterable=None, total=None, desc=None, unit=None, disable=False):
        return iterable if iterable is not None else range(total or 0)

UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
HEADERS = {"Accept": "application/json", "User-Agent": "uniprot_pfam_min/1.1"}


def _norm_ids(ids: Iterable[str]) -> List[str]:
    """
    Normalize a list of UniProt accessions: strip/uppercase and drop duplicates.

    Parameters
    ----------
    ids : Iterable[str]
        A sequence of UniProt accessions (strings); falsy values are ignored.

    Returns
    -------
    list of str
        Normalized accessions (unique, uppercased, in first-seen order).
    """
    seen, out = set(), []
    for x in ids:
        if not x:
            continue
        acc = str(x).strip().upper()
        if acc and acc not in seen:
            seen.add(acc)
            out.append(acc)
    return out


def _extract_pfam_from_json(data: dict) -> Set[str]:
    """
    Extract Pfam IDs (PFxxxxx) from a UniProt JSON entry.

    Parameters
    ----------
    data : dict
        Parsed JSON object for a UniProt entry returned by the REST API.

    Returns
    -------
    set of str
        A set of Pfam IDs present in the entry (e.g., {"PF00001", ...}).
    """
    out: Set[str] = set()
    for x in data.get("uniProtKBCrossReferences", []):
        if x.get("database") == "Pfam":
            pf = x.get("id")
            if isinstance(pf, str) and pf.startswith("PF"):
                out.add(pf.split()[0])
    return out


def _fetch_one(acc: str, *, timeout: int, max_retries: int) -> Tuple[str, Set[str]]:
    """
    Retrieve Pfam IDs for a single UniProt accession with gentle backoff.

    Parameters
    ----------
    acc : str
        The UniProt accession to query.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts on transient errors or rate limits.

    Returns
    -------
    (str, set of str)
        Tuple ``(acc, pfam_ids)`` where ``pfam_ids`` is possibly empty.

    Notes
    -----
    - HTTP 404 yields an empty set (unknown accession).
    - HTTP 429/502/503/504 trigger exponential backoff (honoring `Retry-After` if present).
    - On final failure, returns an empty set for that accession.
    """
    attempt = 0
    with requests.Session() as sess:
        sess.headers.update(HEADERS)
        url = UNIPROT_ENTRY_URL.format(acc=acc)
        while True:
            attempt += 1
            try:
                r = sess.get(url, timeout=timeout)
                if r.status_code == 404:
                    return acc, set()
                if r.status_code in (429, 502, 503, 504):
                    ra = r.headers.get("Retry-After")
                    wait_s = int(ra) if ra and str(ra).isdigit() else min(2 ** attempt, 30)
                    if attempt <= max_retries:
                        time.sleep(wait_s)
                        continue
                r.raise_for_status()
                return acc, _extract_pfam_from_json(r.json())
            except requests.RequestException:
                if attempt <= max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                return acc, set()  # last resort: empty and move on


def get_pfam_for_uniprot_ids(
    ids: Iterable[str],
    *,
    timeout: int = 30,
    max_workers: int = 1,   # 1 = sequential (identical to the original behavior)
    max_retries: int = 3,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
) -> Dict[str, Set[str]]:
    """
    Fetch Pfam IDs for a list of UniProt accessions, with optional progress/ETA reporting.

    Performs either sequential or threaded requests depending on ``max_workers``.
    Optionally displays a progress bar (tqdm) and/or calls a user-supplied callback
    with progress metrics after each accession is processed.

    Parameters
    ----------
    ids : Iterable[str]
        UniProt accessions to query.
    timeout : int, optional
        Per-request timeout in seconds, by default 30.
    max_workers : int, optional
        Number of worker threads (``1`` = sequential), by default 1. Values in the
        16–32 range usually work well for 2k–5k IDs, but adjust based on your limits.
    max_retries : int, optional
        Maximum number of retry attempts for transient failures, by default 3.
    show_progress : bool, optional
        If True, show a tqdm progress bar (works in notebooks and terminals), by default True.
    progress_callback : Callable[[int, int, float, Optional[float]], None], optional
        A function called after each processed accession with:
        ``(processed, total, elapsed_seconds, eta_seconds)``. ``eta_seconds`` may be None
        when ``processed == 0``.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping ``{accession -> {PFxxxxx, ...}}``; empty set if none/failure.

    Notes
    -----
    - ETA is a rough estimate: it assumes the remaining items will take similar time.
    - ``progress_callback`` is useful if you want to log to your UI/logger in addition
      to (or instead of) showing the progress bar.
    """
    accs = _norm_ids(ids)
    total = len(accs)
    mapping: Dict[str, Set[str]] = {a: set() for a in accs}
    if total == 0:
        return mapping

    processed = 0
    t0 = time.monotonic()

    def _update_progress(pbar):
        nonlocal processed
        processed += 1
        elapsed = time.monotonic() - t0
        eta = (elapsed * (total - processed) / processed) if processed > 0 else None
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"{processed}/{total}")
        if progress_callback is not None:
            progress_callback(processed, total, elapsed, eta)

    pbar = tqdm(total=total, desc="Fetching UniProt→Pfam", unit="acc", disable=not show_progress)

    if max_workers <= 1:
        # Sequential mode
        for acc in accs:
            _, pfset = _fetch_one(acc, timeout=timeout, max_retries=max_retries)
            mapping[acc] = pfset
            _update_progress(pbar)
        if pbar is not None:
            pbar.close()
        return mapping

    # Parallel mode
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_one, acc, timeout=timeout, max_retries=max_retries) for acc in accs]
        for fut in as_completed(futures):
            acc, pfset = fut.result()
            mapping[acc] = pfset
            _update_progress(pbar)

    if pbar is not None:
        pbar.close()
    return mapping


def mapping_to_dataframe(mapping: Dict[str, Set[str]]):
    """
    Convert a mapping ``{uniprot -> {PFxxxxx, ...}}`` to a pandas.DataFrame.

    Parameters
    ----------
    mapping : Dict[str, Set[str]]
        Mapping from UniProt accession to a set of Pfam IDs.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with:
        - ``uniprot``: the accession
        - ``pfam_ids``: semicolon-joined Pfam IDs (sorted)
    """
    import pandas as pd
    return pd.DataFrame(
        [{"uniprot": k, "pfam_ids": ";".join(sorted(v))} for k, v in sorted(mapping.items())]
    )
