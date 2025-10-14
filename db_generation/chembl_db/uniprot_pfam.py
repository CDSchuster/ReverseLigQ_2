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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import logging
import requests
import pandas as pd


UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
HEADERS = {"Accept": "application/json", "User-Agent": "uniprot_pfam_min/1.1"}


log = logging.getLogger("generateDB_log")

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

    out = []
    for x in ids:
        if x:
          acc = str(x).strip().upper()
          if acc and acc not in out:
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
    Retrieve Pfam IDs for a single UniProt accession with controlled retries and exponential backoff.

    Parameters
    ----------
    acc : str
        UniProt accession identifier to query.
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts on transient HTTP errors or rate limits.

    Returns
    -------
    Tuple[str, Set[str]]
        A tuple ``(acc, pfam_ids)`` where ``pfam_ids`` is a possibly empty set of Pfam identifiers.

    Notes
    -----
    - HTTP 404 returns an empty set (unknown or deprecated accession).
    - HTTP 429, 502, 503, and 504 trigger exponential backoff (honoring `Retry-After` header if present).
    - If all retries fail, an empty set is returned for that accession.
    """

    result = set()  # default return value (empty set)
    success = False  # flag to know if we got data

    with requests.Session() as sess:
        sess.headers.update(HEADERS)
        url = UNIPROT_ENTRY_URL.format(acc=acc)

        for attempt in range(1, max_retries + 1):
            try:
                r = sess.get(url, timeout=timeout)
                status = r.status_code

                # Not found → no retries, just stop trying
                if status == 404:
                    success = True  # we "succeeded" in knowing it doesn't exist
                    result = set()
                
                # Temporary errors → retry after waiting
                elif status in (429, 502, 503, 504):
                    retry_after = r.headers.get("Retry-After")
                    wait_s = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 30)
                    if attempt < max_retries:
                        time.sleep(wait_s)
                    # no need to change success yet, will retry
                
                # Success
                elif 200 <= status < 300:
                    result = _extract_pfam_from_json(r.json())
                    success = True

                # Any other error → stop trying
                else:
                    success = True

            except requests.RequestException:
                # Network or timeout error
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 30))
                else:
                    success = True  # final failure

            # if we already finished (any of the cases above)
            if success:
                # stop looping without break
                attempt = max_retries

    return result


def get_pfam_for_uniprot_ids(
    ids: Iterable[str],
    *,
    timeout: int = 30,
    max_workers: int = 1,   # 1 = sequential
    max_retries: int = 3,
) -> Dict[str, Set[str]]:
    """
    Fetch Pfam IDs for a list of UniProt accessions.

    Depending on ``max_workers``, requests run sequentially (``max_workers=1``)
    or concurrently using a thread pool (``max_workers>1``).

    Parameters
    ----------
    ids : Iterable[str]
        UniProt accessions to query.
    timeout : int, optional
        Per-request timeout in seconds (default: 30).
    max_workers : int, optional
        Number of worker threads (1 = sequential). Increase for concurrency (default: 1).
    max_retries : int, optional
        Maximum number of retry attempts for transient failures (default: 3).

    Returns
    -------
    Dict[str, Set[str]]
        Mapping ``{accession -> {PFxxxxx, ...}}``; empty set if none or on failure.

    Notes
    -----
    - Concurrency is I/O-bound; using multiple threads can reduce total time.
    """

    accs = _norm_ids(ids)
    total = len(accs)
    mapping: Dict[str, Set[str]] = {a: set() for a in accs}
    if total == 0:
        log.info("No UniProt accessions provided.")
        return mapping

    log.info("Starting UniProt→Pfam retrieval for %d accessions (max_workers=%d)", total, max_workers)

    if max_workers <= 1:
        # Sequential mode
        for i, acc in enumerate(accs, start=1):
            pfset = _fetch_one(acc, timeout=timeout, max_retries=max_retries)
            mapping[acc] = pfset
            log.info("[%d/%d] %s → %d Pfam IDs", i, total, acc, len(pfset))
    else:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_acc = {
                ex.submit(_fetch_one, acc, timeout=timeout, max_retries=max_retries): acc
                for acc in accs
            }
            completed = 0
            for fut in as_completed(future_to_acc):
                acc = future_to_acc[fut]
                try:
                    pfset = fut.result()
                except (requests.exceptions.RequestException, ValueError) as e:
                    log.warning("Error retrieving %s: %s", acc, e)
                    pfset = set()
                mapping[acc] = pfset
                completed += 1
                log.info("[%d/%d] %s → %d Pfam IDs", completed, total, acc, len(pfset))

    log.info("Completed UniProt→Pfam retrieval for %d accessions.", total)
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

    return pd.DataFrame(
        [{"uniprot": k, "pfam_ids": ";".join(sorted(v))} for k, v in sorted(mapping.items())]
    )
