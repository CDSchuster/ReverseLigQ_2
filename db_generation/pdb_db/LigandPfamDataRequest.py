"""
This module contains functions to request PDB IDs that are bound to molecules,
request their ligands' data (SMILES included), and the Pfam domains in those
PDBs.
"""

import os
import json
import logging
import concurrent.futures

import requests
import pandas as pd
from rdkit import Chem


log = logging.getLogger("generateDB_log")


def get_pdb_ids():
    """Retrieves PDB IDs for a specific query several batches

    Returns
    -------
    all_ids : list
        all the PDB IDs that contain some bound molecule
    """

    # Open the JSON file that contains the RCSB search query template
    with open("db_generation/pdb_db/query_pdb.json", "r") as file:
        query_template = json.load(file)

    # Base URL for RCSB search API and headers for JSON body
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    headers = {"Content-Type": "application/json"}

    all_results, found_results = [], True
    start = 0

    # Loop over paginated results until no more are returned
    while found_results:
        # Update pagination fields in the query template
        query_template["request_options"]["paginate"]["start"] = start
        query_template["request_options"]["paginate"]["rows"] = 10000

        # Send POST request with the query
        response = requests.post(url, json=query_template, headers=headers)

        if response.status_code == 200:
            data = response.json()
            # If the result_set key exists and is non-empty, extend results
            if "result_set" in data and data["result_set"]:
                all_results.extend(data["result_set"])
                # Move to the next page offset
                start += 10000
            else:
                # No more results to fetch
                found_results = False
        else:
            # Log non-200 responses for debugging / audit
            log.error(f"Error: {response.status_code}, {response.text}")
            found_results = False

    # Extract 'identifier' field from each result entry
    all_ids = [pdb["identifier"] for pdb in all_results]
    return all_ids


def fetch_url(pdb_id, url):
    """Fetch data from a given URL with error handling.

    Parameters
    ----------
    pdb_id : str
        a valid PDB ID
    url : str
        URL to request data with the PDB ID

    Returns
    -------
    results : tuple
        a tuple containing the PDB ID, URL and the response
    """

    attempts = 0
    # Errors considered transient / worth retrying
    errors_list = ["HTTPSConnectionPool", "500", "503", "504"]

    # Try up to 5 times for transient errors
    while attempts < 5:
        try:
            # Perform GET request with a 60s timeout
            response = requests.get(url, timeout=60)
            # Raise an HTTPError for bad status codes
            response.raise_for_status()
            # On success, return parsed JSON
            results = (pdb_id, url, response.json())
            attempts = 5
        except requests.RequestException as e:
            # Save error message for logging/inspection
            results = (pdb_id, url, str(e))
            # If error message matches known transient errors, retry
            if any(err in str(e) for err in errors_list):
                attempts += 1
            else:
                # For non-retryable errors, stop retrying and log
                attempts = 5
                log.error(e)
            # Mark a fail type so caller can reattempt later if needed
            if any(err in str(e) for err in errors_list):
                failtype = "pfam_fail" if "pfam" in url else "ligand_fail"
                results = (pdb_id, url, failtype)

    return results


def parallelize_pfam_ligand_request(pdb_ids):
    """Takes a list of URLs and parallelizes the retrieval
    of ligand and Pfam data for every PDB ID given.

    Parameters
    ----------
    pdb_ids : list
        a list of valid PDB IDs

    Returns
    -------
    results_dict : dict
        A dictionary containing the data for every PDB ID and every URL with the following keys:
            - pdb_id: The PDB ID.
            - Pfam_url: The Pfam data retrieved for the PDB ID
            - ligand_url: the ligand data retrieved for the PDB ID
       
    fails_dict : dict
        A dictionary containing the errors for Pfam data and ligand data requests
    """

    # Two endpoints: ligand info and Pfam mappings
    urls = [
        "https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}",
        "https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}",
    ]

    # Create (pdb_id, url) task tuples for each combination
    tasks = [(pdb_id, url.format(pdb_id=pdb_id))
             for pdb_id in pdb_ids for url in urls]

    results_dict = {}
    all_ligand_fails, all_pfam_fails = [], []

    # Use threads for IO-bound network calls; max_workers tuned high for concurrency
    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        # Map fetch_url across all tasks and collect results
        results = tuple(executor.map(lambda args: fetch_url(*args), tasks))

        # Iterate over results and store them appropriately
        for pdb_id, url, data in results:
            if data == "pfam_fail":
                # Record PDB IDs whose Pfam request failed
                all_pfam_fails.append(pdb_id)
            elif data == "ligand_fail":
                # Record PDB IDs whose ligand request failed
                all_ligand_fails.append(pdb_id)
            else:
                # Store successful responses in results_dict keyed by pdb_id
                if pdb_id not in results_dict:
                    results_dict[pdb_id] = {}
                if "pfam" in url and isinstance(data, dict):
                    # PDBe returns pfam mappings under a lowercased pdb key
                    results_dict[pdb_id]["Pfam_url"] = (
                        data[pdb_id.lower()]["Pfam"]
                    )
                elif isinstance(data, dict):
                    # Save ligand data under 'ligand_url'
                    results_dict[pdb_id]["ligand_url"] = data[pdb_id.lower()]

    # Prepare a summary of failures
    fails_dict = {"pfam_fails": all_pfam_fails,
                  "ligand_fails": all_ligand_fails}

    return results_dict, fails_dict


def get_bmids_rows(pdb_id, results):
    """For a certain PDB_ID and its ligands requests results,
    creates a list of lists containing the necessary data
    to be used as rows for a dataframe.

    Parameters
    ----------
    pdb_id : str
        a valid PDB ID
    results : list
        a list of dictionaries, each one containing data of a ligand bound to the PDB ID

    Returns
    -------
    new_rows : list
        a list of lists containing PDB ID, Chain ID, ligand hetcode and bm_id
    """

    new_rows = []

    # Each "bm_dict" corresponds to a bound molecule entry (bm_id)
    for bm_dict in results:
        ligands_list = bm_dict["composition"]["ligands"]

        # Iterate through ligands associated with this bound molecule
        for ligand in ligands_list:
            # Build a row matching the ligand_df columns
            new_row = [pdb_id, ligand["chain_id"], ligand["chem_comp_id"],
                       bm_dict["bm_id"]]
            new_rows.append(new_row)

    return new_rows


def get_pfam_rows(pdb_id, results):
    """For a certain PDB_ID and its Pfam requests results,
    creates a list of lists containing the necessary data
    to be used as rows for a dataframe.

    Parameters
    ----------
    pdb_id : str
        a valid PDB ID
    results : list
        a list of dictionaries, each one containing data of a Pfam domain in the PDB ID

    Returns
    -------
    new_rows : list
        a list of lists containing PDB ID, Chain ID, Pfam ID, Pfam name,
        and Pfam domain start and end positions
    """

    new_rows = []
    # results is a dict where keys are pfam IDs and values contain mappings
    for key, value in results.items():
        pfam_id = key
        pfam_name = value["identifier"]
        # For every mapping instance, append a row with start/end residues
        new_rows.extend([[pdb_id, instance["chain_id"], pfam_id, pfam_name,
                          instance["start"]["residue_number"],
                          instance["end"]["residue_number"]]
                         for instance in value["mappings"]])
    return new_rows


def generate_DFs(subset_pdb_ids, ligand_results):
    """Processes a subset of PDB IDs to create ligand and Pfam DataFrames.

    Parameters
    ----------
    subset_pdb_ids : list
        a list of PDB IDs
    ligand_results : dict
        a dictionary containing Pfam and ligand data for every PDB ID

    Returns
    -------
    ligand_df : pandas.DataFrame
        dataframe containing ligand data per PDB ID
    pfam_df : pandas.DataFrame
        dataframe containing Pfam domain data per PDB ID
    """

    # Initialize empty DataFrames with the desired columns
    ligand_df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'ligand_id', "bm_id"])
    pfam_df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'pfam_id',
                                    "pfam_name", "start", "end"])

    for pdb_id in subset_pdb_ids:
        urls = ligand_results.get(pdb_id, {})

        # Build rows for ligand_df from ligand_url results
        new_bmid_rows = get_bmids_rows(pdb_id, urls.get("ligand_url", []))
        ligand_df = pd.concat([ligand_df,
                               pd.DataFrame(new_bmid_rows,
                                            columns=ligand_df.columns)],
                                            ignore_index=True)

        # Build rows for pfam_df from Pfam_url results
        new_pfam_rows = get_pfam_rows(pdb_id, urls.get("Pfam_url", {}))
        pfam_df = pd.concat([pfam_df,
                             pd.DataFrame(new_pfam_rows,
                                          columns=pfam_df.columns)],
                                          ignore_index=True)
        
    return ligand_df, pfam_df


def parallelize_DFs_generation(pdb_ids, ligand_results):
    """Parallelizes ligand and Pfam data DFs generation
    by giving subsets of data to the function :func:`generate_DFs`.

    Parameters
    ----------
    pdb_ids : list
        a list of PDB IDs
    ligand_results : dict
        a dictionary containing Pfam and ligand data for every PDB ID

    Returns
    -------
    ligand_df : pandas.DataFrame
        dataframe containing ligand data per PDB ID
    pfam_df : pandas.DataFrame
        dataframe containing Pfam domain data per PDB ID
    """

    # Determine an upper bound on parallel workers (cap at 12)
    num_workers = min(os.cpu_count(), 12)
    # Compute chunk size to split the pdb_ids across workers
    chunk_size = max(1, len(pdb_ids) // num_workers)
    chunks = [pdb_ids[i:i + chunk_size] for i in range(0, len(pdb_ids),
                                                        chunk_size)]

    # Use multiprocessing (ProcessPoolExecutor) to generate DataFrames in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(generate_DFs,
                                    chunks, [ligand_results] * len(chunks)))

    # Concatenate partial results from all workers into final DataFrames
    ligand_df = pd.concat([res[0] for res in results], ignore_index=True)
    pfam_df = pd.concat([res[1] for res in results], ignore_index=True)

    return ligand_df, pfam_df


def smile_selection(response_json):
    """Selects the SMILE code from the response JSON.
    The SMILE code can be in the RCSB or PDBx data.
    If it is in both, we select the RCSB one.
    If it is in neither, we return None.

    Parameters
    ----------
    response_json : dict
        A json dictionary containing the response from the RCSB API
        with the SMILES for a given PDB ID

    Returns
    -------
    selected_smile : str
        A SMILE code for the ligand in the PDB ID
    """

    selected_smile = None
    # rcsb_smiles may contain fields SMILES and SMILES_stereo
    rcsb_smiles = response_json["data"]["chem_comp"]["rcsb_chem_comp_descriptor"]
    # pdbx_chem_comp_descriptor is typically a list of descriptors
    pdbx_smiles = response_json["data"]["chem_comp"]["pdbx_chem_comp_descriptor"]

    # Prefer the rcsb-provided SMILES if present
    if rcsb_smiles["SMILES"] is not None:
        selected_smile = rcsb_smiles["SMILES"]
    elif rcsb_smiles["SMILES_stereo"] is not None:
        selected_smile = rcsb_smiles["SMILES_stereo"]
    else:
        # Fallback to scanning pdbx descriptors for a SMILES entry
        i = 0
        while selected_smile is None and i < len(pdbx_smiles):
            if pdbx_smiles[i]["type"] in ["SMILES", "SMILES_CANONICAL"]:
                selected_smile = pdbx_smiles[i]["descriptor"]
            i += 1

    return selected_smile


def fetch_SMILE_data(het):
    """Requests SMILE data for a given PDB hetcode.

    Parameters
    ----------
    het : str
        A valid ligand hetcode

    Returns
    -------
    result : tuple
        a tuple containing the hetcode and the request response
    """

    url = "https://data.rcsb.org/graphql"
    query_compounds = """query molecule ($id: String!) {chem_comp(comp_id:$id){
        rcsb_chem_comp_descriptor {SMILES SMILES_stereo}
        pdbx_chem_comp_descriptor {type descriptor}}}"""

    attempts = 0
    # Errors that usually deserve a retry (rate limits, transient network issues)
    errors_list = ["429", "HTTPSConnectionPool", "RemoteDisconnected"]
    while attempts < 5:
        try:
            response = requests.post(
                url,
                json={"query": query_compounds, "variables": {"id": het}},
                timeout=60
            )
            response.raise_for_status()
            # Parse and pick the appropriate SMILES string
            result = (het, smile_selection(response.json()))
            attempts = 5
        except Exception as e:
            # If request fails, return (het, None) and possibly retry
            result = (het, None)
            if any(err in str(e) for err in errors_list):
                attempts += 1
            else:
                attempts = 5
                log.error(e)
            # If the last attempt failed on a retryable error, log it in detail
            if any(err in str(e) for err in errors_list) and attempts == 5:
                log.error(f"{het}: {str(e)}")

    return result


def parallelize_SMILE_request(ligand_df):
    """Parallelizes requests for SMILEs data to RCSB.

    Parameters
    ----------
    ligand_df : pandas.DataFrame
        dataframe containing ligand data (including hetcodes)

    Returns
    -------
    ligand_df : pandas.DataFrame
        the same dataframe as the input but with the SMILEs in a new column
    """

    # Use a small thread pool for parallel GraphQL requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_SMILE_data, list(set(ligand_df.ligand_id)))

    # Build a mapping het -> SMILES and add it to the ligand_df
    raw_smiles_data = dict(results)
    ligand_df["SMILES"] = ligand_df["ligand_id"].map(raw_smiles_data)

    return ligand_df


def count_atoms(smiles):
    """Counts the number of atoms for a SMILE.

    Parameters
    ----------
    smiles : str
        a SMILE code

    Returns
    -------
    atoms_num : int
        the number of atoms in the SMILE molecule
    """

    atoms_num = None
    if isinstance(smiles, str):
        # Use RDKit to parse SMILES. sanitize=False to avoid strict sanitization errors
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        atoms_num = mol.GetNumAtoms() if mol else None
    return atoms_num


def filter_small_ligands(ligand_df):
    """Filters rows of ligands_df if ligands are smaller than 10 atoms.

    Parameters
    ----------
    ligand_df : pandas.DataFrame
        A ligand dataframe that includes the column "SMILES"

    Returns
    -------
    ligand_df : pandas.DataFrame
        the input dataframe with rows filtered based on the ligands number
        of atoms
    """

    # Work on a copy to avoid mutating the caller's DataFrame
    ligand_df = ligand_df.copy()
    # Remove entries with unknown ligand code 'UNL'
    ligand_df = ligand_df[ligand_df["ligand_id"] != "UNL"]
    # Count unique PDB IDs before filtering
    pdb_ligand_num = len(set(ligand_df.pdb_id))
    # Compute number of atoms for each SMILES
    ligand_df["num_atoms"] = ligand_df["SMILES"].apply(count_atoms)
    # Keep only ligands with at least 10 atoms
    ligand_df = ligand_df[ligand_df["num_atoms"].notna() &
                          (ligand_df["num_atoms"] >= 10)].copy()
    # Drop temporary column used for filtering
    ligand_df = ligand_df.drop(columns=["num_atoms"])
    # Compute how many PDB IDs were filtered out
    filtered_pdb_num = pdb_ligand_num - len(set(ligand_df.pdb_id))
    log.info(f"PDB IDs filtered after small ligand filtering: "
             f"{filtered_pdb_num}")
    return ligand_df


def run_requests(pdb_ids):
    """For a given list of PDB IDs, request ligand and Pfam data, as well
    as their ligand SMILEs.

    Parameters
    ----------
    pdb_ids : list
        a list of valid PDB IDs

    Returns
    -------
    ligand_df : pandas.DataFrame
        dataframe containing ligand data per PDB ID
    pfam_df : pandas.DataFrame
        dataframe containing Pfam domain data per PDB ID
    """

    # First request Pfam and ligand raw data
    ligand_results, fails_dict = parallelize_pfam_ligand_request(pdb_ids)
    log.info("Generating ligand and Pfam dataframes")
    # Convert raw results into structured DataFrames in parallel
    ligand_df, pfam_df = parallelize_DFs_generation(pdb_ids, ligand_results)
    log.info("Requesting SMILEs")
    # Fetch SMILES for every unique ligand and attach to ligand_df
    ligand_df = parallelize_SMILE_request(ligand_df)
    return ligand_df, pfam_df, fails_dict


def get_ligand_pfam_data():
    """Gets all PDB IDs with ligands and requests ligand and Pfam data for those.

    Returns
    -------
    results_dict : dict
        a dictionary containing the ligand dataframe, Pfam dataframe and PDB IDs
    """

    log.info("Retrieving PDB IDs with bound molecules")
    pdb_ids = get_pdb_ids()
    
    log.info(f"Total PDB IDs: {len(pdb_ids)}")
    log.info("Requesting ligand and Pfam data")

    ligand_df, pfam_df, fails_dict = run_requests(pdb_ids)
    log.info(f"Successful PDB IDs ligand requests: {len(set(ligand_df.pdb_id))}")
    log.info(f"Successful PDB IDs Pfam requests: {len(set(pfam_df.pdb_id))}")

    results_dict = {"ligand_df": ligand_df,
                    "pfam_df": pfam_df,
                    "fails": fails_dict}

    return results_dict


def retry_lp_request(ligand_df, pfam_df, fails_dict):
    """Requests Pfam and ligand data for PDB IDs that failed request
    originally. Data that could be recovered is added to the input dataframes.

    Parameters
    ----------
    ligand_df : pandas.DataFrame
        dataframe containing ligand data
    pfam_df : pandas.DataFrame
        dataframe containing Pfam domains data
    fails_dict : dict
        dictionary with the lists of PDB IDs that failed ligand and Pfam data requests

    Returns
    -------
    ligand_df : pandas.DataFrame
        dataframe containing ligand data per PDB ID
    pfam_df : pandas.DataFrame
        dataframe containing Pfam domain data per PDB ID
    """
    
    # Extract lists of PDB IDs that failed previously
    ligand_pdb_ids = fails_dict["ligand_fails"]
    pfam_pdb_ids = fails_dict["pfam_fails"]

    log.info(f"{len(ligand_pdb_ids)} ligand PDB IDs failed request that can be recovered")
    log.info(f"{len(pfam_pdb_ids)} Pfam PDB IDs failed request that can be recovered")

    if ligand_pdb_ids:
        # Retry fetching ligand data for the failed PDB IDs
        ligand_df_ligand, _, _ = run_requests(ligand_pdb_ids)
        log.info(f"Recovered {len(set(ligand_df_ligand.pdb_id))} ligand PDB IDs")
        # Merge recovered ligand rows into the provided ligand_df
        ligand_df = pd.concat([ligand_df, ligand_df_ligand], ignore_index=True)

    if pfam_pdb_ids:
        # Retry fetching Pfam data for the failed PDB IDs
        _, pfam_df_pfam, _ = run_requests(pfam_pdb_ids)
        log.info(f"Recovered {len(set(pfam_df_pfam.pdb_id))} Pfam PDB IDs")
        # Merge recovered Pfam rows into the provided pfam_df
        pfam_df = pd.concat([pfam_df, pfam_df_pfam], ignore_index=True)

    return ligand_df, pfam_df
