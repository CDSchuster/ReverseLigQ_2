"""
The functions in this module are designed to request ligand interactions
to proteins and store them in a dataframe.
"""

import logging
import concurrent.futures
import requests
import pandas as pd


log = logging.getLogger("generateDB_log")


def fetch_interaction(pdb_id, bm, url):
    """Get interaction data for a given PDB and bound molecule.

    Parameters
    ----------
    pdb_id : str
        A valid PDB ID.
    bm : str
        Bound molecule ID.
    url : str
        URL to request ligand interaction data.

    Returns
    -------
    tuple
        Contains PDB ID, bm (bound molecule ID), and request response.
    """
    
    attempts = 0
    errors_list = [
        "HTTPSConnectionPool",
        "RemoteDisconnected",
        "Connection reset by peer",
        "500",
        "502",
        "503",
        "504",
    ]

    while attempts < 10:
        try:
            # Save the results if the request is successful
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            results = pdb_id, bm, response.json()
            attempts = 10
        except requests.RequestException as e:
            results = pdb_id, bm, str(e)
            if any(err in str(e) for err in errors_list):
                # If the error is recoverable, we retry
                attempts += 1
                # Return a special value to indicate recoverable failure
                results = pdb_id, bm, "recoverable_fail"
            else:
                # If the error is not recoverable, we stop trying
                attempts = 10
                log.error(e)
    return results


def parallelize_interactions_request(pdb_bm_tuples):
    """Parallelize data retrieval for several PDB IDs and their ligands.

    Parameters
    ----------
    pdb_bm_tuples : list of tuple
        List of (pdb_id, bm_id) pairs.

    Returns
    -------
    tuple
        A dictionary containing interaction data for every bm in every PDB,
        and a list of recoverable failures.
    """

    url_template = (
        "https://www.ebi.ac.uk/pdbe/graph-api/pdb/"
        "bound_molecule_interactions/{pdb_id}/{bm}"
    )

    tasks = [
        (pdb_id, bm, url_template.format(pdb_id=pdb_id, bm=bm))
        for pdb_id, bm in pdb_bm_tuples
    ]

    max_workers = min(500, len(tasks))

    results_dict, recoverable_fails = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda args: fetch_interaction(*args), tasks)

        for pdb_id, bm, data in results:
            if data == "recoverable_fail":
                recoverable_fails.append((pdb_id, bm))
            else:
                if pdb_id not in results_dict:
                    results_dict[pdb_id] = {}
                results_dict[pdb_id][bm] = data[pdb_id.lower()]

    return results_dict, recoverable_fails


def interactions_to_df(interactions_dict):
    """Transform dict of interactions data into a dataframe.

    Parameters
    ----------
    interactions_dict : dict
        A dictionary with interactions data for every bm in every PDB.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing ligand interactions data.
    """

    rows = [
        [
            pdb_id,
            interactions["begin"]["chain_id"],
            interactions["begin"]["chem_comp_id"],
            bm_data[0]["bm_id"],
            interactions["end"]["chain_id"],
            interactions["end"]["chem_comp_id"],
            interactions["end"]["author_residue_number"],
        ]
        for pdb_id, bound_molecules in interactions_dict.items()
        for bm_data in bound_molecules.values()
        for interactions in bm_data[0].get("interactions", [])
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "pdb_id",
            "ligand_chain_id",
            "ligand_id",
            "bm_id",
            "res_chain_id",
            "resid",
            "resnum",
        ],
    )


def get_interaction_data(ligand_df):
    """Retrieve interaction data for ligands bound to PDBs.

    Parameters
    ----------
    ligand_df : pandas.DataFrame
        A dataframe containing ligand data.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing ligand interactions data.
    """

    aas = [
        "ALA", "CYS", "ASP", "GLU", "PHE",
        "GLY", "HIS", "ILE", "LYS", "LEU",
        "MET", "ASN", "GLN", "PRO", "ARG",
        "SER", "THR", "VAL", "TRP", "TYR",
    ]

    tuples = list(zip(ligand_df["pdb_id"], ligand_df["bm_id"]))
    interact_dict, recoverable_fails = parallelize_interactions_request(tuples)

    log.info("Converting interactions data to dataframe")
    interactions_df = interactions_to_df(interact_dict)

    # Retry recoverable fails
    log.info("Retrying %d recoverable fails", len(recoverable_fails))
    recovered_dict, recoverable_fails = parallelize_interactions_request(
        recoverable_fails
    )

    with open("non_recovered_fails.txt", "w", encoding="utf-8") as f:
        for item in recoverable_fails:
            f.write(f"{item[0]}\t{item[1]}\n")

    log.info("Converting recovered interactions data to dataframe")
    recovered_df = interactions_to_df(recovered_dict)
    recovered_num = (
        recovered_df[["pdb_id", "bm_id"]].drop_duplicates().shape[0]
    )
    log.info(
        "Recovered %d pairs of PDB and bound molecule interactions",
        recovered_num,
    )

    # Append recovered interactions to the main dataframe
    interactions_df = pd.concat(
        [interactions_df, recovered_df],
        ignore_index=True,
    )

    log.info("Filter out interactions with non-residue molecules")
    interactions_df = interactions_df[interactions_df["resid"].isin(aas)]

    # Map SMILES to ligands
    log.info("Merging SMILES to interactions dataframe")
    ligand_dict = (
        ligand_df.drop_duplicates("ligand_id")
        .set_index("ligand_id")["SMILES"]
    )
    interactions_df["SMILES"] = interactions_df["ligand_id"].map(ligand_dict)

    return interactions_df
