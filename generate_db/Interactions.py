"""
The functions in this module are designed to request ligand interactions to proteins and store them in a dataframe
"""

import requests
import concurrent.futures
import pandas as pd
import logging


log = logging.getLogger("generateDB_log")


def fetch_interaction(pdb_id, bm, url):
    """Gets interaction data for a given pdb and molecule.

    Parameters
    ----------
    pdb_id : str
        a valid PDB ID
    bm : str
        bound molecule ID
    url : str
        URL to request ligand interactions data

    Returns
    -------
    results : tuple
        contains PDB ID, bmid (bound molecule ID), and request response
    """

    attempts = 0
    errors_list = ["HTTPSConnectionPool", "500", "502", "503", "504"]
    while attempts < 10:
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()  # Ensure we catch HTTP errors
            results = pdb_id, bm, response.json()
            attempts = 10
        except requests.RequestException as e:
            results =  pdb_id, bm, str(e)
            if any(err in str(e) for err in errors_list):
                attempts += 1
            else:
                attempts = 10
                log.error(e)
            if any(err in str(e) for err in errors_list) and attempts==10:
                log.error(f"{pdb_id}: {str(e)}")
    return results


def parallelize_interactions_request(ligand_df):
    """Parallelizes data retrieval for several pdb_ids and their ligands.

    Parameters
    ----------
    ligand_df : dataframe
        dataframe containing ligand data

    Returns
    -------
    results_dict : dict
        a dictionary containing interaction data for every bm in every PDB
    """

    url_template = "https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_molecule_interactions/{pdb_id}/{bm}"

    tuples = list(zip(ligand_df['pdb_id'], ligand_df['bm_id']))
    # We create the tasks to distribute by parallelization
    tasks = [(tup[0], tup[1], url_template.format(pdb_id=tup[0], bm=tup[1])) for tup in tuples]
    
    # Set high concurrency with max_workers (adjust based on system performance)
    MAX_WORKERS = min(500, len(tasks))  # Limits max workers to prevent overload

    # Run all requests in parallel
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(lambda args: fetch_interaction(*args), tasks)
        
        # Store results in a structured dictionary
        for pdb_id, bm, data in results:
            
            if pdb_id not in results_dict:
                results_dict[pdb_id] = {}
            try:
                results_dict[pdb_id][bm] = data[pdb_id.lower()]
            except:
                log.error(f"Could not get interaction data for {pdb_id}/{bm}")

    return results_dict


def interactions_to_DF(interactions_dict):
    """Transforms dict of interactions data into DF.

    Parameters
    ----------
    interactions_dict : dict
        a dictionary with interactions data for every bm in every PDB

    Returns
    -------
    interactions_df : dataframe
        dataframe containing ligand interactions data
    """

    rows = [
        [
            pdb_id,
            interactions["begin"]["chain_id"],
            interactions["begin"]["chem_comp_id"],
            bm_data[0]["bm_id"],
            interactions["end"]["chain_id"],
            interactions["end"]["chem_comp_id"],
            interactions["end"]["author_residue_number"]
        ]
        for pdb_id, bound_molecules in interactions_dict.items()
        for bm_data in bound_molecules.values()
        for interactions in bm_data[0].get("interactions", [])  # Safely handle missing "interactions"
    ]

    interactions_df = pd.DataFrame(rows, columns=["pdb_id", "ligand_chain_id", "ligand_id", "bm_id","res_chain_id", "resid", "resnum"])

    return interactions_df


def get_interaction_data(ligand_df):
    """Retrieves interactions data for ligands bound to PDBs and returns it as a dataframe.

    Parameters
    ----------
    ligand_df : dataframe
        a dataframe containing ligand data

    Returns
    -------
    interactions_df : dataframe
        dataframe containing ligand interactions data
    """

    AAs = ["ALA", "CYS", "ASP", "GLU", "PHE",
           "GLY", "HIS", "ILE", "LYS", "LEU",
           "MET", "ASN", "GLN", "PRO", "ARG",
           "SER", "THR", "VAL", "TRP", "TYR"]

    interact_dict = parallelize_interactions_request(ligand_df)
    log.info("Converting interactions data to dataframe")
    interactions_df = interactions_to_DF(interact_dict)
    log.info("Filter out interactions with non-residue molecules")
    interactions_df = interactions_df[interactions_df['resid'].isin(AAs)] # Keep only interactions with amino acids
    # Map SMILES to ligands in interactions data
    log.info("Merging SMILEs to interactions dataframe")
    interactions_df = interactions_df.merge(ligand_df[['ligand_id', 'SMILES']], on='ligand_id', how='left')

    return interactions_df