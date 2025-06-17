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
        try: # Save the results if the request is successful
            response = requests.get(url, timeout=120)
            response.raise_for_status()  # Ensure we catch HTTP errors
            results = pdb_id, bm, response.json()
            attempts = 10
        except requests.RequestException as e:
            results =  pdb_id, bm, str(e) 
            if any(err in str(e) for err in errors_list):
                attempts += 1 # If the error is recoverable, we retry
            else: # If the error is not recoverable, we stop trying
                attempts = 10
                log.error(e)
            # Save the error if it failed 5 times but it is recoverable
            if any(err in str(e) for err in errors_list):
                # Return a special value to indicate recoverable failure
                results = pdb_id, bm, "recoverable_fail"
    return results


def parallelize_interactions_request(tuples):
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

    # We create the tasks to distribute by parallelization
    tasks = [(tup[0], tup[1], url_template.format(pdb_id=tup[0], bm=tup[1])) for tup in tuples]
    
    # Set high concurrency with max_workers (adjust based on system performance)
    MAX_WORKERS = min(500, len(tasks))  # Limits max workers to prevent overload

    # Run all requests in parallel
    results_dict, recoverable_fails = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(lambda args: fetch_interaction(*args), tasks)
        
        # Store results in a structured dictionary
        for pdb_id, bm, data in results:
            if data == "recoverable_fail":
                recoverable_fails.append((pdb_id, bm))
            else:
                if pdb_id not in results_dict:
                    results_dict[pdb_id] = {}
                results_dict[pdb_id][bm] = data[pdb_id.lower()]

    return results_dict, recoverable_fails


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

    # First attempt to retrieve interactions data for all ligands
    tuples = list(zip(ligand_df['pdb_id'], ligand_df['bm_id']))
    interact_dict, recoverable_fails = parallelize_interactions_request(tuples)
    log.info("Converting interactions data to dataframe")
    interactions_df = interactions_to_DF(interact_dict)

    # Retry for recoverable fails
    log.info(f"Retrying {len(recoverable_fails)} recoverable fails")
    recovered_interact_dict, recoverable_fails = parallelize_interactions_request(recoverable_fails)

    with open("non_recovered_fails.txt", "w") as f:
        for item in recoverable_fails:
            f.write(f"{item[0]}\t{item[1]}\n")
    
    log.info("Converting recovered interactions data to dataframe")
    recovered_interactions_df = interactions_to_DF(recovered_interact_dict)
    recovered_num = recovered_interactions_df[['pdb_id', 'bm_id']].drop_duplicates().shape[0]
    log.info(f"Recovered {recovered_num} pairs of PDB and bound molecule interactions")

    # Append recovered interactions to the main dataframe
    interactions_df = pd.concat([interactions_df, recovered_interactions_df], ignore_index=True)
    log.info("Filter out interactions with non-residue molecules")
    interactions_df = interactions_df[interactions_df['resid'].isin(AAs)] # Keep only interactions with amino acids
    # Map SMILES to ligands in interactions data
    log.info("Merging SMILEs to interactions dataframe")
    # Make a dictionary mapping from ligand_id to SMILES
    ligand_dict = ligand_df.drop_duplicates('ligand_id').set_index('ligand_id')['SMILES']
    # Map it into the interactions_df
    interactions_df['SMILES'] = interactions_df['ligand_id'].map(ligand_dict)

    return interactions_df