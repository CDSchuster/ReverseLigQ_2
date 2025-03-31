import requests
import concurrent.futures
import os
from rdkit import Chem
import json
import pandas as pd


def get_pdb_ids():
    """
    Retrieves PDB IDs for a specific query several batches

    Returns:
        all_ids (list): all the PDB IDs that contain some bound molecule
    """

    # Load query from file
    with open("query_pdb.json", "r") as file:
        query_template = json.load(file)

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    headers = {"Content-Type": "application/json"}
    
    all_results, found_results = [], True
    
    while found_results:
        # We set the batch-size for every iteration of requests
        query_template["request_options"]["paginate"]["start"] = 0
        query_template["request_options"]["paginate"]["rows"] = 10000

        response = requests.post(url, json=query_template, headers=headers)

        if response.status_code == 200:
            data = response.json()
            # Add the requested data to that which we have until there is no more to request
            if "result_set" in data and data["result_set"]:
                all_results.extend(data["result_set"])
                start += 10000
            else:
                found_results = False  # No more results to fetch
        else:
            print(f"Error: {response.status_code}, {response.text}")
            found_results = False
    
    all_ids = [pdb["identifier"] for pdb in all_results]
    return all_ids


def fetch_url(pdb_id, url):
    """
    Fetch data from a given URL with error handling.
    
    Args:
        pdb_id (str): a valid PDB ID
        url (str): URL to request data with the PDB ID
    
    Returns:
        results (tuple): a tuple containing the PDB ID, URL and the response
    """

    attempts = 0
    # In case we encounter one of the following request errors, we try again up to 5 times
    errors_list = ["HTTPSConnectionPool", "500", "503", "504"]
    while attempts < 5:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Ensure we catch HTTP errors
            # In case of success, we store pdb_id, url and response in json format
            results =  pdb_id, url, response.json()
            attempts = 5 # To break the while loop
        except requests.RequestException as e:
            # We save the error in place of the response, and if the error is in the errors_list, we try again
            results =  pdb_id, url, str(e)
            attempts = attempts + 1 if any(err in str(e) for err in errors_list) else (print(e) or 5)
            # Print the error if we failed 5 times
            if any(err in str(e) for err in errors_list) and attempts==5: print(f"{pdb_id}: {str(e)}")
    return results


def parallelize_pfam_ligand_request(pdb_ids):
    """
    Takes a list of URLs and parallelizes the retrieval
    of ligand and Pfam data for every PDB ID given.
    
    Args:
        pdb_ids (list): a list of valid PDB IDs
    
    Returns:
        results_dict (dict): a dictionary containing the data for every PDB ID and every URL with the following keys:
                           - pdb_id (str): the PDB ID
                           - Pfam_url (str): the Pfam data retrieved for the PDB ID
                           - ligand_url (str): the ligand data retrieved for the PDB ID
    """
    
    URLs = ["https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}",
            "https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}"]

    # We create the tasks to distribute by parallelization
    tasks = [(pdb_id, url.format(pdb_id=pdb_id)) for pdb_id in pdb_ids for url in URLs]

    # Run all requests in parallel
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=500) as executor:
        results = executor.map(lambda args: fetch_url(*args), tasks)

        # Store results in a structured dictionary
        for pdb_id, url, data in results:
            
            if pdb_id not in results_dict:
                results_dict[pdb_id] = {}
            if "pfam" in url:
                try:
                    results_dict[pdb_id]["Pfam_url"] = data[pdb_id.lower()]["Pfam"]
                except:
                    print(f"Could not get Pfam data for {pdb_id}")
            else:
                try:
                    results_dict[pdb_id]["ligand_url"] = data[pdb_id.lower()]
                except:
                    print(f"Could not get ligand data for {pdb_id}")
    
    return results_dict


def get_bmids_rows(pdb_id, results):
    """
    For a certain PDB_ID and its ligands requests results,
    creates a list of lists containing the necessary data
    to be used as rows for a dataframe.
    
    Args:
        pdb_id (str): a valid PDB ID
        results (list): a list of dictionaries, each one containing data of a ligand bound to the PDB ID
    
    Returns:
        new_rows (list): a list of lists where each element is going to be a row in a dataframe
    """

    new_rows = []

    for bm_dict in results: # Iterate the dictionaries of each bm_id
        ligands_list = bm_dict["composition"]["ligands"]

        for ligand in ligands_list: # Now we get the data in a list and add it to the new_rows list
            new_row = [pdb_id, ligand["chain_id"], ligand["chem_comp_id"], bm_dict["bm_id"]]
            new_rows.append(new_row)
               
    return new_rows


def get_pfam_rows(pdb_id, results):
    """
    For a certain PDB_ID and its Pfam requests results,
    creates a list of lists containing the necessary data
    to be used as rows for a dataframe.
    
    Args:
        pdb_id (str): a valid PDB ID
        results (list): a list of dictionaries, each one containing data of a Pfam domain in the PDB ID
    
    Returns:
        new_rows (list): a list of lists where each element is going to be a row in a dataframe
    """

    new_rows = []
    for key, value in results.items(): # Iterate each different Pfam domain type
        pfam_id = key
        pfam_name = value["identifier"]
        # Each domain can have multiple instances in many PDB chains
        # We get all the needed data and add it to the new_rows list
        new_rows = [[pdb_id, instance["chain_id"], pfam_id, pfam_name,
                    instance["start"]["residue_number"],
                    instance["end"]["residue_number"]] for instance in value["mappings"]]
    
    return new_rows


def generate_DFs(subset_pdb_ids, ligand_results):
    """
    Processes a subset of PDB IDs to create ligand and Pfam DataFrames.
    
    Args:
        subset_pdb_ids (list): a list of PDB IDs
        ligand_results (dict): a dictionary containing Pfam and ligand data for every PDB ID
    
    Returns:
        (ligand_df, pfam_df): A tuple containing:
    
    - ligand_df (dataframe): dataframe containing ligand data per PDB ID
    - pfam_df (dataframe): dataframe containing Pfam domain data per PDB ID
                
    """
    
    ligand_df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'ligand_id', "bm_id"])
    pfam_df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'pfam_id', "pfam_name", "start", "end"])

    for pdb_id in subset_pdb_ids:
        urls = ligand_results.get(pdb_id, {})

        try:
            new_bmid_rows = get_bmids_rows(pdb_id, urls.get("ligand_url", []))
            ligand_df = pd.concat([ligand_df, pd.DataFrame(new_bmid_rows, columns=ligand_df.columns)], ignore_index=True)
        except Exception:
            print(f"No ligand data for {pdb_id}")

        try:
            new_pfam_rows = get_pfam_rows(pdb_id, urls.get("Pfam_url", {}))
            pfam_df = pd.concat([pfam_df, pd.DataFrame(new_pfam_rows, columns=pfam_df.columns)], ignore_index=True)
        except Exception:
            print(f"No Pfam domains for {pdb_id}")

    return ligand_df, pfam_df


def parallelize_DFs_generation(pdb_ids, ligand_results):
    """
    Parallelizes ligand and Pfam data DFs generation
    by giving subsets of data to the function `generate_DFs`.
    
    Args:
        subset_pdb_ids (list): a list of PDB IDs
        ligand_results (dict): a dictionary containing Pfam and ligand data for every PDB ID
    
    Returns:
        (ligand_df, pfam_df): A tuple containing
            - ligand_df (dataframe): dataframe containing ligand data per PDB ID
            - pfam_df (dataframe): dataframe containing Pfam domain data per PDB ID
    """

    # Define number of workers
    num_workers = min(os.cpu_count(), 12)  # Limit to 12 CPUs
    chunk_size = max(1, len(pdb_ids) // num_workers)  # Divide data into chunks

    # Split PDB IDs into chunks for parallel processing
    chunks = [pdb_ids[i:i + chunk_size] for i in range(0, len(pdb_ids), chunk_size)]

    # Use ProcessPoolExecutor for multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(generate_DFs, chunks, [ligand_results] * len(chunks)))

    # Merge results from all processes
    ligand_df = pd.concat([res[0] for res in results], ignore_index=True)
    pfam_df = pd.concat([res[1] for res in results], ignore_index=True)

    return ligand_df, pfam_df


def fetch_SMILE_data(het):
    """
    Requests SMILE data for a given PDB hetcode.
    
    Args:
        het (str): Description of param1
    
    Returns:
        result (tuple): a tuple containing the hetcode and the request response
    """

    url = "https://data.rcsb.org/graphql"

    query_compounds = """query molecule ($id: String!) {
        chem_comp(comp_id:$id){rcsb_chem_comp_descriptor {SMILES}}}"""

    attempts = 0
    while attempts < 5: # In case there is a 429 request error, it will keep trying up to 5 times
        try:
            response = requests.post(url, json={"query": query_compounds, "variables": {"id": het}})
            response.raise_for_status()  # Raise an error for bad status codes
            result = (het, response.json()["data"]["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES"])
            attempts = 5 # If successful, we have to break the while loop
        except Exception as e:
            result = (het, None)  # Return None in case of an error
            attempts = attempts + 1 if ("429" in str(e)) else (print(e) or 5)
            if ("429" in str(e)) and attempts==5: print(f"{het}: {str(e)}")
            
        return result


def parallelize_SMILE_request(ligand_df):
    """
    Parallelizes requests for SMILEs data to RCSB.
    
    Args:
        ligand_df (dataframe): dataframe containing ligand data (including hetcodes)
    
    Returns:
        ligand_df (dataframe): the same dataframe as the input but with the SMILEs in a new column
    """
    
    raw_smiles_data = {}
    # Use ThreadPoolExecutor to parallelize requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_SMILE_data, list(set(ligand_df.ligand_id)))

    # Store results in dictionary
    for het, data in results:
        raw_smiles_data[het] = data
    
    ligand_df["SMILES"] = ligand_df["ligand_id"].map(raw_smiles_data)
    
    return ligand_df


def count_atoms(smiles):
    """
    Counts the number of atoms for a SMILE.
    
    Args:
        smiles (str): a SMILE code
    
    Returns:
        atoms_num (int): the number of atoms in the SMILE molecule
    """

    if not isinstance(smiles, str):  # Check for NaN or non-string values
        return None  # Mark invalid SMILES
    mol = Chem.MolFromSmiles(smiles)
    atoms_num = mol.GetNumAtoms() if mol else None  # Mark invalid SMILES
    return atoms_num


def filter_small_ligands(ligand_df):
    """
    Filters rows of ligands_df if ligands are smaller than 10 atoms.
    
    Args:
        ligand_df (dataframe): Description of param1
    
    Returns:
        ligand_df (dataframe): the input dataframe with rows filtered based on the ligands number of atoms
    """

    pdb_ligand_num = len(set(ligand_df.pdb_id))
    ligand_df["num_atoms"] = ligand_df["SMILES"].apply(count_atoms)
    ligand_df = ligand_df[ligand_df["num_atoms"].notna() &
                         (ligand_df["num_atoms"] >= 10)].drop(columns=["num_atoms"])
    
    filtered_pdb_num = pdb_ligand_num - len(set(ligand_df.pdb_id))
    print(f"PDB IDs filtered after small ligand filtering: {filtered_pdb_num}")

    return ligand_df


def run_requests(pdb_ids):
    """
    For a given list of PDB IDs request ligand and Pfam data, as well as the SMILEs.
    
    Args:
        pdb_ids (list): a list of valid PDB IDs
    
    Returns:
        (ligand_df, pfam_df): A tuple containing
            - ligand_df (dataframe): dataframe containing ligand data per PDB ID
            - pfam_df (dataframe): dataframe containing Pfam domain data per PDB ID
    """

    ligand_results = parallelize_pfam_ligand_request(pdb_ids)
    ligand_df, pfam_df = parallelize_DFs_generation(pdb_ids, ligand_results)
    ligand_df = parallelize_SMILE_request(ligand_df)
    return ligand_df, pfam_df


def get_ligand_pfam_data():
    """
    Gets all PDB IDs with ligands and requests ligand and Pfam data for those.
    
    Returns:
        results_dict (dict): a dictionary containing the ligand dataframe, Pfam dataframe and PDB IDs
    """

    pdb_ids = get_pdb_ids()
    print(f"Total PDB IDs: {len(pdb_ids)}")

    ligand_df, pfam_df = run_requests(pdb_ids)
    print(f"Successful PDB IDs ligand requests: {len(set(ligand_df.pdb_id))}")
    print(f"Successful PDB IDs Pfam requests: {len(set(pfam_df.pdb_id))}")
    
    results_dict = {"ligand_df":ligand_df, "pfam_df":pfam_df, "pdb_ids":pdb_ids}

    return results_dict


def retry_lp_request(ligand_df, pfam_df, pdb_ids):
    """
    Requests Pfam and ligand data for PDB IDs that failed request originally.
    Data that could be recovered is added to the input dataframes.
    
    Args:
        ligand_df (dataframe): dataframe containing ligand data
        pfam_df (dataframe): dataframe containing Pfam domains data
    
    Returns:
        (ligand_df, pfam_df): A tuple containing
            - ligand_df (dataframe): dataframe containing ligand data per PDB ID
            - pfam_df (dataframe): dataframe containing Pfam domain data per PDB ID
    """

    ligand_pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id not in list(ligand_df.pdb_id)]
    pfam_pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id not in list(pfam_df.pdb_id)]
    
    print(f"{len(ligand_pdb_ids)} ligand PDB IDs failed request")
    print(f"{len(pfam_pdb_ids)} Pfam PDB IDs failed request")

    ligand_df_ligand, ligand_df_pfam = run_requests(ligand_pdb_ids)
    pfam_df_ligand, pfam_df_pfam = run_requests(pfam_pdb_ids)

    print(f"Recovered {len(set(ligand_df_ligand.pdb_id))} ligand PDB IDs")
    print(f"Recovered {len(set(pfam_df_pfam.pdb_id))} Pfam PDB IDs")

    ligand_df = pd.concat([ligand_df, ligand_df_ligand], ignore_index=True)
    pfam_df = pd.concat([pfam_df, pfam_df_pfam], ignore_index=True)

    return ligand_df, pfam_df