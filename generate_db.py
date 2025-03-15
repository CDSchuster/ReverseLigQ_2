import requests
import concurrent.futures
import os
import time
import pickle 
import pandas as pd


QUERY_PDB = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.comp_id",
                            "operator": "exists",
                            "negation": False
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_nonpolymer_instance_annotation.type",
                            "operator": "exact_match",
                            "value": "HAS_NO_COVALENT_LINKAGE",
                            "negation": False
                        }
                    }
                ],
                "label": "nested-attribute"
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "entity_poly.rcsb_entity_polymer_type",
                    "value": "Protein",
                    "operator": "exact_match"
                }
            }
        ],
        "label": "text"
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,  # Will be updated dynamically
            "rows": 10000  # Fetch 10,000 results per request
        },
        "results_content_type": [
            "experimental"
        ],
        "sort": [
            {
                "sort_by": "score",
                "direction": "desc"
            }
        ],
        "scoring_strategy": "combined"
    }
}

QUERY_COMPOUNDS = """query molecule ($id: String!) {
    chem_comp(comp_id:$id){
        rcsb_chem_comp_descriptor {
            SMILES
            
        }
    }
}"""

URL_TEMPLATES = ["https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}",
                 "https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}",
                 "https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_molecule_interactions/{pdb_id}/{bm}"]


def get_pdb_ids(start, batch_size, query_template):
    """Retrieves PDB IDs for a specific query in custom batch sizes"""

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    headers = {"Content-Type": "application/json"}
    
    all_results, found_results = [], True

    while found_results:
        # We set the batch-size for every iteration of requests
        query_template["request_options"]["paginate"]["start"] = start
        query_template["request_options"]["paginate"]["rows"] = batch_size

        response = requests.post(url, json=query_template, headers=headers)

        if response.status_code == 200:
            data = response.json()
            # Add the requested data to that which we have until there is no more to request
            if "result_set" in data and data["result_set"]:
                all_results.extend(data["result_set"])
                start += batch_size
            else:
                found_results = False  # No more results to fetch
        else:
            print(f"Error: {response.status_code}, {response.text}")
            found_results = False
    
    all_ids = [pdb["identifier"] for pdb in all_results]
    return all_ids


def fetch_url(pdb_id, url):
    """Fetch data from a given URL with error handling."""
    attempts = 0
    connection_timeout = "HTTPSConnectionPool(host='www.ebi.ac.uk', port=443): Read timed out. (read timeout=10)"
    while attempts < 5:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Ensure we catch HTTP errors
            results =  pdb_id, url, response.json()
            attempts = 5
        except requests.RequestException as e:
            results =  pdb_id, url, str(e)
            if e == connection_timeout:
                attempts +=1
            else:
                attempts = 5
                print(f"{pdb_id} could not be retrieved for {url.split("/")[-2]}")
                print(e)
            if e == connection_timeout and attempts==5: print(f"Connection timeout for {pdb_id}")
    return results


def parallelize_pfam_ligand_request(pdb_ids, URLs):
    """Takes a list of URLs and retrieves data for every PDB ID given"""
    
    # We create the tasks to distribute by parallelization
    tasks = [(pdb_id, url.format(pdb_id=pdb_id)) for pdb_id in pdb_ids for url in URLs]
    
    # Set high concurrency with max_workers (adjust based on system performance)
    MAX_WORKERS = min(100, len(tasks))  # Limits max workers to prevent overload

    # Run all requests in parallel
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
    """For a certain PDB_ID and its ligands requests results,
    creates a list of lists containing basic info to create a DF"""

    new_rows = []

    for bm_dict in results: # Iterate the dictionaries of each bm_id
        ligands_list = bm_dict["composition"]["ligands"]

        for ligand in ligands_list: # Now we get the data in a list and add it to the new_rows list
            new_row = [pdb_id, ligand["chain_id"], ligand["chem_comp_id"], bm_dict["bm_id"]]
            new_rows.append(new_row)
               
    return new_rows


def get_pfam_rows(pdb_id, results):
    """For a certain PDB_ID and its Pfam requests results, creates
       a list of lists containing basic info to create a DF"""

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
    """Processes a subset of PDB IDs to create ligand and Pfam DataFrames."""
    
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
    """Parallelizes DFs generation"""

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
    """Requests SMILE data for a given PDB hetcode"""

    url = "https://data.rcsb.org/graphql"
    try:
        response = requests.post(url, json={"query": QUERY_COMPOUNDS, "variables": {"id": het}})
        response.raise_for_status()  # Raise an error for bad status codes
        result = (het, response.json()["data"]["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES"])
    except requests.RequestException as e:
        print(f"Error fetching {het}: {e}")
        result = (het, None)  # Return None in case of an error
    return result


def parallelize_SMILE_request(ligand_df):
    """Parallelizes requests for SMILEs data to RCSB"""
    
    raw_smiles_data = {}
    # Use ThreadPoolExecutor to parallelize requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(fetch_SMILE_data, list(set(ligand_df.ligand_id)))  # Limit to first 10 ligands

    # Store results in dictionary
    for het, data in results:
        raw_smiles_data[het] = data
    
    ligand_df["SMILES"] = ligand_df["ligand_id"].map(raw_smiles_data)
    
    return ligand_df


def fetch_interaction(pdb_id, bm, url):
    """Gets interaction data for a given pdb and molecule"""

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()  # Ensure we catch HTTP errors
        results = pdb_id, bm, response.json()
    except requests.RequestException as e:
        results =  pdb_id, bm, str(e)
        print(f"Data could not be retrieved for {url}")
        print(e)
    return results


def parallelize_interactions_request(ligand_df):
    """Parallelizes data retrieval for several pdb_ids and their ligands"""

    tuples = list(zip(ligand_df['pdb_id'], ligand_df['bm_id']))
    # We create the tasks to distribute by parallelization
    tasks = [(tup[0], tup[1], URL_TEMPLATES[2].format(pdb_id=tup[0], bm=tup[1])) for tup in tuples]
    
    # Set high concurrency with max_workers (adjust based on system performance)
    MAX_WORKERS = min(500, len(tasks))  # Limits max workers to prevent overload

    # Run all requests in parallel
    results_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(lambda args: fetch_interaction(*args), tasks)
        
        # Store results in a structured dictionary
        for pdb_id, bm, data in results:
            #print(data.keys(), pdb_id)
            
            if pdb_id not in results_dict:
                results_dict[pdb_id] = {}
            try:
                results_dict[pdb_id][bm] = data[pdb_id.lower()]
            except:
                print(f"Could not get interaction data for {pdb_id}/{bm}")

    return results_dict


def main():
    pdb_ids = get_pdb_ids(start=0, batch_size=10000, query_template=QUERY_PDB)
    ligand_results = parallelize_pfam_ligand_request(pdb_ids, URL_TEMPLATES[:2])
    ligand_df, pfam_df = parallelize_DFs_generation(pdb_ids, ligand_results)
    ligand_df = parallelize_SMILE_request(ligand_df)
    interact_dict = parallelize_interactions_request(ligand_df)
    return ligand_df, pfam_df


# Run the main function
ligand_df, pfam_df = main()
#ligand_df.to_csv("ligand.csv")
#pfam_df.to_csv("pfam.csv")