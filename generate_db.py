import requests
import concurrent.futures
import os
import time
import pickle 
import pandas as pd


QUERY = {
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
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Ensure we catch HTTP errors
        results =  pdb_id, url, response.json()
    except requests.RequestException as e:
        results =  pdb_id, url, str(e)
    return results


def retrieve_data(pdb_ids, URLs):
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
                results_dict[pdb_id]["Pfam_url"] = data[pdb_id.lower()]["Pfam"]
            else:
                results_dict[pdb_id]["ligand_url"] = data[pdb_id.lower()]
    
    return results_dict


def get_bmids_df(pdb_id, results):
    """For a certain PDB_ID and its ligands requests results,
    creates a list of lists containing basic info to create a DF"""

    new_rows = []

    for bm_dict in results: # Iterate the dictionaries of each bm_id
        ligands_list = bm_dict["composition"]["ligands"]

        for ligand in ligands_list: # Now we get the data in a list and add it to the new_rows list
            new_row = [pdb_id, ligand["chain_id"], ligand["chem_comp_id"], bm_dict["bm_id"]]
            new_rows.append(new_row)
               
    return new_rows


def get_pfam_df(pdb_id, results):
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


def generate_DFs(ligand_results):
    """Takes a PDB Pfam and ligands requests results and
    creates 2 dataframes containing all the data needed"""

    # Initialize the dataframes
    ligand_df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'ligand_id', "bm_id"])
    pfam_df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'pfam_id', "pfam_name", "start", "end"])

    for pdb_id, urls in ligand_results.items():

        # For every pdb_id, we get multiple lists of data
        # to be added as rows to the coorespondent dataframe
        new_bmid_rows = get_bmids_df(pdb_id, urls["ligand_url"])
        new_pfam_rows = get_pfam_df(pdb_id, urls["Pfam_url"])

        ligand_df = pd.concat([ligand_df, pd.DataFrame(new_bmid_rows, columns=ligand_df.columns)], ignore_index=True)
        pfam_df = pd.concat([pfam_df, pd.DataFrame(new_pfam_rows, columns=pfam_df.columns)], ignore_index=True)

    return ligand_df, pfam_df
            

def main():
    pdb_ids = get_pdb_ids(start=0, batch_size=10000, query_template=QUERY)
    ligand_results = retrieve_data(pdb_ids[:20], URL_TEMPLATES[:2])
    ligand_df, pfam_df = generate_DFs(ligand_results)
    

main()