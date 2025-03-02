import requests
import concurrent.futures
import os
import time
import pickle 
import pandas as pd


query_template = {
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


def get_pdb_ids(start, batch_size, query_template):
    """Retrieves PDB IDs for a specific query in custom batch sizes"""

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    headers = {"Content-Type": "application/json"}
    
    all_results, found_results = [], True

    while found_results:
        
        query_template["request_options"]["paginate"]["start"] = start
        query_template["request_options"]["paginate"]["rows"] = batch_size

        response = requests.post(url, json=query_template, headers=headers)

        if response.status_code == 200:
            data = response.json()
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
            results_dict[pdb_id][url] = data[pdb_id]
    
    return results_dict


def get_bmids(ligand_results):

    df = pd.DataFrame(columns=['pdb_id', 'chain_id', 'ligand_id', "bm_id"])

    for pdb_id, urls in ligand_results.items():
        
        for url, pdb_data in urls.items():
            print(urls.keys())
            if "bound_excluding_branched" in url:

                for bm_dict in pdb_data:
                    ligands_list = bm_dict["composition"]["ligands"]
                    for ligand in ligands_list:
                        new_row = [pdb_id, ligand["chain_id"], ligand["chem_comp_id"], bm_dict["bm_id"]]
                        df.loc[len(df)] = new_row

                    #for key, value in bm_dict.items():
                    #    print(value)

    print(df)              


            # for pdb_key, bm_list in pdb_data.items():
            #     result[pdb_id] = [
            #         {"bm_id": bm["bm_id"], "chain_ids": [ligand["chain_id"] for ligand in bm["composition"]["ligands"]]}
            #         for bm in bm_list
            #     ]

    #return result


URL_TEMPLATES = ["https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}",
                 "https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}",
                 "https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_molecule_interactions/{pdb_id}/{bm}"]


def main():
    #pdb_ids = get_pdb_ids(start=0, batch_size=10000, query_template=query_template)
    pdb_ids = ["1cqx", "2pgh"]
    ligand_results = retrieve_data(pdb_ids[:3], URL_TEMPLATES[:2])
    #print(ligand_results)
    ligand_bmids = get_bmids(ligand_results)
    #print(ligand_bmids)
    #print(ligands_chains)
    # with open("pdb_results.pkl", "wb") as f:
    #     pickle.dump(results, f)

main()