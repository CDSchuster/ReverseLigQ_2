import requests
import os

url = "https://search.rcsb.org/rcsbsearch/v2/query"
headers = {"Content-Type": "application/json"}

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


def get_pdb_ids(start, batch_size, query_template, url, headers):
    """Retrieves PDB IDs for a specific query in custom batch sizes"""
    
    all_results, found_results = [], True

    while found_results:
        print(start)
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


def download_pdb(pdb_id, save_dir):

    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"  # PDB file URL
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")  # File save path

    # Request the PDB file
    response = requests.get(pdb_url)

    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)  # Save file
        #print(f"Downloaded: {pdb_id}.pdb")
    else:
        print(f"Failed to download: {pdb_id}.pdb (Status Code: {response.status_code})")


def download_multiple_pdbs(pdb_ids, save_dir):

    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    for k, pdb_id in enumerate(pdb_ids[:20]):

        #if k%100==0: print(k)
        download_pdb(pdb_id, save_dir)

    print(f"Download complete! Files are saved in '{save_dir}/'.")


pdb_ids = get_pdb_ids(start=0, batch_size=10000, query_template=query_template, url=url, headers=headers)
download_multiple_pdbs(pdb_ids, "pdb_files")