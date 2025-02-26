import requests
import concurrent.futures
import os
import time
import pickle 
import pandas as pd


rcsb_base = "https://data.rcsb.org/rest/v1/core"
uniprot_base = "https://rest.uniprot.org/uniprotkb"


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


def get_ligand_binding_chains(pdb_id, full_data):
    """For a given PDB ID, gets its ligands and chains"""

    url= f"https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}/"
    response = requests.get(url)
    data = response.json()
    ligands = []
    full_data[pdb_id] = {}
    for dic in data[pdb_id.lower()]:
        ligands_list = dic["composition"]["ligands"]
        for k in ligands_list:
            full_data[pdb_id][k["chain_id"]] = {}
            full_data[pdb_id][k["chain_id"]]["ligand"]=k["chem_comp_id"]
    return full_data


def get_uniprot_accession(pdb_id, full_data):
    """For a given PDB ID and a previously created dictionary,
    returns an updated dictionary with the Uniprot ID for each chain"""

    url = f"https://www.ebi.ac.uk/pdbe/graph-api/mappings/uniprot/{pdb_id}"
    response = requests.get(url)
    data = response.json()[pdb_id.lower()]["UniProt"]
    keys = list(data.keys())
    for k in keys:
        mappings = data[k]["mappings"]
        for mp in mappings:
            if mp["chain_id"] in full_data[pdb_id].keys():
                chain=mp["chain_id"]
                full_data[pdb_id][chain]["UniProt_ID"] = k
    return full_data


def extract_pfam_chains(pdb_id):
    """Gets the Pfam domains of every PDB chain
    and returns a dictionary of those domains"""

    url = f"https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}"
    response = requests.get(url)
    data = response.json()

    chain_pfam_map = {}
    
    for id, pdb_data in data.items():
        if "Pfam" in pdb_data:
            for pfam_id, pfam_data in pdb_data["Pfam"].items():
                for mapping in pfam_data.get("mappings", []):
                    chain_id = mapping["chain_id"]
                    if chain_id not in chain_pfam_map:
                        chain_pfam_map[chain_id] = []
                    chain_pfam_map[chain_id].append(pfam_id)
    
    return chain_pfam_map


def merge_pfam_with_chains(chain_dict, pfam_data):
    """Merges a previously created dictionary to a
       dictionary of Pfam domains for every chain"""

    for pdb_id, chains in chain_dict.items():
        for chain_id, chain_info in chains.items():
            if chain_id in pfam_data:
                chain_info["pfam_domains"] = pfam_data[chain_id]
            else:
                chain_info["pfam_domains"] = []
    
    return chain_dict


def get_PDB_data(pdb_ids):
    """Gets ligands, uniprot ID, chains and Pfam domains
       for every PDB ID that is given in a list and returns
       it as a dictionary of dictionaries"""

    full_data = {}

    for pdb_id in pdb_ids[:50]:
        full_data = get_ligand_binding_chains(pdb_id, full_data)
        full_data = get_uniprot_accession(pdb_id, full_data)
        pfam_data = extract_pfam_chains(pdb_id)
        full_data = merge_pfam_with_chains(full_data, pfam_data)
    
    return full_data


pdb_ids = get_pdb_ids(start=0, batch_size=10000, query_template=query_template)



start = time.time()


# List of API endpoints
URL_TEMPLATES = [
    "https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}",
    "https://www.ebi.ac.uk/pdbe/graph-api/mappings/uniprot/{pdb_id}",
    "https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}",
]

# Generate all URL tasks (combinations of PDB IDs and URLs)
tasks = [(pdb_id, url.format(pdb_id=pdb_id)) for pdb_id in pdb_ids[:1000] for url in URL_TEMPLATES]

def fetch_url(pdb_id, url):
    """Fetch data from a given URL with error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Ensure we catch HTTP errors
        return pdb_id, url, response.json()
    except requests.RequestException as e:
        return pdb_id, url, str(e)

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
        results_dict[pdb_id][url] = data


# Save results as a Pickle file
with open("pdb_results.pkl", "wb") as f:
    pickle.dump(results_dict, f)

print(time.time() - start)