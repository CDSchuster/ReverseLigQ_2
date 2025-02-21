import requests
import pandas as pd
from tqdm import tqdm


# Function to get UniProt ID from PDB ID (Fix for PDB -> ChEMBL mapping)
def get_uniprot_from_pdb(pdb_id):
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        uniprot_ids = list(data.get(pdb_id.lower(), {}).get("UniProt", {}).keys())
        return uniprot_ids if uniprot_ids else []
    return []


# Function to get ChEMBL Target ID from UniProt ID (Corrected)
def get_chembl_from_uniprot(uniprot_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/target?target_components__accession={uniprot_id}"
    headers = {"Accept": "application/json"}  # Ensure JSON response
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return [entry["target_chembl_id"] for entry in data.get("targets", [])]
    return []


# Function to get ligands (compounds) from a ChEMBL target
def get_chembl_ligands(chembl_target_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={chembl_target_id}"
    response = requests.get(url)
    ligands = []
    if response.status_code == 200:
        data = response.json()
        for entry in data.get("activities", []):
            ligand_chembl_id = entry.get("molecule_chembl_id")
            if ligand_chembl_id:
                ligands.append(ligand_chembl_id)
    return list(set(ligands))  # Remove duplicates


# Function to get SMILES from a ChEMBL compound
def get_smiles_from_chembl(chembl_ligand_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_ligand_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("molecule_structures", {}).get("canonical_smiles", "N/A")
    return "N/A"


# Function to get PFAM domains from UniProt
def get_pfam_from_uniprot(uniprot_id):
    url = f"https://www.ebi.ac.uk/proteins/api/features/{uniprot_id}"
    response = requests.get(url, headers={"Accept": "application/json"})
    pfam_domains = []
    if response.status_code == 200:
        data = response.json()
        print("data", data)
        for feature in data.get("features", []):
            if feature["type"] == "DOMAIN" and "Pfam" in feature["description"]:
                pfam_domains.append(feature["description"])
    return pfam_domains


# Function to get PFAM domains from InterPro for a specific UniProt protein
def get_pfam_from_interpro(uniprot_id):
    url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/{uniprot_id}"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)

    # Check for HTTP errors
    if response.status_code != 200:
        print(f"⚠️ Error {response.status_code} for UniProt ID {uniprot_id}")
        return ["N/A"]

    # Check if response is empty
    if not response.text.strip():
        print(f"⚠️ Empty response for UniProt ID {uniprot_id}")
        return ["N/A"]

    try:
        data = response.json()
        pfam_domains = []

        # Extract PFAM matches
        for entry in data.get("matches", []):
            if entry.get("signature_database") == "pfam":
                pfam_domains.append(entry.get("entry", {}).get("name", "Unknown"))

        return pfam_domains if pfam_domains else ["N/A"]

    except requests.exceptions.JSONDecodeError:
        print(f"⚠️ JSON decode error for UniProt ID {uniprot_id}. Response:\n{response.text}")
        return ["N/A"]

# List of PDB IDs (replace with your own)
pdb_ids = ["4HHB"]#, "1A4G", "5XNL"]  # Example PDB IDs 

# Store results
results = []

# Process each PDB ID
for pdb_id in tqdm(pdb_ids, desc="Processing PDBs"):
    print(pdb_id)
    uniprot_ids = get_uniprot_from_pdb(pdb_id)
    print(uniprot_ids)
    for uniprot_id in uniprot_ids:
        chembl_targets = get_chembl_from_uniprot(uniprot_id)
        print("hola", chembl_targets)
        pfam_domains = get_pfam_from_interpro(uniprot_id)
        for chembl_target in chembl_targets:
            ligands = get_chembl_ligands(chembl_target)
            print("chau", ligands)
            for ligand in ligands:
                smiles = get_smiles_from_chembl(ligand)
                results.append({
                    "PDB_ID": pdb_id,
                    "UniProt_ID": uniprot_id,
                    "ChEMBL_Target_ID": chembl_target,
                    "Ligand": ligand,
                    "SMILES": smiles,
                    "PFAM_Domains": ", ".join(pfam_domains) if pfam_domains else "N/A"
                })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv("pdb_chembl_ligands_pfam.csv", index=False)

# Display DataFrame
print(df)
