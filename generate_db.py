import requests
import pandas as pd
from tqdm import tqdm


def get_uniprot_from_pdb(pdb_id):
    """Receives a list of PDB IDs and gets Uniprot IDs for all the chains"""

    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    response = requests.get(url)
    data = response.json()
    uniprot_ids = list(data.get(pdb_id.lower(), {}).get("UniProt", {}).keys())
    return uniprot_ids


def get_chembl_from_uniprot(uniprot_id):
    """Receives a list of Uniprot IDs and gets the
       corresponding target components from Chembl"""
    
    url = f"https://www.ebi.ac.uk/chembl/api/data/target?target_components__accession={uniprot_id}"
    headers = {"Accept": "application/json"}  # Ensure JSON response
    response = requests.get(url, headers=headers)
    data = response.json()
    return [entry["target_chembl_id"] for entry in data.get("targets", [])]


def get_chembl_ligands(chembl_target_id):
    """Get the ligands for chembl target proteins"""

    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={chembl_target_id}"
    response = requests.get(url)
    ligands = []
    data = response.json()
    for entry in data.get("activities", []):
        ligand_chembl_id = entry.get("molecule_chembl_id")
        if ligand_chembl_id:
            ligands.append(ligand_chembl_id)
    return list(set(ligands))


def get_smiles_from_chembl(chembl_ligand_id):
    """Get the smiles for a set of ligands"""

    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_ligand_id}.json"
    response = requests.get(url)
    data = response.json()
    return data.get("molecule_structures", {}).get("canonical_smiles", "N/A")


def get_pfam_from_interpro(uniprot_id):
    """Gets the Pfam domain accessions for a list of uniprot sequences"""

    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{uniprot_id}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()
    results = data.get("results")
    pfam_domains = [res.get("metadata").get("accession") for res in results]
    return pfam_domains


def generate_table(pdbs):
    """Generate a dataset table from a list of PDB IDs"""

    results = []

    # Process each PDB ID
    for pdb_id in tqdm(pdbs, desc="Processing PDBs"):
        
        uniprot_ids = get_uniprot_from_pdb(pdb_id)
        
        for uniprot_id in uniprot_ids:

            chembl_targets = get_chembl_from_uniprot(uniprot_id)
            pfam_domains = get_pfam_from_interpro(uniprot_id)
            
            for chembl_target in chembl_targets:
                ligands = get_chembl_ligands(chembl_target)
                
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

    df = pd.DataFrame(results)
    df.to_csv("pdb_chembl_ligands_pfam.csv", index=False)
    return df

# List of PDB IDs (replace with your own)
pdb_ids = ["4HHB"]#, "1A4G", "5XNL"]  # Example PDB IDs 

df = generate_table(pdb_ids)

df