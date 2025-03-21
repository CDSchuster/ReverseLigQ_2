import re
import LigandPfamModule


def extract_pdb_ids(string, errtype):
    
    pdb_id = None
    if errtype=="pfam":
        match = re.search(r"Could not get Pfam", string)
        start_index = match.start() + 28
        pdb_id = string[start_index:start_index + 4]

    elif errtype=="ligand":
        match = re.search(r"Could not get ligand", string)
        start_index = match.start() + 30
        pdb_id = string[start_index:start_index + 4]
    
    return pdb_id


def load_pfam_ligand_errors(filename):
    pfam_failed_ids = []
    ligand_failed_ids = []

    with open(filename, "r") as inf:
        for line in inf:

            if "Could not get Pfam" in line:
                pdb_id = extract_pdb_ids(line, "pfam")
                pfam_failed_ids.append(pdb_id)

            elif "Could not get ligand" in line:
                pdb_id = extract_pdb_ids(line, "ligand")
                ligand_failed_ids.append(pdb_id)

    inf.close()
    return pfam_failed_ids, ligand_failed_ids


pfam_ids, ligand_ids = load_pfam_ligand_errors("ligand_pfam_output")
ligand_results = LigandPfamModule.parallelize_pfam_ligand_request(ligand_ids)
ligand_df_ligand, ligand_df_pfam = LigandPfamModule.parallelize_DFs_generation(ligand_ids, ligand_results)
ligand_results = LigandPfamModule.parallelize_pfam_ligand_request(pfam_ids)
pfam_df_ligand, pfam_df_pfam = LigandPfamModule.parallelize_DFs_generation(pfam_ids, ligand_results)

ligand_df_ligand.to_csv("ligand_ligand.csv")
ligand_df_pfam.to_csv("ligand_pfam.csv")
pfam_df_ligand.to_csv("pfam_ligand.csv")
pfam_df_pfam.to_csv("pfam_pfam.csv")