"""
This module generates a database of ligand interactions to Pfam domains by using PDB data
"""

import pandas as pd
from LigandPfamModule import get_ligand_pfam_data, filter_small_ligands, retry_lp_request
from Interactions import get_interaction_data


def main():

    # Step 1: Retrieve ligand and Pfam data
    results_dict = get_ligand_pfam_data()
    
    # Step 2: Retry failed ligand and Pfam data requests
    results_dict["ligand_df"], results_dict["pfam_df"] = retry_lp_request(results_dict["ligand_df"],
                                                                          results_dict["pfam_df"],
                                                                          results_dict["pdb_ids"])
    
    # Step 3: filter ligand dataframe rows based on the number of atoms in the ligand
    results_dict["ligand_df"] = filter_small_ligands(results_dict["ligand_df"])

    # Step 4: Retrieve interaction data
    results_dict["interactions_df"] = get_interaction_data(results_dict["ligand_df"])
    
    return results_dict


if __name__ == "__main__":
    results_dict = main()
    results_dict["ligand_df"].to_csv("ligand.csv", index=False)
    results_dict["pfam_df"].to_csv("pfam.csv", index=False)
    results_dict["interactions_df"].to_csv("interactions.csv", index=False)
