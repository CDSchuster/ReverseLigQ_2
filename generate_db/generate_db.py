"""
This module generates a database of ligand interactions to Pfam domains by using PDB data
"""

import pandas as pd
from generate_db.LigandPfamDataRequest import get_ligand_pfam_data, filter_small_ligands, retry_lp_request
from generate_db.Interactions import get_interaction_data


def intersect_data(pfam_data, interactions_data):
    """
    Checks if the residue with which a ligand interacts is in some Pfam domain,
    and then maps the Pfam domain to the interaction

    Parameters
    ----------
    pfam_data : dataframe
        Contains Pfam data for every chain in every PDB ID
    interactions_data : dataframe
        Ligand-protein interaction data retrieved from PDB

    Returns
    -------
    interactions_data : dataframe
        An updated version of interactions_data that includes the Pfam domain data
    """

    pfam_ids, pfam_names = [], [] # In these 2 lists we save the Pfam data to add to interaction_data
    
    for row in interactions_data.itertuples(index=False):
        # Iterate interactions_data rows and keep pdb_id, residue chain and residue number
        pdb_id, res_chain_id, resnum = row.pdb_id, row.res_chain_id, row.resnum
        # For every combination of pdb_id and chain, we check pfam data in that chain
        matching_rows = pfam_data[(pfam_data['pdb_id'] == pdb_id) & (pfam_data['chain_id'] == res_chain_id)]
        
        found = False
        # We iterate the pfam_data rows that have that PDB chain
        for pfrow in matching_rows.itertuples(index=False):
            # Check if the residue with which the ligand interacts is in the Pfam domain
            if pfrow.start < resnum < pfrow.end:
                pfam_ids.append(pfrow.pfam_id)
                pfam_names.append(pfrow.pfam_name)
                found = True
                break  # Stop after first match
        # If there is a match, we keep the Pfam data, otherwise we will put a NA
        if not found:
            pfam_ids.append(pd.NA)
            pfam_names.append(pd.NA)

    # Add the new columns
    interactions_data['pfam_id'] = pfam_ids
    interactions_data['pfam_name'] = pfam_names

    return interactions_data


def main():

    # Step 1: Retrieve ligand and Pfam data
    results_dict = get_ligand_pfam_data()
    
    # Step 2: Retry failed ligand and Pfam data requests
    results_dict["ligand_df"], results_dict["pfam_df"] = retry_lp_request(results_dict["ligand_df"],
                                                                          results_dict["pfam_df"],
                                                                          results_dict["fails"])
    
    # Step 3: filter ligand dataframe rows based on the number of atoms in the ligand
    results_dict["ligand_df"] = filter_small_ligands(results_dict["ligand_df"])

    # Step 4: Retrieve interaction data
    results_dict["interactions_df"] = get_interaction_data(results_dict["ligand_df"])

    # Step 5: Intersect interactions data and Pfam data
    final_df = intersect_data(results_dict["pfam_df"], results_dict["interactions_df"])
    final_df.to_csv("final_df.csv")

    return results_dict


if __name__ == "__main__":
    results_dict = main()