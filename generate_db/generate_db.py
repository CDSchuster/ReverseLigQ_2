"""
This module generates a database of ligand interactions to Pfam domains by using PDB data
"""

import pandas as pd
from generate_db.LigandPfamDataRequest import get_ligand_pfam_data, filter_small_ligands, retry_lp_request
from generate_db.Interactions import get_interaction_data
import logging


log = logging.getLogger("generateDB_log")


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

    # First, merge pfam_data with interactions_data on pdb_id and chain_id
    merged = pd.merge(
        interactions_data,
        pfam_data,
        how="left",
        left_on=["pdb_id", "res_chain_id"],
        right_on=["pdb_id", "chain_id"]
    )

    # Now filter rows where resnum is within [start, end)
    within_domain = (merged['resnum'] >= merged['start']) & (merged['resnum'] < merged['end'])
    matched = merged[within_domain]

    # Drop duplicates to keep the first matching Pfam domain (like your `break`)
    matched_unique = matched.drop_duplicates(subset=interactions_data.columns.tolist())

    # Re-merge back to original size, keeping unmatched rows with NaNs
    interactions_with_pfam = interactions_data.merge(
        matched_unique[["pdb_id", "res_chain_id", "resnum", "pfam_id", "pfam_name"]],
        on=["pdb_id", "res_chain_id", "resnum"],
        how="left"
    )

    return interactions_data


def main():

    # Step 1: Retrieve ligand and Pfam data
    log.info("Starting retrieval of ligand and Pfam data")
    results_dict = get_ligand_pfam_data()
    
    # Step 2: Retry failed ligand and Pfam data requests
    log.info("Retrying failed ligand and Pfam data requests")
    results_dict["ligand_df"], results_dict["pfam_df"] = retry_lp_request(results_dict["ligand_df"],
                                                                          results_dict["pfam_df"],
                                                                          results_dict["fails"])
    
    # Step 3: filter ligand dataframe rows based on the number of atoms in the ligand
    log.info("Filtering small ligands")
    results_dict["ligand_df"] = filter_small_ligands(results_dict["ligand_df"])

    # Step 4: Retrieve interaction data
    log.info("Request interaction data")
    results_dict["interactions_df"] = get_interaction_data(results_dict["ligand_df"])

    # Step 5: Intersect interactions data and Pfam data
    log.info("Merging Pfam and interactions dataframes")
    final_df = intersect_data(results_dict["pfam_df"], results_dict["interactions_df"])
    final_df.to_csv("interactions_DB.csv")
    log.info("Done")


if __name__ == "__main__":
    main()