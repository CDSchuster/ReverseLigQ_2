"""
This module generates a database of ligand interactions to Pfam domains
by using PDB data.
"""

import logging
import pandas as pd

from pdb_db.LigandPfamDataRequest import (
    get_ligand_pfam_data,
    filter_small_ligands,
    retry_lp_request,
)
from pdb_db.Interactions import get_interaction_data


log = logging.getLogger("generateDB_log")


def intersect_data(pfam_data, interactions_data):
    """
    Checks if the residue with which a ligand interacts is in some Pfam domain,
    and then maps the Pfam domain to the interaction.

    Parameters
    ----------
    pfam_data : pandas.DataFrame
        Contains Pfam data for every chain in every PDB ID.
    interactions_data : pandas.DataFrame
        Ligand–protein interaction data retrieved from PDB.

    Returns
    -------
    pandas.DataFrame
        An updated version of interactions_data that includes the Pfam domain
        data.
    """
    
    # Merge pfam_data with interactions_data on pdb_id and chain_id
    merged = pd.merge(
        interactions_data,
        pfam_data,
        how="left",
        left_on=["pdb_id", "res_chain_id"],
        right_on=["pdb_id", "chain_id"],
    )

    # Filter rows where resnum is within [start, end)
    within_domain = (
        (merged["resnum"] >= merged["start"]) & (merged["resnum"] < merged["end"])
    )
    matched = merged[within_domain]

    # Drop duplicates to keep the first matching Pfam domain
    matched_unique = matched.drop_duplicates(
        subset=interactions_data.columns.tolist()
    )

    # Re-merge back to original size, keeping unmatched rows with NaNs
    interactions_data = interactions_data.merge(
        matched_unique[["pdb_id", "res_chain_id", "resnum", "pfam_id", "pfam_name"]],
        on=["pdb_id", "res_chain_id", "resnum"],
        how="left",
    )
    interactions_data = interactions_data.dropna(subset=["SMILES"])

    return interactions_data


def main():
    """Main execution workflow for building the interactions database."""

    # Step 1: Retrieve ligand and Pfam data
    log.info("Starting retrieval of ligand and Pfam data")
    results_dict = get_ligand_pfam_data()

    # Step 2: Retry failed ligand and Pfam data requests
    log.info("Retrying failed ligand and Pfam data requests")
    ligand_df, pfam_df = retry_lp_request(
        results_dict["ligand_df"],
        results_dict["pfam_df"],
        results_dict["fails"],
    )
    results_dict["ligand_df"], results_dict["pfam_df"] = ligand_df, pfam_df

    # Step 3: Filter ligand dataframe rows based on ligand size
    log.info("Filtering small ligands")
    results_dict["ligand_df"] = filter_small_ligands(results_dict["ligand_df"])

    # Step 4: Retrieve interaction data
    log.info("Requesting interaction data")
    results_dict["interactions_df"] = get_interaction_data(
        results_dict["ligand_df"]
    )

    # Step 5: Intersect interactions data and Pfam data
    log.info("Merging Pfam and interactions dataframes")
    final_df = intersect_data(
        results_dict["pfam_df"],
        results_dict["interactions_df"],
    )
    final_df.to_csv("interactions_DB.csv")
    log.info("Done")


if __name__ == "__main__":
    main()
