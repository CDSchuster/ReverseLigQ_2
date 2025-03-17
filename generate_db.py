import pandas as pd
from LigandPfamModule import get_ligand_pfam_data
from Interactions import get_interaction_data


def main():

    # Step 1: Retrieve ligand and Pfam data
    ligand_df, pfam_df = get_ligand_pfam_data()
    
    # Step 2: Retrieve interaction data
    interactions_df = get_interaction_data(ligand_df)
    
    return ligand_df, pfam_df, interactions_df


if __name__ == "__main__":
    ligand_df, pfam_df, interactions_df = main()
    ligand_df.to_csv("ligand.csv", index=False)
    pfam_df.to_csv("pfam.csv", index=False)
    interactions_df.to_csv("interactions.csv", index=False)
