"""
This module contains functions to generate a dataset of active and decoy ligands
based on PDB and ChEMBL data. It includes molecular feature extraction, clustering,
and filtering based on physicochemical properties and Tanimoto similarity.
"""


import itertools
import logging
import random as rnd

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity


# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("generateDB_log")


def extract_molecular_features(df, id_col, smiles_col):
    """
    Generate molecular properties, Morgan fingerprints, and Murcko scaffolds.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing molecular identifiers and SMILES strings.
    id_col : str
        Column name in `df` containing the unique identifier for each molecule.
    smiles_col : str
        Column name in `df` containing the SMILES representation.

    Returns
    -------
    properties_df : pandas.DataFrame
        DataFrame containing the molecular properties of each molecule.
    fingerprints : dict
        Dictionary mapping unique identifiers to Morgan fingerprints.
    scaffolds : dict
        Dictionary mapping SMILES strings to their Murcko scaffold SMILES.
    """

    rows, fingerprints, scaffolds = [], {}, {}
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    for _, row in df.iterrows():
        mol_id = row[id_col]
        smile = row[smiles_col]
        mol = Chem.MolFromSmiles(smile)

        if mol:
            # Generate Morgan fingerprint
            fingerprints[mol_id] = morgan_gen.GetFingerprint(mol)

            # Get Murcko scaffold
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds[smile] = scaffold_smiles

            # Calculate physicochemical properties
            properties = {
                "compound_id": mol_id,
                "smiles": smile,
                "mw": Descriptors.MolWt(mol),
                "logP": Descriptors.MolLogP(mol),
                "rot_bonds": Descriptors.NumRotatableBonds(mol),
                "h_acceptors": Descriptors.NumHAcceptors(mol),
                "h_donors": Descriptors.NumHDonors(mol),
                "charge": Chem.rdmolops.GetFormalCharge(mol),
            }
            rows.append(properties)

    properties_df = pd.DataFrame(rows)
    return properties_df, fingerprints, scaffolds


def bemis_murcko_clustering(smiles, scaffolds):
    """
    Cluster ligands by Murcko scaffolds and return one representative per scaffold.

    Parameters
    ----------
    smiles : list of str
        List of SMILES strings representing the ligands to be clustered.
    scaffolds : dict
        Dictionary mapping SMILES strings to their Murcko scaffold SMILES.

    Returns
    -------
    clustered_ids : list of str
        Representative SMILES strings, one per unique scaffold.
    """

    scaffold_dict = {}
    for smile in smiles:
        scaffold_smiles = scaffolds[smile]
        scaffold_dict.setdefault(scaffold_smiles, []).append(smile)

    # Select first ligand as representative per scaffold
    clustered_ids = [ligands[0] for ligands in scaffold_dict.values()]
    return clustered_ids


def filter_by_properties(properties_df, ligand_properties):
    """
    Filter ligands similar to a reference ligand based on physicochemical properties.

    Filtering follows DUD-E criteria:
    - Molecular weight ±25
    - logP ±1
    - Rotatable bonds ±2
    - H-bond acceptors ±1
    - H-bond donors ±1
    - Same formal charge

    Parameters
    ----------
    properties_df : pandas.DataFrame
        DataFrame with ligand properties.
    ligand_properties : dict
        Dictionary containing the reference ligand's properties.

    Returns
    -------
    filtered_df : pandas.DataFrame
        Subset of ligands from `properties_df` matching the criteria.
    """

    filtered_df = properties_df[
        (properties_df["mw"].between(ligand_properties["mw"] - 25,
                                     ligand_properties["mw"] + 25))
        & (properties_df["logP"].between(ligand_properties["logP"] - 1,
                                         ligand_properties["logP"] + 1))
        & (properties_df["rot_bonds"].between(ligand_properties["rot_bonds"] - 2,
                                              ligand_properties["rot_bonds"] + 2))
        & (properties_df["h_acceptors"].between(ligand_properties["h_acceptors"] - 1,
                                                ligand_properties["h_acceptors"] + 1))
        & (properties_df["h_donors"].between(ligand_properties["h_donors"] - 1,
                                             ligand_properties["h_donors"] + 1))
        & (properties_df["charge"] == ligand_properties["charge"])
    ]
    return filtered_df


def filter_by_tanimoto(ligand_fp, filtered_properties, fps, threshold=0.5):
    """
    Filter ligands based on Tanimoto similarity against a reference ligand.

    Parameters
    ----------
    ligand_fp : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        Fingerprint of the reference ligand.
    filtered_properties : pandas.DataFrame
        Subset of ligands already filtered by physicochemical properties.
    fps : dict
        Dictionary mapping ligand IDs to their Morgan fingerprints.
    threshold : float, default=0.5
        Maximum allowed Tanimoto similarity.

    Returns
    -------
    decoys : list of str
        List of ligand SMILES considered potential decoys.
    """

    pre_decoys_dict = dict(
        zip(filtered_properties["compound_id"], filtered_properties["smiles"])
    )
    pre_decoys = list(filtered_properties["compound_id"])
    pre_decoys_fps = [fps[c] for c in pre_decoys]

    # Compute Tanimoto similarities in bulk
    tanimoto_pre_decoys = BulkTanimotoSimilarity(ligand_fp, pre_decoys_fps)

    # Keep ligands below similarity threshold
    decoys = [
        pre_decoys_dict[pre_decoys[i]]
        for i in range(len(pre_decoys_fps))
        if tanimoto_pre_decoys[i] < threshold
    ]
    return decoys


def get_decoys(
    ligand_fp,
    ligand_props,
    chembl_props_df,
    fingerprints,
    scaffolds,
    threshold,
    max_decoys,
):
    """
    Generate decoys for a ligand using precomputed property datasets.

    Parameters
    ----------
    ligand_fp : rdkit.DataStructs.cDataStructs.ExplicitBitVect
        Fingerprint of the reference ligand.
    ligand_props : dict
        Physicochemical properties of the reference ligand.
    chembl_props_df : pandas.DataFrame
        Properties of ligands derived from ChEMBL.
    fingerprints : dict
        Dictionary mapping ligand IDs to Morgan fingerprints.
    scaffolds : dict
        Dictionary mapping SMILES strings to their Murcko scaffolds.
    threshold : float
        Maximum allowed Tanimoto similarity.
    max_decoys : int
        Maximum number of decoys to return.

    Returns
    -------
    final_decoys : list of str or None
        Representative decoys for the ligand, or None if none were found.
    """

    # Step 1: Filter by physicochemical similarity
    pre_decoys = filter_by_properties(chembl_props_df, ligand_props)

    # Step 2: Filter by Tanimoto similarity
    decoys = filter_by_tanimoto(ligand_fp, pre_decoys, fingerprints, threshold)

    # Step 3: Cluster by Bemis–Murcko scaffold
    final_decoys = bemis_murcko_clustering(decoys, scaffolds)

    # Step 4: Limit number of decoys
    if len(final_decoys) > max_decoys:
        final_decoys = rnd.sample(final_decoys, max_decoys)

    return final_decoys


def get_actives_data(pdb_data, min_actives, max_actives):
    """
    Generate active ligand clusters from PDB-derived data.

    Parameters
    ----------
    pdb_data : str
        Path to CSV file containing `ligand_id`, `SMILES`, and `pfam_id`.
    min_actives : int
        Minimum number of active ligands required for a Pfam cluster.
    max_actives : int
        Maximum number of active ligands allowed for a Pfam cluster.

    Returns
    -------
    actives_data : dict
        Dictionary containing:
        - properties: DataFrame with ligand properties
        - interactions: DataFrame with ligand-Pfam mappings
        - fingerprints: dict of ligand fingerprints
        - Pfam_clusters: dict of Pfam → list of ligand SMILES
        - scaffolds: dict of SMILES → scaffold SMILES
    """

    interactions_db = pd.read_csv(pdb_data)[["ligand_id", "SMILES", "pfam_id"]]
    interactions_db = interactions_db.drop_duplicates()

    # Get physicochemical properties, fingerprints, and scaffolds
    pdb_props_df, pdb_fps, pdb_scaffolds = extract_molecular_features(
        interactions_db[["ligand_id", "SMILES"]].drop_duplicates(),
        "ligand_id",
        "SMILES",
    )

    # Keep only ligands with valid properties
    interactions_db = interactions_db[
        interactions_db.SMILES.isin(pdb_props_df.smiles.unique())
    ]

    # Group ligands by Pfam ID
    pfam_smiles_dict = (
        interactions_db.groupby("pfam_id")["SMILES"].apply(list).to_dict()
    )

    # Cluster ligands within each Pfam and filter by active count
    clusters = {}
    for pfam_id, smiles_list in pfam_smiles_dict.items():
        clustered_ligands = bemis_murcko_clustering(smiles_list, pdb_scaffolds)
        if min_actives < len(clustered_ligands) < max_actives:
            clusters[pfam_id] = clustered_ligands

    # Keep only ligands belonging to valid clusters
    all_actives = {
        ligand for ligand_cluster in clusters.values() for ligand in ligand_cluster
    }
    pdb_props_df = pdb_props_df[pdb_props_df.smiles.isin(all_actives)]
    interactions_db = interactions_db[interactions_db.SMILES.isin(all_actives)]

    actives_data = {
        "properties": pdb_props_df,
        "interactions": interactions_db,
        "fingerprints": pdb_fps,
        "Pfam_clusters": clusters,
        "scaffolds": pdb_scaffolds,
    }
    return actives_data


def get_actives_and_decoys(min_actives, max_actives, threshold, max_decoys):
    """
    Main pipeline to generate actives and decoys datasets.

    Steps:
    1. Generate active clusters from interactions data.
    2. Get ChEMBL ligand properties.
    3. Select decoys for each active ligand.

    Parameters
    ----------
    min_actives : int
        Minimum number of active ligands required for a Pfam cluster.
    max_actives : int
        Maximum number of active ligands allowed for a Pfam cluster.
    threshold : float
        Maximum allowed Tanimoto similarity.
    max_decoys : int
        Maximum number of decoys per ligand.

    Returns
    -------
    pfam_clusters : dict
        Dictionary mapping Pfam IDs to lists of active ligand SMILES.
    decoy_dataset : dict
        Dictionary mapping active ligand SMILES to lists of decoy SMILES.
    """

    log.info("Generating actives dataset")
    actives_data = get_actives_data(
        "input_files/small_interactions_DB.csv", min_actives, max_actives
    )

    # Load ChEMBL data
    chembl_smiles = pd.read_csv("input_files/small_chembl.csv").drop_duplicates()

    # Extract all active ligands
    all_actives = {
        ligand
        for ligand_cluster in actives_data["Pfam_clusters"].values()
        for ligand in ligand_cluster
    }

    # Get ChEMBL properties, fingerprints, and scaffolds
    chembl_props_df, chembl_fps, chembl_scaffolds = extract_molecular_features(
        chembl_smiles, "ChEMBL_ID", "SMILES"
    )

    decoy_dataset = {}
    counter, actives_num = 0, len(all_actives) // 10

    log.info("Generating decoys for actives")
    for ligand in all_actives:

        # Retrieve active ligand properties
        ligand_props = actives_data["properties"][
            actives_data["properties"].smiles == ligand
        ].iloc[0].to_dict()

        ligand_fp = actives_data["fingerprints"][ligand_props["compound_id"]]

        counter += 1
        if counter % actives_num == 0:
            log.info(f"Generating decoys for {counter}/{len(all_actives)} ligands")

        # Generate decoys for the active ligand
        ligand_decoys = get_decoys(
            ligand_fp,
            ligand_props,
            chembl_props_df,
            chembl_fps,
            chembl_scaffolds,
            threshold,
            max_decoys,
        )

        if ligand_decoys:
            decoy_dataset[ligand] = ligand_decoys

    pfam_clusters = actives_data["Pfam_clusters"]
    return pfam_clusters, decoy_dataset


def generate_smiles_pairs_dataset(min_actives, max_actives, threshold, min_decoys, max_decoys):
    """
    Generate a dataset of SMILES pairs (active–active and active–decoy).

    Parameters
    ----------
    min_actives : int
        Minimum number of active ligands required per Pfam cluster.
    max_actives : int
        Maximum number of active ligands per Pfam cluster.
    threshold : float
        Maximum allowed Tanimoto similarity for decoy selection.
    max_decoys : int
        Maximum number of decoys per ligand.

    Returns
    -------
    smiles_df : pandas.DataFrame
        DataFrame containing SMILES pairs and labels:
        - 1 for active-active pairs
        - 0 for active-decoy pairs
    """

    actives_data, decoys_data = get_actives_and_decoys(
        min_actives, max_actives, threshold, max_decoys
    )

    log.info("Generating SMILES pairs dataset")

    rows = []
    for pfam_id, actives in actives_data.items():
        # Active–Active pairs (within the same Pfam)
        for s1, s2 in itertools.combinations(actives, 2):
            rows.append(
                {"smiles_1": s1, "smiles_2": s2, "pfam_id": pfam_id, "label": 1}
            )

        # Active–Decoy pairs
        for active in actives:
            if active in decoys_data and len(decoys_data[active]) > min_decoys:
                for decoy_smile in decoys_data[active]:
                    rows.append(
                        {
                            "smiles_1": active,
                            "smiles_2": decoy_smile,
                            "pfam_id": pfam_id,
                            "label": 0,
                        }
                    )
            else:
                log.error(f"Not enough decoys found for active: {active} (Pfam: {pfam_id})")

    smiles_df = pd.DataFrame(rows)
    smiles_df.to_csv("smiles_pairs_dataset.csv", index=False)
    log.info(f"Total pairs generated: {len(smiles_df)}")

    return smiles_df
