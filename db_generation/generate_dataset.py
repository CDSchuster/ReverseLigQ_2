import random as rnd
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity


def generate_struct_data(df, id_col, smiles_col):
    """Generate molecular properties, Morgan fingerprints, and Murcko scaffolds.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns specified in id_col and smiles_col.
    id_col : str
        Column name in df that contains the unique identifier for each molecule.
    smiles_col : str
        Column name in df that contains the SMILES representation of each molecule.

    Returns
    -------
    properties_df : pandas.DataFrame
        Dataframe containing the molecular properties of each molecule, indexed
        by the unique identifier.
    fingerprints : dict
        Dictionary where keys are the unique identifiers and values are the
        corresponding Morgan fingerprints.
    scaffolds : dict
        Dictionary where keys are the SMILES strings and values are the
        corresponding Murcko scaffold SMILES.
    """

    rows, fingerprints, scaffolds = [], {}, {}
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    for index, row in df.iterrows():
        if index % 10000 == 0:
            print(round(index / len(df) * 100, 2), "%")

        mol_id, smile = row[id_col], row[smiles_col]
        mol = Chem.MolFromSmiles(smile)

        if mol:
            fingerprints[mol_id] = morgan_gen.GetFingerprint(mol)

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds[smile] = scaffold_smiles

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
    print(len(properties_df) / len(df), "of molecules could be processed")

    return properties_df, fingerprints, scaffolds


def bemis_murcko_clustering(smiles, scaffolds):
    """Cluster ligands by Murcko scaffolds, returning one representative per scaffold.

    Parameters
    ----------
    smiles : list
        List of SMILES strings representing the ligands to be clustered.
    scaffolds : dict
        Dictionary where keys are SMILES strings and values are their
        corresponding Murcko scaffold SMILES.

    Returns
    -------
    clustered_ids : list
        List of SMILES strings, each one being a representative ligand for a
        unique scaffold.
    """

    scaffold_dict = {}
    for smile in smiles:
        scaffold_smiles = scaffolds[smile]
        if scaffold_smiles not in scaffold_dict:
            scaffold_dict[scaffold_smiles] = []
        scaffold_dict[scaffold_smiles].append(smile)

    clustered_ids = [ligands[0] for ligands in scaffold_dict.values()]
    return clustered_ids


def filter_ligands(properties_df, ligand_properties):
    """Filter ligands similar to a reference ligand based on physicochemical properties.

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
        Dataframe with ligand properties.
    ligand_properties : dict
        Dictionary containing the reference ligand's properties.

    Returns
    -------
    filtered_df : pandas.DataFrame
        Subset of ligands from properties_df that match the criteria.
    """

    filtered_df = properties_df[
        (properties_df["mw"].between(
            ligand_properties["mw"] - 25, ligand_properties["mw"] + 25
        ))
        & (
            properties_df["logP"].between(
                ligand_properties["logP"] - 1, ligand_properties["logP"] + 1
            )
        )
        & (
            properties_df["rot_bonds"].between(
                ligand_properties["rot_bonds"] - 2,
                ligand_properties["rot_bonds"] + 2,
            )
        )
        & (
            properties_df["h_acceptors"].between(
                ligand_properties["h_acceptors"] - 1,
                ligand_properties["h_acceptors"] + 1,
            )
        )
        & (
            properties_df["h_donors"].between(
                ligand_properties["h_donors"] - 1,
                ligand_properties["h_donors"] + 1,
            )
        )
        & (properties_df["charge"] == ligand_properties["charge"])
    ]
    return filtered_df


def calculate_similarity_filter(ligand_id, filtered_properties, fps, threshold=0.5):
    """Filter ligands based on Tanimoto similarity against a reference ligand.

    Parameters
    ----------
    ligand_id : str
        Unique identifier for the ligand whose fingerprint is used.
    filtered_properties : pandas.DataFrame
        Subset of ligands already filtered by physicochemical properties.
    fps : dict
        Dictionary mapping ligand IDs to their Morgan fingerprints.
    threshold : float, optional (default=0.5)
        Maximum allowed Tanimoto similarity.

    Returns
    -------
    decoys : list
        List of ligand IDs considered potential decoys.
    """

    ligand_fp = fps[ligand_id]
    pre_decoys = list(filtered_properties.index)
    pre_decoys_fps = [fps[c] for c in pre_decoys]

    tanimoto_pre_decoys = BulkTanimotoSimilarity(ligand_fp, pre_decoys_fps)

    decoys = [
        pre_decoys[i]
        for i in range(len(pre_decoys_fps))
        if tanimoto_pre_decoys[i] < threshold
    ]
    return decoys


def generate_decoys_from_properties(
    ligand,
    pdb_props_df,
    chembl_props_df,
    fingerprints,
    scaffolds,
    threshold=0.4,
    max_decoys=100,
):
    """Generate decoys for a ligand using precomputed property datasets.

    Parameters
    ----------
    ligand : str
        Ligand identifier present in pdb_props_df.
    pdb_props_df : pandas.DataFrame
        Properties of ligands derived from PDB data.
    chembl_props_df : pandas.DataFrame
        Properties of ligands derived from ChEMBL data.
    fingerprints : dict
        Dictionary mapping ligand IDs to their Morgan fingerprints.
    scaffolds : dict
        Dictionary mapping SMILES strings to their Murcko scaffolds.
    threshold : float, optional (default=0.4)
        Maximum allowed Tanimoto similarity.
    max_decoys : int, optional (default=100)
        Maximum number of decoys to return.

    Returns
    -------
    final_decoys : list or None
        List of representative decoys for the ligand, or None if no decoys
        could be generated.
    """

    final_decoys = None
    if ligand in pdb_props_df.compound_id.values:
        ligand_props = (
            pdb_props_df[pdb_props_df.compound_id == ligand].iloc[0].to_dict()
        )

        pre_decoys = filter_ligands(chembl_props_df, ligand_props)
        decoys = calculate_similarity_filter(ligand, pre_decoys, fingerprints, threshold)
        final_decoys = bemis_murcko_clustering(decoys, scaffolds)

        if len(final_decoys) > max_decoys:
            final_decoys = rnd.sample(final_decoys, max_decoys)

    return final_decoys


def generate_actives_dataset(pdb_data, min_actives=5, max_actives=100):
    """Generate active ligand clusters from PDB-derived data.

    Parameters
    ----------
    pdb_data : str
        Path to CSV file containing ligand_id, SMILES, and pfam_id.
    min_actives : int, optional (default=5)
        Minimum number of active ligands required for a Pfam cluster.
    max_actives : int, optional (default=100)
        Maximum number of active ligands allowed for a Pfam cluster.

    Returns
    -------
    actives_data : dict
        Dictionary containing:
        - properties: DataFrame with ligand properties
        - interactions: DataFrame with ligand-Pfam mappings
        - fingerprints: dict of ligand fingerprints
        - Pfam_clusters: dict of Pfam → list of ligand SMILES (clustered)
        - scaffolds: dict of SMILES → scaffold SMILES
    """

    interactions_db = pd.read_csv(pdb_data)[["ligand_id", "SMILES", "pfam_id"]]
    interactions_db = interactions_db.drop_duplicates()

    pdb_props_df, pdb_fps, pdb_scaffolds = generate_struct_data(
        interactions_db[["ligand_id", "SMILES"]].drop_duplicates(),
        "ligand_id",
        "SMILES",
    )

    interactions_db = interactions_db[
        interactions_db.SMILES.isin(pdb_props_df.smiles.unique())
    ]

    pfam_smiles_dict = (
        interactions_db.groupby("pfam_id")["SMILES"].apply(list).to_dict()
    )

    clusters = {}
    for pfam_id, smiles_list in pfam_smiles_dict.items():
        clustered_ligands = bemis_murcko_clustering(smiles_list, pdb_scaffolds)
        if min_actives < len(clustered_ligands) < max_actives:
            clusters[pfam_id] = clustered_ligands

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


def generate_data():
    """Main pipeline to generate actives and decoys datasets.

    Steps:
    1. Generate active clusters from PDB data.
    2. Generate ChEMBL ligand properties.
    3. Generate decoys for each active ligand.
    4. Save datasets as pickle files.

    Returns
    -------
    None
    """

    actives_data = generate_actives_dataset("small_interactions_DB.csv")
    print("Actives data generated")

    chembl_smiles = pd.read_csv("small_chembl.csv").drop_duplicates()
    all_actives = {
        ligand
        for ligand_cluster in actives_data["Pfam_clusters"].values()
        for ligand in ligand_cluster
    }

    chembl_props_df, chembl_fps, chembl_scaffolds = generate_struct_data(
        chembl_smiles, "ChEMBL_ID", "SMILES"
    )
    print("ChEMBL data generated")

    decoy_dataset = {}
    counter, actives_num = 0, len(all_actives) // 10

    for ligand in all_actives:
        counter += 1
        if counter % actives_num == 0:
            print(f"Generating decoys for {counter}/{len(all_actives)} ligands")

        ligand_decoys = generate_decoys_from_properties(
            ligand,
            actives_data["properties"],
            chembl_props_df,
            chembl_fps,
            chembl_scaffolds,
        )
        if ligand_decoys:
            decoy_dataset[ligand] = ligand_decoys

    print("Decoys generated")

    pickle.dump(decoy_dataset, open("decoys.pkl", "wb"))
    pickle.dump(
        actives_data["Pfam_clusters"], open("actives_clusters.pkl", "wb")
    )

generate_data()