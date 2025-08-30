import random as rnd
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import Chem
import pickle


def generate_struct_data(df, id_col, smiles_col):
    """For a given dataframe containing SMILES strings, this function generates molecular properties,
    Morgan fingerprints, and Murcko scaffolds. It returns a dataframe with the properties and two
    dictionaries: one for the fingerprints and another for the scaffolds.

    Parameters
    ----------
    df : dataframe
        It must contain the columns specified in id_col and smiles_col
    id_col : str
        The column name in df that contains the unique identifier for each molecule
    smiles_col : str
        The column name in df that contains the SMILES representation of each molecule

    Returns
    -------
    properties_df : dataframe
        A dataframe containing the molecular properties of each molecule, indexed by the unique identifier
    fingerprints : dict
        A dictionary where keys are the unique identifiers and values are the corresponding Morgan fingerprints
    scaffolds : dict
        A dictionary where keys are the SMILES strings and values are the corresponding Murcko scaffold SMILES
    """

    rows, fingerprints, scaffolds = [], {}, {}
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    for index, row in df.iterrows():
        
        if index % 10000 == 0:
            print(round(index/len(df)*100, 2), "%")
        
        mol_id, smile = row[id_col], row[smiles_col]
        mol = Chem.MolFromSmiles(smile)
        
        if mol:
            
            fingerprints[mol_id] = morgan_gen.GetFingerprint(mol)

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds[smile] = scaffold_smiles

            properties = {
                'compound_id': mol_id,
                'smiles': smile,
                'mw': Descriptors.MolWt(mol),
                'logP': Descriptors.MolLogP(mol),
                'rot_bonds': Descriptors.NumRotatableBonds(mol),
                'h_acceptors': Descriptors.NumHAcceptors(mol),
                'h_donors': Descriptors.NumHDonors(mol),
                'charge': Chem.rdmolops.GetFormalCharge(mol)
            }
            rows.append(properties)
    
    properties_df = pd.DataFrame(rows)
    print(len(properties_df)/len(df), "of molecules could be processed")
    
    return properties_df, fingerprints, scaffolds


def bemis_murcko_clustering(smiles, scaffolds):
    """Clusters ligands based on their Murcko scaffolds, returning a list of representative ligands for each scaffold

    Parameters
    ----------
    smiles : list
        A list of SMILES strings representing the ligands to be clustered
    scaffolds : dict
        A dictionary where keys are SMILES strings and values are their corresponding Murcko scaffold SMILES

    Returns
    -------
    clustered_ids : list
        A list of unique identifiers for the ligands, where each identifier corresponds to a unique scaffold
    """

    scaffold_dict = {}
    for smile in smiles:
        scaffold_smiles = scaffolds[smile]
        if scaffold_smiles not in scaffold_dict:
            scaffold_dict[scaffold_smiles] = []
        scaffold_dict[scaffold_smiles].append(smile)
    clustered_ids = [ligands[0] for ligands in scaffold_dict.values()]  # Seleccionar un representante por clúster
    return clustered_ids
    

def filter_ligands(properties_df, ligand_properties):
    """Filters a dataframe of molecular properties to find ligands that are
    similar to a given ligand based on several physicochemical properties.

    Parameters
    ----------
    properties_df : dataframe
        A dataframe containing the properties of various ligands, including molecular weight (mw),
        logP, number of rotatable bonds (rot_bonds), number of hydrogen bond acceptors (h_acceptors),
        number of hydrogen bond donors (h_donors), and charge.
        Each row should represent a different ligand with these properties.
    ligand_properties : dict
        A dictionary containing the properties of a specific ligand, with keys corresponding to the same
        properties as in properties_df (mw, logP, rot_bonds, h_acceptors, h_donors, charge).

    Returns
    -------
    filtered_df : dataframe
        A filtered dataframe containing only those ligands from properties_df that are similar to the
        specified ligand based on the DUD-E criteria for similarity
    """

    filtered_df = properties_df[(properties_df['mw'].between(ligand_properties['mw'] - 25, ligand_properties['mw'] + 25)) &
                                (properties_df['logP'].between(ligand_properties['logP'] - 1, ligand_properties['logP'] + 1)) &
                                (properties_df['rot_bonds'].between(ligand_properties['rot_bonds'] - 2, ligand_properties['rot_bonds'] + 2)) &
                                (properties_df['h_acceptors'].between(ligand_properties['h_acceptors'] - 1, ligand_properties['h_acceptors'] + 1)) &
                                (properties_df['h_donors'].between(ligand_properties['h_donors'] - 1, ligand_properties['h_donors'] + 1)) &
                                (properties_df['charge'] == ligand_properties['charge'])]
    return filtered_df


def calculate_similarity_filter(ligand_id, filtered_properties, fps, threshold=0.5):
    """Calculates the Tanimoto similarity between a given ligand's fingerprint and the fingerprints of a set of filtered ligands.

    Parameters
    ----------
    ligand_id : str
        The unique identifier for the ligand whose fingerprint will be used for similarity calculations.
    filtered_properties : dataframe
        A dataframe containing the properties of ligands that have been filtered based on physicochemical criteria.
    fps : dict
        A dictionary where keys are unique identifiers for ligands and values are their corresponding fingerprints.
    threshold : float, optional
        The maximum Tanimoto threshold to determine whether to keep the potential decoy or not, by default 0.5

    Returns
    -------
    decoys : list
        A list of unique identifiers for ligands that are considered potential decoys based on the Tanimoto similarity threshold.
    """

    ligand_fp = fps[ligand_id]
    pre_decoys = list(filtered_properties.index)
    pre_decoys_fps = [fps[c] for c in pre_decoys]
    tanimoto_pre_decoys = BulkTanimotoSimilarity(ligand_fp,pre_decoys_fps)

    decoys = [pre_decoys[i] for i in range(len(pre_decoys_fps)) if tanimoto_pre_decoys[i]<threshold]
    return decoys


def generate_decoys_from_properties(ligand, pdb_props_df, chembl_props_df, fingerprints, scaffolds, threshold=0.4,max_decoys = 100):
    """Genera decoys para un ligando utilizando una base de datos de propiedades físico-químicas precalculadas."""
    
    final_decoys = None
    if ligand in pdb_props_df.compound_id.values:
        ligand_props = pdb_props_df[pdb_props_df.compound_id==ligand].iloc[0].to_dict()
        pre_decoys = filter_ligands(chembl_props_df, ligand_props)
        decoys = calculate_similarity_filter(ligand, pre_decoys, fingerprints, threshold)
        final_decoys = bemis_murcko_clustering(decoys, scaffolds)

        if len(final_decoys) > max_decoys:
            final_decoys = rnd.sample(final_decoys, max_decoys)

    return final_decoys


def generate_actives_dataset(pdb_data, min_actives=5, max_actives=100):

    interactions_DB = pd.read_csv(pdb_data)[["ligand_id", "SMILES", "pfam_id"]].drop_duplicates()
    pdb_props_df, pdb_fingerprints, pdb_scaffolds = generate_struct_data(interactions_DB[["ligand_id", "SMILES"]].drop_duplicates(), "ligand_id", "SMILES")

    interactions_DB = interactions_DB[interactions_DB.SMILES.isin(pdb_props_df.smiles.unique())]
    pfam_smiles_dict = interactions_DB.groupby('pfam_id')['SMILES'].apply(list).to_dict()

    clusters = {}
    for pfam_id, smiles_list in pfam_smiles_dict.items():
        clustered_ligands = bemis_murcko_clustering(smiles_list, pdb_scaffolds)
        if len(clustered_ligands) > min_actives and len(clustered_ligands) < max_actives:
            clusters[pfam_id] = clustered_ligands

    all_actives = {ligand for ligand_cluster in clusters.values() for ligand in ligand_cluster}
    pdb_props_df = pdb_props_df[pdb_props_df.smiles.isin(all_actives)]
    interactions_DB = interactions_DB[interactions_DB.SMILES.isin(all_actives)]
    
    actives_data = {"properties":pdb_props_df, "interactions":interactions_DB,
                    "fingerprints":pdb_fingerprints, "Pfam_clusters":clusters,
                    "scaffolds":pdb_scaffolds}
    
    return actives_data


def generate_data():

    actives_data = generate_actives_dataset("small_interactions_DB.csv")
    print("Actives data generated")
    
    chembl_smiles = pd.read_csv("small_chembl.csv").drop_duplicates()
    all_actives = {ligand for ligand_cluster in actives_data["Pfam_clusters"].values() for ligand in ligand_cluster}
    chembl_props_df, chembl_fingerprints, chembl_scaffolds = generate_struct_data(chembl_smiles, "ChEMBL_ID", "SMILES")
    print("ChEMBL data generated")
    
    decoy_dataset = {}
    counter, actives_num = 0, len(all_actives) // 10
    
    for ligand in all_actives:
        
        counter += 1
        if counter % actives_num == 0:
            print(f"Generating decoys for {counter}/{len(all_actives)} ligands")

        ligand_decoys = generate_decoys_from_properties(ligand, actives_data["properties"], chembl_props_df, chembl_fingerprints, chembl_scaffolds)
        if ligand_decoys:
            decoy_dataset[ligand] = ligand_decoys
    
    print("Decoys generated")
    pickle.dump(decoy_dataset, open("decoys.pkl", "wb"))
    pickle.dump(actives_data["Pfam_clusters"], open("actives_clusters.pkl", "wb"))

generate_data()