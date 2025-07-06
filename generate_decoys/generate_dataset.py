import random as rnd
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import Chem


def generate_struct_data(df, id_col, smiles_col):
    """Genera un archivo CSV con las propiedades físico-químicas de las moléculas en el diccionario."""

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
    properties_df.set_index('compound_id', inplace=True)
    
    return properties_df, fingerprints, scaffolds


def bemis_murcko_clustering(smiles, scaffolds):
    """Aplica clustering de Bemis-Murcko para identificar estructuras únicas en los ligandos."""
    scaffold_dict = {}
    for smile in smiles:
        scaffold_smiles = scaffolds[smile]
        if scaffold_smiles not in scaffold_dict:
            scaffold_dict[scaffold_smiles] = []
        scaffold_dict[scaffold_smiles].append(smile)
    clustered_ids = [ligands[0] for ligands in scaffold_dict.values()]  # Seleccionar un representante por clúster
    return clustered_ids
    

def filter_ligands(properties_df, ligand_properties):
    """Filtra ligandos aptos según los criterios de DUD-E, basado en las propiedades de un ligando."""
    filtered_df = properties_df[(properties_df['mw'].between(ligand_properties['mw'] - 25, ligand_properties['mw'] + 25)) &
                                (properties_df['logP'].between(ligand_properties['logP'] - 1, ligand_properties['logP'] + 1)) &
                                (properties_df['rot_bonds'].between(ligand_properties['rot_bonds'] - 2, ligand_properties['rot_bonds'] + 2)) &
                                (properties_df['h_acceptors'].between(ligand_properties['h_acceptors'] - 1, ligand_properties['h_acceptors'] + 1)) &
                                (properties_df['h_donors'].between(ligand_properties['h_donors'] - 1, ligand_properties['h_donors'] + 1)) &
                                (properties_df['charge'] == ligand_properties['charge'])]
    return filtered_df


def calculate_similarity_filter(ligand_id, filtered_properties, fps, threshold=0.5):
    """Calcula decoys topológicamente diferentes en un subconjunto filtrado de ligandos."""
    ligand_fp = fps[ligand_id]
    pre_decoys = list(filtered_properties.index)
    pre_decoys_fps = [fps[c] for c in pre_decoys]
    tanimoto_pre_decoys = BulkTanimotoSimilarity(ligand_fp,pre_decoys_fps)

    decoys = [pre_decoys[i] for i in range(len(pre_decoys_fps)) if tanimoto_pre_decoys[i]<threshold]
    return decoys


def generate_decoys_from_properties(ligand, pdb_props_df, chembl_props_df, fingerprints, scaffolds, threshold=0.4,max_decoys = 100):
    """Genera decoys para un ligando utilizando una base de datos de propiedades físico-químicas precalculadas."""
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
    
    actives_data = {"properties":pdb_props_df, "interactions":interactions_DB, "fingerprints":pdb_fingerprints}
    return actives_data


generate_actives_dataset("interactions_DB.csv", min_actives=5, max_actives=100)
