import random as rnd
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import Chem


def generate_properties_csv(df, id_col, smiles_col):
    """Genera un archivo CSV con las propiedades físico-químicas de las moléculas en el diccionario."""
    rows = []
    for index, row in df.iterrows():
        
        if index % 10000 == 0:
            print(round(index/len(df)*100, 2), "%")
        
        mol_id, smile = row[id_col], row[smiles_col]
        mol = Chem.MolFromSmiles(smile)
        if mol:
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
    
    return properties_df


def get_ligand_scaffolds(smiles):
    """Precalcula scaffolds para todos los ligandos en un diccionario de SMILES."""
    scaffolds = {}
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds[smile] = scaffold_smiles
    return scaffolds


def bemis_murcko_clustering(decoys, scaffolds):
    """Aplica clustering de Bemis-Murcko para identificar estructuras únicas en los ligandos."""
    scaffold_dict = {}
    for d in decoys:
        scaffold_smiles = scaffolds[d]
        if scaffold_smiles not in scaffold_dict:
            scaffold_dict[scaffold_smiles] = []
        scaffold_dict[scaffold_smiles].append(d)
    clustered_ids = [ligands[0] for ligands in scaffold_dict.values()]  # Seleccionar un representante por clúster
    return clustered_ids


def retrieve_ligand_properties(ligand_id, properties_df):
    """Recupera las propiedades de un ligando desde la base de datos."""
    if ligand_id in properties_df.index:
        return properties_df.loc[ligand_id].to_dict()
    else:
        raise ValueError(f"Ligando {ligand_id} no encontrado en la base de datos.")
    

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


def generate_decoys_from_properties(l, ligs_props, fps, scaffolds, threshold=0.4,max_decoys = 100):
    """Genera decoys para un ligando utilizando una base de datos de propiedades físico-químicas precalculadas."""
    # Cargar base de datos
    #random.seed(10)
    l_prop = retrieve_ligand_properties(l,ligs_props)
    pre_decoys = filter_ligands(ligs_props,l_prop)
    decoys = calculate_similarity_filter(l, pre_decoys, fps, threshold)
    final_decoys = bemis_murcko_clustering(decoys,scaffolds)
    if len(final_decoys) > max_decoys:
        final_decoys = rnd.sample(final_decoys,max_decoys)

    return final_decoys


chembl_data = pd.read_csv("~/Projects/ReverseLigQ_2/generate_decoys/chembl_smiles.csv")
pdb_data = pd.read_csv("interactions_DB.csv")
pdb_data = pdb_data[["ligand_id", "SMILES"]].drop_duplicates()

pdb_data_properties = generate_properties_csv(pdb_data, "ligand_id", "SMILES")
chembl_data_properties = generate_properties_csv(pdb_data, "ChEMBL_ID", "SMILES")

pdb_data_properties.to_csv("pdb_data_properties.csv", index=True)
chembl_data_properties.to_csv("chembl_data_properties.csv", index=True)