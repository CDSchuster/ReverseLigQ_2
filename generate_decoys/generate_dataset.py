import pandas as pd
from rdkit.Chem import Descriptors
from rdkit import Chem

def generate_properties_csv(chembl_df):
    """Genera un archivo CSV con las propiedades físico-químicas de las moléculas en el diccionario."""
    rows = []
    for index, row in chembl_df.iterrows():
        mol_id, smile = row['ChEMBL ID'], row["Smiles"]
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

data=pd.read_csv("~/Projects/ReverseLigQ_2/generate_decoys/chembl_smiles.csv")
print(generate_properties_csv(data[:10000]))