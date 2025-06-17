#from transformers import AutoTokenizer, AutoModel
from deepchem.feat import MolGraphConvFeaturizer
from rdkit import Chem
from rdkit.Chem import SanitizeMol, SanitizeFlags
import pandas as pd


data = pd.read_csv("interactions_DB.csv")

featurizer = MolGraphConvFeaturizer()
smiles_set = list(set(data["SMILES"]))

valid_features = []

for smi in smiles_set:
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            raise ValueError("RDKit failed to parse")
        
        try:
            # Try full sanitization, or partial sanitization
            Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception as e:
            print(f"[partial molecule] continuing despite: {e}")

        # Generate a SMILES back to use with DeepChem
        fixed_smi = Chem.MolToSmiles(mol)
        feats = featurizer.featurize([fixed_smi])[0]
        valid_features.append(feats)
        
    except Exception as e:
        print(f"[skip] Failed for SMILES {smi}: {e}")


# Check for NaNs or empty strings
missing_smiles = data['SMILES'].isna() | (data['SMILES'].str.strip() == '')

# Print how many
print(f"Number of missing or empty SMILES: {missing_smiles.sum()}")

# Optionally, view the rows
print(set(data[missing_smiles]["ligand_id"]))