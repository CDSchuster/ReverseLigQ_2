from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import rdkit.DataStructs as DataStructs
import pandas as pd
import matplotlib.pyplot as plt


def read_database():
    """
    Reads the database of ligand interactions to Pfam domains
    """

    # Read the database from a CSV file
    db = pd.read_csv("interactions_DB.csv")
    return db


def get_fingerprint(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    # You can use Morgan fingerprint (like ECFP4, radius=2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return fp


def compare_fingerprints():
    db = read_database()
    unique_smiles = db['SMILES'].unique()
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
    with open("estradiol_smile", "r") as f:
        query_smile = f.read().strip()
    query_mol = Chem.MolFromSmiles(query_smile)
    query_fp = morgan_gen.GetFingerprint(query_mol)
    # Step 2: Generate fingerprints for unique SMILES
    smile_to_fp = []
    for smile in unique_smiles:
        if type(smile) == str:
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                fp = morgan_gen.GetFingerprint(mol)
                smile_to_fp.append(fp)


    res = DataStructs.BulkTanimotoSimilarity(query_fp, smile_to_fp)
    res_filt = [x for x in res if x > 0.5]
    plt.hist(res_filt, bins=100)
    plt.show()
compare_fingerprints()