from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import rdkit.DataStructs as DataStructs
import pandas as pd
import matplotlib.pyplot as plt


def compare_fingerprints(query_smile, threhsold=0.9):
    """
    Takes a SMILE, compares its fingerprint to a database of molecules using the Tanimoto coefficient,
    and returns the most similar molecules.

    Parameters
    ----------
    query_smile : str
        A SMILE string to be compared against the database
    threhsold : float, optional
        A threshold for the Tanimoto coefficient, by default 0.9

    Returns
    -------
    filtered_db : pd.DataFrame
        A DataFrame containing the filtered database of molecules that are similar to the query SMILE,
        including their SMILES, interaction data, and Pfam domain information
    """

    db = pd.read_csv("full_DB.csv", index_col=0)
    unique_smiles = db['SMILES'].unique()
    
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
   
    query_mol = Chem.MolFromSmiles(query_smile)
    query_fp = morgan_gen.GetFingerprint(query_mol)
    
    # Step 2: Generate fingerprints for unique SMILES
    target_fps = {}
    for smile in unique_smiles:
        if type(smile) == str:
            mol = Chem.MolFromSmiles(smile)
            if mol != None:
                target_fps[smile] = morgan_gen.GetFingerprint(mol)

    # Bulk similarity
    fps_list = list(target_fps.values())
    smiles_list = list(target_fps.keys())
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps_list)

    # Filter results
    filtered = {smile: sim for smile, sim in zip(smiles_list, similarities) if sim >= threhsold}
    # Get matching SMILES
    matching_smiles = list(filtered.keys())

    # Filter db
    filtered_db = db[db['SMILES'].isin(matching_smiles)]
    
    return filtered_db
    
compare_fingerprints("CC12CCC3c4ccc(cc4CCC3C1CCC2O)O")