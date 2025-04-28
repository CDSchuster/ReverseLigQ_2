from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import rdkit.DataStructs as DataStructs
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
import gzip
import shutil
import subprocess
import tempfile


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

    # Load the database and filter unique SMILES
    db = pd.read_csv("full_DB.csv", index_col=0)
    unique_smiles = db['SMILES'].unique()
    
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
   
    # Generate fingerprint for the query SMILE
    query_mol = Chem.MolFromSmiles(query_smile)
    query_fp = morgan_gen.GetFingerprint(query_mol)
    
    # Generate fingerprints for unique SMILES
    target_fps = {}
    for smile in unique_smiles:
        if type(smile) == str:
            mol = Chem.MolFromSmiles(smile)
            if mol != None:
                target_fps[smile] = morgan_gen.GetFingerprint(mol)

    # Bulk Tanimoto similarity
    fps_list = list(target_fps.values())
    smiles_list = list(target_fps.keys())
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps_list)

    # Filter interactions_db based on Tanimoto similarity
    filtered = {smile: sim for smile, sim in zip(smiles_list, similarities) if sim >= threhsold}
    matching_smiles = list(filtered.keys())
    filtered_db = db[db['SMILES'].isin(matching_smiles)]
    
    return filtered_db

def get_pfam_hmm(pfam_id, release='Pfam35.0'):
    """Download and decompress a Pfam HMM model."""
    url = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_id}?annotation=hmm"
    #base_url = f'https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/{release}/hmm'
    #url = f'{base_url}/{pfam_id}.hmm.gz'
    print(f'Downloading {pfam_id} from {url}')
    temp_dir = tempfile.mkdtemp()
    gz_path = os.path.join(temp_dir, f'{pfam_id}.hmm.gz')
    hmm_path = os.path.join(temp_dir, f'{pfam_id}.hmm')
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f'Failed to download {pfam_id}: HTTP {response.status_code}')
    
    with open(gz_path, 'wb') as f:
        f.write(response.content)
    
    with gzip.open(gz_path, 'rb') as f_in:
        with open(hmm_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    os.remove(gz_path)  # remove the compressed file
    return hmm_path, temp_dir


def run_hmmsearch(hmm_file, fasta_file, output_file):
    """Run hmmsearch."""
    with open(output_file, 'w') as out_f:
        cmd = [
            "hmmsearch",
            hmm_file,
            fasta_file
        ]
        subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, check=True)
    return output_file


def hmmer_to_proteome(pfam_ids, proteome, release='Pfam35.0', output_dir='hmmsearch_results'):
    os.makedirs(output_dir, exist_ok=True)
    for pfam_id in pfam_ids:
        if type(pfam_id) == str and "PF" in pfam_id:
            hmm_path, temp_dir = get_pfam_hmm(pfam_id, release=release)
            output_file = os.path.join(output_dir, f'{pfam_id}_hmmsearch.out')
            run_hmmsearch(hmm_path, proteome, output_file)
        
    

def run_analysis(query_smile, proteome, threshold=0.9, release='Pfam35.0'):
    
    
    filtered_db = compare_fingerprints(query_smile, threshold)
    print(set(filtered_db.pfam_id))
    hmmer_to_proteome(set(filtered_db.pfam_id), proteome)
    
    # Save the filtered database to a CSV file
    filtered_db.to_csv("filtered_DB.csv", index=False)
    

run_analysis("CC12CCC3c4ccc(cc4CCC3C1CCC2O)O", "human_proteome.fasta", threshold=0.9)