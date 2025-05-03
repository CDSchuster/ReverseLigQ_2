from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import rdkit.DataStructs as DataStructs
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
import gzip
import shutil
import subprocess
import tempfile


log = logging.getLogger("run")


def compare_fingerprints(query_smile, threshold=0.9):
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
    log.info(f"Loaded database with {len(db)} entries")
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

    log.info(f"Generated fingerprints for {len(target_fps)} unique SMILES")
    # Bulk Tanimoto similarity
    fps_list = list(target_fps.values())
    smiles_list = list(target_fps.keys())
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps_list)

    # Filter interactions_db based on Tanimoto similarity
    filtered = {smile: sim for smile, sim in zip(smiles_list, similarities) if sim >= threshold}
    matching_smiles = list(filtered.keys())
    filtered_db = db[db['SMILES'].isin(matching_smiles)]
    log.info(f"Filtered database contains {len(filtered_db)} entries with Tanimoto similarity >= {threshold}")
    
    return filtered_db


def get_pfam_hmm(pfam_id):
    """
    Downloads the Pfam HMM file for a given Pfam ID from the InterPro database.

    Parameters
    ----------
    pfam_id : str
        The Pfam ID for which to download the HMM file

    Returns
    -------
    hmm_path : str
        The path to the downloaded HMM file
    temp_dir : str
        The path to the temporary directory where the HMM file was downloaded
    """

    url = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_id}?annotation=hmm"
    log.info(f'Downloading {pfam_id} from {url}')
    
    # Create a temporary directory to store the downloaded files
    temp_dir = tempfile.mkdtemp()
    gz_path = os.path.join(temp_dir, f'{pfam_id}.hmm.gz')
    hmm_path = os.path.join(temp_dir, f'{pfam_id}.hmm')
    
    # Download the HMM file
    response = requests.get(url)
    if response.status_code != 200:
        log.warning(f'Failed to download {pfam_id}: HTTP {response.status_code}')
        result = None
    
    else:
        # Save the gzipped file and extract it
        with open(gz_path, 'wb') as f:
            f.write(response.content)
        
        # Extract the gzipped file
        with gzip.open(gz_path, 'rb') as f_in:
            with open(hmm_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        result = hmm_path, temp_dir
        os.remove(gz_path)  # remove the compressed file

    return result


def run_hmmsearch(hmm_file, fasta_file, output_file):
    """
    Runs HMMER's hmmsearch command on a given HMM file and a FASTA file.

    Parameters
    ----------
    hmm_file : str
        The path to the HMM file
    fasta_file : str
        The path to the FASTA file containing protein sequences
    output_file : str
        The path to the output file where the results will be saved

    Returns
    -------
    output_file : str
        The path to the output file containing the results of the hmmsearch
    """

    with open(output_file, "w") as out_f:
        cmd = ["hmmsearch", hmm_file, fasta_file]
        subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, check=True)
    return output_file


def hmmer_to_proteome(pfam_ids, fasta_file, output_dir):
    """
    Downloads the HMM files for a list of Pfam IDs and runs HMMER's hmmsearch command on a given FASTA file.

    Parameters
    ----------
    pfam_ids : set
        A set of Pfam IDs for which to download the HMM files and run hmmsearch
    fasta_file : str
        The path to the FASTA file containing protein sequences
    output_dir : str, optional
        The directory where the output files will be saved, by default 'hmmsearch_results'
    """
    pfam_ids = list(pfam_ids)+["Foo", 123, "PFXXXX"]
    os.makedirs(output_dir, exist_ok=True)
    for pfam_id in pfam_ids:
        if type(pfam_id) == str and "PF" in pfam_id:
            log.info(f"Processing Pfam ID: {pfam_id}")
            results = get_pfam_hmm(pfam_id)
            if results != None:
                hmm_path, temp_dir = results
                output_file = os.path.join(output_dir, f"{pfam_id}_hmmsearch.out")
                log.info(f"Running hmmsearch for {pfam_id}...")
                run_hmmsearch(hmm_path, fasta_file, output_file)
        else:
            log.warning(f"Invalid Pfam ID: {pfam_id}. Skipping download.")
        
    
def run_analysis(query_file, fasta_file, threshold=0.9, output_dir="results"):
    """
    Takes a SMILE, compares its fingerprint to a database of molecules using the Tanimoto coefficient,
    and returns the most similar molecules. It then downloads the HMM files for the matching Pfam IDs
    and runs HMMER's hmmsearch command on a given FASTA file.

    Parameters
    ----------
    query_smile : str
        A SMILE string to be compared against the database
    fasta_file : str
        The path to the FASTA file containing protein sequences
    threshold : float, optional
        A threshold for the Tanimoto coefficient, by default 0.9
    output_dir : str, optional
        The directory where the output files will be saved, by default 'results'
    """
    
    with open(query_file, "r") as f:
        query_smile = f.read().strip()

    filtered_db = compare_fingerprints(query_smile, threshold)
    hmmer_to_proteome(set(filtered_db.pfam_id), fasta_file, output_dir)
    
    # Save the filtered database to a CSV file
    filtered_db.to_csv(f"{output_dir}/filtered_DB.csv", index=False)