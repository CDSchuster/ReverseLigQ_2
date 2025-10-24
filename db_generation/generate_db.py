from db_generation.pdb_db import generate_pdb_db
from db_generation.generate_dataset import generate_smiles_pairs_dataset
import logging


log = logging.getLogger("generateDB_log")


def run_database_generation(min_actives, max_actives, threshold, min_decoys, max_decoys):
    #generate_pdb_db.main()
    log.info("Starting PDB smiles pairs dataset generation")
    pdb_smiles_dataset = generate_smiles_pairs_dataset(min_actives, max_actives, threshold, min_decoys, max_decoys)