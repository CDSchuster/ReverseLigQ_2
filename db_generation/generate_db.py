from db_generation.pdb_db import generate_pdb_db
from db_generation.generate_dataset import generate_smiles_dataset
import logging


log = logging.getLogger("generateDB_log")


def run_database_generation():
    #generate_pdb_db.main()
    log.info("Starting PDB smiles pairs dataset generation")
    pdb_smiles_dataset = generate_smiles_dataset()