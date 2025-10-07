from db_generation.pdb_db import generate_pdb_db
from db_generation.chembl_db import generate_chembl_db


def run_database_generation(chembl_db):
    #generate_pdb_db.main()
    generate_chembl_db.run_chembl_db_pipeline(chembl_sqlite=chembl_db)