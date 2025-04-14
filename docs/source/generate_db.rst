generate\_db subpackage
=======================

Subpackage contents
-------------------

This package provides functionality for generating a ligand-Pfam interaction database from PDB data.


- :mod:`generate_db.Interactions`: The functions in this module are designed to request ligand interactions to proteins and store them in a dataframe.
- :mod:`generate_db.LigandPfamModule`: This module contains functions to request PDB IDs that are bound to molecules, request their ligands’ data (SMILEs included) and the Pfam domains in those PDBs.
- :mod:`generate_db.generate_db`: This module generates a database of ligand interactions to Pfam domains by using PDB data.


generate\_db.Interactions module
--------------------------------

.. automodule:: generate_db.Interactions
   :members:
   :show-inheritance:
   :undoc-members:

generate\_db.LigandPfamModule module
------------------------------------

.. automodule:: generate_db.LigandPfamModule
   :members:
   :show-inheritance:
   :undoc-members:

generate\_db.generate\_db module
-------------------------------------

.. automodule:: generate_db.generate_db
   :members:
   :show-inheritance:
   :undoc-members: