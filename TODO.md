# RevLigQ 2.0

## TESTS

* Ligand-domain interactions scenarios:
  * One domain in one chain
  * One domain formed by two chains
  * Two domains in one chain
  * Two domains in different chains

## FUNCTIONALITY

* How many PDB IDs have only ligands or only Pfam domains?
* Retry failed SMILE requests
* Retry failed interactions request
* Recover data failed to request by loading it from PDB files
* Determine ligand-domain interactions
* Include Chembl data

## FORMAT

* Add `__init__` modules and organize files and folders following a python package structure

## BUGS

* When requesting data, return the PDB IDs that failed to use them later in retries
