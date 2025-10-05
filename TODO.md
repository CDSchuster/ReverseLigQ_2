# RevLigQ 2.0

## TESTS

## FUNCTIONALITY

### DECOY GENERATION

* Summarise the actives and decoys data as a table

### DATABASE GENERATION

* Why the unrecoverable ligand and pfam data does not cause trouble when creating the dataframes
* Recover data failed to request by loading it from PDB files

## FORMAT

## BUGS

* When getting interactions data from PDB, if no data needs recovery, the function `parallelize_interactions_request()` fails, because it sets max_workers in 0 (since there are 0 tasks to be done)
