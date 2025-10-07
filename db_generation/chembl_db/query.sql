-- ==========================================================
-- ChEMBL → CCD data extraction query
-- ==========================================================
-- This query retrieves all ligand–protein activity pairs
-- for single-protein bioassays ("B" type) from a local ChEMBL
-- SQLite database. It returns only molecules that have either
-- a valid pChEMBL value or an explicit activity annotation
-- ('Active', 'Inactive', etc.), ensuring that both active and
-- inactive compounds are included.
--
-- Output columns:
--   ligand_id  → ChEMBL compound identifier
--   smiles     → Canonical SMILES representation
--   inchikey   → Standardized InChIKey (uppercase)
--   protein    → UniProt accession
--   pchembl    → pChEMBL activity value
--   comment    → Text label (Active / Inactive)
--
-- ==========================================================

SELECT DISTINCT
    md.chembl_id                               AS ligand_id,
    cs.canonical_smiles                        AS smiles,
    UPPER(cs.standard_inchi_key)               AS inchikey,
    csq.accession                              AS protein,
    act.pchembl_value                          AS pchembl,
    act.activity_comment                       AS comment
FROM activities              AS act
JOIN assays                  AS a    ON act.assay_id = a.assay_id
JOIN molecule_dictionary     AS md   ON act.molregno = md.molregno
JOIN compound_structures     AS cs   ON md.molregno = cs.molregno
JOIN target_dictionary       AS td   ON a.tid = td.tid
JOIN target_components       AS tc   ON td.tid = tc.tid
JOIN component_sequences     AS csq  ON tc.component_id = csq.component_id
WHERE a.assay_type = 'B'
  AND td.target_type = 'SINGLE PROTEIN'
  AND (
        act.pchembl_value IS NOT NULL
        OR (
            act.pchembl_value IS NULL
            AND act.activity_comment IN ('Active', 'active', 'inactive', 'Not Active')
        )
      )
  AND cs.canonical_smiles IS NOT NULL
  AND cs.standard_inchi_key IS NOT NULL;
