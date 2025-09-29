import pytest
from unittest.mock import patch
import requests
from pdb_db.LigandPfamDataRequest import parallelize_pfam_ligand_request, fetch_url


# URLs for requests
pfam_base_url = "https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}"
ligand_base_url = "https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}"

# Real PDB IDs for different cases
ligand_fail_ids = ["4HB1", "9B0Q", "3QIK", "9B3A", "5GXI"]
pfam_fail_ids = ["4JPR", "6YFM", "9IW3", "6A28", "3DIP"]
working_pdb_ids = ["3D12", "5LGJ", "2XE6", "8RHW"]

# Each error type is paired with the appropriate exception and message
error_cases = [
    ("500", requests.exceptions.HTTPError("500 Internal Server Error")),
    ("503", requests.exceptions.HTTPError("503 Service Unavailable")),
    ("504", requests.exceptions.HTTPError("504 Gateway Timeout")),
    ("HTTPSConnectionPool", requests.exceptions.ConnectionError("HTTPSConnectionPool(host='example.com', port=443): Max retries exceeded"))
]


@pytest.mark.parametrize("pdb_id", pfam_fail_ids)
def test_fetch_url_pfam_fail(pdb_id):
    # 404 error for Pfam data request
    url = pfam_base_url.format(pdb_id=pdb_id)
    result = fetch_url(pdb_id, url)
    assert "404" in result[2]


@pytest.mark.parametrize("pdb_id", ligand_fail_ids)
def test_fetch_url_ligand_fail(pdb_id):
    # 404 error for Ligand data request
    url = ligand_base_url.format(pdb_id=pdb_id)
    result = fetch_url(pdb_id, url)
    print((result[2]))
    assert "404" in result[2]


@pytest.mark.parametrize("pdb_id", working_pdb_ids)
@pytest.mark.parametrize("error_label, exception", error_cases)
@patch("requests.get")
def test_fetch_url_errors(mock_get, pdb_id, error_label, exception):
    # Simulate 5 failed attempts by always raising the exception
    mock_get.side_effect = exception

    # Alternate between pfam and ligand URLs to cover both failtypes
    url = f"https://www.ebi.ac.uk/pdbe/graph-api/mappings/pfam/{pdb_id}" if int(pdb_id[-1], 36) % 2 == 0 else f"https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_excluding_branched/{pdb_id}"

    result = fetch_url(pdb_id, url)

    expected_failtype = "pfam_fail" if "pfam" in url else "ligand_fail"
    assert result[2] == expected_failtype


@pytest.mark.parametrize("working_pdb_ids", [working_pdb_ids])
def test_parallelize_pfam_ligand_request_success(working_pdb_ids):
    results_dict, fails_dict = parallelize_pfam_ligand_request(working_pdb_ids)
    
    for pdb_id in working_pdb_ids:
        
        assert pdb_id in results_dict
        assert "Pfam_url" in results_dict[pdb_id] or "ligand_url" in results_dict[pdb_id]

    # Verify fails_dict is empty
    assert fails_dict["pfam_fails"] == []
    assert fails_dict["ligand_fails"] == []


@patch("generate_db.LigandPfamDataRequest.fetch_url")
def test_parallelize_pfam_ligand_request_failures(mock_fetch_url):
    # Mock fetch_url to return failures for specific URLs
    def mock_fetch(pdb_id, url):
        if "pfam" in url:
            return (pdb_id, url, "pfam_fail")
        else:
            return (pdb_id, url, "ligand_fail")

    mock_fetch_url.side_effect = mock_fetch

    pdb_ids = ["4HB1", "9B0Q", "3QIK", "9B3A", "5GXI"]
    results_dict, fails_dict = parallelize_pfam_ligand_request(pdb_ids)

    # Verify results_dict is empty
    assert results_dict == {}

    # Verify fails_dict contains all PDB IDs in the appropriate fail lists
    assert set(fails_dict["pfam_fails"]) == set(pdb_ids)
    assert set(fails_dict["ligand_fails"]) == set(pdb_ids)


def test_parallelize_pfam_ligand_request_empty():
    # Test with an empty list of PDB IDs
    fail_ids = ["4HB1", "3QIK"]
    results_dict, fails_dict = parallelize_pfam_ligand_request(ligand_fail_ids)

    # Verify results_dict and fails_dict are both empty
    assert results_dict == {"4HB1":{}, "3QIK":{}}
    assert fails_dict["pfam_fails"] == []
    assert fails_dict["ligand_fails"] == []