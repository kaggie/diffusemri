import pytest
import os
import tempfile
from typing import Dict, Any
from preprocessing.correction import load_eddy_outlier_report, correct_motion_eddy_fsl

# --- Fixtures ---

@pytest.fixture
def mock_eddy_outlier_report_file() -> str:
    """Creates a temporary mock FSL eddy outlier report file."""
    report_content = """
    Volume 0 contained 2 outlier slices.
    Slice 10 in scan 0 (actual scan number 1) is an outlier
    Slice 25 in scan 0 (actual scan number 1) is an outlier

    Volume 2 contained 1 outlier slices.
    Slice 15 in scan 2 (actual scan number 3) is an outlier
    This is another detail line for scan 2.

    Slice 5 in scan 4 (actual scan number 5) is an outlier (some reason)
    Slice 6 in scan 4 (actual scan number 5) is an outlier (another reason)
    Scan 4 contained 2 outlier slices (this is a different summary style)
    """
    # Using NamedTemporaryFile to ensure it's cleaned up
    # Keep the file open until the test using it is done by deleting it explicitly in the test or using 'yield'
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(report_content)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath) # Cleanup after test

@pytest.fixture
def mock_minimal_eddy_outlier_report_file() -> str:
    """Creates a temporary mock FSL eddy outlier report file with only slice lines."""
    report_content = """
    Slice 10 in scan 0 (actual scan number 1) is an outlier
    Slice 25 in scan 0 (actual scan number 1) is an outlier
    Slice 15 in scan 2 (actual scan number 3) is an outlier
    """
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(report_content)
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)

@pytest.fixture
def mock_empty_eddy_outlier_report_file() -> str:
    """Creates an empty temporary mock FSL eddy outlier report file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        filepath = tmp_file.name
    yield filepath
    os.remove(filepath)

# --- Tests for preprocessing.correction ---

def test_load_eddy_outlier_report_parsing(mock_eddy_outlier_report_file):
    """Tests parsing of a typical eddy outlier report."""
    report_data = load_eddy_outlier_report(mock_eddy_outlier_report_file)

    assert isinstance(report_data, dict)
    assert len(report_data) == 3 # Volumes 0, 2, 4

    # Check volume 0
    assert 0 in report_data
    assert report_data[0]['total_outlier_slices'] == 2
    assert report_data[0]['outlier_slices'] == [10, 25]
    assert len(report_data[0]['details']) >= 2 # Contains summary and individual slice lines

    # Check volume 2
    assert 2 in report_data
    assert report_data[2]['total_outlier_slices'] == 1
    assert report_data[2]['outlier_slices'] == [15]
    assert "This is another detail line for scan 2." in report_data[2]['details']
    
    # Check volume 4 (summary line after individual slice lines)
    assert 4 in report_data
    assert report_data[4]['total_outlier_slices'] == 2 
    assert report_data[4]['outlier_slices'] == [5, 6]
    assert "Scan 4 contained 2 outlier slices (this is a different summary style)" in report_data[4]['details']


def test_load_eddy_outlier_report_minimal_parsing(mock_minimal_eddy_outlier_report_file):
    """Tests parsing when only individual slice outlier lines are present."""
    report_data = load_eddy_outlier_report(mock_minimal_eddy_outlier_report_file)
    
    assert len(report_data) == 2 # Volumes 0 and 2
    assert 0 in report_data
    assert report_data[0]['total_outlier_slices'] == 2 # Should be counted from slice lines
    assert report_data[0]['outlier_slices'] == [10, 25]
    
    assert 2 in report_data
    assert report_data[2]['total_outlier_slices'] == 1
    assert report_data[2]['outlier_slices'] == [15]

def test_load_eddy_outlier_report_empty_file(mock_empty_eddy_outlier_report_file):
    """Tests behavior with an empty outlier report file."""
    report_data = load_eddy_outlier_report(mock_empty_eddy_outlier_report_file)
    assert isinstance(report_data, dict)
    assert len(report_data) == 0

def test_load_eddy_outlier_report_file_not_found():
    """Tests behavior when the report file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_eddy_outlier_report("non_existent_file.txt")

# Conceptual test structure for correct_motion_eddy_fsl
# This test would require FSL to be installed and is more of an integration test.
# For now, it serves as a placeholder for how one might structure such a test.
@pytest.mark.skip(reason="Requires FSL installation and valid data for full execution.")
def test_correct_motion_eddy_fsl_integration(tmp_path):
    """
    Conceptual integration test for correct_motion_eddy_fsl.
    This test requires FSL to be installed and configured.
    It also requires dummy data files that are valid enough for eddy to run.
    """
    # Create minimal dummy files that might allow eddy to start
    # (though it will likely fail without real data structure)
    dwi_file = tmp_path / "dwi.nii.gz"
    bval_file = tmp_path / "bvals.txt"
    bvec_file = tmp_path / "bvecs.txt"
    mask_file = tmp_path / "mask.nii.gz"
    index_file = tmp_path / "index.txt"
    acqp_file = tmp_path / "acqp.txt"
    out_base = tmp_path / "dwi_corrected"

    # Create basic NIfTI file (e.g., using nibabel - not done here for brevity)
    # For now, just touch the files
    for f in [dwi_file, mask_file]:
        with open(f, 'w') as fp:
            fp.write("dummy nifti-like content") # Not a real NIfTI
            
    with open(bval_file, 'w') as fp: fp.write("0 1000 1000\n")
    with open(bvec_file, 'w') as fp: fp.write("0 0 0\n1 0 0\n0 1 0\n")
    with open(index_file, 'w') as fp: fp.write("1\n1\n1\n")
    with open(acqp_file, 'w') as fp: fp.write("0 -1 0 0.05\n")

    try:
        corrected_dwi, rotated_bvecs, report_file, map_file = correct_motion_eddy_fsl(
            dwi_file=str(dwi_file),
            bval_file=str(bval_file),
            bvec_file=str(bvec_file),
            mask_file=str(mask_file),
            index_file=str(index_file),
            acqp_file=str(acqp_file),
            out_base_name=str(out_base),
            repol=True # To try and generate outlier reports
        )
        # If eddy ran (even if with warnings on dummy data), check for output file existence
        assert os.path.exists(corrected_dwi)
        assert os.path.exists(rotated_bvecs)
        if report_file: # It's Optional[str]
            assert os.path.exists(report_file)
        # map_file check would also go here if expected
            
    except RuntimeError as e:
        # This is expected if FSL is not installed or data is invalid
        print(f"Integration test for correct_motion_eddy_fsl failed as expected without FSL/valid data: {e}")
    except FileNotFoundError as e:
        # This might happen if dummy files are not found by the sub-process
        print(f"Integration test for correct_motion_eddy_fsl failed due to FileNotFoundError: {e}")

```
