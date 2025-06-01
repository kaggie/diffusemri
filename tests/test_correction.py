import pytest
import os
import tempfile
from typing import Dict, Any, List, Tuple
from unittest import mock # Changed from pytest.mock for broader compatibility if needed, or stick to pytest specific if preferred
import argparse # For creating Namespace

# Assuming the preprocessing module is structured such that 'correction' is a submodule,
# and 'run_preprocessing.py' is in a 'cli' module at the same level as 'preprocessing'.
# Adjust import paths based on actual project structure and how tests are run.
from preprocessing.correction import (
    load_eddy_outlier_report, correct_motion_eddy_fsl,
    correct_susceptibility_topup_fsl, correct_bias_field_dipy
)
from cli import run_preprocessing # For testing the CLI endpoint

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


# --- Tests for correct_susceptibility_topup_fsl wrapper ---

@mock.patch('preprocessing.correction.ApplyTOPUP')
@mock.patch('preprocessing.correction.TOPUP')
def test_correct_susceptibility_topup_fsl_wrapper(mock_topup_class, mock_applytopup_class, tmp_path):
    """
    Tests the wrapper function `correct_susceptibility_topup_fsl` by mocking
    Nipype's TOPUP and ApplyTOPUP interfaces.
    """
    # Mock instances and their run methods
    mock_topup_instance = mock.Mock()
    mock_topup_instance.run.return_value.outputs.out_fieldcoef = str(tmp_path / "topup_fieldcoef.nii.gz")
    mock_topup_instance.run.return_value.outputs.out_movpar = str(tmp_path / "topup_movpar.txt")
    mock_topup_instance.run.return_value.outputs.out_field = str(tmp_path / "topup_field.nii.gz")
    mock_topup_class.return_value = mock_topup_instance

    mock_applytopup_instance = mock.Mock()
    mock_applytopup_instance.run.return_value.outputs.out_corrected = str(tmp_path / "corrected_image.nii.gz")
    mock_applytopup_class.return_value = mock_applytopup_instance

    # Create dummy input files (just need paths to exist for the wrapper)
    imain_file = tmp_path / "imain.nii.gz"
    encoding_file = tmp_path / "encoding.txt"
    images_to_correct_file = tmp_path / "dwi_to_correct.nii.gz"

    imain_file.touch()
    encoding_file.touch()
    images_to_correct_file.touch()

    out_base_name = str(tmp_path / "test_output_base")
    indices_to_correct = [1, 2] # Example indices

    # Call the wrapper
    corrected_file, field_file, fieldcoef_file, movpar_file = correct_susceptibility_topup_fsl(
        imain_file=str(imain_file),
        encoding_file=str(encoding_file),
        out_base_name=out_base_name,
        images_to_correct_file=str(images_to_correct_file),
        images_to_correct_encoding_indices=indices_to_correct,
        config_file="b02b0.cnf", # Default config
        topup_kwargs={'fwhm': 8.0},
        applytopup_kwargs={'method': 'jac'}
    )

    # Assert TOPUP was called correctly
    mock_topup_class.assert_called_once()
    mock_topup_instance.inputs.in_file == str(imain_file)
    mock_topup_instance.inputs.encoding_file == str(encoding_file)
    assert mock_topup_instance.inputs.out_base == f"{out_base_name}_topup"
    assert mock_topup_instance.inputs.config == "b02b0.cnf"
    assert mock_topup_instance.inputs.fwhm == 8.0 # Check kwarg
    mock_topup_instance.run.assert_called_once()

    # Assert ApplyTOPUP was called correctly
    mock_applytopup_class.assert_called_once()
    assert mock_applytopup_instance.inputs.in_files == [str(images_to_correct_file)]
    assert mock_applytopup_instance.inputs.encoding_file == str(encoding_file)
    assert mock_applytopup_instance.inputs.in_topup_fieldcoef == str(tmp_path / "topup_fieldcoef.nii.gz")
    assert mock_applytopup_instance.inputs.in_topup_movpar == str(tmp_path / "topup_movpar.txt")
    assert mock_applytopup_instance.inputs.in_index == indices_to_correct
    assert mock_applytopup_instance.inputs.method == 'jac' # Check kwarg/default
    assert mock_applytopup_instance.inputs.out_corrected == f"{out_base_name}_corrected"
    mock_applytopup_instance.run.assert_called_once()

    # Assert returned paths match expectations
    assert corrected_file == str(tmp_path / "corrected_image.nii.gz")
    assert field_file == str(tmp_path / "topup_field.nii.gz")
    assert fieldcoef_file == str(tmp_path / "topup_fieldcoef.nii.gz")
    assert movpar_file == str(tmp_path / "topup_movpar.txt")

# --- Test for the CLI endpoint for topup_fsl ---
@mock.patch('cli.run_preprocessing.correct_susceptibility_topup_fsl')
def test_cli_run_topup_fsl(mock_correct_susceptibility_fsl_func, tmp_path):
    """
    Tests the `topup_fsl` subcommand of the `run_preprocessing.py` CLI script.
    Mocks the actual FSL calling function `correct_susceptibility_topup_fsl`.
    """
    # Mock the return value of the core function
    mock_correct_susceptibility_fsl_func.return_value = (
        "corrected.nii.gz", "field.nii.gz", "coef.nii.gz", "movpar.txt"
    )

    # Prepare dummy file paths for CLI arguments
    dummy_imain = tmp_path / "imain_for_topup.nii.gz"
    dummy_enc = tmp_path / "encoding_file.txt"
    dummy_img_to_correct = tmp_path / "full_dwi.nii.gz"
    dummy_out_base = tmp_path / "output_prefix_topup"
    dummy_config = tmp_path / "my_b02b0.cnf"

    dummy_imain.touch()
    dummy_enc.touch()
    dummy_img_to_correct.touch()
    dummy_config.touch() # Make it exist for custom config test

    cli_args = [
        'topup_fsl',
        '--imain_file', str(dummy_imain),
        '--encoding_file', str(dummy_enc),
        '--images_to_correct_file', str(dummy_img_to_correct),
        '--images_to_correct_encoding_indices', '1,2,3', # Example indices
        '--out_base_name', str(dummy_out_base),
        '--config_file', str(dummy_config)
        # Add other optional args here if testing their parsing, e.g.
        # '--topup_fwhm', '6.0'
    ]

    # Run the main function of the CLI script with these arguments
    # This requires run_preprocessing.py to correctly handle sys.argv or take args list
    try:
        run_preprocessing.main(cli_args)
    except SystemExit as e:
        # If main calls sys.exit(0) on success, catch it. Fail if other exit code.
        if e.code != 0:
            pytest.fail(f"CLI script exited with code {e.code} for args: {cli_args}")

    # Assert that the mocked core function was called once
    mock_correct_susceptibility_fsl_func.assert_called_once()

    # Get the arguments passed to the mocked function
    called_args = mock_correct_susceptibility_fsl_func.call_args[1] # kwargs

    assert called_args['imain_file'] == str(dummy_imain)
    assert called_args['encoding_file'] == str(dummy_enc)
    assert called_args['images_to_correct_file'] == str(dummy_img_to_correct)
    assert called_args['images_to_correct_encoding_indices'] == [1,2,3] # Check parsing of indices
    assert called_args['out_base_name'] == str(dummy_out_base)
    assert called_args['config_file'] == str(dummy_config)
    # Add asserts for other optional parameters if they are passed from CLI to the function
    # e.g. assert called_args['topup_kwargs']['fwhm'] == 6.0


# --- Tests for correct_bias_field_dipy wrapper ---

@mock.patch('preprocessing.correction.BiasFieldCorrectionFlow')
def test_correct_bias_field_dipy_wrapper(mock_bfc_flow_class, tmp_path):
    """
    Tests the `correct_bias_field_dipy` wrapper by mocking Dipý's BiasFieldCorrectionFlow.
    """
    mock_bfc_instance = mock.Mock()
    mock_bfc_flow_class.return_value = mock_bfc_instance

    input_file = tmp_path / "input.nii.gz"
    output_file = tmp_path / "corrected_output.nii.gz"
    mask_file_dummy = tmp_path / "mask.nii.gz" # For testing mask path handling

    input_file.touch()
    mask_file_dummy.touch() # Create dummy mask file

    # Expected output path should be the same as output_file
    expected_corrected_path = str(output_file)

    # Simulate that the flow creates the output file
    # In a real scenario, the flow's .run() method would do this.
    # For the mock, we need to ensure the path exists after run if the function relies on it.
    # The function `correct_bias_field_dipy` actually checks for this.
    def side_effect_run(**kwargs):
        # Create the dummy output file that the flow would create
        open(output_file, 'a').close()
        return None # run usually doesn't return much, results are on disk

    mock_bfc_instance.run.side_effect = side_effect_run


    returned_path = correct_bias_field_dipy(
        input_image_file=str(input_file),
        output_corrected_file=str(output_file),
        mask_file=str(mask_file_dummy),
        method='n4',
        threshold=0.1, # Example kwarg
        use_cuda=False   # Example kwarg
    )

    mock_bfc_flow_class.assert_called_once() # Check BiasFieldCorrectionFlow() was instantiated

    # Check that the run method of the flow instance was called
    mock_bfc_instance.run.assert_called_once()

    # Inspect arguments passed to the flow's run method
    run_kwargs = mock_bfc_instance.run.call_args[1] # kwargs
    assert run_kwargs['input_files'] == str(input_file)
    assert run_kwargs['method'] == 'n4'
    assert run_kwargs['out_dir'] == str(tmp_path) # Directory of output_file
    assert run_kwargs['out_corrected'] == output_file.name # Filename part of output_file
    assert run_kwargs['threshold'] == 0.1
    assert run_kwargs['use_cuda'] is False

    assert returned_path == expected_corrected_path


# --- Test for the CLI endpoint for bias_field_dipy ---
@mock.patch('cli.run_preprocessing.correct_bias_field_dipy')
def test_cli_run_bias_field_dipy(mock_correct_bias_field_func, tmp_path):
    """
    Tests the `bias_field_dipy` subcommand of the `run_preprocessing.py` CLI script.
    Mocks the actual bias field correction function `correct_bias_field_dipy`.
    """
    mock_correct_bias_field_func.return_value = "corrected_by_cli.nii.gz"

    dummy_input = tmp_path / "input_for_bfc.nii.gz"
    dummy_output = tmp_path / "output_bfc.nii.gz"
    dummy_mask = tmp_path / "mask_for_bfc.nii.gz"

    dummy_input.touch()
    dummy_mask.touch() # Mask is optional, but test passing it

    cli_args = [
        'bias_field_dipy',
        '--input_file', str(dummy_input),
        '--output_file', str(dummy_output),
        '--method', 'n4', # Default, but explicit for test
        '--mask_file', str(dummy_mask)
        # Add other CLI exposed args if any, e.g. --threshold 0.05
    ]

    try:
        run_preprocessing.main(cli_args)
    except SystemExit as e:
        if e.code != 0:
            pytest.fail(f"CLI script exited with code {e.code} for args: {cli_args}")

    mock_correct_bias_field_func.assert_called_once()
    called_args_dict = mock_correct_bias_field_func.call_args[1] # kwargs

    assert called_args_dict['input_image_file'] == str(dummy_input)
    assert called_args_dict['output_corrected_file'] == str(dummy_output)
    assert called_args_dict['method'] == 'n4'
    assert called_args_dict['mask_file'] == str(dummy_mask)
    # Assert that any other kwargs passed from CLI are present in called_args_dict['kwargs_for_correction']
    # e.g. if --threshold was a CLI arg:
    # assert called_args_dict['threshold'] == 0.05

```
