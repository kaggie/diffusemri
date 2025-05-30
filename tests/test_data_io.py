import pytest
import numpy as np
import os
import warnings # For catching UserWarning
import nibabel as nib
from dipy.core.gradients import GradientTable

# Assuming the data_io module and its functions are structured as previously implemented
# and available in the PYTHONPATH for diffusemri.
from diffusemri.data_io import (
    load_nifti_dwi, 
    load_nifti_mask, 
    load_fsl_bvecs, 
    load_fsl_bvals, 
    create_gradient_table,
    load_nifti_study
)

# --- Test Data Generation Utilities ---

def _create_dummy_nifti_file(filepath, data_shape, affine, dtype, data_value_func=None):
    """Helper to create a generic NIfTI file."""
    if data_value_func is None:
        data = np.zeros(data_shape, dtype=dtype)
    else:
        data = data_value_func(data_shape).astype(dtype)
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(filepath)) # Ensure filepath is string

def _create_dummy_bval_file(filepath, content):
    """Writes content to a bval file."""
    with open(str(filepath), 'w') as f:
        f.write(content)

def _create_dummy_bvec_file(filepath, content):
    """Writes content to a bvec file."""
    with open(str(filepath), 'w') as f:
        f.write(content)

# --- Test Class ---

class TestDataIO:
    # --- Test Low-Level Loaders ---
    def test_load_nifti_dwi(self, tmp_path):
        dwi_filepath = tmp_path / "dummy_dwi.nii.gz"
        shape_dwi = (2, 2, 2, 3)
        affine_dwi = np.array([[1,0,0,10],[0,1,0,20],[0,0,1,30],[0,0,0,1]], dtype=float)
        _create_dummy_nifti_file(dwi_filepath, shape_dwi, affine_dwi, dtype=np.float32,
                                 data_value_func=lambda s: np.arange(np.prod(s)).reshape(s) + 1)

        data, affine = load_nifti_dwi(str(dwi_filepath))
        assert data.shape == shape_dwi
        assert data.dtype == np.float32
        assert np.allclose(affine, affine_dwi)

        # Test FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_nifti_dwi(str(tmp_path / "non_existent_dwi.nii.gz"))

        # Test ValueError for wrong dimensions (e.g., loading 3D as DWI)
        mask_filepath_3d = tmp_path / "dummy_3d_for_dwi_test.nii.gz"
        _create_dummy_nifti_file(mask_filepath_3d, (2,2,2), np.eye(4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 4D DWI data"):
            load_nifti_dwi(str(mask_filepath_3d))
            
        # Test nibabel.filebasedimages.ImageFileError (e.g. empty file)
        empty_nifti_path = tmp_path / "empty.nii.gz"
        open(empty_nifti_path, 'a').close() # Create an empty file
        with pytest.raises(nib.filebasedimages.ImageFileError): # Nibabel raises this for empty files
            load_nifti_dwi(str(empty_nifti_path))

    def test_load_nifti_mask(self, tmp_path):
        mask_filepath = tmp_path / "dummy_mask.nii.gz"
        shape_mask = (2, 2, 2)
        affine_mask = np.array([[2,0,0,5],[0,2,0,15],[0,0,2,25],[0,0,0,1]], dtype=float)
        # Create mask with some non-zero values to test boolean conversion
        initial_mask_data_func = lambda s: np.random.choice([0, 1, 2], size=s).reshape(s)
        _create_dummy_nifti_file(mask_filepath, shape_mask, affine_mask, dtype=np.uint8,
                                 data_value_func=initial_mask_data_func)

        data, affine = load_nifti_mask(str(mask_filepath))
        assert data.shape == shape_mask
        assert data.dtype == bool
        assert np.allclose(affine, affine_mask)
        # Check that some values are True if original data had non-zeros
        # Re-generate the initial data to compare
        original_data_for_check = initial_mask_data_func(shape_mask)
        assert np.any(data) if np.any(original_data_for_check) else not np.any(data)


        # Test FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_nifti_mask(str(tmp_path / "non_existent_mask.nii.gz"))

        # Test ValueError for wrong dimensions (e.g., loading 4D as mask)
        dwi_filepath_4d = tmp_path / "dummy_4d_for_mask_test.nii.gz"
        _create_dummy_nifti_file(dwi_filepath_4d, (2,2,2,3), np.eye(4), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 3D mask data"):
            load_nifti_mask(str(dwi_filepath_4d))

        # Test nibabel.filebasedimages.ImageFileError (e.g. malformed file - use text file)
        malformed_nifti_path = tmp_path / "malformed.nii.gz"
        with open(malformed_nifti_path, "w") as f: f.write("this is not a nifti")
        with pytest.raises(nib.filebasedimages.ImageFileError):
            load_nifti_mask(str(malformed_nifti_path))


    def test_load_fsl_bvals(self, tmp_path):
        bval_filepath = tmp_path / "dummy.bval"
        expected_bvals = np.array([0, 1000, 2000])
        
        # Test space separated
        _create_dummy_bval_file(bval_filepath, "0 1000 2000")
        assert np.array_equal(load_fsl_bvals(str(bval_filepath)), expected_bvals)

        # Test comma separated
        _create_dummy_bval_file(bval_filepath, "0,1000,2000")
        assert np.array_equal(load_fsl_bvals(str(bval_filepath)), expected_bvals)

        # Test single line (np.loadtxt handles newlines within numbers as part of the same line if only numbers)
        _create_dummy_bval_file(bval_filepath, "0\n1000\n2000") 
        assert np.array_equal(load_fsl_bvals(str(bval_filepath)), expected_bvals)
        
        # Test single value
        _create_dummy_bval_file(bval_filepath, "1500")
        assert np.array_equal(load_fsl_bvals(str(bval_filepath)), np.array([1500]))


        # Test FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_fsl_bvals(str(tmp_path / "non_existent.bval"))

        # Test ValueError for non-numeric content
        _create_dummy_bval_file(bval_filepath, "0 1000 abc")
        with pytest.raises(ValueError, match="Failed to load or parse bval file"):
            load_fsl_bvals(str(bval_filepath))
            
        # Test ValueError for multi-line content that doesn't squeeze to 1D (np.loadtxt limitation)
        _create_dummy_bval_file(bval_filepath, "0 1000\n2000 3000")
        with pytest.raises(ValueError, match="could not be converted to a 1D array"):
             load_fsl_bvals(str(bval_filepath))


    def test_load_fsl_bvecs(self, tmp_path):
        bvec_filepath = tmp_path / "dummy.bvec"
        
        # Test Nx3 format (space separated)
        content_nx3_space = "0 0 0\n1 0 0\n0 1 0"
        expected_nx3 = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
        _create_dummy_bvec_file(bvec_filepath, content_nx3_space)
        loaded_nx3_space = load_fsl_bvecs(str(bvec_filepath))
        assert np.allclose(loaded_nx3_space, expected_nx3)
        assert loaded_nx3_space.shape == (3,3)

        # Test Nx3 format (comma separated)
        content_nx3_comma = "0,0,0\n1,0,0\n0,1,0"
        _create_dummy_bvec_file(bvec_filepath, content_nx3_comma)
        loaded_nx3_comma = load_fsl_bvecs(str(bvec_filepath))
        assert np.allclose(loaded_nx3_comma, expected_nx3)

        # Test 3xN format (space separated, 3 rows, 4 columns)
        content_3xn_space = "0 1 0 0\n0 0 1 0\n0 0 0 1" 
        expected_3xn_transposed = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
        _create_dummy_bvec_file(bvec_filepath, content_3xn_space)
        loaded_3xn_space = load_fsl_bvecs(str(bvec_filepath))
        assert np.allclose(loaded_3xn_space, expected_3xn_transposed)
        assert loaded_3xn_space.shape == (4,3)
        
        # Test FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_fsl_bvecs(str(tmp_path / "non_existent.bvec"))

        # Test ValueError for non-numeric content
        _create_dummy_bvec_file(bvec_filepath, "0 0 0\n1 0 abc\n0 1 0")
        with pytest.raises(ValueError, match="Failed to load or parse bvec file"):
            load_fsl_bvecs(str(bvec_filepath))

        # Test ValueError for incorrect shapes (e.g., 4xN)
        _create_dummy_bvec_file(bvec_filepath, "0 0 0 0\n1 0 0 0\n0 1 0 0\n0 0 1 0") # 4x4
        with pytest.raises(ValueError, match="b-vectors in .* must be 3xN or Nx3"):
            load_fsl_bvecs(str(bvec_filepath))


    # --- Test create_gradient_table() ---
    def test_create_gradient_table_valid(self):
        bvals = np.array([0, 1000, 1000, 0])
        bvecs = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,0]], dtype=float)
        gtab = create_gradient_table(bvals, bvecs)
        assert isinstance(gtab, GradientTable)
        assert np.array_equal(gtab.bvals, bvals)
        assert np.allclose(gtab.bvecs, bvecs)
        assert np.sum(gtab.b0s_mask) == 2 

    def test_create_gradient_table_invalid(self):
        with pytest.raises(ValueError, match="Number of b-values .* must match the number of b-vectors"):
            create_gradient_table(np.array([0, 1000]), np.array([[0,0,0],[1,0,0],[0,1,0]]))
        with pytest.raises(ValueError, match="bvecs must have shape .*N, 3.*"):
            create_gradient_table(np.array([0,1000]), np.array([[0,0],[1,0]]))
        # Test Dipy's internal ValueError for non-unit bvecs
        with pytest.raises(ValueError, match="Dipy's gradient_table creation failed"):
             create_gradient_table(np.array([1000]), np.array([[2,0,0]]), b0_threshold=0)


    # --- Test High-Level Loader load_nifti_study() ---
    @pytest.fixture
    def full_study_files(self, tmp_path):
        dwi_fp = tmp_path / "dwi.nii.gz"
        bval_fp = tmp_path / "dwi.bval"
        bvec_fp = tmp_path / "dwi.bvec"
        mask_fp = tmp_path / "mask.nii.gz"

        dwi_shape = (3,3,3,4)
        # Make affine non-identity and with different scaling to test affine propagation
        dwi_affine = np.array([
            [2,0,0,10],
            [0,2.5,0,20],
            [0,0,3,30],
            [0,0,0,1]
        ], dtype=float)
        _create_dummy_nifti_file(dwi_fp, dwi_shape, dwi_affine, dtype=np.float32)
        _create_dummy_bval_file(bval_fp, "0 1000 1000 0")
        _create_dummy_bvec_file(bvec_fp, "0 0 0\n1 0 0\n0 1 0\n0 0 0") 
        _create_dummy_nifti_file(mask_fp, dwi_shape[:3], dwi_affine, dtype=np.uint8, 
                                 data_value_func=lambda s: np.ones(s)) # Ensure mask is not all zeros
        
        return str(dwi_fp), str(bval_fp), str(bvec_fp), str(mask_fp), dwi_affine, dwi_shape

    def test_load_nifti_study_basic(self, full_study_files):
        dwi_fp, bval_fp, bvec_fp, mask_fp, dwi_affine_exp, dwi_shape_exp = full_study_files
        
        dwi_data, gtab, mask_data, affine = load_nifti_study(
            dwi_fp, bval_fp, bvec_fp, mask_path=mask_fp
        )
        
        assert dwi_data.shape == dwi_shape_exp
        assert isinstance(gtab, GradientTable)
        assert mask_data.shape == dwi_shape_exp[:3]
        assert mask_data.dtype == bool
        assert np.all(mask_data) # Since we created a mask of ones
        assert np.allclose(affine, dwi_affine_exp)
        assert len(gtab.bvals) == dwi_shape_exp[3]

    def test_load_nifti_study_no_mask(self, full_study_files):
        dwi_fp, bval_fp, bvec_fp, _, _, _ = full_study_files
        _, gtab, mask_data, _ = load_nifti_study(dwi_fp, bval_fp, bvec_fp, mask_path=None)
        assert mask_data is None

    def test_load_nifti_study_errors(self, full_study_files, tmp_path):
        dwi_fp, bval_fp, bvec_fp, mask_fp, _, dwi_shape_exp = full_study_files

        with pytest.raises(FileNotFoundError):
            load_nifti_study(str(tmp_path / "missing.nii.gz"), bval_fp, bvec_fp)
        with pytest.raises(FileNotFoundError):
            load_nifti_study(dwi_fp, str(tmp_path / "missing.bval"), bvec_fp)
        with pytest.raises(FileNotFoundError):
            load_nifti_study(dwi_fp, bval_fp, str(tmp_path / "missing.bvec"))

        bval_short_fp = tmp_path / "short.bval"
        _create_dummy_bval_file(bval_short_fp, "0 1000 0")
        with pytest.raises(ValueError, match="Number of volumes in DWI data .* does not match number of b-values"):
            load_nifti_study(dwi_fp, str(bval_short_fp), bvec_fp)

        mask_wrong_shape_fp = tmp_path / "mask_wrong_shape.nii.gz"
        _create_dummy_nifti_file(mask_wrong_shape_fp, (2,2,2), np.eye(4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Spatial dimensions of DWI data .* and mask .* do not match"):
            load_nifti_study(dwi_fp, bval_fp, bvec_fp, mask_path=str(mask_wrong_shape_fp))

        mask_diff_affine_fp = tmp_path / "mask_diff_affine.nii.gz"
        affine_mask_diff = np.eye(4); affine_mask_diff[0,0]=3 
        _create_dummy_nifti_file(mask_diff_affine_fp, dwi_shape_exp[:3], affine_mask_diff, dtype=np.uint8)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always") # Cause all warnings to always be triggered.
            load_nifti_study(dwi_fp, bval_fp, bvec_fp, mask_path=str(mask_diff_affine_fp))
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "Warning: DWI affine and mask affine are different" in str(w[-1].message)
