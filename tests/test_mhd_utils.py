import unittest
from unittest import mock
import os
import tempfile
import numpy as np
import SimpleITK as sitk
import nibabel as nib

# Adjust import path based on actual project structure
from data_io.mhd_utils import read_mhd_data, write_mhd_data

class TestMhdUtils(unittest.TestCase):

    def create_dummy_mhd_file(self, filepath, data_shape=(10,12,14), is_4d=False, is_dwi=False):
        """Helper to create a dummy MHD/MHA file using SimpleITK for testing."""
        if is_4d:
            actual_shape = data_shape + (5,) if len(data_shape) == 3 else data_shape
        else:
            actual_shape = data_shape

        array_data = np.random.rand(*actual_shape).astype(np.float32)

        # SITK expects data in ZYX or TZYX order
        sitk_data_order = array_data.transpose(list(range(array_data.ndim))[::-1]).copy()
        image = sitk.GetImageFromArray(sitk_data_order)

        # Set some basic spatial metadata
        image.SetSpacing([1.1, 1.2, 1.3][:image.GetDimension()])
        image.SetOrigin([-10.0, -12.0, -14.0][:image.GetDimension()])
        # Identity direction matrix for simplicity in dummy file creation
        # For 3D: [1,0,0, 0,1,0, 0,0,1]
        # For 2D: [1,0, 0,1]
        dir_matrix_flat = np.eye(image.GetDimension()).flatten().tolist()
        image.SetDirection(dir_matrix_flat)

        if is_dwi and image.GetDimension() == 4: # Only add DWI tags if 4D
            image.SetMetaData("modality", "DWMRI")
            image.SetMetaData("DWMRI_b-value", "1000.0") # Example reference b-value
            # Example for 5 gradients
            image.SetMetaData("DWMRI_gradient_0000", "0.0 0.0 0.0 0.0") # b0
            image.SetMetaData("DWMRI_gradient_0001", "1.0 0.0 0.0 1000.0")
            image.SetMetaData("DWMRI_gradient_0002", "0.0 1.0 0.0 1000.0")
            image.SetMetaData("DWMRI_gradient_0003", "0.0 0.0 1.0 1000.0")
            image.SetMetaData("DWMRI_gradient_0004", "0.707 0.707 0.0 1000.0")
            # Measurement frame identity
            image.SetMetaData("measurement frame", "1 0 0 0 1 0 0 0 1")


        # Determine if .mha (data in header) or .mhd (separate .raw file)
        # For simplicity in tests, use .mha to avoid managing separate .raw files.
        # If filepath ends with .mhd, SITK might create .raw. Let's ensure .mha for single file.
        if filepath.endswith(".mhd"):
            # In tests, it's easier if it's self-contained.
            # However, to test .mhd/.raw pair, one might need to handle raw file creation.
            # For now, this helper will produce what SITK does based on extension.
            pass

        sitk.WriteImage(image, filepath)
        return array_data # Return data in NumPy xyz order for comparison

    def test_write_and_read_mhd_basic_3d(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mha_filepath = os.path.join(tmpdir, "test_basic_3d.mha")

            data_np_xyz = np.random.rand(10, 12, 15).astype(np.float32)
            # NIfTI-style affine (RAS)
            affine_nifti = np.array([
                [-1.0, 0.0, 0.0, 10.0],
                [0.0, -1.0, 0.0, 12.0],
                [0.0, 0.0, 1.5, -15.0],
                [0.0, 0.0, 0.0, 1.0]
            ])

            write_mhd_data(mha_filepath, data_np_xyz, affine_nifti)
            self.assertTrue(os.path.exists(mha_filepath))

            read_data_np, read_affine, _, _, read_meta = read_mhd_data(mha_filepath)

            self.assertIsNotNone(read_data_np)
            self.assertIsNotNone(read_affine)
            np.testing.assert_array_almost_equal(read_data_np, data_np_xyz, decimal=5)
            np.testing.assert_array_almost_equal(read_affine, affine_nifti, decimal=5)
            self.assertEqual(read_meta.get('ObjectType'), 'Image') # Default from SITK
            self.assertEqual(read_meta.get('NDims'), '3')

    def test_write_and_read_mhd_dwi_4d(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mha_filepath = os.path.join(tmpdir, "test_dwi_4d.mha")

            data_np_xyzt = np.random.rand(8, 9, 7, 5).astype(np.float32) # X,Y,Z,Time/Grads
            affine_nifti = np.array([
                [2.0, 0.0, 0.0, -80.0],
                [0.0, 2.0, 0.0, -90.0],
                [0.0, 0.0, 2.5, -70.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            bvals_np = np.array([0, 1000, 1000, 2000, 0], dtype=float)
            # Assume bvecs are in NIfTI image coordinate system for writing
            bvecs_img_space = np.array([
                [0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,0,0]
            ], dtype=float)

            custom_fields = {"TestCustomKey": "TestCustomValue"}

            write_mhd_data(mha_filepath, data_np_xyzt, affine_nifti,
                           bvals=bvals_np, bvecs=bvecs_img_space,
                           custom_metadata=custom_fields)
            self.assertTrue(os.path.exists(mha_filepath))

            read_data, read_affine, read_bvals, read_bvecs, read_meta = read_mhd_data(mha_filepath)

            self.assertIsNotNone(read_data)
            np.testing.assert_array_almost_equal(read_data, data_np_xyzt, decimal=5)
            self.assertIsNotNone(read_affine)
            np.testing.assert_array_almost_equal(read_affine, affine_nifti, decimal=5)

            self.assertIsNotNone(read_bvals)
            np.testing.assert_array_almost_equal(read_bvals, bvals_np)

            self.assertIsNotNone(read_bvecs)
            # B-vectors written are transformed to world, then read and re-transformed to image.
            # If all transformations are correct, they should match the original image-space bvecs.
            np.testing.assert_array_almost_equal(read_bvecs, bvecs_img_space, decimal=5)

            self.assertEqual(read_meta.get('modality'), 'DWMRI')
            self.assertEqual(read_meta.get('TestCustomKey'), 'TestCustomValue')
            self.assertTrue(any(k.startswith('DWMRI_gradient_') for k in read_meta.keys()))

    def test_read_mhd_dwi_parsing(self):
        # Test the DWI metadata parsing logic of read_mhd_data more directly
        with tempfile.TemporaryDirectory() as tmpdir:
            mha_filepath = os.path.join(tmpdir, "dwi_parse_test.mha")
            # Create a dummy MHD with specific DWI tags using SITK directly
            # Data shape (x,y,z,t) -> SITK (t,z,y,x)
            data_np_xyzt = np.random.rand(5,5,5,3).astype(np.float32)
            sitk_data = data_np_xyzt.transpose(3,2,1,0).copy()
            img = sitk.GetImageFromArray(sitk_data)
            img.SetSpacing([1.0,1.0,1.0])
            img.SetOrigin([0.0,0.0,0.0])
            img.SetDirection(np.eye(3).flatten().tolist()) # Identity direction for spatial part

            img.SetMetaData("modality", "DWMRI")
            img.SetMetaData("DWMRI_b-value", "1000.0") # Global b-value
            img.SetMetaData("DWMRI_gradient_0000", "0.0 0.0 0.0") # b0, b-value should be 0
            img.SetMetaData("DWMRI_gradient_0001", "1.0 0.0 0.0") # b=1000 due to global
            img.SetMetaData("DWMRI_gradient_0002", "0.0 1.0 0.0 1000.0") # b=1000, explicit in line

            sitk.WriteImage(img, mha_filepath)

            _, _, bvals, bvecs, _ = read_mhd_data(mha_filepath)

            expected_bvals = np.array([0.0, 1000.0, 1000.0])
            expected_bvecs = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
            # Normalize expected bvecs (as the read function re-normalizes after rotation)
            for i in range(expected_bvecs.shape[0]):
                norm = np.linalg.norm(expected_bvecs[i])
                if norm > 1e-6: expected_bvecs[i] /= norm


            np.testing.assert_array_almost_equal(bvals, expected_bvals, decimal=5)
            np.testing.assert_array_almost_equal(bvecs, expected_bvecs, decimal=5)


if __name__ == '__main__':
    unittest.main()
