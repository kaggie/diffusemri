import unittest
from unittest import mock
import os
import tempfile
import numpy as np
import nibabel as nib

# Adjust import path based on actual project structure
from data_io.parrec_utils import read_parrec_data, convert_parrec_to_nifti

class TestParrecUtils(unittest.TestCase):

    @mock.patch('nibabel.load')
    def test_read_parrec_data_success(self, mock_nib_load):
        # Mock the PARRECImage object and its methods/attributes
        mock_parrec_image = mock.Mock(spec=nib.parrec.PARRECImage)
        mock_parrec_image.get_fdata.return_value = np.random.rand(10, 10, 10, 5).astype(np.float32)
        mock_parrec_image.affine = np.eye(4)

        mock_header = mock.Mock(spec=nib.parrec.PARRECHeader)
        mock_header.get_bvals_bvecs.return_value = (
            np.array([0, 1000, 1000, 1000, 0]),
            np.random.rand(5, 3)
        )
        mock_parrec_image.header = mock_header

        mock_nib_load.return_value = mock_parrec_image

        filepath = "dummy.par"
        data, affine, bvals, bvecs, header_info = read_parrec_data(filepath, strict_sort=False, scaling='fp')

        mock_nib_load.assert_called_once_with(filepath, strict_sort=False, scaling='fp')
        mock_parrec_image.get_fdata.assert_called_once_with(dtype=np.float32)
        mock_parrec_image.header.get_bvals_bvecs.assert_called_once()

        self.assertIsNotNone(data)
        self.assertIsNotNone(affine)
        self.assertIsNotNone(bvals)
        self.assertIsNotNone(bvecs)
        self.assertIsNotNone(header_info)
        self.assertEqual(header_info.get("source_format"), "PAR/REC")
        self.assertIs(header_info.get("source_header_object"), mock_header)


    @mock.patch('nibabel.load')
    def test_read_parrec_data_non_parrec_image(self, mock_nib_load):
        mock_nib_load.return_value = mock.Mock(spec=nib.Nifti1Image) # Return a different image type

        filepath = "dummy.nii.gz"
        data, affine, bvals, bvecs, header_info = read_parrec_data(filepath)

        self.assertIsNone(data)
        self.assertIsNone(affine)
        self.assertIsNone(bvals)
        self.assertIsNone(bvecs)
        self.assertIsNone(header_info)
        mock_nib_load.assert_called_once_with(filepath, strict_sort=True, scaling='dv')

    @mock.patch('nibabel.load')
    def test_read_parrec_data_load_error(self, mock_nib_load):
        mock_nib_load.side_effect = Exception("Failed to load")
        filepath = "dummy.par"
        data, affine, bvals, bvecs, header_info = read_parrec_data(filepath)
        self.assertIsNone(data)
        self.assertIsNone(header_info)


    @mock.patch('data_io.parrec_utils.read_parrec_data')
    @mock.patch('nibabel.save')
    @mock.patch('numpy.savetxt')
    @mock.patch('os.makedirs')
    def test_convert_parrec_to_nifti_success_dwi(self, mock_makedirs, mock_savetxt, mock_nib_save, mock_read_parrec):
        dummy_data = np.random.rand(10,10,10,5).astype(np.float32)
        dummy_affine = np.eye(4)
        dummy_bvals = np.array([0, 1000, 1000, 1000, 0])
        dummy_bvecs = np.random.rand(5,3)
        dummy_header_info = {"source_format": "PAR/REC"}
        mock_read_parrec.return_value = (dummy_data, dummy_affine, dummy_bvals, dummy_bvecs, dummy_header_info)

        with tempfile.TemporaryDirectory() as tmpdir:
            par_file = os.path.join(tmpdir, "input.par")
            nii_file = os.path.join(tmpdir, "output.nii.gz")
            bval_file = os.path.join(tmpdir, "output.bval")
            bvec_file = os.path.join(tmpdir, "output.bvec")

            success = convert_parrec_to_nifti(par_file, nii_file, bval_file, bvec_file)

            self.assertTrue(success)
            mock_read_parrec.assert_called_once_with(par_file, strict_sort=True, scaling='dv')
            mock_nib_save.assert_called_once()
            # Check that Nifti1Image was created with correct data and affine
            saved_img_arg = mock_nib_save.call_args[0][0]
            self.assertIsInstance(saved_img_arg, nib.Nifti1Image)
            np.testing.assert_array_equal(saved_img_arg.get_fdata(), dummy_data)
            np.testing.assert_array_equal(saved_img_arg.affine, dummy_affine)

            self.assertEqual(mock_savetxt.call_count, 2)
            # Call 1 for bvals
            self.assertEqual(mock_savetxt.call_args_list[0][0][0], bval_file)
            np.testing.assert_array_equal(mock_savetxt.call_args_list[0][0][1], dummy_bvals.reshape(1,-1))
            # Call 2 for bvecs
            self.assertEqual(mock_savetxt.call_args_list[1][0][0], bvec_file)
            np.testing.assert_array_equal(mock_savetxt.call_args_list[1][0][1], dummy_bvecs)


    @mock.patch('data_io.parrec_utils.read_parrec_data')
    @mock.patch('nibabel.save')
    def test_convert_parrec_to_nifti_no_dwi_info(self, mock_nib_save, mock_read_parrec):
        dummy_data = np.random.rand(10,10,10).astype(np.float32) # 3D data
        dummy_affine = np.eye(4)
        mock_read_parrec.return_value = (dummy_data, dummy_affine, None, None, {}) # No bvals/bvecs

        with tempfile.TemporaryDirectory() as tmpdir:
            par_file = os.path.join(tmpdir, "input.par")
            nii_file = os.path.join(tmpdir, "output.nii.gz")

            success = convert_parrec_to_nifti(par_file, nii_file)
            self.assertTrue(success)
            mock_nib_save.assert_called_once()


    @mock.patch('data_io.parrec_utils.read_parrec_data')
    def test_convert_parrec_to_nifti_read_fail(self, mock_read_parrec):
        mock_read_parrec.return_value = (None, None, None, None, None)
        success = convert_parrec_to_nifti("input.par", "output.nii.gz")
        self.assertFalse(success)

if __name__ == '__main__':
    unittest.main()
