import unittest
from unittest import mock
import numpy as np
import os
# Import the function to test
from data_io.bruker_utils import read_bruker_dwi_data
# Import exceptions for mocking side effects
from brukerapi.exceptions import UnsuportedDatasetType, IncompleteDataset

# Mock the brukerapi.dataset.Dataset class
@mock.patch('data_io.bruker_utils.Dataset')
class TestBrukerUtils(unittest.TestCase):

    def setUp(self):
        # Common mock attributes for visu_pars, method, acqp
        self.mock_visu_pars = mock.Mock()
        self.mock_visu_pars.as_dict.return_value = {'VisuCoreDim': 3, 'VisuCoreWordType': '_16BIT_SGN_INT'}
        self.mock_visu_pars.get_value.side_effect = self.visu_pars_get_value_side_effect

        self.mock_method = mock.Mock()
        self.mock_method.as_dict.return_value = {}
        self.mock_method.get_value.side_effect = self.method_get_value_side_effect

        self.mock_acqp = mock.Mock()
        self.mock_acqp.as_dict.return_value = {}
        # self.mock_acqp.get_value.side_effect = ... # if needed

        # Default values for parameters
        self.default_visu_core_size = [64, 64, 32] # Cols, Rows, Slices
        self.default_visu_core_extent = [128.0, 128.0, 64.0] # mm
        self.default_visu_core_position = [0.0, 0.0, 0.0] # mm
        self.default_visu_core_orientation = np.eye(3).flatten().tolist()

        self.default_pvm_dw_eff_bval = [0.0, 1000.0, 1000.0]
        self.default_pvm_dw_dir = [[0,0,0], [1,0,0], [0,1,0]] # Example b-vectors

        self.dummy_data_shape_3d_dwi = (32, 64, 64, 3) # Slices, Rows, Cols, DWIs
        self.dummy_image_data_3d_dwi = np.random.rand(*self.dummy_data_shape_3d_dwi).astype(np.int16)


    def visu_pars_get_value_side_effect(self, key, default=None):
        if key == 'VisuCoreSize':
            return self.current_visu_core_size
        elif key == 'VisuCoreExtent':
            return self.current_visu_core_extent
        elif key == 'VisuCorePosition':
            return self.current_visu_core_position
        elif key == 'VisuCoreOrientation':
            return self.current_visu_core_orientation
        elif key == 'VisuCoreDim':
            return len(self.current_visu_core_size) # 3 for 3D, 2 for 2D
        return default

    def method_get_value_side_effect(self, key, default=None):
        if key == 'PVM_DwEffBval':
            return self.current_pvm_dw_eff_bval
        elif key == 'PVM_DwDir':
            return self.current_pvm_dw_dir
        return default

    def configure_mock_dataset(self, mock_dataset_instance, image_data, is_dwi=True):
        mock_dataset_instance.data = image_data
        mock_dataset_instance.visu_pars = self.mock_visu_pars
        mock_dataset_instance.method = self.mock_method
        mock_dataset_instance.acqp = self.mock_acqp

        # Setup current test case's specific values
        self.current_visu_core_size = self.default_visu_core_size[:]
        self.current_visu_core_extent = self.default_visu_core_extent[:]
        self.current_visu_core_position = self.default_visu_core_position[:]
        self.current_visu_core_orientation = self.default_visu_core_orientation[:]

        if is_dwi:
            self.current_pvm_dw_eff_bval = self.default_pvm_dw_eff_bval[:]
            self.current_pvm_dw_dir = [row[:] for row in self.default_pvm_dw_dir]
        else:
            self.current_pvm_dw_eff_bval = None # Or single b0
            self.current_pvm_dw_dir = None


    @mock.patch('data_io.bruker_utils.os.path.exists') # Mock os.path.exists
    def test_read_bruker_dwi_data_success_3d_dwi(self, mock_os_exists, MockBrukerDataset):
        mock_os_exists.return_value = True # Assume path to 2dseq exists
        mock_dataset_instance = MockBrukerDataset.return_value
        self.configure_mock_dataset(mock_dataset_instance, self.dummy_image_data_3d_dwi, is_dwi=True)

        result = read_bruker_dwi_data("dummy/path/to/experiment/1")
        self.assertIsNotNone(result)
        img_data, affine, bvals, bvecs, meta = result

        # Expected NIFTI order (Cols, Rows, Slices, DWIs) from (Slices, Rows, Cols, DWIs)
        expected_shape_nifti = (self.dummy_data_shape_3d_dwi[2], # Cols
                                self.dummy_data_shape_3d_dwi[1], # Rows
                                self.dummy_data_shape_3d_dwi[0], # Slices
                                self.dummy_data_shape_3d_dwi[3]) # DWIs
        self.assertEqual(img_data.shape, expected_shape_nifti)
        self.assertIsNotNone(affine)
        self.assertEqual(affine.shape, (4,4))

        self.assertIsNotNone(bvals)
        self.assertEqual(len(bvals), len(self.default_pvm_dw_eff_bval))
        np.testing.assert_array_almost_equal(bvals, self.default_pvm_dw_eff_bval)

        self.assertIsNotNone(bvecs)
        self.assertEqual(bvecs.shape, (len(self.default_pvm_dw_dir), 3))
        # Note: bvec reorientation is complex. Here we test if it returns something shaped correctly.
        # A more detailed test would require known input bvecs and orientation matrix.
        # For now, we assume the simple reorientation or pass-through in the util is tested by shape.

        self.assertIn('visu_pars', meta)
        self.assertIn('method', meta)

        # Check affine calculation basics
        expected_voxel_sizes = np.array(self.default_visu_core_extent) / np.array(self.default_visu_core_size)
        # Extract scales from affine (diagonal of the rotation/scaling part)
        # Affine[:3,:3] = VisuCoreOrientation.T @ diag(voxel_sizes)
        # If VisuCoreOrientation is eye(3), then diag(affine[:3,:3]) == voxel_sizes
        # For default eye(3) orientation matrix:
        np.testing.assert_array_almost_equal(np.diag(affine[:3,:3]), expected_voxel_sizes)


    @mock.patch('data_io.bruker_utils.os.path.exists')
    def test_read_bruker_non_dwi_data(self, mock_os_exists, MockBrukerDataset):
        mock_os_exists.return_value = True
        mock_dataset_instance = MockBrukerDataset.return_value

        dummy_data_3d_anat_shape = (32, 64, 64) # Slices, Rows, Cols
        dummy_image_data_3d_anat = np.random.rand(*dummy_data_3d_anat_shape).astype(np.int16)
        self.configure_mock_dataset(mock_dataset_instance, dummy_image_data_3d_anat, is_dwi=False)

        # Adjust VisuCoreDim for non-DWI (3D) case
        self.mock_visu_pars.get_value.side_effect = lambda key, default=None: {
            'VisuCoreSize': self.default_visu_core_size,
            'VisuCoreExtent': self.default_visu_core_extent,
            'VisuCorePosition': self.default_visu_core_position,
            'VisuCoreOrientation': self.default_visu_core_orientation,
            'VisuCoreDim': 3 # This case is 3D
        }.get(key, default)


        result = read_bruker_dwi_data("dummy/path/to/anat/1")
        self.assertIsNotNone(result)
        img_data, affine, bvals, bvecs, meta = result

        expected_shape_nifti_anat = (dummy_data_3d_anat_shape[2], # Cols
                                     dummy_data_3d_anat_shape[1], # Rows
                                     dummy_data_3d_anat_shape[0]) # Slices
        self.assertEqual(img_data.shape, expected_shape_nifti_anat)
        self.assertIsNone(bvals) # Expect None for non-DWI
        self.assertIsNone(bvecs)


    @mock.patch('data_io.bruker_utils.os.path.exists')
    def test_bruker_dataset_load_failure(self, mock_os_exists, MockBrukerDataset):
        mock_os_exists.return_value = True # Path seems to exist
        # Simulate different ways Dataset initialization might fail
        MockBrukerDataset.side_effect = UnsuportedDatasetType("Test error")
        result = read_bruker_dwi_data("dummy/path/fail1")
        self.assertIsNone(result)

        # Reset side_effect for next test within this method if needed, or use multiple tests
        MockBrukerDataset.side_effect = IncompleteDataset("Test error incomplete")
        # Try with the direct path attempt inside read_bruker_dwi_data
        MockBrukerDataset.side_effect = [UnsuportedDatasetType("Initial fail"), IncompleteDataset("Direct fail")]
        result = read_bruker_dwi_data("dummy/path/fail2")
        self.assertIsNone(result)


    @mock.patch('data_io.bruker_utils.os.path.exists')
    def test_missing_dwi_parameters(self, mock_os_exists, MockBrukerDataset):
        mock_os_exists.return_value = True
        mock_dataset_instance = MockBrukerDataset.return_value
        self.configure_mock_dataset(mock_dataset_instance, self.dummy_image_data_3d_dwi, is_dwi=True)

        # Simulate method file not providing DWI parameters
        self.current_pvm_dw_eff_bval = None
        self.current_pvm_dw_dir = None

        with self.assertLogs(logger='data_io.bruker_utils', level='WARNING') as log_watcher:
            result = read_bruker_dwi_data("dummy/path/dwi_no_params/1")
            self.assertIsNotNone(result) # Should still return data, but bvals/bvecs will be None
            _, _, bvals, bvecs, _ = result
            self.assertIsNone(bvals)
            self.assertIsNone(bvecs)
            # Check for specific log messages
            self.assertTrue(any("b-values (PVM_DwEffBval or PVM_DwBvalEach) not found" in msg for msg in log_watcher.output))


if __name__ == '__main__':
    unittest.main()
