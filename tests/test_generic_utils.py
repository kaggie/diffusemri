import unittest
import os
import tempfile
import numpy as np
import h5py
from scipy.io import loadmat, savemat # For direct comparison if needed

# Adjust import path based on actual project structure
from data_io.generic_utils import (
    save_dict_to_hdf5, load_dict_from_hdf5,
    save_dict_to_mat, load_dict_from_mat
)

class TestGenericUtilsHDF5(unittest.TestCase):

    def test_save_and_load_hdf5_basic(self):
        data_dict = {
            'array1': np.random.rand(10, 5).astype(np.float32),
            'array2': np.random.randint(0, 100, size=(3, 4, 2)).astype(np.int16),
            'array3': np.array([1.0, 2.5, 3.7], dtype=np.float64)
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_data.h5")

            save_dict_to_hdf5(data_dict, filepath)
            self.assertTrue(os.path.exists(filepath))

            loaded_data = load_dict_from_hdf5(filepath)

            self.assertEqual(len(loaded_data), len(data_dict))
            for key in data_dict:
                self.assertIn(key, loaded_data)
                np.testing.assert_array_almost_equal(loaded_data[key], data_dict[key])
                self.assertEqual(loaded_data[key].dtype, data_dict[key].dtype)

    def test_load_hdf5_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            load_dict_from_hdf5("/non/existent/file.h5")

    def test_save_hdf5_invalid_input(self):
        with self.assertRaises(TypeError):
            save_dict_to_hdf5("not_a_dict", "test.h5")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_invalid_value.h5")
            # This should log warnings but not raise an error from save_dict_to_hdf5 itself,
            # rather h5py might raise error if it cannot handle the list.
            # The function skips non-ndarray values.
            with self.assertLogs(level='WARNING') as log_watcher:
                 save_dict_to_hdf5({'array1': [1,2,3]}, filepath) # Value is a list, not ndarray
            # Check if a file was created (it might be empty or only contain valid parts)
            # self.assertTrue(os.path.exists(filepath)) # File might be empty
            # Check that a warning was logged
            self.assertTrue(any("Skipping key 'array1' as its value is not a NumPy array" in msg for msg in log_watcher.output))


class TestGenericUtilsMAT(unittest.TestCase):

    def test_save_and_load_mat_basic(self):
        data_dict = {
            'matrix1': np.array([[1, 2], [3, 4]]),
            'vector_col': np.array([[5],[6],[7]]), # Will be 2D column vector
            'vector_row': np.array([8,9,10]),    # 1D, will be row vector by default in MAT
            'string_data': np.array(['test_string']) # String array
        }
        # Note: SciPy's savemat has limitations on what it can save (e.g. complex nested dicts, objects)
        # and how it represents them. For basic NumPy arrays, it's usually fine.

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_data.mat")

            save_dict_to_mat(data_dict, filepath, oned_as='column')
            self.assertTrue(os.path.exists(filepath))

            loaded_data = load_dict_from_mat(filepath)

            self.assertNotIn('__header__', loaded_data)
            self.assertNotIn('__version__', loaded_data)
            self.assertNotIn('__globals__', loaded_data)

            self.assertIn('matrix1', loaded_data)
            np.testing.assert_array_equal(loaded_data['matrix1'], data_dict['matrix1'])

            self.assertIn('vector_col', loaded_data)
            np.testing.assert_array_equal(loaded_data['vector_col'], data_dict['vector_col'])

            self.assertIn('vector_row', loaded_data)
            # 1D arrays saved as 'column' become 2D column vectors in MATLAB.
            # loadmat will read them back as 2D.
            np.testing.assert_array_equal(loaded_data['vector_row'], data_dict['vector_row'].reshape(-1,1))

            self.assertIn('string_data', loaded_data)
            # String arrays are often loaded as object arrays of strings, or char arrays.
            # For simple strings, it usually works out.
            self.assertEqual(str(loaded_data['string_data'][0]), str(data_dict['string_data'][0]))


    def test_load_mat_non_existent(self):
        with self.assertRaises(FileNotFoundError):
            load_dict_from_mat("/non/existent/file.mat")

    def test_save_mat_invalid_input(self):
        with self.assertRaises(TypeError):
            save_dict_to_mat("not_a_dict", "test.mat")

        # SciPy's savemat might raise errors for unsupported structures within the dict.
        # This test depends on scipy.io.savemat's behavior.
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_scipy_error.mat")
            with self.assertRaises(Exception): # Could be TypeError or other SciPy specific error
                 save_dict_to_mat({'unsupported': {1:2, 3:4}}, filepath) # Dict as value

if __name__ == '__main__':
    unittest.main()
