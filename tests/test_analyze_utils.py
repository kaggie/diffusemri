import unittest
from unittest import mock
import os
import tempfile
import numpy as np
import nibabel as nib

# Adjust import path based on actual project structure
from data_io.analyze_utils import read_analyze_data, write_analyze_data

class TestAnalyzeUtils(unittest.TestCase):

    def create_dummy_analyze_file(self, filepath, data_shape=(10,12,14), data_dtype=np.float32):
        """Helper to create a dummy Analyze file using Nibabel."""
        data = np.random.rand(*data_shape).astype(data_dtype)
        # Create a minimal valid affine for Analyze (often diagonal, or identity for basic tests)
        # Analyze format itself is limited in how it stores orientation.
        # Nibabel tries its best to represent it.
        affine = np.array([
            [-2.0, 0.0, 0.0, (data_shape[0]-1)*2.0/2], # Origin at center of first voxel for X
            [0.0, 2.0, 0.0, -(data_shape[1]-1)*2.0/2], # Origin at center for Y
            [0.0, 0.0, 2.5, -(data_shape[2]-1)*2.5/2], # Origin at center for Z
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Create a minimal Analyze header
        header = nib.analyze.AnalyzeHeader()
        header.set_data_shape(data_shape)
        header.set_data_dtype(data_dtype)
        # Set pixdim (voxel sizes) - first element is for qfac, next 3 are spatial
        header['pixdim'][1:4] = np.abs([affine[0,0], affine[1,1], affine[2,2]])


        img = nib.AnalyzeImage(data, affine, header)
        nib.save(img, filepath)
        return data, affine, header

    def test_write_and_read_analyze_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with .hdr extension (should create .hdr and .img)
            hdr_filepath = os.path.join(tmpdir, "test_basic.hdr")

            original_data, original_affine, _ = self.create_dummy_analyze_file(hdr_filepath)

            self.assertTrue(os.path.exists(hdr_filepath))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_basic.img"))) # Check for .img file

            # Test reading (can use .hdr or .img path)
            read_data, read_affine, read_header = read_analyze_data(hdr_filepath)

            self.assertIsNotNone(read_data)
            self.assertIsNotNone(read_affine)
            self.assertIsNotNone(read_header)

            # Data type from read_analyze_data is forced to float32
            np.testing.assert_array_almost_equal(read_data, original_data.astype(np.float32))
            # Affine comparison can be tricky due to Analyze limitations.
            # Nibabel does its best to reconstruct it. Check for reasonable closeness.
            np.testing.assert_array_almost_equal(read_affine, original_affine, decimal=5)

            self.assertIsInstance(read_header, nib.analyze.AnalyzeHeader)
            np.testing.assert_array_equal(read_header.get_data_shape(), original_data.shape)

    def test_write_analyze_data_different_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_int16 = os.path.join(tmpdir, "test_int16.hdr")
            data_int16 = (np.random.rand(5,5,5) * 100).astype(np.int16)
            affine = np.eye(4)

            write_analyze_data(filepath_int16, data_int16, affine)
            read_data, _, _ = read_analyze_data(filepath_int16)
            # read_analyze_data casts to float32
            self.assertEqual(read_data.dtype, np.float32)
            np.testing.assert_array_almost_equal(read_data, data_int16.astype(np.float32))

    def test_read_analyze_non_existent(self):
        data, affine, header = read_analyze_data("/non/existent/file.hdr")
        self.assertIsNone(data)
        self.assertIsNone(affine)
        self.assertIsNone(header)

    @mock.patch('nibabel.load')
    def test_read_analyze_nibabel_error(self, mock_nib_load):
        mock_nib_load.side_effect = nib.filebasedimages.ImageFileError("Test nibabel error")
        data, affine, header = read_analyze_data("dummy.hdr") # Path doesn't matter due to mock
        self.assertIsNone(data)
        self.assertIsNone(affine)
        self.assertIsNone(header)


if __name__ == '__main__':
    unittest.main()
