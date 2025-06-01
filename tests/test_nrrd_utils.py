import unittest
from unittest import mock
import os
import tempfile
import numpy as np
import nrrd
import nibabel as nib

# Adjust import path based on actual project structure
from data_io.nrrd_utils import read_nrrd_data, write_nrrd_data

class TestNrrdUtils(unittest.TestCase):

    def test_write_and_read_nrrd_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basic.nrrd")

            # Create dummy data and affine
            data_3d = np.random.rand(10, 12, 14).astype(np.float32)
            affine_3d = np.array([
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 12.0],
                [0.0, 0.0, 3.0, 14.0],
                [0.0, 0.0, 0.0, 1.0]
            ])

            header_opts = {'encoding': 'raw'} # Use raw for easier comparison if needed
            write_nrrd_data(filepath, data_3d, affine_3d, nrrd_header_options=header_opts)

            self.assertTrue(os.path.exists(filepath))

            read_data, read_affine, _, _, read_header = read_nrrd_data(filepath)

            self.assertIsNotNone(read_data)
            self.assertIsNotNone(read_affine)
            np.testing.assert_array_almost_equal(read_data, data_3d)
            np.testing.assert_array_almost_equal(read_affine, affine_3d, decimal=5)
            self.assertEqual(read_header['type'], data_3d.dtype.name)
            self.assertEqual(read_header['dimension'], 3)
            np.testing.assert_array_equal(read_header['sizes'], data_3d.shape)
            # Check space directions and origin (derived from affine)
            expected_space_dirs = [affine_3d[:3,0].tolist(), affine_3d[:3,1].tolist(), affine_3d[:3,2].tolist()]
            # self.assertEqual(read_header['space directions'], expected_space_dirs) # pynrrd might store as np.ndarray
            for i in range(3):
                 np.testing.assert_array_almost_equal(read_header['space directions'][i], expected_space_dirs[i], decimal=5)

            np.testing.assert_array_almost_equal(read_header['space origin'], affine_3d[:3,3], decimal=5)


    def test_write_and_read_nrrd_dwi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_dwi.nrrd")
            data_4d = np.random.rand(8, 8, 8, 5).astype(np.float32) # X,Y,Z,Gradients
            affine_4d = np.array([
                [-1.2, 0.0, 0.0, -50.0],
                [0.0, -1.2, 0.0, -40.0],
                [0.0, 0.0, 2.5, 30.0],
                [0.0, 0.0, 0.0, 1.0]
            ])
            bvals_dwi = np.array([0, 1000, 1000, 2000, 2000], dtype=float)
            bvecs_dwi = np.random.rand(5, 3)
            bvecs_dwi[0,:] = 0 # b0
            bvecs_dwi[1:,:] = bvecs_dwi[1:,:] / np.linalg.norm(bvecs_dwi[1:,:], axis=1, keepdims=True) # Normalize

            custom_meta = {"CustomField1": "Value1"}
            header_opts = {'encoding': 'gzip'}

            write_nrrd_data(filepath, data_4d, affine_4d,
                            bvals=bvals_dwi, bvecs=bvecs_dwi,
                            custom_fields=custom_meta, nrrd_header_options=header_opts)

            self.assertTrue(os.path.exists(filepath))

            read_data, read_affine, read_bvals, read_bvecs, read_header = read_nrrd_data(filepath)

            self.assertIsNotNone(read_data)
            np.testing.assert_array_almost_equal(read_data, data_4d)

            self.assertIsNotNone(read_affine)
            np.testing.assert_array_almost_equal(read_affine, affine_4d, decimal=5)

            self.assertIsNotNone(read_bvals)
            np.testing.assert_array_almost_equal(read_bvals, bvals_dwi)

            self.assertIsNotNone(read_bvecs)
            # B-vector reorientation in read_nrrd_data depends on affine and measurement frame.
            # If measurement frame is identity (as written by default), and affine is applied,
            # the read_bvecs should correspond to the original bvecs_dwi if they were in image space.
            # This test is simplified as write_nrrd_data assumes bvecs are in image space.
            # A more thorough test would involve a known non-identity measurement frame.
            self.assertEqual(read_bvecs.shape, bvecs_dwi.shape)
            # For now, just check if they are plausible (e.g. norms are 0 or 1)
            for i in range(read_bvecs.shape[0]):
                norm = np.linalg.norm(read_bvecs[i])
                self.assertTrue(np.isclose(norm, 0.0) or np.isclose(norm, 1.0))


            self.assertEqual(read_header.get('modality'), 'DWMRI')
            self.assertEqual(read_header.get('CustomField1'), 'Value1')
            # Check if some DWMRI_gradient fields exist (count might vary based on how many are written)
            self.assertTrue(any(k.startswith('DWMRI_gradient_') for k in read_header.keys()))

    def test_read_nrrd_dwi_reorientation_simplified(self):
        # This is a simplified test for b-vector reorientation.
        # A full test would require creating a NRRD with b-vectors in a known "patient" space
        # and a known "measurement frame" and "space directions", then verifying the output b-vectors.
        # Here, we simulate reading such a file by mocking the header.

        # Assume b-vectors are stored in NRRD relative to patient axes (e.g. from DICOM)
        bvecs_patient = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        # Image affine (NIfTI style), implies image axes are aligned with world RAS.
        # Let's say image X is Patient R, image Y is Patient A, image Z is Patient S.
        # Affine[:3,:3] is the rotation from image to patient.
        # inv(Affine[:3,:3]) is rotation from patient to image.
        # If NRRD 'space directions' correctly define image axes in patient space,
        # and 'measurement frame' is identity, then b-vectors from DWMRI_gradient_ fields
        # are assumed to be in patient space.

        # Create a dummy NRRD file structure for mocking nrrd.read
        dummy_data = np.zeros((2,2,2,2)) # 2 volumes for 2 bvecs
        dummy_header = {
            'type': 'float32',
            'dimension': 4,
            'space': 'right-anterior-superior', # RAS
            'sizes': np.array([2,2,2,2]),
            # Affine: image X = Pat X, image Y = Pat Y, image Z = Pat Z (simple RAS)
            'space directions': np.array([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]]), # Should make affine identity rotation
            'space origin': np.array([0.,0.,0.]),
            'modality': 'DWMRI',
            'DWMRI_gradient_0000': f"{bvecs_patient[0,0]} {bvecs_patient[0,1]} {bvecs_patient[0,2]}",
            'DWMRI_gradient_0001': f"{bvecs_patient[1,0]} {bvecs_patient[1,1]} {bvecs_patient[1,2]}",
            'DWMRI_b-value': '1000' # Dummy b-value
            # Measurement frame is not set, so assumed identity (b-vecs are in PCS of space directions)
        }

        with mock.patch('nrrd.read', return_value=(dummy_data, dummy_header)) as mock_nrrd_read:
            _, affine_calc, _, bvecs_img, _ = read_nrrd_data("dummy_dwi.nrrd")

            # With identity rotation part in affine, bvecs_img should be same as bvecs_patient
            # because A_rot_inv becomes identity.
            np.testing.assert_array_almost_equal(bvecs_img, bvecs_patient, decimal=5)


if __name__ == '__main__':
    unittest.main()
