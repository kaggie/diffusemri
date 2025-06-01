import unittest
from unittest import mock
import os
import tempfile
import json
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
import nibabel as nib
from typing import List, Dict, Any # For type hints

# Assuming dicom_utils is in data_io, and tests is a top-level or correctly configured path
from data_io.dicom_utils import (
    read_dicom_series, extract_pixel_data_and_affine, convert_dicom_to_nifti_main,
    extract_dwi_metadata, convert_dwi_dicom_to_nifti,
    anonymize_dicom_dataset, anonymize_dicom_file, anonymize_dicom_directory,
    DEFAULT_ANONYMIZATION_TAGS, _REMOVE_TAG_, _EMPTY_STRING_, _ZERO_STRING_, _DEFAULT_DATE_, _DEFAULT_TIME_,
    write_nifti_to_dicom_secondary # Added import
)
from pydicom import uid as pydicom_uid # Added import for SOP Class UID

# Helper to create a dummy DICOM dataset
def create_dummy_dicom_dataset(filename="test.dcm", instance_number=1, series_uid="1.2.3",
                               rows=64, cols=64, pixel_spacing=[1.0, 1.0],
                               image_orientation=[1,0,0,0,1,0], slice_thickness=5.0,
                               image_position=[0,0,0], acquisition_number=1,
                               dwi_specific_tags: Dict[str, Any] = None):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4' # MR Image Storage
    ds.file_meta.MediaStorageSOPInstanceUID = f"1.2.3.4.5.{instance_number}"
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds.InstanceNumber = instance_number
    ds.SeriesInstanceUID = series_uid
    ds.Rows = rows
    ds.Columns = cols
    ds.PixelSpacing = pixel_spacing
    ds.ImageOrientationPatient = image_orientation
    ds.SliceThickness = slice_thickness
    ds.ImagePositionPatient = image_position
    ds.AcquisitionNumber = acquisition_number

    # Add dummy pixel data (e.g., uint16)
    ds.PixelData = np.random.randint(0, 256, size=(rows, cols), dtype=np.uint16).tobytes()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0 # Unsigned

    # Add some basic metadata tags
    ds.RepetitionTime = 1000.0
    ds.EchoTime = 100.0
    ds.FlipAngle = 90.0
    ds.Manufacturer = "TestCorp"

    if dwi_specific_tags:
        for tag_name, value in dwi_specific_tags.items():
            # Pydicom tag names can be keywords or (group, element) tuples
            # For simplicity, this helper assumes tag_name is a string keyword.
            # For actual tags like (0x0018, 0x9087), you'd use ds[tag_tuple] = ...
            if tag_name == "DiffusionBValue": # (0018,9087)
                setattr(ds, tag_name, value)
                # ds[0x0018, 0x9087] = pydicom.DataElement((0x0018,0x9087), 'FD', value) # More robust
            elif tag_name == "DiffusionGradientOrientation": # (0018,0089)
                setattr(ds, tag_name, value)
                # ds[0x0018, 0x9089] = pydicom.DataElement((0x0018,0x9089), 'FD', value)
            else:
                setattr(ds, tag_name, value)

    ds.filename = filename # pydicom uses this attribute
    return ds

class TestDicomUtils(unittest.TestCase):

    @mock.patch('data_io.dicom_utils.os.walk')
    @mock.patch('data_io.dicom_utils.pydicom.dcmread')
    def test_read_dicom_series(self, mock_dcmread, mock_os_walk):
        # Setup mock directory structure and files
        mock_os_walk.return_value = [
            ('/fake/dicom_dir', [], ['slice1.dcm', 'slice2.dcm', 'nodcm.txt']),
        ]

        ds1 = create_dummy_dicom_dataset(filename='slice1.dcm', instance_number=1)
        ds2 = create_dummy_dicom_dataset(filename='slice2.dcm', instance_number=2)

        # Define side effects for pydicom.dcmread
        def dcmread_side_effect(filepath, force=False):
            if filepath.endswith('slice1.dcm'):
                return ds1
            elif filepath.endswith('slice2.dcm'):
                return ds2
            elif filepath.endswith('nodcm.txt'):
                raise InvalidDicomError("Not a DICOM")
            return None
        mock_dcmread.side_effect = dcmread_side_effect

        datasets = read_dicom_series('/fake/dicom_dir')
        self.assertEqual(len(datasets), 2)
        self.assertEqual(datasets[0].InstanceNumber, 1)
        self.assertEqual(datasets[1].InstanceNumber, 2)

        # Test sorting by filename if InstanceNumber is missing
        ds1_no_inst = create_dummy_dicom_dataset(filename='b_slice.dcm')
        del ds1_no_inst.InstanceNumber
        ds2_no_inst = create_dummy_dicom_dataset(filename='a_slice.dcm')
        del ds2_no_inst.InstanceNumber

        mock_os_walk.return_value = [('/fake/dicom_dir2', [], ['b_slice.dcm', 'a_slice.dcm'])]
        def dcmread_side_effect_no_inst(filepath, force=False):
            if filepath.endswith('b_slice.dcm'): return ds1_no_inst
            if filepath.endswith('a_slice.dcm'): return ds2_no_inst
            return None
        mock_dcmread.side_effect = dcmread_side_effect_no_inst

        datasets_sorted_by_name = read_dicom_series('/fake/dicom_dir2')
        self.assertEqual(len(datasets_sorted_by_name), 2)
        # Check if sorted by filename (a_slice.dcm then b_slice.dcm)
        self.assertTrue(datasets_sorted_by_name[0].filename.endswith('a_slice.dcm'))
        self.assertTrue(datasets_sorted_by_name[1].filename.endswith('b_slice.dcm'))


    def test_extract_pixel_data_and_affine(self):
        ds1 = create_dummy_dicom_dataset(instance_number=1, image_position=[0,0,0], slice_thickness=5)
        ds2 = create_dummy_dicom_dataset(instance_number=2, image_position=[0,0,5], slice_thickness=5) # Assuming Z changes by slice_thickness
        sorted_dicoms = [ds1, ds2]

        image_data, affine, metadata = extract_pixel_data_and_affine(sorted_dicoms)

        self.assertIsNotNone(image_data)
        self.assertEqual(image_data.shape, (ds1.Rows, ds1.Columns, 2))
        self.assertEqual(image_data.dtype, ds1.pixel_array.dtype)

        self.assertIsNotNone(affine)
        self.assertEqual(affine.shape, (4,4))
        # Basic affine checks - more detailed checks would require known ground truth
        self.assertAlmostEqual(affine[0,0], ds1.PixelSpacing[1]) # ColSpacing
        self.assertAlmostEqual(affine[1,1], ds1.PixelSpacing[0]) # RowSpacing
        self.assertAlmostEqual(affine[2,2], ds1.SliceThickness) # Approx slice spacing along Z
        self.assertTrue(np.allclose(affine[:3,3], ds1.ImagePositionPatient))

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['TR'], 1000.0)
        self.assertEqual(metadata['Manufacturer'], "TestCorp")

        # Test with NumberOfTemporalPositions for 4D data
        ds1_tp = create_dummy_dicom_dataset(instance_number=1, image_position=[0,0,0], slice_thickness=5)
        ds2_tp = create_dummy_dicom_dataset(instance_number=2, image_position=[0,0,0], slice_thickness=5) # Same position, different time
        ds1_tp.NumberOfTemporalPositions = 2
        ds2_tp.NumberOfTemporalPositions = 2
        # For real 4D, other tags like TriggerTime or AcquisitionTime might vary per volume.
        # InstanceNumber might be per volume, not per slice in time.
        # This test is simplified. Assume slices are ordered slice1_t1, slice2_t1, slice1_t2, slice2_t2
        # Or, more commonly, all slices for t1, then all for t2.
        # The current extract function assumes slices are stacked then reshaped.
        # Let's simulate 2 slices, 2 timepoints.
        slices_t1 = [create_dummy_dicom_dataset(instance_number=i+1, slice_thickness=5, image_position=[0,0,i*5]) for i in range(2)]
        slices_t2 = [create_dummy_dicom_dataset(instance_number=i+3, slice_thickness=5, image_position=[0,0,i*5]) for i in range(2)] # InstNum for t2 follows t1
        for ds_ in slices_t1 + slices_t2: ds_.NumberOfTemporalPositions = 2

        all_slices_for_4d = slices_t1 + slices_t2
        # Sort them by instance number to simulate how read_dicom_series would provide them
        all_slices_for_4d.sort(key=lambda ds: ds.InstanceNumber)

        image_data_4d, _, _ = extract_pixel_data_and_affine(all_slices_for_4d)
        self.assertIsNotNone(image_data_4d)
        self.assertEqual(image_data_4d.shape, (ds1.Rows, ds1.Columns, 2, 2)) # rows, cols, slices_per_vol, num_vols


    @mock.patch('data_io.dicom_utils.read_dicom_series')
    @mock.patch('data_io.dicom_utils.extract_pixel_data_and_affine')
    @mock.patch('data_io.dicom_utils.nib.save')
    @mock.patch('data_io.dicom_utils.json.dump') # Mock json.dump to avoid actual file writing
    def test_convert_dicom_to_nifti_main(self, mock_json_dump, mock_nib_save,
                                         mock_extract_data, mock_read_series):
        # Setup mocks
        mock_read_series.return_value = [create_dummy_dicom_dataset()] # Non-empty list
        dummy_image_data = np.random.rand(64, 64, 10).astype(np.float32)
        dummy_affine = np.eye(4)
        dummy_metadata = {"TR": 1000}
        mock_extract_data.return_value = (dummy_image_data, dummy_affine, dummy_metadata)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_nifti_file = os.path.join(tmpdir, "output.nii.gz")
            success = convert_dicom_to_nifti_main("/fake/dicom_dir", output_nifti_file)

            self.assertTrue(success)
            mock_read_series.assert_called_once_with("/fake/dicom_dir")
            mock_extract_data.assert_called_once_with(mock_read_series.return_value)
            mock_nib_save.assert_called_once()

            # Check NIfTI image data and affine passed to nib.save
            saved_nifti_image = mock_nib_save.call_args[0][0]
            # Check transpose (cols, rows, slices)
            expected_transposed_shape = (dummy_image_data.shape[1], dummy_image_data.shape[0], dummy_image_data.shape[2])
            np.testing.assert_array_equal(saved_nifti_image.get_fdata().shape, expected_transposed_shape)
            np.testing.assert_array_equal(saved_nifti_image.affine, dummy_affine)

            # Check JSON sidecar call
            mock_json_dump.assert_called_once()
            json_sidecar_path = output_nifti_file.replace(".nii.gz", ".json")
            # Check if the call was with the correct path (depends on how open is mocked or handled)
            # For now, check the data passed to json.dump
            self.assertEqual(mock_json_dump.call_args[0][0], dummy_metadata)

    # --- DWI Specific Tests ---
    def test_extract_dwi_metadata(self):
        # Standard b-value and b-vector tags
        dwi_tags_vol1 = {"DiffusionBValue": 0.0, "DiffusionGradientOrientation": [0.0, 0.0, 0.0]}
        dwi_tags_vol2 = {"DiffusionBValue": 1000.0, "DiffusionGradientOrientation": [1.0, 0.0, 0.0]}
        dwi_tags_vol3 = {"DiffusionBValue": 1000.0, "DiffusionGradientOrientation": [0.0, 1.0, 0.0]}

        ds1 = create_dummy_dicom_dataset(instance_number=1, dwi_specific_tags=dwi_tags_vol1)
        ds2 = create_dummy_dicom_dataset(instance_number=2, dwi_specific_tags=dwi_tags_vol2)
        ds3 = create_dummy_dicom_dataset(instance_number=3, dwi_specific_tags=dwi_tags_vol3)

        sorted_dwi_dicoms = [ds1, ds2, ds3]
        # Assume standard ImageOrientationPatient for simplicity in this test of extraction
        # Reorientation complexity is high; here we mainly test if vectors are extracted and shaped correctly.
        ref_iop = [1.0,0.0,0.0,0.0,1.0,0.0]

        b_values, b_vectors, dwi_meta = extract_dwi_metadata(sorted_dwi_dicoms, ref_iop)

        self.assertIsNotNone(b_values)
        self.assertIsNotNone(b_vectors)
        self.assertIsInstance(dwi_meta, dict)

        expected_b_values = np.array([0.0, 1000.0, 1000.0])
        # B-vectors reorientation depends on the matrix derived from ref_iop.
        # If ref_iop is identity [1,0,0,0,1,0], then rotation_matrix_pat_to_img is identity.
        # So, b_vectors should be approximately the same as input if they were already normalized.
        expected_b_vectors_if_identity_iop = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], # Assuming normalized after extraction if needed
            [0.0, 1.0, 0.0]  # Assuming normalized
        ])

        np.testing.assert_array_almost_equal(b_values, expected_b_values)
        self.assertEqual(b_vectors.shape, (3, 3))
        # For this test, since ref_iop is identity, reoriented should be same as raw (if raw normalized)
        np.testing.assert_array_almost_equal(b_vectors, expected_b_vectors_if_identity_iop, decimal=5)


    @mock.patch('data_io.dicom_utils.read_dicom_series')
    @mock.patch('data_io.dicom_utils.extract_pixel_data_and_affine')
    @mock.patch('data_io.dicom_utils.extract_dwi_metadata')
    @mock.patch('data_io.dicom_utils.nib.save')
    @mock.patch('data_io.dicom_utils.np.savetxt') # Mock saving bvals/bvecs
    @mock.patch('data_io.dicom_utils.json.dump')
    def test_convert_dwi_dicom_to_nifti(self, mock_json_dump, mock_np_savetxt,
                                        mock_nib_save, mock_extract_dwi_meta,
                                        mock_extract_pix_affine, mock_read_series):
        # Setup mocks
        dcm1 = create_dummy_dicom_dataset(instance_number=1, dwi_specific_tags={"DiffusionBValue": 0.0})
        dcm2 = create_dummy_dicom_dataset(instance_number=2, dwi_specific_tags={"DiffusionBValue": 1000.0, "DiffusionGradientOrientation": [1,0,0]})
        mock_read_series.return_value = [dcm1, dcm2]

        dummy_image_data = np.random.rand(64, 64, 2).astype(np.float32) # Rows, Cols, NumVols
        dummy_affine = np.eye(4)
        dummy_basic_meta = {"Manufacturer": "TestCorp"}
        mock_extract_pix_affine.return_value = (dummy_image_data, dummy_affine, dummy_basic_meta)

        dummy_bvals = np.array([0.0, 1000.0])
        dummy_bvecs = np.array([[0,0,0], [1,0,0]], dtype=float)
        dummy_dwi_meta = {"NumberOfDiffusionDirections": 1}
        mock_extract_dwi_meta.return_value = (dummy_bvals, dummy_bvecs, dummy_dwi_meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            nifti_file = os.path.join(tmpdir, "dwi.nii.gz")
            bval_file = os.path.join(tmpdir, "dwi.bval")
            bvec_file = os.path.join(tmpdir, "dwi.bvec")

            success = convert_dwi_dicom_to_nifti(
                "/fake_dwi_dir", nifti_file, bval_file, bvec_file
            )
            self.assertTrue(success)
            mock_read_series.assert_called_once_with("/fake_dwi_dir")
            mock_extract_pix_affine.assert_called_once_with([dcm1, dcm2])
            mock_extract_dwi_meta.assert_called_once_with([dcm1, dcm2], mock.ANY) # mock.ANY for ref_iop

            mock_nib_save.assert_called_once()
            # Check NIfTI data transpose: (cols, rows, num_vols)
            saved_nifti_image = mock_nib_save.call_args[0][0]
            expected_nifti_data_shape = (dummy_image_data.shape[1], dummy_image_data.shape[0], dummy_image_data.shape[2])
            self.assertEqual(saved_nifti_image.get_fdata().shape, expected_nifti_data_shape)

            # Check bval/bvec saves
            # np.savetxt is called twice (once for bvals, once for bvecs)
            self.assertEqual(mock_np_savetxt.call_count, 2)
            # First call should be bvals
            np.testing.assert_array_equal(mock_np_savetxt.call_args_list[0][0][1], dummy_bvals.reshape(1,-1))
            self.assertEqual(mock_np_savetxt.call_args_list[0][0][0], bval_file)
            # Second call should be bvecs
            np.testing.assert_array_equal(mock_np_savetxt.call_args_list[1][0][1], dummy_bvecs)
            self.assertEqual(mock_np_savetxt.call_args_list[1][0][0], bvec_file)

            # Check JSON sidecar for combined metadata
            mock_json_dump.assert_called_once()
            expected_combined_meta = {**dummy_basic_meta, **dummy_dwi_meta}
            self.assertEqual(mock_json_dump.call_args[0][0], expected_combined_meta)

    # --- Anonymization Tests ---
    def test_anonymize_dicom_dataset_default_rules(self):
        ds = create_dummy_dicom_dataset()
        # Add more PII tags that are in DEFAULT_ANONYMIZATION_TAGS
        ds.PatientSex = "O"
        ds.PatientAge = "030Y"
        ds.InstitutionName = "Old Hospital"
        ds.ReferringPhysicianName = "Dr. Original"
        ds.SeriesDescription = "Important Series" # Not typically anonymized by default PII
        ds.DiffusionBValue = 1000 # Essential tag

        anonymize_dicom_dataset(ds) # Uses DEFAULT_ANONYMIZATION_TAGS

        self.assertEqual(ds.PatientName, "") # Assuming _EMPTY_STRING_ maps to ""
        self.assertEqual(ds.PatientID, "0") # Assuming _ZERO_STRING_ maps to "0"
        self.assertEqual(ds.PatientBirthDate, "19000101") # Assuming _DEFAULT_DATE_
        self.assertEqual(ds.PatientSex, "")
        self.assertEqual(ds.PatientAge, "")
        self.assertEqual(ds.InstitutionName, "")
        self.assertEqual(ds.ReferringPhysicianName, "")
        self.assertTrue('PatientAddress' not in ds) # Assuming _REMOVE_TAG_

        # Check essential tags are preserved
        self.assertEqual(ds.SeriesDescription, "Important Series")
        self.assertEqual(ds.DiffusionBValue, 1000)
        self.assertTrue('PixelData' in ds) # Pixel data should always be there

    def test_anonymize_dicom_dataset_custom_rules(self):
        ds = create_dummy_dicom_dataset()
        ds.PatientName = "Original Name"
        ds.StudyDescription = "Original Study"
        ds.add_new(0x00189087, 'FD', 1000.0) # DiffusionBValue

        custom_rules = {
            'PatientName': 'ANONYMIZED',
            (0x0008, 0x1030): _REMOVE_TAG_, # StudyDescription
            'DiffusionBValue': 500 # Change an essential tag (for testing custom rules only)
        }
        anonymize_dicom_dataset(ds, anonymization_rules=custom_rules)

        self.assertEqual(ds.PatientName, 'ANONYMIZED')
        self.assertTrue('StudyDescription' not in ds)
        self.assertEqual(ds.DiffusionBValue, 500) # Check if essential tag was changed by custom rule

    def test_anonymize_dicom_dataset_custom_replacements_function(self):
        ds = create_dummy_dicom_dataset()
        ds.PatientAge = "030Y"

        def age_replacer(original_age):
            if original_age and isinstance(original_age, str) and original_age.endswith('Y'):
                return f"A{original_age[:-1]}" # e.g. A030
            return "A000"

        custom_functions = {'PatientAge': age_replacer}
        anonymize_dicom_dataset(ds, custom_replacements=custom_functions) # Uses default rules + custom function

        self.assertEqual(ds.PatientAge, "A030")


    @mock.patch('data_io.dicom_utils.pydicom.dcmread')
    @mock.patch('data_io.dicom_utils.anonymize_dicom_dataset') # Mock the core dataset anonymizer
    @mock.patch('pydicom.dataset.FileDataset.save_as') # Mock save_as directly on FileDataset
    @mock.patch('data_io.dicom_utils.os.makedirs') # Mock makedirs
    def test_anonymize_dicom_file(self, mock_makedirs, mock_save_as, mock_anonymize_ds, mock_dcmread):
        dummy_ds = create_dummy_dicom_dataset()
        mock_dcmread.return_value = dummy_ds

        input_path = "/fake/input/test.dcm"
        output_path = "/fake/output/test_anon.dcm"

        with mock.patch('data_io.dicom_utils.os.path.exists', return_value=True): # Assume input file exists
            success = anonymize_dicom_file(input_path, output_path)

        self.assertTrue(success)
        mock_dcmread.assert_called_once_with(input_path)
        mock_anonymize_ds.assert_called_once_with(dummy_ds, None, None)
        mock_save_as.assert_called_once_with(output_path)
        # Check if makedirs was called if output_dir was present (it is in output_path)
        mock_makedirs.assert_called_with(os.path.dirname(output_path), exist_ok=True)


    @mock.patch('data_io.dicom_utils.os.walk')
    @mock.patch('data_io.dicom_utils.pydicom.dcmread', side_effect=lambda x, stop_before_pixels=False: create_dummy_dicom_dataset(filename=os.path.basename(x))) # Simulate reading
    @mock.patch('data_io.dicom_utils.anonymize_dicom_file')
    def test_anonymize_dicom_directory(self, mock_anonymize_file, mock_dcmread_for_walk, mock_os_walk):
        mock_os_walk.return_value = [
            ('/fake/in_dir', ['subdir'], ['file1.dcm', 'not_dcm.txt']),
            ('/fake/in_dir/subdir', [], ['file2.dcm']),
        ]
        mock_anonymize_file.return_value = True # Assume anonymization of each file succeeds

        with tempfile.TemporaryDirectory() as tmp_out_dir:
            processed, failed = anonymize_dicom_directory("/fake/in_dir", tmp_out_dir, preserve_structure=True)

            self.assertEqual(processed, 2) # file1.dcm, file2.dcm
            self.assertEqual(failed, 0)

            expected_calls = [
                mock.call("/fake/in_dir/file1.dcm", os.path.join(tmp_out_dir, "file1.dcm"), None, None),
                mock.call("/fake/in_dir/subdir/file2.dcm", os.path.join(tmp_out_dir, "subdir", "file2.dcm"), None, None),
            ]
            mock_anonymize_file.assert_has_calls(expected_calls, any_order=True)
            self.assertEqual(mock_anonymize_file.call_count, 2)

            # Test without preserving structure
            mock_anonymize_file.reset_mock()
            processed, failed = anonymize_dicom_directory("/fake/in_dir", tmp_out_dir, preserve_structure=False)
            self.assertEqual(processed, 2)
            self.assertEqual(failed, 0)
            expected_calls_no_struct = [
                mock.call("/fake/in_dir/file1.dcm", os.path.join(tmp_out_dir, "file1.dcm"), None, None),
                mock.call("/fake/in_dir/subdir/file2.dcm", os.path.join(tmp_out_dir, "file2.dcm"), None, None),
            ]
            mock_anonymize_file.assert_has_calls(expected_calls_no_struct, any_order=True)


if __name__ == '__main__':
    unittest.main()
