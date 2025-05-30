import unittest
from unittest import mock
import tempfile
import os
import json
import yaml # Requires PyYAML
import numpy as np
import torch
import nibabel as nib

# Adjust import path based on actual project structure and how tests are run.
# Assuming 'cli' is a top-level directory or package accessible in PYTHONPATH.
from cli import cli_utils 

class TestCliUtils(unittest.TestCase):

    def test_determine_device(self):
        self.assertEqual(cli_utils.determine_device('cpu'), torch.device('cpu'))
        
        with mock.patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(cli_utils.determine_device('cuda'), torch.device('cuda'))
            self.assertEqual(cli_utils.determine_device('auto'), torch.device('cuda'))
        
        with mock.patch('torch.cuda.is_available', return_value=False):
            # Test if warning is printed (optional, can be tricky to assert prints)
            # For now, just check the fallback behavior.
            with mock.patch('builtins.print') as mock_print:
                 self.assertEqual(cli_utils.determine_device('cuda'), torch.device('cpu'))
                 mock_print.assert_any_call("Warning: CUDA requested but not available. Falling back to CPU.")
            self.assertEqual(cli_utils.determine_device('auto'), torch.device('cpu'))

    def test_load_nifti_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.nii.gz")
            dummy_data_f64 = np.random.rand(5, 5, 5).astype(np.float64)
            dummy_affine_f64 = np.eye(4, dtype=np.float64)
            img = nib.Nifti1Image(dummy_data_f64, dummy_affine_f64)
            nib.save(img, filepath)

            # Test loading with ensure_float32=True
            data_f32, affine_f32 = cli_utils.load_nifti_data(filepath, ensure_float32=True)
            self.assertTrue(np.allclose(data_f32, dummy_data_f64.astype(np.float32)))
            self.assertTrue(np.allclose(affine_f32, dummy_affine_f64.astype(np.float32))) # Affine usually float64 by default from nib
            self.assertEqual(data_f32.dtype, np.float32)
            self.assertEqual(affine_f32.dtype, np.float64) # Nibabel affines are often float64

            # Test loading with ensure_float32=False
            data_orig_type, _ = cli_utils.load_nifti_data(filepath, ensure_float32=False)
            self.assertEqual(data_orig_type.dtype, np.float64)

            with self.assertRaises(FileNotFoundError):
                cli_utils.load_nifti_data(os.path.join(tmpdir, "nonexistent.nii.gz"))
    
    def test_load_bvals_bvecs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bval_path = os.path.join(tmpdir, "test.bval")
            bvec_path = os.path.join(tmpdir, "test.bvec")

            bvals_np = np.array([0, 1000, 1000])
            bvecs_standard_np = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float64)
            
            np.savetxt(bval_path, bvals_np, fmt='%d')
            np.savetxt(bvec_path, bvecs_standard_np, fmt='%.6f')
            
            bvals, bvecs = cli_utils.load_bvals_bvecs(bval_path, bvec_path)
            self.assertTrue(np.allclose(bvals, bvals_np))
            self.assertTrue(np.allclose(bvecs, bvecs_standard_np))

            # Test FSL style bvecs (3xN)
            bvecs_fsl_np = bvecs_standard_np.T
            np.savetxt(bvec_path, bvecs_fsl_np, fmt='%.6f')
            bvals_fsl, bvecs_fsl_loaded = cli_utils.load_bvals_bvecs(bval_path, bvec_path)
            self.assertTrue(np.allclose(bvecs_fsl_loaded, bvecs_standard_np))

            # Error cases
            with self.assertRaises(FileNotFoundError):
                cli_utils.load_bvals_bvecs("nonexistent.bval", bvec_path)
            with self.assertRaises(FileNotFoundError):
                cli_utils.load_bvals_bvecs(bval_path, "nonexistent.bvec")
            
            bvecs_wrong_shape_cols = np.array([[1,0],[0,1],[0,0]]) # Nx2
            np.savetxt(bvec_path, bvecs_wrong_shape_cols, fmt='%.6f')
            with self.assertRaisesRegex(ValueError, "not compatible"):
                cli_utils.load_bvals_bvecs(bval_path, bvec_path)

            bvecs_wrong_shape_rows = np.array([[1,0,0],[0,1,0]]) # Mismatch with bvals length (2 vs 3)
            np.savetxt(bvec_path, bvecs_wrong_shape_rows, fmt='%.6f')
            with self.assertRaisesRegex(ValueError, "not compatible"):
                cli_utils.load_bvals_bvecs(bval_path, bvec_path)


    def test_save_nifti_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "output_dir", "test_out.nii.gz") # Test subdirectory creation
            data_to_save = np.random.rand(5,5,5).astype(np.float32)
            affine_to_save = np.eye(4, dtype=np.float32)
            
            cli_utils.save_nifti_data(data_to_save, affine_to_save, filepath)
            self.assertTrue(os.path.exists(filepath))
            
            img_loaded = nib.load(filepath)
            self.assertTrue(np.allclose(img_loaded.get_fdata(), data_to_save))
            self.assertTrue(np.allclose(img_loaded.affine, affine_to_save))

    @mock.patch('cli.cli_utils.save_trk') # Mock Dipy's save_trk
    def test_save_tractogram(self, mock_dipy_save_trk):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "output_dir", "tracts.trk")
            streamlines_to_save = [np.random.rand(10,3).astype(np.float32), 
                                   np.random.rand(15,3).astype(np.float32)]
            affine_np = np.eye(4, dtype=np.float32)
            img_shape = (10,10,10) 

            cli_utils.save_tractogram(streamlines_to_save, affine_np, filepath, image_shape_for_sft=img_shape)
            
            mock_dipy_save_trk.assert_called_once()
            args, kwargs = mock_dipy_save_trk.call_args
            self.assertIsInstance(args[0], cli_utils.StatefulTractogram) # First arg is sft
            self.assertEqual(args[1], filepath) # Second arg is filepath
            self.assertFalse(kwargs.get('bbox_valid_check', True)) # Check if bbox_valid_check is False

            self.assertTrue(os.path.exists(os.path.dirname(filepath))) # Check dir creation

    def test_load_config_from_json_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test.json")
            yaml_path = os.path.join(tmpdir, "test.yaml")
            txt_path = os.path.join(tmpdir, "test.txt")

            json_data = {"key1": "val1", "num_param": 10}
            yaml_data = {"key2": "val2", "list_param": [1,2,3]}

            with open(json_path, 'w') as f: json.dump(json_data, f)
            with open(yaml_path, 'w') as f: yaml.dump(yaml_data, f)
            with open(txt_path, 'w') as f: f.write("text")

            config_json = cli_utils.load_config_from_json_yaml(json_path)
            self.assertEqual(config_json, json_data)

            config_yaml = cli_utils.load_config_from_json_yaml(yaml_path)
            self.assertEqual(config_yaml, yaml_data)

            with self.assertRaises(FileNotFoundError):
                cli_utils.load_config_from_json_yaml("nonexistent.json")
            with self.assertRaisesRegex(ValueError, "Unsupported configuration file format"):
                cli_utils.load_config_from_json_yaml(txt_path)
            
            malformed_yaml_path = os.path.join(tmpdir, "malformed.yaml")
            with open(malformed_yaml_path, 'w') as f: f.write("key: value\n  bad_indent: true") # Malformed YAML
            with self.assertRaisesRegex(ValueError, "Error parsing YAML file"):
                 cli_utils.load_config_from_json_yaml(malformed_yaml_path)

if __name__ == '__main__':
    unittest.main()
```
