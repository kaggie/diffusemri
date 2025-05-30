import unittest
from unittest import mock
import os
import tempfile
import json
import yaml # Requires PyYAML
import numpy as np
import argparse # For type comparison if needed

from cli import run_tracking # The script to test

class TestCliTracking(unittest.TestCase):

    def _get_base_required_args_list(self, output_trk="test_tracts.trk"):
        # Helper for minimal required args
        return [
            '--dwi', 'dwi.nii.gz',
            '--bval', 'bvals.txt',
            '--bvec', 'bvecs.txt',
            '--stopping_metric_map', 'fa.nii.gz',
            '--stopping_threshold', '0.2',
            '--seeds', 'seeds.nii.gz', 
            '--output_tracts', output_trk
        ]

    @mock.patch('cli.run_tracking.run_deterministic_tracking')
    def test_tracking_parsing_required_args(self, mock_run_tracking_func):
        test_argv_list = self._get_base_required_args_list()
        with mock.patch.object(sys, 'argv', ['run_tracking.py'] + test_argv_list):
            run_tracking.main()
        
        mock_run_tracking_func.assert_called_once()
        call_args_ns = mock_run_tracking_func.call_args[0][0]
        
        self.assertEqual(call_args_ns.dwi, 'dwi.nii.gz')
        self.assertEqual(call_args_ns.seeds, 'seeds.nii.gz')
        self.assertEqual(call_args_ns.output_tracts, 'test_tracts.trk')
        self.assertEqual(call_args_ns.stopping_threshold, 0.2)
        
        # Check defaults for optional params set by argparse in main()
        self.assertIsNone(call_args_ns.config_file)
        self.assertEqual(call_args_ns.device, 'auto') # Default from add_device_arg
        self.assertIsNone(call_args_ns.sh_order) 
        self.assertIsNone(call_args_ns.step_size)

    @mock.patch('cli.run_tracking.run_deterministic_tracking')
    def test_tracking_parsing_all_cli_params(self, mock_run_tracking_func):
        test_argv_list = self._get_base_required_args_list(output_trk="all_params.trk") + [
            '--affine', 'affine.txt', 
            '--device', 'cuda',
            '--sh_order', '4',
            '--model_max_peaks', '2',
            '--model_min_separation_angle', '30.0',
            '--model_peak_threshold', '0.4',
            '--step_size', '0.8',
            '--max_crossing_angle', '40.0',
            '--min_length', '5.0',
            '--max_length', '150.0',
            '--max_steps', '500'
        ]
        with mock.patch.object(sys, 'argv', ['run_tracking.py'] + test_argv_list):
            run_tracking.main()
            
        mock_run_tracking_func.assert_called_once()
        call_args_ns = mock_run_tracking_func.call_args[0][0]
        self.assertEqual(call_args_ns.affine, 'affine.txt')
        self.assertEqual(call_args_ns.device, 'cuda')
        self.assertEqual(call_args_ns.sh_order, 4)
        self.assertEqual(call_args_ns.model_max_peaks, 2)
        self.assertEqual(call_args_ns.model_min_separation_angle, 30.0)
        self.assertEqual(call_args_ns.model_peak_threshold, 0.4)
        self.assertEqual(call_args_ns.step_size, 0.8)
        self.assertEqual(call_args_ns.max_crossing_angle, 40.0)
        self.assertEqual(call_args_ns.min_length, 5.0)
        self.assertEqual(call_args_ns.max_length, 150.0)
        self.assertEqual(call_args_ns.max_steps, 500)

    @mock.patch('cli.run_tracking.run_deterministic_tracking')
    # We also mock load_config_from_json_yaml to control what config data run_deterministic_tracking sees
    @mock.patch('cli.cli_utils.load_config_from_json_yaml') 
    def test_tracking_config_file_and_cli_override_check(self, mock_load_config, mock_run_tracking_func):
        # This test verifies that CLI args are parsed correctly and that the config file path is passed.
        # The actual merging logic is inside run_deterministic_tracking, which is mocked.
        # We check that if CLI overrides are present, they are in the args namespace.
        
        mock_config_content = {"step_size": 0.2, "sh_order": 8, "device": "cpu"}
        mock_load_config.return_value = mock_config_content # Mock the loading process
        
        # Create a dummy temp file just so the path exists for argparse
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_config_file:
            yaml.dump(mock_config_content, tmp_config_file)
            tmp_config_path = tmp_config_file.name

        test_argv_list = self._get_base_required_args_list() + [
            '--config_file', tmp_config_path,
            '--step_size', '0.7', # CLI override for step_size
            '--device', 'cuda'     # CLI override for device
        ]
        with mock.patch.object(sys, 'argv', ['run_tracking.py'] + test_argv_list):
            run_tracking.main()
        
        os.remove(tmp_config_path)
        
        mock_run_tracking_func.assert_called_once()
        call_args_ns = mock_run_tracking_func.call_args[0][0]
        
        self.assertEqual(call_args_ns.config_file, tmp_config_path)
        self.assertEqual(call_args_ns.step_size, 0.7) # CLI override should be in Namespace
        self.assertIsNone(call_args_ns.sh_order)      # Not on CLI, so Namespace has None (config applied inside mocked func)
        self.assertEqual(call_args_ns.device, 'cuda') # CLI override should be in Namespace

    @mock.patch('cli.run_tracking.run_deterministic_tracking')
    @mock.patch('cli.run_tracking.parse_seeds') 
    def test_tracking_seed_argument_forwarding(self, mock_parse_seeds, mock_run_tracking_func):
        # We are testing that the --seeds string from CLI is passed to parse_seeds helper.
        # The actual parsing by parse_seeds is tested in test_cli_utils ideally,
        # or we trust its implementation for this CLI test.
        
        # Mock return of parse_seeds to be a simple NumPy array as expected by run_deterministic_tracking
        dummy_parsed_seeds_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
        mock_parse_seeds.return_value = dummy_parsed_seeds_np

        # Test with a file path for seeds
        seed_file_path = 'test_seed_mask.nii.gz'
        test_argv_mask_seeds = self._get_base_required_args_list()
        # Find and replace the default seed arg
        idx = test_argv_mask_seeds.index('--seeds')
        test_argv_mask_seeds[idx+1] = seed_file_path
        
        with mock.patch.object(sys, 'argv', ['run_tracking.py'] + test_argv_mask_seeds):
            run_tracking.main()
        
        # Check that parse_seeds was called with the file path
        # affine_np and device_str are also passed to parse_seeds from within run_deterministic_tracking
        mock_parse_seeds.assert_called_with(seed_file_path, mock.ANY) # mock.ANY for affine_np
        mock_run_tracking_func.assert_called_once()
        
        mock_run_tracking_func.reset_mock()
        mock_parse_seeds.reset_mock()
        mock_parse_seeds.return_value = dummy_parsed_seeds_np # Reset return value for next call

        # Test with a coordinate string for seeds
        seed_coord_str = "10,11,12;13,14,15"
        test_argv_coord_seeds = self._get_base_required_args_list()
        idx = test_argv_coord_seeds.index('--seeds')
        test_argv_coord_seeds[idx+1] = seed_coord_str

        with mock.patch.object(sys, 'argv', ['run_tracking.py'] + test_argv_coord_seeds):
            run_tracking.main()

        mock_parse_seeds.assert_called_with(seed_coord_str, mock.ANY)
        mock_run_tracking_func.assert_called_once()

    def test_main_parser_no_args_triggers_exit(self):
        with mock.patch.object(sys, 'argv', ['run_tracking.py']):
            with self.assertRaises(SystemExit) as cm:
                run_tracking.main()
            self.assertEqual(cm.exception.code, 1) # Custom exit in script's main

    def test_main_parser_missing_required_args_exit(self):
        # Example: Missing --bval, --bvec, etc.
        with mock.patch.object(sys, 'argv', ['run_tracking.py', '--dwi', 'dwi.nii.gz']):
            with self.assertRaises(SystemExit) as cm:
                run_tracking.main()
            self.assertEqual(cm.exception.code, 2) # Argparse error exit

    def test_main_parser_help_command_exit(self):
        with mock.patch.object(sys, 'argv', ['run_tracking.py', '-h']):
            with self.assertRaises(SystemExit) as cm:
                run_tracking.main()
            self.assertEqual(cm.exception.code, 0) # Argparse help exits with 0

if __name__ == '__main__':
    unittest.main()
```
