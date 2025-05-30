import unittest
from unittest import mock
import sys
import os
import tempfile
import json
import yaml # Requires PyYAML
import argparse # For type comparison if needed, not for manual Namespace creation

from cli import run_noddi_fit 

class TestCliNoddiFit(unittest.TestCase):

    def _get_base_required_args_list(self, output_prefix="test_noddi_out/noddi_"):
        return [
            '--dwi', 'dummy_dwi.nii.gz',
            '--bval', 'dummy.bval',
            '--bvec', 'dummy.bvec',
            '--mask', 'dummy_mask.nii.gz', # Mask is required for NODDI CLI
            '--output_prefix', output_prefix
        ]

    @mock.patch('cli.run_noddi_fit.run_noddi_fitting')
    def test_noddi_fit_parsing_required_args(self, mock_run_noddi_fitting_func):
        test_argv_list = self._get_base_required_args_list()
        
        with mock.patch.object(sys, 'argv', ['run_noddi_fit.py'] + test_argv_list):
            run_noddi_fit.main()
        
        mock_run_noddi_fitting_func.assert_called_once()
        call_args_ns = mock_run_noddi_fitting_func.call_args[0][0]
        
        self.assertEqual(call_args_ns.dwi, 'dummy_dwi.nii.gz')
        self.assertEqual(call_args_ns.output_prefix, 'test_noddi_out/noddi_')
        self.assertIsNone(call_args_ns.config_file)
        self.assertEqual(call_args_ns.device, 'auto') # Default from add_device_arg
        self.assertIsNone(call_args_ns.b0_threshold_gtab) # Default for this arg in parser
        
        # Check flags indicating CLI presence
        self.assertFalse(call_args_ns.device_from_cli, 
                         "device_from_cli should be False as --device was not in test_argv_list")
        self.assertFalse(call_args_ns.b0_threshold_gtab_from_cli,
                         "b0_threshold_gtab_from_cli should be False as --b0_threshold_gtab was not in test_argv_list")

    @mock.patch('cli.run_noddi_fit.run_noddi_fitting')
    def test_noddi_fit_parsing_with_cli_options(self, mock_run_noddi_fitting_func):
        test_argv_list = self._get_base_required_args_list() + [
            '--device', 'cuda',
            '--b0_threshold_gtab', '30.0'
        ]
        with mock.patch.object(sys, 'argv', ['run_noddi_fit.py'] + test_argv_list):
            run_noddi_fit.main()
            
        mock_run_noddi_fitting_func.assert_called_once()
        call_args_ns = mock_run_noddi_fitting_func.call_args[0][0]
        self.assertEqual(call_args_ns.device, 'cuda')
        self.assertEqual(call_args_ns.b0_threshold_gtab, 30.0)
        self.assertTrue(call_args_ns.device_from_cli)
        self.assertTrue(call_args_ns.b0_threshold_gtab_from_cli)

    @mock.patch('cli.run_noddi_fit.run_noddi_fitting')
    def test_noddi_fit_config_file_path_parsed(self, mock_run_noddi_fitting_func):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
            json.dump({"device": "cpu"}, tmp_json) # Dummy content
            tmp_json_path = tmp_json.name
        
        test_argv_list = self._get_base_required_args_list() + ['--config_file', tmp_json_path]
        
        with mock.patch.object(sys, 'argv', ['run_noddi_fit.py'] + test_argv_list):
            run_noddi_fit.main()
        
        os.remove(tmp_json_path)
        
        mock_run_noddi_fitting_func.assert_called_once()
        call_args_ns = mock_run_noddi_fitting_func.call_args[0][0]
        self.assertEqual(call_args_ns.config_file, tmp_json_path)
        # CLI flags for device and b0_thresh were not set, so these should be false
        self.assertFalse(call_args_ns.device_from_cli)
        self.assertFalse(call_args_ns.b0_threshold_gtab_from_cli)

    @mock.patch('cli.run_noddi_fit.fit_noddi_volume') # Mock deeper to test run_noddi_fitting's internal logic
    @mock.patch('cli.cli_utils.load_nifti_data')
    @mock.patch('cli.cli_utils.load_bvals_bvecs')
    @mock.patch('cli.cli_utils.PyTorchGradientTable') # Mock classes from utils too
    @mock.patch('cli.cli_utils.determine_device')
    def test_noddi_fit_config_values_and_cli_override_logic(self, mock_determine_device, 
                                                           mock_gtab_class, mock_load_bvals_bvecs,
                                                           mock_load_nifti, mock_fit_noddi_volume_func):
        # This test checks if run_noddi_fitting correctly uses config values and CLI overrides
        # by inspecting what gets passed to fit_noddi_volume.
        
        # Mock return values for loaders
        mock_load_nifti.return_value = (np.zeros((2,2,2,5)), np.eye(4)) # dwi_data, affine
        mock_load_bvals_bvecs.return_value = (np.zeros(5), np.zeros((5,3))) # bvals, bvecs
        mock_gtab_instance = mock.Mock()
        mock_gtab_class.return_value = mock_gtab_instance
        mock_determine_device.return_value = torch.device('cpu') # Assume CPU for simplicity here

        config_data_from_file = {
            "device": "cpu", # From config
            "b0_threshold_gtab": 25.0, # From config
            "min_s0_val": 10.0,
            "fit_params": {"learning_rate": 0.005}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_yaml:
            yaml.dump(config_data_from_file, tmp_yaml)
            tmp_yaml_path = tmp_yaml.name

        # Scenario 1: Config only for device and b0_thresh_gtab
        args_ns_config_only = argparse.Namespace(
            dwi='d.nii', bval='b.bval', bvec='b.bvec', mask='m.nii', output_prefix='p_',
            config_file=tmp_yaml_path,
            device='auto', # Default from parser if not given, effectively CLI not set
            b0_threshold_gtab=None, # Default from parser if not given
            device_from_cli=False, # Simulating --device not on CLI
            b0_threshold_gtab_from_cli=False # Simulating --b0_threshold_gtab not on CLI
        )
        run_noddi_fit.run_noddi_fitting(args_ns_config_only)
        
        call_args_to_fit_noddi_vol = mock_fit_noddi_volume_func.call_args[1] # kwargs
        self.assertEqual(call_args_to_fit_noddi_vol['device'], torch.device('cpu')) # From config
        # Check gtab creation (b0_threshold comes from config)
        mock_gtab_class.assert_called_with(mock.ANY, mock.ANY, b0_threshold=25.0)
        self.assertEqual(call_args_to_fit_noddi_vol['min_s0_val'], 10.0) # From config
        self.assertEqual(call_args_to_fit_noddi_vol['fit_params']['learning_rate'], 0.005) # From config

        mock_fit_noddi_volume_func.reset_mock()
        mock_gtab_class.reset_mock()

        # Scenario 2: CLI overrides config for device and b0_thresh_gtab
        args_ns_cli_override = argparse.Namespace(
            dwi='d.nii', bval='b.bval', bvec='b.bvec', mask='m.nii', output_prefix='p_',
            config_file=tmp_yaml_path,
            device='cuda', # CLI override
            b0_threshold_gtab=77.0, # CLI override
            device_from_cli=True, # Simulating --device was on CLI
            b0_threshold_gtab_from_cli=True # Simulating --b0_threshold_gtab was on CLI
        )
        mock_determine_device.return_value = torch.device('cuda') # Assume determine_device gets 'cuda'
        run_noddi_fit.run_noddi_fitting(args_ns_cli_override)
        
        call_args_to_fit_noddi_vol_override = mock_fit_noddi_volume_func.call_args[1]
        self.assertEqual(call_args_to_fit_noddi_vol_override['device'], torch.device('cuda')) # From CLI
        mock_gtab_class.assert_called_with(mock.ANY, mock.ANY, b0_threshold=77.0) # From CLI
        self.assertEqual(call_args_to_fit_noddi_vol_override['min_s0_val'], 10.0) # Still from config
        
        os.remove(tmp_yaml_path)

    def test_main_parser_no_args_triggers_help_exit(self):
        with mock.patch.object(sys, 'argv', ['run_noddi_fit.py']):
            with self.assertRaises(SystemExit) as cm:
                run_noddi_fit.main()
            self.assertEqual(cm.exception.code, 1) # Custom exit in script

    def test_main_parser_missing_required_args(self):
        with mock.patch.object(sys, 'argv', ['run_noddi_fit.py', '--dwi', 'dwi.nii.gz']):
            with self.assertRaises(SystemExit) as cm:
                run_noddi_fit.main()
            self.assertEqual(cm.exception.code, 2) # Argparse error exit

    def test_main_parser_help_command(self):
        with mock.patch.object(sys, 'argv', ['run_noddi_fit.py', '-h']):
            with self.assertRaises(SystemExit) as cm:
                run_noddi_fit.main()
            self.assertEqual(cm.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
```
