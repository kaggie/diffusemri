import unittest
from unittest import mock
import argparse
import sys
import os # For os.path.join in anonymization test
import tempfile # For anonymization rules file test
import json # For writing dummy rules json

# Assuming 'cli' is a top-level directory or package accessible in PYTHONPATH.
from cli import run_preprocessing 

class TestCliPreprocessing(unittest.TestCase):

    @mock.patch('cli.run_preprocessing.create_brain_mask') # Mock the actual function called by run_masking
    def test_masking_subcommand_parsing(self, mock_create_brain_mask_func):
        # Mock load_nifti_data to avoid actual file loading
        with mock.patch('cli.run_preprocessing.load_nifti_data') as mock_load_nifti:
            mock_load_nifti.return_value = (mock.Mock(), mock.Mock()) # data, affine
            # Test with required args only
            run_preprocessing.main([
                'masking',
                '--dwi', 'input.nii.gz',
                '--output_mask', 'mask.nii.gz'
            ])
        mock_create_brain_mask_func.assert_called_once()
        # Check some args passed to create_brain_mask
        call_args_to_create_mask = mock_create_brain_mask_func.call_args[1] # kwargs
        self.assertEqual(call_args_to_create_mask['median_radius'], 4)
        self.assertEqual(call_args_to_create_mask['numpass'], 4)

        mock_create_brain_mask_func.reset_mock()
        mock_load_nifti.reset_mock()
        with mock.patch('cli.run_preprocessing.load_nifti_data') as mock_load_nifti:
            mock_load_nifti.return_value = (mock.Mock(), mock.Mock()) # data, affine
            # Test with all optional args
            run_preprocessing.main([
                'masking',
                '--dwi', 'in.nii', '--output_mask', 'out.nii',
                '--output_masked_dwi', 'masked.nii',
                '--median_radius', '3', '--numpass', '2'
            ])
        mock_create_brain_mask_func.assert_called_once()
        call_args_to_create_mask_all = mock_create_brain_mask_func.call_args[1]
        self.assertEqual(call_args_to_create_mask_all['median_radius'], 3)
        self.assertEqual(call_args_to_create_mask_all['numpass'], 2)


    @mock.patch('cli.run_preprocessing.denoise_mppca_data') # Mock the actual function
    def test_denoising_subcommand_parsing(self, mock_run_denoising_func):
        with mock.patch('cli.run_preprocessing.load_nifti_data') as mock_load_nifti:
            mock_load_nifti.return_value = (mock.Mock(), mock.Mock()) # data, affine
            # Test with explicit patch_radius
            run_preprocessing.main([
                'denoising_mppca',
                '--dwi', 'input.nii.gz',
                '--output_dwi', 'denoised.nii.gz',
                '--patch_radius', '3'
            ])
        mock_run_denoising_func.assert_called_once()
        call_args_to_denoise = mock_run_denoising_func.call_args[1] #kwargs
        self.assertEqual(call_args_to_denoise['patch_radius'], 3)

        mock_run_denoising_func.reset_mock()
        mock_load_nifti.reset_mock()
        with mock.patch('cli.run_preprocessing.load_nifti_data') as mock_load_nifti:
            mock_load_nifti.return_value = (mock.Mock(), mock.Mock())
            # Test default patch_radius
            run_preprocessing.main([
                'denoising_mppca',
                '--dwi', 'input2.nii.gz',
                '--output_dwi', 'denoised2.nii.gz'
            ])
        mock_run_denoising_func.assert_called_once()
        call_args_to_denoise_default = mock_run_denoising_func.call_args[1]
        self.assertEqual(call_args_to_denoise_default['patch_radius'], 2) # Check default value

    @mock.patch('cli.run_preprocessing.correct_motion_eddy_fsl') # Mock the actual function
    def test_correct_fsl_subcommand_parsing(self, mock_run_correct_fsl_func):
        base_args = [
            'correct_fsl',
            '--dwi', 'd.nii', '--bval', 'bval', '--bvec', 'bvec',
            '--mask', 'm.nii', '--index', 'idx.txt', '--acqp', 'acqp.txt',
            '--out_base', 'outb'
        ]
        # Test with some boolean flags and extra args
        run_preprocessing.main(base_args + ['--use_cuda', '--repol', '--fsl_eddy_extra_args', '--niter=8 --fwhm=1,0'])
        
        mock_run_correct_fsl_func.assert_called_once()
        call_args_to_eddy = mock_run_correct_fsl_func.call_args[1] # kwargs
        self.assertEqual(call_args_to_eddy['dwi_file'], 'd.nii')
        self.assertEqual(call_args_to_eddy['bval_file'], 'bval')
        # ... (check other mandatory args)
        self.assertTrue(call_args_to_eddy['use_cuda'])
        self.assertTrue(call_args_to_eddy['repol'])
        self.assertEqual(call_args_to_eddy['fsl_eddy_extra_args'], '--niter=8 --fwhm=1,0')


        mock_run_correct_fsl_func.reset_mock()
        # Test without optional flags to check defaults
        run_preprocessing.main(base_args)
        mock_run_correct_fsl_func.assert_called_once()
        call_args_defaults = mock_run_correct_fsl_func.call_args[1]
        self.assertFalse(call_args_defaults['use_cuda'])
        self.assertFalse(call_args_defaults['repol'])
        # self.assertEqual(call_args_defaults['fsl_eddy_extra_args'], "") # '' vs None might depend on argparse
        self.assertTrue('fsl_eddy_extra_args' not in call_args_defaults or call_args_defaults['fsl_eddy_extra_args'] == "")


    def test_main_parser_no_command(self):
    def test_main_parser_no_command(self):
        # Test that calling main with no command (sys.argv has only script name)
        # results in SystemExit because the script's main() calls parser.print_help() and sys.exit(1).
        with mock.patch.object(sys, 'argv', ['run_preprocessing.py']):
            with self.assertRaises(SystemExit) as cm:
                run_preprocessing.main()
            self.assertEqual(cm.exception.code, 1) 

    def test_main_parser_invalid_command(self):
        # Test that an invalid command causes argparse to exit.
        # Argparse prints to stderr and exits with code 2 for command line errors.
        with self.assertRaises(SystemExit) as cm:
            run_preprocessing.main(['invalid_command'])
        self.assertEqual(cm.exception.code, 2)

    def test_main_parser_help_command(self):
        # Test that 'help' command for a subparser exits (argparse behavior)
        # For example, 'masking -h' should print help for masking and exit.
        with self.assertRaises(SystemExit) as cm:
            run_preprocessing.main(['masking', '-h'])
        self.assertEqual(cm.exception.code, 0) # Argparse help exits with 0

    # --- Tests for DICOM to NIfTI CLI ---
    @mock.patch('cli.run_preprocessing.convert_dwi_dicom_to_nifti')
    @mock.patch('cli.run_preprocessing.convert_dicom_to_nifti_main')
    def test_dicom_to_nifti_subcommand(self, mock_convert_main, mock_convert_dwi):
        # Test DWI conversion
        run_preprocessing.main([
            'dicom_to_nifti',
            '--input_dicom_dir', '/fake/dwi_dicoms',
            '--output_nifti_file', 'dwi.nii.gz',
            '--is_dwi',
            '--output_bval_file', 'dwi.bval',
            '--output_bvec_file', 'dwi.bvec'
        ])
        mock_convert_dwi.assert_called_once_with(
            dicom_dir='/fake/dwi_dicoms',
            output_nifti_file='dwi.nii.gz',
            output_bval_file='dwi.bval',
            output_bvec_file='dwi.bvec'
        )
        mock_convert_main.assert_not_called()

        mock_convert_dwi.reset_mock()
        # Test non-DWI conversion
        run_preprocessing.main([
            'dicom_to_nifti',
            '--input_dicom_dir', '/fake/anat_dicoms',
            '--output_nifti_file', 'anat.nii.gz'
            # --is_dwi is not present, so should default to False or trigger non-DWI path
        ])
        # Check if is_dwi defaults to False implicitly by not calling convert_dwi_dicom_to_nifti
        # or by explicitly calling convert_dicom_to_nifti_main
        # The run_dicom_to_nifti function in CLI script uses args.is_dwi which is action='store_true'
        # So if --is_dwi is not present, args.is_dwi will be False.
        mock_convert_main.assert_called_once_with(
            dicom_dir='/fake/anat_dicoms',
            output_nifti_file='anat.nii.gz'
        )
        mock_convert_dwi.assert_not_called()


    # --- Tests for Anonymize DICOM CLI ---
    @mock.patch('cli.run_preprocessing.anonymize_dicom_directory')
    @mock.patch('cli.run_preprocessing.anonymize_dicom_file')
    @mock.patch('cli.run_preprocessing.load_config_from_json_yaml') # Mock rules loading
    @mock.patch('cli.run_preprocessing.os.path.isdir') # Mock os.path.isdir for auto-detection
    @mock.patch('cli.run_preprocessing.os.path.isfile') # Mock os.path.isfile
    def test_anonymize_dicom_subcommand(self, mock_isfile, mock_isdir, mock_load_rules, mock_anon_file, mock_anon_dir):
        # Test file anonymization
        mock_isdir.return_value = False
        mock_isfile.return_value = True
        run_preprocessing.main([
            'anonymize_dicom',
            '--input_path', '/fake/dicom.dcm',
            '--output_path', '/fake/anon_dicom.dcm'
        ])
        mock_anon_file.assert_called_once_with(
            input_dicom_path='/fake/dicom.dcm',
            output_dicom_path='/fake/anon_dicom.dcm',
            anonymization_rules=None # No rules file provided
        )
        mock_anon_dir.assert_not_called()
        mock_load_rules.assert_not_called()

        mock_anon_file.reset_mock()
        mock_isdir.reset_mock()
        mock_isfile.reset_mock()

        # Test directory anonymization with custom rules
        mock_isdir.return_value = True
        mock_isfile.return_value = False # Input is a dir
        dummy_rules = {"PatientName": "ANON"}
        mock_load_rules.return_value = dummy_rules

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_rules_file:
            json.dump(dummy_rules, tmp_rules_file)
            rules_file_path = tmp_rules_file.name

        run_preprocessing.main([
            'anonymize_dicom',
            '--input_path', '/fake/dicom_dir',
            '--output_path', '/fake/anon_dir',
            '--is_directory', # Explicitly state it's a directory
            '--rules_json', rules_file_path,
            '--no_preserve_structure'
        ])

        mock_load_rules.assert_called_once_with(rules_file_path)
        mock_anon_dir.assert_called_once_with(
            input_dir='/fake/dicom_dir',
            output_dir='/fake/anon_dir',
            anonymization_rules=dummy_rules,
            preserve_structure=False
        )
        mock_anon_file.assert_not_called()

        os.remove(rules_file_path)


if __name__ == '__main__':
    unittest.main()
```
