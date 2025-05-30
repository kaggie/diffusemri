import unittest
from unittest import mock
import argparse # For comparing Namespace if needed, not strictly for creating
import sys

# Assuming 'cli' is a top-level directory or package accessible in PYTHONPATH.
from cli import run_preprocessing 

class TestCliPreprocessing(unittest.TestCase):

    @mock.patch('cli.run_preprocessing.run_masking')
    def test_masking_subcommand_parsing(self, mock_run_masking_func):
        # Test with required args only
        run_preprocessing.main([
            'masking', 
            '--dwi', 'input.nii.gz', 
            '--output_mask', 'mask.nii.gz'
        ])
        mock_run_masking_func.assert_called_once()
        call_args = mock_run_masking_func.call_args[0][0]
        self.assertEqual(call_args.dwi, 'input.nii.gz')
        self.assertEqual(call_args.output_mask, 'mask.nii.gz')
        self.assertEqual(call_args.median_radius, 4) # Default
        self.assertEqual(call_args.numpass, 4) # Default
        self.assertIsNone(call_args.output_masked_dwi) # Default

        mock_run_masking_func.reset_mock()
        # Test with all optional args
        run_preprocessing.main([
            'masking', 
            '--dwi', 'in.nii', '--output_mask', 'out.nii',
            '--output_masked_dwi', 'masked.nii',
            '--median_radius', '3', '--numpass', '2'
        ])
        mock_run_masking_func.assert_called_once()
        call_args_all = mock_run_masking_func.call_args[0][0]
        self.assertEqual(call_args_all.dwi, 'in.nii')
        self.assertEqual(call_args_all.output_mask, 'out.nii')
        self.assertEqual(call_args_all.output_masked_dwi, 'masked.nii')
        self.assertEqual(call_args_all.median_radius, 3)
        self.assertEqual(call_args_all.numpass, 2)

    @mock.patch('cli.run_preprocessing.run_denoising_mppca')
    def test_denoising_subcommand_parsing(self, mock_run_denoising_func):
        # Test with explicit patch_radius
        run_preprocessing.main([
            'denoising_mppca',
            '--dwi', 'input.nii.gz',
            '--output_dwi', 'denoised.nii.gz',
            '--patch_radius', '3' 
        ])
        mock_run_denoising_func.assert_called_once()
        call_args = mock_run_denoising_func.call_args[0][0]
        self.assertEqual(call_args.dwi, 'input.nii.gz')
        self.assertEqual(call_args.output_dwi, 'denoised.nii.gz')
        self.assertEqual(call_args.patch_radius, 3)

        mock_run_denoising_func.reset_mock()
        # Test default patch_radius
        run_preprocessing.main([
            'denoising_mppca',
            '--dwi', 'input2.nii.gz',
            '--output_dwi', 'denoised2.nii.gz'
        ])
        mock_run_denoising_func.assert_called_once()
        call_args_default = mock_run_denoising_func.call_args[0][0]
        self.assertEqual(call_args_default.dwi, 'input2.nii.gz')
        self.assertEqual(call_args_default.patch_radius, 2) # Check default value

    @mock.patch('cli.run_preprocessing.run_correct_fsl')
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
        call_args = mock_run_correct_fsl_func.call_args[0][0]
        self.assertEqual(call_args.dwi, 'd.nii')
        self.assertEqual(call_args.bval, 'bval')
        self.assertEqual(call_args.bvec, 'bvec')
        self.assertEqual(call_args.mask, 'm.nii')
        self.assertEqual(call_args.index, 'idx.txt')
        self.assertEqual(call_args.acqp, 'acqp.txt')
        self.assertEqual(call_args.out_base, 'outb')
        self.assertTrue(call_args.use_cuda)
        self.assertTrue(call_args.repol)
        self.assertFalse(call_args.cnr_maps) # Default is False
        self.assertFalse(call_args.residuals) # Default is False
        self.assertEqual(call_args.fsl_eddy_extra_args, '--niter=8 --fwhm=1,0')

        mock_run_correct_fsl_func.reset_mock()
        # Test without optional flags to check defaults
        run_preprocessing.main(base_args)
        mock_run_correct_fsl_func.assert_called_once()
        call_args_defaults = mock_run_correct_fsl_func.call_args[0][0]
        self.assertFalse(call_args_defaults.use_cuda)
        self.assertFalse(call_args_defaults.repol)
        self.assertEqual(call_args_defaults.fsl_eddy_extra_args, "")


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

if __name__ == '__main__':
    unittest.main()
```
