import unittest
from unittest import mock
import sys
import argparse # For type hinting expected args if needed, not for creating Namespace manually

# Assuming 'cli' is a top-level directory or package accessible in PYTHONPATH.
from cli import run_dti_fit 

class TestCliDtiFit(unittest.TestCase):

    @mock.patch('cli.run_dti_fit.run_dti_fitting')
    def test_dti_fit_parsing_required_args(self, mock_run_dti_fitting_func):
        test_argv = [
            '--dwi', 'dwi.nii.gz',
            '--bval', 'bvals.txt',
            '--bvec', 'bvecs.txt',
            '--output_prefix', 'out/dti_'
        ]
        # Simulate how main() in run_dti_fit.py would be called with these args
        # by directly calling the parser logic if main is not designed for list input,
        # or by patching sys.argv.
        # The run_dti_fit.py's main() calls parser.parse_args() which defaults to sys.argv[1:].
        # So, we need to mock sys.argv for these tests.
        
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py'] + test_argv):
            try:
                run_dti_fit.main()
            except SystemExit: # Catch sys.exit if main calls it, though mock should prevent full run
                pass 
                # This might happen if args.func is not found by a test runner 
                # but the mock should ensure run_dti_fitting is called before any exit.

        mock_run_dti_fitting_func.assert_called_once()
        call_args = mock_run_dti_fitting_func.call_args[0][0]
        
        self.assertEqual(call_args.dwi, 'dwi.nii.gz')
        self.assertEqual(call_args.bval, 'bvals.txt')
        self.assertEqual(call_args.bvec, 'bvecs.txt')
        self.assertEqual(call_args.output_prefix, 'out/dti_')
        # Check defaults for other args
        self.assertIsNone(call_args.mask)
        self.assertEqual(call_args.b0_threshold, 50.0)
        self.assertEqual(call_args.min_s0_threshold, 1.0)

    @mock.patch('cli.run_dti_fit.run_dti_fitting')
    def test_dti_fit_parsing_all_args(self, mock_run_dti_fitting_func):
        test_argv = [
            '--dwi', 'dwi.nii.gz',
            '--bval', 'bvals.txt',
            '--bvec', 'bvecs.txt',
            '--mask', 'mask.nii.gz',
            '--output_prefix', 'out/dti_',
            '--b0_threshold', '40.0',
            '--min_s0_threshold', '5.0'
        ]
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py'] + test_argv):
            run_dti_fit.main()
        
        mock_run_dti_fitting_func.assert_called_once()
        call_args = mock_run_dti_fitting_func.call_args[0][0]

        self.assertEqual(call_args.dwi, 'dwi.nii.gz')
        self.assertEqual(call_args.bval, 'bvals.txt')
        self.assertEqual(call_args.bvec, 'bvecs.txt')
        self.assertEqual(call_args.output_prefix, 'out/dti_')
        self.assertEqual(call_args.mask, 'mask.nii.gz')
        self.assertEqual(call_args.b0_threshold, 40.0)
        self.assertEqual(call_args.min_s0_threshold, 5.0)

    def test_main_parser_no_args_triggers_help_exit(self):
        # Test that calling main with no arguments (sys.argv has only script name)
        # results in SystemExit because the script's main() calls parser.print_help() and sys.exit(1).
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py']): # Simulate no CLI args
            with self.assertRaises(SystemExit) as cm:
                run_dti_fit.main()
            self.assertEqual(cm.exception.code, 1) 

    def test_main_parser_missing_required_args(self):
        # Test with missing required arguments (e.g., --dwi is present, but --bval is missing)
        # Argparse itself should cause a SystemExit with code 2.
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py', '--dwi', 'dwi.nii.gz']):
            with self.assertRaises(SystemExit) as cm:
                run_dti_fit.main()
            self.assertEqual(cm.exception.code, 2)
            
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py', '--dwi', 'd.nii', '--bval', 'b.bval', '--bvec', 'b.bvec']): # Missing --output_prefix
            with self.assertRaises(SystemExit) as cm:
                run_dti_fit.main()
            self.assertEqual(cm.exception.code, 2)

    def test_main_parser_help_command(self):
        # Test that '-h' or '--help' causes SystemExit with code 0
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py', '-h']):
            with self.assertRaises(SystemExit) as cm:
                run_dti_fit.main()
            self.assertEqual(cm.exception.code, 0)
        
        with mock.patch.object(sys, 'argv', ['run_dti_fit.py', '--help']):
            with self.assertRaises(SystemExit) as cm:
                run_dti_fit.main()
            self.assertEqual(cm.exception.code, 0)

if __name__ == '__main__':
    unittest.main()
```
