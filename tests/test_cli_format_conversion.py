import unittest
from unittest import mock
import os
import tempfile
import numpy as np # Added for np.array in mock returns
import argparse

# Adjust import path based on actual project structure
from cli import run_format_conversion

class TestCliFormatConversion(unittest.TestCase):

    @mock.patch('cli.run_format_conversion.read_nrrd_data')
    @mock.patch('cli.run_format_conversion.save_nifti_data') # This is from data_io.cli_utils via diffusemri
    @mock.patch('cli.run_format_conversion.np.savetxt') # Mock for bval/bvec saving
    def test_cli_nrrd_to_nifti(self, mock_savetxt, mock_save_nifti, mock_read_nrrd):
        # Mock return values
        mock_read_nrrd.return_value = (
            mock.Mock(name="imageData", dtype=float), # Mock data
            mock.Mock(name="affineMatrix"),          # Mock affine
            np.array([0, 1000]),                     # Mock bvals
            np.array([[0,0,0],[1,0,0]]),             # Mock bvecs
            {"modality": "DWMRI"}                    # Mock header
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_nrrd = os.path.join(tmpdir, "input.nrrd")
            output_nifti = os.path.join(tmpdir, "output.nii.gz")
            output_bval = os.path.join(tmpdir, "output.bval")
            output_bvec = os.path.join(tmpdir, "output.bvec")

            # Create dummy input nrrd for os.path.exists checks if any in CLI (not in this case)
            # open(input_nrrd, 'a').close()

            cli_args = [
                'nrrd2nii',
                '--input_nrrd', input_nrrd,
                '--output_nifti', output_nifti,
                '--output_bval', output_bval,
                '--output_bvec', output_bvec
            ]

            try:
                run_format_conversion.main(cli_args)
            except SystemExit as e:
                self.fail(f"CLI script exited unexpectedly with code {e.code} for args: {cli_args}")

            mock_read_nrrd.assert_called_once_with(input_nrrd)
            mock_save_nifti.assert_called_once_with(
                mock_read_nrrd.return_value[0], # image_data
                mock_read_nrrd.return_value[1], # affine_matrix
                output_nifti
            )
            # Check bval/bvec saves
            self.assertEqual(mock_savetxt.call_count, 2)
            # Note: call_args_list records calls in order.
            # First call for bvals
            self.assertEqual(mock_savetxt.call_args_list[0][0][0], output_bval)
            np.testing.assert_array_equal(mock_savetxt.call_args_list[0][0][1], np.array([0, 1000]).reshape(1,-1))
            # Second call for bvecs
            self.assertEqual(mock_savetxt.call_args_list[1][0][0], output_bvec)
            np.testing.assert_array_equal(mock_savetxt.call_args_list[1][0][1], np.array([[0,0,0],[1,0,0]]))


    @mock.patch('cli.run_format_conversion.nib.load')
    @mock.patch('cli.run_format_conversion.np.loadtxt')
    @mock.patch('cli.run_format_conversion.write_nrrd_data')
    def test_cli_nifti_to_nrrd(self, mock_write_nrrd, mock_loadtxt, mock_nib_load):
        # Mock nib.load
        mock_img_instance = mock.Mock()
        mock_img_instance.get_fdata.return_value = np.random.rand(10,10,10,5).astype(np.float32)
        mock_img_instance.affine = np.eye(4)
        mock_nib_load.return_value = mock_img_instance

        # Mock np.loadtxt for bval and bvec
        # bval first, then bvec
        mock_loadtxt.side_effect = [
            np.array([0, 1000, 1000, 1000, 1000]), # bvals
            np.random.rand(5,3) # bvecs (Nx3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_nifti = os.path.join(tmpdir, "input.nii.gz")
            output_nrrd = os.path.join(tmpdir, "output.nrrd")
            input_bval = os.path.join(tmpdir, "input.bval")
            input_bvec = os.path.join(tmpdir, "input.bvec")

            # Create dummy files for os.path.exists checks in the CLI function
            open(input_nifti, 'a').close()
            open(input_bval, 'a').close()
            open(input_bvec, 'a').close()

            cli_args = [
                'nii2nrrd',
                '--input_nifti', input_nifti,
                '--output_nrrd', output_nrrd,
                '--input_bval', input_bval,
                '--input_bvec', input_bvec,
                '--nrrd_encoding', 'raw' # Test an optional arg
            ]

            try:
                run_format_conversion.main(cli_args)
            except SystemExit as e:
                 self.fail(f"CLI script exited unexpectedly with code {e.code} for args: {cli_args}")

            mock_nib_load.assert_called_once_with(input_nifti)
            self.assertEqual(mock_loadtxt.call_count, 2)
            mock_loadtxt.assert_any_call(input_bval)
            mock_loadtxt.assert_any_call(input_bvec)

            mock_write_nrrd.assert_called_once()
            call_kwargs = mock_write_nrrd.call_args[1] # Get kwargs
            self.assertEqual(call_kwargs['output_filepath'], output_nrrd)
            np.testing.assert_array_equal(call_kwargs['data'], mock_img_instance.get_fdata.return_value)
            np.testing.assert_array_equal(call_kwargs['affine'], mock_img_instance.affine)
            np.testing.assert_array_equal(call_kwargs['bvals'], mock_loadtxt.side_effect[0])
            np.testing.assert_array_equal(call_kwargs['bvecs'], mock_loadtxt.side_effect[1])
            self.assertEqual(call_kwargs['nrrd_header_options']['encoding'], 'raw')


    @mock.patch('cli.run_format_conversion.read_mhd_data')
    @mock.patch('cli.run_format_conversion.save_nifti_data')
    @mock.patch('cli.run_format_conversion.np.savetxt')
    def test_cli_mhd_to_nifti(self, mock_savetxt, mock_save_nifti, mock_read_mhd):
        mock_read_mhd.return_value = (
            mock.Mock(name="imageDataMhd", dtype=float),
            mock.Mock(name="affineMatrixMhd"),
            np.array([0, 2000]),
            np.array([[0,0,0],[0,0,1]]),
            {"Modality": "DWMRI"}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_mhd = os.path.join(tmpdir, "input.mhd")
            output_nifti = os.path.join(tmpdir, "output_from_mhd.nii.gz")
            output_bval = os.path.join(tmpdir, "output_from_mhd.bval")
            output_bvec = os.path.join(tmpdir, "output_from_mhd.bvec")
            open(input_mhd, 'a').close() # Ensure file exists for CLI checks if any

            cli_args = [
                'mhd2nii',
                '--input_mhd', input_mhd,
                '--output_nifti', output_nifti,
                '--output_bval', output_bval,
                '--output_bvec', output_bvec
            ]
            try:
                run_format_conversion.main(cli_args)
            except SystemExit as e:
                self.fail(f"CLI script mhd2nii exited unexpectedly: {e.code}")

            mock_read_mhd.assert_called_once_with(input_mhd)
            mock_save_nifti.assert_called_once_with(
                mock_read_mhd.return_value[0], mock_read_mhd.return_value[1], output_nifti
            )
            self.assertEqual(mock_savetxt.call_count, 2)
            # Call 1 for bvals
            self.assertEqual(mock_savetxt.call_args_list[0][0][0], output_bval)
            np.testing.assert_array_equal(mock_savetxt.call_args_list[0][0][1], np.array([0, 2000]).reshape(1,-1))
            # Call 2 for bvecs
            self.assertEqual(mock_savetxt.call_args_list[1][0][0], output_bvec)
            np.testing.assert_array_equal(mock_savetxt.call_args_list[1][0][1], np.array([[0,0,0],[0,0,1]]))


    @mock.patch('cli.run_format_conversion.nib.load')
    @mock.patch('cli.run_format_conversion.np.loadtxt')
    @mock.patch('cli.run_format_conversion.write_mhd_data')
    def test_cli_nifti_to_mhd(self, mock_write_mhd, mock_loadtxt, mock_nib_load):
        mock_img_instance = mock.Mock()
        mock_img_instance.get_fdata.return_value = np.random.rand(10,10,10,3).astype(np.float32)
        mock_img_instance.affine = np.eye(4)
        mock_nib_load.return_value = mock_img_instance

        mock_loadtxt.side_effect = [ np.array([0, 1000, 2000]), np.random.rand(3,3) ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_nifti = os.path.join(tmpdir, "input_for_mhd.nii.gz")
            output_mhd = os.path.join(tmpdir, "output.mhd")
            input_bval = os.path.join(tmpdir, "input_for_mhd.bval")
            input_bvec = os.path.join(tmpdir, "input_for_mhd.bvec")
            open(input_nifti, 'a').close()
            open(input_bval, 'a').close()
            open(input_bvec, 'a').close()

            cli_args = [
                'nii2mhd',
                '--input_nifti', input_nifti,
                '--output_mhd', output_mhd,
                '--input_bval', input_bval,
                '--input_bvec', input_bvec
            ]
            try:
                run_format_conversion.main(cli_args)
            except SystemExit as e:
                self.fail(f"CLI script nii2mhd exited unexpectedly: {e.code}")

            mock_nib_load.assert_called_once_with(input_nifti)
            self.assertEqual(mock_loadtxt.call_count, 2)

            mock_write_mhd.assert_called_once()
            call_kwargs = mock_write_mhd.call_args[1]
            self.assertEqual(call_kwargs['output_filepath'], output_mhd)
            np.testing.assert_array_equal(call_kwargs['data'], mock_img_instance.get_fdata.return_value)
            np.testing.assert_array_equal(call_kwargs['affine'], mock_img_instance.affine)
            np.testing.assert_array_equal(call_kwargs['bvals'], mock_loadtxt.side_effect[0])
            np.testing.assert_array_equal(call_kwargs['bvecs'], mock_loadtxt.side_effect[1])


    @mock.patch('cli.run_format_conversion.read_analyze_data')
    @mock.patch('cli.run_format_conversion.save_nifti_data') # Using this more specific mock
    def test_cli_analyze_to_nifti(self, mock_save_nifti_data, mock_read_analyze):
        mock_read_analyze.return_value = (
            np.random.rand(10,10,10).astype(np.float32), # data
            np.eye(4), # affine
            mock.Mock() # header
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_analyze = os.path.join(tmpdir, "input.hdr")
            output_nifti = os.path.join(tmpdir, "output_from_analyze.nii.gz")
            open(input_analyze, 'a').close()

            cli_args = [
                'analyze2nii',
                '--input_analyze', input_analyze,
                '--output_nifti', output_nifti
            ]
            try:
                run_format_conversion.main(cli_args)
            except SystemExit as e:
                self.fail(f"CLI script analyze2nii exited unexpectedly: {e.code}")

            mock_read_analyze.assert_called_once_with(input_analyze)
            # save_nifti_data is called by the CLI handler run_analyze_to_nifti
            mock_save_nifti_data.assert_called_once_with(
                mock_read_analyze.return_value[0], # image_data
                mock_read_analyze.return_value[1], # affine
                output_nifti
            )

    @mock.patch('cli.run_format_conversion.nib.load')
    @mock.patch('cli.run_format_conversion.write_analyze_data')
    def test_cli_nifti_to_analyze(self, mock_write_analyze, mock_nib_load):
        mock_img_instance = mock.Mock()
        mock_img_instance.get_fdata.return_value = np.random.rand(10,10,10).astype(np.float32)
        mock_img_instance.affine = np.eye(4)
        mock_img_instance.header = mock.Mock() # Mock the header object
        mock_nib_load.return_value = mock_img_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            input_nifti = os.path.join(tmpdir, "input_for_analyze.nii.gz")
            output_analyze = os.path.join(tmpdir, "output.hdr")
            open(input_nifti, 'a').close()

            cli_args = [
                'nii2analyze',
                '--input_nifti', input_nifti,
                '--output_analyze', output_analyze
            ]
            try:
                run_format_conversion.main(cli_args)
            except SystemExit as e:
                self.fail(f"CLI script nii2analyze exited unexpectedly: {e.code}")

            mock_nib_load.assert_called_once_with(input_nifti)
            mock_write_analyze.assert_called_once()
            call_kwargs = mock_write_analyze.call_args[1]
            self.assertEqual(call_kwargs['output_filepath'], output_analyze)
            np.testing.assert_array_equal(call_kwargs['data'], mock_img_instance.get_fdata.return_value)
            np.testing.assert_array_equal(call_kwargs['affine'], mock_img_instance.affine)
            self.assertEqual(call_kwargs['header'], mock_img_instance.header)


if __name__ == '__main__':
    unittest.main()
