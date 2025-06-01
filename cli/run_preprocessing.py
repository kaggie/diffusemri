import argparse
import sys # For exiting
import os
import numpy as np # For saving mask as int8

# Assuming cli_utils.py is in the same directory or package
from .cli_utils import (
    load_nifti_data, save_nifti_data
)

# Import core library functions
# These paths assume 'diffusemri' is the top-level package accessible in PYTHONPATH
# If running cli scripts directly, relative paths from project root might be needed for imports.
try:
    from diffusemri.preprocessing.masking import create_brain_mask
    from diffusemri.preprocessing.denoising import denoise_mppca_data, correct_gibbs_ringing_dipy
    from diffusemri.preprocessing.denoising import denoise_mppca_data, correct_gibbs_ringing_dipy
    from diffusemri.preprocessing.correction import (
        correct_motion_eddy_fsl, load_eddy_outlier_report,
        correct_susceptibility_topup_fsl, correct_bias_field_dipy
    )
except ImportError:
    # Fallback for direct script execution if 'diffusemri' is not in PYTHONPATH
    # This allows running from 'python cli/run_preprocessing.py ...' if diffusemri is project root
    # This is a common pattern but can be tricky. Proper packaging is better.
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    # Corrected fallback for when cli is a sub-package of diffusemri, or diffusemri is the root.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) 
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from diffusemri.preprocessing.masking import create_brain_mask
    from diffusemri.preprocessing.denoising import denoise_mppca_data
    from diffusemri.preprocessing.correction import (
        correct_motion_eddy_fsl, load_eddy_outlier_report,
        correct_susceptibility_topup_fsl, correct_bias_field_dipy
    )


def setup_masking_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--output_mask', required=True, help="Path to save the output 3D brain mask NIFTI file.")
    parser.add_argument('--output_masked_dwi', help="Path to save the output 4D masked DWI NIFTI file (optional).")
    parser.add_argument('--median_radius', type=int, default=4, help="Radius for the median filter (default: 4).")
    parser.add_argument('--numpass', type=int, default=4, help="Number of passes for the median filter (default: 4).")
    parser.set_defaults(func=run_masking)

def run_masking(args):
    print(f"Running brain masking for DWI: {args.dwi}")
    try:
        dwi_data, affine = load_nifti_data(args.dwi)
        
        # Voxel size is not directly used by the current create_brain_mask implementation
        # which uses pytorch_median_otsu -> scipy.ndimage.median_filter.
        # Passing a placeholder or None if the API allows. The current API expects it.
        voxel_size_placeholder = (1.0, 1.0, 1.0) # Or derive from affine if strictly needed

        brain_mask_np, masked_dwi_np = create_brain_mask(
            dmri_data=dwi_data,
            voxel_size=voxel_size_placeholder, 
            median_radius=args.median_radius,
            numpass=args.numpass
        )
        
        save_nifti_data(brain_mask_np.astype(np.int8), affine, args.output_mask) 
        if args.output_masked_dwi:
            save_nifti_data(masked_dwi_np, affine, args.output_masked_dwi)
        print("Brain masking complete.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during masking: {e}", file=sys.stderr)
        sys.exit(1)


def setup_denoising_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--output_dwi', required=True, help="Path to save the output denoised 4D DWI NIFTI file.")
    parser.add_argument('--patch_radius', type=int, default=2, 
                        help="Radius for local patches in MP-PCA (e.g., 2 for 5x5x5 patch, default: 2).")
    parser.set_defaults(func=run_denoising_mppca)

def run_denoising_mppca(args):
    print(f"Running MP-PCA denoising for DWI: {args.dwi}")
    try:
        dwi_data, affine = load_nifti_data(args.dwi) 
        
        denoised_dwi_np = denoise_mppca_data(
            dmri_data=dwi_data,
            patch_radius=args.patch_radius
        )
        
        save_nifti_data(denoised_dwi_np, affine, args.output_dwi)
        print("MP-PCA denoising complete.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during MP-PCA denoising: {e}", file=sys.stderr)
        sys.exit(1)

def setup_correct_fsl_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--bval', required=True, help="Path to the b-values file.")
    parser.add_argument('--bvec', required=True, help="Path to the b-vectors file.")
    parser.add_argument('--mask', required=True, help="Path to the brain mask NIFTI file.")
    parser.add_argument('--index', required=True, help="Path to the FSL index file.")
    parser.add_argument('--acqp', required=True, help="Path to the FSL acquisition parameters file.")
    parser.add_argument('--out_base', required=True, help="Base name for FSL eddy output files (e.g., 'dwi_corrected/eddy_corrected_data').")
    parser.add_argument('--use_cuda', action='store_true', help="Attempt to use FSL's eddy_cuda.")
    parser.add_argument('--repol', action='store_true', help="Detect and replace outlier slices (FSL eddy --repol).")
    parser.add_argument('--cnr_maps', action='store_true', help="Output CNR maps (FSL eddy --cnr_maps).")
    parser.add_argument('--residuals', action='store_true', help="Output residuals (FSL eddy --residuals).")
    parser.add_argument('--fsl_eddy_extra_args', type=str, default="", 
                        help="String of additional arguments to pass to FSL eddy (e.g., '--niter=8 --fwhm=0'). Quote the string.")
    parser.set_defaults(func=run_correct_fsl)

def run_correct_fsl(args):
    print(f"Running FSL eddy motion/eddy current correction for DWI: {args.dwi}")
    
    eddy_kwargs = {
        'repol': args.repol,
        'cnr_maps': args.cnr_maps,
        'residuals': args.residuals
    }
    eddy_kwargs = {k: v for k, v in eddy_kwargs.items() if v is not False} # Keep only if True

    if args.fsl_eddy_extra_args:
        eddy_kwargs['args'] = args.fsl_eddy_extra_args
        
    try:
        corrected_dwi_file, rotated_bvec_file, outlier_report_file, _ = correct_motion_eddy_fsl(
            dwi_file=args.dwi,
            bval_file=args.bval,
            bvec_file=args.bvec,
            mask_file=args.mask,
            index_file=args.index,
            acqp_file=args.acqp,
            out_base_name=args.out_base,
            use_cuda=args.use_cuda,
            **eddy_kwargs 
        )
        print("FSL eddy correction process initiated successfully.")
        print(f"  Corrected DWI will be at: {corrected_dwi_file}")
        print(f"  Rotated b-vectors will be at: {rotated_bvec_file}")
        if outlier_report_file:
            print(f"  Outlier report will be at: {outlier_report_file}")
            # report_data = load_eddy_outlier_report(outlier_report_file)
            # print(f"Successfully loaded outlier report: {len(report_data)} entries.")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e: # Nipype often raises RuntimeError for command execution issues
        print(f"Error: FSL eddy execution failed. Ensure FSL is installed and configured correctly. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e: 
        print(f"An unexpected error occurred during FSL eddy correction: {e}", file=sys.stderr)
        sys.exit(1)


def setup_topup_fsl_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--imain_file', required=True, help="Path to the 4D NIFTI file containing images with opposing PE directions (e.g., AP/PA b0s) for TOPUP.")
    parser.add_argument('--encoding_file', required=True, help="Path to the text file specifying acquisition parameters for TOPUP (PE direction and total readout time).")
    parser.add_argument('--images_to_correct_file', required=True, help="Path to the 4D NIFTI image series to which the susceptibility correction will be applied (e.g., the full DWI dataset).")
    parser.add_argument('--images_to_correct_encoding_indices', required=True, type=str, help="Comma-separated list of 1-based indices for ApplyTOPUP's --inindex (e.g., '1' or '1,1,2,2').")
    parser.add_argument('--out_base_name', required=True, help="Base name for output files. TOPUP outputs will be prefixed with '_topup'.")
    parser.add_argument('--config_file', default="b02b0.cnf", help="Path to FSL TOPUP configuration file (default: b02b0.cnf).")
    # Add other relevant optional TOPUP/ApplyTOPUP arguments here if desired
    # For example:
    # parser.add_argument('--topup_fwhm', type=float, help="FWHM for TOPUP.")
    # parser.add_argument('--applytopup_method', choices=['jac', 'lsr'], default='jac', help="Method for ApplyTOPUP (default: jac).")
    parser.set_defaults(func=run_topup_fsl)

def run_topup_fsl(args):
    print(f"Running FSL TOPUP and ApplyTOPUP for susceptibility correction on: {args.images_to_correct_file}")
    try:
        # Parse encoding indices
        indices_list_str = args.images_to_correct_encoding_indices.split(',')
        indices_list_int = [int(idx.strip()) for idx in indices_list_str]
        if not indices_list_int:
            raise ValueError("images_to_correct_encoding_indices cannot be empty.")

        # For simplicity, topup_kwargs and applytopup_kwargs are not exposed in this basic CLI example
        # but could be added by parsing additional arguments.
        corrected_file, field_file, fieldcoef_file, movpar_file = correct_susceptibility_topup_fsl(
            imain_file=args.imain_file,
            encoding_file=args.encoding_file,
            out_base_name=args.out_base_name,
            images_to_correct_file=args.images_to_correct_file,
            images_to_correct_encoding_indices=indices_list_int,
            config_file=args.config_file
            # topup_kwargs={},
            # applytopup_kwargs={'method': args.applytopup_method} # Example if method was an arg
        )
        print("FSL TOPUP/ApplyTOPUP process initiated successfully.")
        print(f"  Corrected Image: {corrected_file}")
        print(f"  Field Map (Hz): {field_file}")
        print(f"  Field Coefficients: {fieldcoef_file}")
        print(f"  Movement Parameters: {movpar_file}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e: # For issues like parsing indices
        print(f"Error: Parameter issue. {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e: # For Nipype execution errors
        print(f"Error: FSL TOPUP/ApplyTOPUP execution failed. Ensure FSL is installed and configured correctly. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during TOPUP/ApplyTOPUP processing: {e}", file=sys.stderr)
        sys.exit(1)


def setup_bias_field_dipy_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_file', required=True, help="Path to the NIFTI file to be corrected (e.g., mean b0 or T1w).")
    parser.add_argument('--output_file', required=True, help="Path to save the bias field corrected NIFTI file.")
    parser.add_argument('--method', default='n4', const='n4', nargs='?', choices=['n4'],
                        help="Bias field correction method (default: n4). Currently only 'n4' (via Dipý's BiasFieldCorrectionFlow) is supported.")
    parser.add_argument('--mask_file', help="Path to an optional brain mask NIFTI file. Note: For 'n4', if masking is desired, it's often best to provide a pre-masked input_file.")
    # Add common kwargs for BiasFieldCorrectionFlow or N4 if desired, e.g.:
    # parser.add_argument('--threshold', type=float, help="Threshold for internal mask generation by the flow (if no mask provided to N4).")
    # parser.add_argument('--use_cuda', action='store_true', help="Attempt to use CUDA if supported by the method.")
    # parser.add_argument('--n4_num_threads', type=int, help="Number of threads for N4 (if flow passes it).")
    parser.set_defaults(func=run_bias_field_correction_dipy)

def run_bias_field_correction_dipy(args):
    print(f"Running Dipý-based bias field correction (method: {args.method}) on: {args.input_file}")
    try:
        # Collect relevant kwargs for correct_bias_field_dipy
        kwargs_for_correction = {}
        # Example of passing specific kwargs if they were added to CLI parser:
        # if args.threshold is not None: kwargs_for_correction['threshold'] = args.threshold
        # if args.use_cuda: kwargs_for_correction['use_cuda'] = True
        # if args.n4_num_threads: kwargs_for_correction['num_threads'] = args.n4_num_threads # Example for N4

        corrected_output_file = correct_bias_field_dipy(
            input_image_file=args.input_file,
            output_corrected_file=args.output_file,
            method=args.method,
            mask_file=args.mask_file, # Pass mask_file, function will log if unused by method
            **kwargs_for_correction
        )
        print(f"Bias field correction complete. Corrected image saved to: {corrected_output_file}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Bias field correction execution failed. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during bias field correction: {e}", file=sys.stderr)
        sys.exit(1)


def setup_gibbs_ringing_dipy_parser(parser: argparse.ArgumentParser):
    parser.add_argument('--input_file', required=True, help="Path to the NIFTI file to be corrected for Gibbs ringing.")
    parser.add_argument('--output_file', required=True, help="Path to save the Gibbs ringing corrected NIFTI file.")
    parser.add_argument('--slice_axis', type=int, default=2, choices=[0, 1, 2],
                        help="Axis along which slices were acquired (0 for X, 1 for Y, 2 for Z, default: 2).")
    parser.add_argument('--n_points', type=int, default=3,
                        help="Number of neighboring points on each side for local TV calculation (default: 3).")
    parser.add_argument('--num_processes', type=int, default=1,
                        help="Number of processes to use for parallel computation (default: 1, use None in code for auto).")
    parser.set_defaults(func=run_gibbs_ringing_correction_dipy)

def run_gibbs_ringing_correction_dipy(args):
    print(f"Running Dipý-based Gibbs ringing correction on: {args.input_file}")
    try:
        num_processes_arg = args.num_processes
        if num_processes_arg <= 0: # Dipý's gibbs_removal expects None or positive int
            num_processes_arg = None
            print("Interpreting num_processes <= 0 as auto-detect for Dipý.")

        corrected_output_file = correct_gibbs_ringing_dipy(
            input_image_file=args.input_file,
            output_corrected_file=args.output_file,
            slice_axis=args.slice_axis,
            n_points=args.n_points,
            num_processes=num_processes_arg
        )
        print(f"Gibbs ringing correction complete. Corrected image saved to: {corrected_output_file}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Gibbs ringing correction execution failed. Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during Gibbs ringing correction: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing tools for diffusion MRI data.",
        formatter_class=argparse.RawTextHelpFormatter # Preserves newlines in help text
    )
    subparsers = parser.add_subparsers(title="Available Commands", dest="command")
    # Making subparsers required in Python 3.7+ can be done by setting dest and required=True in add_subparsers
    # For older versions, or for more robust handling if no command is given:
    # subparsers.required = True # For Python 3.7+
    # subparsers.dest = 'command' 

    mask_parser = subparsers.add_parser(
        "masking", 
        help="Create a brain mask from DWI data using median Otsu.",
        description="Creates a brain mask from DWI data. Uses a median filter and Otsu's thresholding."
    )
    setup_masking_parser(mask_parser)

    denoise_parser = subparsers.add_parser(
        "denoising_mppca", 
        help="Denoise DWI data using Marchenko-Pastur PCA (MP-PCA).",
        description="Applies MP-PCA based denoising to 4D DWI data."
    )
    setup_denoising_parser(denoise_parser)

    correct_fsl_parser = subparsers.add_parser(
        "correct_fsl", 
        help="Perform motion and eddy current correction using FSL's eddy tool.",
        description="Wraps FSL's 'eddy' tool for motion and eddy current correction."
    )
    setup_correct_fsl_parser(correct_fsl_parser)

    topup_fsl_parser = subparsers.add_parser(
        "topup_fsl",
        help="Perform susceptibility distortion correction using FSL's TOPUP and ApplyTOPUP.",
        description="Wraps FSL's TOPUP for field estimation and ApplyTOPUP for correction."
    )
    setup_topup_fsl_parser(topup_fsl_parser)

    bias_field_parser = subparsers.add_parser(
        "bias_field_dipy",
        help="Perform bias field correction using Dipý's BiasFieldCorrectionFlow (e.g., N4 method).",
        description="Wraps Dipý's BiasFieldCorrectionFlow, typically using N4 for bias field correction."
    )
    setup_bias_field_dipy_parser(bias_field_parser)

    gibbs_parser = subparsers.add_parser(
        "gibbs_ringing_dipy",
        help="Perform Gibbs ringing correction using Dipý's `gibbs_removal`.",
        description="Wraps Dipý's `gibbs_removal` for correcting Gibbs ringing artifacts."
    )
    setup_gibbs_ringing_dipy_parser(gibbs_parser)

    if len(sys.argv) <= 1: # No arguments or just script name
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        # This case should ideally not be reached if subparsers are required
        # or if a default func is set for the main parser.
        # However, if a command is recognized but no func was set (should not happen with current setup)
        print(f"No function associated with command: {args.command}", file=sys.stderr)
        parser.print_help(sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```
