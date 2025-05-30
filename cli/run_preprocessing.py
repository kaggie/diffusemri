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
    from diffusemri.preprocessing.denoising import denoise_mppca_data
    from diffusemri.preprocessing.correction import correct_motion_eddy_fsl, load_eddy_outlier_report
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
    from diffusemri.preprocessing.correction import correct_motion_eddy_fsl, load_eddy_outlier_report


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
