import argparse
import sys
import os
import numpy as np

# Assuming cli_utils.py is in the same directory or package
from .cli_utils import (
    load_nifti_data, save_nifti_data,
    load_bvals_bvecs
)

# Import core library functions
try:
    from diffusemri.fitting.dti_fitter import fit_dti_volume
except ImportError:
    # Fallback for direct script execution
    # This allows running from 'python cli/run_dti_fit.py ...' if diffusemri is project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) 
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from diffusemri.fitting.dti_fitter import fit_dti_volume


def run_dti_fitting(args):
    print("Starting DTI model fitting...")
    
    # Device determination is not critical here as dti_fitter is NumPy based.

    # Load data
    print(f"Loading DWI data from: {args.dwi}")
    dwi_data, affine = load_nifti_data(args.dwi) # Returns np.ndarray

    print(f"Loading b-values from: {args.bval} and b-vectors from: {args.bvec}")
    bvals, bvecs = load_bvals_bvecs(args.bval, args.bvec) # Returns np.ndarray

    mask_data = None
    dwi_data_to_fit = dwi_data # Default to using unmasked data

    if args.mask:
        print(f"Loading brain mask from: {args.mask}")
        mask_data, mask_affine = load_nifti_data(args.mask, ensure_float32=False) # Mask can be int/bool
        if mask_data.shape != dwi_data.shape[:3]:
            print(f"Error: Mask shape {mask_data.shape} does not match DWI spatial shape {dwi_data.shape[:3]}.", file=sys.stderr)
            sys.exit(1)
        if not np.allclose(affine, mask_affine, atol=1e-3): # Check if affines are reasonably close
             print(f"Warning: Affine of DWI and mask do not match closely. DWI affine:\n{affine}\nMask affine:\n{mask_affine}", file=sys.stderr)
             # Decide if this should be a fatal error or just a warning. For now, a warning.

        print("Applying mask to DWI data before fitting...")
        # Create a new array for masked DWI data to avoid modifying original dwi_data if it's used elsewhere
        dwi_data_masked = np.zeros_like(dwi_data)
        mask_bool = mask_data.astype(bool) 
        
        # Apply mask to each volume in the 4D DWI data
        for i in range(dwi_data.shape[3]):
            volume_slice = dwi_data[..., i]
            masked_volume_slice = dwi_data_masked[..., i]
            masked_volume_slice[mask_bool] = volume_slice[mask_bool]
            dwi_data_masked[..., i] = masked_volume_slice
        dwi_data_to_fit = dwi_data_masked
    else:
        print("No brain mask provided. Fitting DTI to all voxels (or as per S0 thresholding in fitter).")

    print("Fitting DTI model...")
    dti_maps = fit_dti_volume(
        image_data_4d=dwi_data_to_fit,
        b_values=bvals,
        b_vectors=bvecs,
        b0_threshold=args.b0_threshold,
        min_S0_intensity_threshold=args.min_s0_threshold 
    )

    if dti_maps is None or not dti_maps: # Check if it's None or empty dict
        print("DTI fitting failed or produced no maps.", file=sys.stderr)
        sys.exit(1)

    output_prefix_dir = os.path.dirname(args.output_prefix)
    if output_prefix_dir and not os.path.exists(output_prefix_dir):
        os.makedirs(output_prefix_dir, exist_ok=True)
        print(f"Created output directory: {output_prefix_dir}")

    maps_to_save = ["FA", "MD", "AD", "RD"] 
    
    for map_name in maps_to_save:
        if map_name in dti_maps and dti_maps[map_name] is not None:
            map_filepath = f"{args.output_prefix}{map_name}.nii.gz"
            # Ensure map data is float32 for saving, as some Dipy functions might return float64
            save_nifti_data(dti_maps[map_name].astype(np.float32), affine, map_filepath)
        else:
            print(f"Warning: DTI map '{map_name}' not found in fitting results or is None.")

    if "D_tensor_map" in dti_maps and dti_maps["D_tensor_map"] is not None:
        tensor_filepath = f"{args.output_prefix}DT.nii.gz"
        save_nifti_data(dti_maps["D_tensor_map"].astype(np.float32), affine, tensor_filepath)
    
    print("DTI fitting and map saving complete.")
    print(f"Output files saved with prefix: {args.output_prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Diffusion Tensor Imaging (DTI) model to dMRI data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--bval', required=True, help="Path to the input b-values file.")
    parser.add_argument('--bvec', required=True, help="Path to the input b-vectors file.")
    parser.add_argument('--mask', help="Path to the input 3D brain mask NIFTI file (optional). "
                                       "If provided, DTI fitting will be performed only within the mask.")
    
    parser.add_argument('--output_prefix', required=True, 
                        help="Prefix for saving output DTI maps (e.g., 'outputs/subject01_dti_'). "
                             "Resulting files will be like 'prefixFA.nii.gz', 'prefixMD.nii.gz', etc.")

    parser.add_argument('--b0_threshold', type=float, default=50.0,
                        help="b-value threshold to identify b0 images (default: 50.0). Passed to DTI fitter.")
    parser.add_argument('--min_s0_threshold', type=float, default=1.0,
                        help="Minimum S0 intensity for a voxel to be processed by DTI fitter (default: 1.0).")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    try:
        run_dti_fitting(args)
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Input data issue. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```
