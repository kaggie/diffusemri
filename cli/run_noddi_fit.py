import argparse
import sys
import os
import numpy as np
import torch
import yaml # For load_config_from_json_yaml

# Assuming cli_utils.py is in the same directory or package
from .cli_utils import (
    load_nifti_data, save_nifti_data,
    load_bvals_bvecs, load_config_from_json_yaml,
    add_device_arg, determine_device
)

# Import core library functions
try:
    from diffusemri.fitting.noddi_fitter import fit_noddi_volume
    from diffusemri.utils.pytorch_gradient_utils import PyTorchGradientTable
except ImportError:
    # Fallback for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) 
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from diffusemri.fitting.noddi_fitter import fit_noddi_volume
    from diffusemri.utils.pytorch_gradient_utils import PyTorchGradientTable

def run_noddi_fitting(args):
    print("Starting NODDI model fitting...")
    
    config = {}
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        try:
            config = load_config_from_json_yaml(args.config_file)
        except Exception as e:
            print(f"Error loading configuration file: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Determine device: CLI > config > auto default from determine_device
    final_device_str = 'auto' # Default for determine_device if nothing else is set
    if args.device_from_cli: # If --device was passed on CLI
        final_device_str = args.device 
    elif 'device' in config: # If device is in config file (and not overridden by CLI)
        final_device_str = config['device']
    
    final_torch_device = determine_device(final_device_str)
    print(f"Using effective computation device: {final_torch_device}")

    print(f"Loading DWI data from: {args.dwi}")
    dwi_data_np, affine = load_nifti_data(args.dwi)

    print(f"Loading b-values from: {args.bval} and b-vectors from: {args.bvec}")
    bvals_np, bvecs_np = load_bvals_bvecs(args.bval, args.bvec)

    mask_data_np = None
    if args.mask:
        print(f"Loading brain mask from: {args.mask}")
        mask_data_np, mask_affine = load_nifti_data(args.mask, ensure_float32=False)
        if not np.allclose(affine, mask_affine, atol=1e-3):
             print("Warning: DWI and mask affines differ significantly. Ensure they are aligned.", file=sys.stderr)
        if mask_data_np.shape != dwi_data_np.shape[:3]:
            print(f"Error: Mask shape {mask_data_np.shape} does not match DWI spatial shape {dwi_data_np.shape[:3]}.", file=sys.stderr)
            sys.exit(1)
    else:
        print("No brain mask provided. Consider providing one for optimal results.")

    b0_thresh_gtab_val = 50.0 # PyTorchGradientTable's own default
    if args.b0_threshold_gtab_from_cli and args.b0_threshold_gtab is not None:
        b0_thresh_gtab_val = args.b0_threshold_gtab
    elif 'b0_threshold_gtab' in config:
        b0_thresh_gtab_val = float(config['b0_threshold_gtab'])
    print(f"Using b0_threshold for PyTorchGradientTable: {b0_thresh_gtab_val}")
    gtab_torch = PyTorchGradientTable(bvals_np, bvecs_np, b0_threshold=b0_thresh_gtab_val)

    min_s0_val = float(config.get('min_s0_val', 1.0))
    batch_size_fit = int(config.get('batch_size_fit', 512))
    d_intra_val = float(config.get('d_intra', 1.7e-3)) 
    d_iso_val = float(config.get('d_iso', 3.0e-3))     
    fit_params_dict = config.get('fit_params', None) 
    
    initial_orientation_map_np = None
    initial_orientation_map_path = config.get('initial_orientation_map_path', None)
    if initial_orientation_map_path:
        if os.path.exists(initial_orientation_map_path):
            print(f"Loading initial orientation map from: {initial_orientation_map_path}")
            initial_orientation_map_np, orient_affine = load_nifti_data(initial_orientation_map_path)
            if not np.allclose(affine, orient_affine, atol=1e-3):
                 print("Warning: DWI and initial orientation map affines differ. Ensure alignment.", file=sys.stderr)
            if initial_orientation_map_np.ndim == 4 and initial_orientation_map_np.shape[3] == 3:
                pass 
            elif initial_orientation_map_np.ndim == 3 and initial_orientation_map_np.shape[-1] !=3 : # Try to fix common case
                 temp_map = initial_orientation_map_np[..., np.newaxis] # Add last dim
                 if temp_map.ndim == 4 and temp_map.shape[3] == 3: # Check if it worked out
                      initial_orientation_map_np = temp_map
                 else: # If not, warn and skip
                      print(f"Warning: initial_orientation_map has shape {initial_orientation_map_np.shape}, expected (X,Y,Z,3). Skipping.", file=sys.stderr)
                      initial_orientation_map_np = None
            elif initial_orientation_map_np.ndim == 5 and initial_orientation_map_np.shape[3] == 1 and initial_orientation_map_np.shape[4] == 3: # X,Y,Z,1,3
                initial_orientation_map_np = initial_orientation_map_np.squeeze(axis=3) # X,Y,Z,3
                if initial_orientation_map_np.ndim != 4 or initial_orientation_map_np.shape[-1] != 3 :
                     print(f"Warning: initial_orientation_map (after squeeze) has shape {initial_orientation_map_np.shape}, expected (X,Y,Z,3). Skipping.", file=sys.stderr)
                     initial_orientation_map_np = None
            else:
                print(f"Warning: initial_orientation_map has shape {initial_orientation_map_np.shape}, expected (X,Y,Z,3). Skipping.", file=sys.stderr)
                initial_orientation_map_np = None
        else:
            print(f"Warning: initial_orientation_map_path '{initial_orientation_map_path}' not found. Skipping.", file=sys.stderr)

    print("Fitting NODDI model...")
    noddi_maps = fit_noddi_volume(
        dwi_data=dwi_data_np, 
        gtab=gtab_torch,      
        mask=mask_data_np,    
        min_s0_val=min_s0_val,
        batch_size=batch_size_fit,
        fit_params=fit_params_dict,
        device=final_torch_device, 
        d_intra=d_intra_val,
        d_iso=d_iso_val,
        initial_orientation_map=initial_orientation_map_np
    )

    if noddi_maps is None or not noddi_maps:
        print("NODDI fitting failed or produced no maps.", file=sys.stderr)
        sys.exit(1)

    output_prefix_dir = os.path.dirname(args.output_prefix)
    if output_prefix_dir and not os.path.exists(output_prefix_dir):
        os.makedirs(output_prefix_dir, exist_ok=True)
        print(f"Created output directory: {output_prefix_dir}")

    for map_name, map_data in noddi_maps.items():
        if map_data is not None:
            map_filepath = f"{args.output_prefix}{map_name}.nii.gz"
            save_nifti_data(map_data.astype(np.float32), affine, map_filepath)
        else:
            print(f"Warning: NODDI map '{map_name}' not found in results or is None.")
            
    print("NODDI fitting and map saving complete.")
    print(f"Output files saved with prefix: {args.output_prefix}")

def main(cli_args_list=None): # Renamed for clarity in testing
    parser = argparse.ArgumentParser(
        description="Fit Neurite Orientation Dispersion and Density Imaging (NODDI) model to dMRI data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--bval', required=True, help="Path to the input b-values file.")
    parser.add_argument('--bvec', required=True, help="Path to the input b-vectors file.")
    parser.add_argument('--mask', required=True, help="Path to the input 3D brain mask NIFTI file.") # Made required for NODDI
    
    parser.add_argument('--output_prefix', required=True, 
                        help="Prefix for saving output NODDI maps (e.g., 'outputs/subject01_noddi_').")

    parser.add_argument('--config_file', help="Path to a JSON or YAML configuration file for detailed fitting parameters.")
    
    parser.add_argument('--b0_threshold_gtab', type=float, default=None, 
                        help="b-value threshold for PyTorchGradientTable (overrides config or internal default of 50.0).")

    add_device_arg(parser) # Adds --device with default 'auto'
    
    # Determine effective args for parsing (respects testing or direct CLI call)
    effective_args_to_parse = cli_args_list if cli_args_list is not None else sys.argv[1:]

    if not effective_args_to_parse and '-h' not in sys.argv and '--help' not in sys.argv : # No args and not help
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args(args=effective_args_to_parse)
    
    # Check if device or b0_threshold_gtab were explicitly set via CLI to manage override logic
    args.device_from_cli = '--device' in effective_args_to_parse
    args.b0_threshold_gtab_from_cli = '--b0_threshold_gtab' in effective_args_to_parse
    
    try:
        run_noddi_fitting(args)
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e: # For data shape issues etc.
        print(f"Error: Data or parameter issue. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during NODDI fitting: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```
