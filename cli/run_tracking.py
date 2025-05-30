import argparse
import sys
import os
import numpy as np
import torch # For device determination

# Assuming cli_utils.py is in the same directory or package
from .cli_utils import (
    load_nifti_data, save_tractogram,
    load_bvals_bvecs, load_config_from_json_yaml,
    add_device_arg, determine_device # determine_device might not be used if pipeline handles it
)

# Import core library functions
try:
    from diffusemri.tracking.deterministic import track_deterministic_oudf
    from dipy.core.gradients import gradient_table as create_dipy_gradient_table
    from dipy.tracking.streamline import Streamlines as DipyStreamlines 
except ImportError:
    # Fallback for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) 
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from diffusemri.tracking.deterministic import track_deterministic_oudf
    from dipy.core.gradients import gradient_table as create_dipy_gradient_table
    from dipy.tracking.streamline import Streamlines as DipyStreamlines


def parse_seeds(seed_arg: str, affine_np: np.ndarray) -> np.ndarray:
    """
    Parses the --seeds argument.
    If it's a path to a NIFTI file, loads it as a mask.
    If it's a string of coordinates (e.g., "10,12,14;11,13,15"), parses them.
    Returns seeds as a NumPy array (either boolean mask or Nx3 coordinates).
    """
    if os.path.exists(seed_arg):
        print(f"Loading seed mask from: {seed_arg}")
        mask_data, mask_affine = load_nifti_data(seed_arg, ensure_float32=False)
        if not np.allclose(affine_np, mask_affine, atol=1e-3): # Compare with DWI affine
            print("Warning: DWI and seed mask affines differ. Ensure consistent space for voxel coordinates.", file=sys.stderr)
        return mask_data.astype(bool) 
    else:
        try:
            coord_list = []
            for point_str in seed_arg.split(';'):
                if not point_str.strip(): continue # Skip empty parts
                coords = [float(c.strip()) for c in point_str.split(',')]
                if len(coords) == 3:
                    coord_list.append(coords)
                else:
                    raise ValueError("Each coordinate point must have 3 values.")
            if not coord_list:
                raise ValueError("No coordinates found in seed string.")
            print(f"Using seed coordinates (voxel space): {coord_list}")
            return np.array(coord_list, dtype=np.float32)
        except ValueError as e:
            print(f"Error: --seeds argument '{seed_arg}' is not a valid file path "
                  f"nor a valid coordinate string (e.g., 'x1,y1,z1;x2,y2,z2'). Details: {e}", file=sys.stderr)
            sys.exit(1)


def run_deterministic_tracking(args):
    print("Starting Deterministic Tractography...")
    
    config = {}
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        try:
            config = load_config_from_json_yaml(args.config_file)
        except Exception as e:
            print(f"Error loading configuration file: {e}", file=sys.stderr)
            sys.exit(1)

    # Note: The track_deterministic_oudf wrapper currently auto-detects device.
    # If we want CLI to control it, the wrapper needs a 'device' param.
    # For now, this CLI --device arg is informational for the user, or for future use.
    effective_torch_device = determine_device(args.device)
    print(f"Requested computation device (passed to PyTorch components if wrapper supports it): {effective_torch_device}")


    print(f"Loading DWI data from: {args.dwi}")
    dwi_data_np, affine_np = load_nifti_data(args.dwi)

    print(f"Loading b-values from: {args.bval} and b-vectors from: {args.bvec}")
    bvals_np, bvecs_np = load_bvals_bvecs(args.bval, args.bvec)
    
    gtab_b0_thresh = float(config.get('b0_threshold_gtab', 50.0)) # For gtab creation
    gtab_dipy = create_dipy_gradient_table(bvals_np, bvecs_np, b0_threshold=gtab_b0_thresh)

    print(f"Loading stopping metric map from: {args.stopping_metric_map}")
    metric_map_np, metric_affine = load_nifti_data(args.stopping_metric_map)
    if not np.allclose(affine_np, metric_affine, atol=1e-3):
        print("Warning: DWI and stopping metric map affines differ. Ensure alignment.", file=sys.stderr)

    # Seeds are parsed into a NumPy array (mask or coordinates)
    # The track_deterministic_oudf wrapper expects seeds in this NumPy format.
    seeds_np_parsed = parse_seeds(args.seeds, affine_np)

    # Consolidate parameters: CLI overrides config, then config overrides function defaults
    def get_param(cli_val, config_key, default_val, type_cast=None):
        val = cli_val if cli_val is not None else config.get(config_key, default_val)
        return type_cast(val) if type_cast and val is not None else val

    step_size = get_param(args.step_size, 'step_size', 0.5, float)
    sh_order = get_param(args.sh_order, 'sh_order', 6, int) # Wrapper default is 8, pipeline default is 8
    response_val = config.get('response', None) # Special handling if complex object
    model_max_peaks = get_param(args.model_max_peaks, 'model_max_peaks', 5, int)
    model_min_sep_angle = get_param(args.model_min_separation_angle, 'model_min_separation_angle', 25, float)
    model_peak_thresh = get_param(args.model_peak_threshold, 'model_peak_threshold', 0.5, float)
    max_crossing_angle_trk = get_param(args.max_crossing_angle, 'max_crossing_angle', 60.0, float)
    min_length_trk = get_param(args.min_length, 'min_length', 10.0, float)
    max_length_trk = get_param(args.max_length, 'max_length', 250.0, float)
    max_steps_trk = get_param(args.max_steps, 'max_steps', 1000, int)
    
    print("Running deterministic tractography via wrapper...")
    # The wrapper `track_deterministic_oudf` handles internal PyTorch conversions and device.
    streamlines_obj = track_deterministic_oudf(
        dwi_data=dwi_data_np,
        gtab=gtab_dipy,
        seeds=seeds_np_parsed, 
        affine=affine_np,
        metric_map_for_stopping=metric_map_np,
        stopping_threshold_value=args.stopping_threshold, # Directly from CLI
        step_size=step_size,
        sh_order=sh_order,
        response=response_val, 
        model_max_peaks=model_max_peaks,
        model_min_separation_angle=model_min_sep_angle,
        model_peak_threshold=model_peak_thresh,
        max_crossing_angle=max_crossing_angle_trk,
        min_length=min_length_trk,
        max_length=max_length_trk,
        max_steps=max_steps_trk
        # If wrapper is updated to take device: pass effective_torch_device.type (e.g. 'cpu' or 'cuda')
    )

    if streamlines_obj is None or len(streamlines_obj) == 0:
        print("No streamlines generated or kept after filtering.", file=sys.stderr)
    
    save_tractogram(list(streamlines_obj), affine_np, args.output_tracts, 
                    image_shape_for_sft=dwi_data_np.shape[:3])
    print(f"Deterministic tractography complete. Output saved to {args.output_tracts}")

def main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Perform deterministic fiber tractography using ODF peaks.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--bval', required=True, help="Path to the input b-values file.")
    parser.add_argument('--bvec', required=True, help="Path to the input b-vectors file.")
    parser.add_argument('--stopping_metric_map', required=True, 
                        help="Path to a 3D NIFTI file containing the metric for stopping criteria (e.g., FA map).")
    parser.add_argument('--stopping_threshold', required=True, type=float,
                        help="Threshold value for the stopping_metric_map.")
    parser.add_argument('--seeds', required=True, 
                        help="Seed points for tracking. Can be a path to a 3D NIFTI mask file, "
                             "or a string of voxel coordinates (e.g., 'x1,y1,z1;x2,y2,z2').")
    parser.add_argument('--output_tracts', required=True, help="Path to save the output tractogram (.trk file).")

    parser.add_argument('--config_file', help="Path to JSON/YAML config file for detailed parameters.")
    
    # Model Parameters (overriding config)
    parser.add_argument('--sh_order', type=int, default=None, help="Spherical harmonic order.")
    parser.add_argument('--model_max_peaks', type=int, default=None, help="Max peaks for CSD model.")
    parser.add_argument('--model_min_separation_angle', type=float, default=None, help="Min separation angle for CSD peaks (degrees).")
    parser.add_argument('--model_peak_threshold', type=float, default=None, help="Relative peak threshold for CSD model.")

    # Tracking Parameters (overriding config)
    parser.add_argument('--step_size', type=float, default=None, help="Step size in mm.")
    parser.add_argument('--max_crossing_angle', type=float, default=None, help="Max crossing angle in degrees.")
    parser.add_argument('--min_length', type=float, default=None, help="Min streamline length in mm.")
    parser.add_argument('--max_length', type=float, default=None, help="Max streamline length in mm.")
    parser.add_argument('--max_steps', type=int, default=None, help="Max steps per streamline.")
    
    add_device_arg(parser) # Adds --device, default 'auto'

    effective_args_to_parse = cli_args if cli_args is not None else sys.argv[1:]
    if not effective_args_to_parse and not any(arg in ['-h', '--help'] for arg in sys.argv):
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args(effective_args_to_parse)
    
    try:
        run_deterministic_tracking(args)
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Data or parameter issue. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during tracking: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```
