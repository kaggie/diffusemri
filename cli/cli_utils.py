import argparse
import json
import yaml # Requires PyYAML to be installed
import numpy as np
import nibabel as nib
import os
import torch

# It's better to import save_trk from dipy.io.stateful_tractogram as save_trk for modern Dipy versions
# from dipy.io.streamline import save_trk <- Older import
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_trk

# --- Argument Parsing Helpers ---

def add_common_input_args(parser: argparse.ArgumentParser):
    """Adds common input file arguments to an ArgumentParser."""
    parser.add_argument('--dwi', required=True, help="Path to the input 4D DWI NIFTI file.")
    parser.add_argument('--bval', required=True, help="Path to the input b-values file.")
    parser.add_argument('--bvec', required=True, help="Path to the input b-vectors file.")
    parser.add_argument('--mask', help="Path to the input 3D brain mask NIFTI file (optional).")
    return parser

def add_common_output_args(parser: argparse.ArgumentParser):
    """Adds common output arguments to an ArgumentParser."""
    parser.add_argument('--output_prefix', help="Prefix for output files (e.g., 'subject_id/dti_'). Directory will be created if it doesn't exist.")
    parser.add_argument('--output_dir', help="Directory to save output files. If 'output_prefix' includes a path, this is ignored. Defaults to current directory.")
    return parser
    
def add_device_arg(parser: argparse.ArgumentParser):
    """Adds a --device argument (cpu/cuda)."""
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto', 
        help="Computation device: 'cpu', 'cuda', or 'auto' to auto-select CUDA if available (default: 'auto')."
    )
    return parser

def determine_device(requested_device: str) -> torch.device:
    """Determines the torch.device based on user request and availability."""
    if requested_device.lower() == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    elif requested_device.lower() == 'cpu':
        return torch.device('cpu')
    # Default 'auto'
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- Data Loading Helpers ---

def load_nifti_data(filepath: str, ensure_float32: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Loads data and affine from a NIFTI file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIFTI file not found: {filepath}")
    img = nib.load(filepath)
    data = img.get_fdata()
    if ensure_float32 and data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, img.affine

def load_bvals_bvecs(bval_filepath: str, bvec_filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads b-values and b-vectors from specified file paths."""
    if not os.path.exists(bval_filepath):
        raise FileNotFoundError(f"bval file not found: {bval_filepath}")
    if not os.path.exists(bvec_filepath):
        raise FileNotFoundError(f"bvec file not found: {bvec_filepath}")
    
    bvals = np.loadtxt(bval_filepath)
    bvecs = np.loadtxt(bvec_filepath)
    if bvecs.shape[0] == 3 and bvecs.shape[1] == bvals.shape[0]: # If bvecs are FSL format (3xN)
        bvecs = bvecs.T
    elif bvecs.shape[0] != bvals.shape[0] or bvecs.shape[1] != 3:
        raise ValueError(f"bvecs shape {bvecs.shape} is not compatible with bvals shape {bvals.shape} or N_gradients x 3.")
        
    return bvals, bvecs

# --- Data Saving Helpers ---

def save_nifti_data(data: np.ndarray, affine: np.ndarray, filepath: str):
    """Saves data as a NIFTI file."""
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filepath)
    print(f"Saved NIFTI file: {filepath}")

def save_tractogram(streamlines: list, affine: np.ndarray, filepath: str, image_shape_for_sft: tuple = None):
    """
    Saves streamlines as a .trk file using Dipy's StatefulTractogram.

    Args:
        streamlines (list): List of NumPy arrays, where each array is a streamline (Nx3 points).
                            Assumed to be in world (RASMM) coordinates.
        affine (np.ndarray): Affine transformation of the reference image (voxel to RASMM).
        filepath (str): Path to save the .trk file.
        image_shape_for_sft (tuple, optional): Shape of the reference image's data (e.g., DWI spatial dims).
                                              Used to provide dimensional context in the TRK header.
                                              If None, a generic header is used.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Streamlines are already in world (RASMM) space.
    # We provide image_dims and affine for header information.
    if image_shape_for_sft:
        sft = StatefulTractogram(streamlines, image_dims=image_shape_for_sft, 
                                 affine=affine, space=Space.RASMM)
    else: 
        # Create a header with just the affine if shape is not known
        header = nib.Nifti1Header()
        header.set_sform(affine)
        sft = StatefulTractogram(streamlines, header, space=Space.RASMM)
    
    save_trk(sft, filepath, bbox_valid_check=False)
    print(f"Saved tractogram file: {filepath}")


# --- Configuration File Loading ---

def load_config_from_json_yaml(filepath: str) -> dict:
    """Loads parameters from a JSON or YAML configuration file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    config = {}
    with open(filepath, 'r') as f:
        if ext == '.json':
            config = json.load(f)
        elif ext in ['.yaml', '.yml']:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {filepath}: {e}")
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}. Use .json or .yaml.")
    return config

if __name__ == '__main__':
    # Example of using these utilities
    print("--- CLI Utilities Example ---")
    
    # 1. ArgumentParser setup (simulated)
    parser = argparse.ArgumentParser(description="CLI Utility Test App")
    parser = add_common_input_args(parser)
    parser = add_common_output_args(parser)
    parser = add_device_arg(parser)
    parser.add_argument('--extra_param', type=int, default=10, help="An extra test parameter.")
    
    print("Simulating argument parsing (not actually parsing cmd line)...")
    # Example: args = parser.parse_args(['--dwi', 'd.nii', '--bval', 'b.bval', '--bvec', 'b.bvec'])
    # print(f"Example parsed args (if provided): {args}")
    
    # Example: Determine device
    # requested_device_example = 'auto'
    # selected_torch_device = determine_device(requested_device_example)
    # print(f"Selected torch device for '{requested_device_example}': {selected_torch_device}")


    # 2. Config file loading (create dummy files for this)
    dummy_config_json = {"param1": "value1", "learning_rate": 0.01, "iterations": 500}
    dummy_config_yaml = {"param2": "value2", "threshold": 0.2, "options": [1,2,3]}
    
    json_path = "temp_test_config.json"
    yaml_path = "temp_test_config.yaml"

    with open(json_path, 'w') as f:
        json.dump(dummy_config_json, f)
    with open(yaml_path, 'w') as f:
        yaml.dump(dummy_config_yaml, f)
        
    try:
        print("\nLoading JSON config...")
        config_json = load_config_from_json_yaml(json_path)
        print(f"  JSON Config: {config_json}")
        assert config_json["learning_rate"] == 0.01

        print("\nLoading YAML config...")
        config_yaml = load_config_from_json_yaml(yaml_path)
        print(f"  YAML Config: {config_yaml}")
        assert config_yaml["threshold"] == 0.2
        
        print("\nTesting unsupported format...")
        with open("temp_test_config.txt", "w") as f: f.write("test")
        try:
            load_config_from_json_yaml("temp_test_config.txt")
        except ValueError as e:
            print(f"  Caught expected error for .txt: {e}")

    finally:
        if os.path.exists(json_path): os.remove(json_path)
        if os.path.exists(yaml_path): os.remove(yaml_path)
        if os.path.exists("temp_test_config.txt"): os.remove("temp_test_config.txt")
    
    print("\nCLI utilities example finished.")
```
