import torch
import numpy as np
from utils.pytorch_gradient_utils import PyTorchGradientTable # New import
from typing import Optional, Tuple, Dict, Any

from models.noddi_model import NoddiModelTorch
from models import noddi_signal # For synthetic data generation

EPS = 1e-8 # Small epsilon for numerical stability


def preprocess_noddi_input(
    dwi_data: np.ndarray,
    gtab: PyTorchGradientTable, # Changed type hint
    mask: Optional[np.ndarray] = None,
    # b0_threshold: float = EPS, # Removed: b0_threshold is now inherent to gtab
    s0_norm_method: str = "mean",
    min_s0_val: float = 1.0
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Preprocesses DWI data for NODDI model fitting.

    This function performs several key steps:
    1.  Identifies b0 images from the gradient table.
    2.  Calculates the S0 (signal intensity at b=0) map.
    3.  Creates a mask of valid voxels based on an optional input brain mask
        and a minimum S0 intensity threshold (`min_s0_val`).
    4.  Extracts DWI signals and S0 values for these valid voxels.
    5.  Normalizes the DWI signals by their corresponding S0 values (S_dw / S0).
    6.  Clips normalized signals to a plausible range to handle noise.
    7.  Converts the processed signals and S0 values to PyTorch tensors.

    Args:
        dwi_data (np.ndarray): The 4D DWI data array with shape (X, Y, Z, N_gradients).
        gtab (PyTorchGradientTable): PyTorchGradientTable object corresponding to the DWI data.
                                     The b0_threshold used for its initialization determines b0s.
        mask (Optional[np.ndarray], optional): A 3D boolean brain mask. Voxels outside
            this mask (if provided) are excluded from processing. Shape: (X, Y, Z).
            Defaults to None, in which case all voxels are initially considered.
        s0_norm_method (str, optional): Method to calculate S0 from multiple b0 images.
            - "mean": Averages the signal intensity across all identified b0 images (default).
            - "first": Uses the signal intensity of the first identified b0 image.
            Defaults to "mean".
        min_s0_val (float, optional): Minimum S0 value for a voxel to be included in
            the fitting process. Voxels with S0 below this threshold are excluded.
            Defaults to 1.0.

    Returns:
        Tuple[torch.Tensor, np.ndarray, torch.Tensor]: A tuple containing:
            - dwi_signals_normalized_torch (torch.Tensor): A 2D tensor of normalized DWI
              signals (S_dw / S0) for valid voxels. Shape: (N_valid_voxels, N_gradients).
            - valid_voxel_coords (np.ndarray): A 2D NumPy array containing the (x, y, z)
              coordinates (indices) of the valid voxels. Shape: (N_valid_voxels, 3).
            - s0_values_torch (torch.Tensor): A 1D tensor of S0 values for the
              valid voxels. Shape: (N_valid_voxels,).

    Raises:
        ValueError: If `dwi_data` is not 4D, if `mask` shape doesn't match DWI spatial
            dimensions, if no b0 images are found, or if no valid voxels are found.
    """
    if dwi_data.ndim != 4:
        raise ValueError("dwi_data must be a 4D array.")
    
    img_shape = dwi_data.shape[:3]

    if mask is None:
        mask = np.ones(img_shape, dtype=bool)
    else:
        if mask.shape != img_shape:
            raise ValueError("Mask shape must match DWI data spatial dimensions.")
    
    # Use b0s_mask directly from the PyTorchGradientTable
    b0_mask_gtab = gtab.b0s_mask # This is a PyTorch boolean tensor

    # Indexing NumPy array dwi_data with a PyTorch boolean tensor works directly.
    s0_images = dwi_data[..., b0_mask_gtab.cpu().numpy()] # Convert to numpy for indexing if issue, but often works
    if s0_images.shape[-1] == 0: # Check if any b0 images were found
        raise ValueError(
            f"No b0 images found based on gtab.b0s_mask. "
            f"Number of b0s identified by gtab: {torch.sum(gtab.b0s_mask).item()}"
        )

    if s0_norm_method == "mean":
        s0_map = np.mean(s0_images, axis=-1)
    elif s0_norm_method == "first":
        s0_map = s0_images[..., 0]
    else:
        raise ValueError(f"Unknown s0_norm_method: {s0_norm_method}. Choose 'mean' or 'first'.")

    # Apply mask and S0 threshold to find valid voxels for fitting
    valid_mask = mask & (s0_map >= min_s0_val) # Use >= for min_s0_val
    valid_voxel_coords = np.array(np.where(valid_mask)).T  # Shape: (N_valid_voxels, 3)

    if valid_voxel_coords.shape[0] == 0:
        raise ValueError("No valid voxels found after masking and S0 thresholding.")

    # Extract signals and S0 for valid voxels
    dwi_signals_valid_voxels = dwi_data[valid_mask]  # Shape: (N_valid_voxels, N_gradients)
    s0_values_valid_voxels = s0_map[valid_mask]      # Shape: (N_valid_voxels,)

    # Normalize DWI signals: S_dw / S0
    # Add EPS to s0_values to prevent division by zero, though min_s0_val should largely prevent S0 being zero.
    s0_for_norm = s0_values_valid_voxels[:, np.newaxis] + EPS
    dwi_signals_normalized = dwi_signals_valid_voxels / s0_for_norm
    
    # Clamp normalized signals to be physically plausible.
    # Signals S/S0 should ideally be between 0 and 1.
    # Clamping to a slightly larger upper bound (e.g., 1.5) can accommodate noise in S0 or S_dw.
    dwi_signals_normalized = np.clip(dwi_signals_normalized, EPS, 1.5)

    # Convert to PyTorch tensors
    dwi_signals_normalized_torch = torch.from_numpy(dwi_signals_normalized).float()
    s0_values_torch = torch.from_numpy(s0_values_valid_voxels).float()
    
    return dwi_signals_normalized_torch, valid_voxel_coords, s0_values_torch


def fit_noddi_volume(
    dwi_data: np.ndarray,
    gtab: PyTorchGradientTable, # Changed type hint
    mask: Optional[np.ndarray] = None,
    # b0_threshold: float = EPS, # Removed
    s0_norm_method: str = "mean",
    min_s0_val: float = 1.0,
    batch_size: int = 512,
    fit_params: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    d_intra: float = noddi_signal.d_intra_default,
    d_iso: float = noddi_signal.d_iso_default,
    initial_orientation_map: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Fits the NODDI model to an entire 3D volume from 4D DWI data.

    This function orchestrates the NODDI model fitting process for a whole brain volume:
    1.  Calls `preprocess_noddi_input` to get normalized signals for valid voxels.
    2.  Initializes the `NoddiModelTorch` (PyTorch-based NODDI model).
    3.  Iterates through the valid voxels in batches of specified `batch_size`.
    4.  For each batch, calls `NoddiModelTorch.fit_batch()` to perform the optimization.
        Optionally uses `initial_orientation_map` to provide initial guesses for `mu`.
        Regularization parameters (L1/L2) can also be passed to `fit_batch`
        via the `fit_params` dictionary.
    5.  Stores the fitted parameters (e.g., f_intra, f_iso, ODI, kappa, orientation angles)
        into full 3D NumPy arrays at their original voxel locations.
    6.  Voxels outside the valid mask or with S0 below threshold will have default (zero)
        values in the output parameter maps.

    Args:
        dwi_data (np.ndarray): The 4D DWI data array (X, Y, Z, N_gradients).
        gtab (PyTorchGradientTable): PyTorchGradientTable object for the DWI data.
                                     Its internal b0_threshold determines b0 identification.
        mask (Optional[np.ndarray], optional): 3D boolean brain mask (X, Y, Z).
            If None, attempts to fit all voxels that meet S0 criteria. Defaults to None.
        s0_norm_method (str, optional): Method for S0 calculation ("mean" or "first").
            Defaults to "mean".
        min_s0_val (float, optional): Minimum S0 value for a voxel to be processed.
            Defaults to 1.0.
        batch_size (int, optional): Number of voxels to fit in each batch passed to the
            PyTorch model. Adjust based on available GPU memory. Defaults to 512.
        fit_params (Optional[Dict[str, Any]], optional): Dictionary of parameters to pass
            to `NoddiModelTorch.fit_batch()`. This can include 'learning_rate',
            'n_iterations', as well as regularization parameters like 'l1_penalty_weight',
            'l2_penalty_weight', and 'l1_params_to_regularize'.
            If None, uses default fitting parameters from `NoddiModelTorch`. Defaults to None.
        device (Optional[torch.device], optional): PyTorch device to use for fitting
            (e.g., torch.device('cuda'), torch.device('cpu')). If None, auto-detects CUDA
            availability. Defaults to None.
        d_intra (float, optional): Fixed intrinsic diffusivity for the intra-cellular compartment.
            Defaults to `noddi_signal.d_intra_default`.
        d_iso (float, optional): Fixed diffusivity for the isotropic compartment.
            Defaults to `noddi_signal.d_iso_default`.
        initial_orientation_map (Optional[np.ndarray], optional): A 4D NumPy array
            (X, Y, Z, 3) providing initial guesses for the mean neurite orientation (mu)
            in Cartesian coordinates (e.g., from DTI primary eigenvector). If provided,
            these orientations are used to initialize the `mu_theta` and `mu_phi` parameters
            for fitting. Defaults to None (standard initialization).

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are strings representing NODDI
            parameter names (e.g., 'f_intra', 'f_iso', 'odi', 'kappa', 'mu_theta',
            'mu_phi', 'f_extra', 'S0_fit') and values are the corresponding 3D NumPy arrays
            of the fitted parameters for the entire volume.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if fit_params is None:
        fit_params = {'learning_rate': 0.01, 'n_iterations': 500} # Default fit params

    # 1. Preprocess data
    print("Preprocessing DWI data...")
    # Call preprocess_noddi_input without b0_threshold, as it's now derived from gtab
    norm_signals_torch, valid_coords, s0_values_torch = preprocess_noddi_input(
        dwi_data, gtab, mask, s0_norm_method, min_s0_val
    )
    n_valid_voxels = norm_signals_torch.shape[0]
    print(f"Found {n_valid_voxels} valid voxels for fitting.")

    if n_valid_voxels == 0:
        print("No voxels to fit. Returning empty parameter maps.")
        return {} 

    # Prepare initial orientation batch if map is provided
    initial_mu_orientations_flat = None
    if initial_orientation_map is not None:
        if initial_orientation_map.shape[:3] != dwi_data.shape[:3] or initial_orientation_map.shape[3] != 3:
            raise ValueError("initial_orientation_map must have shape (X, Y, Z, 3)")
        # Extract the orientations for the valid voxels
        initial_mu_orientations_flat = initial_orientation_map[valid_mask] # Shape (N_valid_voxels, 3)
        initial_mu_orientations_flat = torch.from_numpy(initial_mu_orientations_flat).float()
        print("Using initial orientation map for fitting.")

    # 2. Initialize NODDI model
    noddi_torch_model = NoddiModelTorch(gtab, d_intra=d_intra, d_iso=d_iso)
    noddi_torch_model.to(device)

    # 3. Prepare output maps
    img_shape = dwi_data.shape[:3]
    param_names = ['f_intra', 'f_iso', 'odi', 'kappa', 'mu_theta', 'mu_phi', 'f_extra']
    fitted_param_maps = {name: np.zeros(img_shape, dtype=np.float32) for name in param_names}
    fitted_param_maps['S0_fit'] = np.zeros(img_shape, dtype=np.float32)

    # 4. Iterate through batches of voxels
    print(f"Starting NODDI fitting in batches of {batch_size}...")
    for i in range(0, n_valid_voxels, batch_size):
        batch_start_idx = i
        batch_end_idx = min(i + batch_size, n_valid_voxels)
        
        batch_signals = norm_signals_torch[batch_start_idx:batch_end_idx, :].to(device)
        
        current_initial_mu_batch = None
        if initial_mu_orientations_flat is not None:
            current_initial_mu_batch = initial_mu_orientations_flat[batch_start_idx:batch_end_idx, :].to(device)

        try:
            # Fit the batch
            fitted_batch_params = noddi_torch_model.fit_batch(
                dwi_signals_normalized=batch_signals,
                initial_mu_batch=current_initial_mu_batch, # Pass initial orientations
                **fit_params
            )
            
            # Store fitted parameters back into 3D maps
            current_coords = valid_coords[batch_start_idx:batch_end_idx]
            for param_name in param_names:
                if param_name in fitted_batch_params:
                    param_values_cpu = fitted_batch_params[param_name].cpu().numpy()
                    fitted_param_maps[param_name][current_coords[:,0], current_coords[:,1], current_coords[:,2]] = param_values_cpu
            
            # Store S0 values for these voxels
            s0_batch_cpu = s0_values_torch[batch_start_idx:batch_end_idx].cpu().numpy()
            fitted_param_maps['S0_fit'][current_coords[:,0], current_coords[:,1], current_coords[:,2]] = s0_batch_cpu

        except Exception as e:
            print(f"Error fitting batch starting at voxel index {batch_start_idx}: {e}")
            # Optionally, fill with NaNs or skip, or re-raise
            # For now, problematic voxels in batch will remain zero in maps

    print("NODDI fitting completed.")
    return fitted_param_maps


if __name__ == '__main__':
    import time
    # --- Synthetic Data Generation ---
    img_dim = (10, 10, 3) # Small volume for testing
    n_voxels_total = np.prod(img_dim)
    
    # Gradient table
    n_grads_sim = 64
    bvals_sim = np.random.uniform(0, 2500, n_grads_sim).astype(np.float32)
    bvals_sim[0:5] = 0 # ~5 b0 images
    bvecs_sim = np.random.randn(n_grads_sim, 3).astype(np.float32)
    bvecs_sim[bvals_sim == 0] = 0 # Ensure b0 bvecs are [0,0,0]
    bvecs_norm_factor = np.linalg.norm(bvecs_sim[bvals_sim > 0], axis=1, keepdims=True) + EPS # Normalize only non-b0
    bvecs_sim[bvals_sim > 0] = bvecs_sim[bvals_sim > 0] / bvecs_norm_factor
    # PyTorchGradientTable will use its default b0_threshold (e.g., 50.0) if not specified.
    # For bvals_sim where b0s are exactly 0, this default is fine.
    gtab_sim = PyTorchGradientTable(bvals_sim, bvecs_sim) 

    # True NODDI parameters for the synthetic volume
    true_params_vol = {
        'f_intra': np.full(img_dim, 0.6, dtype=np.float32),
        'f_iso': np.full(img_dim, 0.15, dtype=np.float32),
        'kappa': np.full(img_dim, 2.5, dtype=np.float32), # ODI ~ 0.25
        'mu_theta': np.full(img_dim, np.pi / 3, dtype=np.float32), # 60 deg
        'mu_phi': np.full(img_dim, np.pi / 4, dtype=np.float32),   # 45 deg
    }
    # Add some variation to one parameter for testing
    true_params_vol['f_intra'][3:6, 3:6, 1] = 0.4
    true_params_vol['kappa'][3:6, 3:6, 1] = 1.0 # Higher ODI section

    # Generate synthetic 4D DWI data
    dwi_synthetic_flat = np.zeros((n_voxels_total, n_grads_sim), dtype=np.float32)
    
    # Convert 3D true param maps to flat arrays for noddi_signal_model
    flat_params_for_signal_gen = {}
    for key in ['f_intra', 'f_iso', 'kappa', 'mu_theta', 'mu_phi']:
        flat_params_for_signal_gen[key] = torch.from_numpy(true_params_vol[key].ravel())

    # Use noddi_signal.noddi_signal_model (requires torch tensors)
    # gtab_sim is now PyTorchGradientTable, so its .bvals and .bvecs are already tensors.
    b_values_torch = gtab_sim.bvals 
    b_vectors_torch = gtab_sim.bvecs

    print("Generating synthetic signals for volume...")
    with torch.no_grad():
        signals_norm_flat_torch = noddi_signal.noddi_signal_model(
            params=flat_params_for_signal_gen,
            b_values=b_values_torch,
            gradient_directions=b_vectors_torch,
            d_intra_val=noddi_signal.d_intra_default,
            d_iso_val=noddi_signal.d_iso_default
        )
    signals_norm_flat_np = signals_norm_flat_torch.numpy()

    # Assign S0 and add noise
    s0_gt_map = np.full(img_dim, 1500.0, dtype=np.float32)
    s0_gt_map[0:2, 0:2, 0] = 0 # Some invalid voxels due to low S0
    
    dwi_synthetic_flat = signals_norm_flat_np * s0_gt_map.ravel()[:, np.newaxis]
    
    # Add Rician noise (approximate with Gaussian for simplicity here if complex numbers are an issue)
    noise_sigma = 20 # SD of noise in real and imag channels
    # For Rician: sqrt( (S_true + N_real)^2 + N_imag^2 )
    # Simpler Gaussian noise on S_true for now:
    dwi_synthetic_flat_noisy = dwi_synthetic_flat + np.random.normal(0, noise_sigma, dwi_synthetic_flat.shape)
    dwi_synthetic_flat_noisy = np.clip(dwi_synthetic_flat_noisy, 0, None) # Ensure non-negative

    dwi_synthetic_4d = dwi_synthetic_flat_noisy.reshape(img_dim + (n_grads_sim,))
    print(f"Synthetic 4D DWI data shape: {dwi_synthetic_4d.shape}")

    # Create a brain mask (e.g., all valid S0 voxels)
    brain_mask_sim = s0_gt_map > 50 # Mask where S0 is reasonably high
    
    # --- Test fit_noddi_volume ---
    print("\nTesting fit_noddi_volume...")
    start_time = time.time()
    
    # Use a smaller batch size for small test volume
    test_batch_size = 16 
    test_fit_params = {'learning_rate': 0.02, 'n_iterations': 750} # Iterations might need tuning

    fitted_maps = fit_noddi_volume(
        dwi_data=dwi_synthetic_4d,
        gtab=gtab_sim, # This is now a PyTorchGradientTable instance
        mask=brain_mask_sim,
        min_s0_val=50.0, # Match mask criteria
        batch_size=test_batch_size,
        fit_params=test_fit_params,
        # device=torch.device('cpu') # Force CPU for testing if desired
        # b0_threshold argument is removed from fit_noddi_volume
    )
    end_time = time.time()
    print(f"Volume fitting took {end_time - start_time:.2f} seconds.")

    # --- Compare fitted maps to ground truth for a few voxels ---
    if fitted_maps:
        print("\n--- Comparison of True and Fitted Parameters (Sample Voxels) ---")
        # Choose some valid coordinates to check based on brain_mask_sim
        sample_coords = []
        valid_indices = np.array(np.where(brain_mask_sim)).T
        if len(valid_indices) > 0:
            for i in range(min(3, len(valid_indices))): # Check up to 3 sample voxels
                 sample_coords.append(tuple(valid_indices[np.random.choice(len(valid_indices))]))
        
        if not sample_coords: # If mask was too restrictive or all zero S0
            sample_coords = [(img_dim[0]//2, img_dim[1]//2, img_dim[2]//2)] # Fallback coordinate

        for coord in sample_coords:
            print(f"\nVoxel {coord}:")
            for param_name in ['f_intra', 'f_iso', 'odi', 'kappa', 'mu_theta', 'mu_phi']:
                true_val_str = "N/A"
                if param_name == 'odi':
                    true_kappa_val = true_params_vol['kappa'][coord]
                    true_val = (2.0 / np.pi) * np.arctan(1.0 / (true_kappa_val + EPS))
                    true_val_str = f"{true_val:.3f}"
                elif param_name in true_params_vol:
                    true_val_str = f"{true_params_vol[param_name][coord]:.3f}"
                
                fitted_val = fitted_maps[param_name][coord]
                print(f"  {param_name:<10}: True = {true_val_str}, Fitted = {fitted_val:.3f}")
    else:
        print("Fitting did not produce output maps.")

```
