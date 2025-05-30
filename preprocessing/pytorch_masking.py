import torch
import numpy as np
from scipy.ndimage import median_filter # For median filtering step

# Small epsilon for numerical stability in Otsu's method if needed
EPS_OTSU = 1e-8

def pytorch_otsu_threshold(image_tensor: torch.Tensor) -> int:
    """
    Computes Otsu's threshold for a 2D or 3D PyTorch tensor (intensities).
    Assumes the input tensor contains non-negative values, typically representing
    image intensities, scaled to a reasonable range (e.g., 0-255 or 0-1).

    Args:
        image_tensor (torch.Tensor): A 2D or 3D PyTorch tensor.
                                     Must be convertible to float for calculations.

    Returns:
        int: The Otsu threshold value. Returns 0 if the image is flat.
    """
    if image_tensor.numel() == 0:
        return 0 # Or raise error

    # Ensure tensor is float for calculations and on CPU
    img_flat = image_tensor.view(-1).float().cpu()
    
    min_val_tensor = img_flat.min()
    max_val_tensor = img_flat.max()

    min_val = min_val_tensor.item()
    max_val = max_val_tensor.item()


    if abs(min_val - max_val) < EPS_OTSU: # Flat image, no threshold can be found
        return int(min_val) # Or 0, or max_val. Convention might vary.

    # Normalize to 0-255 range for histogram calculation, common for Otsu.
    img_norm = (img_flat - min_val_tensor) / (max_val_tensor - min_val_tensor) * 255.0
    
    # Calculate histogram
    # Ensure bins cover the 0-255 range. torch.histc needs min/max for range.
    # Using 256 bins for values 0 through 255.
    hist = torch.histc(img_norm, bins=256, min=0, max=255)
    hist_norm = hist.float() / hist.sum() # Normalized histogram

    # Cumulative sums
    omega = torch.cumsum(hist_norm, dim=0) # Cumulative sum of probabilities (weights)
    # Ensure arange is on the same device as hist (CPU)
    mu_t = torch.cumsum(hist_norm * torch.arange(0, 256, device=hist.device).float(), dim=0) 

    mu_total = mu_t[-1] # Total mean of the image

    # Calculate between-class variance for all possible thresholds
    # omega can be 0 or 1 for some thresholds, leading to division by zero if not handled.
    # Add EPS_OTSU to denominators to prevent this.
    denominator = omega * (1 - omega)
    sigma_b_squared = torch.zeros_like(denominator)
    
    # Calculate variance only for valid omega values
    valid_omega_mask = denominator > EPS_OTSU
    sigma_b_squared[valid_omega_mask] = \
        (mu_total * omega[valid_omega_mask] - mu_t[valid_omega_mask])**2 / denominator[valid_omega_mask]
    
    # Find threshold that maximizes between-class variance
    optimal_thresh_norm = torch.argmax(sigma_b_squared).item()

    # Denormalize threshold back to original image scale
    optimal_thresh = int(min_val + (optimal_thresh_norm / 255.0) * (max_val - min_val))
            
    return optimal_thresh


def pytorch_median_otsu(
    image_data: torch.Tensor, 
    median_radius: int = 4, 
    numpass: int = 4
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a brain mask from a 3D volume using a median filter followed by
    Otsu's thresholding, implemented primarily with PyTorch and SciPy for median filter.

    Args:
        image_data (torch.Tensor): The 3D input data (e.g., a mean DWI volume).
                                   Expected to be a PyTorch tensor.
        median_radius (int, optional): Radius (in voxels) of the applied median filter.
                                       Defaults to 4.
        numpass (int, optional): Number of passes of the median filter. Defaults to 4.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - brain_mask (torch.Tensor): The 3D binary brain mask (boolean PyTorch tensor).
            - masked_image_data (torch.Tensor): The input image_data after applying the
                                                brain_mask (PyTorch tensor).
    """
    if not isinstance(image_data, torch.Tensor):
        raise TypeError("Input image_data must be a PyTorch tensor.")
    if image_data.ndim != 3:
        raise ValueError("Input image_data must be a 3D tensor.")

    device = image_data.device # Keep track of original device

    # 1. Median Filtering (using SciPy as a placeholder for dipy.segment.mask.median_filter behavior)
    # Convert to NumPy for SciPy's median_filter
    volume_np = image_data.cpu().numpy()
    
    # Apply median filter multiple times (numpass)
    # SciPy's median_filter footprint size is 2*radius + 1
    footprint_size = 2 * median_radius + 1
    filtered_volume_np = volume_np
    for _ in range(numpass):
        filtered_volume_np = median_filter(filtered_volume_np, size=footprint_size)
    
    filtered_volume_torch = torch.from_numpy(filtered_volume_np).to(device)

    # 2. Otsu Thresholding
    # pytorch_otsu_threshold expects a 2D or 3D tensor of intensities.
    threshold_value = pytorch_otsu_threshold(filtered_volume_torch)
    
    # 3. Create Mask
    brain_mask = filtered_volume_torch > threshold_value
    
    # 4. Apply Mask to original image_data
    masked_image_data = image_data * brain_mask 

    return brain_mask.bool(), masked_image_data


if __name__ == '__main__':
    # Example Usage
    print("--- PyTorch Median Otsu Example ---")
    
    # Create a synthetic 3D volume (e.g., sphere in noise)
    shape = (64, 64, 64)
    center = tuple(s // 2 for s in shape)
    radius = shape[0] // 4
    
    # Create coordinates
    coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing='ij')
    
    # Create sphere
    sphere_mask = sum([(coords[i] - center[i])**2 for i in range(3)]) < radius**2
    synthetic_volume = torch.zeros(shape, dtype=torch.float32)
    synthetic_volume[sphere_mask] = 150.0 # Sphere intensity
    synthetic_volume[~sphere_mask] = 50.0  # Background intensity
    
    # Add some noise
    synthetic_volume += torch.randn_like(synthetic_volume) * 20.0
    synthetic_volume = synthetic_volume.clamp(0, 255) # Clamp to typical intensity range

    print(f"Synthetic volume created with shape: {synthetic_volume.shape}")
    print(f"Min intensity: {synthetic_volume.min():.2f}, Max intensity: {synthetic_volume.max():.2f}")

    # Apply median_otsu
    try:
        brain_mask_torch, masked_volume_torch = pytorch_median_otsu(
            synthetic_volume, 
            median_radius=3, # Smaller radius for faster example
            numpass=2        # Fewer passes for faster example
        )
        
        print(f"Output brain_mask shape: {brain_mask_torch.shape}, dtype: {brain_mask_torch.dtype}")
        print(f"Output masked_volume shape: {masked_volume_torch.shape}, dtype: {masked_volume_torch.dtype}")
        print(f"Number of True voxels in mask: {torch.sum(brain_mask_torch).item()}")
        
        # Basic check: mask should not be all True or all False for this example
        assert torch.sum(brain_mask_torch).item() > 0, "Mask is all False"
        assert torch.sum(brain_mask_torch).item() < brain_mask_torch.numel(), "Mask is all True"

        # Check if masked volume has zeros where mask is False
        # (or original value * 0, which is 0 for float types)
        if torch.sum(~brain_mask_torch).item() > 0 : # If there are False values in mask
             assert torch.all(masked_volume_torch[~brain_mask_torch].abs() < EPS_OTSU), \
                    "Masked volume not zeroed out where mask is False"

        print("pytorch_median_otsu example executed successfully.")

    except Exception as e:
        print(f"Error during pytorch_median_otsu example: {e}")
        raise
```
