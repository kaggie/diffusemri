import torch
import torch.nn.functional as F
import numpy as np # For patch extraction, can be replaced later if pure PyTorch is desired

# Small epsilon for numerical stability
MPPCA_EPS = 1e-8

def _extract_patches_4d(image_4d_torch: torch.Tensor, patch_radius: int) -> tuple[torch.Tensor, tuple]:
    """
    Extracts overlapping patches from a 4D image (X, Y, Z, N_gradients).
    A simpler implementation using looping for each spatial dimension.
    This version will be memory intensive for large images or large patch_radius.

    Args:
        image_4d_torch (torch.Tensor): The 4D input image (X, Y, Z, N_gradients).
        patch_radius (int): The radius of the patch (e.g., 1 means 3x3x3 patch).

    Returns:
        torch.Tensor: A tensor of patches. Shape: (N_patches, patch_size_x, patch_size_y, patch_size_z, N_gradients).
                      N_patches = X_out * Y_out * Z_out.
        tuple: Original spatial dimensions (X, Y, Z).
    """
    if image_4d_torch.ndim != 4:
        raise ValueError("Input image must be 4D.")
    
    X, Y, Z, N_gradients = image_4d_torch.shape
    patch_size = 2 * patch_radius + 1
    
    # Pad the image to handle borders
    # Amount of padding on each side is patch_radius
    padded_image = F.pad(image_4d_torch.permute(3, 0, 1, 2), 
                         (patch_radius, patch_radius, 
                          patch_radius, patch_radius, 
                          patch_radius, patch_radius), 
                         mode='reflect').permute(1, 2, 3, 0) # Back to X,Y,Z,N_grad

    patches_list = []
    # Loop through the center of each possible patch in the original image dimensions
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                # The coordinates are for the top-left-front corner of the patch in the *padded* image
                patch = padded_image[x : x + patch_size, 
                                     y : y + patch_size, 
                                     z : z + patch_size, :]
                patches_list.append(patch)
    
    # Stack all patches
    # N_patches = X * Y * Z
    # Each patch is (patch_size, patch_size, patch_size, N_gradients)
    if not patches_list: # Should not happen if X,Y,Z > 0
        # Create an empty tensor with the expected patch dimensions if input image is empty in spatial dims
        return torch.empty((0, patch_size, patch_size, patch_size, N_gradients), 
                           dtype=image_4d_torch.dtype, device=image_4d_torch.device), (X,Y,Z)

    patches_tensor = torch.stack(patches_list, dim=0)
    
    return patches_tensor, (X,Y,Z)


def _aggregate_patches_4d(
    denoised_patches: torch.Tensor, 
    original_spatial_dims: tuple[int, int, int], 
    patch_radius: int
) -> torch.Tensor:
    """
    Aggregates overlapping denoised patches back into a 4D image.
    Uses averaging for overlapping regions.
    This version assumes patches were extracted with a stride of 1.

    Args:
        denoised_patches (torch.Tensor): Tensor of denoised patches.
            Shape: (N_patches, patch_size_x, patch_size_y, patch_size_z, N_gradients).
            N_patches = X * Y * Z (original dimensions).
        original_spatial_dims (tuple[int, int, int]): Original spatial dimensions (X, Y, Z).
        patch_radius (int): The radius of the patches used.

    Returns:
        torch.Tensor: The reconstructed 4D image of shape (X, Y, Z, N_gradients).
    """
    X, Y, Z = original_spatial_dims
    if X == 0 or Y == 0 or Z == 0: # Handle empty original image
        return torch.empty((X,Y,Z, denoised_patches.shape[-1]), 
                            dtype=denoised_patches.dtype, device=denoised_patches.device)
        
    patch_size = 2 * patch_radius + 1
    N_gradients = denoised_patches.shape[-1]
    
    # Initialize output image and a counter for averaging overlaps
    # Padded size for accumulation
    padded_X, padded_Y, padded_Z = X + 2 * patch_radius, Y + 2 * patch_radius, Z + 2 * patch_radius
    
    output_image = torch.zeros((padded_X, padded_Y, padded_Z, N_gradients), 
                               dtype=denoised_patches.dtype, 
                               device=denoised_patches.device)
    overlap_counts = torch.zeros((padded_X, padded_Y, padded_Z, 1), # Count per voxel, not per gradient
                                 dtype=torch.float32, 
                                 device=denoised_patches.device)

    patch_idx = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if patch_idx < denoised_patches.shape[0]:
                    output_image[x : x + patch_size, 
                                 y : y + patch_size, 
                                 z : z + patch_size, :] += denoised_patches[patch_idx]
                    overlap_counts[x : x + patch_size, 
                                   y : y + patch_size, 
                                   z : z + patch_size, :] += 1.0
                patch_idx += 1
    
    # Avoid division by zero where overlap_counts is zero
    overlap_counts[overlap_counts == 0] = 1.0 # Should affect only areas outside any patch center
    
    reconstructed_image_padded = output_image / overlap_counts
    
    # Crop back to original dimensions (by removing the padding)
    reconstructed_image = reconstructed_image_padded[patch_radius : X + patch_radius,
                                                     patch_radius : Y + patch_radius,
                                                     patch_radius : Z + patch_radius, :]
    return reconstructed_image


def _denoise_patch_mppca(patch_4d_torch: torch.Tensor, n_gradients: int) -> torch.Tensor:
    """
    Denoises a single 4D patch using Marchenko-Pastur PCA logic.

    The method involves centering the patch data (voxel signals across gradients),
    performing SVD, estimating noise variance from the singular value spectrum,
    calculating an MP-based threshold, applying hard thresholding to the
    singular values (effectively keeping components with eigenvalues above this
    threshold), and then reconstructing the patch.

    Args:
        patch_4d_torch (torch.Tensor): A single patch of shape 
                                     (patch_size_x, patch_size_y, patch_size_z, N_gradients).
        n_gradients (int): Number of gradients (Ng), the last dimension of the patch.

    Returns:
        torch.Tensor: The denoised patch, with the same shape as the input.

    Algorithm Details:
    1. Reshapes the patch to (P, Ng) where P is num_voxels_in_patch.
    2. Centers data by subtracting row-wise means (original means stored).
    3. Computes SVD: U, S_singular_values, Vh = svd(centered_patch).
    4. Estimates noise variance `sigma_sq_est` using the formula:
       `sigma_sq_est = torch.median(S_singular_values**2) / torch.sqrt(P/Ng)`.
       (Note: This specific estimator is based on user-provided interpretation).
    5. Calculates the MP eigenvalue threshold `lambda_mp_thresh` using:
       `lambda_mp_thresh = sigma_sq_est * (1 + torch.sqrt(P/Ng))**2`.
    6. Applies hard thresholding: Squared singular values <= `lambda_mp_thresh`
       (and their corresponding singular values) are set to zero.
    7. Reconstructs the patch with the thresholded singular values.
    8. Adds back the stored row-wise means.
    9. Reshapes the patch to its original 4D shape.
    Includes error handling for SVD failures or invalid singular values.
    """
    patch_shape = patch_4d_torch.shape
    P = patch_shape[0] * patch_shape[1] * patch_shape[2] # Number of voxels in patch
    Ng = n_gradients # Number of measurements / gradients

    if P == 0 or Ng == 0:
        return patch_4d_torch # Empty patch or no gradients

    # Reshape patch to (P, Ng)
    patch_matrix = patch_4d_torch.reshape(P, Ng)

    # Center data by subtracting row-wise mean (mean signal for each voxel over gradients)
    row_means = torch.mean(patch_matrix, dim=1, keepdim=True)
    patch_centered = patch_matrix - row_means

    try:
        # Perform SVD on the centered patch matrix
        # U: (P, K), S_singular_values: (K,), Vh: (K, Ng), where K = min(P, Ng)
        U, S_singular_values, Vh = torch.linalg.svd(patch_centered, full_matrices=False)
    except torch.linalg.LinAlgError:
        # SVD failed, return original (uncentered) patch
        return patch_4d_torch 
    
    if torch.any(torch.isnan(S_singular_values)) or torch.any(torch.isinf(S_singular_values)):
        # Invalid singular values, return original (uncentered) patch
        return patch_4d_torch

    # Eigenvalues of the covariance matrix are singular_values squared (scaled by P or Ng)
    # For MP law, we consider eigenvalues of X^T X / P or X X^T / Ng
    # Let eigenvalues_observed = S_singular_values**2 / P (if P < Ng, related to X X^T)
    # or eigenvalues_observed = S_singular_values**2 / Ng (if Ng < P, related to X^T X)
    # The choice of scaling affects sigma_sq_est interpretation.
    # For simplicity, let's work with squared singular values directly S_sq = S_singular_values**2.

    S_sq = S_singular_values**2

    # Estimate noise variance (sigma_sq_est)
    # Based on user's formula: sigma_sq_est = median(Sigma^2) / sqrt(gamma)
    # where Sigma^2 are eigenvalues (S_sq here), and gamma = P / Ng.
    # This formula was for 'sigma' in `lambda_MP = sigma^2 * (1 + sqrt(gamma))^2`
    # So, `sigma_sq_est_for_formula = median_of_eigenvalues / sqrt(gamma_aspect_ratio)`
    
    if P == 0 or Ng == 0: # Should have been caught earlier, but for safety
        return patch_4d_torch
        
    gamma = P / Ng # Aspect ratio

    if gamma < MPPCA_EPS: # Avoid division by zero if Ng is huge or P is zero
        return patch_4d_torch

    # sigma_sq_estimator_val is the term that plays role of sigma^2 in MP formula
    # User: sigma_sq_estimator_val = median(S_sq) / torch.sqrt(torch.tensor(gamma, device=S_sq.device))
    # This estimates the variance of the underlying noise elements if the matrix was pure noise.
    # Let's verify the typical use of median of eigenvalues for noise variance.
    # If K = min(P,Ng), and if P > Ng (more voxels than gradients), then there are P-Ng zero eigenvalues
    # for X X^T if data is rank Ng.
    # Noise estimation is critical. A common approach: if M > N, sigma^2 is the median of the
    # last M-N eigenvalues of the sample covariance matrix Y = (1/N) X X^T.
    # Veraart et al. (2016) use a specific way to determine the number of noise eigenvalues
    # and then average them.
    # Given the user's formula: sigma_sq_est = median(S_sq) / sqrt(gamma)
    # This is the `sigma^2` term in their MP threshold formula.
    
    if S_sq.numel() == 0: # No singular values (e.g. P=0 or Ng=0)
        return patch_4d_torch
        
    # Ensure gamma is a tensor for sqrt
    gamma_tensor = torch.tensor(gamma, device=S_sq.device, dtype=S_sq.dtype)
    if torch.sqrt(gamma_tensor) < MPPCA_EPS : # Avoid division by zero from sqrt(gamma)
        sigma_sq_est = torch.median(S_sq) # Fallback if gamma is tiny or P is very small relative to Ng
    else:
        sigma_sq_est = torch.median(S_sq) / torch.sqrt(gamma_tensor)


    # Marchenko-Pastur critical threshold for eigenvalues
    # lambda_MP_max_noise_eval = sigma_sq_est * (1 + torch.sqrt(gamma_tensor))**2
    # The user provided this as `sigma^2 * (1 + sqrt(gamma))^2` where `sigma^2` was the `median(Sigma^2)/sqrt(gamma)` term
    # So, lambda_MP_max_noise_eval = sigma_sq_est * (1 + torch.sqrt(gamma_tensor))**2
    # This means sigma_sq_est is the variance of the matrix elements if it were white noise.
    # The eigenvalues of such a matrix (scaled by 1/N or 1/M) would follow MP dist.
    # Let's assume sigma_sq_est is indeed the variance of the noise in the elements of X_centered.
    # The eigenvalues of Cov(X_centered) = (1/Ng) X_centered^T X_centered are S_sq/Ng.
    # These eigenvalues are what MP law describes.
    # So, if lambda_i = s_i^2/Ng, then lambda_i_noise_max_bound = sigma_elementwise_sq * (1+sqrt(P/Ng))^2 if P/Ng is used as beta for (1/P) X X^T
    # Or sigma_elementwise_sq * (1+sqrt(Ng/P))^2 if Ng/P is used as beta for (1/Ng) X^T X.
    # This is getting confusing. Let's simplify based on Dipy's approach or common RMT application:
    # 1. Estimate element-wise noise std dev `sigma_e`. `sigma_e = torch.median(S_singular_values) / sqrt(max(P,Ng))`. (Heuristic for Gaussian noise)
    # 2. `sigma_e_sq = sigma_e**2`.
    # 3. Critical MP eigenvalue: `lambda_plus = sigma_e_sq * (1 + torch.sqrt(torch.tensor(min(P,Ng)/max(P,Ng), device=S_sq.device)))**2`.
    # 4. Threshold: Keep `s_i` if `s_i**2 / max(P,Ng) > lambda_plus`. (Scaling S_singular_values to eigenvalues).

    # Let's try to stick to the user's most recent formulation of lambda_MP based on their sigma estimation.
    # `sigma_sq_est = median(S_sq) / sqrt(gamma)` where `gamma = P/Ng`.
    # `lambda_MP = sigma_sq_est * (1 + sqrt(gamma))^2`.
    # This `lambda_MP` is a threshold for *squared singular values* (eigenvalues).
    
    lambda_mp_threshold = sigma_sq_est * (1 + torch.sqrt(gamma_tensor))**2
    
    S_denoised_sq = S_sq.clone()
    S_denoised_sq[S_sq <= lambda_mp_threshold] = 0.0
    
    S_denoised = torch.sqrt(S_denoised_sq) # Get back to singular values

    # Reconstruct the patch
    if S_denoised.numel() > 0:
        denoised_patch_centered = U @ torch.diag(S_denoised) @ Vh
    else:
        denoised_patch_centered = torch.zeros_like(patch_centered)

    # Add back the mean
    denoised_patch = denoised_patch_centered + row_means
    
    # Reshape back to original patch dimensions
    return denoised_patch.reshape(patch_shape)


def pytorch_mppca(
    image_4d_torch: torch.Tensor, 
    patch_radius: int = 2,
    verbose: bool = False
) -> torch.Tensor:
    """
    Denoises a 4D dMRI image using Marchenko-Pastur Principal Component Analysis (MP-PCA).
    This implementation processes the image by extracting overlapping patches,
    denoising each patch using MP-PCA, and then aggregating the results.

    Note: The core MP-PCA logic in `_denoise_patch_mppca` implements a
    Marchenko-Pastur PCA based denoising algorithm using Random Matrix Theory
    principles for noise estimation and singular value thresholding. Further
    refinements or validation against specific literature models (e.g., Veraart et al., 2016)
    may be beneficial for optimal performance.

    Args:
        image_4d_torch (torch.Tensor): The 4D dMRI data (X, Y, Z, N_gradients) as a PyTorch tensor.
        patch_radius (int, optional): Radius of the cubic patch for local PCA.
                                      E.g., patch_radius=1 means a 3x3x3 patch. Defaults to 2 (5x5x5 patch).
        verbose (bool, optional): If True, prints progress information. Defaults to False.

    Returns:
        torch.Tensor: The denoised 4D dMRI data as a PyTorch tensor.
    """
    if not isinstance(image_4d_torch, torch.Tensor):
        raise TypeError("Input image_data must be a PyTorch tensor.")
    if image_4d_torch.ndim != 4:
        raise ValueError("Input image_data must be a 4D tensor.")
    if patch_radius < 0: # patch_radius = 0 means 1x1x1 patch (single voxel)
        raise ValueError("patch_radius must be non-negative.")

    device = image_4d_torch.device
    X, Y, Z, N_gradients = image_4d_torch.shape
    
    if X == 0 or Y == 0 or Z == 0: # Handle empty image
        return image_4d_torch.clone()

    if verbose:
        print(f"Starting MP-PCA denoising for image of shape {image_4d_torch.shape} with patch_radius {patch_radius}.")
        print(f"Using device: {device}")

    # 1. Extract patches
    if verbose: print("Extracting patches...")
    patches, orig_spatial_dims = _extract_patches_4d(image_4d_torch, patch_radius)
    num_patches = patches.shape[0]
    
    if num_patches == 0 : # No patches extracted (e.g. if image dims are smaller than patch size effectively)
        if verbose: print("No patches extracted, returning original image.")
        return image_4d_torch.clone()

    if verbose: print(f"Extracted {num_patches} patches of shape {patches.shape[1:]}")

    # 2. Denoise each patch
    denoised_patches_list = []
    for i in range(num_patches):
        if verbose and num_patches > 10 and (i + 1) % (max(1, num_patches // 10)) == 0:
            print(f"Denoising patch {i+1}/{num_patches}...")
        
        current_patch = patches[i].to(device) # Ensure patch is on correct device
        denoised_patch = _denoise_patch_mppca(current_patch, N_gradients)
        denoised_patches_list.append(denoised_patch.cpu()) # Collect on CPU

    denoised_patches_tensor = torch.stack(denoised_patches_list, dim=0).to(device)

    # 3. Aggregate patches
    if verbose: print("Aggregating denoised patches...")
    denoised_image = _aggregate_patches_4d(denoised_patches_tensor, orig_spatial_dims, patch_radius)
    
    if verbose: print("MP-PCA denoising completed.")
    return denoised_image.to(device)


if __name__ == '__main__':
    print("--- PyTorch MP-PCA Denoising Example ---")
    
    # Create a synthetic 4D volume
    shape_4d = (10, 10, 8, 20) # Small volume: X, Y, Z, N_gradients
    
    base_signal = torch.rand(shape_4d) * 50 + 100 
    base_signal[..., :3] = torch.rand(shape_4d[:-1] + (3,)) * 100 + 200 
    
    noise_sigma = 15.0
    noisy_dwi_data = base_signal + torch.randn_like(base_signal) * noise_sigma
    noisy_dwi_data = noisy_dwi_data.clamp(min=0) 

    print(f"Synthetic noisy DWI data created with shape: {noisy_dwi_data.shape}")
    print(f"Noisy data stats: min={noisy_dwi_data.min():.2f}, max={noisy_dwi_data.max():.2f}, mean={noisy_dwi_data.mean():.2f}")

    print("\nApplying MP-PCA (placeholder)...")
    try:
        denoised_dwi_data = pytorch_mppca(noisy_dwi_data, patch_radius=1, verbose=True)
        
        print(f"\nDenoised DWI data shape: {denoised_dwi_data.shape}")
        print(f"Denoised data stats: min={denoised_dwi_data.min():.2f}, max={denoised_dwi_data.max():.2f}, mean={denoised_dwi_data.mean():.2f}")

        assert denoised_dwi_data.shape == noisy_dwi_data.shape
        
        print("\nMP-PCA (placeholder) example executed successfully.")

    except Exception as e:
        print(f"Error during MP-PCA (placeholder) example: {e}")
        raise
