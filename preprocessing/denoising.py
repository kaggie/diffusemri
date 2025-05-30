import numpy as np
import torch # Added import
from .pytorch_denoising import pytorch_mppca # New import

def denoise_mppca_data(dmri_data: np.ndarray, patch_radius: int = 3) -> np.ndarray:
    """
    Denoises 4D dMRI data using a PyTorch-based implementation of Marchenko-Pastur 
    Principal Component Analysis (MP-PCA).

    Parameters
    ----------
    dmri_data : np.ndarray
        The 4D dMRI data array (x, y, z, g) to be denoised.
    patch_radius : int, optional
        The radius of the local patch to be used for PCA (e.g., radius 3 means 7x7x7 patch,
        as pytorch_mppca uses patch_size = 2*radius+1). Default is 3.

    Returns
    -------
    denoised_data : np.ndarray
        The denoised 4D dMRI data array.
    """
    if not isinstance(dmri_data, np.ndarray) or dmri_data.ndim != 4:
        raise ValueError("dMRI data must be a 4D NumPy array.")

    # Convert NumPy array to PyTorch tensor
    # Ensure data is float32 for processing, common for dMRI.
    dmri_data_torch = torch.from_numpy(dmri_data.astype(np.float32)).float()

    # Call the PyTorch MP-PCA implementation
    # The patch_radius in pytorch_mppca has a default of 2.
    # Here, we pass the patch_radius from this function's signature (defaulting to 3).
    denoised_data_torch = pytorch_mppca(
        image_4d_torch=dmri_data_torch,
        patch_radius=patch_radius
        # verbose=False # Can be exposed in this function's signature if needed
    )

    # Convert the result back to a NumPy array
    denoised_data_np = denoised_data_torch.cpu().numpy()
    
    return denoised_data_np
