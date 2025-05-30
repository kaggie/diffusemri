import numpy as np
import torch # Added import
from .pytorch_masking import pytorch_median_otsu # New import

def create_brain_mask(dmri_data, voxel_size, median_radius=4, numpass=4):
    """
    Computes a brain mask from dMRI data using the median_otsu method
    and applies the mask to the dMRI data.

    Parameters
    ----------
    dmri_data : ndarray
        The 4D dMRI data array (x, y, z, g).
    voxel_size : tuple or list
        The voxel size in mm (e.g., (2.0, 2.0, 2.0)).
    median_radius : int, optional
        Radius (in voxels) of the applied median filter. Default is 4.
    numpass : int, optional
        Number of passes of the median filter. Default is 4.

    Returns
    -------
    brain_mask : ndarray
        The 3D binary brain mask (boolean).
    masked_dmri_data : ndarray
        The 4D dMRI data array with the mask applied (non-brain voxels zeroed out).
    """
    if dmri_data.ndim != 4:
        raise ValueError("dMRI data must be a 4D array.")

    # We need a 3D volume for median_otsu.
    # If there are multiple volumes, we can average them or use the first one.
    # Averaging is generally more robust if multiple b0s are present
    # or if it's mean DWI data.
    # For this example, we'll average across the last dimension (gradients/volumes).
    # If specific b0 volumes are known, they should ideally be used.
    mean_dwi_data = np.mean(dmri_data, axis=3)

    # Compute the brain mask using median_otsu
    # The median_otsu function in dipy expects vol_idx=None if data is already 3D.
    # It also takes voxel_size directly if available.
    # However, the standard median_otsu call is (data, median_radius, numpass)
    # and for newer versions, it might be (data, voxel_size, median_radius, numpass, dilate)
    # Let's assume a version of dipy where voxel_size is not directly passed to median_otsu
    # or it's handled internally if data has affine. Voxel_size is not used by
    # pytorch_median_otsu either, but kept in signature for context.

    # Convert mean_dwi_data to PyTorch tensor
    mean_dwi_data_torch = torch.from_numpy(mean_dwi_data).float()

    # Call the new PyTorch-based median_otsu
    # It returns (brain_mask_torch, masked_mean_volume_torch)
    # We are interested in the brain_mask_torch to apply to the original 4D data.
    brain_mask_torch, _ = pytorch_median_otsu(
        mean_dwi_data_torch,
        median_radius=median_radius,
        numpass=numpass
    )

    # Convert the resulting mask back to NumPy boolean array
    brain_mask_np = brain_mask_torch.cpu().numpy().astype(bool)

    # Apply the 3D mask to the 4D dMRI data (original logic)
    # The brain_mask_np is 3D (x, y, z). We need to apply it to each volume in the 4D data.
    # We can do this by adding a new axis to brain_mask_np for broadcasting.
    masked_dmri_data = dmri_data * brain_mask_np[..., np.newaxis]

    return brain_mask_np, masked_dmri_data
