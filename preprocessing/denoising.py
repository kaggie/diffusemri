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


import nibabel as nib
import os
import logging
from dipy.denoise.gibbs import gibbs_removal

logger_gibbs = logging.getLogger(__name__) # Use a specific logger if desired, or root logger

def correct_gibbs_ringing_dipy(
    input_image_file: str,
    output_corrected_file: str,
    slice_axis: int = 2,
    n_points: int = 3,
    num_processes: Optional[int] = 1, # Allow None for auto-detection by dipy
    **kwargs: Any
) -> str:
    """
    Corrects Gibbs ringing artifacts in an image using Dipý's `gibbs_removal`.

    Gibbs ringing artifacts appear as spurious oscillations near sharp intensity
    transitions in MR images, often due to truncation of k-space data.

    Args:
        input_image_file (str): Path to the NIFTI file to be corrected.
            Can be 3D or 4D. If 4D, correction is applied volume-wise.
        output_corrected_file (str): Path where the corrected NIFTI file will be saved.
        slice_axis (int, optional): Axis along which slices were acquired (0, 1, or 2).
            This is crucial for the algorithm to correctly identify the direction
            of ringing. Defaults to 2 (axial).
        n_points (int, optional): Number of neighboring points on each side of a voxel
            to access for local Total Variation (TV) calculation. Default is 3.
        num_processes (Optional[int], optional): Number of processes to use for parallel
            computation. If None, Dipý attempts to determine the number of
            available CPUs. If 1, runs single-threaded. Defaults to 1.
        **kwargs: Additional keyword arguments. `gibbs_removal` itself has few other
                  parameters (`inplace` is forced to False). This is for future flexibility.

    Returns:
        str: Path to the corrected image file (`output_corrected_file`).

    Raises:
        FileNotFoundError: If `input_image_file` does not exist.
        RuntimeError: If the `gibbs_removal` process fails or if saving the
                      output image fails.
    """
    if not os.path.exists(input_image_file):
        raise FileNotFoundError(f"Input image file not found: {input_image_file}")

    logger_gibbs.info(f"Starting Gibbs ringing correction for: {input_image_file}")
    logger_gibbs.info(f"  Slice axis: {slice_axis}, N points: {n_points}, Num processes: {num_processes}")
    logger_gibbs.info(f"  Output will be saved to: {output_corrected_file}")

    try:
        img = nib.load(input_image_file)
        data = img.get_fdata() # Load data as is, usually float64 by nibabel
        affine = img.affine
        header = img.header
    except Exception as e:
        logger_gibbs.error(f"Failed to load NIFTI image from {input_image_file}: {e}")
        raise RuntimeError(f"Failed to load NIFTI image: {e}")

    # Ensure data is float for gibbs_removal if it requires it (Dipy usually handles this)
    # gibbs_removal typically expects float data.
    if not np.issubdtype(data.dtype, np.floating):
        logger_gibbs.info(f"Input data dtype is {data.dtype}, converting to float32 for Gibbs removal.")
        data = data.astype(np.float32)


    logger_gibbs.info("Applying Dipý's gibbs_removal...")
    try:
        # Ensure num_processes is handled correctly (None or int)
        if num_processes is not None and num_processes < 1:
            logger_gibbs.warning(f"num_processes ({num_processes}) is invalid, setting to 1.")
            num_processes = 1

        # Call gibbs_removal, ensuring inplace is False
        corrected_data = gibbs_removal(
            data,
            slice_axis=slice_axis,
            n_points=n_points,
            num_processes=num_processes,
            inplace=False # Explicitly ensure data is copied and modified
        )
    except Exception as e:
        logger_gibbs.error(f"Error during Dipý's gibbs_removal: {e}")
        raise RuntimeError(f"Dipý gibbs_removal failed: {e}")

    logger_gibbs.info("Gibbs ringing correction complete. Saving corrected image...")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_corrected_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger_gibbs.info(f"Created output directory: {output_dir}")

        corrected_img = nib.Nifti1Image(corrected_data, affine, header=header)
        nib.save(corrected_img, output_corrected_file)
        logger_gibbs.info(f"Corrected image saved to: {output_corrected_file}")
    except Exception as e:
        logger_gibbs.error(f"Failed to save corrected NIFTI image to {output_corrected_file}: {e}")
        raise RuntimeError(f"Failed to save corrected image: {e}")

    return output_corrected_file
