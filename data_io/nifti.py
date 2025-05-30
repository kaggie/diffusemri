import numpy as np
import nibabel as nib
import os

def load_nifti_dwi(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads Diffusion-Weighted Imaging (DWI) data from a NIfTI file.

    Parameters
    ----------
    filepath : str
        Path to the NIfTI file containing DWI data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - data (np.ndarray): The 4D DWI data array (X, Y, Z, N_volumes), as float32.
        - affine (np.ndarray): The 4x4 affine transformation matrix.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    nibabel.filebasedimages.ImageFileError
        If Nibabel cannot load the file (e.g., not a NIfTI or corrupted).
    ValueError
        If the loaded data is not 4-dimensional.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIfTI DWI file not found at: {filepath}")
    
    try:
        img = nib.load(filepath)
        data = img.get_fdata(dtype=np.float32)
        affine = img.affine
    except nib.filebasedimages.ImageFileError as e:
        raise nib.filebasedimages.ImageFileError(f"Failed to load NIfTI file at {filepath}: {e}")
    except Exception as e: # Catch other potential nibabel errors during loading
        raise RuntimeError(f"An unexpected error occurred while loading {filepath}: {e}")

    if data.ndim != 4:
        raise ValueError(f"Expected 4D DWI data, but got {data.ndim}D data from {filepath}.")
    
    return data, affine

def load_nifti_mask(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a brain mask or general mask from a NIfTI file.

    Parameters
    ----------
    filepath : str
        Path to the NIfTI file containing the mask.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - mask_data (np.ndarray): The 3D boolean mask array (X, Y, Z).
        - affine (np.ndarray): The 4x4 affine transformation matrix.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    nibabel.filebasedimages.ImageFileError
        If Nibabel cannot load the file.
    ValueError
        If the loaded data is not 3-dimensional.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIfTI mask file not found at: {filepath}")

    try:
        img = nib.load(filepath)
        # Dtype bool will cast any non-zero values to True, and zero to False.
        mask_data = img.get_fdata(dtype=bool) 
        affine = img.affine
    except nib.filebasedimages.ImageFileError as e:
        raise nib.filebasedimages.ImageFileError(f"Failed to load NIfTI file at {filepath}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading {filepath}: {e}")

    if mask_data.ndim != 3:
        raise ValueError(f"Expected 3D mask data, but got {mask_data.ndim}D data from {filepath}.")
        
    return mask_data, affine

def load_fsl_bvecs(filepath: str) -> np.ndarray:
    """
    Loads b-vectors from an FSL-formatted text file.

    FSL b-vectors can be space or comma-separated. This function expects
    the file to contain either 3 rows (X, Y, Z) and N columns (volumes),
    or N rows and 3 columns. It will always return an Nx3 array.

    Parameters
    ----------
    filepath : str
        Path to the FSL bvec file.

    Returns
    -------
    np.ndarray
        An Nx3 NumPy array of b-vectors.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the loaded b-vectors do not conform to 3xN or Nx3 shape,
        or if the file is empty or malformed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FSL bvec file not found at: {filepath}")

    try:
        bvecs = np.loadtxt(filepath)
        if bvecs.size == 0:
            raise ValueError(f"bvec file is empty: {filepath}")
    except Exception as e: # Catches errors from loadtxt (e.g. malformed, not numbers)
        raise ValueError(f"Failed to load or parse bvec file {filepath}: {e}")

    # Check shape and transpose if necessary (3xN -> Nx3)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    elif bvecs.shape[1] == 3 and bvecs.shape[0] != 3:
        pass # Already Nx3
    elif bvecs.shape[0] == 3 and bvecs.shape[1] == 3:
        # Ambiguous 3x3 case. Assume it's Nx3 (N=3).
        pass
    # Check if it's a single b-vector (1x3 or 3x1)
    elif bvecs.ndim == 1 and bvecs.shape[0] == 3: # Single vector loaded as 1D array of 3 elements
        bvecs = bvecs.reshape(1, 3) # Reshape to 1x3
    else: # Other shapes are invalid
        raise ValueError(
            f"b-vectors in {filepath} must be 3xN or Nx3 (or a single 1x3/3x1 vector). Got shape {bvecs.shape}."
        )

    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        # This secondary check ensures the final shape is Nx3
        raise ValueError(
            f"Processed b-vectors must be Nx3. Got shape {bvecs.shape} from {filepath}."
        )
        
    return bvecs

def load_fsl_bvals(filepath: str) -> np.ndarray:
    """
    Loads b-values from an FSL-formatted text file.

    FSL b-values are typically in a single row or column, space or comma-separated.
    This function returns a 1D NumPy array.

    Parameters
    ----------
    filepath : str
        Path to the FSL bval file.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of b-values.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the loaded b-values cannot be coerced into a 1D array,
        or if the file is empty or malformed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FSL bval file not found at: {filepath}")

    try:
        bvals = np.loadtxt(filepath)
        if bvals.size == 0:
            raise ValueError(f"bval file is empty: {filepath}")
    except Exception as e:
        raise ValueError(f"Failed to load or parse bval file {filepath}: {e}")

    bvals = bvals.squeeze() 
    if bvals.ndim == 0: 
        bvals = bvals.reshape(1,)
    elif bvals.ndim != 1:
        raise ValueError(
            f"b-values in {filepath} could not be converted to a 1D array. Got shape {bvals.shape} after squeeze."
        )
        
    return bvals

from .gradients import create_gradient_table # Assuming gradients.py is in the same directory
from dipy.core.gradients import GradientTable # For type hinting

def load_nifti_study(
    dwi_path: str, 
    bval_path: str, 
    bvec_path: str, 
    mask_path: str = None, 
    b0_threshold: float = 50.0, 
    atol: float = 1e-2
) -> tuple[np.ndarray, GradientTable, np.ndarray | None, np.ndarray]:
    """
    Loads a complete dMRI study from NIfTI DWI, FSL bval/bvec files,
    and an optional NIfTI mask.

    Parameters
    ----------
    dwi_path : str
        Path to the NIfTI file containing DWI data.
    bval_path : str
        Path to the FSL-formatted bval file.
    bvec_path : str
        Path to the FSL-formatted bvec file.
    mask_path : str, optional
        Path to the NIfTI file containing the brain mask. If None (default),
        no mask is loaded.
    b0_threshold : float, optional
        Threshold used to identify b0 volumes when creating the gradient table.
        Default is 50.0.
    atol : float, optional
        Absolute tolerance used for various checks in gradient table creation.
        Default is 1e-2.

    Returns
    -------
    tuple[np.ndarray, GradientTable, np.ndarray | None, np.ndarray]
        A tuple containing:
        - dwi_data (np.ndarray): The 4D DWI data array (X, Y, Z, N_volumes).
        - gtab (GradientTable): The Dipy GradientTable object.
        - mask_data (np.ndarray | None): The 3D boolean mask array, or None
                                         if `mask_path` was not provided.
        - affine (np.ndarray): The 4x4 affine transformation matrix from the DWI NIfTI.

    Raises
    ------
    FileNotFoundError
        If any of the required file paths do not exist.
    nibabel.filebasedimages.ImageFileError
        If Nibabel cannot load a NIfTI file.
    ValueError
        If there are inconsistencies in data shapes/dimensions (e.g., DWI vs. bvals,
        DWI vs. mask) or if gradient table creation fails.
    RuntimeError
        For other unexpected errors during file loading or processing.
    """
    # Load DWI data and main affine
    dwi_data, affine = load_nifti_dwi(dwi_path)

    # Load b-values and b-vectors
    bvals = load_fsl_bvals(bval_path)
    bvecs = load_fsl_bvecs(bvec_path)

    # --- Consistency Checks for DWI, bvals, bvecs ---
    if dwi_data.shape[3] != len(bvals):
        raise ValueError(
            f"Number of volumes in DWI data ({dwi_data.shape[3]}) "
            f"does not match number of b-values ({len(bvals)})."
        )
    
    if dwi_data.shape[3] != bvecs.shape[0]:
        raise ValueError(
            f"Number of volumes in DWI data ({dwi_data.shape[3]}) "
            f"does not match number of b-vectors ({bvecs.shape[0]})."
        )
    # Note: create_gradient_table will also check if len(bvals) == len(bvecs)

    # Create GradientTable
    gtab = create_gradient_table(bvals, bvecs, b0_threshold=b0_threshold, atol=atol)

    # Load optional mask
    mask_data = None
    if mask_path is not None:
        loaded_mask_data, mask_affine = load_nifti_mask(mask_path)
        
        # Consistency Check: Affines
        if not np.allclose(affine, mask_affine, atol=1e-5): # Using a small tolerance for float comparison
            # Consider using warnings.warn for user warnings instead of print
            import warnings
            warnings.warn(
                f"DWI affine and mask affine are different.\n"
                f"DWI affine:\n{affine}\n"
                f"Mask affine:\n{mask_affine}\n"
                f"Using DWI affine for all spatial interpretations.", UserWarning
            )
            # Decision to use DWI affine is implicit as `affine` is what's returned.

        # Consistency Check: Spatial Shapes
        if dwi_data.shape[:3] != loaded_mask_data.shape:
            raise ValueError(
                f"Spatial dimensions of DWI data {dwi_data.shape[:3]} "
                f"and mask {loaded_mask_data.shape} do not match."
            )
        mask_data = loaded_mask_data
        
    return dwi_data, gtab, mask_data, affine
