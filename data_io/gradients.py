import numpy as np
from dipy.core.gradients import gradient_table, GradientTable # Import GradientTable for type hinting

def create_gradient_table(bvals: np.ndarray, bvecs: np.ndarray, b0_threshold: float = 50.0, atol: float = 1e-2) -> GradientTable:
    """
    Creates a Dipy GradientTable object from b-values and b-vectors arrays.

    The GradientTable object is a core Dipy structure used in most reconstruction
    and model fitting workflows to encapsulate gradient information.

    Parameters
    ----------
    bvals : np.ndarray
        A 1D NumPy array containing the b-value for each diffusion gradient acquisition.
    bvecs : np.ndarray
        A 2D NumPy array of shape (N, 3) containing the b-vector (gradient direction)
        for each acquisition. N should match the length of `bvals`.
    b0_threshold : float, optional
        The threshold used to identify b0 volumes (volumes with b-value close to zero).
        Volumes with b-values less than or equal to this threshold are considered b0s.
        Default is 50.0.
    atol : float, optional
        Absolute tolerance used by Dipy's `gradient_table` function for various checks,
        including b0 identification (relative to `b0_threshold`) and ensuring
        b-vectors for non-b0 volumes are unit vectors. Default is 1e-2.

    Returns
    -------
    dipy.core.gradients.GradientTable
        A Dipy GradientTable object.

    Raises
    ------
    ValueError
        If `bvals` is not a 1D NumPy array.
        If `bvecs` is not a 2D NumPy array with shape (N, 3).
        If the number of b-values does not match the number of b-vectors.
        If Dipy's `gradient_table` function raises an error due to inconsistent
        inputs (e.g., b-vector normalization issues).
    RuntimeError
        For other unexpected errors during Dipy's gradient_table creation.
    """
    # --- Input Validation ---
    if not isinstance(bvals, np.ndarray) or bvals.ndim != 1:
        raise ValueError("bvals must be a 1D NumPy array.")
    
    if not isinstance(bvecs, np.ndarray) or bvecs.ndim != 2:
        raise ValueError("bvecs must be a 2D NumPy array.")
    
    if bvecs.shape[1] != 3:
        raise ValueError(f"bvecs must have shape (N, 3), but got shape {bvecs.shape}.")
        
    if len(bvals) != bvecs.shape[0]:
        raise ValueError(
            f"Number of b-values ({len(bvals)}) must match the number of "
            f"b-vectors ({bvecs.shape[0]})."
        )

    # --- GradientTable Creation ---
    try:
        gtab = gradient_table(
            bvals=bvals,
            bvecs=bvecs,
            b0_threshold=b0_threshold,
            atol=atol
        )
    except ValueError as e:
        # Dipy's gradient_table can raise ValueError for issues like:
        # - bvecs not being unit vectors for bvals > b0_threshold
        # - other internal consistency checks
        raise ValueError(f"Dipy's gradient_table creation failed: {e}")
    except Exception as e: # Catch any other unexpected errors from Dipy
        raise RuntimeError(f"An unexpected error occurred during Dipy gradient_table creation: {e}")
        
    return gtab
