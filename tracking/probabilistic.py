import numpy as np
import nibabel as nib # For affine, if needed (though not explicitly used in current logic beyond seeds_from_mask)
from dipy.data import get_sphere, Sphere # Sphere for type hinting
from dipy.direction.probabilistic_direction_getter import ProbabilisticDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines
from dipy.tracking.stopping_criterion import StoppingCriterion # For type hinting
from dipy.tracking.utils import seeds_from_mask
# For type hinting odf_fit_object, we can check for 'shm_coeff' attribute
# from dipy.reconst.csdeconv import CsdFit
# from dipy.reconst.qball import QballFit

def track_probabilistic_odf(
    odf_fit_object,
    seeds,
    stopping_criterion,
    sphere,
    step_size=0.5,
    affine=np.eye(4),
    samples_per_voxel=5,
    max_angle=30.0,
    pmf_threshold=0.1,
    min_length=10,
    max_length=300
):
    """
    Performs probabilistic tractography from ODF models.

    This function generates streamlines by sampling directions from a probability
    mass function (PMF) derived from the ODF model at each step.

    Parameters
    ----------
    odf_fit_object : object
        Fitted Dipy ODF model object (e.g., from Dipy's `CsdModel.fit()` or
        `QballModel.fit()`, typically the `_fit_object` from
        `diffusemri.models.CsdModel` or `QballModel`). Must have an `shm_coeff`
        attribute containing the spherical harmonic coefficients of the ODFs.
    seeds : np.ndarray
        A 3D binary mask (NumPy array) or an Nx3 array of seed coordinates
        in world/scanner space.
    stopping_criterion : StoppingCriterion
        An instance of a Dipy stopping criterion class (e.g.,
        `BinaryStoppingCriterion`, `ThresholdStoppingCriterion`).
    sphere : dipy.data.Sphere
        Dipy `Sphere` object used for the ProbabilisticDirectionGetter.
    step_size : float, optional
        Tracking step size in mm. Default is 0.5.
    affine : np.ndarray, optional
        NumPy array (4x4) representing the affine transformation from voxel
        space to world/scanner space. Important if using a mask for seeds
        or if streamlines should be in world space. Default is identity.
    samples_per_voxel : int, optional
        Number of streamline seeds to generate per voxel (if `seeds` is a mask)
        or per coordinate (if `seeds` is an array of coordinates). Default is 5.
    max_angle : float, optional
        Maximum allowed angle (in degrees) between successive steps for the
        ProbabilisticDirectionGetter. Default is 30.0.
    pmf_threshold : float, optional
        Threshold for the probability mass function (PMF) used by the
        ProbabilisticDirectionGetter. Default is 0.1.
    min_length : float, optional
        Minimum streamline length in mm. Default is 10.
    max_length : float, optional
        Maximum streamline length in mm. Default is 300.

    Returns
    -------
    dipy.tracking.streamline.Streamlines
        A Dipy `Streamlines` object containing the generated tracts.

    Raises
    ------
    ValueError
        If input parameters are invalid (e.g., incorrect types or shapes).
    AttributeError
        If `odf_fit_object` does not have the `shm_coeff` attribute.
    TypeError
        If `stopping_criterion` or `sphere` are not of expected types.
    RuntimeError
        If direction getter setup or streamline generation fails.
    """

    # --- Input Validation ---
    if not hasattr(odf_fit_object, 'shm_coeff'):
        raise AttributeError("odf_fit_object must have an 'shm_coeff' attribute.")
    
    shm_coeff = odf_fit_object.shm_coeff
    if not isinstance(shm_coeff, np.ndarray) or shm_coeff.ndim < 3: # SH coeffs are at least 3D (x,y,coeffs) or 4D (x,y,z,coeffs)
        raise ValueError("odf_fit_object.shm_coeff must be a NumPy array of at least 3 dimensions.")


    if not isinstance(seeds, np.ndarray):
        raise ValueError("seeds must be a NumPy array (either a 3D mask or Nx3 coordinates).")
    if seeds.ndim not in [2, 3]:
        raise ValueError("seeds must be a 3D mask or an Nx3 array of coordinates.")
    if seeds.ndim == 2 and seeds.shape[1] != 3:
        raise ValueError("If seeds is an array of coordinates, it must have shape Nx3.")
    if seeds.ndim == 3 and not np.issubdtype(seeds.dtype, np.integer) and not np.issubdtype(seeds.dtype, np.bool_):
        raise ValueError("If seeds is a mask, it must be an integer or boolean NumPy array.")

    if not isinstance(stopping_criterion, StoppingCriterion):
        raise TypeError("stopping_criterion must be an instance of a Dipy StoppingCriterion class.")
    
    if not isinstance(sphere, Sphere):
        raise TypeError("sphere must be an instance of Dipy's Sphere class.")

    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("affine must be a 4x4 NumPy array.")
    
    if not (isinstance(step_size, (float, int)) and step_size > 0):
        raise ValueError("step_size must be a positive number.")
    if not (isinstance(samples_per_voxel, int) and samples_per_voxel > 0):
        raise ValueError("samples_per_voxel must be a positive integer.")
    if not (isinstance(max_angle, (float, int)) and 0 < max_angle < 180):
        raise ValueError("max_angle must be a number between 0 and 180.")
    if not (isinstance(pmf_threshold, (float, int)) and 0 <= pmf_threshold <= 1):
        raise ValueError("pmf_threshold must be a number between 0 and 1.")
    if not (isinstance(min_length, (float, int)) and min_length >= 0):
        raise ValueError("min_length must be a non-negative number.")
    if not (isinstance(max_length, (float, int)) and max_length > 0 and max_length >= min_length):
        raise ValueError("max_length must be a positive number greater than or equal to min_length.")


    # --- Seed Generation ---
    if seeds.ndim == 3: # Mask
        actual_seeds = seeds_from_mask(seeds.astype(bool), affine, density=samples_per_voxel)
    elif seeds.ndim == 2: # Coordinates
        actual_seeds = np.repeat(seeds, samples_per_voxel, axis=0)
    else: # Should be caught by initial validation
        raise ValueError("Invalid seeds format.") # Should not be reached

    if actual_seeds.size == 0:
        print("Warning: No seed points generated. Returning empty streamlines.")
        return Streamlines()

    # --- Direction Getter Setup ---
    try:
        prob_dg = ProbabilisticDirectionGetter.from_shcoeff(
            shm_coeff=shm_coeff,
            max_angle=max_angle,
            sphere=sphere,
            pmf_threshold=pmf_threshold
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create ProbabilisticDirectionGetter: {e}")

    # --- Streamline Generation ---
    try:
        streamlines_generator = LocalTracking(
            direction_getter=prob_dg,
            stopping_criterion=stopping_criterion,
            seed_points=actual_seeds,
            affine=affine,
            step_size=step_size
        )
        streamlines = Streamlines(streamlines_generator)
    except Exception as e:
        raise RuntimeError(f"Failed during streamline generation: {e}")

    if not streamlines:
        print("Warning: No streamlines generated.")
        return Streamlines()

    # --- Filter Streamlines by Length ---
    # Dipy's Streamlines object allows direct boolean indexing based on lengths.
    original_count = len(streamlines)
    
    # Apply min_length filter
    lengths = streamlines.length()
    streamlines_min_filtered = streamlines[lengths >= min_length]
    
    if not streamlines_min_filtered:
        if original_count > 0: # Only print warning if there were streamlines to begin with
            print(f"Warning: All {original_count} streamlines were shorter than min_length ({min_length}mm).")
        return Streamlines()
        
    # Apply max_length filter to the already min-filtered streamlines
    lengths_min_filtered = streamlines_min_filtered.length()
    streamlines_max_filtered = streamlines_min_filtered[lengths_min_filtered <= max_length]
    
    if not streamlines_max_filtered and len(streamlines_min_filtered) > 0:
         print(f"Warning: All remaining streamlines after min_length filter were longer than max_length ({max_length}mm).")
         return Streamlines()

    return streamlines_max_filtered
