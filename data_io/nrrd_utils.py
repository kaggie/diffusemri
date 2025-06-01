import os
import logging
import numpy as np
import nrrd # For reading and writing NRRD files
import nibabel as nib # For NIfTI interaction and affine transformations if needed

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def read_nrrd_data(nrrd_filepath: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """
    Reads data and metadata from a NRRD file.

    Attempts to parse spatial metadata to construct a NIfTI-compatible affine matrix
    and extracts DWI-specific information (b-values, b-vectors) if present.

    Args:
        nrrd_filepath (str): Path to the NRRD file.

    Returns:
        tuple:
            - image_data (np.ndarray | None): The image data.
            - affine (np.ndarray | None): NIfTI-compatible affine matrix.
            - bvals (np.ndarray | None): Array of b-values, if DWI.
            - bvecs (np.ndarray | None): Array of b-vectors (Nx3, reoriented), if DWI.
            - header (dict): The full NRRD header.
            Returns (None, None, None, None, {}) if reading fails.
    """
    if not os.path.exists(nrrd_filepath):
        logger.error(f"NRRD file not found: {nrrd_filepath}")
        return None, None, None, None, {}

    try:
        data, header = nrrd.read(nrrd_filepath)
        logger.info(f"Successfully read NRRD file: {nrrd_filepath}")
    except Exception as e:
        logger.error(f"Failed to read NRRD file {nrrd_filepath}: {e}")
        return None, None, None, None, {}

    # --- Construct Affine Matrix ---
    # NRRD header provides 'space directions' (vectors for each axis) and 'space origin'.
    # 'space directions': an array of vectors. Each vector is a column in the affine matrix's
    #                     rotational/scaling part, when considering data array axes (e.g., i, j, k).
    # 'space origin': the coordinates of the center of the first voxel.

    affine = np.eye(4)
    try:
        if 'space directions' in header and header['space directions'] is not None:
            space_directions = np.asarray(header['space directions'])
            if data.ndim == 3: # (k,j,i) order for nrrd.read vs (i,j,k) for nifti typically
                if space_directions.shape == (3, 3): # Check if 3x3
                    # NRRD 'space directions' are columns:
                    # First col is direction of first data axis (e.g. 'i' if data[i,j,k])
                    # Second col is direction of second data axis
                    # Third col is direction of third data axis
                    # Nifti affine maps voxel coords (i,j,k) to world (x,y,z)
                    # Affine columns are:
                    # Col 0: mapping of X-axis of data array
                    # Col 1: mapping of Y-axis of data array
                    # Col 2: mapping of Z-axis of data array
                    # Data from nrrd.read is often (ax0, ax1, ax2, ...)
                    # Let's assume data is (ax0, ax1, ax2) for 3D.
                    # Space directions: [vec_ax0, vec_ax1, vec_ax2]
                    # If data is ordered (ax0, ax1, ax2) and we want NIFTI (i,j,k) -> (x,y,z)
                    # and typically NIFTI data is (i,j,k), we might need to transpose data later
                    # or adjust affine construction.
                    # For now, assume space_directions correspond to data axes as read by nrrd.
                    affine[0:3, 0:3] = space_directions.T # Transpose to make them rows of the direction part
                                                        # Or directly assign columns: affine[0:3, i] = space_directions[i]
                    # No, space_directions are the vectors along each data axis.
                    # So, affine columns should be these vectors.
                    for i in range(3):
                        affine[:3, i] = space_directions[i]

                else:
                    logger.warning(f"NRRD 'space directions' field has unexpected shape: {space_directions.shape}. Expected (3,3) for 3D data.")
            elif data.ndim == 4: # Assuming 4th dim is non-spatial (e.g. time, gradients)
                 if space_directions.shape == (3, 3): # Only 3 spatial directions
                      for i in range(3):
                        affine[:3, i] = space_directions[i]
                 elif space_directions.shape == (4,4) and space_directions[3,3] == 1: # Full affine in space directions
                     # This is less common for 'space directions' but possible if it's a full matrix.
                     # More often, it's (Ndim, Ndim) for spatial dims, or (Ndim, Nvectors_per_dim)
                     logger.info("Found 4x4 'space directions', interpreting as full affine (rotation/scale part).")
                     affine[:3,:3] = space_directions[:3,:3]
                 elif space_directions.shape[0] == data.ndim: # e.g. (4,3) for 4D data, last one ignored for spatial affine
                     logger.warning(f"NRRD 'space directions' has {space_directions.shape[0]} vectors for {data.ndim}D data. Using first 3 for spatial affine.")
                     for i in range(3):
                        affine[:3, i] = space_directions[i] # This might be wrong if the 4th direction is spatial
                 else:
                    logger.warning(f"NRRD 'space directions' field has shape {space_directions.shape}. Using first 3 for 3D affine if possible.")
                    if space_directions.shape[1] == 3 and space_directions.shape[0] >=3 : # (N,3)
                         for i in range(3):
                            affine[:3, i] = space_directions[i]
            else: # Other dimensions not explicitly handled for affine
                logger.warning(f"Data has {data.ndim} dimensions. Affine construction might be basic.")

        elif 'space_directions' in header and header['space_directions'] is not None: # Older PyNrrd might use underscore
            logger.warning("Found 'space_directions' (with underscore). Preferring 'space directions' (with space).")
            # Similar logic as above could be applied if 'space directions' was None.

        if 'space origin' in header and header['space origin'] is not None:
            space_origin = np.asarray(header['space origin'])
            if space_origin.size == 3:
                affine[0:3, 3] = space_origin
            else:
                logger.warning(f"NRRD 'space origin' field has unexpected size: {space_origin.size}. Expected 3.")

        logger.info(f"Constructed affine from NRRD header:\n{affine}")

    except Exception as e:
        logger.error(f"Error constructing affine from NRRD header: {e}. Using identity affine.")
        affine = np.eye(4) # Fallback to identity

    # --- DWI Metadata Extraction ---
    bvals = None
    bvecs = None # Will be reoriented to image space

    # Check for common DWI modality indicator
    is_dwi = header.get('modality', '').upper() == 'DWMRI' or \
             any(key.lower().startswith('dwmri_b-value') for key in header.keys()) or \
             any(key.lower().startswith('dwmri_gradient_') for key in header.keys())

    if is_dwi or 'DWMRI_b-value' in header: # Check specific b-value tag too
        logger.info("DWI-related fields found or modality is DWMRI. Attempting to extract bvals/bvecs.")

        # B-value
        if 'DWMRI_b-value' in header:
            try:
                bvals = np.array([float(header['DWMRI_b-value'])], dtype=float)
                # If data is 4D, and only one b-value is specified, assume it applies to all non-b0 volumes.
                # This is a common simplification. True multi-shell would have per-volume b-values.
                # For now, if data is 4D, we'll need per-volume b-values.
                # This single 'DWMRI_b-value' might be a reference b-value for DWIs.
                # We need to find per-gradient b-values.
                # Often, b-values are implicitly part of the gradient lines or need to be inferred.
                # Let's assume if this tag exists, it's the primary b-value for all DWIs.
                # A more robust parser would look for per-gradient b-values if this is just a reference.
                logger.info(f"Found 'DWMRI_b-value': {bvals[0]}. This might be a reference value.")
                # This logic is too simple for real multi-shell or varying b-value data.
                # For now, we'll assume if this is present, all DWIs (non-b0) have this b-value.
                # A proper solution needs to parse gradient-specific b-values if they exist.
            except ValueError:
                logger.warning("Could not parse 'DWMRI_b-value' as float.")

        # Gradient Vectors and potentially per-gradient b-values
        gradients_data = [] # list of (bval, bvec) tuples
        gradient_keys = sorted([key for key in header if key.lower().startswith('dwmri_gradient_')])

        if gradient_keys:
            temp_bvals = []
            temp_bvecs_patient = []
            for key in gradient_keys:
                try:
                    parts = header[key].strip().split()
                    # Format can be "gx gy gz bval" or just "gx gy gz" if bval is global or in another field.
                    if len(parts) >= 3:
                        vec = [float(p) for p in parts[:3]]
                        temp_bvecs_patient.append(vec)
                        # Try to get b-value from the same line if present (often for Siemens)
                        current_bval = None
                        if len(parts) >= 4:
                            try:
                                current_bval = float(parts[3])
                            except ValueError:
                                logger.warning(f"Could not parse 4th element of gradient '{key}' as b-value: '{parts[3]}'")

                        # If DWMRI_b-value exists and no per-gradient bval, use it.
                        if current_bval is None and bvals is not None: # bvals here is the single DWMRI_b-value
                            # If the vector is non-zero, assign the global DWMRI_b-value.
                            # If vector is zero, it's a b0, so b-value is 0.
                            if np.linalg.norm(vec) > 1e-6:
                                temp_bvals.append(bvals[0])
                            else:
                                temp_bvals.append(0.0)
                        elif current_bval is not None:
                            temp_bvals.append(current_bval)
                        else: # No global DWMRI_b-value, no per-gradient b-value
                            logger.warning(f"No b-value found for gradient '{key}'. Assuming 0.")
                            temp_bvals.append(0.0)
                    else:
                        logger.warning(f"Could not parse gradient line '{key}': {header[key]}")
                except Exception as e:
                    logger.warning(f"Error parsing gradient line '{key}': {header[key]} - {e}")

            if temp_bvals: bvals = np.array(temp_bvals, dtype=float)
            if temp_bvecs_patient: bvecs_patient_coord = np.array(temp_bvecs_patient, dtype=float)

            # B-vector reorientation
            if bvecs_patient_coord is not None and bvecs_patient_coord.size > 0:
                measurement_frame = header.get('measurement frame')
                if measurement_frame is not None and np.array(measurement_frame).shape == (3,3):
                    mf = np.array(measurement_frame)
                    logger.info(f"Applying measurement frame to b-vectors:\n{mf}")
                    # If b-vectors are defined relative to measurement frame, transform them to PCS
                    # b_pcs = MF * b_mf
                    bvecs_pcs_temp = np.zeros_like(bvecs_patient_coord)
                    for i, b_mf in enumerate(bvecs_patient_coord):
                         bvecs_pcs_temp[i,:] = mf @ b_mf # Matrix multiplication
                    bvecs_patient_coord = bvecs_pcs_temp

                # Reorient from patient coordinate system (PCS) to image coordinate system (ICS)
                # Affine maps voxel coords (i,j,k) to world/PCS (x,y,z): x_pcs = A * x_vox
                # We need to transform b-vectors from PCS to ICS.
                # The rotation part of the affine (A_rot) maps ICS axes to PCS axes.
                # So, b_ics = A_rot.T * b_pcs (or inv(A_rot) * b_pcs)
                A_rot = affine[:3,:3].copy()
                # Remove scaling from A_rot to get pure rotation (or rotation+reflection)
                for col_idx in range(A_rot.shape[1]):
                    norm = np.linalg.norm(A_rot[:, col_idx])
                    if norm > 1e-6: A_rot[:, col_idx] /= norm

                try:
                    A_rot_inv = np.linalg.inv(A_rot)
                except np.linalg.LinAlgError:
                    logger.warning("Could not invert rotation part of affine for b-vector reorientation. Using pseudo-inverse.")
                    A_rot_inv = np.linalg.pinv(A_rot)

                bvecs_reoriented_list = []
                for i, b_vec_pcs in enumerate(bvecs_patient_coord):
                    if np.any(np.abs(b_vec_pcs) > 1e-6): # If it's not a zero vector (b0)
                        b_vec_ics = A_rot_inv @ b_vec_pcs
                        norm_ics = np.linalg.norm(b_vec_ics)
                        if norm_ics > 1e-6: b_vec_ics /= norm_ics
                        else: b_vec_ics = np.array([0.,0.,0.])
                        bvecs_reoriented_list.append(b_vec_ics)
                    else:
                        bvecs_reoriented_list.append(np.array([0.,0.,0.]))
                bvecs = np.array(bvecs_reoriented_list, dtype=float)
                logger.info("Reoriented b-vectors to image coordinate system.")

        elif bvals is not None and data.ndim == 4 and data.shape[3] > 1 and bvals.size == 1:
            # If only a single global DWMRI_b-value was found, but data is 4D (multiple volumes)
            # and no per-gradient information was found. This is ambiguous.
            # We can assume it's a single shell acquisition, but we lack directions.
            logger.warning("Single 'DWMRI_b-value' found for 4D data, but no gradient directions. Cannot form bvecs.")
            # Set bvals to repeat for all volumes, but bvecs will be None or zeros.
            num_volumes = data.shape[3]
            b0_mask_implicit = np.zeros(num_volumes, dtype=bool) # Assume first one might be b0 if no other info
            if num_volumes > 0 : b0_mask_implicit[0] = True # Crude assumption

            new_bvals = np.full(num_volumes, bvals[0])
            new_bvals[b0_mask_implicit] = 0
            bvals = new_bvals
            bvecs = np.zeros((num_volumes, 3), dtype=float) # No gradient info available
            logger.warning(f"Assuming b-value {bvals[0]} for all non-b0 volumes, with zeroed b-vectors due to missing directionality.")


    # Ensure bvals and bvecs have consistent length with the last data dimension if data is 4D
    if data.ndim == 4:
        num_volumes = data.shape[3]
        if bvals is not None and len(bvals) != num_volumes:
            logger.warning(f"Number of b-values ({len(bvals)}) does not match data's 4th dimension ({num_volumes}). Clearing DWI info.")
            bvals, bvecs = None, None
        if bvecs is not None and bvecs.shape[0] != num_volumes:
            logger.warning(f"Number of b-vectors ({bvecs.shape[0]}) does not match data's 4th dimension ({num_volumes}). Clearing DWI info.")
            bvals, bvecs = None, None
    elif bvals is not None or bvecs is not None: # Data is 3D but DWI info found
        logger.warning("Data is 3D, but DWI information (bvals/bvecs) was found. Clearing DWI info as it's inconsistent.")
        bvals, bvecs = None, None

    # Final check: if bvals is all zeros, bvecs should also be all zeros or None
    if bvals is not None and np.all(bvals < 1e-3) and bvecs is not None: # Use small threshold for "all zeros"
        logger.info("All b-values are effectively zero. Setting b-vectors to zeros.")
        bvecs = np.zeros_like(bvecs)


    return data, affine, bvals, bvecs, header


def write_nrrd_data(output_filepath: str,
                      data: np.ndarray,
                      affine: np.ndarray,
                      bvals: np.ndarray = None,
                      bvecs: np.ndarray = None,
                      custom_fields: dict = None,
                      nrrd_header_options: dict = None):
    """
    Writes image data and metadata to a NRRD file.

    Constructs a NRRD header from the affine matrix and optionally DWI information
    (b-values, b-vectors) and other custom fields.

    Args:
        output_filepath (str): Path to save the output NRRD file.
        data (np.ndarray): Image data (3D or 4D).
        affine (np.ndarray): 4x4 NIfTI-style affine matrix.
        bvals (np.ndarray, optional): 1D array of b-values for DWI.
        bvecs (np.ndarray, optional): Nx3 array of b-vectors for DWI (image space).
        custom_fields (dict, optional): Additional custom key-value pairs for the header.
        nrrd_header_options (dict, optional): Options for `nrrd.write` like 'type',
                                             'endian', 'encoding'.

    Raises:
        ValueError: If input data or affine is invalid.
        Exception: If `nrrd.write` fails.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input 'data' must be a NumPy array.")
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("Input 'affine' must be a 4x4 NumPy array.")

    logger.info(f"Preparing to write NRRD file: {output_filepath}")

    # Initialize header dictionary
    header = {}

    # Basic NRRD header fields from data shape and type
    header['type'] = data.dtype.name
    if data.dtype == np.bool_: # nrrd.write might not handle bool directly, convert
        logger.info("Converting boolean data to uint8 for NRRD saving.")
        data = data.astype(np.uint8)
        header['type'] = 'uint8'

    header['dimension'] = data.ndim
    header['sizes'] = np.array(data.shape) # Ensure it's a NumPy array for py_nrrd

    # --- Derive NRRD spatial metadata from NIfTI-style affine ---
    # 'space': Defines the anatomical orientation (e.g., 'left-posterior-superior')
    # This needs to be inferred from the affine's rotation matrix or set by user.
    # For simplicity, common default or user-provided via nrrd_header_options.
    # Example: nib.aff2axcodes(affine) can give ('R','A','S'), then join to "right-anterior-superior"
    # Defaulting to LPS as it's common.
    axcodes = nib.aff2axcodes(affine)
    space_map = {'L': 'left', 'R': 'right', 'P': 'posterior', 'A': 'anterior', 'S': 'superior', 'I': 'inferior'}
    # NRRD space often specified as origin-axis1-axis2-axis3 e.g. LPS = left-posterior-superior
    # If axcodes are (R,A,S), NRRD space could be right-anterior-superior
    # Order for NRRD space is typically X, Y, Z.
    try:
        header['space'] = "-".join(space_map[c] for c in axcodes)
    except KeyError:
        logger.warning(f"Could not determine NRRD space from axcodes {axcodes}. Defaulting to 'left-posterior-superior'.")
        header['space'] = 'left-posterior-superior'


    # 'space directions': Voxel dimensions and orientation. Columns of the rotation/scaling part of affine.
    # Ensure these are lists of lists for JSON compatibility in some NRRD readers, or py_nrrd handles np.ndarray.
    # space_directions_vectors = [affine[:3, i].tolist() for i in range(data.ndim if data.ndim <=3 else 3)]
    # For py-nrrd, this should be a list of lists (or Nx3 array) where N is dimension of data
    # It should be (Ndim_spatial, Ndim_spatial) if diagonal, or list of Ndim_spatial vectors
    num_spatial_dims = min(data.ndim, 3) # Consider only spatial dimensions for space directions
    space_directions_vectors = [affine[:3, i].copy() for i in range(num_spatial_dims)]
    # NRRD standard: "If the field is a (dimension)x(dimension) matrix, then each ROW is a vector."
    # However, py-nrrd docs and examples seem to use columns.
    # And NIfTI affine columns are the axes in world space.
    # Let's stick to affine columns for space directions.
    header['space directions'] = space_directions_vectors


    # 'space origin': Translation part of the affine (center of the first voxel).
    header['space origin'] = affine[:3, 3].tolist()

    # 'endian', 'encoding': Can be set via nrrd_header_options or defaults used by nrrd.write
    if nrrd_header_options:
        if 'endian' in nrrd_header_options: header['endian'] = nrrd_header_options['endian']
        if 'encoding' in nrrd_header_options: header['encoding'] = nrrd_header_options['encoding']
        if 'type' in nrrd_header_options: header['type'] = nrrd_header_options['type'] # Allow override

    # --- DWI Metadata ---
    if bvals is not None and bvecs is not None:
        if len(bvals) != bvecs.shape[0]:
            raise ValueError("Length of bvals must match number of rows in bvecs.")
        if bvecs.shape[1] != 3:
            raise ValueError("bvecs must have 3 columns.")
        if data.ndim == 4 and data.shape[3] != len(bvals):
            raise ValueError(f"4th dimension of data ({data.shape[3]}) must match length of bvals/bvecs ({len(bvals)}).")

        header['modality'] = 'DWMRI'
        # Store the primary b-value (often the max or most representative non-zero)
        # This is a convention, not strictly defined.
        non_zero_bvals = bvals[bvals > 50] # Crude threshold for non-zero
        if non_zero_bvals.size > 0:
            header['DWMRI_b-value'] = str(non_zero_bvals.max()) # Store max non-zero b-value as reference
        else:
            header['DWMRI_b-value'] = str(0.0)

        # Measurement frame: Use identity if b-vectors are already in image space.
        # If b-vectors were in patient space, this should be the rotation matrix from image to patient.
        # For now, assume bvecs are provided in image coordinates (as expected for NIfTI bvec files).
        header['measurement frame'] = np.eye(3).tolist()

        for i in range(len(bvals)):
            # Store each gradient and its b-value (some formats store b-value per gradient)
            # Format: DWMRI_gradient_0000:=gx gy gz [bval]
            # The b-value part is optional in this tag if a global DWMRI_b-value is set
            # or if b-values are implicitly zero for b0s.
            # For clarity, let's store b-value with each gradient if it's non-zero,
            # or rely on the fact that b0s have zero vectors.
            # Pynrrd handles float values for custom fields correctly.
            # header[f'DWMRI_gradient_{i:04d}'] = f"{bvecs[i,0]:.8f} {bvecs[i,1]:.8f} {bvecs[i,2]:.8f} {bvals[i]:.1f}"
            # Simpler: just store vectors, b-value is separate or inferred
            header[f'DWMRI_gradient_{i:04d}'] = f"{bvecs[i,0]:.8f} {bvecs[i,1]:.8f} {bvecs[i,2]:.8f}"
            # One could also add a separate list of b-values if the format expects it, e.g. as a string
            # header['DWMRI_b-values_list'] = " ".join(map(str, bvals))

        # If we decide to store all bvals as a single string:
        # header['DWMRI_bvalues_all'] = ' '.join([f"{b:.2f}" for b in bvals])


    # Add other custom fields
    if custom_fields:
        for key, value in custom_fields.items():
            # Ensure value is string for NRRD header if it's not numeric or list of numerics
            if not isinstance(value, (str, int, float, list, np.ndarray)):
                header[key] = str(value)
            else:
                header[key] = value

    logger.info(f"Writing NRRD data of shape {data.shape} with header: {header}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    try:
        nrrd.write(output_filepath, data, header=header)
        logger.info(f"NRRD file saved successfully: {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to write NRRD file {output_filepath}: {e}")
        raise # Re-raise the exception to be caught by calling function if needed

# Placeholder for CLI functions
# def nrrd_to_nifti_cli(...): pass
# def nifti_to_nrrd_cli(...): pass
