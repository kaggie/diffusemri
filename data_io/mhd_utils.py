import os
import logging
import numpy as np
import SimpleITK as sitk
import nibabel as nib # For NIfTI interaction and affine transformations

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def read_mhd_data(mhd_filepath: str) \
        -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    """
    Reads data and metadata from a MHD/MHA file using SimpleITK.

    Attempts to parse spatial metadata to construct a NIfTI-compatible affine matrix
    and extracts DWI-specific information (b-values, b-vectors) if present.

    Args:
        mhd_filepath (str): Path to the MHD/MHA file.

    Returns:
        tuple:
            - image_data_numpy (np.ndarray | None): The image data, potentially transposed to NIfTI-like order (x,y,z,...).
            - affine (np.ndarray | None): NIfTI-compatible affine matrix.
            - bvals (np.ndarray | None): Array of b-values, if DWI.
            - bvecs (np.ndarray | None): Array of b-vectors (Nx3, reoriented to image space), if DWI.
            - metadata (dict): Dictionary of metadata from the MHD header.
            Returns (None, None, None, None, {}) if reading fails.
    """
    if not os.path.exists(mhd_filepath):
        logger.error(f"MHD/MHA file not found: {mhd_filepath}")
        return None, None, None, None, {}

    try:
        image = sitk.ReadImage(mhd_filepath)
        logger.info(f"Successfully read MHD/MHA file: {mhd_filepath}")
    except Exception as e:
        logger.error(f"Failed to read MHD/MHA file {mhd_filepath}: {e}")
        return None, None, None, None, {}

    # --- Extract Image Data ---
    # SimpleITK GetArrayFromImage returns array with order [z,y,x] or [t,z,y,x] etc.
    # NIfTI / Nibabel typically expect [x,y,z] or [x,y,z,t].
    # We need to transpose the data.
    image_data_sitk_order = sitk.GetArrayFromImage(image)

    # Transpose to NIfTI-like order: (x, y, z, ...) from (..., z, y, x)
    # If 3D: (z,y,x) -> (x,y,z) by transposing (2,1,0)
    # If 4D: (t,z,y,x) -> (x,y,z,t) by transposing (3,2,1,0)
    # In general, reverse the order of axes.
    axes_order = list(range(image_data_sitk_order.ndim))[::-1]
    image_data_numpy = image_data_sitk_order.transpose(axes_order)
    logger.info(f"Original SITK data shape: {image_data_sitk_order.shape}, "
                f"Transposed NumPy data shape: {image_data_numpy.shape}")

    # --- Construct Affine Matrix ---
    # Get Spacing (X, Y, Z order)
    spacing = np.array(image.GetSpacing())
    # Get Origin (X, Y, Z order)
    origin = np.array(image.GetOrigin())
    # Get Direction Cosine Matrix (3x3 for 3D, potentially larger if >3D image in SITK)
    dimension = image.GetDimension()
    direction_cosines_flat = image.GetDirection()

    # Ensure we handle spatial dimensions correctly, typically 3 for medical images.
    # If image is 4D (e.g. DWI), GetDimension() is 4. Direction matrix is 16 elements (4x4).
    # We are interested in the spatial part for the NIfTI affine.
    spatial_dims = min(dimension, 3)
    direction_matrix = np.array(direction_cosines_flat).reshape(dimension, dimension)

    # The NIfTI affine maps voxel coordinates to world coordinates.
    # Affine[:spatial_dims, :spatial_dims] = Rotation/Scaling part
    # Affine[:spatial_dims, 3] = Translation part

    affine = np.eye(4)

    # Rotation/Scaling part: DirectionMatrix * SpacingDiagonal
    # SimpleITK Direction matrix is already the rotation matrix.
    # Spacing needs to be applied to it.
    # Affine_rot_scale = DirectionMatrix @ np.diag(Spacing_spatial)
    # where Spacing_spatial are the spacings for the spatial dimensions.

    # Spatial part of direction matrix and spacing
    spatial_direction_matrix = direction_matrix[:spatial_dims, :spatial_dims]
    spatial_spacing = spacing[:spatial_dims]

    # Calculate the rotation/scaling part of the affine
    affine_rot_scale = spatial_direction_matrix @ np.diag(spatial_spacing)

    affine[:spatial_dims, :spatial_dims] = affine_rot_scale

    # Translation part (origin)
    # The origin in SimpleITK is the coordinate of the center of voxel (0,0,0) in world space.
    # NIfTI affine's translation part also maps voxel (0,0,0) [center or corner depending on convention] to world.
    # If SimpleITK and NIfTI assume center of voxel (0,0,0), then origin can be used directly.
    # However, the data array was transposed. If data was (z,y,x) and is now (x,y,z),
    # the origin and direction matrix might need adjustment if they were relative to the original SITK array order.
    # SimpleITK's GetOrigin() and GetDirection() are defined with respect to its voxel indexing (x,y,z,...).
    # Since we transposed data from (...,z,y,x) to (x,y,...,z), we must ensure affine matches the new data order.
    # The constructed affine should map the *new* data array's voxel indices to the *same* world space.

    # Let's use nibabel's helper if possible or be very careful.
    # A robust way: form affine assuming data is (x,y,z) from SITK metadata.
    # If data was originally (z,y,x) from GetArrayFromImage, then:
    # affine_sitk_convention = np.eye(4)
    # affine_sitk_convention[:spatial_dims, :spatial_dims] = spatial_direction_matrix @ np.diag(spatial_spacing)
    # affine_sitk_convention[:spatial_dims, 3] = origin[:spatial_dims]
    # Now, this affine_sitk_convention is for data in (x,y,z) order if SITK internally uses (x,y,z) for these.
    # But GetArrayFromImage gives (z,y,x). So if we transpose to (x,y,z), this affine should be correct.

    affine[:spatial_dims, 3] = origin[:spatial_dims]
    logger.info(f"Constructed affine from MHD header:\n{affine}")

    # --- Extract Metadata ---
    metadata = {}
    for key in image.GetMetaDataKeys():
        metadata[key] = image.GetMetaData(key)

    # --- DWI Metadata Extraction (Placeholder - to be refined) ---
    bvals = None
    bvecs = None # Should be reoriented to image space (relative to new data array axes)

    # Example parsing for DWI fields (highly dependent on how they are stored in MHD)
    # This is a common convention but not strictly standardized for DWI in MHD.
    if metadata.get('modality', '').upper() == 'DWMRI' or \
       metadata.get('DWMRI_b-value') is not None or \
       any(k.lower().startswith('dwmri_gradient_') for k in metadata.keys()):

        logger.info("DWI-related metadata keys found. Attempting to parse bvals/bvecs.")

        # B-value (often a single reference value, or per-gradient in gradient lines)
        # If a single "DWMRI_b-value" is present
        if 'DWMRI_b-value' in metadata:
            try:
                # This might be a single value, or a string of space-separated values
                bval_str = metadata['DWMRI_b-value']
                bvals_parsed = np.array([float(v) for v in bval_str.split()], dtype=float)
                if bvals_parsed.size == 1 and image_data_numpy.ndim == 4 and image_data_numpy.shape[-1] > 1:
                    # If single b-value for 4D data, it's ambiguous.
                    # It might be a reference b-value for all DWIs.
                    # We'll need per-gradient b-values for a full bval array.
                    logger.info(f"Found single 'DWMRI_b-value': {bvals_parsed[0]}. Needs per-gradient info for full bval array.")
                    # Store it for now; gradient parsing might override or use it.
                    # bvals = bvals_parsed
                else:
                    bvals = bvals_parsed
            except Exception as e:
                logger.warning(f"Could not parse 'DWMRI_b-value' ('{metadata.get('DWMRI_b-value')}') as float or list of floats: {e}")

        # Gradient vectors (and potentially b-values per gradient)
        gradient_keys = sorted([key for key in metadata if key.lower().startswith('dwmri_gradient_')])
        if gradient_keys:
            temp_bvecs_mhd = []
            temp_bvals_from_grads = []
            is_bval_in_gradient_line = False

            for i, key in enumerate(gradient_keys):
                parts = metadata[key].strip().split()
                try:
                    if len(parts) >= 3:
                        vec = [float(p) for p in parts[:3]]
                        temp_bvecs_mhd.append(vec)
                        if len(parts) >= 4: # Check for b-value in the gradient line
                            temp_bvals_from_grads.append(float(parts[3]))
                            if i == 0: is_bval_in_gradient_line = True # Assume consistent format
                        elif bvals is not None and bvals.size == 1: # Use single reference b-value
                             # If vector is non-zero, use the reference b-value, else 0 for b0
                            temp_bvals_from_grads.append(bvals[0] if np.linalg.norm(vec) > 1e-6 else 0.0)
                        else: # Fallback b-value if not in line and no global ref
                            temp_bvals_from_grads.append(0.0) # Or some other default / error
                    else:
                        logger.warning(f"Could not parse gradient line '{key}': {metadata[key]} - too few parts.")
                except ValueError:
                    logger.warning(f"Could not parse floats from gradient line '{key}': {metadata[key]}")

            if temp_bvecs_mhd:
                bvecs_mhd_coord = np.array(temp_bvecs_mhd, dtype=float)

                # If b-values were successfully parsed from gradient lines, use them
                if is_bval_in_gradient_line and len(temp_bvals_from_grads) == len(bvecs_mhd_coord):
                    bvals = np.array(temp_bvals_from_grads, dtype=float)
                elif bvals is not None and bvals.size==1 and image_data_numpy.ndim == 4 and image_data_numpy.shape[-1] == len(bvecs_mhd_coord):
                    # If a single reference b-value was found, and we have b-vectors for each volume
                    num_volumes = image_data_numpy.shape[-1]
                    ref_bval = bvals[0]
                    bvals = np.full(num_volumes, ref_bval)
                    for i in range(num_volumes): # Set b0s based on bvec norm
                        if np.linalg.norm(bvecs_mhd_coord[i,:]) < 1e-6:
                            bvals[i] = 0.0
                elif bvals is None and image_data_numpy.ndim == 4 and image_data_numpy.shape[-1] == len(bvecs_mhd_coord):
                    logger.warning("B-vectors found, but no clear b-values. Assuming all are b0 or default DWI b-value if set.")
                    bvals = np.zeros(len(bvecs_mhd_coord)) # Fallback: assume all b0 if no b-value info.

                # Reorient b-vectors
                # MHD b-vectors might be in LPS or RAS (defined by "space" field) or relative to a "measurement frame"
                # For NIfTI, b-vectors are usually relative to the image axes after applying the affine.
                # This part is complex and requires clear conventions for MHD DWI.
                # Assuming bvecs_mhd_coord are in the same coordinate system as defined by MHD header's
                # 'space origin' and 'space directions' (i.e., the world/patient coordinate system).
                # We need to transform them to be relative to the image axes defined by the NIfTI affine.
                # NIfTI affine maps voxel -> world. Inverse of rotation part maps world -> voxel relative axes.
                rot_matrix = affine[:3, :3].copy()
                # Normalize columns of rotation matrix to remove scaling, get pure rotation
                for col_idx in range(rot_matrix.shape[1]):
                    norm = np.linalg.norm(rot_matrix[:, col_idx])
                    if norm > 1e-6: rot_matrix[:, col_idx] /= norm

                try:
                    inv_rot_matrix = np.linalg.inv(rot_matrix)
                except np.linalg.LinAlgError:
                    logger.warning("Cannot invert rotation matrix for b-vector reorientation. Using pseudo-inverse.")
                    inv_rot_matrix = np.linalg.pinv(rot_matrix)

                bvecs_reoriented_list = []
                for b_vec_world in bvecs_mhd_coord:
                    if np.linalg.norm(b_vec_world) > 1e-6: # If not a b0 vector
                        b_vec_img_space = inv_rot_matrix @ b_vec_world
                        norm_img_space = np.linalg.norm(b_vec_img_space)
                        if norm_img_space > 1e-6: b_vec_img_space /= norm_img_space
                        else: b_vec_img_space = np.array([0.,0.,0.])
                        bvecs_reoriented_list.append(b_vec_img_space)
                    else:
                        bvecs_reoriented_list.append(np.array([0.,0.,0.]))
                bvecs = np.array(bvecs_reoriented_list, dtype=float)
                logger.info("Reoriented b-vectors to align with NIfTI image axes convention.")

        # Final consistency check for DWI data
        if data.ndim == 4 and bvals is not None and bvecs is not None:
            num_gradients_data = data.shape[-1]
            if len(bvals) != num_gradients_data or bvecs.shape[0] != num_gradients_data:
                logger.warning(f"Mismatch after parsing: Data has {num_gradients_data} gradients, "
                               f"found {len(bvals)} b-values and {bvecs.shape[0]} b-vectors. Clearing DWI info.")
                bvals, bvecs = None, None
        elif data.ndim != 4 and (bvals is not None or bvecs is not None):
             logger.warning(f"Data is not 4D (shape {data.shape}), but DWI info found. Clearing DWI info.")
             bvals, bvecs = None, None


    return image_data_numpy, affine, bvals, bvecs, metadata


def write_mhd_data(output_filepath: str,
                     data: np.ndarray,
                     affine: np.ndarray,
                     bvals: np.ndarray = None,
                     bvecs: np.ndarray = None,
                     custom_metadata: dict = None,
                     mhd_header_options: dict = None):
    """
    Writes image data and metadata to an MHD/MHA file using SimpleITK.

    Constructs MHD header information from NIfTI-style affine and optionally
    DWI information (b-values, b-vectors) and other custom metadata.

    Args:
        output_filepath (str): Path to save the output MHD/MHA file.
                               The extension (.mhd or .mha) determines if raw data is separate.
        data (np.ndarray): Image data (typically NIfTI-style x,y,z,... order).
        affine (np.ndarray): 4x4 NIfTI-style affine matrix.
        bvals (np.ndarray, optional): 1D array of b-values for DWI.
        bvecs (np.ndarray, optional): Nx3 array of b-vectors for DWI (image space).
        custom_metadata (dict, optional): Additional custom key-value pairs for metadata.
        mhd_header_options (dict, optional): Options directly for SimpleITK image metadata
                                           or to guide header construction (e.g., 'ObjectType',
                                           'NDims', 'BinaryDataByteOrderMSB', 'CompressedData').
                                           Note: 'ObjectType', 'NDims' are set by SimpleITK from data.

    Raises:
        ValueError: If input data or affine is invalid.
        RuntimeError: If `sitk.WriteImage` fails.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input 'data' must be a NumPy array.")
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("Input 'affine' must be a 4x4 NumPy array.")

    logger.info(f"Preparing to write MHD/MHA file: {output_filepath}")

    # --- Convert NumPy data to SimpleITK Image ---
    # SimpleITK expects data in [z,y,x] or [t,z,y,x] order.
    # NumPy data from Nibabel is typically [x,y,z] or [x,y,z,t].
    # Transpose data to SimpleITK's expected order.
    axes_order_to_sitk = list(range(data.ndim))[::-1]
    data_sitk_order = data.transpose(axes_order_to_sitk).copy() # Use copy for C-contiguous array

    try:
        sitk_image = sitk.GetImageFromArray(data_sitk_order)
    except Exception as e:
        logger.error(f"Failed to create SimpleITK image from NumPy array: {e}")
        raise RuntimeError(f"SimpleITK GetImageFromArray failed: {e}")

    # --- Set Spatial Metadata from NIfTI-style Affine ---
    # NIfTI affine: Voxel_coord_NIfTI -> World_coord
    # SimpleITK: Voxel_coord_SITK -> World_coord (where Voxel_coord_SITK is z,y,x based for GetArrayFromImage)

    # Spacing: Usually positive, norms of the first 3 columns of affine, scaled by direction.
    # The first three columns of the NIFTI affine represent the mapping from voxel coordinates (i,j,k) to world coordinates.
    # The length of these column vectors gives the voxel spacing along each NIFTI data axis (x,y,z).
    # SimpleITK spacing is (x_spacing, y_spacing, z_spacing).
    nifti_x_axis, nifti_y_axis, nifti_z_axis = affine[:3, 0], affine[:3, 1], affine[:3, 2]
    spacing_nifti_order = [np.linalg.norm(nifti_x_axis), np.linalg.norm(nifti_y_axis), np.linalg.norm(nifti_z_axis)]
    sitk_image.SetSpacing(spacing_nifti_order[:sitk_image.GetDimension()]) # SITK expects spacing in its (x,y,z) order.

    # Origin: The world coordinate of the center of the voxel (0,0,0) in NIFTI.
    # SimpleITK's origin is also the world coordinate of the (0,0,0) voxel.
    sitk_image.SetOrigin(affine[:sitk_image.GetDimension(), 3].copy())

    # Direction Cosine Matrix:
    # Normalized columns of the NIFTI affine's rotation/scaling part.
    direction_matrix_nifti = np.zeros((3,3))
    for i in range(3):
        vec = affine[:3, i].copy()
        norm = np.linalg.norm(vec)
        if norm > 1e-6: direction_matrix_nifti[:, i] = vec / norm
        else: direction_matrix_nifti[:, i] = vec # Avoid division by zero if a dimension has zero extent

    # SimpleITK direction is a flat list (row-major for 3x3 matrix: Rxx,Rxy,Rxz,Ryx,Ryy,Ryz,Rzx,Rzy,Rzz)
    # It should correspond to the spatial dimensions of the SITK image.
    if sitk_image.GetDimension() == 3:
        sitk_image.SetDirection(direction_matrix_nifti.flatten().tolist())
    elif sitk_image.GetDimension() == 2: # e.g. a single slice
        sitk_image.SetDirection(direction_matrix_nifti[:2,:2].flatten().tolist())
    # For 4D, direction is usually for the spatial part only. SITK handles this.

    logger.info(f"Set SITK Spacing: {sitk_image.GetSpacing()}")
    logger.info(f"Set SITK Origin: {sitk_image.GetOrigin()}")
    logger.info(f"Set SITK Direction: {sitk_image.GetDirection()}")

    # --- DWI Metadata ---
    if bvals is not None and bvecs is not None:
        if len(bvals) != bvecs.shape[0] or bvecs.shape[1] != 3:
            logger.warning("bvals/bvecs inconsistent or malformed. Skipping DWI metadata writing.")
        elif data.ndim == 4 and data.shape[-1] != len(bvals): # Check against last dim of original data array
             logger.warning(f"4th dim of data ({data.shape[-1]}) does not match bvals/bvecs length ({len(bvals)}). Skipping DWI metadata.")
        else:
            sitk_image.SetMetaData("modality", "DWMRI")
            non_zero_bvals = bvals[bvals > 50]
            if non_zero_bvals.size > 0:
                sitk_image.SetMetaData("DWMRI_b-value", str(non_zero_bvals.max()))
            else:
                sitk_image.SetMetaData("DWMRI_b-value", str(0.0))

            # B-vectors need to be reoriented from NIfTI image space to the MHD world/patient space
            # defined by the MHD header's Direction and Origin (LPS or RAS).
            # The NIfTI bvecs are relative to NIfTI image axes.
            # We need to transform them to the PCS used by MHD.
            # If MHD uses LPS, and NIfTI used RAS, this is complex.
            # For now, assume bvecs are provided in a coordinate system that is consistent
            # with the chosen 'space' for the NRRD header or relative to a 'measurement frame'.
            # A common practice is to store b-vectors in patient space in MHD.
            # If input bvecs are in image space (relative to NIFTI axes), transform to world/patient:
            # bvec_world = Affine_rot_scale_part @ bvec_img
            # where Affine_rot_scale_part is affine[:3,:3]

            # Reorient bvecs from image space (assuming input bvecs are that) to world/patient space for MHD
            affine_rotation_scaling = affine[:3, :3]
            bvecs_world_space = np.array([(affine_rotation_scaling @ b_img) for b_img in bvecs])
            # Normalize again
            for i in range(bvecs_world_space.shape[0]):
                norm = np.linalg.norm(bvecs_world_space[i])
                if norm > 1e-6 : bvecs_world_space[i] /= norm
                else: bvecs_world_space[i] = np.array([0.,0.,0.]) # Handle b0s

            for i in range(len(bvals)):
                # Store b-vector in world space, and optionally the b-value again
                sitk_image.SetMetaData(f'DWMRI_gradient_{i:04d}',
                                       f"{bvecs_world_space[i,0]:.8f} {bvecs_world_space[i,1]:.8f} {bvecs_world_space[i,2]:.8f}")
            # Storing a measurement frame of identity implies the gradients are in the patient/world coordinate system
            # as defined by the rest of the MHD spatial tags (Origin, Spacing, Direction).
            sitk_image.SetMetaData("measurement frame", "1 0 0 0 1 0 0 0 1")


    # Add other custom metadata
    if custom_metadata:
        for key, value in custom_metadata.items():
            sitk_image.SetMetaData(str(key), str(value)) # Values must be strings

    # MHD Header Options (less common to set manually, SITK handles most)
    if mhd_header_options:
        if 'ElementSpacing' in mhd_header_options: # Already handled by SetSpacing
            pass
        # Other options like 'CompressedData', 'BinaryDataByteOrderMSB' can be set via general metadata
        # if they are standard ITK MHD tags and SimpleITK doesn't have direct setters.
        # For example, to suggest compression (though WriteImage handles this with filename extension):
        # if 'CompressedData' in mhd_header_options and mhd_header_options['CompressedData']:
        #    sitk_image.SetMetaData('CompressedData', 'True')


    logger.info(f"Writing MHD/MHA data of shape {data.shape} (transposed to {data_sitk_order.shape} for SITK)")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    try:
        sitk.WriteImage(sitk_image, output_filepath)
        logger.info(f"MHD/MHA file saved successfully: {output_filepath}")
    except Exception as e:
        logger.error(f"Failed to write MHD/MHA file {output_filepath}: {e}")
        raise # Re-raise

# Placeholder for CLI functions
# def mhd_to_nifti_cli(...): pass
# def nifti_to_mhd_cli(...): pass
