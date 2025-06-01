import os
import logging
import numpy as np
import nibabel as nib
from brukerapi.dataset import Dataset
from brukerapi.exceptions import UnsuportedDatasetType, IncompleteDataset, DatasetTypeMissmatch

logger = logging.getLogger(__name__)

def read_bruker_dwi_data(bruker_dir_path: str) -> tuple | None:
    """
    Reads Bruker ParaVision DWI dataset and extracts image data, affine,
    b-values, b-vectors, and metadata.

    Args:
        bruker_dir_path (str): Path to the Bruker experiment directory
                               (e.g., a folder named '1' containing 'pdata', 'scan.xml')
                               or directly to the '2dseq' file, or its parent 'pdata/1/'.

    Returns:
        tuple | None: A tuple containing:
            - image_data (np.ndarray): 4D or 5D NumPy array (rows, cols, slices, [time/volumes]).
            - affine (np.ndarray): 4x4 affine matrix.
            - bvals (np.ndarray | None): 1D array of b-values.
            - bvecs (np.ndarray | None): 2D array (N, 3) of b-vectors.
            - metadata_dict (dict): Dictionary with extracted parameters.
        Returns None if critical DWI information cannot be loaded or parsed.
    """
    dataset = None
    try:
        # Try loading assuming bruker_dir_path is the experiment folder or pdata/1 folder
        potential_2dseq_path = os.path.join(bruker_dir_path, 'pdata', '1', '2dseq')
        if os.path.exists(potential_2dseq_path):
            dataset = Dataset(potential_2dseq_path)
        elif os.path.exists(os.path.join(bruker_dir_path, '2dseq')): # If path is to pdata/1
             dataset = Dataset(os.path.join(bruker_dir_path, '2dseq'))
        elif os.path.isfile(bruker_dir_path) and bruker_dir_path.endswith('2dseq'): # If path is directly to 2dseq
            dataset = Dataset(bruker_dir_path)
        else: # If path is to the experiment folder (e.g. 'subject/study/1')
            dataset = Dataset(bruker_dir_path, мясо='fid') # Try to load fid to get path context
            if hasattr(dataset, 'path_to_2dseq_file') and dataset.path_to_2dseq_file:
                 dataset = Dataset(dataset.path_to_2dseq_file) # Re-load with 2dseq
            else: # Fallback if 2dseq path not found, try loading with directory directly
                dataset = Dataset(bruker_dir_path)


    except (FileNotFoundError, UnsuportedDatasetType, IncompleteDataset, DatasetTypeMissmatch) as e:
        logger.error(f"Error initializing Bruker dataset from path {bruker_dir_path}: {e}")
        # Try a more direct approach if the above failed, common for brukerapi usage
        try:
            dataset = Dataset(bruker_dir_path) # Path to expnum folder
        except (FileNotFoundError, UnsuportedDatasetType, IncompleteDataset, DatasetTypeMissmatch) as e_direct:
            logger.error(f"Directly loading Bruker dataset from path {bruker_dir_path} also failed: {e_direct}")
            return None

    if dataset is None:
        logger.error(f"Could not initialize Bruker dataset for path: {bruker_dir_path}")
        return None

    try:
        dataset.load_data() # Load 2dseq image data
        # Brukerapi often loads visu_pars by default with 2dseq. Explicitly load others.
        dataset.add_parameter_file('acqp')
        dataset.add_parameter_file('method')
        # dataset.add_parameter_file('reco') # reco parameters for reconstructed data
        dataset.load_properties() # Ensure derived properties are calculated
    except FileNotFoundError as e:
        logger.warning(f"Could not load acqp, method, or reco files: {e}. DWI info might be missing or incomplete.")
    except Exception as e:
        logger.error(f"Error loading Bruker data or parameter files: {e}")
        return None

    image_data_raw = dataset.data
    if image_data_raw is None:
        logger.error("Image data is None after loading.")
        return None

    metadata_dict = {
        'visu_pars': dataset.visu_pars.as_dict() if dataset.visu_pars else {},
        'method': dataset.method.as_dict() if dataset.method else {},
        'acqp': dataset.acqp.as_dict() if dataset.acqp else {}
    }

    # Extract Affine Matrix
    # Based on VisuCore parameters (typically in mm for extent/position)
    visu_core_size = np.array(dataset.visu_pars.get_value('VisuCoreSize', [0,0,0]))
    visu_core_extent = np.array(dataset.visu_pars.get_value('VisuCoreExtent', [0,0,0]))
    visu_core_position = np.array(dataset.visu_pars.get_value('VisuCorePosition', [0,0,0]))
    # VisuCoreOrientation is row-major in file, reshape to 3x3 matrix
    # Each row is a unit vector for X, Y, Z image axes in world space
    visu_core_orientation_matrix = np.array(dataset.visu_pars.get_value('VisuCoreOrientation', np.eye(3).flatten())).reshape(3, 3)

    if np.any(visu_core_size == 0) or np.any(visu_core_extent == 0):
        logger.error("VisuCoreSize or VisuCoreExtent is zero, cannot compute affine.")
        return None

    voxel_sizes = visu_core_extent / visu_core_size

    affine = np.eye(4)
    # Rotation and scaling part. visu_core_orientation_matrix rows are axis vectors.
    # For NIFTI affine, columns are axis vectors. So, transpose is needed.
    rotation_scaling = visu_core_orientation_matrix.T @ np.diag(voxel_sizes)
    affine[:3, :3] = rotation_scaling

    # Translation: NIFTI affine translation is the world coordinate of the center of the *first voxel*.
    # VisuCorePosition is often the center of the FOV.
    # Transformation: T_nifti = VisuCorePosition_world - RotationScaling @ (VisuCoreSize_vox / 2 - 0.5)
    # (VisuCoreSize_vox / 2 - 0.5) gives vector from first voxel center to FOV center in voxel space.
    image_center_voxels = (visu_core_size - 1) / 2.0
    fov_center_to_first_voxel_vox = -image_center_voxels
    # If VisuCorePosition is FOV center:
    first_voxel_center_world = visu_core_position + (rotation_scaling @ fov_center_to_first_voxel_vox)
    affine[:3, 3] = first_voxel_center_world

    # --- DWI Parameters ---
    bvals, bvecs_scanner = None, None
    # Try to get DWI parameters from the 'method' file (PVM_... tags)
    if dataset.method:
        # PVM_DwEffBval usually contains all effective b-values used in the experiment
        pvm_dw_eff_bval = dataset.method.get_value('PVM_DwEffBval')
        if pvm_dw_eff_bval is not None:
            bvals = np.array(pvm_dw_eff_bval, dtype=float)
        else: # Fallback for older ParaVision or different storage
            pvm_dw_bval_each = dataset.method.get_value('PVM_DwBvalEach')
            if pvm_dw_bval_each is not None:
                 bvals = np.array(pvm_dw_bval_each, dtype=float)

        # PVM_DwDir contains gradient directions in scanner coordinate system
        pvm_dw_dir = dataset.method.get_value('PVM_DwDir')
        if pvm_dw_dir is not None:
            # Expected shape (num_directions, 3) or flattened list
            try:
                bvecs_scanner = np.array(pvm_dw_dir, dtype=float).reshape(-1, 3)
            except ValueError:
                logger.warning(f"Could not reshape PVM_DwDir into (N,3) array. Value: {pvm_dw_dir}")


    if bvals is None:
        logger.warning("b-values (PVM_DwEffBval or PVM_DwBvalEach) not found in 'method' file. Assuming non-DWI or incomplete data.")
    if bvecs_scanner is None:
        logger.warning("b-vectors (PVM_DwDir) not found in 'method' file. Assuming non-DWI or incomplete data.")

    # Data Shape and Transposition
    # Bruker data often (slices, rows, cols, ...) or (rows, cols, slices, ...)
    # NIFTI standard is (rows, cols, slices, volumes/time) -> (X, Y, Z, T)
    # VisuCoreDim tells the dimensionality of each frame.
    # VisuCoreSize is [dim1, dim2, dim3, ...] where dim1=cols, dim2=rows, dim3=slices for 3D.
    # However, `dataset.data` order might be different, e.g. (slices, rows, cols, ...)
    # We need to match this to visu_core_size order for NIFTI.
    # Example: if data is (slices, rows, cols, vols) and VisuCoreSize is (cols, rows, slices)
    # then transpose to (rows, cols, slices, vols) would be (1, 2, 0, 3)

    # Assuming VisuCoreSize is [cols, rows, slices] as per some Bruker conventions for visu_pars
    # And if dataset.data is (slices, rows, cols, [volumes])
    # NIFTI standard order: (cols, rows, slices, [volumes]) if affine maps voxel (i,j,k)
    # or (rows, cols, slices, [volumes]) if affine maps voxel (j,i,k) after transpose.
    # Let's aim for (X, Y, Z, Time/DWIs) where X=cols, Y=rows, Z=slices

    # Default NIFTI orientation for display is often RAS.
    # The affine matrix handles the world coordinate mapping.
    # The order of data in the numpy array should correspond to the voxel iteration.
    # If VisuCoreSize = [Nc, Nr, Ns], data could be (Ns, Nr, Nc, Nvols).
    # We want (Nc, Nr, Ns, Nvols) for NIFTI if affine is identity-like for axes.
    # Or (Nr, Nc, Ns, Nvols) if that's more common with tools.
    # Let's assume VisuCoreSize is [dimX, dimY, dimZ] and data is [dimZ, dimY, dimX, ...rest]
    # Then transpose to [dimX, dimY, dimZ, ...rest] is (2,1,0, ...) for spatial dims.

    img_dims = image_data_raw.ndim
    num_dw_volumes_from_data = image_data_raw.shape[-1] if img_dims > (dataset.visu_pars.get_value('VisuCoreDim',3) -1) else 1

    # Check consistency of bvals/bvecs with data dimensions
    if bvals is not None:
        if len(bvals) != num_dw_volumes_from_data:
            logger.warning(f"Number of b-values ({len(bvals)}) does not match number of volumes in image data ({num_dw_volumes_from_data}). Adjusting b-values.")
            # This could be complex. If PVM_DwEffBval has more values than actual image frames,
            # it might be due to averaging or other processing.
            # For now, truncate or pad bvals/bvecs if there's a clear mismatch with the last data dimension.
            # A more robust solution would use PVM_NRepetitions, PVM_NAverages, etc. to understand structure.
            if len(bvals) > num_dw_volumes_from_data:
                bvals = bvals[:num_dw_volumes_from_data]
            # else: bvals might be too short, which is an issue.

            if bvecs_scanner is not None and len(bvecs_scanner) != len(bvals):
                 if len(bvecs_scanner) > len(bvals): # More bvecs than bvals (after bval adjustment)
                     bvecs_scanner = bvecs_scanner[:len(bvals)]
                 else: # Fewer bvecs than bvals - this is problematic.
                     logger.error("Fewer bvecs than (adjusted) bvals. Cannot proceed with DWI info.")
                     bvecs_scanner = None # Invalidate bvecs

    # Data transposition:
    # Bruker data from `dataset.data` is often (slices, rows, cols, volumes)
    # VisuCoreSize is often [cols, rows, slices]
    # NIFTI standard is (cols, rows, slices, volumes) -> (X,Y,Z,T)
    # So, if raw data is (Ns, Nr, Nc, Nv), transpose to (Nc, Nr, Ns, Nv) is (2,1,0,3)

    visu_dim = dataset.visu_pars.get_value('VisuCoreDim',3)
    image_data_nifti_order = image_data_raw # Assume it's already in a good order or simple

    if visu_dim == 3 and img_dims == 3: # (slices, rows, cols)
        image_data_nifti_order = image_data_raw.transpose(2, 1, 0) # -> (cols, rows, slices)
    elif visu_dim == 3 and img_dims == 4: # (slices, rows, cols, volumes/DWIs)
        image_data_nifti_order = image_data_raw.transpose(2, 1, 0, 3) # -> (cols, rows, slices, volumes)
    elif visu_dim == 2 and img_dims == 2: # (rows, cols)
        image_data_nifti_order = image_data_raw.transpose(1,0) # -> (cols, rows)
    elif visu_dim == 2 and img_dims == 3: # (rows, cols, volumes/DWIs)
        image_data_nifti_order = image_data_raw.transpose(1,0,2) # -> (cols, rows, volumes)
    else:
        logger.warning(f"Unexpected VisuCoreDim ({visu_dim}) and image_data.ndim ({img_dims}). Using raw data order.")
        # No transpose, assuming data is already (X,Y,Z,T) or (X,Y,Z)

    # Reorient b-vectors
    # bvecs_scanner are in Bruker's gradient coordinate system.
    # Need to rotate them to the NIFTI image coordinate system.
    # This involves:
    # 1. Rotation from gradient frame to scanner's physical/world frame (ACQ_gradient_matrix or similar, if different from image axes).
    # 2. Rotation from scanner's physical/world frame to image frame (inverse of visu_core_orientation_matrix).
    # For simplicity, if ACQ_gradient_matrix is identity or aligned with VisuCoreOrientation,
    # then bvecs_scanner are effectively in a frame aligned with visu_core_orientation_matrix axes.
    # To get them into image space (relative to NIFTI image axes after data transposition),
    # we can try: bvecs_img = (inv(visu_core_orientation_matrix.T) @ bvecs_scanner.T).T
    # Or, if bvecs_scanner are already relative to the same axes defined by visu_core_orientation_matrix rows:
    # bvecs_nifti = bvecs_scanner (no rotation needed if data axes and bvec axes are already aligned by convention)
    # This is a common point of complexity.
    # A robust solution often uses `dataset.method.get_value('ACQ_grad_matrix')` if available.
    # For now, assume bvecs are in a coordinate system that can be rotated by visu_core_orientation_matrix
    # to align with the *world* axes that the *image axes* are defined in by visu_core_orientation_matrix.
    # If visu_core_orientation_matrix.T maps voxel axes to world axes, then
    # bvecs_world = bvecs_scanner (if scanner gradient system = world system)
    # bvecs_voxel = inv(visu_core_orientation_matrix.T) @ bvecs_world

    bvecs_nifti = None
    if bvecs_scanner is not None:
        try:
            # This assumes bvecs_scanner are defined in the PCS (Patient Coordinate System)
            # and visu_core_orientation_matrix maps image axes to PCS axes.
            # We need b-vectors relative to the image axes.
            # If a voxel (i,j,k) maps to world via Affine, bvecs should be in that world system.
            # Or, if bvecs are relative to image axes, no rotation needed after data is ordered.
            # Let's assume bvecs from PVM_DwDir are in the scanner frame.
            # The affine[:3,:3] part is visu_core_orientation_matrix.T @ diag(voxel_sizes).
            # So, inv(affine[:3,:3]) maps from world to voxel grid.
            # If PVM_DwDir are in world/scanner, rotate by inv(visu_core_orientation_matrix.T)
            # rotation_matrix_inv = np.linalg.inv(visu_core_orientation_matrix.T) # from world to image-aligned frame
            # bvecs_nifti = (rotation_matrix_inv @ bvecs_scanner.T).T

            # Simpler assumption often made: bvecs are in the same frame as VisuCoreOrientation defines axes.
            # We need to ensure they are unit vectors.
            bvecs_nifti = bvecs_scanner.copy()
            for i in range(len(bvecs_nifti)):
                norm = np.linalg.norm(bvecs_nifti[i])
                if norm > 1e-6:
                    bvecs_nifti[i] /= norm
            logger.info("b-vectors taken from PVM_DwDir. Normalization applied. Reorientation may be needed depending on specific Bruker setup.")

        except Exception as e:
            logger.error(f"Error reorienting b-vectors: {e}")
            bvecs_nifti = None

    return image_data_nifti_order, affine, bvals, bvecs_nifti, metadata_dict

if __name__ == '__main__':
    # Example usage (conceptual - requires a real Bruker dataset path)
    # logging.basicConfig(level=logging.INFO)
    # bruker_path = "/path/to/your/bruker_experiment_folder_or_2dseq"
    # if os.path.exists(bruker_path):
    #     result = read_bruker_dwi_data(bruker_path)
    #     if result:
    #         img_data, aff, bvals, bvecs, meta = result
    #         logger.info(f"Image data shape: {img_data.shape}")
    #         logger.info(f"Affine:\n{aff}")
    #         if bvals is not None:
    #             logger.info(f"b-values: {bvals}")
    #         if bvecs is not None:
    #             logger.info(f"b-vectors (sample):\n{bvecs[:5]}")
    # else:
    #     logger.warning(f"Path not found: {bruker_path}. Cannot run example.")
    pass
