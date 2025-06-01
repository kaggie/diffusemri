import os
import logging
import json # For metadata saving
import numpy as np
import pydicom
import nibabel as nib
from pydicom.errors import InvalidDicomError

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def read_dicom_series(dicom_dir: str) -> list[pydicom.FileDataset]:
    """
    Reads all DICOM files from a directory, sorts them, and returns a list of datasets.

    Args:
        dicom_dir (str): Path to the directory containing DICOM files.

    Returns:
        list[pydicom.FileDataset]: A sorted list of pydicom.FileDataset objects.
                                   Returns an empty list if directory is not found,
                                   contains no DICOM files, or an error occurs.
    """
    if not os.path.isdir(dicom_dir):
        logger.error(f"DICOM directory not found: {dicom_dir}")
        return []

    dicom_datasets: list[pydicom.FileDataset] = []
    filepaths: list[str] = []

    logger.info(f"Reading DICOM files from directory: {dicom_dir}")
    for root, _, files in os.walk(dicom_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                # Attempt to read the file as DICOM.
                # pydicom.dcmread raises InvalidDicomError if it's not a DICOM file
                # or if it's a DICOM file that cannot be parsed.
                ds = pydicom.dcmread(filepath, force=True) # force=True to try reading non-conformant files

                # Basic check for PixelData, as we are interested in image series.
                # Some DICOM files might be structured reports, etc.
                if 'PixelData' not in ds:
                    logger.debug(f"Skipping non-image DICOM file (missing PixelData): {filepath}")
                    continue

                dicom_datasets.append(ds)
                filepaths.append(filepath) # Keep track of original filepaths for sorting if needed
            except InvalidDicomError:
                logger.debug(f"Skipping non-DICOM or invalid DICOM file: {filepath}")
            except Exception as e:
                logger.warning(f"Could not read or parse file {filepath} as DICOM: {e}")

    if not dicom_datasets:
        logger.warning(f"No valid DICOM files found in directory: {dicom_dir}")
        return []

    # Sort the datasets.
    # Primary sort key: InstanceNumber (0020,0013)
    # Secondary sort key (fallback): AcquisitionNumber (0020,0012)
    # Tertiary sort key (fallback): Filename (if InstanceNumber is missing or inconsistent)

    # Check if InstanceNumber is reliably present for sorting
    has_instance_number = all(hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None for ds in dicom_datasets)
    has_acq_number = all(hasattr(ds, 'AcquisitionNumber') and ds.AcquisitionNumber is not None for ds in dicom_datasets)

    if has_instance_number:
        logger.info("Sorting DICOM series by InstanceNumber.")
        dicom_datasets.sort(key=lambda ds: int(ds.InstanceNumber))
    elif has_acq_number:
        logger.warning("InstanceNumber missing or inconsistent across DICOM files. Attempting to sort by AcquisitionNumber.")
        # If InstanceNumber is missing, AcquisitionNumber might be used, but it's less standard for slice ordering.
        # Sometimes AcquisitionTime is also useful if slice order is temporal.
        # For now, let's try AcquisitionNumber if InstanceNumber is not good.
        dicom_datasets.sort(key=lambda ds: int(ds.AcquisitionNumber))
    else:
        # Fallback to sorting by original file path if essential sorting tags are missing
        logger.warning("InstanceNumber and AcquisitionNumber are missing or inconsistent. "
                       "Attempting to sort by filename. This might not be accurate for slice order.")
        # Create a list of tuples (dataset, original_filepath)
        # This assumes filepaths were stored in the same order as dicom_datasets before sorting
        # To ensure this, we should sort by filepaths first, then if tags are available, re-sort.
        # Or, more simply, if we have to resort to filename sorting, use the filepaths list.

        # A more robust way for filename fallback:
        # Zip datasets with their filepaths, sort by filepath, then extract datasets.
        # This ensures that if we fall back to filename sorting, it's based on the actual files.
        # However, the current structure already appends to dicom_datasets and filepaths in order.
        # So, if we need to sort by filename, we can create pairs and sort.

        # For simplicity, if critical tags are missing, we'll sort based on the order files were read,
        # which is OS-dependent, or by filename directly.
        # Let's use the filepaths list captured in parallel with dicom_datasets.

        # Create pairs of (dataset, original_filepath_index) to ensure stable sort if names are same
        indexed_datasets = []
        for i, ds_item in enumerate(dicom_datasets):
            # Find the filepath for this ds_item. This is tricky if ds_items are not unique by content.
            # Assuming ds.filename (if available and unique) or original filepath list helps.
            # The simplest here is to assume the order of filepaths matches current dicom_datasets.
            indexed_datasets.append((ds_item, filepaths[i]))

        indexed_datasets.sort(key=lambda item: item[1]) # Sort by filepath
        dicom_datasets = [item[0] for item in indexed_datasets] # Extract sorted datasets


    logger.info(f"Successfully read and sorted {len(dicom_datasets)} DICOM datasets.")
    return dicom_datasets


def extract_pixel_data_and_affine(sorted_dicoms: list[pydicom.FileDataset]) \
        -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Extracts pixel data, constructs an affine matrix, and gathers basic metadata
    from a sorted list of DICOM datasets.

    Args:
        sorted_dicoms (list[pydicom.FileDataset]): A list of DICOM datasets,
            sorted by slice order (e.g., InstanceNumber).

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, dict]:
            - image_data (np.ndarray | None): The stacked pixel data (3D or 4D).
            - affine (np.ndarray | None): The constructed 4x4 affine matrix.
            - metadata (dict): Extracted metadata (TR, TE, FlipAngle, etc.).
            Returns (None, None, {}) if critical information is missing or inconsistent.
    """
    if not sorted_dicoms:
        logger.error("Cannot extract data: DICOM list is empty.")
        return None, None, {}

    # --- Consistency Checks (using the first DICOM as reference) ---
    ref_ds = sorted_dicoms[0]
    required_tags_series = ['SeriesInstanceUID', 'Rows', 'Columns', 'PixelSpacing',
                            'ImageOrientationPatient', 'SliceThickness']
    # ImagePositionPatient is per-slice, so not checked for series consistency here.

    for tag_name in required_tags_series:
        if not hasattr(ref_ds, tag_name):
            logger.error(f"Reference DICOM missing required tag for series consistency: {tag_name}")
            return None, None, {}

    series_uid = ref_ds.SeriesInstanceUID
    rows = ref_ds.Rows
    cols = ref_ds.Columns
    pixel_spacing = ref_ds.PixelSpacing # [RowSpacing, ColumnSpacing]
    image_orientation_patient = ref_ds.ImageOrientationPatient # [Rx, Ry, Rz, Cx, Cy, Cz]
    slice_thickness = ref_ds.SliceThickness

    for i, ds in enumerate(sorted_dicoms[1:]):
        if ds.SeriesInstanceUID != series_uid:
            logger.warning(f"Inconsistent SeriesInstanceUID at index {i+1}. Expected {series_uid}, got {ds.SeriesInstanceUID}.")
            # Allow processing but log warning, as sometimes series are split.
        if ds.Rows != rows or ds.Columns != cols:
            logger.error(f"Inconsistent Rows/Columns at index {i+1}. Expected ({rows},{cols}), got ({ds.Rows},{ds.Columns}).")
            return None, None, {}
        if list(ds.PixelSpacing) != list(pixel_spacing):
            logger.error(f"Inconsistent PixelSpacing at index {i+1}. Expected {pixel_spacing}, got {ds.PixelSpacing}.")
            return None, None, {}
        if list(ds.ImageOrientationPatient) != list(image_orientation_patient):
            logger.error(f"Inconsistent ImageOrientationPatient at index {i+1}.")
            return None, None, {}
        if ds.SliceThickness != slice_thickness:
            logger.warning(f"Inconsistent SliceThickness at index {i+1}. Expected {slice_thickness}, got {ds.SliceThickness}. Using first slice's thickness.")
            # Not necessarily fatal, could be variable slice thickness, but affine construction assumes constant for now.

    # --- Extract Pixel Data ---
    num_slices = len(sorted_dicoms)
    image_shape = (int(rows), int(cols), int(num_slices)) # Explicitly cast to int

    # Determine data type from first DICOM, assume consistency
    # BitsAllocated, PixelRepresentation (0=unsigned, 1=signed), SamplesPerPixel
    # For simplicity, assume MONOCHROME2 and single sample per pixel.
    # More complex logic needed for RGB, multi-sample, different bit depths.
    # pydicom.pixel_data_handlers.util.apply_modality_lut performs rescale/intercept
    # pydicom.pixel_data_handlers.util.apply_voi_lut performs windowing (not usually needed for NIfTI conversion)

    # Use pixel_array which applies RescaleSlope, RescaleIntercept, and handles bit depth.
    try:
        first_pixel_array = ref_ds.pixel_array
        image_dtype = first_pixel_array.dtype
    except Exception as e:
        logger.error(f"Failed to get pixel_array from reference DICOM: {e}")
        return None, None, {}

    # Initialize volume
    volume_data = np.zeros(image_shape, dtype=image_dtype)

    for i, ds in enumerate(sorted_dicoms):
        try:
            slice_data = ds.pixel_array
            if slice_data.shape != (rows, cols):
                logger.error(f"Slice {i} data shape {slice_data.shape} mismatch with expected ({rows},{cols}).")
                return None, None, {}
            volume_data[:, :, i] = slice_data
        except Exception as e:
            logger.error(f"Error processing pixel data for slice {i} ({ds.filename if hasattr(ds, 'filename') else 'Unknown'}): {e}")
            return None, None, {}

    logger.info(f"Successfully stacked {num_slices} slices into a volume of shape {volume_data.shape}.")

    # --- Construct Affine Matrix ---
    # This is a simplified construction. For robust conversion, especially with
    # oblique slices or gantry tilt, a more sophisticated method like that in
    # nibabel.nicom.dicomreaders.mosaic_to_nii or dcm2niix is recommended.

    # ImagePositionPatient of the first slice (X, Y, Z coordinates of the center of the first voxel)
    try:
        pos_first_slice = np.array(ref_ds.ImagePositionPatient, dtype=float)
    except AttributeError:
        logger.error("Missing ImagePositionPatient in the first DICOM slice. Cannot construct affine.")
        return volume_data, None, {} # Return data even if affine fails, but log

    # ImageOrientationPatient: [Rx, Ry, Rz, Cx, Cy, Cz]
    # Rx,Ry,Rz are components of the first row vector (direction cosines)
    # Cx,Cy,Cz are components of the first column vector (direction cosines)
    row_cosine = np.array(image_orientation_patient[:3], dtype=float)
    col_cosine = np.array(image_orientation_patient[3:], dtype=float)

    # Pixel spacing: [RowSpacing, ColumnSpacing]
    delta_r, delta_c = float(pixel_spacing[0]), float(pixel_spacing[1])

    # Slice thickness (used for spacing between slices along Z_image axis)
    # For slice_spacing, if SliceLocation is available and varies, can calculate more accurately.
    # Otherwise, use SliceThickness or SpacingBetweenSlices.
    slice_spacing = float(slice_thickness)
    if hasattr(ref_ds, 'SpacingBetweenSlices') and ref_ds.SpacingBetweenSlices is not None:
        slice_spacing = float(ref_ds.SpacingBetweenSlices)
        logger.info(f"Using SpacingBetweenSlices ({slice_spacing}mm) for Z-axis spacing.")
    else:
        logger.info(f"Using SliceThickness ({slice_spacing}mm) for Z-axis spacing.")

    # Construct the rotational part of the affine
    affine = np.eye(4, dtype=float)
    affine[0, 0] = row_cosine[0] * delta_c # X-component of column vector * ColSpacing
    affine[1, 0] = row_cosine[1] * delta_c # Y-component of column vector * ColSpacing
    affine[2, 0] = row_cosine[2] * delta_c # Z-component of column vector * ColSpacing

    affine[0, 1] = col_cosine[0] * delta_r # X-component of row vector * RowSpacing
    affine[1, 1] = col_cosine[1] * delta_r # Y-component of row vector * RowSpacing
    affine[2, 1] = col_cosine[2] * delta_r # Z-component of row vector * RowSpacing

    # Calculate Z-axis vector (cross product of row and column cosines)
    # This is the direction normal to the slices.
    z_vector = np.cross(row_cosine, col_cosine)
    z_vector_normalized = z_vector / (np.linalg.norm(z_vector) or 1.0) # Avoid div by zero

    affine[0, 2] = z_vector_normalized[0] * slice_spacing
    affine[1, 2] = z_vector_normalized[1] * slice_spacing
    affine[2, 2] = z_vector_normalized[2] * slice_spacing

    # Set translation part (origin)
    affine[:3, 3] = pos_first_slice

    logger.info(f"Constructed affine matrix:\n{affine}")

    # --- Extract Basic Metadata (from the first DICOM file) ---
    metadata = {}
    metadata_tags = {
        'RepetitionTime': 'TR',
        'EchoTime': 'TE',
        'FlipAngle': 'FlipAngle',
        'PixelBandwidth': 'PixelBandwidth',
        'Manufacturer': 'Manufacturer',
        'ManufacturerModelName': 'ManufacturerModelName',
        'AcquisitionDate': 'AcquisitionDate',
        'AcquisitionTime': 'AcquisitionTime',
        'SeriesDescription': 'SeriesDescription',
        'ProtocolName': 'ProtocolName',
        'ScanningSequence': 'ScanningSequence', # e.g., GR, SE, EP
        'SequenceVariant': 'SequenceVariant',   # e.g., SK, SP, MP
    }
    for dcm_tag, meta_key in metadata_tags.items():
        if hasattr(ref_ds, dcm_tag):
            value = getattr(ref_ds, dcm_tag)
            # pydicom values can be complex objects, try to convert to simple types for JSON
            if isinstance(value, pydicom.multival.MultiValue):
                metadata[meta_key] = [str(v) for v in value] if value else None
            elif isinstance(value, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)) : # DecimalString, IntegerString
                 metadata[meta_key] = float(value) if '.' in str(value) else int(value)
            elif isinstance(value, (float, int, str)):
                metadata[meta_key] = value
            else: # For other types like PersonName, Date, Time, etc.
                metadata[meta_key] = str(value)
        else:
            metadata[meta_key] = None # Tag not present

    logger.info(f"Extracted metadata: {metadata}")

    # Note: This implementation assumes a single 3D volume.
    # If the DICOM series represents multiple volumes (e.g., fMRI, multi-echo),
    # volume_data would need to be 4D, and affine logic might need adjustment
    # or be assumed constant across volumes if they are part of the same acquisition.
    # For now, it assumes slices form one 3D volume.
    # If there are multiple temporal positions or echo times, they should be handled.
    # Check for 'NumberOfTemporalPositions' or multiple 'EchoTime' values if expecting 4D.

    # If multiple temporal positions exist, reshape.
    # This is a very basic check. Real 4D DICOM (fMRI, multi-echo) needs more robust handling.
    num_temporal_positions = getattr(ref_ds, 'NumberOfTemporalPositions', 1)
    if num_temporal_positions > 1 and num_slices % num_temporal_positions == 0:
        num_actual_slices_per_volume = num_slices // num_temporal_positions
        logger.info(f"Detected {num_temporal_positions} temporal positions. Reshaping volume.")
        new_shape = (int(rows), int(cols), int(num_actual_slices_per_volume), int(num_temporal_positions))
        try:
            volume_data = volume_data.reshape(new_shape, order='F') # Fortran order for slice-by-slice stacking
            logger.info(f"Reshaped volume data to: {volume_data.shape}")
        except ValueError as e:
            logger.error(f"Failed to reshape volume data for temporal positions: {e}. Keeping as 3D stack.")
            # If reshape fails, it remains a 3D stack of all slices.

    return volume_data, affine, metadata


def extract_dwi_metadata(sorted_dicoms: list[pydicom.FileDataset], image_orientation_patient: list[float]) \
        -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """
    Extracts DWI-specific metadata (b-values, b-vectors) from a sorted list of DICOM datasets.

    Args:
        sorted_dicoms (list[pydicom.FileDataset]): A list of DICOM datasets, sorted.
        image_orientation_patient (list[float]): The ImageOrientationPatient tag value from a reference DICOM,
                                                used for b-vector reorientation.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None, dict]:
            - b_values (np.ndarray | None): Array of b-values.
            - b_vectors (np.ndarray | None): Array of b-vectors (Nx3, reoriented to image space).
            - dwi_metadata (dict): Dictionary of other DWI metadata.
    """
    if not sorted_dicoms:
        return None, None, {}

    b_values_list = []
    b_vectors_list = [] # Store raw b-vectors first
    dwi_metadata = {}

    # Standard DICOM tags for diffusion
    bval_tag = (0x0018, 0x9087) # DiffusionBValue
    bvec_tag = (0x0018, 0x9089) # DiffusionGradientOrientation

    # Siemens specific handling often involves CSA headers
    # Example: Accessing Siemens Shadow Header (0029,1010) - "CSAImageHeaderInfo"
    #          and (0029,1020) - "CSASeriesHeaderInfo"
    # These are private tags and parsing them is complex.
    # For now, we'll focus on standard tags and make notes for vendor specifics.

    logger.info("Attempting to extract DWI metadata (b-values, b-vectors)...")
    for i, ds in enumerate(sorted_dicoms):
        b_value = None
        b_vector = None

        # Standard B-value tag
        if bval_tag in ds:
            b_value = float(ds[bval_tag].value)
        else:
            # Placeholder for Siemens B-value extraction from CSA
            # E.g., if 'B_value' in ds.CSASeriesHeaderInfo (pseudo-code)
            # This would require parsing the CSA header, which is non-trivial.
            # For now, if standard tag is missing, assume it might be a b0 or data is incomplete.
            logger.debug(f"Standard DiffusionBValue (0018,9087) not found in DICOM slice {i}. Assuming b-value=0 if no other source.")
            # If it's a b0, b_value should be ~0. If other tags provide it, they'll override.

        # Standard B-vector tag
        if bvec_tag in ds:
            b_vector_raw = np.array(ds[bvec_tag].value, dtype=float)
            if b_vector_raw.shape == (3,):
                b_vector = b_vector_raw
            else:
                logger.warning(f"DiffusionGradientOrientation (0018,9089) in slice {i} has unexpected shape {b_vector_raw.shape}. Expected (3,).")
        else:
            # Placeholder for Siemens B-vector extraction
            logger.debug(f"Standard DiffusionGradientOrientation (0018,9089) not found in DICOM slice {i}.")

        # If b_value is effectively zero, b_vector should be [0,0,0]
        if b_value is not None and b_value < 50: # Assuming a common b0 threshold
            b_vector = np.array([0.0, 0.0, 0.0])
            if b_value < 5: b_value = 0.0 # Clean up small b-values for b0s
        elif b_vector is None: # If it's a DWI but b-vector is missing
            logger.warning(f"Slice {i} appears to be a DWI (b-value={b_value}) but b-vector is missing. Setting b-vector to [0,0,0].")
            b_vector = np.array([0.0, 0.0, 0.0]) # Fallback, though this makes it a b0 effectively

        if b_value is None: # If after all attempts, b_value is still None, default to 0
            b_value = 0.0
            if b_vector is None: b_vector = np.array([0.0, 0.0, 0.0])


        b_values_list.append(b_value)
        if b_vector is not None:
            b_vectors_list.append(b_vector)
        else: # Should have been handled by fallback above
             b_vectors_list.append(np.array([0.,0.,0.]))


    b_values_np = np.array(b_values_list, dtype=float)
    b_vectors_raw_np = np.array(b_vectors_list, dtype=float)

    if b_vectors_raw_np.shape[0] != len(sorted_dicoms) or b_vectors_raw_np.shape[1] != 3:
        logger.error(f"Final b-vectors array shape mismatch. Expected ({len(sorted_dicoms)}, 3), got {b_vectors_raw_np.shape}")
        return b_values_np, None, dwi_metadata

    # --- B-vector reorientation ---
    # DICOM b-vectors are typically in the patient coordinate system (LPH or LPS).
    # NIfTI standard (RAS+) requires b-vectors to be in the image coordinate system's frame of reference
    # if they are to be used directly with tools like FSL.
    # This means rotating them by the inverse of the patient-to-image orientation part of the affine.
    # More simply, if ImageOrientationPatient defines the orientation of image axes in PCS,
    # then b-vectors in PCS need to be projected onto these image axes.

    # ImageOrientationPatient: [Rx, Ry, Rz, Cx, Cy, Cz]
    row_cosines = np.array(image_orientation_patient[:3], dtype=float)
    col_cosines = np.array(image_orientation_patient[3:], dtype=float)

    # Create rotation matrix from image orientation
    # This matrix transforms coordinates from image space to patient space
    # R = [Rx*dc, Cx*dr, Nx*ds; Ry*dc, Cy*dr, Ny*ds; Rz*dc, Cz*dr, Nz*ds] where N is cross(Row, Col)
    # We need the inverse of this rotation to bring b-vectors from patient to image space.
    # If b-vectors are defined in PCS, and image axes are defined by IOP,
    # then b_img = R_patient_to_image * b_patient.
    # R_patient_to_image is derived from IOP. Let M_iop be the matrix whose columns are [row_cosines, col_cosines, cross_product].
    # Then b_vec_image = inv(M_iop) * b_vec_patient (if b_vec_patient is col vector)
    # Or b_vec_image_row = b_vec_patient_row * inv(M_iop_row_vectors)

    # Let's use a simpler approach common in converters:
    # If M is the rotation part of the DICOM affine (first 3x3 block, without scaling by pixel spacing),
    # then b_image = M.T @ b_patient_dicom. (or inv(M) @ b_patient_dicom)
    # The exact transformation depends on the definition of b-vectors in DICOM standard for that specific sequence.
    # For now, assume b-vectors from (0018,9089) are in PCS.

    # A common approach is to create a rotation matrix from ImageOrientationPatient:
    # Col1 = RowCosines (direction of X axis of image in patient coords)
    # Col2 = ColCosines (direction of Y axis of image in patient coords)
    # Col3 = CrossProduct(RowCosines, ColCosines) (direction of Z axis of image in patient coords)
    # This matrix R transforms from image space to patient space: P = R @ I
    # So, to transform a vector from patient space to image space: I = inv(R) @ P = R.T @ P

    rotation_matrix_img_to_pat = np.zeros((3,3), dtype=float)
    rotation_matrix_img_to_pat[:,0] = row_cosines
    rotation_matrix_img_to_pat[:,1] = col_cosines
    rotation_matrix_img_to_pat[:,2] = np.cross(row_cosines, col_cosines)

    # Check if rotation_matrix is valid (e.g. determinant close to 1)
    if abs(np.linalg.det(rotation_matrix_img_to_pat)) < 1e-3:
        logger.warning("Singular rotation matrix from ImageOrientationPatient. B-vector reorientation may be incorrect.")
        # Fallback: use identity, meaning b-vectors are assumed to be already in image space or reorientation failed.
        rotation_matrix_pat_to_img = np.eye(3)
    else:
        try:
            rotation_matrix_pat_to_img = np.linalg.inv(rotation_matrix_img_to_pat)
        except np.linalg.LinAlgError:
            logger.warning("Could not invert rotation matrix from ImageOrientationPatient. Using pseudo-inverse.")
            rotation_matrix_pat_to_img = np.linalg.pinv(rotation_matrix_img_to_pat)


    b_vectors_reoriented_list = []
    for b_vec_patient in b_vectors_raw_np:
        # Only reorient non-zero b-vectors
        if np.any(np.abs(b_vec_patient) > 1e-6): # If it's not a zero vector
            b_vec_image = rotation_matrix_pat_to_img @ b_vec_patient
            # Normalize again after rotation, as rotation matrix might not be perfectly orthogonal due to source data
            norm = np.linalg.norm(b_vec_image)
            if norm > 1e-6:
                b_vec_image /= norm
            else: # Should not happen if original b_vec_patient was non-zero and rotation is valid
                b_vec_image = np.array([0.,0.,0.])
            b_vectors_reoriented_list.append(b_vec_image)
        else:
            b_vectors_reoriented_list.append(np.array([0.,0.,0.])) # Keep zero vectors as is

    b_vectors_reoriented_np = np.array(b_vectors_reoriented_list, dtype=float)
    logger.info(f"Reoriented b-vectors to image space. Example original: {b_vectors_raw_np[b_values_np > 50][0] if np.any(b_values_np > 50) else 'N/A'}, "
                f"Example reoriented: {b_vectors_reoriented_np[b_values_np > 50][0] if np.any(b_values_np > 50) else 'N/A'}")

    dwi_metadata['NumberOfDiffusionDirections'] = int(np.sum(b_values_np > 50.0)) # Example threshold for DWIs
    # Further metadata like number of shells could be derived from unique b-values.

    return b_values_np, b_vectors_reoriented_np, dwi_metadata


# --- DICOM Anonymization ---

# Define special markers for actions
_REMOVE_TAG_ = object() # Sentinel object for removing a tag
_EMPTY_STRING_ = object() # Sentinel for setting to empty string
_ZERO_STRING_ = object() # Sentinel for setting to "0"
_DEFAULT_DATE_ = object() # Sentinel for setting to "19000101"
_DEFAULT_TIME_ = object() # Sentinel for setting to "000000"

DEFAULT_ANONYMIZATION_TAGS = {
    # Patient Identifying Information
    'PatientName': _EMPTY_STRING_,
    'PatientID': _ZERO_STRING_,
    'PatientBirthDate': _DEFAULT_DATE_,
    'PatientSex': _EMPTY_STRING_,
    'PatientAge': _EMPTY_STRING_,
    'PatientAddress': _REMOVE_TAG_,
    'PatientTelephoneNumbers': _REMOVE_TAG_,
    'OtherPatientIDs': _REMOVE_TAG_,
    'OtherPatientNames': _REMOVE_TAG_,
    'PatientBirthName': _REMOVE_TAG_,
    'PatientMotherBirthName': _REMOVE_TAG_,

    # Study Information (some might be PII or linkable)
    'StudyDate': _DEFAULT_DATE_, # Or a function to shift dates
    'SeriesDate': _DEFAULT_DATE_,
    'AcquisitionDate': _DEFAULT_DATE_,
    'ContentDate': _DEFAULT_DATE_,
    'StudyTime': _DEFAULT_TIME_,
    'SeriesTime': _DEFAULT_TIME_,
    'AcquisitionTime': _DEFAULT_TIME_,
    'ContentTime': _DEFAULT_TIME_,
    'AccessionNumber': _REMOVE_TAG_,
    'ReferringPhysicianName': _EMPTY_STRING_,
    'PhysiciansOfRecord': _EMPTY_STRING_,
    'PerformingPhysicianName': _EMPTY_STRING_,
    'NameOfPhysiciansReadingStudy': _EMPTY_STRING_,
    'OperatorsName': _EMPTY_STRING_,
    'InstitutionalDepartmentName': _EMPTY_STRING_,
    'InstitutionName': _EMPTY_STRING_,
    'InstitutionAddress': _REMOVE_TAG_,

    # Equipment Information (less likely PII, but can be identifying)
    # 'Manufacturer': _EMPTY_STRING_, # Often kept for compatibility
    # 'ManufacturerModelName': _EMPTY_STRING_, # Often kept
    'StationName': _EMPTY_STRING_,
    'DeviceSerialNumber': _EMPTY_STRING_,

    # UID related tags - UIDs should generally be replaced with new ones if anonymizing.
    # For simplicity here, we might remove some potentially linkable study/series UIDs
    # but SOPInstanceUID must be kept or regenerated.
    # For now, focusing on PII text fields. UID handling is complex.
    # 'StudyInstanceUID': _REMOVE_TAG_, # Potentially regenerate
    # 'SeriesInstanceUID': _REMOVE_TAG_, # Potentially regenerate

    # Other potentially identifying information
    'RequestingPhysician': _EMPTY_STRING_,
    'PerformedProcedureStepDescription': _EMPTY_STRING_,
    'CommentsOnPerformedProcedureStep': _EMPTY_STRING_,
    'ImageComments': _EMPTY_STRING_,
    'ProtocolName': _EMPTY_STRING_, # Can contain identifying info
    'StudyDescription': _EMPTY_STRING_,
    # 'SeriesDescription': _EMPTY_STRING_, # Often useful, e.g. "FLAIR", "T1w" - decide if anonymize

    # Curves and Overlays - these can sometimes contain burned-in annotations
    # (0x50xx, 0xxxx) groups for curves - remove all by default
    # (0x60xx, 0xxxx) groups for overlays - remove all by default
    # This requires more specific logic to iterate and remove groups.
    # For now, specific tags in these groups if known.
    # (0070,0001) GraphicAnnotationSequence - remove
    (0x0070, 0x0001): _REMOVE_TAG_,


    # Tags that should NOT be removed by default (essential for image interpretation/processing)
    # PixelData, ImageOrientationPatient, ImagePositionPatient, PixelSpacing, SliceThickness,
    # Rows, Columns, SamplesPerPixel, PhotometricInterpretation, BitsAllocated, BitsStored, HighBit,
    # WindowCenter, WindowWidth, RescaleIntercept, RescaleSlope,
    # SpecificCharacterSet (handle with care),
    # SOPClassUID, SOPInstanceUID (must be preserved or regenerated carefully),
    # TransferSyntaxUID, Modality,
    # DiffusionBValue (0018,9087), DiffusionGradientOrientation (0018,9089)
    # and vendor-specific diffusion tags.
}


def anonymize_dicom_dataset(dataset: pydicom.FileDataset,
                            anonymization_rules: dict = None,
                            custom_replacements: dict = None) -> None:
    """
    Anonymizes a DICOM dataset in-place based on specified rules.

    Args:
        dataset (pydicom.FileDataset): The DICOM dataset to anonymize.
        anonymization_rules (dict, optional): Rules for anonymization.
            Keys are DICOM tag keywords (str) or (group, element) tuples.
            Values are replacement values or special markers like _REMOVE_TAG_.
            If None, `DEFAULT_ANONYMIZATION_TAGS` is used.
        custom_replacements (dict, optional): Dictionary of functions for custom
            tag value modifications. Keys are tag keywords/tuples, values are functions
            `f(original_value) -> new_value`.
    """
    rules = anonymization_rules if anonymization_rules is not None else DEFAULT_ANONYMIZATION_TAGS
    if custom_replacements is None:
        custom_replacements = {}

    for tag_key, action_or_value in rules.items():
        tag_address = None
        if isinstance(tag_key, str): # Keyword
            try:
                # pydicom keywords might not always match the exact casing.
                # Iterate through dataset elements to find matching keyword (case-insensitive)
                found_tag = False
                for elem in dataset:
                    if elem.keyword == tag_key:
                        tag_address = elem.tag
                        found_tag = True
                        break
                if not found_tag:
                    # logger.debug(f"Tag keyword '{tag_key}' not found in dataset.")
                    continue
            except Exception: # Broad exception if keyword lookup itself fails for some reason
                logger.warning(f"Could not resolve keyword '{tag_key}' to a tag address.")
                continue
        elif isinstance(tag_key, tuple) and len(tag_key) == 2: # (group, element) tuple
            tag_address = tag_key
        else:
            logger.warning(f"Invalid tag key format in rules: {tag_key}. Must be keyword string or (group,element) tuple.")
            continue

        if tag_address not in dataset:
            # logger.debug(f"Tag {tag_address} (from key '{tag_key}') not found in dataset.")
            continue

        try:
            if tag_key in custom_replacements: # Custom function takes precedence
                original_value = dataset[tag_address].value
                new_value = custom_replacements[tag_key](original_value)
                dataset[tag_address].value = new_value
                logger.debug(f"Applied custom replacement for tag {tag_address} ('{tag_key}').")
            elif action_or_value is _REMOVE_TAG_:
                del dataset[tag_address]
                logger.debug(f"Removed tag {tag_address} ('{tag_key}').")
            elif action_or_value is _EMPTY_STRING_:
                dataset[tag_address].value = ""
                logger.debug(f"Emptied tag {tag_address} ('{tag_key}').")
            elif action_or_value is _ZERO_STRING_:
                dataset[tag_address].value = "0" # Usually for IDs like PatientID
                logger.debug(f"Zeroed tag {tag_address} ('{tag_key}').")
            elif action_or_value is _DEFAULT_DATE_:
                dataset[tag_address].value = "19000101"
                logger.debug(f"Set tag {tag_address} ('{tag_key}') to default date.")
            elif action_or_value is _DEFAULT_TIME_:
                dataset[tag_address].value = "000000"
                logger.debug(f"Set tag {tag_address} ('{tag_key}') to default time.")
            else: # Direct replacement with the provided value
                dataset[tag_address].value = action_or_value
                logger.debug(f"Replaced tag {tag_address} ('{tag_key}') with value '{action_or_value}'.")
        except Exception as e:
            logger.error(f"Error processing tag {tag_address} ('{tag_key}'): {e}")

    # Example: Remove specific sequences if they exist and are known to contain PII
    # This needs to be done carefully based on actual DICOM structure.
    # For instance, ReferencedPerformedProcedureStepSequence (0008,1111)
    if (0x0008, 0x1111) in dataset:
        try:
            del dataset[(0x0008, 0x1111)]
            logger.info("Removed ReferencedPerformedProcedureStepSequence (0008,1111).")
        except KeyError:
            pass # Tag wasn't there

    # Update SOPInstanceUID with a new one if required by anonymization profile
    # This is critical for true de-identification but complex.
    # For this example, we are not regenerating UIDs by default.
    # dataset.SOPInstanceUID = pydicom.uid.generate_uid()

    # Modify PatientID to ensure it's not empty if it was just set to "" by rules
    # and an empty string is not desired for this specific field by some systems.
    if 'PatientID' in rules and rules['PatientID'] is _EMPTY_STRING_ and hasattr(dataset, 'PatientID') and not dataset.PatientID:
        dataset.PatientID = "00000000" # A common placeholder
        logger.debug("Set PatientID to default placeholder '00000000' as it was emptied by rules.")

    # Ensure SpecificCharacterSet is appropriate if string values were changed significantly.
    # For simplicity, we assume ASCII or compatible characters for replacements.
    # If characters outside ASCII are used for replacement, SpecificCharacterSet might need update.
    # For now, we do not modify SpecificCharacterSet.
    if 'SpecificCharacterSet' in dataset:
        logger.info(f"SpecificCharacterSet is '{dataset.SpecificCharacterSet}'. Ensure replacement strings are compatible.")


def anonymize_dicom_file(input_dicom_path: str, output_dicom_path: str,
                         anonymization_rules: dict = None,
                         custom_replacements: dict = None) -> bool:
    """
    Reads a DICOM file, anonymizes its dataset, and saves it to a new file.

    Args:
        input_dicom_path (str): Path to the input DICOM file.
        output_dicom_path (str): Path to save the anonymized DICOM file.
        anonymization_rules (dict, optional): Rules for anonymization.
            Passed to `anonymize_dicom_dataset`.
        custom_replacements (dict, optional): Custom replacement functions.
            Passed to `anonymize_dicom_dataset`.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not os.path.exists(input_dicom_path):
        logger.error(f"Input DICOM file not found: {input_dicom_path}")
        return False

    try:
        dataset = pydicom.dcmread(input_dicom_path)
    except Exception as e:
        logger.error(f"Failed to read DICOM file {input_dicom_path}: {e}")
        return False

    # Anonymize the dataset in-place
    anonymize_dicom_dataset(dataset, anonymization_rules, custom_replacements)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_dicom_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory for anonymized file: {output_dir}")
        except OSError as e:
            logger.error(f"Could not create output directory {output_dir}: {e}")
            return False

    try:
        dataset.save_as(output_dicom_path)
        logger.info(f"Anonymized DICOM file saved to: {output_dicom_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save anonymized DICOM file {output_dicom_path}: {e}")
        return False


def anonymize_dicom_directory(input_dir: str, output_dir: str,
                              anonymization_rules: dict = None,
                              custom_replacements: dict = None,
                              preserve_structure: bool = True) -> tuple[int, int]:
    """
    Scans an input directory for DICOM files, anonymizes them, and saves them
    to an output directory, optionally preserving the subdirectory structure.

    Args:
        input_dir (str): Path to the root directory containing DICOM files.
        output_dir (str): Path to the root directory where anonymized DICOM files
                          will be saved.
        anonymization_rules (dict, optional): Rules for anonymization.
        custom_replacements (dict, optional): Custom replacement functions.
        preserve_structure (bool, optional): If True, replicates the subdirectory
            structure from `input_dir` within `output_dir`. Defaults to True.

    Returns:
        tuple[int, int]: A tuple containing:
            - number_of_files_processed (int)
            - number_of_files_failed (int)
    """
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return 0, 0

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created base output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Could not create base output directory {output_dir}: {e}")
            return 0, 0

    processed_count = 0
    failed_count = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_filepath = os.path.join(root, filename)

            # Attempt to read as DICOM to confirm it's a DICOM file before anonymizing
            try:
                pydicom.dcmread(input_filepath, stop_before_pixels=True) # Quick check
            except InvalidDicomError:
                logger.debug(f"Skipping non-DICOM file during anonymization scan: {input_filepath}")
                continue
            except Exception: # Other read errors
                logger.warning(f"Could not quickly verify DICOM format for {input_filepath}, attempting full read in anonymize_dicom_file.")


            relative_path = os.path.relpath(root, input_dir)
            current_output_dir = output_dir
            if preserve_structure and relative_path != '.':
                current_output_dir = os.path.join(output_dir, relative_path)

            if not os.path.exists(current_output_dir):
                try:
                    os.makedirs(current_output_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"Could not create subdirectory {current_output_dir}: {e}")
                    failed_count += 1
                    continue # Skip this file if its output dir can't be made

            output_filepath = os.path.join(current_output_dir, filename)

            if anonymize_dicom_file(input_filepath, output_filepath,
                                    anonymization_rules, custom_replacements):
                processed_count += 1
            else:
                failed_count += 1
                logger.error(f"Failed to anonymize file: {input_filepath}")

    logger.info(f"DICOM directory anonymization complete. Processed: {processed_count}, Failed: {failed_count}")
    return processed_count, failed_count


def convert_dwi_dicom_to_nifti(dicom_dir: str, output_nifti_file: str,
                               output_bval_file: str, output_bvec_file: str) -> bool:
    """
    Converts a DICOM series of DWI data to NIfTI format, along with bval/bvec files.

    Args:
        dicom_dir (str): Path to the directory containing the DICOM series.
        output_nifti_file (str): Path to save the output NIfTI file.
        output_bval_file (str): Path to save the output b-values file.
        output_bvec_file (str): Path to save the output b-vectors file.
                                A JSON sidecar for metadata will also be saved
                                with the same base name as the NIfTI file.
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    logger.info(f"Starting DWI DICOM to NIfTI conversion for directory: {dicom_dir}")
    logger.info(f"Output NIfTI: {output_nifti_file}")
    logger.info(f"Output bval: {output_bval_file}")
    logger.info(f"Output bvec: {output_bvec_file}")

    # 1. Read and sort DICOM series
    sorted_datasets = read_dicom_series(dicom_dir)
    if not sorted_datasets:
        logger.error(f"Failed to read or sort DICOM series from {dicom_dir}.")
        return False

    # 2. Extract pixel data, affine, and basic metadata
    # extract_pixel_data_and_affine is expected to handle 4D volume if NumberOfTemporalPositions is > 1
    # For DWI, the 4th dimension is typically gradient directions, not time, so NumberOfTemporalPositions might be 1.
    # The number of datasets in sorted_dicoms will correspond to the number of volumes.
    image_data, affine_matrix, basic_metadata = extract_pixel_data_and_affine(sorted_datasets)

    if image_data is None:
        logger.error("Failed to extract image data from DICOM series.")
        return False
    if image_data.ndim != 4:
        # If extract_pixel_data_and_affine produced a 3D volume but we expect 4D for DWI (slices stacked into volumes)
        # this might indicate an issue or that the data is single-volume DWI (unlikely for series).
        # The current logic in extract_pixel_data_and_affine stacks all slices into the 3rd dim,
        # then reshapes to 4D if NumberOfTemporalPositions > 1.
        # For DWI, the "volumes" are implicitly the number of DICOM files if each is a 3D volume.
        # If each DICOM is a slice, then image_data will be (H, W, NumSlices).
        # We need to ensure image_data is (H, W, NumSlices, NumGradients) or similar before transposing for NIfTI.
        # The previous implementation of extract_pixel_data_and_affine forms a (H,W,Slices) volume.
        # For DWI, if each file is one volume (already 3D), then len(sorted_dicoms) is NumGradients,
        # and image_data would need to be stacked differently.
        # Let's assume for now `extract_pixel_data_and_affine` correctly gives a 4D [H,W,Slices,NumGradients]
        # if `NumberOfTemporalPositions` was used, or if it returns a 3D [H,W,Slices*NumGradients]
        # which needs to be reshaped based on len(b_values).
        # This part is tricky and depends heavily on how DICOMs are structured (mosaic vs. slice-by-slice vs. volume-by-volume).
        # For now, we trust `extract_pixel_data_and_affine` to return a stack that, after transpose, is (X,Y,Z,NumVolumes).
        # If it's (X,Y,Z), it means it's likely a single volume (e.g. mean DWI, not a series for bval/bvec).
        logger.warning(f"Expected 4D image data for DWI, but got {image_data.ndim}D. Attempting to proceed.")
        # If it's 3D, it might be a single volume that needs bval/bvec for that single volume.
        # This logic might need refinement based on typical DWI DICOM structures.

    if affine_matrix is None:
        logger.warning("Failed to construct affine matrix. NIfTI header will be less informative.")
        affine_matrix = np.eye(4)

    # 3. Extract DWI-specific metadata
    # Need ImageOrientationPatient from a reference DICOM for b-vector reorientation.
    ref_iop = getattr(sorted_dicoms[0], 'ImageOrientationPatient', [1.0,0.0,0.0,0.0,1.0,0.0]) # Default if missing
    b_values, b_vectors, dwi_specific_metadata = extract_dwi_metadata(sorted_datasets, ref_iop)

    if b_values is None or b_vectors is None:
        logger.error("Failed to extract b-values or b-vectors.")
        return False

    # Consistency check: number of volumes in image data vs. b-values/b-vectors
    # The last dimension of image_data (after potential reshape in extract_pixel_data_and_affine for time)
    # or the number of slices if it's a 3D stack of volumes, should match len(b_values).
    # If image_data is (H, W, Slices) and len(b_values) == Slices, it implies each slice is a volume.
    # If image_data is (H, W, SlicesPerVol, NumVols) and len(b_values) == NumVols.
    # The current extract_pixel_data_and_affine returns (H,W,Slices) or (H,W,SlicesPerVol, TimePos).
    # For DWI, we assume each DICOM file corresponds to one gradient direction / volume.
    # So, len(sorted_dicoms) should be the number of volumes.
    # And image_data should be (Rows, Cols, len(sorted_dicoms)) if each DICOM is a slice,
    # or (Rows, Cols, SlicesPerVolume, len(sorted_dicoms)) if each DICOM is a 3D volume (unlikely for series).
    # Let's assume image_data from extract_pixel_data_and_affine is (Rows, Cols, NumVolumes) if each DICOM is a slice.

    num_volumes_in_image = image_data.shape[2] if image_data.ndim == 3 else image_data.shape[3]
    if num_volumes_in_image != len(b_values):
        logger.error(f"Mismatch between number of image volumes ({num_volumes_in_image}) "
                     f"and number of b-values/b-vectors ({len(b_values)}).")
        logger.error(f"Image data shape: {image_data.shape}")
        return False

    # 4. Save NIfTI image
    try:
        if image_data.ndim == 3: # (rows, cols, num_volumes) -> means each slice was a volume
            image_data_nifti_order = image_data.transpose(1, 0, 2) # (cols, rows, num_volumes)
        elif image_data.ndim == 4: # (rows, cols, slices_per_vol, num_volumes) -> from temporal reshape
            image_data_nifti_order = image_data.transpose(1, 0, 2, 3) # (cols, rows, slices_per_vol, num_volumes)
        else:
            logger.error(f"Unsupported image data dimension for NIfTI saving: {image_data.ndim}")
            return False

        logger.info(f"Saving DWI NIfTI image with data shape {image_data_nifti_order.shape} and affine:\n{affine_matrix}")
        nifti_image = nib.Nifti1Image(image_data_nifti_order.astype(np.float32), affine_matrix) # Ensure float32

        output_dir = os.path.dirname(output_nifti_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        nib.save(nifti_image, output_nifti_file)
        logger.info(f"DWI NIfTI file saved successfully: {output_nifti_file}")

        # 5. Save b-values and b-vectors
        # Bvals: simple text file, space separated
        np.savetxt(output_bval_file, b_values.reshape(1, -1), fmt='%g') # Reshape to 1xN for single line
        logger.info(f"b-values saved to: {output_bval_file}")

        # Bvecs: text file, FSL format (3xN, space separated) or (Nx3)
        # We have Nx3 from extract_dwi_metadata. FSL prefers 3xN for some tools, but Nx3 is also common.
        # Let's save as Nx3 as it's more direct. Many tools handle both.
        # For FSL eddy, it expects bvecs with each column being a vector (3xN).
        # For Dipy, it's Nx3. Let's stick to Nx3 for now as it's common with Dipy/MRtrix.
        # If FSL compatibility is paramount, transpose: b_vectors.T
        np.savetxt(output_bvec_file, b_vectors, fmt='%.8f')
        logger.info(f"b-vectors saved to: {output_bvec_file} (format: Nx3)")

        # 6. Save combined metadata to JSON sidecar
        combined_metadata = {**basic_metadata, **dwi_specific_metadata}
        json_sidecar_file = output_nifti_file.replace(".nii.gz", ".json").replace(".nii", ".json")
        try:
            with open(json_sidecar_file, 'w') as f:
                json.dump(combined_metadata, f, indent=4)
            logger.info(f"Combined metadata JSON sidecar saved successfully: {json_sidecar_file}")
        except Exception as e:
            logger.error(f"Failed to save combined metadata JSON sidecar: {e}")

        return True

    except Exception as e:
        logger.error(f"An error occurred during DWI NIfTI creation or saving: {e}")
        return False


def convert_dicom_to_nifti_main(dicom_dir: str, output_nifti_file: str) -> bool:
    """
    Orchestrates DICOM series reading, data/affine extraction, and NIfTI conversion.
    Saves basic metadata to a JSON sidecar file.

    Args:
        dicom_dir (str): Path to the directory containing the DICOM series.
        output_nifti_file (str): Path to save the output NIfTI file.
                                 A JSON sidecar will be saved with the same base name.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    logger.info(f"Starting DICOM to NIfTI conversion for directory: {dicom_dir}")
    logger.info(f"Output NIfTI will be saved to: {output_nifti_file}")

    # 1. Read and sort DICOM series
    sorted_datasets = read_dicom_series(dicom_dir)
    if not sorted_datasets:
        logger.error(f"Failed to read or sort DICOM series from {dicom_dir}.")
        return False

    # 2. Extract pixel data, affine, and metadata
    image_data, affine_matrix, metadata = extract_pixel_data_and_affine(sorted_datasets)

    if image_data is None:
        logger.error("Failed to extract image data from DICOM series.")
        return False

    if affine_matrix is None:
        logger.warning("Failed to construct affine matrix. NIfTI header will be less informative.")
        # Continue with a default identity affine if desired, or fail.
        # For now, let's allow saving without a perfect affine but log it.
        affine_matrix = np.eye(4) # Default affine if construction failed

    # 3. Create NIfTI image and save
    try:
        # Ensure data is in a standard orientation for NIfTI (e.g., RAS)
        # The affine matrix should handle the orientation.
        # Nibabel expects data in (x, y, z, ...) order, Fortran-style if coming from DICOM slices.
        # Our stacking is (rows, cols, slices), which is (y, x, z) if DICOM image is axial.
        # The affine should correctly map this.
        # If data is 4D (e.g. from temporal positions), ensure it's handled.

        # If data is 3D (X,Y,Z), nibabel handles it.
        # If data is 4D (X,Y,Z,Time), nibabel handles it.
        # The current `extract_pixel_data_and_affine` tries to reshape to 4D if temporal positions are found.

        # Transpose image_data from (rows, cols, slices, [time]) to (cols, rows, slices, [time])
        # This is a common convention if rows=Y, cols=X.
        # (y, x, z, [t]) -> (x, y, z, [t])
        if image_data.ndim == 3: # (rows, cols, slices)
            image_data_nifti_order = image_data.transpose(1, 0, 2)
        elif image_data.ndim == 4: # (rows, cols, slices, time)
            image_data_nifti_order = image_data.transpose(1, 0, 2, 3)
        else:
            logger.error(f"Unsupported image data dimension: {image_data.ndim}")
            return False

        logger.info(f"Saving NIfTI image with data shape {image_data_nifti_order.shape} and affine:\n{affine_matrix}")
        nifti_image = nib.Nifti1Image(image_data_nifti_order, affine_matrix)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_nifti_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        nib.save(nifti_image, output_nifti_file)
        logger.info(f"NIfTI file saved successfully: {output_nifti_file}")

        # 4. Save metadata to JSON sidecar
        json_sidecar_file = output_nifti_file.replace(".nii.gz", ".json").replace(".nii", ".json")
        try:
            with open(json_sidecar_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata JSON sidecar saved successfully: {json_sidecar_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata JSON sidecar: {e}")
            # Continue, as NIfTI saving was the primary goal.

        return True

    except Exception as e:
        logger.error(f"An error occurred during NIfTI creation or saving: {e}")
        return False
