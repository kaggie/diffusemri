import os
import logging
import numpy as np
import nibabel as nib

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def read_parrec_data(parrec_filepath: str, strict_sort: bool = True, scaling: str = 'dv') \
        -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict | None]:
    """
    Reads data, affine, b-values, b-vectors, and header from a Philips PAR/REC file.

    Args:
        parrec_filepath (str): Path to the PAR or REC file.
        strict_sort (bool, optional): Whether to use strict sorting of volumes
            based on acquisition time and other tags. Passed to `nib.load`.
            Defaults to True.
        scaling (str, optional): Scaling method for PAR/REC data ('dv' or 'fp').
            'dv': uses display values. 'fp': uses floating point values.
            Passed to `nib.load`. Defaults to 'dv'.

    Returns:
        tuple:
            - image_data (np.ndarray | None): The image data as a NumPy array (float32).
            - affine (np.ndarray | None): The NIfTI-compatible affine matrix.
            - bvals (np.ndarray | None): Array of b-values, if DWI.
            - bvecs (np.ndarray | None): Array of b-vectors (Nx3), if DWI.
            - header_info (dict | None): Dictionary containing PAR/REC header information.
            Returns (None, None, None, None, None) if reading fails.
    """
    if not os.path.exists(parrec_filepath):
        logger.error(f"PAR/REC file not found: {parrec_filepath}")
        return None, None, None, None, None

    try:
        logger.info(f"Loading PAR/REC file: {parrec_filepath} with strict_sort={strict_sort}, scaling='{scaling}'")
        img_obj = nib.load(parrec_filepath, strict_sort=strict_sort, scaling=scaling)

        if not isinstance(img_obj, nib.parrec.PARRECImage):
            logger.error(f"File is not a recognized Philips PAR/REC format image: {parrec_filepath}")
            return None, None, None, None, None

        image_data = img_obj.get_fdata(dtype=np.float32)
        affine = img_obj.affine

        # Extract b-values and b-vectors if available
        # The get_bvals_bvecs() method returns None for bvals and bvecs if not a diffusion image
        # or if the information is not found in the header.
        bvals, bvecs = img_obj.header.get_bvals_bvecs()

        # Convert PARRECHeader to a more generic dict for metadata, if needed, or pass as is.
        # For simplicity, we can pass the nibabel header object.
        # To make it JSON serializable for sidecars, a dict conversion would be needed.
        header_info = {"source_format": "PAR/REC"}
        # Example: extract some specific fields if desired
        # header_info['RepetitionTime'] = img_obj.header.get_echo_train_length() # This is wrong, just an example
        # header_info['EchoTime'] = img_obj.header.get_echo_times()

        logger.info(f"Successfully read PAR/REC file: {parrec_filepath}")
        logger.info(f"Data shape: {image_data.shape}, Affine:\n{affine}")
        if bvals is not None:
            logger.info(f"Extracted b-values (count: {len(bvals)})")
        if bvecs is not None:
            logger.info(f"Extracted b-vectors (shape: {bvecs.shape})")

        return image_data, affine, bvals, bvecs, {"source_header_object": img_obj.header, "parsed_info": header_info}

    except Exception as e:
        logger.error(f"Failed to read PAR/REC file {parrec_filepath}: {e}")
        return None, None, None, None, None


def convert_parrec_to_nifti(parrec_filepath: str,
                              output_nifti_file: str,
                              output_bval_file: str = None,
                              output_bvec_file: str = None,
                              strict_sort: bool = True,
                              scaling: str = 'dv') -> bool:
    """
    Converts a Philips PAR/REC file to NIfTI format, and optionally saves
    b-values and b-vectors if it's a DWI acquisition.

    Args:
        parrec_filepath (str): Path to the input PAR or REC file.
        output_nifti_file (str): Path to save the output NIfTI file.
        output_bval_file (str, optional): Path to save the output b-values file.
                                          Required if DWI data is present and to be saved.
        output_bvec_file (str, optional): Path to save the output b-vectors file.
                                          Required if DWI data is present and to be saved.
        strict_sort (bool, optional): Passed to `read_parrec_data`. Defaults to True.
        scaling (str, optional): Passed to `read_parrec_data`. Defaults to 'dv'.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    logger.info(f"Starting PAR/REC to NIfTI conversion for: {parrec_filepath}")

    image_data, affine, bvals, bvecs, _ = read_parrec_data(
        parrec_filepath, strict_sort=strict_sort, scaling=scaling
    )

    if image_data is None or affine is None:
        logger.error("Failed to read data or affine from PAR/REC file.")
        return False

    try:
        # Ensure output directory exists for NIfTI file
        output_nifti_dir = os.path.dirname(output_nifti_file)
        if output_nifti_dir and not os.path.exists(output_nifti_dir):
            os.makedirs(output_nifti_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_nifti_dir}")

        nifti_image = nib.Nifti1Image(image_data.astype(np.float32), affine) # Ensure float32
        nib.save(nifti_image, output_nifti_file)
        logger.info(f"NIfTI file saved successfully: {output_nifti_file}")

        if bvals is not None and output_bval_file:
            output_bval_dir = os.path.dirname(output_bval_file)
            if output_bval_dir and not os.path.exists(output_bval_dir):
                os.makedirs(output_bval_dir, exist_ok=True)
            np.savetxt(output_bval_file, bvals.reshape(1, -1), fmt='%g')
            logger.info(f"b-values saved to: {output_bval_file}")

        if bvecs is not None and output_bvec_file:
            output_bvec_dir = os.path.dirname(output_bvec_file)
            if output_bvec_dir and not os.path.exists(output_bvec_dir):
                os.makedirs(output_bvec_dir, exist_ok=True)
            # Nibabel's PARREC loader should provide b-vectors in RAS_ τότε FSL format (3xN)
            # For consistency with other tools, often Nx3 is preferred for text files.
            # Let's check the shape from Nibabel and save appropriately.
            # FSL expects 3 rows, N columns. Dipy/MRtrix often use N rows, 3 columns.
            # The `get_bvals_bvecs` method in nibabel.parrec returns bvecs as (N, 3)
            np.savetxt(output_bvec_file, bvecs, fmt='%.8f') # Save as Nx3
            logger.info(f"b-vectors saved to: {output_bvec_file} (format: Nx3)")

        return True

    except Exception as e:
        logger.error(f"An error occurred during NIfTI creation or saving bval/bvec: {e}")
        return False
