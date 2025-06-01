import os
import logging
import numpy as np
import nibabel as nib

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def read_analyze_data(analyze_filepath: str) -> tuple[np.ndarray | None, np.ndarray | None, nib.analyze.AnalyzeHeader | None]:
    """
    Reads data, affine, and header from an Analyze 7.5 format file (.hdr/.img).

    Args:
        analyze_filepath (str): Path to the Analyze file (either .hdr or .img).

    Returns:
        tuple:
            - image_data (np.ndarray | None): The image data as a NumPy array (float32).
            - affine (np.ndarray | None): The NIfTI-compatible affine matrix.
            - header (nib.analyze.AnalyzeHeader | None): The Nibabel AnalyzeHeader object.
            Returns (None, None, None) if reading fails.
    """
    if not os.path.exists(analyze_filepath):
        logger.error(f"Analyze file not found: {analyze_filepath}")
        return None, None, None

    try:
        img_obj = nib.load(analyze_filepath)
        if not isinstance(img_obj, nib.AnalyzeImage):
            logger.error(f"File is not an Analyze 7.5 format image: {analyze_filepath}")
            return None, None, None

        # Ensure data is float32 for consistency
        image_data = img_obj.get_fdata(dtype=np.float32)
        affine = img_obj.affine
        header = img_obj.header

        logger.info(f"Successfully read Analyze file: {analyze_filepath}")
        logger.info(f"Data shape: {image_data.shape}, Affine:\n{affine}")
        return image_data, affine, header
    except Exception as e:
        logger.error(f"Failed to read Analyze file {analyze_filepath}: {e}")
        return None, None, None


def write_analyze_data(output_filepath: str,
                         data: np.ndarray,
                         affine: np.ndarray,
                         header: nib.analyze.AnalyzeHeader = None):
    """
    Writes image data and affine to an Analyze 7.5 format file (.hdr/.img).

    Args:
        output_filepath (str): Path to save the output Analyze file (e.g., 'image.hdr').
                               The corresponding '.img' file will be created automatically.
        data (np.ndarray): Image data (3D or 4D NumPy array).
        affine (np.ndarray): 4x4 NIfTI-style affine matrix.
        header (nib.analyze.AnalyzeHeader, optional): A Nibabel AnalyzeHeader object.
            If None, a minimal header will be created by Nibabel. It's recommended
            to provide one if specific header information (like voxel dimensions from `pixdim`)
            needs to be precisely controlled, as affine to header conversion can be lossy.

    Raises:
        ValueError: If input data or affine is invalid.
        Exception: If `nib.save` fails.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input 'data' must be a NumPy array.")
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("Input 'affine' must be a 4x4 NumPy array.")

    logger.info(f"Preparing to write Analyze file: {output_filepath}")
    logger.info(f"Data shape: {data.shape}, Affine:\n{affine}")

    try:
        # Data type consideration: Analyze 7.5 has limited data type support.
        # Common types: uint8, int16, float32. Nibabel will handle conversion if possible.
        # Forcing data to float32 if not already, as it's a common intermediate.
        if not np.issubdtype(data.dtype, np.floating):
            logger.warning(f"Data type {data.dtype} may not be directly supported by Analyze 7.5. Casting to float32.")
            data = data.astype(np.float32)
        elif data.dtype != np.float32: # If already float, ensure it's float32
            data = data.astype(np.float32)


        analyze_image = nib.AnalyzeImage(data, affine, header=header)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        nib.save(analyze_image, output_filepath)
        logger.info(f"Analyze file saved successfully: {output_filepath} (and corresponding .img)")
    except Exception as e:
        logger.error(f"Failed to write Analyze file {output_filepath}: {e}")
        raise
