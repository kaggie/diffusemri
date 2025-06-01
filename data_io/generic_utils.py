import os
import logging
import numpy as np
import h5py
from scipy.io import savemat, loadmat # For .MAT file handling

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers(): # Avoid adding multiple handlers if already configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def save_dict_to_hdf5(data_dict: dict, hdf5_filepath: str):
    """
    Saves a dictionary of NumPy arrays to an HDF5 file.

    Args:
        data_dict (dict): Dictionary where keys are strings (dataset names)
                          and values are NumPy arrays.
        hdf5_filepath (str): Path to save the HDF5 file.

    Raises:
        TypeError: If data_dict is not a dictionary or values are not NumPy arrays.
        Exception: For HDF5 file operations or dataset creation errors.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("Input 'data_dict' must be a dictionary.")

    try:
        with h5py.File(hdf5_filepath, 'w') as hf:
            logger.info(f"Saving data to HDF5 file: {hdf5_filepath}")
            for key, value in data_dict.items():
                if not isinstance(key, str):
                    logger.warning(f"Skipping key '{key}' as it's not a string.")
                    continue
                if not isinstance(value, np.ndarray):
                    logger.warning(f"Skipping key '{key}' as its value is not a NumPy array (type: {type(value)}).")
                    continue
                hf.create_dataset(key, data=value)
                logger.debug(f"Saved dataset '{key}' with shape {value.shape} and dtype {value.dtype}")
        logger.info("HDF5 file saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save dictionary to HDF5 file {hdf5_filepath}: {e}")
        raise


def load_dict_from_hdf5(hdf5_filepath: str) -> dict:
    """
    Loads data from an HDF5 file into a dictionary of NumPy arrays.

    Args:
        hdf5_filepath (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary where keys are dataset names and values are NumPy arrays.
              Returns an empty dictionary if the file cannot be read or is empty.

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        Exception: For HDF5 file operations or dataset reading errors.
    """
    if not os.path.exists(hdf5_filepath):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_filepath}")

    loaded_dict = {}
    try:
        with h5py.File(hdf5_filepath, 'r') as hf:
            logger.info(f"Loading data from HDF5 file: {hdf5_filepath}")
            if not hf.keys():
                logger.warning(f"HDF5 file is empty: {hdf5_filepath}")
                return {}
            for key in hf.keys():
                try:
                    loaded_dict[key] = hf[key][...] # Use [...] to read full dataset into memory
                    logger.debug(f"Loaded dataset '{key}' with shape {loaded_dict[key].shape} and dtype {loaded_dict[key].dtype}")
                except Exception as e:
                    logger.error(f"Failed to load dataset '{key}' from HDF5 file {hdf5_filepath}: {e}")
                    # Optionally, continue to load other datasets or re-raise
        logger.info("HDF5 file loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dictionary from HDF5 file {hdf5_filepath}: {e}")
        raise # Re-raise after logging

    return loaded_dict


def save_dict_to_mat(data_dict: dict, mat_filepath: str, oned_as: str = 'column'):
    """
    Saves a dictionary of NumPy arrays to a .MAT file (version 5).

    Args:
        data_dict (dict): Dictionary where keys are strings (variable names)
                          and values are NumPy arrays.
        mat_filepath (str): Path to save the .MAT file.
        oned_as (str, optional): How to store 1D arrays ('column' or 'row').
                                 Defaults to 'column'.

    Raises:
        TypeError: If data_dict is not a dictionary.
        Exception: For SciPy I/O errors.
    """
    if not isinstance(data_dict, dict):
        raise TypeError("Input 'data_dict' must be a dictionary.")

    try:
        logger.info(f"Saving data to .MAT file: {mat_filepath}")
        # Ensure all keys are strings and values are suitable for savemat
        # savemat can handle various NumPy array types.
        # It might have issues with complex objects or non-NumPy arrays directly.
        # For simplicity, this wrapper assumes values are primarily NumPy arrays.
        # Consider adding checks or conversions if more complex dicts are expected.
        savemat(mat_filepath, data_dict, oned_as=oned_as)
        logger.info(".MAT file saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save dictionary to .MAT file {mat_filepath}: {e}")
        raise


def load_dict_from_mat(mat_filepath: str) -> dict:
    """
    Loads data from a .MAT file into a dictionary of NumPy arrays.

    This function filters out MAT-file specific metadata keys (starting with '__').

    Args:
        mat_filepath (str): Path to the .MAT file.

    Returns:
        dict: A dictionary where keys are variable names and values are NumPy arrays.
              Returns an empty dictionary if the file cannot be read.

    Raises:
        FileNotFoundError: If the .MAT file does not exist.
        Exception: For SciPy I/O errors.
    """
    if not os.path.exists(mat_filepath):
        raise FileNotFoundError(f".MAT file not found: {mat_filepath}")

    loaded_data_with_meta = {}
    try:
        logger.info(f"Loading data from .MAT file: {mat_filepath}")
        loaded_data_with_meta = loadmat(mat_filepath)
    except Exception as e:
        logger.error(f"Failed to load .MAT file {mat_filepath}: {e}")
        raise # Re-raise after logging

    # Filter out MAT-file specific metadata keys
    # and ensure values are primarily NumPy arrays (loadmat usually ensures this for data)
    data_dict = {
        key: value for key, value in loaded_data_with_meta.items()
        if not key.startswith('__') # Filter out __header__, __version__, __globals__
        # Optionally, add further checks e.g. isinstance(value, np.ndarray)
        # but loadmat structures (like structs/cells) can be complex.
        # For this generic util, we keep what loadmat provides minus the __ prefixed keys.
    }

    if not data_dict:
        logger.warning(f".MAT file {mat_filepath} contained no loadable data variables (or only metadata).")

    logger.info(".MAT file loaded successfully.")
    return data_dict
