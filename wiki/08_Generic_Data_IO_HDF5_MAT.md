# 08: Generic Data I/O (HDF5 and .MAT)

This section describes utilities for saving and loading collections of NumPy arrays to/from HDF5 and MATLAB .MAT files. These functions are useful for storing intermediate results, processed parameter maps, or any collection of array data in a structured way.

## HDF5 I/O Utilities

HDF5 (Hierarchical Data Format version 5) is a versatile file format designed to store and organize large amounts of numerical data.

### Key Functions:
*   **`diffusemri.data_io.generic_utils.save_dict_to_hdf5(data_dict: dict, hdf5_filepath: str)`**:
    *   **Purpose:** Saves a Python dictionary, where keys are strings and values are NumPy arrays, to an HDF5 file. Each key-value pair in the dictionary becomes a dataset in the HDF5 file.
    *   **Arguments:**
        *   `data_dict` (dict): The dictionary to save. Values must be NumPy arrays.
        *   `hdf5_filepath` (str): Path to the output HDF5 file (e.g., `my_data.h5`).
    *   **Dependencies:** Requires `h5py`.

*   **`diffusemri.data_io.generic_utils.load_dict_from_hdf5(hdf5_filepath: str) -> dict`**:
    *   **Purpose:** Loads all datasets from an HDF5 file into a Python dictionary, where keys are the dataset names and values are the NumPy arrays.
    *   **Arguments:**
        *   `hdf5_filepath` (str): Path to the input HDF5 file.
    *   **Returns:** A dictionary containing the loaded datasets.
    *   **Dependencies:** Requires `h5py`.

### Conceptual Python Usage (HDF5):
```python
import numpy as np
# from diffusemri.data_io.generic_utils import save_dict_to_hdf5, load_dict_from_hdf5 # Actual import

# Example data
# fa_map = np.random.rand(10, 10, 10).astype(np.float32)
# md_map = np.random.rand(10, 10, 10).astype(np.float32)
# data_to_save = {
#     "FA": fa_map,
#     "MD": md_map,
# }
# hdf5_file = "path/to/my_processed_data.h5"

# Save data
# save_dict_to_hdf5(data_to_save, hdf5_file)
# print(f"Data saved to {hdf5_file}")

# Load data back
# loaded_data = load_dict_from_hdf5(hdf5_file)
# if loaded_data and 'FA' in loaded_data:
#     print(f"Loaded FA map shape: {loaded_data['FA'].shape}")
# else:
#     print("Conceptual HDF5 example - see examples/04_Generic_HDF5_MAT_IO.ipynb for runnable code.")
print("Conceptual HDF5 example - see examples/04_Generic_HDF5_MAT_IO.ipynb for runnable code.")
```

## MATLAB .MAT File I/O Utilities

MATLAB .MAT files (version 5 by default with SciPy) are commonly used for storing numerical arrays and can be a convenient way to exchange data with MATLAB or other environments that support this format.

### Key Functions:
*   **`diffusemri.data_io.generic_utils.save_dict_to_mat(data_dict: dict, mat_filepath: str, oned_as: str = 'column')`**:
    *   **Purpose:** Saves a Python dictionary, where keys are strings (variable names) and values are NumPy arrays (or other compatible types like strings, scalars), to a .MAT file.
    *   **Arguments:**
        *   `data_dict` (dict): The dictionary to save.
        *   `mat_filepath` (str): Path to the output .MAT file (e.g., `my_data.mat`).
        *   `oned_as` (str, optional): How to store 1D arrays ('column' or 'row'). Defaults to 'column'.
    *   **Dependencies:** Requires `scipy.io`.

*   **`diffusemri.data_io.generic_utils.load_dict_from_mat(mat_filepath: str) -> dict`**:
    *   **Purpose:** Loads variables from a .MAT file into a Python dictionary. It filters out MAT-file specific metadata keys (like `__header__`, `__version__`, `__globals__`).
    *   **Arguments:**
        *   `mat_filepath` (str): Path to the input .MAT file.
    *   **Returns:** A dictionary containing the loaded variables.
    *   **Dependencies:** Requires `scipy.io`.

### Conceptual Python Usage (.MAT):
```python
import numpy as np
# from diffusemri.data_io.generic_utils import save_dict_to_mat, load_dict_from_mat # Actual import

# Example data
# tract_lengths = np.array([10.5, 22.1, 15.3])
# connectivity_matrix = np.random.randint(0, 100, size=(5,5))
# data_to_save_mat = {
#     "tract_lengths": tract_lengths,
#     "connectivity": connectivity_matrix
# }
# mat_file = "path/to/my_analysis_results.mat"

# Save data
# save_dict_to_mat(data_to_save_mat, mat_file)
# print(f"Data saved to {mat_file}")

# Load data back
# loaded_mat_data = load_dict_from_mat(mat_file)
# if loaded_mat_data and 'tract_lengths' in loaded_mat_data:
#     print(f"Loaded tract lengths: {loaded_mat_data['tract_lengths']}")
# else:
#     print("Conceptual .MAT example - see examples/04_Generic_HDF5_MAT_IO.ipynb for runnable code.")
print("Conceptual .MAT example - see examples/04_Generic_HDF5_MAT_IO.ipynb for runnable code.")
```

**Note:** These generic I/O utilities are designed for flexibility in saving and loading collections of NumPy arrays and basic Python types (for .MAT). They do not store rich NIfTI header information (like affine transformations or detailed imaging parameters) directly within the HDF5 or .MAT structures unless explicitly included as separate datasets/variables by the user. When loading data for further spatial processing, ensure you have the necessary spatial context (e.g., affine, dimensions) from other sources if it's not part of the saved arrays.

---
For detailed, runnable examples of these I/O utilities, please refer to the `examples/04_Generic_HDF5_MAT_IO.ipynb` Jupyter notebook.
