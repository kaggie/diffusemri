# 09: Bruker ParaVision Data I/O

This section details the functionalities for reading Bruker ParaVision datasets, commonly used in preclinical MRI. The `diffusemri` library leverages the `brukerapi` Python package to parse these complex datasets.

## Overview of Bruker ParaVision Data Structure

Bruker ParaVision datasets are typically stored in a specific directory structure for each experiment (often a numbered folder). Key files and directories involved in DWI data include:

*   **`2dseq`**: This binary file contains the raw image data. It's usually located within a `pdata/<reco_num>/` subdirectory (e.g., `pdata/1/2dseq`).
*   **`visu_pars`**: A text file (JCAMP-DX format) containing visualization parameters, image properties like matrix size, FOV, orientation, slice information, etc. Usually located alongside `2dseq`.
*   **`method`**: A text file (JCAMP-DX format) containing acquisition method parameters, including crucial DWI information like b-values (`PVM_DwEffBval` or `PVM_DwBvalEach`) and gradient directions (`PVM_DwDir`). Located in the main experiment folder.
*   **`acqp`**: A text file (JCAMP-DX format) containing acquisition parameters related to hardware and timing, but less commonly used for basic DWI metadata extraction compared to `method` and `visu_pars`. Located in the main experiment folder.
*   Other files like `reco` (reconstruction parameters), `fid` (raw k-space data), and `subject` (subject-related information) may also be present.

## Reading Bruker ParaVision DWI Data

The `diffusemri` library provides a function to read Bruker DWI datasets and extract the necessary components for conversion to formats like NIfTI.

### Key Python Function:
*   **`diffusemri.data_io.bruker_utils.read_bruker_dwi_data(bruker_dir_path: str) -> tuple | None`**:
    *   **Purpose:** Reads a Bruker ParaVision DWI dataset from the specified experiment directory path.
    *   **Arguments:**
        *   `bruker_dir_path` (str): Path to the Bruker experiment folder (e.g., `/path/to/study/exp_num`) or directly to the `pdata/1/` folder containing `2dseq`, or even to the `2dseq` file itself. The function attempts to locate the necessary files based on these inputs.
    *   **Method:**
        1.  Initializes a `brukerapi.dataset.Dataset` object.
        2.  Loads parameter files: `visu_pars` (usually loaded by default with `2dseq`), `acqp`, and `method`.
        3.  Loads the image data from `2dseq`.
        4.  Extracts image data (NumPy array).
        5.  Constructs a NIfTI-compatible affine matrix using `VisuCoreSize`, `VisuCoreExtent`, `VisuCorePosition`, and `VisuCoreOrientation` from `visu_pars`.
        6.  Extracts b-values (typically from `PVM_DwEffBval` or `PVM_DwBvalEach` in `method`).
        7.  Extracts b-vectors (typically from `PVM_DwDir` in `method`). These are assumed to be in the scanner/gradient coordinate system.
        8.  Performs an initial reorientation of b-vectors to align with the NIfTI image axes (further refinement might be needed based on specific scanner setups and `ACQ_grad_matrix`).
        9.  Transposes image data to a NIfTI-compatible order (e.g., X, Y, Z, Time/DWIs). The exact transposition depends on the `VisuCoreDim` and the raw data order from `brukerapi`.
        10. Gathers key parameters into a metadata dictionary.
    *   **Returns:** A tuple `(image_data, affine, bvals, bvecs, metadata_dict)` or `None` if critical information is missing or an error occurs.
    *   **Dependencies:** Requires the `brukerapi` library.

### Conceptual Python Usage:
```python
# from diffusemri.data_io.bruker_utils import read_bruker_dwi_data # Actual import

# bruker_experiment_path = "/path/to/your_bruker_study/experiment_number"
# # This path should point to the folder containing 'acqp', 'method', and the 'pdata' subdirectory.

# try:
#     result = read_bruker_dwi_data(bruker_experiment_path)
#     if result:
#         image_data, affine, bvals, bvecs, metadata = result
#         print(f"Image data shape: {image_data.shape}")
#         print(f"Affine matrix:\\n{affine}")
#         if bvals is not None:
#             print(f"b-values (first 5): {bvals[:5]}")
#         if bvecs is not None:
#             print(f"b-vectors (first 5):\\n{bvecs[:5,:]}")
#         # print(f"Sample metadata (VisuCoreDim): {metadata.get('visu_pars', {}).get('VisuCoreDim')}")
#     else:
#         print("Failed to read Bruker DWI data.")
# except Exception as e:
#     print(f"An error occurred: {e}")
print("Note: Bruker data reading example is conceptual. Requires a valid Bruker dataset and `brukerapi`.")
```

## Command-Line Interface (CLI) for Bruker to NIfTI Conversion

The `run_format_conversion.py` script provides a subcommand to convert Bruker ParaVision DWI datasets to NIfTI format, also saving b-values and b-vectors.

### `bruker2nii` (alias: `bruker_to_nifti`)
*   **Purpose:** Converts a Bruker ParaVision DWI dataset into NIfTI format, along with associated `.bval` and `.bvec` files.
*   **CLI Tool:** `python cli/run_format_conversion.py bruker2nii ...` (or `bruker_to_nifti`)
*   **Key Arguments:**
    *   `--input_bruker_dir`: Path to the Bruker experiment directory.
    *   `--output_nifti`: Path to save the output NIfTI file.
    *   `--output_bval`: Path to save the output b-values file.
    *   `--output_bvec`: Path to save the output b-vectors file.

*   **Conceptual CLI Usage:**
    ```bash
    python cli/run_format_conversion.py bruker2nii \\
        --input_bruker_dir /path/to/your_bruker_study/experiment_number \\
        --output_nifti /path/to/output/dwi_from_bruker.nii.gz \\
        --output_bval /path/to/output/dwi_from_bruker.bval \\
        --output_bvec /path/to/output/dwi_from_bruker.bvec
    ```

**Important Considerations for Bruker Data:**
*   **`brukerapi` Version:** Ensure you have a compatible version of `brukerapi` installed.
*   **b-vector Reorientation:** The reorientation of b-vectors from the scanner frame to the image frame is complex and can depend on specific scanner parameters (e.g., `ACQ_grad_matrix`, `VisuCoreOrientation`). The current implementation provides a common first-pass reorientation; always validate your b-vectors post-conversion, especially if performing quantitative tractography or analyses sensitive to gradient directions.
*   **Data Transposition:** The order of dimensions in the raw Bruker `2dseq` file can vary. The `read_bruker_dwi_data` function attempts a common transposition to align with NIfTI conventions (e.g., X, Y, Z, Time/DWIs). Verify the output dimensions and orientation.
*   **Multi-echo / Multi-repetition Data:** Complex sequences with multiple echoes, repetitions, or segments might require more specialized parsing logic if not handled transparently by `brukerapi` for the `2dseq` (processed image) data. The current reader primarily assumes a final reconstructed image series in `2dseq`.

---
For a runnable example demonstrating the use of these utilities (with mocked Bruker data for testing purposes), please refer to the Jupyter notebook `examples/TODO_Bruker_IO.ipynb` (Note: This notebook would need to be created as part of future work if a fully runnable example with mocked data is desired, as direct Bruker data simulation is complex for a simple example).
