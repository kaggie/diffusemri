# 07: Image Format Conversion Utilities

This section details the functionalities within the `diffusemri` library for converting medical imaging data between various formats. These tools primarily leverage the `cli/run_format_conversion.py` script for command-line operations and corresponding functions in `data_io/` for Python API usage. Supported formats include NIfTI, NRRD, MHD/MHA, Analyze 7.5, Philips PAR/REC, ISMRMRD (placeholder), and DICOM Secondary Capture.

## General Notes
*   **DWI Metadata:** Special attention is given to preserving Diffusion-Weighted Imaging (DWI) metadata (b-values and b-vectors) during conversions where applicable. However, the accuracy of this depends on the source format's ability to store such information and the conventions used.
*   **Orientation and Affine Information:** Conversion of spatial orientation (affine matrices) is handled, but users should always verify the results, especially when converting to/from formats with limited orientation support (like Analyze 7.5).
*   **CLI Script:** Most command-line examples below use `python cli/run_format_conversion.py <subcommand> ...`. Ensure you are in the root directory of the `diffusemri` project or adjust paths accordingly.

## NIfTI <-> NRRD Conversion

NRRD (Nearly Raw Raster Data) is a flexible format often used in research, particularly with tools like 3D Slicer.

### Key Python Functions:
*   `data_io.nrrd_utils.read_nrrd_data()`: Reads NRRD, extracts image, affine, DWI metadata, and header.
*   `data_io.nrrd_utils.write_nrrd_data()`: Writes NumPy data to NRRD, embedding affine and optional DWI metadata.

### CLI Subcommands:
*   **`nrrd2nii`**: Converts NRRD to NIfTI.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py nrrd2nii \\
        --input_nrrd /path/to/your/image.nrrd \\
        --output_nifti /path/to/output/image.nii.gz \\
        # Optional for DWI:
        # --output_bval /path/to/output/dwi.bval \\
        # --output_bvec /path/to/output/dwi.bvec
    ```
*   **`nii2nrrd`**: Converts NIfTI to NRRD.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py nii2nrrd \\
        --input_nifti /path/to/your/image.nii.gz \\
        --output_nrrd /path/to/output/image.nrrd \\
        # Optional for DWI:
        # --input_bval /path/to/your/dwi.bval \\
        # --input_bvec /path/to/your/dwi.bvec
    ```

## NIfTI <-> MHD/MHA Conversion

MHD/MHA (MetaImageHeader) is another common research format, often used by ITK and related software.

### Key Python Functions:
*   `data_io.mhd_utils.read_mhd_data()`: Reads MHD/MHA using SimpleITK, extracts image, affine, DWI metadata, and header.
*   `data_io.mhd_utils.write_mhd_data()`: Writes NumPy data to MHD/MHA using SimpleITK, embedding affine and optional DWI metadata.

### CLI Subcommands:
*   **`mhd2nii`**: Converts MHD/MHA to NIfTI.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py mhd2nii \\
        --input_mhd /path/to/your/image.mhd \\
        --output_nifti /path/to/output/image.nii.gz \\
        # Optional for DWI:
        # --output_bval /path/to/output/dwi.bval \\
        # --output_bvec /path/to/output/dwi.bvec
    ```
*   **`nii2mhd`**: Converts NIfTI to MHD/MHA.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py nii2mhd \\
        --input_nifti /path/to/your/image.nii.gz \\
        --output_mhd /path/to/output/image.mha \\
        # Optional for DWI:
        # --input_bval /path/to/your/dwi.bval \\
        # --input_bvec /path/to/your/dwi.bvec
    ```

## NIfTI <-> Analyze 7.5 Conversion

Analyze 7.5 (`.hdr`/`.img`) is an older format with significant limitations in storing orientation and metadata.

### Key Python Functions:
*   `data_io.analyze_utils.read_analyze_data()`: Reads Analyze using Nibabel.
*   `data_io.analyze_utils.write_analyze_data()`: Writes Analyze using Nibabel.

### CLI Subcommands:
*   **`analyze2nii`**: Converts Analyze 7.5 to NIfTI.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py analyze2nii \\
        --input_analyze /path/to/your/image.hdr \\
        --output_nifti /path/to/output/image.nii.gz
    ```
*   **`nii2analyze`**: Converts NIfTI to Analyze 7.5.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py nii2analyze \\
        --input_nifti /path/to/your/image.nii.gz \\
        --output_analyze /path/to/output/image.hdr
        # .img file will be created alongside
    ```
*   **Warning:** DWI metadata (bvals/bvecs) will be lost when converting to Analyze. Orientation information may also be compromised.

## Philips PAR/REC to NIfTI Conversion

Supports reading Philips PAR/REC files (versions 4.0-4.2) and converting them to NIfTI.

### Key Python Functions:
*   `data_io.parrec_utils.read_parrec_data()`: Reads PAR/REC using Nibabel, extracts image, affine, and DWI metadata.
*   `data_io.parrec_utils.convert_parrec_to_nifti()`: Orchestrates the conversion and saving process.

### CLI Subcommand:
*   **`parrec2nii`**: Converts PAR/REC to NIfTI.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py parrec2nii \\
        --input_parrec /path/to/your/image.par \\
        --output_nifti /path/to/output/image.nii.gz \\
        # Optional for DWI:
        --output_bval /path/to/output/dwi.bval \\
        --output_bvec /path/to/output/dwi.bvec \\
        # Optional PAR/REC loading parameters:
        # --scaling_method fp \\
        # --no_strict_sort
    ```

## NIfTI to DICOM Secondary Capture Conversion

Converts NIfTI images into a series of DICOM Secondary Capture files, useful for integrating processed data into PACS.

### Key Python Function:
*   `data_io.dicom_utils.write_nifti_to_dicom_secondary()`: Takes a NIfTI image object and creates DICOM SC files.

### CLI Subcommand:
*   **`nii2dicom_sec`**: Converts NIfTI to DICOM Secondary Capture.
    ```bash
    # Conceptual CLI Usage:
    python cli/run_format_conversion.py nii2dicom_sec \\
        --input_nifti /path/to/your/image.nii.gz \\
        --output_dicom_dir /path/to/output/dicom_series_directory/ \\
        --patient-id "PAT001" \\
        --series-description "Converted NIfTI Data"
        # Other options like --study-uid, --series-uid, --series-number are available
    ```
*   **Important Note:** DICOM Secondary Capture objects are not for primary diagnostic use and do not retain original acquisition metadata.

## ISMRMRD Support (.h5) - Placeholder

ISMRMRD (International Society for Magnetic Resonance in Medicine Raw Data Format) is used for raw and reconstructed MR data.

*   **Current Status:** `Placeholder - Not Implemented`
*   **Intended Python Functions:** `data_io.ismrmrd_utils.read_ismrmrd_file()`, `data_io.ismrmrd_utils.convert_ismrmrd_to_nifti_and_metadata()`.
*   **Intended CLI Subcommand:** `ismrmrd_convert`
    ```bash
    # Conceptual CLI Usage (Placeholder):
    # python cli/run_format_conversion.py ismrmrd_convert \\
    #     --input_ismrmrd /path/to/input.h5 \\
    #     --output_base /path/to/output/scan_base
    ```
*   **Dependencies:** Will require the `ismrmrd` Python package.

---
For detailed examples of using these functions and CLI tools, please refer to the Jupyter notebooks in the `examples/` directory, particularly:
*   `01_DICOM_IO_and_Anonymization.ipynb` (for `dicom_to_nifti` via `run_preprocessing`)
*   `02_NRRD_MHD_Analyze_IO.ipynb`
*   `03_PARREC_Input.ipynb`
*   `05_NIfTI_to_DICOM_Secondary.ipynb`
