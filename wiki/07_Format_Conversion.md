# 07: Format Conversion (NRRD, MHD, Analyze, ISMRMRD)

This section details the functionalities for converting imaging data between various formats including NRRD (Nearly Raw Raster Data), MHD (MetaImageHeader), Analyze 7.5, ISMRMRD, and NIfTI, with special considerations for diffusion MRI (dMRI) data.

## NRRD File Support

The `diffusemri` library provides utilities to read and write NRRD files. This is particularly useful for interoperability with software and datasets that use the NRRD format, such as Slicer.

### Key Functions:
*   **`diffusemri.data_io.nrrd_utils.read_nrrd_data()`**:
    *   Reads image data and header information from a `.nrrd` or `.nhdr` file.
    *   Constructs a NIfTI-compatible affine matrix from NRRD spatial metadata (`space`, `space directions`, `space origin`).
    *   Attempts to parse DWI-specific information if present, including:
        *   b-values (e.g., from `DWMRI_b-value` or per-gradient fields).
        *   Gradient vectors (e.g., from `DWMRI_gradient_XXXX` fields). These are reoriented based on the image affine and `measurement frame` (if present) to align with typical NIfTI conventions (image coordinate system).
    *   Returns the image data (NumPy array), affine matrix (4x4 NumPy array), b-values (1D NumPy array or None), b-vectors (Nx3 NumPy array or None), and the full NRRD header dictionary.

*   **`diffusemri.data_io.nrrd_utils.write_nrrd_data()`**:
    *   Writes image data (NumPy array) to a `.nrrd` file.
    *   Constructs a NRRD header using a provided NIfTI-style affine matrix. This includes setting `space`, `space directions`, and `space origin`.
    *   If b-values and b-vectors are provided, they are stored in the NRRD header using common conventions (e.g., `modality:=DWMRI`, `DWMRI_b-value`, `DWMRI_gradient_XXXX`, `measurement frame`).
    *   Allows specification of custom header fields and NRRD writing options (e.g., `encoding`, `endian`).

### DWI Metadata in NRRD
NRRD files can store DWI information in their headers using key-value pairs. Common conventions, particularly from tools like Slicer, include:
*   `modality:=DWMRI`
*   `DWMRI_b-value:=<value>`: Often a single reference b-value for the DWI acquisition.
*   `DWMRI_gradient_0000:=<gx> <gy> <gz> [bval]`: Per-gradient information. The b-value component here is sometimes used for multi-shell data. Our reader attempts to parse this.
*   `measurement frame:=<xx> <xy> <xz> <yx> <yy> <yz> <zx> <zy> <zz>`: A 3x3 matrix indicating the coordinate system in which the `DWMRI_gradient_XXXX` vectors are expressed. If present, b-vectors are transformed from this frame to the patient coordinate system (PCS) before being reoriented to the image coordinate system. If not present, gradients are often assumed to be in the PCS.

The `read_nrrd_data` function attempts to interpret these fields to extract b-values and reorient b-vectors appropriately for use with tools expecting NIfTI-like conventions. The `write_nrrd_data` function stores DWI information using these fields to facilitate interoperability.

## Command-Line Interface (CLI) for Format Conversion

A dedicated CLI script `run_format_conversion.py` is provided for NRRD and NIfTI format conversions.

### `nrrd2nii`: NRRD to NIfTI Conversion
*   **Purpose:** Converts a NRRD file (potentially DWI) into NIfTI format. If DWI information is successfully parsed from the NRRD header, corresponding `.bval` and `.bvec` files are also generated.
*   **CLI Tool:** `run_format_conversion nrrd2nii`
*   **Conceptual CLI Usage:**
    ```bash
    # Convert a general NRRD file to NIfTI
    python cli/run_format_conversion.py nrrd2nii \\
        --input_nrrd /path/to/input.nrrd \\
        --output_nifti /path/to/output.nii.gz

    # Convert a DWI NRRD file to NIfTI, also saving bval/bvec
    python cli/run_format_conversion.py nrrd2nii \\
        --input_nrrd /path/to/dwi.nrrd \\
        --output_nifti /path/to/output/dwi.nii.gz \\
        --output_bval /path/to/output/dwi.bval \\
        --output_bvec /path/to/output/dwi.bvec
    ```

### `nii2nrrd`: NIfTI to NRRD Conversion
*   **Purpose:** Converts a NIfTI file into NRRD format. If associated `.bval` and `.bvec` files are provided for a DWI NIfTI, this information is embedded into the output NRRD header.
*   **CLI Tool:** `run_format_conversion nii2nrrd`
*   **Conceptual CLI Usage:**
    ```bash
    # Convert a general NIfTI file to NRRD
    python cli/run_format_conversion.py nii2nrrd \\
        --input_nifti /path/to/input.nii.gz \\
        --output_nrrd /path/to/output.nrrd

    # Convert a DWI NIfTI file (with bval/bvec) to NRRD
    python cli/run_format_conversion.py nii2nrrd \\
        --input_nifti /path/to/dwi.nii.gz \\
        --input_bval /path/to/dwi.bval \\
        --input_bvec /path/to/dwi.bvec \\
        --output_nrrd /path/to/output/dwi.nrrd \\
        --nrrd_encoding gzip # Optional: specify NRRD encoding
    ```

**Note:** The accuracy of DWI information (especially b-vector orientation) conversion depends heavily on the correctness and completeness of the header information in the source NRRD or NIfTI file and the associated b-vector/affine conventions.

## MHD/MHA File Support

The library also supports reading and writing MHD (MetaImageHeader, `.mhd` with a separate `.raw` or `.zraw` file) and MHA (single file `.mha`) formats using SimpleITK.

### Key Functions:
*   **`diffusemri.data_io.mhd_utils.read_mhd_data()`**:
    *   Reads image data and metadata from a `.mhd` or `.mha` file using SimpleITK.
    *   Converts the SimpleITK image object to a NumPy array, transposing axis order from SimpleITK's `[z,y,x]` or `[t,z,y,x]` to NumPy/Nibabel's typical `[x,y,z]` or `[x,y,z,t]`.
    *   Constructs a NIfTI-compatible affine matrix from SimpleITK's spatial metadata (`Spacing`, `Origin`, `Direction`).
    *   Attempts to parse DWI-specific information from metadata fields if present (e.g., `modality:=DWMRI`, `DWMRI_b-value`, `DWMRI_gradient_XXXX`). B-vectors are reoriented to the image coordinate system.
    *   Returns the image data (NumPy array), affine matrix, b-values (or None), b-vectors (or None), and the metadata dictionary.
*   **`diffusemri.data_io.mhd_utils.write_mhd_data()`**:
    *   Writes NumPy image data to a `.mhd` or `.mha` file using SimpleITK.
    *   Converts the NIfTI-style affine matrix to SimpleITK's `Spacing`, `Origin`, and `Direction` cosines.
    *   If b-values and b-vectors (assumed to be in image coordinate space) are provided, they are reoriented to world/patient space (consistent with the MHD header's spatial information) and stored in the MHD header using `DWMRI_` prefixed keys.
    *   Allows specification of custom metadata.

### DWI Metadata in MHD
Similar to NRRD, MHD files can store DWI information in their headers. Common conventions include:
*   `modality:=DWMRI`
*   `DWMRI_b-value:=<value>`
*   `DWMRI_gradient_0000:=<gx> <gy> <gz>` (often in world/patient coordinate system)
*   `measurement frame:=<xx> <xy> <xz> <yx> <yy> <yz> <zx> <zy> <zz>` (may define relation of gradients to world/patient axes)

The `read_mhd_data` function attempts to interpret these and reorient b-vectors to the NIfTI image space. The `write_mhd_data` function reorients NIfTI image-space b-vectors to world/patient space before saving them in the MHD header.

## Command-Line Interface (CLI) for Format Conversion

The `run_format_conversion.py` script handles NRRD and MHD conversions.

### `mhd2nii`: MHD/MHA to NIfTI Conversion
*   **Purpose:** Converts an MHD/MHA file (potentially DWI) into NIfTI format. If DWI information is parsed, corresponding `.bval` and `.bvec` files are generated.
*   **CLI Tool:** `run_format_conversion mhd2nii`
*   **Conceptual CLI Usage:**
    ```bash
    # Convert a general MHD/MHA file to NIfTI
    python cli/run_format_conversion.py mhd2nii \\
        --input_mhd /path/to/input.mha \\
        --output_nifti /path/to/output.nii.gz

    # Convert a DWI MHD/MHA file to NIfTI, also saving bval/bvec
    python cli/run_format_conversion.py mhd2nii \\
        --input_mhd /path/to/dwi.mhd \\
        --output_nifti /path/to/output/dwi.nii.gz \\
        --output_bval /path/to/output/dwi.bval \\
        --output_bvec /path/to/output/dwi.bvec
    ```

### `nii2mhd`: NIfTI to MHD/MHA Conversion
*   **Purpose:** Converts a NIfTI file into MHD/MHA format. If associated `.bval` and `.bvec` files are provided, this DWI information is embedded into the output MHD header.
*   **CLI Tool:** `run_format_conversion nii2mhd`
*   **Conceptual CLI Usage:**
    ```bash
    # Convert a general NIfTI file to MHD/MHA
    python cli/run_format_conversion.py nii2mhd \\
        --input_nifti /path/to/input.nii.gz \\
        --output_mhd /path/to/output.mha

    # Convert a DWI NIfTI file (with bval/bvec) to MHD/MHA
    python cli/run_format_conversion.py nii2mhd \\
        --input_nifti /path/to/dwi.nii.gz \\
        --input_bval /path/to/dwi.bval \\
        --input_bvec /path/to/dwi.bvec \\
        --output_mhd /path/to/output/dwi.mha
    ```

**Note:** The accuracy of DWI information (especially b-vector orientation) conversion depends heavily on the correctness and completeness of the header information in the source files and the associated b-vector/affine conventions. Axis ordering differences between SimpleITK (typically ZYX for arrays) and Nibabel (typically XYZ for arrays) are handled during conversion.

## Analyze 7.5 File Support (.hdr/.img)

The library provides basic support for reading and writing Analyze 7.5 format files using Nibabel.

### Key Functions:
*   **`diffusemri.data_io.analyze_utils.read_analyze_data()`**:
    *   Reads image data, affine, and header from an Analyze 7.5 file pair (`.hdr`/`.img`).
    *   Returns the image data (as `np.float32`), a NIfTI-compatible affine matrix, and the Nibabel AnalyzeHeader object.
*   **`diffusemri.data_io.analyze_utils.write_analyze_data()`**:
    *   Writes NumPy image data and a NIfTI-style affine to Analyze 7.5 format.
    *   A minimal Analyze header is created if not provided.

### Limitations of Analyze 7.5 Format:
*   **Orientation Information:** The Analyze 7.5 format has very limited support for storing image orientation and coordinate system information compared to NIfTI. While Nibabel attempts to construct a best-effort affine, this can be ambiguous or inaccurate if the original Analyze header is non-standard or lacks sufficient information. SForm/QForm information, standard in NIfTI for precise orientation, is not part of the Analyze 7.5 standard. Conversions involving Analyze format may lead to loss or misinterpretation of spatial orientation.
*   **Metadata:** Analyze 7.5 has minimal support for rich metadata. It does not natively store DWI-specific information like b-values or b-vectors. Therefore, when converting DWI data to Analyze format, this crucial information will be lost in the image header itself.
*   **Data Types:** Supports a limited range of data types. The provided utilities typically convert data to `float32` for consistency.

### Command-Line Interface (CLI) for Analyze Conversion

The `run_format_conversion.py` script also includes subcommands for Analyze 7.5 format.

#### `analyze2nii`: Analyze to NIfTI Conversion
*   **Purpose:** Converts an Analyze 7.5 file (`.hdr`/`.img`) into NIfTI format.
*   **CLI Tool:** `run_format_conversion analyze2nii`
*   **Conceptual CLI Usage:**
    ```bash
    python cli/run_format_conversion.py analyze2nii \\
        --input_analyze /path/to/input_image.hdr \\
        --output_nifti /path/to/output_image.nii.gz
    ```

#### `nii2analyze`: NIfTI to Analyze Conversion
*   **Purpose:** Converts a NIfTI file into Analyze 7.5 format. Be aware of the potential loss of orientation accuracy and metadata. DWI information (bvals/bvecs) from the NIfTI will not be stored in the Analyze header.
*   **CLI Tool:** `run_format_conversion nii2analyze`
*   **Conceptual CLI Usage:**
    ```bash
    python cli/run_format_conversion.py nii2analyze \\
        --input_nifti /path/to/input_image.nii.gz \\
        --output_analyze /path/to/output_image.hdr
        # (.img file will be created alongside)
    ```

## ISMRMRD File Support (.h5)

*   **Purpose:** To support reading data from ISMRMRD (International Society for Magnetic Resonance in Medicine Raw Data Format) files, typically HDF5 (`.h5`) files. This format is used for storing raw k-space data and reconstructed image data along with acquisition parameters. The initial goal is to extract usable image volumes and relevant metadata, potentially including DWI information if available and interpretable.
*   **Current Status:** `Placeholder - Not Implemented`
*   **Key Functions (Placeholders):**
    *   `diffusemri.data_io.ismrmrd_utils.read_ismrmrd_file()`: Intended to read an ISMRMRD file and extract image data, affine, DWI information (if possible), and metadata. Currently returns placeholder values.
    *   `diffusemri.data_io.ismrmrd_utils.convert_ismrmrd_to_nifti_and_metadata()`: Intended to convert ISMRMRD data to NIfTI and save metadata. Currently a placeholder.
*   **Dependencies:** Requires `ismrmrd` Python package and its underlying HDF5 dependencies. Full implementation will require detailed knowledge of ISMRMRD data structures.
*   **CLI Tool (Placeholder):** `run_format_conversion ismrmrd_convert`
    *   **Purpose:** Converts an ISMRMRD file to NIfTI format and extracts metadata.
    *   **Conceptual CLI Usage (Placeholder):**
        ```bash
        python cli/run_format_conversion.py ismrmrd_convert \\
            --input_ismrmrd /path/to/input.h5 \\
            --output_base /path/to/output/scan_base
            # Output might be scan_base.nii.gz, scan_base_metadata.json, etc.
        ```
    *   **Note:** This CLI command and its underlying functionality are currently placeholders and not implemented.

## Philips PAR/REC File Support

The library supports reading Philips PAR/REC files (versions 4.0, 4.1, 4.2) using Nibabel. This allows conversion of PAR/REC image data, including DWI information, to NIfTI format.

### Key Functions:
*   **`diffusemri.data_io.parrec_utils.read_parrec_data()`**:
    *   Reads image data, affine, b-values, b-vectors (if present), and header information from a PAR/REC file pair.
    *   Uses `nibabel.load()` with options for `strict_sort` (volume sorting) and `scaling` ('dv' or 'fp').
    *   Returns image data (as `np.float32`), affine matrix, b-values array (or None), b-vectors array (or None), and a dictionary containing the source header object.
*   **`diffusemri.data_io.parrec_utils.convert_parrec_to_nifti()`**:
    *   Orchestrates the conversion from PAR/REC to NIfTI.
    *   Calls `read_parrec_data` to load the data.
    *   Saves the image data as a NIfTI file.
    *   If b-values and b-vectors are extracted, they are saved to specified output text files.

### DWI Metadata in PAR/REC
Nibabel's PAR/REC loader (`nibabel.parrec.PARRECImage`) handles the extraction of b-values and b-vectors directly from the PAR file header if the acquisition was a diffusion scan. These are made available via the `header.get_bvals_bvecs()` method. The b-vectors are typically provided in an RAS (scanner) coordinate system.

### Command-Line Interface (CLI) for PAR/REC Conversion

The `run_format_conversion.py` script includes a subcommand for PAR/REC to NIfTI conversion.

#### `parrec2nii`: PAR/REC to NIfTI Conversion
*   **Purpose:** Converts a Philips PAR/REC file pair into NIfTI format. If the PAR/REC is DWI data, it also extracts and saves b-values and b-vectors to separate files.
*   **CLI Tool:** `run_format_conversion parrec2nii`
*   **Conceptual CLI Usage:**
    ```bash
    # Convert PAR/REC (e.g., anatomical or DWI) to NIfTI
    python cli/run_format_conversion.py parrec2nii \\
        --input_parrec /path/to/input_image.par \\
        --output_nifti /path/to/output_image.nii.gz \\
        # Optional: specify bval/bvec output paths if DWI is expected
        --output_bval /path/to/output_dwi.bval \\
        --output_bvec /path/to/output_dwi.bvec \\
        # Optional: control PAR/REC loading parameters
        --scaling_method fp \\
        --no_strict_sort
    ```
*   **Note:** Provide the path to either the `.par` or `.rec` file; Nibabel will locate the corresponding pair.

## NIfTI to DICOM Secondary Capture Conversion

DICOM Secondary Capture (SC) format is used to store image data that is not directly acquired from a modality but is derived or converted from other image formats. This is useful for integrating images from various sources into a PACS environment.

The `diffusemri` library provides a utility to convert 3D or 4D NIfTI images into a series of DICOM Secondary Capture files.

### Key Function:
*   **`diffusemri.data_io.dicom_utils.write_nifti_to_dicom_secondary()`**:
    *   Takes a `nibabel.Nifti1Image` object (3D or 4D) and an output directory.
    *   For 3D NIfTI, each slice becomes a separate DICOM SC instance.
    *   For 4D NIfTI, each slice within each volume becomes a separate DICOM SC instance, with instance numbers incrementing accordingly.
    *   Populates essential DICOM tags including Patient Information, Study/Series UIDs, SOP Class UID (Secondary Capture Image Storage), and image pixel data characteristics.
    *   Allows customization of several DICOM tags through parameters.

### Command-Line Interface (CLI) for NIfTI to DICOM SC Conversion

The `run_format_conversion.py` script includes a subcommand for this conversion.

#### `nii2dicom_sec`: NIfTI to DICOM Secondary Capture
*   **Purpose:** Converts a 3D or 4D NIfTI file into a series of DICOM Secondary Capture files, which are saved in a specified output directory.
*   **CLI Tool:** `run_format_conversion nii2dicom_sec`
*   **Conceptual CLI Usage:**
    ```bash
    python cli/run_format_conversion.py nii2dicom_sec \\
        --input_nifti /path/to/input_image.nii.gz \\
        --output_dicom_dir /path/to/output_dicom_series_directory/ \\
        --patient-id "PAT001" \\
        --study-uid "1.2.826.0.1.3680043.8.498.12345" \\
        --series-uid "1.2.826.0.1.3680043.8.498.54321" \\
        --series-description "Converted NIfTI" \\
        --series-number 101 \\
        --create-dirs
    ```
*   **Key Arguments:**
    *   `--input_nifti`: Path to the input NIfTI file (.nii, .nii.gz).
    *   `--output_dicom_dir`: Directory where the output DICOM files will be saved.
    *   `--patient-id` (optional): Patient ID for the DICOM files. Default: "Anonymous".
    *   `--study-uid` (optional): Study Instance UID. If not provided, a new UID will be generated.
    *   `--series-uid` (optional): Series Instance UID. If not provided, a new UID will be generated.
    *   `--series-description` (optional): Series Description. Default: "NIfTI Secondary Capture".
    *   `--series-number` (optional): Series Number. Default: 999.
    *   `--sop-instance-uid-prefix` (optional): Custom prefix for SOP Instance UIDs.
    *   `--initial-instance-number` (optional): Starting instance number for the DICOM series. Default: 1.
    *   `--create-dirs` (flag): Create the output directory if it doesn't exist.
