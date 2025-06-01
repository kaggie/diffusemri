# 07: Format Conversion (NRRD & MHD)

This section details the functionalities for converting imaging data between NRRD (Nearly Raw Raster Data), MHD (MetaImageHeader), and NIfTI formats, with special considerations for diffusion MRI (dMRI) data.

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
