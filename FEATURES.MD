# `diffusemri` Library Features and Status

This document details the current features and status of the `diffusemri` library components.

## CLI (Command Line Interface)

### `cli.cli_utils`
*   **`add_common_input_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser`**
    *   Description: Adds common input file arguments (DWI, bval, bvec, mask) to an `ArgumentParser`.
    *   Status: `Implemented`
*   **`add_common_output_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser`**
    *   Description: Adds common output arguments (output_prefix, output_dir) to an `ArgumentParser`.
    *   Status: `Implemented`
*   **`add_device_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser`**
    *   Description: Adds a `--device` argument (cpu/cuda/auto) to an `ArgumentParser`.
    *   Status: `Implemented`
*   **`determine_device(requested_device: str) -> torch.device`**
    *   Description: Determines the `torch.device` (CPU/CUDA) based on user request and hardware availability.
    *   Status: `Implemented`
*   **`load_nifti_data(filepath: str, ensure_float32: bool = True) -> tuple[np.ndarray, np.ndarray]`**
    *   Description: Loads data and affine from a NIFTI file, with an option to ensure data is float32.
    *   Status: `Implemented`
*   **`load_bvals_bvecs(bval_filepath: str, bvec_filepath: str) -> tuple[np.ndarray, np.ndarray]`**
    *   Description: Loads b-values and b-vectors, automatically handling FSL's 3xN b-vector format.
    *   Status: `Implemented`
*   **`save_nifti_data(data: np.ndarray, affine: np.ndarray, filepath: str)`**
    *   Description: Saves NumPy data as a NIFTI file, creating output directories if needed.
    *   Status: `Implemented`
*   **`save_tractogram(streamlines: list, affine: np.ndarray, filepath: str, image_shape_for_sft: tuple = None)`**
    *   Description: Saves streamlines to a .trk file using Dipy's `StatefulTractogram`.
    *   Status: `Implemented (Wraps Dipy)`
*   **`load_config_from_json_yaml(filepath: str) -> dict`**
    *   Description: Loads parameters from a JSON or YAML configuration file.
    *   Status: `Implemented`

### `cli.run_dti_fit`
*   **`main()` (and internal `run_dti_fitting`)**
    *   Description: Command-line interface for DTI model fitting. Parses arguments, loads data, performs DTI fitting using `fitting.dti_fitter.fit_dti_volume`, and saves output DTI parameter maps (FA, MD, AD, RD, Tensor).
    *   Status: `Implemented`

### `cli.run_noddi_fit`
*   **`main()` (and internal `run_noddi_fitting`)**
    *   Description: Command-line interface for NODDI model fitting. Parses arguments, manages configuration (including JSON config file for detailed model parameters), loads data, performs NODDI fitting using `fitting.noddi_fitter.fit_noddi_volume` (PyTorch-based), and saves output parameter maps.
    *   Status: `Implemented`

### `cli.run_preprocessing`
*   **Subcommand `create_mask` (internal `run_masking`)**
    *   Description: Creates a brain mask from DWI data using a PyTorch-based median filter and Otsu thresholding.
    *   Status: `Implemented (PyTorch based)`
*   **Subcommand `denoise_mppca` (internal `run_denoising_mppca`)**
    *   Description: Denoises DWI data using Marchenko-Pastur PCA (MP-PCA) (PyTorch-based).
    *   Status: `Implemented (PyTorch based)`
*   **Subcommand `correct_fsl` (internal `run_correct_fsl`)**
    *   Description: Performs motion and eddy current correction by wrapping FSL's `eddy` tool via Nipype.
    *   Status: `Implemented (Relies on FSL)`
*   **Subcommand `topup_fsl` (internal `run_topup_fsl`)**
    *   Description: Performs susceptibility distortion correction using FSL's `topup` and `applytopup` tools via Nipype.
    *   Status: `Implemented (Relies on FSL)`
*   **Subcommand `bias_field_dipy` (internal `run_bias_field_correction_dipy`)**
    *   Description: Performs bias field correction using Dipý's N4-based `BiasFieldCorrectionFlow`.
    *   Status: `Implemented (Wraps Dipy)`
*   **Subcommand `gibbs_ringing_dipy` (internal `run_gibbs_ringing_correction_dipy`)**
    *   Description: Performs Gibbs ringing correction using Dipý's `gibbs_removal` function.
    *   Status: `Implemented (Wraps Dipy)`
*   **Subcommand `dicom_to_nifti` (internal `run_dicom_to_nifti`)**
    *   Description: Converts DICOM series (anatomical or DWI) to NIfTI format. For DWI, extracts and saves b-values and b-vectors.
    *   Status: `Implemented`
*   **Subcommand `anonymize_dicom` (internal `run_anonymize_dicom`)**
    *   Description: Anonymizes DICOM files (single or directory) by editing tags based on default or custom rules.
    *   Status: `Implemented`
*   **`main()`**
    *   Description: Main CLI entry point for `run_preprocessing.py`, dispatching to various subcommands.
    *   Status: `Implemented`

### `cli.run_format_conversion`
*   **Subcommand `nrrd2nii` (internal `run_nrrd_to_nifti`)**
    *   Description: Converts a NRRD file to NIfTI format, optionally extracting DWI information.
    *   Status: `Implemented`
*   **Subcommand `nii2nrrd` (internal `run_nifti_to_nrrd`)**
    *   Description: Converts a NIfTI file to NRRD format, optionally embedding DWI information.
    *   Status: `Implemented`
*   **Subcommand `mhd2nii` (internal `run_mhd_to_nifti`)**
    *   Description: Converts a MHD/MHA file to NIfTI format, optionally extracting DWI information.
    *   Status: `Implemented`
*   **Subcommand `nii2mhd` (internal `run_nifti_to_mhd`)**
    *   Description: Converts a NIfTI file to MHD/MHA format, optionally embedding DWI information.
    *   Status: `Implemented`
*   **Subcommand `analyze2nii` (internal `run_analyze_to_nifti`)**
    *   Description: Converts an Analyze 7.5 file (.hdr/.img) to NIfTI format.
    *   Status: `Implemented`
*   **Subcommand `nii2analyze` (internal `run_nifti_to_analyze`)**
    *   Description: Converts a NIfTI file to Analyze 7.5 format (.hdr/.img).
    *   Status: `Implemented`
*   **Subcommand `parrec2nii` (internal `run_parrec_to_nifti`)**
    *   Description: Converts a Philips PAR/REC file to NIfTI format, extracting DWI information if available.
    *   Status: `Implemented`
*   **Subcommand `nii2dicom_sec` (internal `run_nifti_to_dicom_secondary`)**
    *   Description: Converts a 3D or 4D NIfTI file to a series of DICOM Secondary Capture files.
    *   Status: `Implemented`
*   **Subcommand `ismrmrd_convert` (internal `run_ismrmrd_convert`)**
    *   Description: Placeholder CLI for converting ISMRMRD files to NIfTI and extracting metadata.
    *   Status: `Placeholder - Not Implemented`
*   **Subcommand `bruker2nii` (aliases: `bruker_to_nifti`) (internal `run_bruker_to_nifti`)**
    *   Description: Converts Bruker ParaVision DWI data to NIfTI format, including b-values and b-vectors.
    *   Status: `Implemented`
*   **`main()`**
    *   Description: Main CLI entry point for `run_format_conversion.py`, dispatching to various subcommands.
    *   Status: `Implemented`

### `cli.run_tracking`
*   **`parse_seeds(seed_arg: str, affine_np: np.ndarray) -> np.ndarray`**
    *   Description: Parses seed arguments from CLI, loading from a NIFTI mask file or a string of coordinates.
    *   Status: `Implemented`
*   **Subcommand `det_oudf` (internal `run_deterministic_tracking`)**
    *   Description: Runs deterministic tractography using ODFs. Manages parameters from CLI and config file, loads data, and saves output streamlines. Uses the `tracking.deterministic.track_deterministic_oudf` function.
    *   Status: `Implemented`
*   **`main(cli_args=None)`**
    *   Description: Main CLI entry point for `run_tracking.py`, dispatching to tracking algorithm subcommands.
    *   Status: `Implemented`

## Data I/O Modules

### `data_io.nifti` (and `data_io.load_dmri`)
*   **`load_nifti_dwi(filepath: str) -> tuple[np.ndarray, np.ndarray]`**
    *   Description: Loads 4D DWI data (as float32) and affine from a NIFTI file.
    *   Status: `Implemented`
*   **`load_nifti_mask(filepath: str) -> tuple[np.ndarray, np.ndarray]`**
    *   Description: Loads a 3D mask (as boolean) and affine from a NIFTI file.
    *   Status: `Implemented`
*   **`load_fsl_bvecs(filepath: str) -> np.ndarray`**
    *   Description: Loads b-vectors from an FSL-formatted text file (handles 3xN or Nx3).
    *   Status: `Implemented`
*   **`load_fsl_bvals(filepath: str) -> np.ndarray`**
    *   Description: Loads b-values from an FSL-formatted text file.
    *   Status: `Implemented`
*   **`load_nifti_study(...)` (from `load_dmri`)**
    *   Description: Loads a complete dMRI study (DWI, bvals, bvecs, optional mask) into memory.
    *   Status: `Implemented`

### `data_io.dicom_utils`
*   **`read_dicom_series(dicom_dir: str) -> list[pydicom.FileDataset]`**
    *   Description: Reads and sorts DICOM files from a directory by InstanceNumber or filename.
    *   Status: `Implemented`
*   **`extract_pixel_data_and_affine(...)`**
    *   Description: Extracts pixel data, constructs affine, and gathers basic metadata from a sorted DICOM series.
    *   Status: `Implemented`
*   **`extract_dwi_metadata(...)`**
    *   Description: Extracts b-values, b-vectors (reoriented to image space), and DWI metadata from a sorted DICOM series.
    *   Status: `Implemented`
*   **`convert_dicom_to_nifti_main(...)`**
    *   Description: Orchestrates non-DWI DICOM to NIfTI conversion, saving image and a JSON metadata sidecar.
    *   Status: `Implemented`
*   **`convert_dwi_dicom_to_nifti(...)`**
    *   Description: Orchestrates DWI DICOM to NIfTI conversion, saving image, bvals, bvecs, and a JSON metadata sidecar.
    *   Status: `Implemented`
*   **`anonymize_dicom_dataset(...)`**
    *   Description: Anonymizes a `pydicom.FileDataset` object in-place based on specified rules.
    *   Status: `Implemented`
*   **`anonymize_dicom_file(...)`**
    *   Description: Reads, anonymizes, and saves a single DICOM file.
    *   Status: `Implemented`
*   **`anonymize_dicom_directory(...)`**
    *   Description: Anonymizes all DICOM files in a directory, optionally preserving structure.
    *   Status: `Implemented`
*   **`write_nifti_to_dicom_secondary(...)`**
    *   Description: Converts a NIfTI image object (3D or 4D) into a series of DICOM Secondary Capture files.
    *   Status: `Implemented`

### `data_io.nrrd_utils`
*   **`read_nrrd_data(nrrd_filepath: str) -> tuple`**
    *   Description: Reads NRRD files, parsing image data, NIfTI-compatible affine, DWI metadata, and header.
    *   Status: `Implemented (Wraps pynrrd)`
*   **`write_nrrd_data(...)`**
    *   Description: Writes image data to NRRD format, constructing a header from affine and optional DWI info.
    *   Status: `Implemented (Wraps pynrrd)`

### `data_io.mhd_utils`
*   **`read_mhd_data(mhd_filepath: str) -> tuple`**
    *   Description: Reads MHD/MHA files using SimpleITK, parsing image data, NIfTI-compatible affine, DWI metadata, and header.
    *   Status: `Implemented (Wraps SimpleITK)`
*   **`write_mhd_data(...)`**
    *   Description: Writes image data to MHD/MHA format using SimpleITK, constructing header from affine and optional DWI metadata.
    *   Status: `Implemented (Wraps SimpleITK)`

### `data_io.analyze_utils`
*   **`read_analyze_data(analyze_filepath: str) -> tuple`**
    *   Description: Reads Analyze 7.5 files (.hdr/.img) using Nibabel, returning image data, affine, and header.
    *   Status: `Implemented (Wraps Nibabel)`
*   **`write_analyze_data(...)`**
    *   Description: Writes image data to Analyze 7.5 format using Nibabel.
    *   Status: `Implemented (Wraps Nibabel)`

### `data_io.parrec_utils`
*   **`read_parrec_data(...)`**
    *   Description: Reads Philips PAR/REC files using Nibabel, returning image data, affine, bvals/bvecs (if DWI), and header.
    *   Status: `Implemented (Wraps Nibabel)`
*   **`convert_parrec_to_nifti(...)`**
    *   Description: Converts PAR/REC data to NIfTI, saving bvals/bvecs if present.
    *   Status: `Implemented`

### `data_io.ismrmrd_utils`
*   **`read_ismrmrd_file(filepath: str) -> tuple`**
    *   Description: Placeholder for reading ISMRMRD files.
    *   Status: `Placeholder - Not Implemented`
*   **`convert_ismrmrd_to_nifti_and_metadata(...)`**
    *   Description: Placeholder for converting ISMRMRD data to NIfTI and extracting metadata.
    *   Status: `Placeholder - Not Implemented`

### `data_io.generic_utils`
*   **`save_dict_to_hdf5(data_dict: dict, hdf5_filepath: str)`**
    *   Description: Saves a dictionary of NumPy arrays to an HDF5 file.
    *   Status: `Implemented (Wraps h5py)`
*   **`load_dict_from_hdf5(hdf5_filepath: str) -> dict`**
    *   Description: Loads data from an HDF5 file into a dictionary of NumPy arrays.
    *   Status: `Implemented (Wraps h5py)`
*   **`save_dict_to_mat(data_dict: dict, mat_filepath: str, ...)`**
    *   Description: Saves a dictionary of NumPy arrays to a .MAT file (version 5).
    *   Status: `Implemented (Wraps SciPy)`
*   **`load_dict_from_mat(mat_filepath: str) -> dict`**
    *   Description: Loads data from a .MAT file into a dictionary, filtering MAT-specific metadata.
    *   Status: `Implemented (Wraps SciPy)`

### `data_io.gradients` (Consolidated)
*   **`create_gradient_table(bvals: np.ndarray, bvecs: np.ndarray, ...) -> dipy.core.gradients.GradientTable`**
    *   Description: Creates a Dipy `GradientTable` object from b-values and b-vectors arrays.
    *   Status: `Implemented (Wraps Dipy)`

## Preprocessing Modules

### `preprocessing.correction`
*   **`correct_motion_eddy_fsl(...)`**
    *   Description: Wraps FSL's `eddy` tool for motion and eddy current correction.
    *   Status: `Implemented (Relies on FSL)`
*   **`load_eddy_outlier_report(...)`**
    *   Description: Parses FSL eddy's outlier report text file.
    *   Status: `Implemented`
*   **`correct_susceptibility_topup_fsl(...)`**
    *   Description: Wraps FSL's `topup` and `applytopup` for susceptibility distortion correction.
    *   Status: `Implemented (Relies on FSL)`
*   **`correct_bias_field_dipy(...)`**
    *   Description: Performs bias field correction using Dipý's `BiasFieldCorrectionFlow` (N4 method).
    *   Status: `Implemented (Wraps Dipy)`

### `preprocessing.denoising`
*   **`denoise_mppca_data(dmri_data: np.ndarray, ...) -> np.ndarray`**
    *   Description: Denoises 4D dMRI data using PyTorch-based Marchenko-Pastur PCA (MP-PCA).
    *   Status: `Implemented (PyTorch based)`
*   **`correct_gibbs_ringing_dipy(...)`**
    *   Description: Corrects Gibbs ringing artifacts using Dipý's `gibbs_removal`.
    *   Status: `Implemented (Wraps Dipy)`

### `preprocessing.masking`
*   **`create_brain_mask(dmri_data: np.ndarray, ...)`**
    *   Description: Computes a 3D brain mask from dMRI data using PyTorch-based median Otsu.
    *   Status: `Implemented (PyTorch based, uses SciPy)`

### `preprocessing.pytorch_denoising` (Internal module for `denoise_mppca_data`)
*   **`pytorch_mppca(...)`**: Core PyTorch MP-PCA implementation.
    *   Status: `Implemented (PyTorch based)`

### `preprocessing.pytorch_masking` (Internal module for `create_brain_mask`)
*   **`pytorch_median_otsu(...)`**: Core PyTorch median Otsu implementation.
    *   Status: `Implemented (PyTorch based, uses SciPy)`

## Fitting Modules

### `fitting.dti_fitter`
*   **`fit_dti_voxel(...)`**
    *   Description: Fits DTI model to a single voxel's signals.
    *   Status: `Implemented`
*   **`fit_dti_volume(...)`**
    *   Description: Fits DTI model voxel-wise to a 4D dMRI dataset, returning parameter maps.
    *   Status: `Implemented`

### `fitting.noddi_fitter`
*   **`preprocess_noddi_input(...)`**
    *   Description: Preprocesses DWI data for NODDI fitting (S0 calculation, normalization, masking).
    *   Status: `Implemented (PyTorch based)`
*   **`fit_noddi_volume(...)`**
    *   Description: Fits the NODDI model to 4D DWI data in batches using PyTorch.
    *   Status: `Implemented (PyTorch based)`

## Model Definitions (`models.*`)

### `models.dti`
*   Contains helper functions for DTI calculations (tensor elements, metrics from tensor).
*   Status: `Implemented`

### `models.dki`
*   **`DkiModel(gtab)`**: Wrapper for Dipy's `DiffusionKurtosisModel`.
    *   `.fit(data, mask)`: Fits model.
    *   Properties: `.fa`, `.md`, `.mk`, `.ak`, `.rk`, `.ka`.
    *   Status: `Implemented (Wraps Dipy)`

### `models.csd`
*   **`CsdModel(gtab, response, sh_order_max)` (renamed from `MultiTissueCsdModel` for clarity if it's single-tissue focused, or verify if it's truly MSMT-CSD wrapper)**
    *   Description: Wrapper for Dipy's CSD model (and potentially MSMT-CSD if `response_wm`, `response_gm`, `response_csf` are handled).
    *   `.fit(data, mask)`: Fits model.
    *   Properties/Methods: `.gfa`, `.odf()`. For MSMT-CSD: `.wm_odf()`, `.gm_fraction()`, `.csf_fraction()`.
    *   Status: `Implemented (Wraps Dipy)` (Clarify if single or multi-tissue based on actual wrapper code)

### `models.qball`
*   **`QballModel(gtab, sh_order_max, smooth)`**: Wrapper for Dipy's `QballModel` (CSA type).
    *   `.fit(data, mask)`: Fits model.
    *   Properties/Methods: `.gfa`, `.odf()`.
    *   Status: `Implemented (Wraps Dipy)`

### `models.mapmri`
*   **`MapmriModel(gtab, ...)`**: Wrapper for Dipy's `MapmriModel`.
    *   `.fit(data, mask)`: Fits model.
    *   Properties: `.rtop`, `.rtap`, `.rtpp`, `.msd`, `.qiv`, `.ng`, `.gfa`. Methods: `.odf()`.
    *   Status: `Implemented (Wraps Dipy)`

### `models.noddi_model` (PyTorch Implementation)
*   **`NoddiModelTorch(gtab, d_intra, d_iso)`**: Core PyTorch `nn.Module` for NODDI.
    *   `.fit_batch(...)`: Fits a batch of voxels.
    *   Status: `Implemented (PyTorch based)`

### `models.noddi_signal` (PyTorch Implementation)
*   Contains functions for computing NODDI compartment signals and the total signal model.
*   Status: `Implemented (PyTorch based)`

## Tracking Modules

### `tracking.deterministic`
*   **`track_deterministic_oudf(...)`**: Performs deterministic tractography using an internally fitted ODF model (CSD-like, PyTorch-based).
    *   Status: `Implemented (PyTorch based)`

### `tracking.probabilistic`
*   **`track_probabilistic_odf(...)`**: Performs probabilistic tractography using ODFs from a Dipy fitted model object.
    *   Status: `Implemented (Wraps Dipy)`

### `tracking.pytorch_tracking_pipeline` (Internal module for `track_deterministic_oudf`)
*   Contains PyTorch-based components for peak extraction, direction getting, stopping criteria, and local tracking.
*   Status: `Implemented (PyTorch based)`

## Utility Modules

### `utils.pytorch_gradient_utils`
*   **`PyTorchGradientTable(bvals, bvecs, ...)`**: Class to manage gradient table information using PyTorch tensors, for use with PyTorch-based models/fitters.
    *   Status: `Implemented (PyTorch based)`

---
*Note: Some function signatures might be slightly simplified for brevity in this document. Refer to source code for full details. The `diffusemri.` prefix is omitted from module paths for conciseness.*
*Class methods/properties listed under their respective classes are typically public unless prefixed with an underscore.*
*Status indicates current implementation state. "Wraps Dipy/SimpleITK/etc." means it primarily uses that library's functions.*
