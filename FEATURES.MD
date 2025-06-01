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
*   **`run_dti_fitting(args: argparse.Namespace)`**
    *   Description: Performs DTI model fitting using loaded DWI data, b-values, b-vectors, and an optional mask. Saves output DTI parameter maps (FA, MD, AD, RD, Tensor).
    *   Status: `Implemented`
*   **`main()`**
    *   Description: Command-line interface entry point for DTI model fitting. Parses arguments and calls `run_dti_fitting`.
    *   Status: `Implemented`

### `cli.run_noddi_fit`
*   **`run_noddi_fitting(args: argparse.Namespace)`**
    *   Description: Performs NODDI model fitting using loaded dMRI data. Supports parameter specification via a configuration file and CLI overrides for device and b0_threshold. Handles initial orientation map loading.
    *   Status: `Implemented (PyTorch based)`
*   **`main(cli_args_list=None)`**
    *   Description: Command-line interface entry point for NODDI model fitting. Parses arguments, manages configuration, and calls `run_noddi_fitting`.
    *   Status: `Implemented`

### `cli.run_preprocessing`
*   **`run_masking(args: argparse.Namespace)`**
    *   Description: Creates a brain mask from DWI data using median filter and Otsu thresholding.
    *   Status: `Implemented (PyTorch based)`
*   **`run_denoising_mppca(args: argparse.Namespace)`**
    *   Description: Denoises DWI data using Marchenko-Pastur PCA (MP-PCA).
    *   Status: `Implemented (PyTorch based)`
*   **`run_correct_fsl(args: argparse.Namespace)`**
    *   Description: Performs motion and eddy current correction by wrapping FSL's `eddy` tool via Nipype.
    *   Status: `Implemented (Relies on FSL)`
*   **`run_topup_fsl(args: argparse.Namespace)`**
    *   Description: Performs susceptibility distortion correction using FSL's TOPUP and ApplyTOPUP.
    *   Status: `Implemented (Relies on FSL)`
*   **`run_bias_field_correction_dipy(args: argparse.Namespace)`**
    *   Description: Performs bias field correction using Dipý's BiasFieldCorrectionFlow (e.g., N4 method).
    *   Status: `Implemented (Wraps Dipy)`
*   **`run_gibbs_ringing_correction_dipy(args: argparse.Namespace)`**
    *   Description: Performs Gibbs ringing correction using Dipý's `gibbs_removal` function.
    *   Status: `Implemented (Wraps Dipy)`
*   **`run_dicom_to_nifti(args: argparse.Namespace)`**
    *   Description: Converts DICOM series to NIfTI format, with options for DWI data (bvals/bvecs).
    *   Status: `Implemented`
*   **`run_anonymize_dicom(args: argparse.Namespace)`**
    *   Description: Anonymizes DICOM files (single or directory) by editing tags based on default or custom rules.
    *   Status: `Implemented`
*   **`main()`**
    *   Description: Command-line interface entry point for various preprocessing tasks including DICOM utilities. Uses subparsers for each task.
    *   Status: `Implemented`

### `cli.run_format_conversion`
*   **`run_nrrd_to_nifti(args: argparse.Namespace)`**
    *   Description: Converts a NRRD file to NIfTI format, optionally extracting DWI information (bvals/bvecs) if present in NRRD header.
    *   Status: `Implemented`
*   **`run_nifti_to_nrrd(args: argparse.Namespace)`**
    *   Description: Converts a NIfTI file to NRRD format, optionally embedding DWI bvals/bvecs from text files into NRRD header.
    *   Status: `Implemented`
*   **`main()`**
    *   Description: Command-line interface entry point for format conversion tasks (NRRD <-> NIfTI).
    *   Status: `Implemented`

### `cli.run_format_conversion`
*   **`run_nrrd_to_nifti(args: argparse.Namespace)`**
    *   Description: Converts a NRRD file to NIfTI format, optionally extracting DWI information (bvals/bvecs) if present in NRRD header.
    *   Status: `Implemented`
*   **`run_nifti_to_nrrd(args: argparse.Namespace)`**
    *   Description: Converts a NIfTI file to NRRD format, optionally embedding DWI bvals/bvecs from text files into NRRD header.
    *   Status: `Implemented`
*   **`run_mhd_to_nifti(args: argparse.Namespace)`**
    *   Description: Converts a MHD/MHA file to NIfTI format, optionally extracting DWI information (bvals/bvecs) if present in MHD header.
    *   Status: `Implemented`
*   **`run_nifti_to_mhd(args: argparse.Namespace)`**
    *   Description: Converts a NIfTI file to MHD/MHA format, optionally embedding DWI bvals/bvecs from text files into MHD header.
    *   Status: `Implemented`
*   **`run_analyze_to_nifti(args: argparse.Namespace)`**
    *   Description: Converts an Analyze 7.5 file (.hdr/.img) to NIfTI format.
    *   Status: `Implemented`
*   **`run_nifti_to_analyze(args: argparse.Namespace)`**
    *   Description: Converts a NIfTI file to Analyze 7.5 format (.hdr/.img).
    *   Status: `Implemented`
*   **`main()`**
    *   Description: Command-line interface entry point for format conversion tasks (NRRD, MHD, Analyze <-> NIfTI).
    *   Status: `Implemented`

### `cli.run_tracking`
*   **`parse_seeds(seed_arg: str, affine_np: np.ndarray) -> np.ndarray`**
    *   Description: Parses seed arguments, loading from a NIFTI mask file or a string of coordinates.
    *   Status: `Implemented`
*   **`run_deterministic_tracking(args: argparse.Namespace)`**
    *   Description: Runs deterministic tractography using ODFs. Manages parameters from CLI and config file, loads data, and saves output streamlines. Uses the `diffusemri.tracking.deterministic.track_deterministic_oudf` wrapper.
    *   Status: `Implemented`
*   **`main(cli_args=None)`**
    *   Description: Command-line interface entry point for deterministic tractography.
    *   Status: `Implemented`

## Data I/O

### `data_io.gradients`
*   **`create_gradient_table(bvals: np.ndarray, bvecs: np.ndarray, b0_threshold: float = 50.0, atol: float = 1e-2) -> dipy.core.gradients.GradientTable`**
    *   Description: Creates a Dipy `GradientTable` object from b-values and b-vectors arrays, with input validation.
    *   Status: `Implemented (Wraps Dipy)`

### `data_io.load_dmri`
*   **`load_nifti_dmri_data(nifti_filepath: str, bval_filepath: str, bvec_filepath: str) -> tuple`**
    *   Description: Loads dMRI data from a NIFTI file, b-values, and b-vectors. It handles basic validation like checking 4D NIFTI, matching number of volumes with b-values/vectors, and transposing FSL-style b-vectors. Returns `(image_data, b_values, b_vectors)` or `(None, None, None)` on error.
    *   Status: `Implemented`

### `data_io.dicom_utils`
*   **`read_dicom_series(dicom_dir: str) -> list[pydicom.FileDataset]`**
    *   Description: Reads and sorts DICOM files from a directory.
    *   Status: `Implemented`
*   **`extract_pixel_data_and_affine(sorted_dicoms: list[pydicom.FileDataset]) -> tuple[np.ndarray | None, np.ndarray | None, dict]`**
    *   Description: Extracts pixel data, constructs affine, and gathers basic metadata from DICOM series.
    *   Status: `Implemented`
*   **`extract_dwi_metadata(sorted_dicoms: list[pydicom.FileDataset], image_orientation_patient: list[float]) -> tuple[np.ndarray | None, np.ndarray | None, dict]`**
    *   Description: Extracts b-values, b-vectors (reoriented to image space), and DWI metadata from DICOM series.
    *   Status: `Implemented`
*   **`convert_dicom_to_nifti_main(dicom_dir: str, output_nifti_file: str) -> bool`**
    *   Description: Orchestrates non-DWI DICOM to NIfTI conversion, saving image and a JSON metadata sidecar.
    *   Status: `Implemented`
*   **`convert_dwi_dicom_to_nifti(dicom_dir: str, output_nifti_file: str, output_bval_file: str, output_bvec_file: str) -> bool`**
    *   Description: Orchestrates DWI DICOM to NIfTI conversion, saving image, bvals, bvecs, and a JSON metadata sidecar.
    *   Status: `Implemented`
*   **`anonymize_dicom_dataset(dataset: pydicom.FileDataset, anonymization_rules: dict = None, custom_replacements: dict = None) -> None`**
    *   Description: Anonymizes a DICOM dataset in-place based on specified rules (default or custom).
    *   Status: `Implemented`
*   **`anonymize_dicom_file(input_dicom_path: str, output_dicom_path: str, anonymization_rules: dict = None, custom_replacements: dict = None) -> bool`**
    *   Description: Reads, anonymizes, and saves a single DICOM file.
    *   Status: `Implemented`
*   **`anonymize_dicom_directory(input_dir: str, output_dir: str, anonymization_rules: dict = None, custom_replacements: dict = None, preserve_structure: bool = True) -> tuple[int, int]`**
    *   Description: Anonymizes all DICOM files in a directory, optionally preserving structure.
    *   Status: `Implemented`

### `data_io.gradients`
*   **`create_gradient_table(bvals: np.ndarray, bvecs: np.ndarray, b0_threshold: float = 50.0, atol: float = 1e-2) -> dipy.core.gradients.GradientTable`**
    *   Description: Creates a Dipy `GradientTable` object from b-values and b-vectors arrays, with input validation.
    *   Status: `Implemented (Wraps Dipy)`

### `data_io.load_dmri`
*   **`load_nifti_dmri_data(nifti_filepath: str, bval_filepath: str, bvec_filepath: str) -> tuple`**
    *   Description: Loads dMRI data from a NIFTI file, b-values, and b-vectors. It handles basic validation like checking 4D NIFTI, matching number of volumes with b-values/vectors, and transposing FSL-style b-vectors. Returns `(image_data, b_values, b_vectors)` or `(None, None, None)` on error.
    *   Status: `Implemented`

### `data_io.nifti`
*   **`load_nifti_dwi(filepath: str) -> tuple[np.ndarray, np.ndarray]`**
    *   Description: Loads 4D DWI data (as float32) and affine from a NIFTI file, with dimension validation.
    *   Status: `Implemented`
*   **`load_nifti_mask(filepath: str) -> tuple[np.ndarray, np.ndarray]`**
    *   Description: Loads a 3D mask (as boolean) and affine from a NIFTI file, with dimension validation.
    *   Status: `Implemented`
*   **`load_fsl_bvecs(filepath: str) -> np.ndarray`**
    *   Description: Loads b-vectors from an FSL-formatted text file, handling 3xN or Nx3 formats and returning Nx3.
    *   Status: `Implemented`
*   **`load_fsl_bvals(filepath: str) -> np.ndarray`**
    *   Description: Loads b-values from an FSL-formatted text file into a 1D NumPy array.
    *   Status: `Implemented`
*   **`load_nifti_study(dwi_path: str, bval_path: str, bvec_path: str, mask_path: str = None, b0_threshold: float = 50.0, atol: float = 1e-2) -> tuple[np.ndarray, dipy.core.gradients.GradientTable, np.ndarray | None, np.ndarray]`**
    *   Description: Loads a complete dMRI study (DWI, bvals, bvecs, optional mask) and returns DWI data, Dipy GradientTable, optional mask data, and DWI affine. Includes consistency checks.
    *   Status: `Implemented`

### `data_io.nrrd_utils`
*   **`read_nrrd_data(nrrd_filepath: str) -> tuple`**
    *   Description: Reads NRRD files, attempting to parse image data, NIfTI-compatible affine, DWI metadata (bvals/bvecs), and the full header.
    *   Status: `Implemented (Wraps nrrd)`
*   **`write_nrrd_data(output_filepath: str, data: np.ndarray, affine: np.ndarray, bvals: np.ndarray = None, bvecs: np.ndarray = None, custom_fields: dict = None, nrrd_header_options: dict = None)`**
    *   Description: Writes image data to NRRD format, constructing a header from affine, optional DWI info, and custom fields.
    *   Status: `Implemented (Wraps nrrd)`

### `data_io.mhd_utils`
*   **`read_mhd_data(mhd_filepath: str) -> tuple`**
    *   Description: Reads MHD/MHA files using SimpleITK, parses image data, NIfTI-compatible affine, DWI metadata, and header.
    *   Status: `Implemented (Wraps SimpleITK)`
*   **`write_mhd_data(output_filepath: str, data: np.ndarray, affine: np.ndarray, bvals: np.ndarray = None, bvecs: np.ndarray = None, custom_metadata: dict = None)`**
    *   Description: Writes image data to MHD/MHA format using SimpleITK, constructing header from affine and optional DWI/custom metadata.
    *   Status: `Implemented (Wraps SimpleITK)`

### `data_io.analyze_utils`
*   **`read_analyze_data(analyze_filepath: str) -> tuple`**
    *   Description: Reads Analyze 7.5 files (.hdr/.img) using Nibabel, returning image data, affine, and header.
    *   Status: `Implemented (Wraps Nibabel)`
*   **`write_analyze_data(output_filepath: str, data: np.ndarray, affine: np.ndarray, header: nib.analyze.AnalyzeHeader = None)`**
    *   Description: Writes image data to Analyze 7.5 format using Nibabel.
    *   Status: `Implemented (Wraps Nibabel)`

## Fitting

### `fitting.dti_fitter`
*   **`fit_dti_voxel(voxel_signals_dw: np.ndarray, S0_voxel: float, b_values_dw: np.ndarray, b_vectors_dw: np.ndarray, min_signal_threshold: float = 1e-6) -> tuple`**
    *   Description: Fits the DTI model to diffusion signals from a single voxel. Returns DTI metrics (MD, FA, AD, RD), the diffusion tensor, eigenvalues, and eigenvectors.
    *   Status: `Implemented`
*   **`fit_dti_volume(image_data_4d: np.ndarray, b_values: np.ndarray, b_vectors: np.ndarray, b0_threshold: float = 50.0, min_S0_intensity_threshold: float = 1.0) -> dict`**
    *   Description: Fits the DTI model voxel-wise to a 4D dMRI dataset. Returns a dictionary of 3D parameter maps (MD, FA, AD, RD, D_tensor_map).
    *   Status: `Implemented`

### `fitting.noddi_fitter`
*   **`preprocess_noddi_input(dwi_data: np.ndarray, gtab: utils.pytorch_gradient_utils.PyTorchGradientTable, mask: Optional[np.ndarray] = None, s0_norm_method: str = "mean", min_s0_val: float = 1.0) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]`**
    *   Description: Preprocesses DWI data for NODDI fitting: calculates S0, creates a valid voxel mask, normalizes DWI signals by S0, and converts outputs to PyTorch tensors.
    *   Status: `Implemented (PyTorch based)`
*   **`fit_noddi_volume(dwi_data: np.ndarray, gtab: utils.pytorch_gradient_utils.PyTorchGradientTable, mask: Optional[np.ndarray] = None, s0_norm_method: str = "mean", min_s0_val: float = 1.0, batch_size: int = 512, fit_params: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None, d_intra: float = ..., d_iso: float = ..., initial_orientation_map: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]`**
    *   Description: Fits the NODDI model to a 3D volume from 4D DWI data in batches. Manages preprocessing, model initialization, batch fitting (with optional initial orientation map and regularization), and returns 3D parameter maps.
    *   Status: `Implemented (PyTorch based)`

## GUI

### `gui.dmrifittingwidget`
*   **`DMRIFittingWidget(QWidget)`**
    *   Description: A PyQt5 widget for loading dMRI data, selecting diffusion models (DTI, QBall, MT-CSD), fitting them, and displaying results (scalar maps or ODF placeholders).
    *   Status: `GUI Component`
    *   **`handle_load_dmri_data()`**: Loads NIfTI, bval, bvec files.
    *   **`handle_fit_volume()`**: Fits the selected diffusion model to the loaded data.
    *   **`update_output_display()`**: Updates the display area based on the fitted model and selected output type (e.g., FA map, GM fraction, ODF placeholder).
    *   **`populate_map_selector()`**: Populates a dropdown for selecting DTI-specific scalar maps (FA, MD etc.) when legacy DTI fitting is used.
    *   **`handle_map_selection_changed()`**: Handles display of selected legacy DTI maps.

## Models

### `models.csd`
*   **`MultiTissueCsdModel(gtab: dipy.core.gradients.GradientTable)`**
    *   Description: A wrapper for Dipy's Multi-Shell Multi-Tissue Constrained Spherical Deconvolution (MSMT-CSD) model. It can estimate multi-tissue response functions using D'Hollander's algorithm if not provided, and then fits the MSMT-CSD model.
    *   Status: `Implemented (Wraps Dipy)`
    *   **`fit(self, data: np.ndarray, response_wm=None, response_gm=None, response_csf=None, **dhollander_kwargs)`**: Fits the MSMT-CSD model to 4D DWI data.
    *   **`wm_odf(self, sphere: dipy.data.Sphere = None) -> np.ndarray`**: Calculates the White Matter (WM) Orientation Distribution Functions (ODFs) from the fitted model.
    *   **`gm_fraction(self) -> np.ndarray`**: Calculates the Grey Matter (GM) volume fraction map from the fitted model.
    *   **`csf_fraction(self) -> np.ndarray`**: Calculates the Cerebrospinal Fluid (CSF) volume fraction map from the fitted model.

### `models.dki`
*   **`DkiModel(gtab: dipy.core.gradients.GradientTable)`**
    *   Description: A wrapper class for Dipy's `DiffusionKurtosisModel`.
    *   Status: `Implemented (Wraps Dipy)`
    *   **`fit(self, data: np.ndarray)`**: Fits the DKI model to 4D DWI data.
    *   **`fa` (property)**: Returns the 3D Fractional Anisotropy (FA) map.
    *   **`md` (property)**: Returns the 3D Mean Diffusivity (MD) map.
    *   **`mk` (property)**: Returns the 3D Mean Kurtosis (MK) map.
    *   **`ak` (property)**: Returns the 3D Axial Kurtosis (AK) map.
    *   **`rk` (property)**: Returns the 3D Radial Kurtosis (RK) map.
    *   **`ka` (property)**: Returns the 3D Kurtosis Anisotropy (KA) map.

### `models.dti`
*   **`calculate_D_elements(log_S_ratio: np.ndarray, b_design_matrix: np.ndarray) -> np.ndarray | None`**
    *   Description: Estimates the 6 unique diffusion tensor elements (Dxx, Dyy, Dzz, Dxy, Dxz, Dyz) using linear least squares.
    *   Status: `Implemented`
*   **`tensor_elements_to_matrix(D_elements: np.ndarray) -> np.ndarray | None`**
    *   Description: Converts the 6 unique diffusion tensor elements into a 3x3 symmetric diffusion tensor matrix.
    *   Status: `Implemented`
*   **`calculate_dti_metrics(D_tensor_matrix: np.ndarray) -> tuple`**
    *   Description: Calculates DTI metrics (MD, FA, AD, RD, sorted eigenvalues, sorted eigenvectors) from a 3x3 diffusion tensor matrix. Handles eigenvalue clamping for stability.
    *   Status: `Implemented`

### `models.mapmri`
*   **`MapmriModel(gtab: dipy.core.gradients.GradientTable, radial_order: int = 6, laplacian_regularization: bool = True, laplacian_weighting: float = 0.2)`**
    *   Description: A wrapper class for Dipy's `MapmriModel`. MAP-MRI (Mean Apparent Propagator MRI) estimates various microstructure-sensitive parameters and requires multi-shell data.
    *   Status: `Implemented (Wraps Dipy)`
    *   **`fit(self, data: np.ndarray)`**: Fits the MAP-MRI model to 4D DWI data.
    *   **`rtop` (property)**: Returns the 3D Return to Origin Probability (RTOP) map.
    *   **`rtap` (property)**: Returns the 3D Return to Axis Probability (RTAP) map.
    *   **`rtpp` (property)**: Returns the 3D Return to Plane Probability (RTPP) map.
    *   **`msd` (property)**: Returns the 3D Mean Squared Displacement (MSD) map.
    *   **`qiv` (property)**: Returns the 3D Q-space Inverse Variance (QIV) map.
    *   **`ng` (property)**: Returns the 3D Non-Gaussianity (NG) map.
    *   **`odf(self, sphere: dipy.data.Sphere = None) -> np.ndarray`**: Calculates Orientation Distribution Functions (ODFs) from the fitted model.
    *   **`gfa` (property)**: Returns the 3D Generalized Fractional Anisotropy (GFA) map, computed as `sqrt(QIV) * RTAP`.

### `models.noddi_model`
*   **`NoddiModelTorch(gtab: utils.pytorch_gradient_utils.PyTorchGradientTable, d_intra: float = ..., d_iso: float = ...)`**
    *   Description: A PyTorch module for fitting the NODDI model. It handles parameter transformations for constrained optimization and includes a batch-wise fitting procedure using gradient descent.
    *   Status: `Implemented (PyTorch based)`
    *   **`kappa_to_odi(self, kappa: torch.Tensor) -> torch.Tensor`**: Converts Watson concentration parameter kappa to Orientation Dispersion Index (ODI).
    *   **`fit_batch(self, dwi_signals_normalized: torch.Tensor, learning_rate: float = 1e-2, n_iterations: int = 500, initial_params: Optional[nn.ParameterDict] = None, initial_mu_batch: Optional[torch.Tensor] = None, l1_penalty_weight: float = 0.0, l2_penalty_weight: float = 0.0, l1_params_to_regularize: Optional[List[str]] = None) -> Dict[str, torch.Tensor]`**: Fits the NODDI model to a batch of normalized DWI signals. Supports initial parameter guesses, initial mean orientations, and L1/L2 regularization.

### `models.noddi_signal`
*   **`spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor`**
    *   Description: Converts spherical coordinates (theta, phi) to Cartesian unit vectors (x, y, z).
    *   Status: `Implemented (PyTorch based)`
*   **`cartesian_to_spherical_mu(cart_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**
    *   Description: Converts Cartesian unit vectors (x,y,z) to spherical coordinates (theta, phi) for NODDI's mean orientation mu.
    *   Status: `Implemented (PyTorch based)`
*   **`compute_intra_signal(b_values: torch.Tensor, gradient_directions: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, d_intra: float) -> torch.Tensor`**
    *   Description: Computes the normalized signal from the intra-cellular compartment (sticks with Watson distribution) in the NODDI model using a Legendre polynomial expansion.
    *   Status: `Implemented (PyTorch based)`
*   **`compute_iso_signal(b_values: torch.Tensor, d_iso: float) -> torch.Tensor`**
    *   Description: Computes the normalized signal from the isotropic compartment (e.g., CSF) in NODDI using an exponential decay model.
    *   Status: `Implemented (PyTorch based)`
*   **`compute_extra_signal(b_values: torch.Tensor, gradient_directions: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, f_intra: torch.Tensor, d_intra: float) -> torch.Tensor`**
    *   Description: Computes the normalized signal from the extra-cellular compartment in NODDI, modeling anisotropic diffusion hindered by neurites (tortuosity model) and oriented by a Watson distribution. Uses a Legendre polynomial expansion.
    *   Status: `Implemented (PyTorch based)`
*   **`noddi_signal_model(params: dict, b_values: torch.Tensor, gradient_directions: torch.Tensor, d_intra_val: float = ..., d_iso_val: float = ...) -> torch.Tensor`**
    *   Description: Computes the total normalized signal for the NODDI model by combining signals from intra-cellular, extra-cellular, and isotropic compartments based on their volume fractions.
    *   Status: `Implemented (PyTorch based)`

### `models.qball`
*   **`QballModel(gtab: dipy.core.gradients.GradientTable, smooth: float = 0.006)`**
    *   Description: A wrapper class for Dipy's `QballModel` (Constant Solid Angle - CSA type). Q-ball models estimate ODFs from single-shell HARDI data.
    *   Status: `Implemented (Wraps Dipy)`
    *   **`fit(self, data: np.ndarray, sh_order_max: int = 8)`**: Fits the Q-ball model to 4D DWI data.
    *   **`odf(self, sphere: dipy.data.Sphere = None) -> np.ndarray`**: Calculates Orientation Distribution Functions (ODFs) from the fitted model.
    *   **`gfa` (property)**: Returns the 3D Generalized Fractional Anisotropy (GFA) map, calculated from the SH coefficients.

## Preprocessing

### `preprocessing.correction`
*   **`correct_motion_eddy_fsl(dwi_file: str, bval_file: str, bvec_file: str, mask_file: str, index_file: str, acqp_file: str, out_base_name: str, use_cuda: bool = False, output_type: str = "NIFTI_GZ", **kwargs: Any) -> Tuple[str, str, Optional[str], Optional[str]]`**
    *   Description: Wraps FSL's `eddy` tool using Nipype for motion and eddy current correction of DWI data. Requires FSL installation. Handles various `eddy` options via kwargs.
    *   Status: `Implemented (Relies on FSL)`
*   **`load_eddy_outlier_report(outlier_report_file: str) -> Dict[int, Dict[str, Any]]`**
    *   Description: Parses FSL eddy's outlier report text file into a structured dictionary, summarizing outlier slices per volume.
    *   Status: `Implemented`
*   **`correct_susceptibility_topup_fsl(imain_file: str, encoding_file: str, out_base_name: str, images_to_correct_file: str, images_to_correct_encoding_indices: list[int], output_type: str = "NIFTI_GZ", config_file: str = "b02b0.cnf", topup_kwargs: Optional[Dict[str, Any]] = None, applytopup_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str, str]`**
    *   Description: Performs susceptibility-induced distortion correction using FSL's `topup` and `applytopup` tools via Nipype. Requires FSL installation.
    *   Status: `Implemented (Relies on FSL)`
*   **`correct_bias_field_dipy(input_image_file: str, output_corrected_file: str, method: str = 'n4', mask_file: Optional[str] = None, **kwargs: Any) -> str`**
    *   Description: Performs bias field correction using Dipý's `BiasFieldCorrectionFlow` (e.g., N4 method). Requires Dipý and its dependencies (like ANTs/ITK for N4).
    *   Status: `Implemented (Wraps Dipy)`

### `preprocessing.denoising`
*   **`denoise_mppca_data(dmri_data: np.ndarray, patch_radius: int = 3) -> np.ndarray`**
    *   Description: Denoises 4D dMRI data using a PyTorch-based implementation of Marchenko-Pastur PCA (MP-PCA). This function wraps `preprocessing.pytorch_denoising.pytorch_mppca`.
    *   Status: `Implemented (PyTorch based)`
*   **`correct_gibbs_ringing_dipy(input_image_file: str, output_corrected_file: str, slice_axis: int = 2, n_points: int = 3, num_processes: Optional[int] = 1, **kwargs: Any) -> str`**
    *   Description: Corrects Gibbs ringing artifacts in a NIFTI image using Dipý's `gibbs_removal` function.
    *   Status: `Implemented (Wraps Dipy)`

### `preprocessing.masking`
*   **`create_brain_mask(dmri_data: np.ndarray, voxel_size, median_radius: int = 4, numpass: int = 4) -> Tuple[np.ndarray, np.ndarray]`**
    *   Description: Computes a 3D brain mask from 4D dMRI data (using mean of volumes) via median Otsu method (wrapping `preprocessing.pytorch_masking.pytorch_median_otsu`) and applies it to the dMRI data.
    *   Status: `Implemented (PyTorch based, uses SciPy)`

### `preprocessing.pytorch_denoising`
*   **`pytorch_mppca(image_4d_torch: torch.Tensor, patch_radius: int = 2, verbose: bool = False) -> torch.Tensor`**
    *   Description: Denoises a 4D dMRI image tensor using Marchenko-Pastur PCA (MP-PCA) implemented in PyTorch. It processes the image in overlapping patches, denoises each patch, and aggregates the results. The core logic involves SVD, noise estimation, and singular value thresholding based on Random Matrix Theory.
    *   Status: `Implemented (PyTorch based)`

### `preprocessing.pytorch_masking`
*   **`pytorch_otsu_threshold(image_tensor: torch.Tensor) -> int`**
    *   Description: Computes Otsu's threshold for a 2D or 3D PyTorch tensor. Handles normalization to 0-255 range internally and returns the threshold in the original image's scale.
    *   Status: `Implemented (PyTorch based)`
*   **`pytorch_median_otsu(image_data: torch.Tensor, median_radius: int = 4, numpass: int = 4) -> tuple[torch.Tensor, torch.Tensor]`**
    *   Description: Computes a brain mask from a 3D PyTorch tensor using median filtering (via SciPy) followed by Otsu's thresholding (PyTorch-based). Returns the boolean mask and the masked input tensor.
    *   Status: `Implemented (PyTorch based, uses SciPy)`

## Tracking

### `tracking.deterministic`
*   **`track_deterministic_oudf(dwi_data: np.ndarray, gtab: dipy.core.gradients.GradientTable, seeds: np.ndarray, affine: np.ndarray, metric_map_for_stopping: np.ndarray, stopping_threshold_value: float, step_size: float = 0.5, sh_order: int = 8, response: Optional[tuple] = None, model_max_peaks: int = 5, model_min_separation_angle: float = 25, model_peak_threshold: float = 0.5, max_crossing_angle: Optional[float] = 60.0, min_length: float = 10.0, max_length: float = 250.0, max_steps: int = 1000, device: Optional[str] = None) -> dipy.tracking.streamline.Streamlines`**
    *   Description: Performs deterministic tractography by wrapping a PyTorch-based pipeline (`tracking.pytorch_tracking_pipeline.track_deterministic_oudf`). This wrapper handles NumPy to PyTorch tensor conversions for main data inputs.
    *   Status: `Implemented (PyTorch based)`

### `tracking.probabilistic`
*   **`track_probabilistic_odf(odf_fit_object, seeds: np.ndarray, stopping_criterion: dipy.tracking.stopping_criterion.StoppingCriterion, sphere: dipy.data.Sphere, step_size: float = 0.5, affine: np.ndarray = np.eye(4), samples_per_voxel: int = 5, max_angle: float = 30.0, pmf_threshold: float = 0.1, min_length: float = 10, max_length: float = 300) -> dipy.tracking.streamline.Streamlines`**
    *   Description: Performs probabilistic tractography using ODFs from a fitted Dipy model object (e.g., CSD or Q-ball fit). It uses Dipy's `ProbabilisticDirectionGetter` and `LocalTracking`. Handles mask or coordinate seeds, and streamline length filtering.
    *   Status: `Implemented (Wraps Dipy)`

### `tracking.pytorch_tracking_pipeline`
*   **`PeaksFromModel(data_tensor: torch.Tensor, gradient_table: dipy.core.gradients.GradientTable, sh_order: int = 8, max_peaks: int = 5, min_separation_angle: float = 25, peak_threshold: float = 0.5, response: tuple = None, device: str ='cpu')`**
    *   Description: PyTorch implementation for extracting fiber orientation peaks from dMRI data using a CSD-like spherical harmonics model. Handles SH basis computation, ODF evaluation, and peak finding.
    *   Status: `Implemented (PyTorch based)`
    *   **`compute(self) -> tuple[torch.Tensor, torch.Tensor]`**: Computes and returns all peaks (tensor of shape X,Y,Z,N_peaks*3) and their values (tensor of shape X,Y,Z,N_peaks).
*   **`PeaksDirectionGetter(peak_data: torch.Tensor, affine: torch.Tensor, device='cpu')`**
    *   Description: A PyTorch-based direction getter for tractography that retrieves valid peak directions from precomputed peak data at a given point.
    *   Status: `Implemented (PyTorch based)`
*   **`ThresholdStoppingCriterion(metric_data: torch.Tensor, threshold: float, device='cpu')`**
    *   Description: A PyTorch-based stopping criterion for tractography that stops tracking if the metric at the current point falls below a threshold.
    *   Status: `Implemented (PyTorch based)`
*   **`LocalTracking(direction_getter: DirectionGetter, stopping_criterion: StoppingCriterion, seeds: torch.Tensor, affine: torch.Tensor, step_size: float, max_crossing_angle: Optional[float] = None, min_length: float = 0, max_length: float = float('inf'), max_steps: int = 1000, device: str ='cpu')`**
    *   Description: PyTorch-based local tracking algorithm. Generates streamlines from seeds using a given direction getter and stopping criterion. Handles step size, angle constraints, and length filtering.
    *   Status: `Implemented (PyTorch based)`
*   **`track_deterministic_oudf(dwi_data: np.ndarray, gtab: dipy.core.gradients.GradientTable, affine: np.ndarray, stopping_metric_map: torch.Tensor, stopping_threshold_value: float, seeds_input: Optional[torch.Tensor] = None, step_size: float = 0.5, sh_order: int = 8, response: Optional[tuple] = None, model_max_peaks: int = 5, model_min_separation_angle: float = 25, model_peak_threshold: float = 0.5, max_crossing_angle: Optional[float] = 60, min_length: float = 10.0, max_length: float = 250.0, max_steps: int = 1000, device: Optional[str] = None) -> dipy.tracking.streamline.Streamlines`**
    *   Description: Performs deterministic tractography using internally derived ODF peaks. This function orchestrates `PeaksFromModel` for CSD-like modeling and peak extraction, and `LocalTracking` for streamline generation, all primarily using PyTorch. It's the core PyTorch-based tracking pipeline.
    *   Status: `Implemented (PyTorch based)`

## Utils

### `utils.pytorch_gradient_utils`
*   **`PyTorchGradientTable(bvals: np.ndarray, bvecs: np.ndarray, b0_threshold: float = 50.0)`**
    *   Description: A class to store and manage gradient table information (b-values, b-vectors, b0/DWI masks, and indices) using PyTorch tensors. Provides properties to access these attributes.
    *   Status: `Implemented (PyTorch based)`

---
*Note: Some function signatures might be slightly simplified for brevity in this document. Refer to source code for full details. The `diffusemri.` prefix is omitted from module paths for conciseness.*
*Class methods/properties listed under their respective classes are typically public unless prefixed with an underscore.*
