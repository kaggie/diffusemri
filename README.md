# diffusemri

## Project Overview

The `diffusemri` project is a Python library dedicated to the analysis of diffusion Magnetic Resonance Imaging (dMRI) data. Its primary objective is to provide a user-friendly and efficient platform for researchers and clinicians to perform common dMRI processing and analysis tasks. The scope of the project includes data loading, model fitting, parameter map generation, and visualization.

## Table of Contents

* [Project Overview](#project-overview)
* [Installation Instructions](#installation-instructions)
* [Getting Started / Usage Examples](#getting-started--usage-examples)
* [Loading Your Data](#loading-your-data)
* [Core Data Structures](#core-data-structures)
* [Data Requirements](#data-requirements)
* [Supported Models and Features](#supported-models-and-features)
    * [Current Models](#current-models)
    * [Key Features](#key-features)
    * [Preprocessing Tools](#preprocessing-tools)
* [Tractography](#tractography)
    * [Deterministic Tractography](#deterministic-tractography)
    * [Probabilistic Tractography](#probabilistic-tractography)
* [Contribution Guidelines](#contribution-guidelines)
* [Code of Conduct](#code-of-conduct)

## Installation Instructions

### Setting up the environment

It is recommended to use a virtual environment to manage dependencies.

```bash
python -m venv diffusemri_env
source diffusemri_env/bin/activate  # On Windows use `diffusemri_env\Scripts\activate`
```

### Dependencies

The following packages are required:

*   `nibabel`
*   `numpy`
*   `scipy`
*   `matplotlib`
*   `dipy`

### Installation

You can install `diffusemri` and its dependencies using pip:

```bash
pip install diffusemri
```

(Note: This assumes the package will be published to PyPI. For local development, you might use `pip install .` from the project root.)

### System Requirements

There are no special system requirements beyond a standard Python installation and the ability to install the listed dependencies.

## Getting Started / Usage Examples

Here's a minimal working example demonstrating a typical workflow:

```python
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
import matplotlib.pyplot as plt

# 1. Load dMRI data, b-values, and b-vectors
img = nib.load('path/to/your/dwi.nii.gz')
data = img.get_fdata()
bvals, bvecs = read_bvals_bvecs('path/to/your/bvals', 'path/to/your/bvecs')
gtab = gradient_table(bvals, bvecs)

# 2. Fit a DTI model
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data)

# 3. Calculate FA and MD maps
fa_map = tenfit.fa
md_map = tenfit.md

# 4. Visualize a parameter map (e.g., FA)
slice_index = data.shape[2] // 2  # Example: middle axial slice
plt.imshow(fa_map[:, :, slice_index].T, cmap='gray', origin='lower')
plt.colorbar()
plt.title(f"FA Map - Slice {slice_index}")
plt.show()
```

## Loading Your Data

The primary way to load your NIfTI-based dMRI dataset into `diffusemri` is by using the `load_nifti_study` function from the `diffusemri.data_io` module. This function handles loading of the DWI data, b-values, b-vectors, and an optional brain mask.

```python
from diffusemri.data_io import load_nifti_study

dwi_file = "path/to/your/dwi.nii.gz"
bval_file = "path/to/your/bvals.bval" # FSL format text file
bvec_file = "path/to/your/bvecs.bvec" # FSL format text file
mask_file = "path/to/your/mask.nii.gz" # Optional NIfTI mask file

# Load the study data
dwi_data, gtab, mask_data, affine = load_nifti_study(
    dwi_path=dwi_file,
    bval_path=bval_file,
    bvec_path=bvec_file,
    mask_path=mask_file, # Omit this argument or set its value to None if no mask is used
    b0_threshold=50.0   # Optional: b-value threshold to identify b0s for gtab creation
)

# Returned variables:
# dwi_data: 4D NumPy array (float32) - The diffusion-weighted image data.
# gtab: Dipy GradientTable object - Contains b-values, b-vectors, and b0 information.
# mask_data: 3D NumPy array (bool), or None - The brain mask if provided, otherwise None.
# affine: 4x4 NumPy array - The affine transformation matrix from the DWI NIfTI header.
```

This function provides a convenient way to get all necessary data components ready for analysis.

For users who need more granular control over data loading (e.g., if data is in a non-standard arrangement or if individual components need to be loaded separately), the `diffusemri.data_io` module also provides lower-level functions such as:
*   `load_nifti_dwi`: For loading just the DWI NIfTI image and its affine.
*   `load_nifti_mask`: For loading just a NIfTI mask and its affine.
*   `load_fsl_bvals`: For loading b-values from an FSL-formatted file.
*   `load_fsl_bvecs`: For loading b-vectors from an FSL-formatted file.
*   `create_gradient_table`: For creating a Dipy `GradientTable` from b-values and b-vectors arrays.

For detailed information on the in-memory representation of these data components (like `dwi_data`, `gtab`, `affine`, etc.), please refer to the [Core Data Structures](#core-data-structures) section.

## Core Data Structures

The `diffusemri` library primarily operates on a set of core data structures, many of which are standard in the diffusion MRI field. Understanding these is key to effectively using the library:

*   **Diffusion-Weighted Image (DWI) Data:**
    *   **Description:** A 4D NumPy array (`numpy.ndarray`) containing the diffusion-weighted signal.
    *   **Expected Type:** `numpy.ndarray`
    *   **Typical Shape/Dimensions:** `(X, Y, Z, N_volumes)`, where `X, Y, Z` are the spatial dimensions of the image and `N_volumes` is the number of acquired diffusion-weighted volumes (including b0 images).
    *   **Data Type:** Floating-point, typically `float32` or `float64`.

*   **b-values:**
    *   **Description:** A 1D NumPy array representing the b-value (strength of diffusion sensitization) for each volume in the DWI data.
    *   **Expected Type:** `numpy.ndarray`
    *   **Typical Shape/Dimensions:** `(N_volumes,)`
    *   **Data Type:** Numeric (integer or float).

*   **b-vectors:**
    *   **Description:** A 2D NumPy array representing the gradient direction for each volume in the DWI data. For volumes with b=0 (b0 images), the corresponding b-vector should ideally be `[0, 0, 0]`.
    *   **Expected Type:** `numpy.ndarray`
    *   **Typical Shape/Dimensions:** `(N_volumes, 3)`
    *   **Data Type:** Floating-point.

*   **Brain Mask (Optional):**
    *   **Description:** A 3D NumPy array spatially aligned with the DWI data, used to delineate the region of interest (typically the brain) for analysis.
    *   **Expected Type:** `numpy.ndarray`
    *   **Typical Shape/Dimensions:** `(X, Y, Z)`
    *   **Data Type:** Boolean (`bool`) or integer (`int`, typically 0 for background and 1 for the mask region).

*   **Affine Transformation Matrix:**
    *   **Description:** A 4x4 NumPy array representing the affine transformation that maps voxel coordinates of the DWI data to a world or scanner reference space. This is typically obtained from NIfTI image headers and is crucial for correct spatial interpretation and orientation.
    *   **Expected Type:** `numpy.ndarray`
    *   **Typical Shape/Dimensions:** `(4, 4)`
    *   **Data Type:** Floating-point.

*   **Gradient Table (`gtab`):**
    *   **Description:** A `dipy.core.gradients.GradientTable` object. This object encapsulates b-values, b-vectors, information about b0 volumes, and potentially other gradient-related parameters (like pulse duration or separation, though not always used by all models).
    *   **Expected Type:** `dipy.core.gradients.GradientTable`
    *   **Creation:** Typically created using `dipy.core.gradients.gradient_table(bvals, bvecs, ...)`.
    *   **Usage:** Most `diffusemri` models and functions requiring gradient information expect this object as input, as it provides a standardized way to access and validate gradient information.

## Data Requirements

### Expected Input Data Formats

*   Diffusion-weighted images should be in NIfTI format (`.nii` or `.nii.gz`).

### Necessary Input Files

*   **Diffusion-weighted images (DWIs)**: A 4D NIfTI file where the last dimension corresponds to the different gradient acquisitions.
*   **b-values**: A text file (typically FSL format, e.g., `.bval`) listing the b-value for each volume in the DWI series.
*   **b-vectors**: A text file (typically FSL format, e.g., `.bvec`) where each row (or column, depending on formatting convention) specifies the gradient direction (x, y, z) for each volume.

### Preprocessing

Users are expected to perform essential preprocessing steps before using this library. These may include:

*   **Motion correction**: To correct for subject movement during the scan.
*   **Eddy current correction**: To correct for distortions caused by eddy currents.
*   **Brain extraction/masking**: To isolate the brain tissue from the skull and other non-brain tissues.

Tools like **FSL** (`topup`, `eddy`, `bet`) or **Dipy** offer functionalities for these preprocessing steps.

## Supported Models and Features

### Current Models

*   **Diffusion Tensor Imaging (DTI)**: A widely used model to characterize anisotropic water diffusion by fitting a tensor to the diffusion signal. It provides parameters like FA and MD.
*   **Diffusion Kurtosis Imaging (DKI)**: Extends DTI to characterize non-Gaussian water diffusion, offering more detailed insights into tissue microstructure. It requires multi-shell acquisition schemes (at least two non-zero b-values). The implementation (`diffusemri.models.DkiModel`) leverages Dipy for its core calculations.
*   **Constrained Spherical Deconvolution (CSD)**: An advanced model capable of resolving crossing fiber populations within a voxel by estimating Orientation Distribution Functions (ODFs). Its main output is the ODF, which describes the angular profile of diffusion. The implementation (`diffusemri.models.CsdModel`) leverages Dipy's CSD functionalities and response function estimation.
*   **Q-Ball Imaging (QBI)**: Another method for reconstructing Orientation Distribution Functions (ODFs) to characterize fiber orientations, often suitable for single-shell diffusion data. Like CSD, its main output is the ODF, providing an angular profile of diffusion. The implementation (`diffusemri.models.QballModel`) leverages Dipy's Q-Ball (specifically Constant Solid Angle - CSA type) functionalities.

*   **Mean Apparent Propagator MRI (MAP-MRI)**: An advanced diffusion model that provides a framework to probe the three-dimensional q-space information and can characterize complex microstructural environments beyond the capabilities of DTI or DKI. It can estimate various scalar indices such as Return to Origin/Axis/Plane Probability (RTOP/RTAP/RTPP), Mean Squared Displacement (MSD), Q-space Inverse Variance (QIV), and Non-Gaussianity (NG). It can also be used to reconstruct Orientation Distribution Functions (ODFs). The implementation (`diffusemri.models.MapmriModel`) leverages Dipy's `MapmriModel`.
*   **Neurite Orientation Dispersion and Density Imaging (NODDI)**: A multi-compartment model that estimates parameters like Neurite Density Index (NDI), Orientation Dispersion Index (ODI), and isotropic volume fraction (Fiso).
    *   The `diffusemri` library includes a custom implementation of the NODDI model based on the original formulation by Zhang et al. (2012).
    *   Fitting is performed using PyTorch for efficiency, supporting batch-wise processing of voxels and GPU acceleration. The primary fitting function is `fitting.noddi_fitter.fit_noddi_volume`.
    *   **Advanced Features for NODDI fitting:**
        *   **Smart Initialization:** The mean neurite orientation (`mu`) can be initialized using an `initial_orientation_map` (e.g., derived from DTI primary eigenvectors), potentially improving convergence and accuracy. This is configurable in `fit_noddi_volume`.
        *   **Regularization:** L1 and L2 regularization options are available during the fitting process. L1 can be applied to specific constrained parameters (e.g., `f_iso` for promoting sparsity), and L2 is applied to unconstrained parameters for weight decay. These are configurable via the `fit_params` argument in `fit_noddi_volume`.


### Key Features

*   **Parameter Map Generation**:
    *   From DTI:
        *   Fractional Anisotropy (FA)
        *   Mean Diffusivity (MD)
        *   Axial Diffusivity (AD)
        *   Radial Diffusivity (RD)
        *   RGB DEC FA maps (Red-Green-Blue Directionally Encoded Color FA)
    *   From DKI (in addition to FA and MD, which can also be derived from its tensor component):
        *   Mean Kurtosis (MK)
        *   Axial Kurtosis (AK)
        *   Radial Kurtosis (RK)
        *   Kurtosis Anisotropy (KA)

    *   From CSD, QBI, & MAP-MRI:
        *   Orientation Distribution Functions (ODFs)
        *   Generalized Fractional Anisotropy (GFA) - *Note: GFA from MAP-MRI is calculated differently, e.g., using RTAP and QIV.*
    *   From MAP-MRI:
        *   Return to Origin Probability (RTOP)
        *   Return to Axis Probability (RTAP)
        *   Return to Plane Probability (RTPP)
        *   Mean Squared Displacement (MSD)
        *   Q-space Inverse Variance (QIV)
        *   Non-Gaussianity (NG)

    *   From NODDI:
        *   Neurite Density Index (NDI, also f_intra or vic)
        *   Orientation Dispersion Index (ODI)
        *   Isotropic Volume Fraction (Fiso, also f_iso or viso)
        *   Mean orientation of neurites (mu_theta, mu_phi)
        *   Watson model concentration parameter (kappa)
    *   From CSD & QBI:
        *   Orientation Distribution Functions (ODFs)
    *   From Multi-Tissue CSD (MT-CSD):
        *   White Matter (WM) fODFs
        *   Grey Matter (GM) Volume Fraction Maps
        *   Cerebrospinal Fluid (CSF) Volume Fraction Maps
*   **Model Fitting Enhancements**:
    *   Smart initialization options for certain model parameters (e.g., NODDI `mu` orientation).
    *   Regularization options (L1/L2) for improved stability in model fitting (e.g., for NODDI).

*   **Basic Visualization**:
    *   Displaying 2D slices of parameter maps.
    *   GUI allows selection of different output maps from MT-CSD for visualization.

### Preprocessing Tools

The library also includes tools to help prepare your dMRI data for analysis:

*   **Brain Masking**:
    *   **Purpose**: To isolate brain tissue from non-brain areas in dMRI images, which is crucial for accurate downstream analysis.
    *   **Function**: `diffusemri.preprocessing.create_brain_mask()`
    *   **Method**: Implements a brain masking algorithm based on Dipy's `median_otsu` method, which is effective for DWI data.
*   **Noise Reduction**:
    *   **Purpose**: To improve data quality by reducing random noise typically present in dMRI acquisitions, leading to more reliable model fitting and parameter estimation.
    *   **Function**: `diffusemri.preprocessing.denoise_mppca_data()`
    *   **Method**: Implements the Marchenko-Pastur Principal Component Analysis (MP-PCA) algorithm for noise reduction, utilizing Dipy's `localpca` functionality.
*   **Motion and Eddy Current Correction**:
    *   **Purpose**: To correct for subject motion, eddy current distortions, and related susceptibility artifacts in DWI data. This step is critical for improving data quality and the accuracy of subsequently fitted diffusion models. It also ensures that b-vectors are correctly rotated to align with the corrected image data.
    *   **Function**: `diffusemri.preprocessing.correction.correct_motion_eddy_fsl()`
    *   **Method**: This function serves as a wrapper for FSL's powerful `eddy` tool (typically `eddy_openmp` or `eddy_cuda`). It uses Nipype to interface with the FSL command-line tool. `eddy` can model and correct complex distortions, handle multi-shell data, and detect/replace outlier slices (e.g., by passing `repol=True` in `**kwargs`). The wrapper returns paths to the corrected DWI image, the rotated b-vectors, and optionally outlier reports.
    *   **Dependency**: This feature requires a local installation of FSL (with `eddy` available in the system PATH) and the `nipype` library. Users must ensure compliance with FSL's licensing terms.
    *   **Outlier Information**: A helper function `diffusemri.preprocessing.correction.load_eddy_outlier_report()` is available to parse the textual outlier report produced by `eddy`.
    *   **Susceptibility Distortion Correction (FSL TOPUP)**:
        *   **Purpose**: Corrects for geometric distortions caused by susceptibility-induced field inhomogeneities, typically using images with opposing phase-encoding directions (e.g., AP-PA b0s).
        *   **Function**: `diffusemri.preprocessing.correction.correct_susceptibility_topup_fsl()`
        *   **Method**: Wraps FSL's `topup` (for field estimation) and `applytopup` (to apply correction) tools via Nipype.
        *   **Dependency**: Requires FSL and Nipype.
    *   **Bias Field Correction (Dipý N4)**:
        *   **Purpose**: Corrects for low-frequency intensity non-uniformities (bias field) in MR images.
        *   **Function**: `diffusemri.preprocessing.correction.correct_bias_field_dipy()`
        *   **Method**: Wraps Dipý's `BiasFieldCorrectionFlow` using the 'n4' method (N4ITK algorithm).
        *   **Dependency**: Requires Dipý and its N4 dependencies (e.g., ANTs/ITK).
    *   **Gibbs Ringing Correction (Dipý)**:
        *   **Purpose**: Reduces Gibbs ringing artifacts, which are spurious oscillations near sharp intensity transitions.
        *   **Function**: `diffusemri.preprocessing.denoising.correct_gibbs_ringing_dipy()`
        *   **Method**: Wraps Dipý's `dipy.denoise.gibbs_removal` function.
        *   **Dependency**: Requires Dipý.

### DICOM Utilities
The library provides tools for handling DICOM data:
*   **DICOM to NIfTI Conversion**:
    *   Convert DICOM series to NIfTI format.
    *   For DWI data, automatically extracts and saves b-values and b-vectors.
    *   Relevant functions: `diffusemri.data_io.dicom_utils.convert_dwi_dicom_to_nifti()`, `diffusemri.data_io.dicom_utils.convert_dicom_to_nifti_main()`.
*   **DICOM Anonymization**:
    *   Remove or modify patient-identifying information from DICOM tags.
    *   Supports default and custom anonymization rules.
    *   Relevant functions: `diffusemri.data_io.dicom_utils.anonymize_dicom_directory()`, `diffusemri.data_io.dicom_utils.anonymize_dicom_file()`.
*   **NRRD Format Conversion**:
    *   Conversion between NIfTI and NRRD formats.
    *   Supports DWI data (b-values/b-vectors) during conversion.
    *   Relevant functions: `diffusemri.data_io.nrrd_utils.read_nrrd_data()`, `diffusemri.data_io.nrrd_utils.write_nrrd_data()`.
*   **MHD/MHA Format Conversion**:
    *   Conversion between NIfTI and MHD/MHA formats using SimpleITK.
    *   Supports DWI data (b-values/b-vectors) during conversion.
    *   Relevant functions: `diffusemri.data_io.mhd_utils.read_mhd_data()`, `diffusemri.data_io.mhd_utils.write_mhd_data()`.
*   **Analyze 7.5 Format Conversion**:
    *   Basic conversion between NIfTI and Analyze 7.5 formats (`.hdr`/`.img`).
    *   Note: Analyze 7.5 has limited support for orientation and no standard DWI metadata.
    *   Relevant functions: `diffusemri.data_io.analyze_utils.read_analyze_data()`, `diffusemri.data_io.analyze_utils.write_analyze_data()`.
*   **Philips PAR/REC Format Conversion**:
    *   Conversion of Philips PAR/REC files to NIfTI, including DWI metadata (b-values/b-vectors) extraction.
    *   Relevant functions: `diffusemri.data_io.parrec_utils.read_parrec_data()`, `diffusemri.data_io.parrec_utils.convert_parrec_to_nifti()`.
*   **ISMRMRD Format Support (Placeholder)**:
    *   Basic placeholder functions for reading ISMRMRD `.h5` files are included. Full implementation for image and k-space data extraction is planned for future development.
    *   Relevant functions: `diffusemri.data_io.ismrmrd_utils.read_ismrmrd_file()`.
*   **NIfTI to DICOM Secondary Capture Conversion**:
    *   Converts 3D or 4D NIfTI files into a series of DICOM Secondary Capture files.
    *   Useful for integrating NIfTI-based images into PACS systems.
    *   Relevant function: `diffusemri.data_io.dicom_utils.write_nifti_to_dicom_secondary()`.
*   **Generic HDF5 and .MAT I/O**:
    *   Utilities to save and load dictionaries of NumPy arrays to/from HDF5 (`.h5`) and MATLAB (`.mat`) files.
    *   Useful for storing intermediate results or collections of processed data.
    *   Relevant functions: `diffusemri.data_io.generic_utils.save_dict_to_hdf5()`, `load_dict_from_hdf5()`, `save_dict_to_mat()`, `load_dict_from_mat()`.
*   **Format Conversion CLI**:
    *   The `run_format_conversion` CLI tool provides subcommands for `nrrd2nii`, `nii2nrrd`, `mhd2nii`, `nii2mhd`, `analyze2nii`, `nii2analyze`, `parrec2nii`, `nii2dicom_sec`, and a placeholder for `ismrmrd_convert`.

For more details on these preprocessing tools and various data format utilities (DICOM, NRRD, MHD, Analyze, PAR/REC, ISMRMRD, HDF5, MAT), including their command-line interfaces, please refer to the [Preprocessing Tools Wiki](wiki/01_Preprocessing_Tools.md), the [Format Conversion Wiki](wiki/07_Format_Conversion.md), and the [Generic Data I/O Wiki](wiki/08_Generic_Data_IO_HDF5_MAT.md).

## Tractography

This section details the functionalities available for reconstructing white matter pathways (fiber tracking).

### Deterministic Tractography

Deterministic tractography algorithms reconstruct streamlines by iteratively following the most probable orientation from a local diffusion model at each step.

*   **Purpose**: To reconstruct white matter pathways by iteratively following orientation information (peaks) derived from Orientation Distribution Function (ODF) models like CSD or Q-Ball.
*   **Key Function**: `diffusemri.tracking.track_deterministic_oudf`
*   **Main Inputs**:
    *   `odf_fit_object`: A fitted ODF model object from Dipy (e.g., from `CsdModel` or `QballModel` in this library) that provides ODF information.
    *   `seeds`: Seed points for initiating tracking, which can be provided as either a 3D binary mask or an N x 3 array of coordinates in world/scanner space.
    *   `stopping_criterion`: A Dipy `StoppingCriterion` object (e.g., `BinaryStoppingCriterion` based on a mask, or `ThresholdStoppingCriterion` based on a scalar map like GFA). This determines when tracking should terminate.
    *   `sphere`: A Dipy `Sphere` object (e.g., `get_sphere('repulsion724')`) used for ODF peak extraction.
    *   Additional parameters: `step_size`, `affine` transformation, ODF peak extraction parameters (`relative_peak_threshold`, `min_separation_angle`), and streamline length filters (`min_length`, `max_length`).
*   **Output**: A Dipy `Streamlines` object, which is a list-like container of the generated tracts.
*   **Methodology**: The function leverages Dipy's `peaks_from_model` for ODF peak extraction and `LocalTracking` algorithm for streamline generation.


### Probabilistic Tractography

Probabilistic tractography algorithms reconstruct streamlines by sampling from orientation distributions (e.g., ODFs) at each step. This approach can better represent uncertainty in fiber directions, especially in complex regions with crossing, fanning, or kissing fibers.

*   **Purpose**: To reconstruct white matter pathways by sampling from orientation distributions (ODFs), which can better represent uncertainty in fiber directions, especially in complex regions.
*   **Key Function**: `diffusemri.tracking.track_probabilistic_odf`
*   **Main Inputs**:
    *   `odf_fit_object`: A fitted ODF model object (e.g., from `CsdModel` or `QballModel` in this library) that provides spherical harmonic coefficients (`shm_coeff`) of the ODF.
    *   `seeds`: Seed points for initiating tracking, which can be provided as either a 3D binary mask or an N x 3 array of coordinates in world/scanner space.
    *   `stopping_criterion`: A Dipy `StoppingCriterion` object (e.g., `BinaryStoppingCriterion` based on a mask, or `ThresholdStoppingCriterion` based on a scalar map like GFA). This determines when tracking should terminate.
    *   `sphere`: A Dipy `Sphere` object (e.g., `get_sphere('repulsion724')`) used by the probabilistic direction getter.
    *   Additional parameters: `samples_per_voxel` (number of seeds per voxel/coordinate), `step_size`, `affine` transformation, probabilistic tracking parameters (`max_angle`, `pmf_threshold`), and streamline length filters (`min_length`, `max_length`).
*   **Output**: A Dipy `Streamlines` object, which is a list-like container of the generated tracts.
*   **Methodology**: The function leverages Dipy's `ProbabilisticDirectionGetter` (specifically from spherical harmonic coefficients) and `LocalTracking` algorithm for streamline generation.

## Validation and Benchmarking

Ensuring the accuracy and reliability of dMRI models is crucial. `diffusemri` provides tools and encourages users to validate its implementations.
*   **Synthetic Data Tests:** The test suite (`tests/`) includes various tests using synthetic data to verify model fitting accuracy against known ground truths for models like DTI and NODDI. These tests cover a range of parameter values and noise conditions.
*   **Cross-Model Consistency:** Where applicable, tests aim to ensure consistency between different models (e.g., orientation estimates from the custom NODDI model are compared against DTI in simple fiber configurations).
*   **Benchmarking Guidance:** For users interested in comparing `diffusemri`'s custom model implementations (like NODDI) against other software packages or more extensive ground truth datasets, a detailed guide is available in [`docs/benchmarking_noddi.md`](docs/benchmarking_noddi.md). This document outlines strategies for synthetic and real data comparisons and highlights important considerations for ensuring fair and informative benchmarking.


## Contribution Guidelines

We welcome contributions to `diffusemri`! If you'd like to contribute, please follow these steps:

1.  **Fork the repository** on GitHub.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b my-new-feature`.
3.  **Make your changes** and ensure they adhere to existing coding styles.
4.  **Add tests** for your new feature or bug fix. Tests are located in the `tests/` directory. You can run tests using a test runner like `pytest`.
5.  **Ensure your code passes all tests.**
6.  **Commit your changes**: `git commit -am 'Add some feature'`.
7.  **Push to the branch**: `git push origin my-new-feature`.
8.  **Submit a Pull Request (PR)** to the `main` (or `develop`) branch of the original repository.

### Coding Standards

Please follow the **PEP 8** style guide for Python code.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for everyone. All contributors and users are expected to adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please report any unacceptable behavior.