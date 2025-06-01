# `diffusemri` - Diffusion MRI Analysis Toolkit

## Project Overview

The `diffusemri` project is a Python library dedicated to the processing, analysis, and management of diffusion Magnetic Resonance Imaging (dMRI) data. Its primary objective is to provide a comprehensive, user-friendly, and efficient platform for researchers and clinicians. The scope of the project includes:

*   **Data I/O and Format Conversion:** Handling various dMRI and related medical imaging formats (DICOM, NIfTI, NRRD, MHD, Analyze, PAR/REC, ISMRMRD placeholder), including conversion between them and specialized DICOM utilities.
*   **Preprocessing:** A suite of tools for preparing dMRI data for analysis, including brain masking, denoising (MP-PCA, Gibbs ringing removal), bias field correction (N4), and wrappers for advanced FSL-based corrections (Eddy, TOPUP).
*   **Model Fitting:** Implementation and wrappers for several diffusion models, including DTI, DKI, CSD, Q-Ball, and a PyTorch-based NODDI fitter.
*   **Parameter Map Generation:** Calculation of common scalar metrics from fitted models (FA, MD, AD, RD, NDI, ODI, Fiso, GFA, etc.).
*   **Tractography:** Tools for deterministic fiber tracking.
*   **Command-Line Interface (CLI):** Scripts for accessing many of the library's functionalities directly from the terminal.
*   **Examples and Documentation:** Jupyter notebooks and wiki pages to guide users.

## Table of Contents

* [Project Overview](#project-overview)
* [Installation Instructions](#installation-instructions)
* [Getting Started & Examples](#getting-started--examples)
* [Core Library Structure](#core-library-structure)
* [Key Features and Capabilities](#key-features-and-capabilities)
    * [Data I/O and Format Conversion](#data-io-and-format-conversion)
    * [Preprocessing Tools](#preprocessing-tools)
    * [Diffusion Models](#diffusion-models)
    * [Tractography](#tractography)
* [Command-Line Interface (CLI)](#command-line-interface-cli)
* [Wiki Documentation](#wiki-documentation)
* [Contribution Guidelines](#contribution-guidelines)
* [Code of Conduct](#code-of-conduct)

## Installation Instructions

### Python Version
It is recommended to use **Python 3.8 or newer**.

### Setting up a Virtual Environment

Using a virtual environment is strongly recommended to manage dependencies:
```bash
python -m venv diffusemri_env
source diffusemri_env/bin/activate  # On Windows use `diffusemri_env\Scripts\activate`
```

### Installing the Library and Dependencies

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <repository_url>
    cd diffusemri
    ```
2.  **Install in editable mode (for development) or from `requirements.txt`:**
    *   For development (recommended): This installs the package and its dependencies. Changes to the source code are immediately reflected.
        ```bash
        pip install -e .
        ```
    *   Alternatively, to install dependencies directly (e.g., for a specific environment setup before installing the package itself, or if not installing in editable mode):
        ```bash
        pip install -r requirements.txt
        # Followed by: pip install .  (if not using -e)
        ```

### Key Dependencies
The `requirements.txt` file lists all Python dependencies. Key ones include:
*   `numpy`: Core numerical operations.
*   `scipy`: Scientific and technical computing (used for .MAT I/O, optimizations).
*   `nibabel`: Reading/writing neuroimaging formats (NIfTI, Analyze, PAR/REC).
*   `pydicom`: Reading/writing DICOM files.
*   `dipy`: Major dMRI analysis library, used for many core algorithms and models.
*   `torch`: (PyTorch) For tensor computations and neural networks (used for NODDI fitter, some preprocessing). Installation might vary based on your CUDA setup if GPU support is desired.
*   `matplotlib`: Plotting and visualization.
*   `SimpleITK`: Image processing and I/O (used for MHD format).
*   `pynrrd`: Reading/writing NRRD files.
*   `h5py`: Interacting with HDF5 files.
*   `nipype`: (Optional, for FSL wrappers) Interfacing with external neuroimaging tools.
*   `ismrmrd`: (Optional, for ISMRMRD format) Reading ISMRMRD files.

*(Note: `torch` and `nipype` might be commented out in `requirements.txt` if they are treated as optional installations to be handled by the user based on their specific needs, especially concerning GPU/CUDA for PyTorch or FSL installation for Nipype.)*

### External Software Dependencies
*   **FSL (FMRIB Software Library):** Required for advanced preprocessing steps like Eddy current correction (`eddy`) and susceptibility distortion correction (`topup`). FSL must be installed separately, and its tools need to be available in your system's PATH. See the [FSL Installation Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation).

### Future PyPI Installation (Placeholder)
```bash
# pip install diffusemri
```
(This will be available if the package is published to the Python Package Index).

## Getting Started & Examples

The `diffusemri` library provides a collection of Jupyter Notebooks in the `examples/` directory to help you get started with its various functionalities. These notebooks offer practical, executable code for common tasks:

*   `00_Setup_and_Dependencies.ipynb`: Guides through environment setup and dependencies.
*   `01_DICOM_IO_and_Anonymization.ipynb`: DICOM reading, NIfTI conversion, anonymization.
*   `02_NRRD_MHD_Analyze_IO.ipynb`: Handling NRRD, MHD, and Analyze 7.5 formats.
*   `03_PARREC_Input.ipynb`: Reading Philips PAR/REC files.
*   `04_Generic_HDF5_MAT_IO.ipynb`: Saving and loading data using HDF5 and .MAT files.
*   `05_NIfTI_to_DICOM_Secondary.ipynb`: Converting NIfTI images to DICOM Secondary Capture.
*   `06_Basic_Preprocessing.ipynb`: Brain masking, MP-PCA denoising, Gibbs unringing.
*   `07_Advanced_Preprocessing_FSL.ipynb`: Conceptual guide to FSL `eddy` and `topup` wrappers.
*   `08_Bias_Field_Correction.ipynb`: N4 bias field correction using Dipý.
*   `09_DTI_Fitting_and_Metrics.ipynb`: DTI model fitting and common metrics.
*   `10_NODDI_Fitting_and_Metrics.ipynb`: NODDI model fitting (PyTorch-based).
*   `11_Other_Models_CSD_DKI_QBall.ipynb`: Using CSD, DKI, and Q-Ball model wrappers (Dipý-based).
*   `12_Deterministic_Tractography.ipynb`: Performing deterministic fiber tracking.

We recommend exploring these notebooks to understand the library's capabilities and how to use them.

A minimal working example demonstrating a typical workflow using Dipy (as `diffusemri` often wraps or uses similar patterns):
```python
import nibabel as nib
import numpy as np
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel # Example using Dipy's DTI
import matplotlib.pyplot as plt

# 1. Load dMRI data (NIfTI), b-values, and b-vectors
# Replace with your actual file paths or use I/O functions from diffusemri.data_io
# img = nib.load('path/to/your/dwi.nii.gz')
# data = img.get_fdata()
# bvals, bvecs = read_bvals_bvecs('path/to/your/bvals.bval', 'path/to/your/bvecs.bvec')
# gtab = gradient_table(bvals, bvecs)

# For a quick test, let's create dummy data:
shape = (10, 10, 10, 7) # X,Y,Z,num_volumes
data = np.random.rand(*shape).astype(np.float32) * 100
bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
bvecs = np.random.rand(7, 3)
bvecs[0,:] = 0 # b0 vector
gtab = gradient_table(bvals, bvecs)


# 2. Fit a DTI model (using Dipy's TensorModel as an example)
# diffusemri.fitting.dti_fitter.fit_dti_volume would be used for the library's implementation
tenmodel = TensorModel(gtab)
tenfit = tenmodel.fit(data) # Assuming data is masked if necessary

# 3. Calculate FA and MD maps
fa_map = tenfit.fa
md_map = tenfit.md

# 4. Visualize a parameter map (e.g., FA)
# slice_index = data.shape[2] // 2
# plt.imshow(fa_map[:, :, slice_index].T, cmap='gray', origin='lower')
# plt.colorbar()
# plt.title(f"FA Map - Slice {slice_index}")
# plt.show()
print("Minimal example executed (actual plotting commented out for brevity in README).")
```

## Core Library Structure
(This section can be expanded to describe the main modules like `data_io`, `preprocessing`, `fitting`, `models`, `tracking`, `cli`, `utils`.)

## Key Features and Capabilities

### Data I/O and Format Conversion
`diffusemri` provides extensive support for various medical imaging formats:
*   **NIfTI:** Core format for DWI data, masks, and parameter maps. Standard loading utilities are available (e.g., `data_io.load_nifti_study`).
*   **DICOM:**
    *   Conversion of DICOM series (anatomical and DWI) to NIfTI, including b-value/b-vector extraction (`data_io.dicom_utils`).
    *   DICOM anonymization for single files or entire directories, with support for default and custom anonymization rules (`data_io.dicom_utils`).
    *   Conversion of NIfTI images to DICOM Secondary Capture series, useful for integrating processed maps into PACS (`data_io.dicom_utils`).
*   **NRRD (.nrrd, .nhdr):** Reading and writing NRRD files, including NIfTI conversion and handling of DWI metadata stored in NRRD headers (`data_io.nrrd_utils`).
*   **MHD/MHA (.mhd, .raw, .zraw, .mha):** Reading and writing MetaImageHeader files using SimpleITK, with NIfTI conversion and DWI metadata support (`data_io.mhd_utils`).
*   **Analyze 7.5 (.hdr, .img):** Basic reading and writing capabilities for the Analyze 7.5 format, including NIfTI conversion. Users should be aware of Analyze's limitations regarding orientation and metadata (`data_io.analyze_utils`).
*   **Philips PAR/REC:** Reading Philips PAR/REC files (typically versions 4.0-4.2) via Nibabel, enabling conversion to NIfTI and extraction of DWI metadata (`data_io.parrec_utils`).
*   **ISMRMRD (.h5):** Placeholder support for reading ISMRMRD raw/reconstructed data. Full implementation is a future goal (`data_io.ismrmrd_utils`).
*   **Generic Formats:**
    *   **HDF5 (.h5):** Utilities to save and load Python dictionaries of NumPy arrays, suitable for structured storage of multiple datasets (`data_io.generic_utils`).
    *   **MATLAB .MAT (.mat):** Functions to save and load dictionaries for interoperability with MATLAB environments (`data_io.generic_utils`).

### Preprocessing Tools
A suite of tools to prepare dMRI data for analysis:
*   **Basic Preprocessing:**
    *   **Brain Masking:** PyTorch-based implementation using a median Otsu method (`preprocessing.masking`).
    *   **MP-PCA Denoising:** Marchenko-Pastur PCA based denoising, also leveraging PyTorch (`preprocessing.denoising`).
    *   **Gibbs Ringing Removal:** Wrapper for Dipý's Gibbs unringing algorithm (`preprocessing.denoising`).
    *   **N4 Bias Field Correction:** Wrapper for Dipý's N4ITK bias field correction (`preprocessing.correction`).
*   **Advanced FSL-based Correction (requires FSL installation and Nipype):**
    *   **Susceptibility Distortion Correction:** Wrapper for FSL's `topup` and `applytopup` tools (`preprocessing.correction`).
    *   **Eddy Current and Motion Correction:** Wrapper for FSL's `eddy` tool (`preprocessing.correction`).

### Diffusion Models

Implementation and wrappers for various diffusion models:
*   **DTI (Diffusion Tensor Imaging):** Standard tensor fitting with FA, MD, AD, RD calculation (`fitting.dti_fitter`, `models.dti`).
*   **DKI (Diffusion Kurtosis Imaging):** Wrapper for Dipý's DKI model, providing kurtosis metrics (`models.dki`). Requires multi-shell data.
*   **CSD (Constrained Spherical Deconvolution):** Wrapper for Dipý's CSD for fODF estimation (`models.csd`). Suitable for HARDI data.
*   **Q-Ball Imaging (QBI):** Wrapper for Dipý's Q-Ball model (CSA type) for ODF reconstruction (`models.qball`). Suitable for HARDI data.
*   **MAP-MRI (Mean Apparent Propagator MRI):** Wrapper for Dipý's MAPMRI model (`models.mapmri`). Requires multi-shell, high b-value data.
*   **NODDI (Neurite Orientation Dispersion and Density Imaging):** PyTorch-based implementation for efficient fitting of NDI, ODI, and Fiso (`fitting.noddi_fitter`, `models.noddi_model`, `models.noddi_signal`). Supports advanced features like smart initialization and regularization.

Derived parameter maps include FA, MD, AD, RD (from DTI/DKI tensor), MK, AK, RK (from DKI), GFA, ODFs (from CSD, QBI, MAP-MRI), NDI, ODI, Fiso (from NODDI), and more.

### Tractography

*   **Deterministic Tractography:** Using ODF peaks (e.g., from CSD results) to reconstruct fiber pathways (`tracking.deterministic.track_deterministic_oudf`). This function internally uses a CSD model to estimate ODFs.
*   **Probabilistic Tractography:** (If implemented or planned) Wrapper for Dipy's probabilistic methods to sample from orientation distributions.

## Command-Line Interface (CLI)
`diffusemri` offers several command-line scripts for key functionalities, enabling batch processing and integration into larger pipelines:
*   **`run_preprocessing.py`:** Access various preprocessing tools (masking, denoising, FSL wrappers, bias field correction, DICOM conversion/anonymization).
*   **`run_format_conversion.py`:** Convert between NIfTI, NRRD, MHD, Analyze, PAR/REC, and NIfTI to DICOM SC.
*   **`run_dti_fit.py`:** Perform DTI model fitting and save scalar maps.
*   **`run_noddi_fit.py`:** Perform NODDI model fitting and save parameter maps.
*   **`run_tracking.py`:** Perform tractography (e.g., deterministic ODF-based).

Refer to the respective `--help` option for each script and the [Wiki Documentation](#wiki-documentation) for detailed usage.

## Wiki Documentation
For more in-depth information, tutorials, and advanced usage, please refer to the [project Wiki](../../wiki) (assuming this relative link works, or use full URL if known):
*   [01 - Preprocessing Tools](wiki/01_Preprocessing_Tools.md)
*   [02 - Diffusion Models](wiki/02_Diffusion_Models.md)
*   [04 - Tractography](wiki/04_Tractography.md)
*   [07 - Format Conversion](wiki/07_Format_Conversion.md)
*   [08 - Generic Data I/O (HDF5, MAT)](wiki/08_Generic_Data_IO_HDF5_MAT.md)
*   ... and other pages covering specific features and examples.

(Note: `Validation and Benchmarking`, `Contribution Guidelines`, `Coding Standards`, and `Code of Conduct` sections from the original README would typically follow here. For brevity in this diff, assuming they remain largely unchanged unless specified otherwise.)

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