# 03: Key Features - Parameter Maps

The `diffusemri` library enables the generation of a variety of quantitative parameter maps from different diffusion MRI models. These maps provide voxel-wise information about tissue microstructure and properties. For details on the underlying models and how to fit them, please refer to `02_Diffusion_Models.md` and the example Jupyter notebooks in the `examples/` directory (e.g., `09_DTI_Fitting_and_Metrics.ipynb`, `10_NODDI_Fitting_and_Metrics.ipynb`, etc.).

The fitting functions (e.g., `fit_dti_volume`, `fit_noddi_volume`) typically return a dictionary where keys are strings representing the metric name (e.g., "FA", "f_intra") and values are the corresponding NumPy arrays (the parameter maps).

## From Diffusion Tensor Imaging (DTI)

DTI provides well-established metrics that characterize water diffusion anisotropy and magnitude.

*   **`FA` (Fractional Anisotropy):**
    *   *Description:* A scalar value between 0 and 1 measuring the degree of anisotropic diffusion.
*   **`MD` (Mean Diffusivity):**
    *   *Description:* The average magnitude of water diffusion in a voxel.
*   **`AD` (Axial Diffusivity):**
    *   *Description:* Diffusivity along the principal direction of the diffusion tensor ($\lambda_1$).
*   **`RD` (Radial Diffusivity):**
    *   *Description:* Average diffusivity perpendicular to the principal direction ($\frac{\lambda_2 + \lambda_3}{2}$).
*   **`D_tensor_map`:**
    *   *Description:* The fitted 3x3 diffusion tensor for each voxel, usually stored as a 5D array (X, Y, Z, 3, 3).
*   **Directionally Encoded Color FA (DEC FA):**
    *   *Description:* An FA map where voxel color indicates the principal diffusion direction (Red: L-R, Green: A-P, Blue: S-I), intensity-weighted by FA. (Note: Direct generation of RGB DEC FA images might be a separate utility or require post-processing of FA and eigenvector maps).

## From Diffusion Kurtosis Imaging (DKI)

DKI extends DTI by quantifying the non-Gaussianity of water diffusion. The `diffusemri.models.DkiModel` (wrapping Dipy) provides these:

*   *(DTI metrics like FA and MD can also be derived from the diffusion tensor component of DKI via the Dipy fit object accessed through the wrapper.)*
*   **`mk` (Mean Kurtosis):**
    *   *Description:* The average degree of non-Gaussian diffusion.
*   **`ak` (Axial Kurtosis):**
    *   *Description:* Kurtosis along the principal diffusion direction.
*   **`rk` (Radial Kurtosis):**
    *   *Description:* Average kurtosis in directions perpendicular to the principal axis.
*   **`ka` (Kurtosis Anisotropy):**
    *   *Description:* Anisotropy of diffusion kurtosis.

## From Neurite Orientation Dispersion and Density Imaging (NODDI)

NODDI provides estimates of specific microstructural properties related to neurites. Output keys from `fit_noddi_volume` typically include:

*   **`f_intra` (Neurite Density Index - NDI or `vic`):**
    *   *Description:* Intra-cellular volume fraction, an estimate of neurite density.
*   **`odi` (Orientation Dispersion Index):**
    *   *Description:* Quantifies angular variation of neurite orientations (0: aligned, 1: isotropic dispersion). Derived from `kappa`.
*   **`f_iso` (Isotropic Volume Fraction or `viso`):**
    *   *Description:* Volume fraction of tissue with isotropic diffusion (e.g., free water).
*   **`mu_theta`, `mu_phi`:**
    *   *Description:* Spherical coordinates of the mean neurite orientation.
*   **`kappa`:**
    *   *Description:* Watson distribution concentration parameter, inversely related to ODI.

## From Constrained Spherical Deconvolution (CSD) & Q-Ball Imaging (QBI)

These models estimate the Orientation Distribution Function (ODF). Wrappers `diffusemri.models.CsdModel` and `diffusemri.models.QballModel` provide:

*   **ODFs (Spherical Harmonic Coefficients):**
    *   *Description:* The ODF itself, typically stored as spherical harmonic (SH) coefficients (e.g., key `shm_coeff` in the fit object).
*   **`gfa` (Generalized Fractional Anisotropy):**
    *   *Description:* A measure of ODF anisotropy. Higher GFA indicates more sharply defined ODF peaks.
*   **Peaks from ODFs:**
    *   *Description:* Dominant fiber orientations extracted from the ODFs, used for tractography. (Typically obtained via methods of the fit object or separate peak extraction functions from Dipy).

## From Multi-Tissue CSD (MT-CSD)

If using a multi-tissue CSD approach (e.g., via `diffusemri.models.CsdModel` if it supports MSMT functionality by handling multiple response functions):

*   **White Matter (WM) fODFs:** Fiber ODF specific to white matter.
*   **Grey Matter (GM) Volume Fraction Maps:** Proportion of grey matter.
*   **Cerebrospinal Fluid (CSF) Volume Fraction Maps:** Proportion of CSF.

This list provides a summary. For implementation details on how to generate these maps by fitting the respective models, please refer to `02_Diffusion_Models.md` and the runnable Jupyter notebooks in the `examples/` directory.
