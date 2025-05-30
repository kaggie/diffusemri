# 03: Key Features - Parameter Maps

The `diffusemri` library enables the generation of a variety of quantitative parameter maps from different diffusion MRI models. These maps provide voxel-wise information about tissue microstructure and properties. For details on the underlying models and how to fit them, please refer to `02_Diffusion_Models.md`.

## From Diffusion Tensor Imaging (DTI)

DTI provides well-established metrics that characterize water diffusion anisotropy and magnitude. These are typically derived from the diffusion tensor fitted in each voxel.

*   **Fractional Anisotropy (FA):**
    *   *Description:* A scalar value between 0 and 1 that measures the degree of anisotropic diffusion. Higher FA indicates more directional coherence (e.g., in white matter tracts), while lower FA suggests more isotropic diffusion (e.g., in gray matter or CSF).
*   **Mean Diffusivity (MD):**
    *   *Description:* The average magnitude of water diffusion in a voxel, irrespective of direction. It reflects the overall diffusivity within the tissue.
*   **Axial Diffusivity (AD):**
    *   *Description:* The magnitude of diffusion along the principal direction of the diffusion tensor (parallel to the primary fiber orientation).
*   **Radial Diffusivity (RD):**
    *   *Description:* The average magnitude of diffusion perpendicular to the principal direction of the diffusion tensor.
*   **RGB DEC FA maps (Red-Green-Blue Directionally Encoded Color FA):**
    *   *Description:* An FA map where the color of each voxel is determined by the principal diffusion direction, and the intensity by the FA value. Typically:
        *   Red: Left-Right orientation
        *   Green: Anterior-Posterior orientation
        *   Blue: Superior-Inferior orientation

## From Diffusion Kurtosis Imaging (DKI)

DKI extends DTI by quantifying the non-Gaussianity of water diffusion, providing additional microstructural details. DKI maps are typically derived from the kurtosis tensor, in addition to the diffusion tensor.

*   *(DTI metrics like FA and MD can also be derived from the diffusion tensor component of DKI)*
*   **Mean Kurtosis (MK):**
    *   *Description:* The average degree of non-Gaussian diffusion across all directions. It is sensitive to tissue complexity and restrictions.
*   **Axial Kurtosis (AK):**
    *   *Description:* The kurtosis along the principal diffusion direction.
*   **Radial Kurtosis (RK):**
    *   *Description:* The average kurtosis in directions perpendicular to the principal diffusion direction.
*   **Kurtosis Anisotropy (KA):**
    *   *Description:* A measure of the anisotropy of diffusion kurtosis, analogous to FA for diffusivity.

## From Neurite Orientation Dispersion and Density Imaging (NODDI)

NODDI is a multi-compartment model that provides estimates of specific microstructural properties related to neurites.

*   **Neurite Density Index (NDI):**
    *   *Description:* Also referred to as `f_intra` or `vic`. Represents the intra-cellular volume fraction, interpreted as an estimate of neurite density.
*   **Orientation Dispersion Index (ODI):**
    *   *Description:* Quantifies the angular variation or dispersion of neurite orientations around a mean direction. ODI ranges from 0 (perfectly aligned) to 1 (isotropically dispersed).
*   **Isotropic Volume Fraction (Fiso):**
    *   *Description:* Also `f_iso` or `viso`. Represents the volume fraction of tissue with isotropic diffusion, often interpreted as free water or CSF contamination.
*   **Mean Orientation of Neurites (`mu_theta`, `mu_phi`):**
    *   *Description:* The primary orientation of neurites within a voxel, represented in spherical coordinates (polar angle `mu_theta`, azimuthal angle `mu_phi`).
*   **Watson Model Concentration Parameter (`kappa`):**
    *   *Description:* A parameter from the Watson distribution used to model neurite orientation dispersion. It is inversely related to ODI; higher `kappa` means lower dispersion (more coherence).

## From Constrained Spherical Deconvolution (CSD) & Q-Ball Imaging (QBI)

CSD and QBI are used to estimate the Orientation Distribution Function (ODF), which describes the angular profile of diffusion, particularly useful for resolving crossing fibers.

*   **Orientation Distribution Functions (ODFs):**
    *   *Description:* A function on a sphere representing the probability of water diffusing in any given direction. ODFs can have multiple peaks in voxels with crossing fibers. They are often represented by spherical harmonic coefficients.
*   *(Scalar metrics like GFA and peaks for tractography are also derived from ODFs, see `02_Diffusion_Models.md` and `04_Tractography.md`)*

## From Multi-Tissue CSD (MT-CSD)

Multi-Tissue CSD extends CSD by modeling contributions from different tissue types (e.g., White Matter, Grey Matter, CSF).

*   **White Matter (WM) fODFs:**
    *   *Description:* The fiber ODF specific to white matter, resolving crossing fibers within WM.
*   **Grey Matter (GM) Volume Fraction Maps:**
    *   *Description:* A map indicating the proportion of grey matter within each voxel.
*   **Cerebrospinal Fluid (CSF) Volume Fraction Maps:**
    *   *Description:* A map indicating the proportion of CSF within each voxel.

This list provides a summary. For implementation details on how to generate these maps by fitting the respective models, please refer to the model-specific documentation in `02_Diffusion_Models.md`.
```
