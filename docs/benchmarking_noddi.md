# Benchmarking the Custom NODDI Implementation

This document outlines strategies for users to benchmark and validate the custom PyTorch-based NODDI model implemented in the `diffusemri` library against other existing NODDI implementations or known ground truths.

## Introduction

The NODDI (Neurite Orientation Dispersion and Density Imaging) model (Zhang et al., 2012) has several available implementations (e.g., the original MATLAB toolbox, AMICO, implementations in other libraries). It's important for users to be able to assess the performance and consistency of the `diffusemri` implementation.

Benchmarking can be approached using synthetic data with known ground truth parameters and by comparing results on real DWI datasets with those from other established toolboxes.

## 1. Synthetic Data Comparison

Using synthetic data allows for quantitative comparison against a known ground truth.

### a. Generating Synthetic DWI Data
1.  **Define Ground Truth Parameters:** Create 3D maps (or batches of voxels) for NODDI parameters:
    *   Neurite Density Index (NDI, or `f_intra`): e.g., values ranging from 0.1 to 0.9.
    *   Orientation Dispersion Index (ODI): e.g., values ranging from 0.05 (highly coherent) to 0.9 (highly dispersed). This will correspond to a `kappa` value for the Watson distribution.
    *   Isotropic Volume Fraction (Fiso, or `f_iso`): e.g., values from 0.0 to 0.9, ensuring `f_intra + f_iso < 1`.
    *   Mean Neurite Orientation (`mu`): Define orientation vectors (theta, phi) across the synthetic volume.
2.  **DWI Signal Generation:**
    *   Use a known DWI signal simulator. The `diffusemri.models.noddi_signal.noddi_signal_model` itself can be used to generate ideal signals if you trust its formulation for this purpose.
    *   Alternatively, if comparing against another toolbox (e.g., AMICO or MATLAB NODDI), use that toolbox's signal generator if available to create the synthetic signals based on your ground truth parameters. This helps isolate differences in fitting versus differences in signal generation.
    *   Ensure the same b-values, b-vectors (gradient table), and intrinsic diffusivity parameters (e.g., `d_intra`, `d_iso`) are used as will be used for fitting.
3.  **Add Noise:** Add Rician or Gaussian noise to the synthetic signals to simulate realistic data conditions. Varying SNR levels can be tested.

### b. Fitting and Comparison
1.  **Fit with `diffusemri` NODDI:**
    *   Use `diffusemri.fitting.noddi_fitter.fit_noddi_volume` (or `NoddiModelTorch.fit_batch`) to fit the custom NODDI model to the synthetic DWI data.
    *   Ensure model settings (intrinsic diffusivities) match those used for signal generation.
2.  **Fit with Other Implementation (e.g., AMICO, MATLAB NODDI):**
    *   Fit the NODDI model using the other implementation on the *exact same* synthetic DWI data and brain mask.
    *   Ensure all model settings (e.g., fixed diffusivities, number of iterations, convergence criteria if applicable) are matched as closely as possible.
3.  **Quantitative Comparison:**
    *   **Against Ground Truth:** For both implementations, compare the fitted parameter maps (NDI, ODI, Fiso, `mu`) against the original ground truth maps.
        *   Calculate error metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
        *   For orientation `mu`, calculate the mean angular difference between the fitted `mu` and true `mu`.
        *   Use correlation plots and scatter plots to visualize agreement.
    *   **Between Implementations:** Directly compare the parameter maps from `diffusemri`'s NODDI and the other implementation.
        *   Calculate MAE/RMSE and correlations between the maps.
        *   Bland-Altman plots can be useful for assessing agreement and biases.

## 2. Real Data Comparison

Comparing on real DWI data provides a qualitative assessment of the model's behavior in practical scenarios.

### a. Data Selection and Preprocessing
1.  **Dataset:** Choose an open-access, multi-shell DWI dataset suitable for NODDI analysis (e.g., from the Human Connectome Project, or other publicly available datasets).
2.  **Preprocessing:** Apply identical preprocessing steps to the data before fitting with either NODDI implementation. This includes:
    *   Denoising.
    *   Motion and eddy current correction (e.g., using FSL's `eddy`). Ensure b-vectors are correctly rotated.
    *   Brain masking.

### b. Fitting and Comparison
1.  **Fit with `diffusemri` NODDI:** Run the `diffusemri` NODDI fitting pipeline.
2.  **Fit with Other Implementation:** Run the other NODDI implementation using the same preprocessed data, mask, and gradient table. Match model settings as closely as possible.
3.  **Qualitative Comparison:**
    *   **Visual Inspection:** Visually compare the generated parameter maps (NDI, ODI, Fiso) from both implementations. Look for overall similarity in anatomical contrast and parameter ranges.
    *   **Histograms:** Plot histograms of parameter values within specific Regions of Interest (ROIs), such as major white matter tracts (e.g., corpus callosum, corticospinal tract), gray matter, and CSF. Compare the distributions.
    *   **Profile Plots:** If specific tracts are segmented, compare parameter profiles along these tracts.
    *   **Orientation Coherence:** For `mu` (mean orientation), visualize it using Directionally Encoded Color (DEC) maps or by overlaying orientation vectors/glyphs on anatomical images. Compare the coherence and anatomical plausibility of orientations.

## 3. Key Considerations for Fair Comparison

*   **Input Data Consistency:** Ensure identical DWI data, b-values, b-vectors (gradient table), and brain masks are used for all implementations being compared.
*   **Model Parameter Equivalency:**
    *   **Intrinsic Diffusivities:** Ensure `d_intra` (parallel diffusivity for sticks/zeppelins) and `d_iso` (isotropic CSF diffusivity) are set to the same fixed values in all models. The default in Zhang et al. (2012) is `d_intra = 1.7e-3 mm^2/s` and `d_iso = 3.0e-3 mm^2/s`.
    *   **S0 Handling:** Different implementations might handle S0 (signal at b=0) differently. Some might normalize the signal by S0 internally, while others expect raw signals. Strive for consistency or understand how this might affect parameter estimates (especially volume fractions). The `diffusemri` implementation uses S/S0 normalized signals for fitting.
    *   **Parameter Constraints:** Check if constraints on parameters (e.g., range of volume fractions, kappa) are handled differently.
*   **Fitting Algorithm Details:** If possible, try to match settings like the number of iterations, convergence criteria, and optimization algorithms. However, these are often internal to the specific implementation.
*   **Software Versions:** Note the versions of all software and libraries used, as updates can lead to changes in results.

## 4. Disclaimer

*   When comparing with other software packages (e.g., AMICO, MATLAB toolboxes), users must ensure they have the appropriate licenses for those tools and are complying with their usage terms.
*   Small numerical differences between implementations are common due to different numerical optimization strategies, convergence criteria, or slight variations in the exact mathematical approximations used (e.g., for spherical integrals). The focus should be on overall consistency and plausibility of the biophysical parameters.

By following these strategies, users can gain confidence in the `diffusemri` NODDI implementation and understand its characteristics relative to other available tools.
