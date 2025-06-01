# 04: Tractography

Tractography, also known as fiber tracking, is a set of computational methods used to reconstruct white matter pathways (fiber tracts or streamlines) in the brain from diffusion MRI (dMRI) data. This library provides tools for deterministic tractography.

**Note on Data Loading and Prerequisite Model Fitting:** The examples in this section assume you have already loaded your dMRI data (e.g., into NumPy arrays and a Dipy `GradientTable` or `PyTorchGradientTable`) and, for some tractography methods, fitted an appropriate local diffusion model (like CSD to get ODFs or their spherical harmonic coefficients). Please refer to previous wiki pages (`02_Diffusion_Models.md`) and example notebooks (`examples/`) for details on data loading, format conversion, and model fitting. For instance, the `track_deterministic_oudf` function described below internally fits a CSD model, so it requires raw DWI data.

## Deterministic Tractography

Deterministic tractography algorithms reconstruct streamlines by iteratively following the most probable orientation from a local diffusion model at each step.

*   **Purpose:** To reconstruct white matter pathways by modeling local fiber orientations (e.g., from ODF peaks) and tracking step-by-step from seed points.
*   **Main Function:** `diffusemri.tracking.deterministic.track_deterministic_oudf()`
    *   Note: This function acts as a wrapper to the core PyTorch-based tracking pipeline implemented in `diffusemri.tracking.pytorch_tracking_pipeline.track_deterministic_oudf`.

*   **Core Methodology & Key Inputs:**
    The current implementation performs ODF modeling, peak extraction, and tracking primarily using PyTorch-based components.
    *   **Input Data:**
        *   `dwi_data` (np.ndarray): The 4D dMRI data.
        *   `gtab` (dipy.core.gradients.GradientTable): A Dipy GradientTable object.
        *   `seeds` (np.ndarray): Seed points for initiating tracking (3D binary mask or N x 3 voxel coordinates).
        *   `affine` (np.ndarray): 4x4 affine transformation matrix.
        *   `metric_map_for_stopping` (np.ndarray): A 3D scalar map (e.g., FA or GFA) used for stopping.
        *   `stopping_threshold_value` (float): Threshold for `metric_map_for_stopping`.
    *   **Internal ODF Modeling (CSD-like, PyTorch-based):**
        *   The function uses `PeaksFromModel` (from `tracking.pytorch_tracking_pipeline`) to fit a CSD-like model, evaluate ODFs, and extract peak orientations. Controlled by `sh_order`, `response` (for CSD), peak extraction parameters.
    *   **Streamline Generation (PyTorch-based):**
        *   Uses `PyTorchPeaksDirectionGetter`, `PyTorchThresholdStoppingCriterion`, and `PyTorchLocalTracking`. Controlled by `step_size`, `max_crossing_angle`, length filters.
*   **Output:**
    *   A Dipy `Streamlines` object (`dipy.tracking.streamline.Streamlines`).

*   **Conceptual Python Example:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table # For gtab creation
    # from diffusemri.tracking.deterministic import track_deterministic_oudf # Actual import

    # --- Assume dwi_data_np, gtab, affine_np, seeds_np, fa_map_np are loaded/prepared ---
    # vol_shape = (5,5,5); num_gradients = 7
    # dwi_data_np = np.random.rand(vol_shape[0],vol_shape[1],vol_shape[2],num_gradients).astype(np.float32)
    # bvals_np = np.array([0,1000,1000,1000,1000,1000,1000], dtype=float)
    # bvecs_np = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=float)
    # gtab = gradient_table(bvals_np, bvecs_np)
    # affine_np = np.eye(4)
    # seeds_np = np.array([[2,2,2]], dtype=np.float32)
    # fa_map_np = np.ones(vol_shape, dtype=np.float32) * 0.5
    # fa_threshold = 0.1

    # Perform deterministic tractography (conceptual call)
    # streamlines_obj = track_deterministic_oudf(
    #     dwi_data=dwi_data_np,
    #     gtab=gtab,
    #     seeds=seeds_np,
    #     affine=affine_np,
    #     metric_map_for_stopping=fa_map_np,
    #     stopping_threshold_value=fa_threshold,
    #     sh_order=4, model_max_peaks=1, min_length=2.0, max_length=20.0
    # )
    # print(f"Generated {len(streamlines_obj) if 'streamlines_obj' in locals() else 'N/A (conceptual)'} streamlines.")
    print("Note: Tractography example is conceptual. See '12_Deterministic_Tractography.ipynb' for a runnable version.")
    ```
*   **CLI Usage:** Deterministic tractography can be run via `python cli/run_tracking.py det_oudf ...`. Refer to the script's help for arguments.

## Probabilistic Tractography (Conceptual)

While the primary `track_deterministic_oudf` is PyTorch-based, the library also contains a Dipy-based probabilistic tracking wrapper: `diffusemri.tracking.probabilistic.track_probabilistic_odf`.

*   **Purpose:** Reconstructs pathways by sampling from orientation distributions (ODFs), representing uncertainty.
*   **Key Function:** `diffusemri.tracking.probabilistic.track_probabilistic_odf()`
*   **Methodology:** Leverages Dipy's `ProbabilisticDirectionGetter` and `LocalTracking`.
*   **Main Inputs:**
    *   `odf_fit_object`: A fitted ODF model object from Dipy (e.g., from `CsdModel` or `QballModel` wrappers if they expose the Dipy fit object).
    *   `seeds` (np.ndarray): Seed points or mask.
    *   `stopping_criterion` (dipy.tracking.stopping_criterion.StoppingCriterion).
    *   `sphere` (dipy.data.Sphere).
*   **Output:** Dipy `Streamlines` object.
*   **Note:** This function requires a pre-fitted Dipy model object that provides ODF information (e.g., SH coefficients). The `diffusemri.models.CsdModel` and `QballModel` would need to expose the underlying Dipy fit object for use here.

For detailed examples and runnable code, please refer to the Jupyter notebooks in the `examples/` directory, especially `12_Deterministic_Tractography.ipynb`.
