# 04: Tractography

Tractography, also known as fiber tracking, is a set of computational methods used to reconstruct white matter pathways (fiber tracts or streamlines) in the brain from diffusion MRI (dMRI) data. This library provides tools for deterministic tractography.

## Deterministic Tractography

Deterministic tractography algorithms reconstruct streamlines by iteratively following the most probable orientation from a local diffusion model at each step.

*   **Purpose:** To reconstruct white matter pathways by modeling local fiber orientations (e.g., from ODF peaks) and tracking step-by-step from seed points.
*   **Main Function:** `diffusemri.tracking.deterministic.track_deterministic_oudf()`
    *   Note: This function acts as a wrapper to the core PyTorch-based tracking pipeline implemented in `diffusemri.tracking.pytorch_tracking_pipeline.track_deterministic_oudf`.

*   **Core Methodology & Key Inputs:**
    The current implementation performs ODF modeling, peak extraction, and tracking primarily using PyTorch-based components.
    *   **Input Data:**
        *   `dwi_data` (np.ndarray): The 4D dMRI data. This is converted to a PyTorch tensor internally.
        *   `gtab` (dipy.core.gradients.GradientTable): A Dipy GradientTable object containing b-values and b-vectors. This is used by the internal peak extraction model.
        *   `seeds` (np.ndarray): Seed points for initiating tracking. Can be a 3D binary mask (NumPy array) or an N x 3 array of coordinates in voxel space. This is converted to PyTorch tensors for internal processing; mask-based seeds are generated using Dipy's `seeds_from_mask`.
        *   `affine` (np.ndarray): A 4x4 affine transformation matrix mapping voxel coordinates to world/scanner space. Converted to a PyTorch tensor internally.
        *   `stopping_metric_map` (np.ndarray): A 3D scalar map (e.g., FA or a binary mask) used for the stopping criterion. Converted to a PyTorch tensor internally.
        *   `stopping_threshold_value` (float): Threshold applied to the `stopping_metric_map` to determine when tracking should terminate.
    *   **Internal ODF Modeling and Peak Extraction (PyTorch-based):**
        *   The function internally uses the `PeaksFromModel` class (from `diffusemri.tracking.pytorch_tracking_pipeline`) to:
            *   Perform a CSD-like Spherical Harmonic (SH) fit to the input `dwi_data` and `gtab`.
            *   Evaluate the Orientation Distribution Function (ODF).
            *   Extract peak orientations from the ODF in each voxel.
        *   This process is controlled by parameters like `sh_order`, `model_max_peaks`, `model_min_separation_angle`, and `model_peak_threshold`.
        *   The `PeaksFromModel` class itself uses Dipy's `get_sphere` internally.
    *   **Streamline Generation (PyTorch-based):**
        *   The extracted PyTorch peak data is then used by the `PyTorchPeaksDirectionGetter`.
        *   A `PyTorchThresholdStoppingCriterion` is initialized using the `stopping_metric_map` and `stopping_threshold_value`.
        *   The `PyTorchLocalTracking` class (both from `diffusemri.tracking.pytorch_tracking_pipeline`) performs the step-by-step streamline generation.
        *   This process is controlled by parameters like `step_size`, `max_crossing_angle`, `min_length`, `max_length`, and `max_steps`.
*   **Output:**
    *   A Dipy `Streamlines` object (`dipy.tracking.streamline.Streamlines`), which is a list-like container of the generated tracts (each tract being a NumPy array of 3D points in world coordinates). This ensures compatibility with Dipy's ecosystem for further analysis or visualization.

*   **Example Usage:**
    ```python
    import numpy as np
    # from dipy.core.gradients import gradient_table # For gtab creation
    # from diffusemri.tracking.deterministic import track_deterministic_oudf # Conceptual
    
    # # --- This is a conceptual example ---
    # # Assume dwi_data_np (4D NumPy), gtab (Dipy GradientTable), 
    # # affine_np (4x4 NumPy), seeds_np (NumPy mask or Nx3 coords),
    # # and fa_map_np (3D NumPy for stopping) are loaded/created.

    # # fa_threshold = 0.2

    # # streamlines_obj = track_deterministic_oudf(
    # #     dwi_data=dwi_data_np,
    # #     gtab=gtab,
    # #     seeds=seeds_np, # This is passed to the wrapper
    # #     affine=affine_np,
    # #     metric_map_for_stopping=fa_map_np, # Wrapper expects np.ndarray
    # #     stopping_threshold_value=fa_threshold,
    # #     # Model parameters (for internal PeaksFromModel)
    # #     sh_order=6,
    # #     model_max_peaks=3,
    # #     model_min_separation_angle=25,
    # #     model_peak_threshold=0.4,
    # #     # Tracking parameters
    # #     step_size=0.5,
    # #     max_crossing_angle=30,
    # #     min_length=10.0,
    # #     max_length=200.0,
    # #     max_steps=500 
    # # )
    
    # # print(f"Generated {len(streamlines_obj)} streamlines.")
    # # if streamlines_obj:
    # #     # Further processing, saving, or visualization using Dipy tools
    # #     pass
    print("Note: To run the tractography example, ensure 'track_deterministic_oudf' and input data are correctly set up.")
    ```
