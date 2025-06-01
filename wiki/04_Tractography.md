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
    from dipy.core.gradients import gradient_table
    from diffusemri.tracking.deterministic import track_deterministic_oudf
    
    # Create minimal synthetic data for tractography
    vol_shape = (5, 5, 5) # A small 3D volume
    num_gradients = 7 # 1 b0, 6 DWI

    # DWI data: Random data, b0 higher signal
    dwi_data_np = np.random.rand(vol_shape[0], vol_shape[1], vol_shape[2], num_gradients).astype(np.float32) * 500
    dwi_data_np[..., 0] = 1000 # b0 signal
    # Simulate some anisotropy for tracking: make one direction have lower signal (higher diffusivity)
    # For example, make signal lower along a simulated "bundle" in x-direction for y=2, z=2
    dwi_data_np[2, 2, :, 1] = 300 # Lower signal for bvec (1,0,0)
    dwi_data_np[2, 2, :, 4] = 300 # Lower signal for bvec (-1,0,0)


    # Gradient table (same as DTI example for simplicity)
    bvals_np = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float32)
    bvecs_np = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]
    ], dtype=np.float32)
    gtab = gradient_table(bvals_np, bvecs_np)

    # Affine matrix (identity for simple voxel space)
    affine_np = np.eye(4)

    # Seeds: a single seed point in the middle of the volume
    seeds_np = np.array([[2, 2, 2]], dtype=np.float32) # Voxel coordinates

    # Stopping metric map: FA map (or simple ones for this example)
    # For simplicity, use a map of ones, so tracking stops based on other criteria (angle, length)
    # A more realistic FA map would be derived from DTI fitting.
    fa_map_np = np.ones(vol_shape, dtype=np.float32)
    fa_threshold = 0.1 # Low threshold as FA map is artificial

    # Perform deterministic tractography
    # Parameters are adjusted for the small synthetic data
    streamlines_obj = track_deterministic_oudf(
        dwi_data=dwi_data_np,
        gtab=gtab,
        seeds=seeds_np,
        affine=affine_np,
        metric_map_for_stopping=fa_map_np,
        stopping_threshold_value=fa_threshold,
        # Model parameters
        sh_order=4,             # Lower SH order for small data
        model_max_peaks=1,      # Expect simple structure
        model_min_separation_angle=15,
        model_peak_threshold=0.3,
        # Tracking parameters
        step_size=0.5,          # Standard step size
        max_crossing_angle=60,  # Allow for some curvature
        min_length=2.0,         # Shorter min length for small volume
        max_length=20.0,        # Shorter max length
        max_steps=40            # Max steps = max_length / step_size
    )
    
    print(f"Generated {len(streamlines_obj)} streamlines.")
    if len(streamlines_obj) > 0:
        print(f"First streamline has {len(streamlines_obj[0])} points.")
        # Example: print first point of the first streamline
        # print(f"First point of first streamline: {streamlines_obj[0][0]}")
    ```
