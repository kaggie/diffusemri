# 02: Diffusion Models

The `diffusemri` library provides implementations and interfaces for several diffusion MRI models. These models allow for the characterization of water diffusion in tissue, providing insights into microstructure and connectivity.

## Diffusion Tensor Imaging (DTI)

*   **Purpose:** A widely used model to characterize anisotropic water diffusion by fitting a tensor to the diffusion signal at each voxel. It is sensitive to the primary direction of diffusion and the degree of anisotropy.
*   **Key Library Components:** (Assuming `diffusemri.models.DtiModel` or similar, and fitting via `fitting.dti_fitter.py`)
    *   Model Class: e.g., `diffusemri.models.DtiModel`
    *   Fitting Function: e.g., `diffusemri.fitting.dti_fitter.fit_dti_volume()`
*   **Method:** Typically involves a linear least-squares fit to the log-transformed diffusion signal. The library's implementation uses NumPy for these calculations (as seen in `dti_fitter.py` and `models/dti.py`).
*   **Outputs:** A 3x3 diffusion tensor per voxel, from which scalar metrics are derived.
*   **Common Parameters Generated:**
    *   Fractional Anisotropy (FA)
    *   Mean Diffusivity (MD)
    *   Axial Diffusivity (AD)
    *   Radial Diffusivity (RD)
    *   Directionally Encoded Color FA (DEC FA) maps
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.fitting.dti_fitter import fit_dti_volume

    # Create minimal synthetic data for DTI
    # Shape: (x, y, z, num_gradients) -> e.g., a 2x2x2 volume with 7 scans (1 b0, 6 DWI)
    dwi_data_np = np.zeros((2, 2, 2, 7), dtype=np.float32)
    # Simulate some signal: b0 images are bright, DWI are dimmer
    dwi_data_np[..., 0] = 1500 # b0 signal
    dwi_data_np[..., 1:] = np.random.uniform(500, 800, size=(2, 2, 2, 6))

    # b-values: one b0 and six b=1000 s/mm^2
    bvals_np = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float32)

    # b-vectors: b0 has (0,0,0), others are non-collinear
    bvecs_np = np.array([
        [0, 0, 0],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1]
    ], dtype=np.float32)
    # For Dipy, bvecs should be (N, 3) where N is number of volumes.
    # If b0s are present, their bvecs are often (0,0,0) or actual vectors if measured.
    # Here, we ensure the bvecs array matches the number of volumes.

    gtab = gradient_table(bvals_np, bvecs_np)
    
    # Fit DTI model
    # Note: fit_dti_volume expects bvals and bvecs directly, not the gtab object for now.
    # This might differ from Dipy's typical model.fit(gtab, data) pattern.
    # Let's assume fit_dti_volume takes dwi_data, gtab.bvals, gtab.bvecs
    dti_params_dict = fit_dti_volume(dwi_data_np, gtab.bvals, gtab.bvecs)
    
    fa_map = dti_params_dict["FA"]
    md_map = dti_params_dict["MD"]

    print(f"DTI FA map shape: {fa_map.shape}")
    print(f"DTI MD map shape: {md_map.shape}")
    print(f"FA at first voxel: {fa_map[0,0,0] if fa_map is not None else 'N/A'}")
    ```

## Diffusion Kurtosis Imaging (DKI)

*   **Purpose:** Extends DTI to characterize non-Gaussian water diffusion, offering more detailed insights into tissue microstructure by quantifying the degree to which the diffusion displacement profile deviates from a Gaussian distribution.
*   **Data Requirements:** Requires multi-shell acquisition schemes (at least two non-zero b-values, typically including a high b-value shell e.g., b=2000 s/mm² or higher).
*   **Key Library Components:**
    *   Model Class: `diffusemri.models.DkiModel`
*   **Method:** The implementation (`diffusemri.models.DkiModel`) leverages Dipy for its core calculations to fit the diffusion kurtosis tensor.
*   **Outputs:** A diffusion kurtosis tensor per voxel.
*   **Common Parameters Generated (in addition to DTI metrics):**
    *   Mean Kurtosis (MK)
    *   Axial Kurtosis (AK)
    *   Radial Kurtosis (RK)
    *   Kurtosis Anisotropy (KA)
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.models import DkiModel

    # Create minimal synthetic data for DKI
    # Shape: (x, y, z, num_gradients) -> e.g., a 2x2x2 volume
    # Needs at least 2 non-zero b-values. e.g., 1 b0, 6 b=1000, 6 b=2000
    num_b0 = 1
    num_b1000_dirs = 6
    num_b2000_dirs = 6
    total_volumes = num_b0 + num_b1000_dirs + num_b2000_dirs
    dwi_data_np = np.zeros((2, 2, 2, total_volumes), dtype=np.float32)

    # Simulate signal
    dwi_data_np[..., 0] = 2000 # b0
    dwi_data_np[..., 1:1+num_b1000_dirs] = np.random.uniform(600, 900, size=(2,2,2,num_b1000_dirs)) # b=1000
    dwi_data_np[..., 1+num_b1000_dirs:] = np.random.uniform(300, 500, size=(2,2,2,num_b2000_dirs)) # b=2000

    bvals_list = [0]*num_b0 + [1000]*num_b1000_dirs + [2000]*num_b2000_dirs
    bvals_np = np.array(bvals_list, dtype=np.float32)

    # Simple b-vectors (non-collinear for each shell)
    bvecs_list = [[0,0,0]]*num_b0
    base_bvecs = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]])
    bvecs_list.extend(base_bvecs[:num_b1000_dirs])
    bvecs_list.extend(base_bvecs[:num_b2000_dirs]) # Can reuse or use different for more realism
    bvecs_np = np.array(bvecs_list, dtype=np.float32)

    gtab = gradient_table(bvals_np, bvecs_np)
    
    # Fit DKI model
    # DkiModel from diffusemri.models uses Dipy's DkiModel internally
    dki_model_instance = DkiModel(gtab)
    # The .fit method in diffusemri.models.DkiModel wraps dipy's fit.
    # It might return a Dipy DkiFit object or a dictionary of maps.
    # Assuming it returns a Dipy DkiFit object based on current structure.
    dki_fit_result = dki_model_instance.fit(dwi_data_np)
    
    mk_map = dki_fit_result.mk() # Standard Dipy DkiFit method
    # fa_map_from_dki = dki_fit_result.fa # DTI params can also be derived
    # md_map_from_dki = dki_fit_result.md

    print(f"DKI MK map shape: {mk_map.shape}")
    print(f"MK at first voxel: {mk_map[0,0,0] if mk_map is not None else 'N/A'}")
    ```

## Constrained Spherical Deconvolution (CSD)

*   **Purpose:** An advanced model capable of resolving crossing fiber populations within a voxel by estimating Orientation Distribution Functions (ODFs). The ODF describes the angular profile of diffusion, revealing one or more dominant fiber orientations.
*   **Key Library Components:**
    *   Model Class: `diffusemri.models.CsdModel`
*   **Method:** The implementation (`diffusemri.models.CsdModel`) leverages Dipy's CSD functionalities and response function estimation (e.g., `auto_response_ssst`).
*   **Outputs:** An ODF (represented as spherical harmonic coefficients) per voxel. From the ODF, fiber orientations (peaks) can be extracted.
*   **Common Parameters Generated:**
    *   Orientation Distribution Functions (ODFs)
    *   Generalized Fractional Anisotropy (GFA)
    *   Peaks from ODFs (used for tractography)
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.models import CsdModel # Assuming CsdModel is in diffusemri.models
    from dipy.reconst.shm import calculate_max_order # To estimate sh_order for response

    # Create minimal synthetic data for CSD
    # Shape: (x, y, z, num_gradients) -> e.g., a 2x2x2 volume
    # Needs HARDI data (e.g., b=1000 or higher, >=30 directions ideally)
    # For simplicity, 1 b0, 15 b=1000 directions
    num_b0 = 1
    num_b1000_dirs = 15 # Minimal for CSD
    total_volumes = num_b0 + num_b1000_dirs
    dwi_data_np = np.zeros((2, 2, 2, total_volumes), dtype=np.float32)

    dwi_data_np[..., 0] = 1800 # b0
    # Simulate anisotropic signal for CSD
    dwi_data_np[..., 1:8] = np.random.uniform(400, 700, size=(2,2,2,7)) # Lower signal along some dirs
    dwi_data_np[..., 8:] = np.random.uniform(700, 1000, size=(2,2,2,8)) # Higher signal along others

    bvals_np = np.array([0]*num_b0 + [1000]*num_b1000_dirs, dtype=np.float32)

    # Generate some semi-random b-vectors (more realistic would use Dipy's sphere_vf_to_cartesian)
    phi = np.random.uniform(0, np.pi, num_b1000_dirs)
    theta = np.random.uniform(0, 2 * np.pi, num_b1000_dirs)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    generated_bvecs = np.vstack((x,y,z)).T

    bvecs_list = [[0,0,0]]*num_b0
    bvecs_list.extend(generated_bvecs)
    bvecs_np = np.array(bvecs_list, dtype=np.float32)

    gtab = gradient_table(bvals_np, bvecs_np)

    # Fit CSD model
    # diffusemri.models.CsdModel wraps Dipy's CsdModel.
    # It might auto-estimate response or require it. Let's assume auto-response for simplicity.
    # sh_order might be determined internally or specifiable.
    # A common sh_order for CSD is 6 or 8.
    # The response function is crucial for CSD. Dipy's auto_response_ssst is often used.
    # The wrapper diffusemri.models.CsdModel handles this internally.
    csd_model_instance = CsdModel(gtab, sh_order=6) # Example sh_order
    
    # The .fit method in diffusemri.models.CsdModel fits the model and stores results.
    # It likely computes SH coefficients for ODFs and GFA.
    csd_model_instance.fit(dwi_data_np)
    
    odf_sh_coeffs = csd_model_instance.shm_coeff # Spherical harmonic coefficients
    gfa_map = csd_model_instance.gfa # Generalized Fractional Anisotropy

    print(f"CSD ODF SH Coeffs shape: {odf_sh_coeffs.shape}")
    print(f"CSD GFA map shape: {gfa_map.shape}")
    print(f"GFA at first voxel: {gfa_map[0,0,0] if gfa_map is not None else 'N/A'}")
    ```

## Q-Ball Imaging (QBI)

*   **Purpose:** Another method for reconstructing Orientation Distribution Functions (ODFs) to characterize fiber orientations, often suitable for single-shell diffusion data. Like CSD, its main output is the ODF, providing an angular profile of diffusion.
*   **Key Library Components:**
    *   Model Class: `diffusemri.models.QballModel`
*   **Method:** The implementation (`diffusemri.models.QballModel`) leverages Dipy's Q-Ball functionalities (specifically Constant Solid Angle - CSA type).
*   **Outputs:** An ODF (represented as spherical harmonic coefficients) per voxel.
*   **Common Parameters Generated:**
    *   Orientation Distribution Functions (ODFs)
    *   Generalized Fractional Anisotropy (GFA)
    *   Peaks from ODFs
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.models import QballModel # Assuming QballModel is in diffusemri.models

    # Create minimal synthetic data for Q-Ball
    # Shape: (x, y, z, num_gradients) -> e.g., a 2x2x2 volume
    # Typically single-shell HARDI data (e.g., b=1000 or 2000 or 3000, >=30 directions)
    # For simplicity, 1 b0, 15 b=1000 directions
    num_b0 = 1
    num_b1000_dirs = 15 # Minimal for Q-Ball
    total_volumes = num_b0 + num_b1000_dirs
    dwi_data_np = np.zeros((2, 2, 2, total_volumes), dtype=np.float32)

    dwi_data_np[..., 0] = 1900 # b0
    # Simulate anisotropic signal
    dwi_data_np[..., 1:8] = np.random.uniform(300, 600, size=(2,2,2,7))
    dwi_data_np[..., 8:] = np.random.uniform(600, 900, size=(2,2,2,8))

    bvals_np = np.array([0]*num_b0 + [1000]*num_b1000_dirs, dtype=np.float32)
    
    # Reusing b-vector generation logic from CSD example for simplicity
    phi = np.random.uniform(0, np.pi, num_b1000_dirs)
    theta = np.random.uniform(0, 2 * np.pi, num_b1000_dirs)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    generated_bvecs = np.vstack((x,y,z)).T
    
    bvecs_list = [[0,0,0]]*num_b0
    bvecs_list.extend(generated_bvecs)
    bvecs_np = np.array(bvecs_list, dtype=np.float32)

    gtab = gradient_table(bvals_np, bvecs_np)

    # Fit Q-Ball model (Constant Solid Angle type via Dipy)
    # diffusemri.models.QballModel wraps Dipy's CsaOdfModel.
    # sh_order is a key parameter.
    qball_model_instance = QballModel(gtab, sh_order=6) # Example sh_order

    # The .fit method fits the model and stores results.
    qball_model_instance.fit(dwi_data_np)

    odf_sh_coeffs_qball = qball_model_instance.shm_coeff # SH coeffs for ODF
    gfa_map_qball = qball_model_instance.gfa # GFA from Q-Ball ODFs

    print(f"Q-Ball ODF SH Coeffs shape: {odf_sh_coeffs_qball.shape}")
    print(f"Q-Ball GFA map shape: {gfa_map_qball.shape}")
    print(f"Q-Ball GFA at first voxel: {gfa_map_qball[0,0,0] if gfa_map_qball is not None else 'N/A'}")
    ```

## Neurite Orientation Dispersion and Density Imaging (NODDI)

*   **Purpose:** A multi-compartment biophysical model that estimates parameters like Neurite Density Index (NDI), Orientation Dispersion Index (ODI), and isotropic (e.g., CSF) volume fraction (Fiso). It aims to provide more specific measures of tissue microstructure than DTI.
*   **Key Library Components:**
    *   Model Fitting Function: `diffusemri.fitting.noddi_fitter.fit_noddi_volume()`
    *   Underlying PyTorch Model: `diffusemri.models.noddi_model.NoddiModelTorch`
*   **Method:** The `diffusemri` library includes a custom implementation of the NODDI model based on the original formulation by Zhang et al. (2012). Fitting is performed using PyTorch for efficiency, supporting batch-wise processing of voxels and GPU acceleration. The gradient information is handled by `diffusemri.utils.pytorch_gradient_utils.PyTorchGradientTable`.
*   **Outputs:** Voxel-wise maps of NODDI parameters.
*   **Common Parameters Generated:**
    *   Neurite Density Index (NDI, also `f_intra` or `vic`)
    *   Orientation Dispersion Index (ODI)
    *   Isotropic Volume Fraction (Fiso, also `f_iso` or `viso`)
    *   Mean orientation of neurites (`mu_theta`, `mu_phi` - spherical coordinates)
    *   Watson model concentration parameter (`kappa`, related to ODI)
*   **Advanced Fitting Features:**
    *   **Smart Initialization:** The mean neurite orientation (`mu`) can be initialized using an `initial_orientation_map` (e.g., derived from DTI primary eigenvectors) passed to `fit_noddi_volume`. This can potentially improve convergence speed and accuracy.
    *   **Regularization:** L1 and L2 regularization options are available during the fitting process. L1 can be applied to specific constrained parameters (e.g., `f_iso` for promoting sparsity), and L2 is applied to unconstrained parameters for weight decay. These are configurable via the `fit_params` argument in `fit_noddi_volume` (e.g., `fit_params={'l1_penalty_weight': 0.01, 'l1_params_to_regularize': ['f_iso']}`).
*   **Example Usage (Basic NODDI fitting):**
    ```python
    import numpy as np
    import torch
    from diffusemri.fitting.noddi_fitter import fit_noddi_volume
    from diffusemri.utils.pytorch_gradient_utils import PyTorchGradientTable

    # Create minimal synthetic data for NODDI
    # Shape: (x, y, z, num_gradients) -> e.g., a 2x1x1 volume for speed
    # Needs multi-shell data, e.g., 2 b0s, 1 shell b=700 (low), 1 shell b=2000 (high)
    num_b0 = 2
    num_b700_dirs = 8  # Minimal directions
    num_b2000_dirs = 8 # Minimal directions
    total_volumes = num_b0 + num_b700_dirs + num_b2000_dirs

    # Using a smaller volume (2,1,1) to make example faster
    dwi_data_np = np.zeros((2, 1, 1, total_volumes), dtype=np.float32)

    # Simulate signal: high for b0, decreasing with b-value
    dwi_data_np[..., :num_b0] = 2500
    # Simulate some orientation dependence for b700 and b2000 shells
    for i in range(num_b700_dirs):
        dwi_data_np[..., num_b0 + i] = np.random.uniform(1000, 1500) - (i%2 * 200) # Simple anisotropy
    for i in range(num_b2000_dirs):
        dwi_data_np[..., num_b0 + num_b700_dirs + i] = np.random.uniform(400, 700) - (i%3 * 100)


    bvals_list = [0]*num_b0 + [700]*num_b700_dirs + [2000]*num_b2000_dirs
    bvals_np = np.array(bvals_list, dtype=np.float32)

    # Simple b-vectors
    bvecs_list = [[0,0,0]]*num_b0
    base_bvecs_shell1 = np.array([
        [1,0,0], [0,1,0], [0,0,1], [-1,0,0],
        [0,-1,0], [0,0,-1], [1,1,0]/np.sqrt(2), [-1,1,0]/np.sqrt(2)
    ])[:num_b700_dirs]
    base_bvecs_shell2 = np.array([
        [1,0,1]/np.sqrt(2), [0,1,1]/np.sqrt(2), [1,1,1]/np.sqrt(3), [-1,-1,0]/np.sqrt(2),
        [0,-1,-1]/np.sqrt(2), [-1,0,-1]/np.sqrt(2), [1,-1,0]/np.sqrt(2), [-1,-1,-1]/np.sqrt(3)
    ])[:num_b2000_dirs]

    bvecs_list.extend(base_bvecs_shell1)
    bvecs_list.extend(base_bvecs_shell2)
    bvecs_np = np.array(bvecs_list, dtype=np.float32)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # The PyTorchGradientTable is used by the NODDI fitter
    gtab_torch = PyTorchGradientTable(bvals_np, bvecs_np)

    # Basic fitting
    # fit_noddi_volume takes numpy dwi_data and PyTorchGradientTable
    noddi_maps = fit_noddi_volume(
        dwi_data=dwi_data_np,
        gtab=gtab_torch,
        device=device_str, # Pass device string: "cuda" or "cpu"
        fit_params={'n_iterations': 50} # Reduce iterations for faster example
    )
    ndi_map = noddi_maps["f_intra"]
    odi_map = noddi_maps["odi"]
    fiso_map = noddi_maps["f_iso"]

    print(f"NODDI NDI map shape: {ndi_map.shape}")
    print(f"NODDI ODI map shape: {odi_map.shape}")
    print(f"NDI at first voxel: {ndi_map[0,0,0] if ndi_map is not None else 'N/A'}")
    print(f"ODI at first voxel: {odi_map[0,0,0] if odi_map is not None else 'N/A'}")
    ```
*   **Example Usage (NODDI with Smart Initialization and Regularization):**
    ```python
    import numpy as np
    import torch
    from diffusemri.fitting.noddi_fitter import fit_noddi_volume
    from diffusemri.utils.pytorch_gradient_utils import PyTorchGradientTable

    # --- Using the same synthetic data as basic NODDI example above for brevity ---
    # Re-generate data and gtab if running this section independently
    num_b0 = 2; num_b700_dirs = 8; num_b2000_dirs = 8; total_volumes = num_b0 + num_b700_dirs + num_b2000_dirs
    dwi_data_np = np.zeros((2, 1, 1, total_volumes), dtype=np.float32)
    dwi_data_np[..., :num_b0] = 2500
    for i in range(num_b700_dirs): dwi_data_np[..., num_b0 + i] = np.random.uniform(1000, 1500) - (i%2 * 200)
    for i in range(num_b2000_dirs): dwi_data_np[..., num_b0 + num_b700_dirs + i] = np.random.uniform(400, 700) - (i%3 * 100)
    bvals_list = [0]*num_b0 + [700]*num_b700_dirs + [2000]*num_b2000_dirs
    bvals_np = np.array(bvals_list, dtype=np.float32)
    bvecs_list = [[0,0,0]]*num_b0
    base_bvecs_shell1 = np.array([[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1], [1,1,0]/np.sqrt(2), [-1,1,0]/np.sqrt(2)])[:num_b700_dirs]
    base_bvecs_shell2 = np.array([[1,0,1]/np.sqrt(2), [0,1,1]/np.sqrt(2), [1,1,1]/np.sqrt(3), [-1,-1,0]/np.sqrt(2), [0,-1,-1]/np.sqrt(2), [-1,0,-1]/np.sqrt(2), [1,-1,0]/np.sqrt(2), [-1,-1,-1]/np.sqrt(3)])[:num_b2000_dirs]
    bvecs_list.extend(base_bvecs_shell1); bvecs_list.extend(base_bvecs_shell2)
    bvecs_np = np.array(bvecs_list, dtype=np.float32)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    gtab_torch = PyTorchGradientTable(bvals_np, bvecs_np)
    # --- End of data re-generation ---

    # Create a dummy initial orientation map (e.g., from DTI v1 or just random for example)
    # Shape should be (X, Y, Z, 3) for the main orientations (v1s)
    initial_orientation_map_np = np.random.rand(2, 1, 1, 3).astype(np.float32)
    # Normalize the vectors
    norm = np.linalg.norm(initial_orientation_map_np, axis=-1, keepdims=True)
    initial_orientation_map_np = initial_orientation_map_np / (norm + 1e-6) # add epsilon to avoid div by zero

    advanced_fit_params = {
        'learning_rate': 0.005, # Smaller LR often better for complex fits
        'n_iterations': 100, # Reduce iterations for faster example
        'l1_penalty_weight': 0.001,
        'l1_params_to_regularize': ['f_iso'],
        'l2_penalty_weight': 0.0001
    }

    noddi_maps_advanced = fit_noddi_volume(
        dwi_data=dwi_data_np,
        gtab=gtab_torch,
        initial_orientation_map=initial_orientation_map_np,
        fit_params=advanced_fit_params,
        device=device_str
    )
    ndi_map_adv = noddi_maps_advanced["f_intra"]
    odi_map_adv = noddi_maps_advanced["odi"]
    fiso_map_adv = noddi_maps_advanced["f_iso"]

    print(f"Advanced NODDI NDI map shape: {ndi_map_adv.shape}")
    print(f"Advanced NODDI ODI map shape: {odi_map_adv.shape}")
    print(f"Advanced NODDI FISO map shape: {fiso_map_adv.shape}")
    print(f"NDI (adv) at first voxel: {ndi_map_adv[0,0,0] if ndi_map_adv is not None else 'N/A'}")
    ```
