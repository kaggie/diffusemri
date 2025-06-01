# 02: Diffusion Models

The `diffusemri` library provides implementations and interfaces for several diffusion MRI models. These models allow for the characterization of water diffusion in tissue, providing insights into microstructure and connectivity.

**Note on Data Loading:** The examples in this section create minimal synthetic data for brevity. For analyzing your own data, please use the comprehensive data loading and format conversion utilities described in the `01_Preprocessing_Tools.md` (for DICOM to NIfTI), `07_Format_Conversion.md` (for NRRD, MHD, Analyze, PAR/REC to NIfTI etc.) wiki pages, and the Jupyter notebooks in the `examples/` directory. These tools will help you load your data into the NumPy arrays and Dipy `GradientTable` (or `PyTorchGradientTable` for NODDI) objects expected by the model fitting functions.

## Diffusion Tensor Imaging (DTI)

*   **Purpose:** A widely used model to characterize anisotropic water diffusion by fitting a tensor to the diffusion signal at each voxel. It is sensitive to the primary direction of diffusion and the degree of anisotropy.
*   **Key Library Components:**
    *   Fitting Function: `diffusemri.fitting.dti_fitter.fit_dti_volume()`
    *   Underlying Logic: `diffusemri.models.dti` (contains functions for tensor calculation and metric derivation).
*   **Method:** Typically involves a linear least-squares fit to the log-transformed diffusion signal. The library's implementation uses NumPy for these calculations.
*   **Outputs:** A 3x3 diffusion tensor per voxel, from which scalar metrics are derived.
*   **Common Parameters Generated:**
    *   Fractional Anisotropy (FA)
    *   Mean Diffusivity (MD)
    *   Axial Diffusivity (AD)
    *   Radial Diffusivity (RD)
    *   Diffusion Tensor elements (`D_tensor_map`)
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table # For creating gtab from bvals/bvecs
    from diffusemri.fitting.dti_fitter import fit_dti_volume

    # Create minimal synthetic data for DTI
    # Shape: (x, y, z, num_gradients) -> e.g., a 2x2x2 volume with 7 scans (1 b0, 6 DWI)
    dwi_data_np = np.zeros((2, 2, 2, 7), dtype=np.float32)
    dwi_data_np[..., 0] = 1500 # b0 signal
    dwi_data_np[..., 1:] = np.random.uniform(500, 800, size=(2, 2, 2, 6))

    bvals_np = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float32)
    bvecs_np = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]
    ], dtype=np.float32)
    
    # Fit DTI model
    # fit_dti_volume takes dwi_data (numpy), bvals (numpy), and bvecs (numpy)
    dti_params_dict = fit_dti_volume(dwi_data_np, bvals_np, bvecs_np)
    
    fa_map = dti_params_dict.get("FA") # Use .get for safety
    md_map = dti_params_dict.get("MD")

    if fa_map is not None:
        print(f"DTI FA map shape: {fa_map.shape}")
        print(f"FA at first voxel: {fa_map[0,0,0]}")
    if md_map is not None:
        print(f"DTI MD map shape: {md_map.shape}")
    # For a runnable example, see examples/09_DTI_Fitting_and_Metrics.ipynb
    ```

## Diffusion Kurtosis Imaging (DKI)

*   **Purpose:** Extends DTI to characterize non-Gaussian water diffusion, offering more detailed insights into tissue microstructure.
*   **Data Requirements:** Requires multi-shell acquisition schemes (at least two non-zero b-values).
*   **Key Library Components:**
    *   Model Class: `diffusemri.models.DkiModel` (wraps Dipy's `DiffusionKurtosisModel`).
*   **Method:** Fits the diffusion kurtosis tensor using Dipy's underlying algorithms.
*   **Common Parameters Generated (in addition to DTI metrics like FA, MD):**
    *   Mean Kurtosis (MK)
    *   Axial Kurtosis (AK)
    *   Radial Kurtosis (RK)
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.models import DkiModel

    # Create minimal synthetic data for DKI (e.g., 1 b0, 6 b1000, 6 b2000)
    num_b0, num_b1000, num_b2000 = 1, 6, 6
    total_vols = num_b0 + num_b1000 + num_b2000
    dwi_data_np = np.zeros((2,2,2, total_vols), dtype=np.float32)
    dwi_data_np[...,0] = 2000 # b0
    dwi_data_np[...,num_b0:num_b0+num_b1000] = 800
    dwi_data_np[...,num_b0+num_b1000:] = 400

    bvals_np = np.array([0]*num_b0 + [1000]*num_b1000 + [2000]*num_b2000, dtype=float)
    bvecs_np = np.random.rand(total_vols, 3) * 2 - 1
    bvecs_np[0:num_b0,:] = 0
    bvecs_np[num_b0:] /= np.linalg.norm(bvecs_np[num_b0:], axis=1, keepdims=True)

    gtab = gradient_table(bvals_np, bvecs_np)
    
    dki_model = DkiModel(gtab)
    # The .fit() method in the wrapper calls Dipy's fit and stores the Dipy DkiFit object
    # dki_model.fit(dwi_data_np) # This would fit the model
    # mk_map = dki_model.mk # Access metrics via properties
    
    # print(f"DKI MK map shape: {mk_map.shape if 'mk_map' in locals() else 'N/A (conceptual fit)'}")
    print("Note: DKI example conceptual fit. See examples/11_Other_Models_CSD_DKI_QBall.ipynb for runnable version.")
    ```

## Constrained Spherical Deconvolution (CSD)

*   **Purpose:** Estimates Orientation Distribution Functions (ODFs) to resolve crossing fiber populations.
*   **Data Requirements:** Typically HARDI data (single high b-value shell with many directions, or multi-shell).
*   **Key Library Components:**
    *   Model Class: `diffusemri.models.CsdModel` (wraps Dipy's CSD/MSMT-CSD).
*   **Method:** Uses Dipy's CSD algorithms. Can auto-estimate response functions or use a provided one.
*   **Common Parameters Generated:** ODFs (as spherical harmonic coefficients), GFA, fiber peaks.
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.models import CsdModel

    # Minimal HARDI-like data (1 b0, 15 b1000)
    num_b0, num_b1000_csd = 1, 15
    total_vols_csd = num_b0 + num_b1000_csd
    dwi_data_np = np.zeros((2,2,2, total_vols_csd), dtype=np.float32)
    dwi_data_np[...,0] = 1800
    dwi_data_np[...,1:] = 700

    bvals_np = np.array([0]*num_b0 + [1000]*num_b1000_csd, dtype=float)
    bvecs_np = np.random.rand(total_vols_csd, 3)
    bvecs_np[0:num_b0,:] = 0
    bvecs_np[num_b0:] /= np.linalg.norm(bvecs_np[num_b0:], axis=1, keepdims=True)

    gtab = gradient_table(bvals_np, bvecs_np)
    
    # CsdModel wrapper handles response estimation or uses provided one.
    csd_model = CsdModel(gtab, sh_order_max=6) # Example sh_order
    # csd_model.fit(dwi_data_np)
    # gfa_map = csd_model.gfa
    
    # print(f"CSD GFA map shape: {gfa_map.shape if 'gfa_map' in locals() else 'N/A (conceptual fit)'}")
    print("Note: CSD example conceptual fit. See examples/11_Other_Models_CSD_DKI_QBall.ipynb for runnable version.")
    ```

## Q-Ball Imaging (QBI)

*   **Purpose:** Reconstructs ODFs, often from single-shell HARDI data.
*   **Key Library Components:**
    *   Model Class: `diffusemri.models.QballModel` (wraps Dipy's `QballModel`, CSA type).
*   **Method:** Uses Dipy's Constant Solid Angle ODF reconstruction.
*   **Common Parameters Generated:** ODFs (SH coefficients), GFA, fiber peaks.
*   **Example Usage:**
    ```python
    import numpy as np
    from dipy.core.gradients import gradient_table
    from diffusemri.models import QballModel
    
    # Can use similar HARDI data as CSD example
    # (Reusing gtab from CSD example for brevity)
    # dwi_data_np_qball = dwi_data_np # from CSD example
    
    qball_model = QballModel(gtab, sh_order_max=6) # gtab from CSD example
    # qball_model.fit(dwi_data_np_qball)
    # gfa_map_qball = qball_model.gfa

    # print(f"Q-Ball GFA map shape: {gfa_map_qball.shape if 'gfa_map_qball' in locals() else 'N/A (conceptual fit)'}")
    print("Note: Q-Ball example conceptual fit. See examples/11_Other_Models_CSD_DKI_QBall.ipynb for runnable version.")
    ```

## Neurite Orientation Dispersion and Density Imaging (NODDI)

*   **Purpose:** A multi-compartment biophysical model estimating neurite density (NDI), orientation dispersion (ODI), and isotropic volume fraction (Fiso).
*   **Data Requirements:** Multi-shell data (typically 2-3 shells including b0).
*   **Key Library Components:**
    *   Fitting Function: `diffusemri.fitting.noddi_fitter.fit_noddi_volume()`
    *   Gradient Table: `diffusemri.utils.pytorch_gradient_utils.PyTorchGradientTable`
    *   Underlying PyTorch Model: `diffusemri.models.noddi_model.NoddiModelTorch`
*   **Method:** Uses a PyTorch-based iterative optimization to fit the NODDI model. Can leverage GPU if available.
*   **Common Parameters Generated:** NDI (`f_intra`), ODI, Fiso (`f_iso`), mean neurite orientation (`mu_theta`, `mu_phi`), Watson concentration (`kappa`).
*   **Advanced Features:** Supports smart initialization of mean orientation and L1/L2 regularization via `fit_params`.
*   **Example Usage:**
    ```python
    import numpy as np
    import torch
    from diffusemri.fitting.noddi_fitter import fit_noddi_volume
    from diffusemri.utils.pytorch_gradient_utils import PyTorchGradientTable

    # Minimal multi-shell data for NODDI (e.g., 2 b0s, 8 dirs b=700, 8 dirs b=2000)
    num_b0, n_s1, n_s2 = 2, 8, 8
    total_vols_noddi = num_b0 + n_s1 + n_s2
    dwi_data_np = np.zeros((2,1,1, total_vols_noddi), dtype=np.float32) # Tiny volume
    dwi_data_np[...,:num_b0] = 2500
    dwi_data_np[...,num_b0:num_b0+n_s1] = 1200
    dwi_data_np[...,num_b0+n_s1:] = 500

    bvals_np = np.array([0]*num_b0 + [700]*n_s1 + [2000]*n_s2, dtype=float)
    bvecs_np = np.random.rand(total_vols_noddi, 3)
    bvecs_np[:num_b0,:] = 0
    bvecs_np[num_b0:] /= np.linalg.norm(bvecs_np[num_b0:], axis=1, keepdims=True)

    gtab_torch = PyTorchGradientTable(bvals_np, bvecs_np)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # noddi_maps = fit_noddi_volume(
    #     dwi_data=dwi_data_np,
    #     gtab=gtab_torch,
    #     device=device_str,
    #     fit_params={'n_iterations': 50} # Fewer iterations for quick example
    # )
    # ndi_map = noddi_maps.get("f_intra")
    # print(f"NODDI NDI map shape: {ndi_map.shape if 'ndi_map' in locals() else 'N/A (conceptual fit)'}")
    print("Note: NODDI example conceptual fit. See examples/10_NODDI_Fitting_and_Metrics.ipynb for runnable version.")
    ```
For detailed, runnable examples of fitting these models and visualizing their outputs, please refer to the Jupyter notebooks in the `examples/` directory.
