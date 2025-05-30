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
    # from diffusemri.fitting.dti_fitter import fit_dti_volume # Conceptual import
    # from dipy.core.gradients import gradient_table # For gtab creation

    # # --- This is a conceptual example ---
    # # Assume dwi_data_np (4D), bvals_np, bvecs_np are loaded
    # # gtab = gradient_table(bvals_np, bvecs_np)
    
    # # dti_params_dict = fit_dti_volume(dwi_data_np, gtab.bvals, gtab.bvecs)
    # # fa_map = dti_params_dict["FA"]
    # # md_map = dti_params_dict["MD"]
    
    print("Note: To run DTI example, ensure 'fit_dti_volume' and data are correctly set up.")
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
    # import numpy as np
    # from diffusemri.models import DkiModel # Conceptual import
    # from dipy.core.gradients import gradient_table

    # # --- Conceptual example ---
    # # Assume dwi_data_np (4D multi-shell), bvals_np, bvecs_np are loaded
    # # gtab = gradient_table(bvals_np, bvecs_np) # Ensure gtab has multiple b-values > 0
    
    # # dki_model_instance = DkiModel(gtab)
    # # dki_fit_result = dki_model_instance.fit(dwi_data_np) 
    # # mk_map = dki_fit_result.mk() # Example, actual method might vary
    
    print("Note: To run DKI example, ensure 'DkiModel' and multi-shell data are correctly set up.")
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
    # import numpy as np
    # from diffusemri.models import CsdModel # Conceptual import
    # from dipy.core.gradients import gradient_table

    # # --- Conceptual example ---
    # # Assume dwi_data_np (4D HARDI), bvals_np, bvecs_np are loaded
    # # gtab = gradient_table(bvals_np, bvecs_np)
    
    # # csd_model_instance = CsdModel(gtab)
    # # csd_model_instance.fit(dwi_data_np) # Fitting might involve response func estimation
    # # odf_sh_coeffs = csd_model_instance.shm_coeff # Example attribute for SH coeffs
    # # gfa_map = csd_model_instance.gfa
    
    print("Note: To run CSD example, ensure 'CsdModel' and HARDI data are correctly set up.")
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
    # import numpy as np
    # from diffusemri.models import QballModel # Conceptual import
    # from dipy.core.gradients import gradient_table

    # # --- Conceptual example ---
    # # Assume dwi_data_np (4D single-shell HARDI), bvals_np, bvecs_np are loaded
    # # gtab = gradient_table(bvals_np, bvecs_np)
    
    # # qball_model_instance = QballModel(gtab, sh_order=6) # Example sh_order
    # # qball_model_instance.fit(dwi_data_np)
    # # odf_sh_coeffs_qball = qball_model_instance.shm_coeff
    
    print("Note: To run QBI example, ensure 'QballModel' and HARDI data are correctly set up.")
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
    # from diffusemri.fitting.noddi_fitter import fit_noddi_volume # Conceptual
    # from diffusemri.utils.pytorch_gradient_utils import PyTorchGradientTable # Conceptual

    # # --- Conceptual example ---
    # # Assume dwi_data_np (4D multi-shell), bvals_np, bvecs_np are loaded
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # # dwi_data_torch = torch.from_numpy(dwi_data_np).float().to(device) # Not needed for fit_noddi_volume input
    # # gtab_torch = PyTorchGradientTable(bvals_np, bvecs_np) # Using the PyTorch gradient table

    # # # Basic fitting
    # # noddi_maps = fit_noddi_volume(
    # #     dwi_data=dwi_data_np, # fit_noddi_volume takes np array for dwi_data
    # #     gtab=gtab_torch,      # but PyTorch gtab
    # #     device=device # Pass device string to fit_noddi_volume
    # # )
    # # ndi_map = noddi_maps["f_intra"]
    # # odi_map = noddi_maps["odi"]
    
    print("Note: To run NODDI example, ensure 'fit_noddi_volume', 'PyTorchGradientTable', and data are correctly set up.")
    ```
*   **Example Usage (NODDI with Smart Initialization and Regularization):**
    ```python
    # # --- Conceptual example continued ---
    # # Assume dti_v1_map (X,Y,Z,3) from DTI primary eigenvectors is available as np.ndarray
    # # initial_orientation_map_np = dti_v1_map 
    
    # # advanced_fit_params = {
    # #     'learning_rate': 0.01,
    # #     'n_iterations': 750,
    # #     'l1_penalty_weight': 0.005,
    # #     'l1_params_to_regularize': ['f_iso'], # Example: L1 on isotropic fraction
    # #     'l2_penalty_weight': 0.001
    # # }
    
    # # noddi_maps_advanced = fit_noddi_volume(
    # #     dwi_data=dwi_data_np,
    # #     gtab=gtab_torch,
    # #     initial_orientation_map=initial_orientation_map_np, # Provide DTI V1s
    # #     fit_params=advanced_fit_params,
    # #     device=device
    # # )
    # # ndi_map_adv = noddi_maps_advanced["f_intra"]
    # # odi_map_adv = noddi_maps_advanced["odi"]
    # # fiso_map_adv = noddi_maps_advanced["f_iso"]

    print("Note: Advanced NODDI fitting shows conceptual parameters.")
    ```
