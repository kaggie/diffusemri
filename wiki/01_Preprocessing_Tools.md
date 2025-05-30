# 01: Preprocessing Tools

This section details the tools available in the `diffusemri` library to help prepare your dMRI data for analysis. Effective preprocessing is crucial for accurate and reliable results from downstream modeling and tractography.

## Brain Masking

*   **Purpose:** To isolate brain tissue from non-brain areas (e.g., skull, CSF outside the brain, background) in dMRI images. This is crucial for accurate analysis as it restricts computations to relevant voxels, saves processing time, and can improve the performance of various algorithms.
*   **Function:** `diffusemri.preprocessing.masking.create_brain_mask()`
*   **Method:** This function implements a brain masking algorithm. Internally, it computes the mean of the diffusion-weighted images (or uses b0 images if available and specified), applies a median filter (using `scipy.ndimage.median_filter`), and then performs Otsu's thresholding (using a PyTorch-based implementation) to create a binary brain mask. This approach is effective for generating masks from DWI data.
*   **Key Parameters for `create_brain_mask`:**
    *   `dmri_data` (np.ndarray): The 4D dMRI data.
    *   `voxel_size` (tuple or list): Voxel dimensions (currently not used by the core median_otsu logic but kept for API consistency).
    *   `median_radius` (int): Radius for the median filter.
    *   `numpass` (int): Number of passes for the median filter.
*   **Returns:** A 3D boolean NumPy array (the brain mask) and the 4D dMRI data multiplied by this mask.

*   **Example Usage:**
    ```python
    import numpy as np
    # from diffusemri.preprocessing.masking import create_brain_mask # Assuming correct import path

    # --- Assuming create_brain_mask is in a place accessible like this for example ---
    # This is a placeholder for where the actual function would be imported from
    # For the purpose of this documentation, we'll assume it's available.
    # In the actual library, it might be:
    # from dmri_library_name.preprocessing.masking import create_brain_mask
    
    # Example: Create dummy 4D dMRI data (e.g., 10x10x10 volume, 5 gradients)
    dummy_dmri_data = np.random.rand(10, 10, 10, 5).astype(np.float32) * 2000
    dummy_voxel_size = (2.0, 2.0, 2.0)

    # brain_mask, masked_data = create_brain_mask(
    #     dummy_dmri_data, 
    #     dummy_voxel_size, 
    #     median_radius=4, 
    #     numpass=4
    # )
    
    # print(f"Brain mask shape: {brain_mask.shape}")
    # print(f"Masked data shape: {masked_data.shape}")
    # print(f"Number of voxels in mask: {np.sum(brain_mask)}")
    print("Note: To run the brain masking example, ensure 'create_brain_mask' is correctly imported from the library.")
    ```

## Noise Reduction

*   **Purpose:** To improve data quality by reducing random noise typically present in dMRI acquisitions. Noise can significantly affect the accuracy of model fitting, parameter estimation, and tractography.
*   **Function:** `diffusemri.preprocessing.denoising.denoise_mppca_data()`
*   **Method:** This function implements the Marchenko-Pastur Principal Component Analysis (MP-PCA) algorithm for noise reduction. The core logic, found in `diffusemri.preprocessing.pytorch_denoising.pytorch_mppca`, is PyTorch-based and operates on local patches from the dMRI data. For each patch, it performs PCA and uses Random Matrix Theory (RMT) principles to estimate noise and threshold singular values, thereby separating signal from noise components. This method is effective even without prior knowledge of the noise level.
*   **Key Parameters for `denoise_mppca_data`:**
    *   `dmri_data` (np.ndarray): The 4D dMRI data.
    *   `patch_radius` (int): Radius for the local patches used in PCA (e.g., radius 2 means a 5x5x5 patch).
*   **Returns:** A 4D NumPy array containing the denoised dMRI data.

*   **Example Usage:**
    ```python
    import numpy as np
    # from diffusemri.preprocessing.denoising import denoise_mppca_data # Assuming correct import path
    
    # Example: Create dummy 4D dMRI data
    dummy_dmri_data = np.random.rand(10, 10, 10, 20).astype(np.float32) * 100
    # Add some noise
    dummy_dmri_data += np.random.normal(0, 15, dummy_dmri_data.shape) 
    dummy_dmri_data = np.clip(dummy_dmri_data, 0, None)

    # denoised_data = denoise_mppca_data(dummy_dmri_data, patch_radius=2)
    
    # print(f"Denoised data shape: {denoised_data.shape}")
    # print(f"Original data mean: {np.mean(dummy_dmri_data):.2f}, Denoised data mean: {np.mean(denoised_data):.2f}")
    print("Note: To run the noise reduction example, ensure 'denoise_mppca_data' is correctly imported from the library.")
    ```

## Motion and Eddy Current Correction

*   **Purpose:** To correct for distortions in dMRI data caused by subject motion during the scan and by eddy currents induced by strong diffusion gradients. These corrections are critical for improving data quality, the accuracy of fitted diffusion models, and the reliability of tractography. This step also ensures that b-vectors are correctly rotated to align with the corrected image data.
*   **Function:** `diffusemri.preprocessing.correction.correct_motion_eddy_fsl()`
*   **Method:** This function serves as a wrapper for FSL's powerful `eddy` tool (typically `eddy_openmp` or `eddy_cuda`). It uses [Nipype](https://nipype.readthedocs.io/en/latest/) to interface with the FSL command-line tool. `eddy` is capable of modeling and correcting complex distortions, handling multi-shell data, and detecting/replacing outlier slices (e.g., by passing `repol=True` in `**kwargs` to the function). The wrapper returns paths to the corrected DWI image, the rotated b-vectors, and optionally outlier reports.
*   **Dependencies:** This feature requires a local installation of FSL (with `eddy` available in the system PATH) and the `nipype` library. Users must ensure compliance with FSL's licensing terms.
*   **Key Parameters for `correct_motion_eddy_fsl`:**
    *   `dwi_file` (str): Path to the input 4D DWI NIfTI file.
    *   `bval_file` (str): Path to the b-values file.
    *   `bvec_file` (str): Path to the b-vectors file.
    *   `mask_file` (str): Path to a brain mask NIfTI file.
    *   `index_file` (str): Path to the FSL index file (specifies acquisition parameters for each volume).
    *   `acqp_file` (str): Path to the FSL acquisition parameters file (encodes phase-encoding, readout time).
    *   `out_base_name` (str): Base name for output files.
    *   `use_cuda` (bool): Whether to attempt using `eddy_cuda`.
    *   `**kwargs`: Additional arguments to pass to FSL `eddy` (e.g., `repol=True`, `cnr_maps=True`).
*   **Returns:** Paths to the corrected DWI file, rotated b-vectors file, and optional outlier report files.
*   **Outlier Information:** A helper function `diffusemri.preprocessing.correction.load_eddy_outlier_report()` is available to parse the textual outlier report produced by `eddy`.

*   **Conceptual Example Usage:**
    ```python
    # from diffusemri.preprocessing.correction import correct_motion_eddy_fsl, load_eddy_outlier_report
    
    # # --- This is a conceptual example. Actual paths and FSL installation are needed. ---
    # dwi_nifti_path = "path/to/your/dwi.nii.gz"
    # bvals_path = "path/to/your/bvals.bval" # Or .txt
    # bvecs_path = "path/to/your/bvecs.bvec" # Or .txt
    # brain_mask_path = "path/to/your/brain_mask.nii.gz"
    # # Create index_file: text file, one line per DWI volume, number indicates row in acqp_file
    # # e.g., if 64 volumes all use 1st acqp line: 64 lines of "1"
    # index_file_path = "path/to/your/index.txt" 
    # # Create acqp_file: text file, e.g., "0 -1 0 0.062" (for AP phase-encode, total readout 62ms)
    # acqp_file_path = "path/to/your/acqp.txt" 
    # output_base = "path/to/output/dwi_corrected_by_fsl"

    # try:
    #     corr_dwi, rot_bvecs, outlier_rep, _ = correct_motion_eddy_fsl(
    #         dwi_file=dwi_nifti_path,
    #         bval_file=bvals_path,
    #         bvec_file=bvecs_path,
    #         mask_file=brain_mask_path,
    #         index_file=index_file_path,
    #         acqp_file=acqp_file_path,
    #         out_base_name=output_base,
    #         repol=True, # Example of passing an eddy argument
    #         cnr_maps=True 
    #     )
    #     print(f"Corrected DWI: {corr_dwi}")
    #     print(f"Rotated b-vectors: {rot_bvecs}")
    #     if outlier_rep:
    #         print(f"Outlier report: {outlier_rep}")
    #         # outlier_summary = load_eddy_outlier_report(outlier_rep)
    #         # print(f"Parsed outlier summary (first few entries): {list(outlier_summary.items())[:2]}")
    # except RuntimeError as e:
    #     print(f"FSL eddy correction requires FSL installed and configured. Error: {e}")
    # except FileNotFoundError as e:
    #     print(f"One of the input files was not found: {e}")
    print("Note: To run the motion/eddy correction example, ensure FSL is installed, input files exist, and function is correctly imported.")
    ```
