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
    from diffusemri.preprocessing.masking import create_brain_mask

    # Example: Create dummy 4D dMRI data (e.g., 10x10x10 volume, 5 gradients)
    dummy_dmri_data = np.random.rand(10, 10, 10, 5).astype(np.float32) * 2000
    dummy_voxel_size = (2.0, 2.0, 2.0)

    brain_mask, masked_data = create_brain_mask(
        dummy_dmri_data,
        dummy_voxel_size,
        median_radius=4,
        numpass=4
    )
    
    print(f"Brain mask shape: {brain_mask.shape}")
    print(f"Masked data shape: {masked_data.shape}")
    print(f"Number of voxels in mask: {np.sum(brain_mask)}")
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
    from diffusemri.preprocessing.denoising import denoise_mppca_data
    
    # Example: Create dummy 4D dMRI data
    dummy_dmri_data = np.random.rand(10, 10, 10, 20).astype(np.float32) * 100
    # Add some noise
    dummy_dmri_data += np.random.normal(0, 15, dummy_dmri_data.shape) 
    dummy_dmri_data = np.clip(dummy_dmri_data, 0, None)

    denoised_data = denoise_mppca_data(dummy_dmri_data, patch_radius=2)
    
    print(f"Denoised data shape: {denoised_data.shape}")
    print(f"Original data mean: {np.mean(dummy_dmri_data):.2f}, Denoised data mean: {np.mean(denoised_data):.2f}")
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
    from diffusemri.preprocessing.correction import correct_motion_eddy_fsl, load_eddy_outlier_report
    print("Successfully imported correct_motion_eddy_fsl and load_eddy_outlier_report.")
    
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
    print("Note: The main part of the motion/eddy correction example requires FSL installed and actual data paths.")
    ```

## Susceptibility Distortion Correction (FSL TOPUP)

*   **Purpose:** To correct for geometric distortions in dMRI data caused by susceptibility-induced magnetic field inhomogeneities. These distortions are particularly problematic in regions with tissue-air interfaces (e.g., near sinuses, ear canals) and are often characterized by stretching or compression of the image along phase-encoding directions. Correction typically requires images acquired with opposing phase-encoding directions (e.g., AP-PA pairs).
*   **Function:** `diffusemri.preprocessing.correction.correct_susceptibility_topup_fsl()`
*   **Method:** This function serves as a wrapper for FSL's `topup` and `applytopup` tools.
    *   `topup`: Estimates the susceptibility-induced off-resonance field from pairs of images acquired with opposing phase-encoding directions (typically b0 images).
    *   `applytopup`: Applies the estimated field correction to a target image series (which can be the same images used for `topup`, or a full DWI dataset that has similar PE direction as one of the `topup` inputs).
    It uses [Nipype](https://nipype.readthedocs.io/en/latest/) to interface with these FSL command-line tools.
*   **Dependencies:** This feature requires a local installation of FSL (with `topup` and `applytopup` available in the system PATH) and the `nipype` library. Users must ensure compliance with FSL's licensing terms.
*   **Key Parameters for `correct_susceptibility_topup_fsl`:**
    *   `imain_file` (str): Path to the 4D NIFTI file containing images for `topup` (e.g., concatenated AP and PA b0s).
    *   `encoding_file` (str): Path to the text file describing acquisition parameters for `topup` (PE direction, total readout time for each volume in `imain_file`).
    *   `out_base_name` (str): Base name for output files.
    *   `images_to_correct_file` (str): Path to the 4D NIFTI image series to which correction will be applied.
    *   `images_to_correct_encoding_indices` (list[int]): 1-based indices indicating which line(s) in `encoding_file` correspond to `images_to_correct_file`.
    *   `config_file` (str, optional): FSL `topup` configuration file (e.g., "b02b0.cnf").
    *   `output_type` (str, optional): FSL output type (e.g., "NIFTI_GZ").
    *   `topup_kwargs` (dict, optional): Additional arguments for the `TOPUP` Nipype interface.
    *   `applytopup_kwargs` (dict, optional): Additional arguments for the `ApplyTOPUP` Nipype interface.
*   **Returns:** Paths to the corrected image, field map, topup field coefficients, and movement parameters.

*   **Conceptual Example Usage:**
    ```python
    # from diffusemri.preprocessing.correction import correct_susceptibility_topup_fsl
    # print("Successfully imported correct_susceptibility_topup_fsl.") # Verify import

    # # --- This is a conceptual example. Actual paths and FSL installation are needed. ---
    # # imain_file: e.g., a NIFTI file with 2 volumes: first b0 with AP, second b0 with PA
    # topup_input_images_path = "path/to/your/ap_pa_b0s.nii.gz"
    # # encoding_file: text file describing acquisition parameters for topup_input_images_path
    # # Example content for 2 volumes (AP then PA):
    # # 0 1 0 0.065  (AP: y-direction phase encode, total readout 65ms)
    # # 0 -1 0 0.065 (PA: y-direction phase encode, total readout 65ms)
    # pe_encoding_file_path = "path/to/your/phase_encode_params.txt"
    # # images_to_correct_file: your full DWI dataset that needs correction
    # dwi_to_correct_path = "path/to/your/full_dwi.nii.gz"
    # # images_to_correct_encoding_indices: If full_dwi.nii.gz was acquired with AP phase encoding (like the 1st vol in topup_input_images_path)
    # # and your encoding_file has AP as the first line (index 1 for FSL tools).
    # # Assuming all volumes in dwi_to_correct_path share this PE scheme.
    # dwi_pe_indices = [1] # Use [1] if all volumes in dwi_to_correct_path match 1st line of pe_encoding_file_path

    # output_corrected_base = "path/to/output/dwi_corrected_topup"

    # try:
    #     corrected_img, field_map, field_coef, mov_params = correct_susceptibility_topup_fsl(
    #         imain_file=topup_input_images_path,
    #         encoding_file=pe_encoding_file_path,
    #         out_base_name=output_corrected_base,
    #         images_to_correct_file=dwi_to_correct_path,
    #         images_to_correct_encoding_indices=dwi_pe_indices,
    #         # config_file="b02b0.cnf", # Default
    #         # topup_kwargs={'fwhm': 8}, # Example topup kwarg
    #         # applytopup_kwargs={'method': 'jac'} # Default method
    #     )
    #     print(f"Susceptibility corrected image: {corrected_img}")
    #     print(f"Field map (Hz): {field_map}")
    #     print(f"Field coefficients: {field_coef}")
    #     print(f"Movement parameters: {mov_params}")
    # except RuntimeError as e:
    #     print(f"FSL TOPUP/ApplyTOPUP correction requires FSL installed and configured. Error: {e}")
    # except FileNotFoundError as e:
    #     print(f"One of the input files for TOPUP/ApplyTOPUP was not found: {e}")
    print("Note: The TOPUP/ApplyTOPUP example requires FSL installed, correctly configured input files, and the function imported.")
    ```

## Bias Field Correction (Dipý N4)

*   **Purpose:** To correct for low-frequency intensity non-uniformities (bias field) in MR images. Bias fields can arise from imperfections in the scanner hardware or subject-induced field disturbances, leading to spatially varying intensity levels that can affect segmentation, registration, and quantitative analysis.
*   **Function:** `diffusemri.preprocessing.correction.correct_bias_field_dipy()`
*   **Method:** This function wraps Dipý's `BiasFieldCorrectionFlow`, which uses the 'n4' method (N4ITK algorithm, a variant of N3) for bias field correction. N4 is widely used and effective for various types of MR images, including T1w, T2w, and mean b0 images from DWI.
*   **Dependencies:** This feature requires Dipý to be installed. The 'n4' method itself relies on ANTs/ITK being installed and accessible by Dipý.
*   **Key Parameters for `correct_bias_field_dipy`:**
    *   `input_image_file` (str): Path to the NIFTI file to be corrected.
    *   `output_corrected_file` (str): Path to save the corrected NIFTI file.
    *   `method` (str, optional): Correction method, defaults to 'n4'.
    *   `mask_file` (str, optional): Path to a brain mask. While N4 can use a mask (often improving results by focusing computation on relevant tissue), this wrapper notes that `BiasFieldCorrectionFlow` might not directly pass it for 'n4'. Users might need to provide a pre-masked input image if masked correction is desired.
    *   `**kwargs`: Additional arguments for Dipý's `BiasFieldCorrectionFlow.run()` (e.g., `threshold`, `use_cuda`, N4-specific parameters like `n_iters`, `convergence_threshold`).
*   **Returns:** Path to the corrected image file.

*   **Conceptual Example Usage:**
    ```python
    # from diffusemri.preprocessing.correction import correct_bias_field_dipy
    # print("Successfully imported correct_bias_field_dipy.") # Verify import

    # # --- This is a conceptual example. Actual paths and dependencies (Dipy, ANTs/ITK) are needed. ---
    # # input_image_file: e.g., a T1w image or a mean b0 image from DWI
    # image_to_correct_path = "path/to/your/image_needing_correction.nii.gz"
    # # optional_mask_path = "path/to/your/brain_mask.nii.gz" # Optional
    # output_bias_corrected_path = "path/to/output/image_corrected_n4.nii.gz"

    # try:
    #     corrected_file = correct_bias_field_dipy(
    #         input_image_file=image_to_correct_path,
    #         output_corrected_file=output_bias_corrected_path,
    #         # mask_file=optional_mask_path, # If providing a mask
    #         method='n4',
    #         # Example of passing kwargs for N4, check Dipy docs for specific BiasFieldCorrectionFlow options
    #         # n_iters="[50,50,30,20]", # Example: N4 iterations schedule
    #         # convergence_threshold=1e-6,
    #         verbose=True
    #     )
    #     print(f"Bias field corrected image saved to: {corrected_file}")
    # except RuntimeError as e:
    #     print(f"Dipý N4 bias field correction requires Dipy and underlying N4 (ANTs/ITK) to be installed and configured. Error: {e}")
    # except FileNotFoundError as e:
    #     print(f"One of the input files for N4 bias field correction was not found: {e}")
    print("Note: The N4 bias field correction example requires Dipý (and its N4 dependencies like ANTs/ITK) installed, input files, and the function imported.")
    ```

## Gibbs Ringing Correction (Dipý)

*   **Purpose:** To reduce Gibbs ringing artifacts, which are spurious oscillations typically seen near sharp intensity transitions (e.g., tissue interfaces) in MR images. These artifacts arise from the truncation of k-space data during image acquisition.
*   **Function:** `diffusemri.preprocessing.denoising.correct_gibbs_ringing_dipy()`
*   **Method:** This function wraps Dipý's `dipy.denoise.gibbs_removal` function. This algorithm works by identifying voxels likely affected by Gibbs ringing based on local total variation (TV) and then correcting their intensities.
*   **Dependencies:** This feature requires Dipý to be installed.
*   **Key Parameters for `correct_gibbs_ringing_dipy`:**
    *   `input_image_file` (str): Path to the NIFTI file to be corrected (can be 3D or 4D).
    *   `output_corrected_file` (str): Path to save the corrected NIFTI file.
    *   `slice_axis` (int, optional): Axis along which slices were acquired (0 for X, 1 for Y, 2 for Z). Default is 2.
    *   `n_points` (int, optional): Number of neighboring points for local TV calculation. Default is 3.
    *   `num_processes` (int, optional): Number of processes for parallel computation. Default is 1.
*   **Returns:** Path to the corrected image file.

*   **Conceptual Example Usage:**
    ```python
    # from diffusemri.preprocessing.denoising import correct_gibbs_ringing_dipy
    # print("Successfully imported correct_gibbs_ringing_dipy.") # Verify import

    # # --- This is a conceptual example. Actual paths and Dipý installation are needed. ---
    # # input_image_file: e.g., a T2w image or a DWI volume susceptible to ringing
    # image_with_ringing_path = "path/to/your/image_with_ringing.nii.gz"
    # output_unring_path = "path/to/output/image_unringed.nii.gz"

    # try:
    #     corrected_file = correct_gibbs_ringing_dipy(
    #         input_image_file=image_with_ringing_path,
    #         output_corrected_file=output_unring_path,
    #         slice_axis=2, # Assuming axial acquisition
    #         n_points=3,
    #         num_processes=None # Use None for auto-detection of CPUs by Dipy
    #     )
    #     print(f"Gibbs ringing corrected image saved to: {corrected_file}")
    # except RuntimeError as e:
    #     print(f"Dipý Gibbs ringing correction failed. Error: {e}")
    # except FileNotFoundError as e:
    #     print(f"One of the input files for Gibbs ringing correction was not found: {e}")
    print("Note: The Gibbs ringing correction example requires Dipý installed, input files, and the function imported.")
    ```

## DICOM to NIfTI Conversion

*   **Purpose:** To convert medical images from DICOM format (typically a series of 2D files) into the NIfTI format (often a single 3D or 4D file), which is commonly used in neuroimaging research. For diffusion-weighted imaging (DWI), this process also involves extracting b-values and b-vectors.
*   **CLI Tool:** `run_preprocessing dicom_to_nifti`
*   **Key Functions:**
    *   `diffusemri.data_io.dicom_utils.convert_dwi_dicom_to_nifti()`: For DWI data, extracts image data, b-values, b-vectors, and metadata, saving them as NIfTI, .bval, .bvec, and a JSON sidecar file.
    *   `diffusemri.data_io.dicom_utils.convert_dicom_to_nifti_main()`: For non-DWI data (e.g., anatomical T1w), converts DICOM series to NIfTI and saves basic metadata to a JSON sidecar.
*   **Method:** Reads DICOM files from a directory, sorts them, extracts pixel data and relevant metadata (including DWI-specific information if applicable), constructs an affine matrix, and saves the data in NIfTI format. B-values and b-vectors are saved to separate text files for DWI.
*   **Dependencies:** Requires `pydicom` for reading DICOM files and `nibabel` for NIfTI handling.
*   **Conceptual CLI Usage:**
    ```bash
    # For DWI data
    python cli/run_preprocessing.py dicom_to_nifti \\
        --input_dicom_dir /path/to/dwi_dicom_series \\
        --output_nifti_file /path/to/output/dwi.nii.gz \\
        --output_bval_file /path/to/output/dwi.bval \\
        --output_bvec_file /path/to/output/dwi.bvec \\
        --is_dwi

    # For non-DWI data (e.g., anatomical T1w)
    python cli/run_preprocessing.py dicom_to_nifti \\
        --input_dicom_dir /path/to/anatomical_dicom_series \\
        --output_nifti_file /path/to/output/t1w.nii.gz
        # Note: --is_dwi is not specified, so it defaults to False or non-DWI path
    ```

## DICOM Anonymization

*   **Purpose:** To de-identify DICOM files by removing or modifying specific patient-identifying information (PII) tags. This is crucial for sharing data while protecting patient privacy.
*   **CLI Tool:** `run_preprocessing anonymize_dicom`
*   **Key Functions:**
    *   `diffusemri.data_io.dicom_utils.anonymize_dicom_directory()`: Anonymizes all DICOM files within a specified directory, optionally preserving the directory structure.
    *   `diffusemri.data_io.dicom_utils.anonymize_dicom_file()`: Anonymizes a single DICOM file.
*   **Method:** The tool uses a default profile of common PII tags to either remove them or replace their values with generic placeholders (e.g., empty strings, "000000", "19000101"). Users can also provide a custom JSON file specifying their own rules for tag modification or removal. Essential image processing tags (like orientation, spacing, diffusion parameters) are generally preserved by default.
*   **Dependencies:** Requires `pydicom`.
*   **Note on Compliance:** This tool performs tag editing for de-identification. For strict compliance with regulations like HIPAA, ensure your anonymization rules and procedures are thoroughly vetted and meet all legal requirements. This tool provides a technical means for tag modification but does not guarantee legal compliance on its own.
*   **Conceptual CLI Usage:**
    ```bash
    # Anonymize a single DICOM file with default rules
    python cli/run_preprocessing.py anonymize_dicom \\
        --input_path /path/to/single.dcm \\
        --output_path /path/to/output/anonymized.dcm

    # Anonymize an entire DICOM directory with default rules, preserving structure
    python cli/run_preprocessing.py anonymize_dicom \\
        --input_path /path/to/dicom_directory \\
        --output_path /path/to/output_anonymized_directory \\
        --is_directory

    # Anonymize with custom rules from a JSON file
    # Example my_rules.json:
    # {
    #   "PatientName": "PatientX",
    #   "InstitutionName": "_REMOVE_TAG_",
    #   "(0x0010,0x0020)": "CustomPatientID"
    # }
    python cli/run_preprocessing.py anonymize_dicom \\
        --input_path /path/to/dicom_directory \\
        --output_path /path/to/output_anonymized_directory \\
        --is_directory \\
        --rules_json /path/to/my_rules.json
    ```
