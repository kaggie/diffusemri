# 01: Preprocessing Tools

This section details the tools available in the `diffusemri` library to help prepare your dMRI data for analysis. Effective preprocessing is crucial for accurate and reliable results from downstream modeling and tractography.

## Brain Masking

*   **Purpose:** To isolate brain tissue from non-brain areas (e.g., skull, CSF outside the brain, background) in dMRI images. This is crucial for accurate analysis as it restricts computations to relevant voxels, saves processing time, and can improve the performance of various algorithms.
*   **Python Function:** `diffusemri.preprocessing.masking.create_brain_mask()`
*   **CLI Tool:** `python cli/run_preprocessing.py create_mask ...`
*   **Method:** This function implements a brain masking algorithm. Internally, it computes the mean of the diffusion-weighted images (or uses b0 images if available and specified), applies a median filter (using `scipy.ndimage.median_filter`), and then performs Otsu's thresholding (using a PyTorch-based implementation) to create a binary brain mask.
*   **Key Parameters for `create_brain_mask`:**
    *   `dmri_data` (np.ndarray): The 4D dMRI data.
    *   `voxel_size` (tuple or list): Voxel dimensions.
    *   `median_radius` (int): Radius for the median filter.
    *   `numpass` (int): Number of passes for the median filter.
*   **Returns:** A 3D boolean NumPy array (the brain mask) and the 4D dMRI data multiplied by this mask.

*   **Conceptual Python Example:**
    ```python
    import numpy as np
    # from diffusemri.preprocessing.masking import create_brain_mask # Actual import

    # For demonstration, assume create_brain_mask is available
    # Create dummy 4D dMRI data (e.g., 10x10x10 volume, 5 gradients)
    # dummy_dmri_data = np.random.rand(10, 10, 10, 5).astype(np.float32) * 2000
    # dummy_voxel_size = (2.0, 2.0, 2.0)

    # brain_mask, masked_data = create_brain_mask(
    #     dummy_dmri_data,
    #     dummy_voxel_size,
    #     median_radius=4,
    #     numpass=4
    # )
    # print(f"Brain mask shape: {brain_mask.shape if 'brain_mask' in locals() else 'N/A'}")
    print("Note: Brain masking example is conceptual. See '06_Basic_Preprocessing.ipynb' for a runnable version.")
    ```
*   **Conceptual CLI Example:**
    ```bash
    # python cli/run_preprocessing.py create_mask \\
    #   --input_dwi /path/to/your/dwi.nii.gz \\
    #   --output_mask /path/to/output/brain_mask.nii.gz \\
    #   --output_masked_dwi /path/to/output/masked_dwi.nii.gz \\
    #   --median_radius 4 --numpass 4
    ```

## Noise Reduction

### MP-PCA Denoising
*   **Purpose:** To improve data quality by reducing random noise typically present in dMRI acquisitions.
*   **Python Function:** `diffusemri.preprocessing.denoising.denoise_mppca_data()`
*   **CLI Tool:** `python cli/run_preprocessing.py denoise_mppca ...`
*   **Method:** Implements the Marchenko-Pastur Principal Component Analysis (MP-PCA) algorithm (PyTorch-based).
*   **Key Parameters for `denoise_mppca_data`:**
    *   `dmri_data` (np.ndarray): The 4D dMRI data.
    *   `patch_radius` (int): Radius for local patches.
*   **Returns:** A 4D NumPy array (denoised dMRI data).

*   **Conceptual Python Example:**
    ```python
    import numpy as np
    # from diffusemri.preprocessing.denoising import denoise_mppca_data # Actual import

    # dummy_dmri_data = np.random.rand(10,10,10,20).astype(np.float32) * 100
    # denoised_data = denoise_mppca_data(dummy_dmri_data, patch_radius=2)
    # print(f"Denoised data shape: {denoised_data.shape if 'denoised_data' in locals() else 'N/A'}")
    print("Note: MP-PCA denoising example is conceptual. See '06_Basic_Preprocessing.ipynb' for a runnable version.")
    ```
*   **Conceptual CLI Example:**
    ```bash
    # python cli/run_preprocessing.py denoise_mppca \\
    #   --input_dwi /path/to/your/dwi.nii.gz \\
    #   --output_denoised_dwi /path/to/output/dwi_denoised_mppca.nii.gz \\
    #   --patch_radius 2
    ```

### Gibbs Ringing Correction (Dipý)
*   **Purpose:** To reduce Gibbs ringing artifacts near sharp intensity transitions.
*   **Python Function:** `diffusemri.preprocessing.denoising.correct_gibbs_ringing_dipy()`
*   **CLI Tool:** `python cli/run_preprocessing.py gibbs_ringing_dipy ...`
*   **Method:** Wraps Dipý's `dipy.denoise.gibbs_removal` function.
*   **Dependencies:** Requires Dipý.
*   **Key Parameters:** `input_image_file`, `output_corrected_file`, `slice_axis`, `n_points`, `num_processes`.
*   **Returns:** Path to the corrected image file.

*   **Conceptual Python Example:**
    ```python
    # from diffusemri.preprocessing.denoising import correct_gibbs_ringing_dipy # Actual import
    # image_with_ringing_path = "path/to/your/image_with_ringing.nii.gz"
    # output_unring_path = "path/to/output/image_unringed.nii.gz"
    # try:
    #     corrected_file = correct_gibbs_ringing_dipy(
    #         input_image_file=image_with_ringing_path,
    #         output_corrected_file=output_unring_path
    #     )
    #     print(f"Gibbs corrected image saved to: {corrected_file}")
    # except Exception as e:
    #     print(f"Error: {e}")
    print("Note: Gibbs ringing correction example is conceptual. See '06_Basic_Preprocessing.ipynb' for a runnable version.")
    ```
*   **Conceptual CLI Example:**
    ```bash
    # python cli/run_preprocessing.py gibbs_ringing_dipy \\
    #   --input_file /path/to/your/image_with_ringing.nii.gz \\
    #   --output_file /path/to/output/image_unringed.nii.gz \\
    #   --slice_axis 2
    ```

## Distortion Correction & Field Correction

### Motion and Eddy Current Correction (FSL `eddy`)
*   **Purpose:** Corrects distortions from subject motion and eddy currents. Rotates b-vectors accordingly.
*   **Python Function:** `diffusemri.preprocessing.correction.correct_motion_eddy_fsl()`
*   **CLI Tool:** `python cli/run_preprocessing.py correct_fsl ...`
*   **Method:** Wraps FSL's `eddy` tool via Nipype.
*   **Dependencies:** Requires FSL installation and Nipype.
*   **Key Parameters:** `dwi_file`, `bval_file`, `bvec_file`, `mask_file`, `index_file`, `acqp_file`, `out_base_name`, `use_cuda`, `**kwargs` for `eddy`.
*   **Returns:** Paths to corrected DWI, rotated b-vectors, and optional outlier reports.
*   **Outlier Parsing:** `diffusemri.preprocessing.correction.load_eddy_outlier_report()` can parse `eddy`'s outlier report.

*   **Conceptual Python Example:**
    ```python
    # from diffusemri.preprocessing.correction import correct_motion_eddy_fsl # Actual import
    # print("Conceptual call to correct_motion_eddy_fsl. Requires FSL and valid data.")
    # # dwi_nifti_path = "path/to/dwi.nii.gz"
    # # ... other paths ...
    # # output_base = "path/to/output/dwi_corrected_eddy"
    # # try:
    # #     corr_dwi, rot_bvecs, outlier_rep, _ = correct_motion_eddy_fsl(
    # #         dwi_file=dwi_nifti_path, ..., out_base_name=output_base, repol=True)
    # #     print(f"Corrected DWI: {corr_dwi}")
    # # except Exception as e: print(f"Error: {e}")
    print("Note: FSL Eddy example is conceptual. See '07_Advanced_Preprocessing_FSL.ipynb'.")
    ```
*   **Conceptual CLI Example:**
    ```bash
    # python cli/run_preprocessing.py correct_fsl \\
    #   --dwi_file path/to/dwi.nii.gz \\
    #   --bval_file path/to/bvals.bval \\
    #   --bvec_file path/to/bvecs.bvec \\
    #   --mask_file path/to/mask.nii.gz \\
    #   --index_file path/to/index.txt \\
    #   --acqp_file path/to/acqparams.txt \\
    #   --out_base_name path/to/output/eddy_corrected \\
    #   --repol True
    ```

### Susceptibility Distortion Correction (FSL `topup`)
*   **Purpose:** Corrects geometric distortions from susceptibility-induced field inhomogeneities using images with opposing phase-encoding directions.
*   **Python Function:** `diffusemri.preprocessing.correction.correct_susceptibility_topup_fsl()`
*   **CLI Tool:** `python cli/run_preprocessing.py topup_fsl ...`
*   **Method:** Wraps FSL's `topup` (field estimation) and `applytopup` (correction) via Nipype.
*   **Dependencies:** Requires FSL installation and Nipype.
*   **Key Parameters:** `imain_file` (e.g., AP/PA b0s), `encoding_file`, `out_base_name`, `images_to_correct_file`, `images_to_correct_encoding_indices` (0-indexed for Python wrapper).
*   **Returns:** Paths to corrected image, field map, field coefficients, movement parameters.

*   **Conceptual Python Example:**
    ```python
    # from diffusemri.preprocessing.correction import correct_susceptibility_topup_fsl # Actual import
    # print("Conceptual call to correct_susceptibility_topup_fsl. Requires FSL and valid data.")
    # # topup_input_images_path = "path/to/ap_pa_b0s.nii.gz"
    # # pe_encoding_file_path = "path/to/phase_encode_params.txt"
    # # dwi_to_correct_path = "path/to/full_dwi.nii.gz"
    # # dwi_pe_indices = [0] # 0-indexed for Python wrapper, e.g., if full_dwi matches 1st line of pe_encoding_file
    # # output_corrected_base = "path/to/output/dwi_corrected_topup"
    # # try:
    # #    corrected_img, _, _, _ = correct_susceptibility_topup_fsl(
    # #        imain_file=topup_input_images_path, encoding_file=pe_encoding_file_path,
    # #        out_base_name=output_corrected_base, images_to_correct_file=dwi_to_correct_path,
    # #        images_to_correct_encoding_indices=dwi_pe_indices)
    # #    print(f"Susceptibility corrected image: {corrected_img}")
    # # except Exception as e: print(f"Error: {e}")
    print("Note: FSL TOPUP example is conceptual. See '07_Advanced_Preprocessing_FSL.ipynb'.")
    ```
*   **Conceptual CLI Example:**
    ```bash
    # python cli/run_preprocessing.py topup_fsl \\
    #   --imain_file path/to/ap_pa_b0s.nii.gz \\
    #   --encoding_file path/to/phase_encode_params.txt \\
    #   --images_to_correct_file path/to/full_dwi.nii.gz \\
    #   --images_to_correct_encoding_indices 0 # Example: use first line of encoding file for all vols in images_to_correct_file
    #   --out_base_name path/to/output/topup_corrected
    ```

### Bias Field Correction (Dipý N4)
*   **Purpose:** Corrects low-frequency intensity non-uniformities (bias field).
*   **Python Function:** `diffusemri.preprocessing.correction.correct_bias_field_dipy()`
*   **CLI Tool:** `python cli/run_preprocessing.py bias_field_dipy ...`
*   **Method:** Wraps Dipý's `BiasFieldCorrectionFlow` using the 'n4' method (N4ITK algorithm).
*   **Dependencies:** Requires Dipý and its N4 dependencies (e.g., ANTs/ITK).
*   **Key Parameters:** `input_image_file`, `output_corrected_file`, `method` ('n4'), `mask_file` (optional).
*   **Returns:** Path to the corrected image file.

*   **Conceptual Python Example:**
    ```python
    # from diffusemri.preprocessing.correction import correct_bias_field_dipy # Actual import
    # print("Conceptual call to correct_bias_field_dipy. Requires Dipy/ANTs-ITK and valid data.")
    # # image_to_correct_path = "path/to/image_with_bias.nii.gz"
    # # output_bias_corrected_path = "path/to/output/image_corrected_n4.nii.gz"
    # # try:
    # #    corrected_file = correct_bias_field_dipy(input_image_file=image_to_correct_path,
    # #                                           output_corrected_file=output_bias_corrected_path)
    # #    print(f"Bias field corrected image: {corrected_file}")
    # # except Exception as e: print(f"Error: {e}")
    print("Note: N4 Bias Field example is conceptual. See '08_Bias_Field_Correction.ipynb' for a runnable version.")
    ```
*   **Conceptual CLI Example:**
    ```bash
    # python cli/run_preprocessing.py bias_field_dipy \\
    #   --input_file path/to/image_with_bias.nii.gz \\
    #   --output_file path/to/output/corrected_image_n4.nii.gz \\
    #   --method n4
    #   # Optional: --mask_file path/to/mask.nii.gz
    ```

## DICOM Utilities (Handled by `run_preprocessing` CLI)

The `run_preprocessing.py` script also includes subcommands for DICOM to NIfTI conversion and DICOM anonymization. For detailed Python API usage and examples of these utilities, please refer to the `01_DICOM_IO_and_Anonymization.ipynb` notebook and the [Format Conversion Wiki](wiki/07_Format_Conversion.md).

### DICOM to NIfTI Conversion
*   **Purpose:** Converts DICOM series to NIfTI, extracting b-values/b-vectors for DWI.
*   **CLI Tool:** `python cli/run_preprocessing.py dicom_to_nifti ...`
*   **Key Python Functions:** `data_io.dicom_utils.convert_dwi_dicom_to_nifti()`, `data_io.dicom_utils.convert_dicom_to_nifti_main()`.

*   **Conceptual CLI Example:**
    ```bash
    # For DWI data
    # python cli/run_preprocessing.py dicom_to_nifti \\
    #     --input_dicom_dir /path/to/dwi_dicom_series \\
    #     --output_nifti_file /path/to/output/dwi.nii.gz \\
    #     --output_bval_file /path/to/output/dwi.bval \\
    #     --output_bvec_file /path/to/output/dwi.bvec \\
    #     --is_dwi

    # For non-DWI data (e.g., anatomical T1w)
    # python cli/run_preprocessing.py dicom_to_nifti \\
    #     --input_dicom_dir /path/to/anatomical_dicom_series \\
    #     --output_nifti_file /path/to/output/t1w.nii.gz
    ```

### DICOM Anonymization
*   **Purpose:** De-identifies DICOM files by removing or modifying PII tags.
*   **CLI Tool:** `python cli/run_preprocessing.py anonymize_dicom ...`
*   **Key Python Functions:** `data_io.dicom_utils.anonymize_dicom_directory()`, `data_io.dicom_utils.anonymize_dicom_file()`.

*   **Conceptual CLI Example:**
    ```bash
    # Anonymize a DICOM directory with default rules
    # python cli/run_preprocessing.py anonymize_dicom \\
    #     --input_path /path/to/dicom_directory \\
    #     --output_path /path/to/output_anonymized_directory \\
    #     --is_directory

    # Anonymize with custom rules from a JSON file
    # python cli/run_preprocessing.py anonymize_dicom \\
    #     --input_path /path/to/dicom_directory \\
    #     --output_path /path/to/output_anonymized_directory \\
    #     --is_directory \\
    #     --rules_json /path/to/my_rules.json
    ```
*   **Note on Compliance:** Always ensure your anonymization procedures meet legal and ethical requirements. This tool provides technical means for tag modification.

For more detailed examples of all these preprocessing steps, please refer to the Jupyter notebooks in the `examples/` directory.
