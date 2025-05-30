# Dipy Function Usage in This Library

This document tracks the usage of functions from the Dipy library. The goal is to gradually replace these with native PyTorch implementations to improve independence and stability.

## Identified Dipy Functions and Their Locations:

1.  **`dipy.core.gradients.gradient_table`**
    *   **Purpose:** Used for representing and accessing b-values, b-vectors, and identifying b0 images.
    *   **Original Dipy Location:** `fitting/noddi_fitter.py` and `models/noddi_model.py`.
    *   **Status: Replaced.**
    *   **PyTorch Replacement:** The `NoddiModelTorch` class (`models/noddi_model.py`) and relevant functions in `fitting/noddi_fitter.py` now utilize `utils.pytorch_gradient_utils.PyTorchGradientTable`. This class manages b-values and b-vectors as PyTorch tensors and provides related properties like `b0s_mask`.

2.  **`dipy.denoise.localpca.localpca`**
    *   **Purpose:** Used for Marchenko-Pastur Principal Component Analysis (MP-PCA) denoising of dMRI data.
    *   **Original Dipy Location:** `preprocessing/denoising.py` (function `denoise_mppca_data`).
    *   **Status: Replaced.**
    *   **PyTorch Replacement:** The `denoise_mppca_data` function now utilizes `preprocessing.pytorch_denoising.pytorch_mppca`. This PyTorch version implements a patch-based MP-PCA algorithm that includes:
        *   Local patch extraction and aggregation.
        *   Per-patch data centering, SVD.
        *   Noise variance estimation from singular values (using `median(S^2)/sqrt(P/Ng)`).
        *   Marchenko-Pastur eigenvalue threshold calculation (using `sigma_est^2 * (1 + sqrt(P/Ng))^2`).
        *   Hard thresholding of singular values for denoising.
        *   This implementation is based on an interpretation of RMT principles for this context. Further validation against specific literature models (e.g., Veraart et al., 2016) could refine performance.

3.  **`dipy.segment.mask.median_otsu`**
    *   **Purpose:** Used for creating a brain mask from dMRI data by applying a median filter and Otsu's thresholding.
    *   **Original Dipy Location:** `preprocessing/masking.py` (function `create_brain_mask`).
    *   **Status: Replaced.**
    *   **PyTorch/SciPy Replacement:** The `create_brain_mask` function now utilizes `preprocessing.pytorch_masking.pytorch_median_otsu`. This function internally uses `scipy.ndimage.median_filter` for the median filtering step and a PyTorch-based implementation for Otsu's thresholding.

4.  **`dipy.direction.peaks_from_model`**
    *   **Purpose:** Extracts peaks from an Orientation Distribution Function (ODF) model.
    *   **Original Dipy Location:** `tracking/deterministic.py` (previously used directly via a Dipy ODF model object).
    *   **Status: Replaced (Core logic for CSD-based peak extraction).**
    *   **PyTorch Replacement:** The `tracking.deterministic.track_deterministic_oudf` function no longer directly calls `dipy.direction.peaks_from_model`. Instead, it now takes raw DWI data and a Dipy `GradientTable` and uses the new PyTorch-based `models.pytorch_csd_peaks.PeaksFromModel` class internally to perform CSD-like SH fitting and peak extraction. This new class handles ODF calculation and peak finding in PyTorch.
    *   **Note on Dipy Dependencies:** The `PeaksFromModel` PyTorch class itself currently still utilizes `dipy.core.gradients.GradientTable` as an input type for gradient information and `dipy.data.get_sphere` internally for acquiring spherical sampling schemes.

5.  **`dipy.tracking.local_tracking.LocalTracking`**
    *   **Purpose:** Core algorithm for generating streamlines using a local tracking approach.
    *   **Original Dipy Location:** `tracking/deterministic.py` (previously used directly).
    *   **Status: Replaced (Core tracking logic).**
    *   **PyTorch Replacement:** The core step-by-step tracking logic previously handled by `dipy.tracking.local_tracking.LocalTracking` within `tracking.deterministic.track_deterministic_oudf` has been replaced by a new PyTorch-based implementation in `tracking.pytorch_local_tracking.LocalTracking`. This includes the main tracking loop, direction selection, and application of constraints like step size, max crossing angle, and length filters.

6.  **`dipy.tracking.streamline.Streamlines`**
    *   **Purpose:** Data structure for representing and managing collections of streamlines.
    *   **Location:** `tracking/deterministic.py` (used as the output format of `track_deterministic_oudf`).
    *   **Status:** Retained (Dipy dependent, output format).
    *   **Note:** While the internal tracking is now PyTorch-based, the `track_deterministic_oudf` function currently wraps the PyTorch-generated streamlines (list of NumPy arrays) into a `dipy.tracking.streamline.Streamlines` object before returning, to maintain compatibility with downstream Dipy-based tools or workflows.

7.  **`dipy.tracking.stopping_criterion.StoppingCriterion` (and `BinaryStoppingCriterion`)**
    *   **Purpose:** Defines the conditions under which streamline generation should terminate. `BinaryStoppingCriterion` is a specific implementation.
    *   **Original Dipy Location:** `tracking/deterministic.py` (previously accepted as a direct parameter).
    *   **Status: Partially Replaced (Internal usage).**
    *   **PyTorch Replacement/Adaptation:** The `tracking.deterministic.track_deterministic_oudf` function no longer accepts Dipy `StoppingCriterion` objects directly. Instead, it now takes parameters like a metric map and a threshold value, and internally uses PyTorch-based stopping criteria (e.g., `tracking.pytorch_local_tracking.PyTorchThresholdStoppingCriterion`). Dipy's `StoppingCriterion` might still be used as a reference for expected behavior or for components not yet fully transitioned.

8.  **`dipy.tracking.utils.seeds_from_mask`**
    *   **Purpose:** Generates seed points for tractography from a binary mask.
    *   **Location:**
        *   `tracking/deterministic.py`: Imported and used in `track_deterministic_oudf` to generate seeds if a mask is provided.
    *   **Status: Retained (Dipy dependent). Refer to the 'Tracking Module Dependencies' section for details.**

9.  **`dipy.data.Sphere` (and potentially `get_sphere`)**
    *   **Purpose:** `Sphere` is used for representing spherical sampling schemes, necessary for ODF-related calculations. `get_sphere` is a utility to fetch standard sphere definitions.
    *   **Location:** Internally used by `models.pytorch_csd_peaks.PeaksFromModel` (via `dipy.data.get_sphere`). The `track_deterministic_oudf` function no longer takes a `Sphere` object as a direct argument.
    *   **Status: Retained (Dipy dependent). Refer to the 'Tracking Module Dependencies' section for details.**
    *   **Note:** This Dipy utility is crucial for the internal ODF representation and peak extraction process within the PyTorch-based `PeaksFromModel`.

---
*This document will be updated as replacements are implemented or new usages are identified.*
## Tracking Module Dependencies (`tracking/deterministic.py`)

The `tracking/deterministic.py` module handles streamline generation. While the core tracking loop and stopping criteria are now PyTorch-based, several Dipy components are still utilized for pre-processing steps or for their established utility:

*   **CSD ODF Modeling & Peak Extraction:** The core logic for fitting a CSD-like model to DWI data and extracting ODF peaks has been replaced by the PyTorch-based `models.pytorch_csd_peaks.PeaksFromModel`. However, this class itself still uses:
    *   `dipy.core.gradients.GradientTable`: As an input type for gradient information.
    *   `dipy.data.get_sphere`: Internally, for acquiring spherical sampling schemes necessary for ODF calculations.
*   **Streamline Output:** `dipy.tracking.streamline.Streamlines` is used as the final output format for compatibility.
*   **Seed Generation:** `dipy.tracking.utils.seeds_from_mask` is used if seeds are generated from a mask.
*   **Type Hinting:** `dipy.core.gradients.GradientTable` is used for type hinting the `gtab` input in `track_deterministic_oudf`.


**Status and Future Work:**

Replacing these tracking-related Dipy functionalities with native PyTorch implementations would be a **very significant research and development effort**. It would involve reimplementing complex algorithms for fiber orientation modeling, streamline generation, and related utilities.

For the foreseeable future, these Dipy dependencies within the `tracking` module will remain. A future project could potentially undertake a phased replacement of these components if there's a strong need for a completely Dipy-independent tracking module. This would likely start with simpler utilities and progressively address the more complex algorithmic parts.

---

*This note clarifies the status of Dipy dependencies in the tracking module.*
