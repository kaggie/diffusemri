# Future Development and Potential Features

This file lists potential features and improvements that could be added to the `diffusemri` project in the future.

## Core Algorithm Enhancements
*   **GPU Acceleration for MT-CSD**: Porting parts of the MT-CSD computation (especially fitting and ODF reconstruction) to GPU using libraries like CuPy could significantly reduce processing time for large datasets.
*   **Support for Other Multi-Tissue CSD Algorithms**: Investigate and potentially implement alternative multi-tissue CSD algorithms (e.g., SS3T-CSD) to provide users with more modeling choices.
*   **Advanced Response Function Estimation**: Explore or implement more advanced techniques for response function estimation, potentially with automatic outlier rejection or user-guided selection.

## Visualization and User Interface
*   **Advanced Multi-Tissue Visualization**:
    *   Implement direct visualization of WM fODFs in the GUI slicer views (e.g., rendering glyphs on slices) instead of just logging their generation.
    *   Allow overlaying tissue fraction maps with fODFs.
    *   Provide options for adjusting transparency and color maps for different tissue compartments.
*   **Quantitative Output Export**: Add functionality to easily export scalar maps (GM/CSF fractions, DTI metrics, etc.) in common medical imaging formats (e.g., NIfTI).
*   **Batch Processing UI**: Develop a user interface for setting up and running batch processing of multiple datasets.

## Tractography
*   **Multi-Tissue Tractography**: Integrate tractography algorithms that can leverage the multi-tissue information from MT-CSD (e.g., by using WM fODFs and respecting GM/CSF boundaries).
*   **Probabilistic Tractography for MT-CSD**: Extend tracking capabilities to include probabilistic approaches suitable for MT-CSD outputs.

## Validation and Quality Control
*   **Cross-Validation Tools**: Implement tools for k-fold cross-validation or similar techniques to assess model fit and robustness.
*   **Phantom Data Generation/Analysis**: Include utilities for generating synthetic phantom data with known ground truth for validation of MT-CSD and other models.
*   **Quality Metrics Display**: Calculate and display quality metrics for the fitted models (e.g., goodness-of-fit, residual error maps).

## General
*   **Plugin Architecture**: Refactor the codebase to support a plugin architecture, making it easier to add new models, algorithms, or visualization tools.
*   **Comprehensive Documentation**: Further expand user and developer documentation, including more tutorials and API references.
