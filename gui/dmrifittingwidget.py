from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QTextEdit, QGroupBox, QLineEdit, QApplication) # Added QApplication
import pyqtgraph as pg # Assuming pyqtgraph is available
import numpy as np

# Attempt relative imports for plugin structure
try:
    from ..data_io.load_dmri import load_nifti_dmri_data
    from ..fitting.dti_fitter import fit_dti_volume
    # Attempt to import models (DTI, QBall, MultiTissueCsdModel)
    from ..models.dti import DtiModel # Assuming DtiModel exists for consistency
    from ..models.qball import QballModel # Assuming QballModel exists
    from ..models.csd import MultiTissueCsdModel # Updated CSD model
    from dipy.core.gradients import GradientTable # For gtab creation
    from dipy.data import get_sphere # For ODF visualization sphere

except ImportError:
    # Fallback for direct script execution
    import sys
    import os
    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    from dmri_plugin.data_io.load_dmri import load_nifti_dmri_data
    from dmri_plugin.fitting.dti_fitter import fit_dti_volume # This is a direct function, not a class based model
    # For fallback, models would also need to be findable.
    # This might require adding the parent of `dmri_plugin` to sys.path if models are in `dmri_plugin.models`
    # For simplicity in this example, direct execution might have issues with model imports
    # unless the dmri_plugin is installed or paths are carefully managed.
    # Assuming DtiModel, QballModel, MultiTissueCsdModel are available through dmri_plugin.models
    from dmri_plugin.models.dti import DtiModel
    from dmri_plugin.models.qball import QballModel
    from dmri_plugin.models.csd import MultiTissueCsdModel
    from dipy.core.gradients import GradientTable
    from dipy.data import get_sphere


class DMRIFittingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("dMRI Fitting Controls")

        # Initialize data and results attributes
        self.image_data = None # This is the raw 4D diffusion data loaded from NIfTI
        self.b_values = None
        self.b_vectors = None
        self.gtab = None # GradientTable object
        self.data_volume = None # This will be self.image_data, renamed for clarity with model API
        
        self.fitted_model = None # Stores the instance of the fitted model (e.g., DtiModel, MultiTissueCsdModel)
        self.parameter_maps = None # For DTI legacy, stores dict of maps. For new models, this might be less used.
        self.sphere = get_sphere('repulsion724') # Default sphere for ODF visualizations
        self.current_view_type = None # 'scalar' or 'odf'

        main_layout = QVBoxLayout(self)

        # Data Loading Section
        data_group = QGroupBox("Load dMRI Data")
        data_layout = QVBoxLayout()
        # NIfTI file
        nifti_layout = QHBoxLayout()
        nifti_layout.addWidget(QLabel("NIfTI Image:"))
        self.nifti_path_edit = QLineEdit()
        self.nifti_path_edit.setPlaceholderText("Path to .nii or .nii.gz")
        nifti_layout.addWidget(self.nifti_path_edit)
        self.browse_nifti_button = QPushButton("Browse...")
        nifti_layout.addWidget(self.browse_nifti_button)
        data_layout.addLayout(nifti_layout)
        # bval file
        bval_layout = QHBoxLayout()
        bval_layout.addWidget(QLabel("bval File:"))
        self.bval_path_edit = QLineEdit()
        self.bval_path_edit.setPlaceholderText("Path to .bval")
        bval_layout.addWidget(self.bval_path_edit)
        self.browse_bval_button = QPushButton("Browse...")
        bval_layout.addWidget(self.browse_bval_button)
        data_layout.addLayout(bval_layout)
        # bvec file
        bvec_layout = QHBoxLayout()
        bvec_layout.addWidget(QLabel("bvec File:"))
        self.bvec_path_edit = QLineEdit()
        self.bvec_path_edit.setPlaceholderText("Path to .bvec")
        bvec_layout.addWidget(self.bvec_path_edit)
        self.browse_bvec_button = QPushButton("Browse...")
        bvec_layout.addWidget(self.browse_bvec_button)
        data_layout.addLayout(bvec_layout)
        
        self.load_dmri_button = QPushButton("Load dMRI Data") # This button will trigger the actual loading logic
        self.load_dmri_button = QPushButton("Load dMRI Data") 
        self.load_dmri_button.clicked.connect(self.handle_load_dmri_data) # Connect signal
        data_layout.addWidget(self.load_dmri_button)
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)

        # Model & Fitting Section
        fit_group = QGroupBox("Fitting")
        fit_layout = QVBoxLayout()
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Diffusion Model:"))
        self.model_combo = QComboBox()
        # Define available models - assuming DtiModel and QballModel classes exist
        self.models = {
            "DTI": DtiModel,  # Placeholder if DTI is also a class-based model
            "QBall": QballModel, # Placeholder
            "MT-CSD": MultiTissueCsdModel
        }
        self.model_combo.addItems(self.models.keys())
        self.model_combo.currentTextChanged.connect(self._on_model_changed) # Connect change handler
        model_select_layout.addWidget(self.model_combo)
        fit_layout.addLayout(model_select_layout)

        # MT-CSD Specific Controls (initially hidden)
        self.mt_csd_output_type_combo = QComboBox()
        self.mt_csd_output_type_combo.addItems(["WM fODFs", "GM Fraction", "CSF Fraction"])
        self.mt_csd_output_type_combo.setVisible(False) # Initially hidden
        self.mt_csd_output_type_combo.currentIndexChanged.connect(self.update_output_display) # Update display on change
        fit_layout.addWidget(self.mt_csd_output_type_combo) # Add to layout

        self.fit_button = QPushButton("Fit Volume")
        self.fit_button.clicked.connect(self.handle_fit_volume) 
        fit_layout.addWidget(self.fit_button)
        fit_group.setLayout(fit_layout)
        main_layout.addWidget(fit_group)

        # Results Visualization Section
        results_group = QGroupBox("Results Viewer")
        results_layout = QHBoxLayout()
        
        # Left panel for map selection (legacy DTI) or general controls
        display_controls_layout = QVBoxLayout()
        self.map_select_label = QLabel("Display Map (DTI):") # Label for DTI maps
        display_controls_layout.addWidget(self.map_select_label)
        self.map_combo = QComboBox() # For DTI parameter maps
        self.map_combo.currentIndexChanged.connect(self.handle_map_selection_changed)
        display_controls_layout.addWidget(self.map_combo)
        display_controls_layout.addStretch()
        results_layout.addLayout(display_controls_layout)

        # Unified Image View (can show scalar maps or ODFs if adapted)
        # For simplicity, let's assume pg.ImageView is for scalar maps.
        # ODFs might need a dedicated viewer or adaptation of this one.
        # For now, we'll use self.map_image_view for scalars and potentially clear/repurpose for ODFs.
        self.scalar_slicer_widget = pg.ImageView() # Renaming for clarity
        self.scalar_slicer_widget.ui.roiBtn.hide()
        self.scalar_slicer_widget.ui.menuBtn.hide()
        results_layout.addWidget(self.scalar_slicer_widget, stretch=1)
        
        # Placeholder for ODF Slicer - In a real app, this would be a proper ODF viewer
        # For this example, we'll simulate its presence with a simple widget or clear map_image_view
        # self.odf_slicer_widget = SomeODFViewerWidget() # Ideal
        # For now, let's assume we'll try to display ODFs textually or clear scalar view
        self.odf_display_status = QLabel("ODF Display Area (Conceptual)") # Placeholder
        # results_layout.addWidget(self.odf_display_status, stretch=1) # Add if using separate ODF area

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Initial visibility based on default model
        self._on_model_changed(self.model_combo.currentText())


        # Status Area
        status_group = QGroupBox("Status & Logs")
        status_layout = QVBoxLayout()
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Status messages and logs will appear here...")
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Connect browse buttons
        self.browse_nifti_button.clicked.connect(lambda: self._browse_file(self.nifti_path_edit, "NIfTI files (*.nii *.nii.gz)"))
        self.browse_bval_button.clicked.connect(lambda: self._browse_file(self.bval_path_edit, "bval files (*.bval)"))
        self.browse_bvec_button.clicked.connect(lambda: self._browse_file(self.bvec_path_edit, "bvec files (*.bvec)"))

    def _browse_file(self, path_edit_widget, file_filter):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if filepath:
            path_edit_widget.setText(filepath)

    def _on_model_changed(self, model_name):
        # Show/hide MT-CSD specific controls
        is_mt_csd = (model_name == "MT-CSD")
        self.mt_csd_output_type_combo.setVisible(is_mt_csd)
        
        # Show/hide DTI specific controls (like the map_combo for FA, MD etc.)
        is_dti_legacy = (model_name == "DTI_legacy_fit") # Assuming old DTI fit is different
        # If DTI model is class-based and also outputs to self.fitted_model, this logic might change.
        # For now, assume map_combo is for the old DTI style.
        self.map_select_label.setVisible(not is_mt_csd) # Hide if MT-CSD, show for others
        self.map_combo.setVisible(not is_mt_csd)       # Hide if MT-CSD, show for others
        
        if not is_mt_csd: # If not MT-CSD, clear its specific displays potentially
            # self.odf_slicer_widget.clear() # If you have a dedicated ODF viewer
            pass # Scalar slicer will be updated by other models or cleared

        self.update_output_display() # Update display based on new model (e.g. clear if no fit)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    # Ensure QApplication instance exists for standalone testing
    if QApplication.instance() is None:
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    main_widget = DMRIFittingWidget()
    main_widget.setGeometry(100, 100, 800, 700) 
    main_widget.show()
    
    # For standalone test, QFileDialog might need QApplication.exec_() if not run from main app.
    # However, this test just shows the widget.
    if not hasattr(app, '_already_running_for_test'): # Basic check to avoid recursive exec_
        app._already_running_for_test = True
        # sys.exit(app.exec_()) # Only if this is the main entry point and not imported

    # New methods to be added below this line in the class

    def handle_load_dmri_data(self):
        nifti_fp = self.nifti_path_edit.text()
        bval_fp = self.bval_path_edit.text()
        bvec_fp = self.bvec_path_edit.text()

        if not all([nifti_fp, bval_fp, bvec_fp]):
            self.status_text.append("Error: NIfTI, bval, and bvec file paths must all be specified.")
            return

        self.status_text.setText(f"Loading dMRI data...\n  NIfTI: {nifti_fp}\n  bval: {bval_fp}\n  bvec: {bvec_fp}")
        QApplication.processEvents()

        try:
            loaded_image_data, b_vals, b_vecs = load_nifti_dmri_data(nifti_fp, bval_fp, bvec_fp)
            if loaded_image_data is not None:
                self.image_data = loaded_image_data
                self.data_volume = self.image_data # Use data_volume for model fitting
                self.b_values = b_vals
                self.b_vectors = b_vecs
                # Create GradientTable
                self.gtab = GradientTable(self.b_values, self.b_vectors, b0_threshold=50) # Assuming a b0_threshold
                
                self.fitted_model = None # Clear previous model and results
                self.parameter_maps = None 
                self.map_combo.clear()
                self.scalar_slicer_widget.clear()
                # self.odf_slicer_widget.clear() # If separate ODF viewer
                self.status_text.append("dMRI data loaded successfully.")
                self.status_text.append(f"  Image shape: {self.data_volume.shape}")
                self.status_text.append(f"  b-values count: {len(self.b_values)}")
                self.status_text.append(f"  b-vectors shape: {self.b_vectors.shape}")
            else:
                # Clear all data attributes
                self._clear_loaded_data()
                self.status_text.append("Error loading dMRI data. Check paths/formats. See console for details.")
        except Exception as e:
            self._clear_loaded_data()
            self.status_text.append(f"An unexpected error occurred during data loading: {e}")

    def _clear_loaded_data(self):
        self.image_data = None
        self.data_volume = None
        self.b_values = None
        self.b_vectors = None
        self.gtab = None
        self.fitted_model = None
        self.parameter_maps = None
        self.map_combo.clear()
        self.scalar_slicer_widget.clear()
        # self.odf_slicer_widget.clear() # If separate ODF viewer

    def handle_fit_volume(self):
        if self.data_volume is None or self.gtab is None:
            self.status_text.append("Error: dMRI data not loaded or gtab not created. Please load data first.")
            return

        selected_model_name = self.model_combo.currentText()
        ModelClass = self.models.get(selected_model_name)

        if not ModelClass:
            self.status_text.append(f"Error: Model '{selected_model_name}' is not recognized.")
            return

        self.status_text.append(f"Starting {selected_model_name} model fitting...")
        QApplication.processEvents()

        try:
            self.fitted_model = ModelClass(self.gtab) # Instantiate the model
            # Special handling for DTI legacy fit_dti_volume if it's not a class based model
            if selected_model_name == "DTI_legacy_fit": # Example: if DTI is still function-based
                 self.parameter_maps = fit_dti_volume(self.data_volume, self.b_values, self.b_vectors)
                 self.fitted_model = None # No class instance for this legacy one
                 self.status_text.append("DTI (legacy) fitting complete.")
                 self.populate_map_selector() # For DTI maps
                 if self.map_combo.count() > 0:
                     self.map_combo.setCurrentIndex(0)
                     self.handle_map_selection_changed()
            else:
                # For class-based models like MultiTissueCsdModel, QballModel, DtiModel (if class)
                self.fitted_model.fit(self.data_volume) # Call the fit method of the model instance
                self.status_text.append(f"{selected_model_name} fitting complete.")
                # Clear old DTI maps if a new model is fit
                self.parameter_maps = None 
                self.map_combo.clear()
            
            self.update_output_display() # Update display for the newly fitted model

        except Exception as e:
            self.fitted_model = None
            self.parameter_maps = None
            self.map_combo.clear()
            self.scalar_slicer_widget.clear()
            # self.odf_slicer_widget.clear()
            self.status_text.append(f"An error occurred during {selected_model_name} fitting: {e}")


    def update_output_display(self):
        self.scalar_slicer_widget.clear()
        # self.odf_slicer_widget.clear() # If you have a separate ODF viewer
        # self.odf_display_status.setText("ODF Display Area (Conceptual)") # Reset ODF placeholder

        if self.fitted_model is None:
            # This can happen if fitting failed or no model is fit yet, or for DTI legacy.
            # If parameter_maps exist (from DTI legacy), handle_map_selection_changed will manage display.
            if self.parameter_maps:
                self.handle_map_selection_changed() # Try to show DTI map if available
            else:
                self.scalar_slicer_widget.clear()
            return

        model_name = self.model_combo.currentText() # Get current selected model in combo

        if isinstance(self.fitted_model, MultiTissueCsdModel) and model_name == "MT-CSD":
            selected_output = self.mt_csd_output_type_combo.currentText()
            self.status_text.append(f"Updating display for MT-CSD: {selected_output}")
            try:
                if selected_output == "WM fODFs":
                    # This is tricky with pg.ImageView. ODFs are 4D (x,y,z, N_coeffs/vertices).
                    # A real ODF slicer is needed. For now, we can't display them in scalar_slicer_widget.
                    # We could try to calculate a scalar derived from ODFs (e.g., peak FA) or show placeholder.
                    wm_odfs = self.fitted_model.wm_odf(sphere=self.sphere) # Shape (x,y,z, N_sphere_vertices)
                    # For now, let's just log that we have them.
                    self.status_text.append(f"  WM fODFs generated, shape: {wm_odfs.shape}. (Display not implemented in scalar view)")
                    # self.odf_slicer_widget.set_odf_data(wm_odfs, sphere=self.sphere) # Ideal
                    self.scalar_slicer_widget.clear() # Clear scalar display
                    self.current_view_type = 'odf'
                elif selected_output == "GM Fraction":
                    gm_map = self.fitted_model.gm_fraction() # Should be 3D
                    if gm_map.ndim == 3:
                        self.scalar_slicer_widget.setImage(np.transpose(gm_map, (2,0,1)), autoRange=True, autoLevels=True)
                        self.status_text.append("  Displaying GM Volume Fraction.")
                    else:
                         self.status_text.append(f"  GM Fraction map is not 3D, shape: {gm_map.shape}")
                    # self.odf_slicer_widget.clear()
                    self.current_view_type = 'scalar'
                elif selected_output == "CSF Fraction":
                    csf_map = self.fitted_model.csf_fraction() # Should be 3D
                    if csf_map.ndim == 3:
                        self.scalar_slicer_widget.setImage(np.transpose(csf_map, (2,0,1)), autoRange=True, autoLevels=True)
                        self.status_text.append("  Displaying CSF Volume Fraction.")
                    else:
                        self.status_text.append(f"  CSF Fraction map is not 3D, shape: {csf_map.shape}")
                    # self.odf_slicer_widget.clear()
                    self.current_view_type = 'scalar'
            except Exception as e:
                self.status_text.append(f"Error generating MT-CSD output '{selected_output}': {e}")
        
        elif isinstance(self.fitted_model, QballModel) and model_name == "QBall":
            # Example: QBall might produce GFA or ODFs
            try:
                gfa_map = self.fitted_model.gfa # Assuming QballModel has a .gfa property (3D map)
                if gfa_map.ndim == 3:
                    self.scalar_slicer_widget.setImage(np.transpose(gfa_map, (2,0,1)), autoRange=True, autoLevels=True)
                    self.status_text.append("Displaying GFA from QBall model.")
                else:
                    self.status_text.append(f"QBall GFA map is not 3D, shape: {gfa_map.shape}")
                self.current_view_type = 'scalar'
            except Exception as e:
                self.status_text.append(f"Error getting GFA from QBall model: {e}")

        elif isinstance(self.fitted_model, DtiModel) and model_name == "DTI":
            # Example: DTI class model might output FA
            try:
                fa_map = self.fitted_model.fa # Assuming DtiModel has a .fa property (3D map)
                if fa_map.ndim == 3:
                    self.scalar_slicer_widget.setImage(np.transpose(fa_map, (2,0,1)), autoRange=True, autoLevels=True)
                    self.status_text.append("Displaying FA from DTI model.")
                else:
                     self.status_text.append(f"DTI FA map is not 3D, shape: {fa_map.shape}")
                self.current_view_type = 'scalar'
            except Exception as e:
                self.status_text.append(f"Error getting FA from DTI model: {e}")
        else:
            # This case handles if the model is selected but not yet fit, or if it's a model type
            # not specifically handled above for direct property display.
            # DTI legacy maps are handled by populate_map_selector and handle_map_selection_changed.
            if not self.parameter_maps: # If no DTI legacy maps either
                 self.scalar_slicer_widget.clear()
                 self.status_text.append("Select a model and fit data, or select a DTI map if available.")


    def populate_map_selector(self): # Used for DTI legacy fit_dti_volume results
        self.map_combo.clear()
        if self.parameter_maps is not None: # self.parameter_maps is for DTI legacy
            for map_name in self.parameter_maps.keys():
                map_data = self.parameter_maps[map_name]
                if isinstance(map_data, np.ndarray) and map_data.ndim == 3:
                    self.map_combo.addItem(map_name)
            if self.map_combo.count() == 0:
                 self.status_text.append("No 3D scalar maps found in DTI legacy fitting results.")


    def handle_map_selection_changed(self): # Used for DTI legacy fit_dti_volume results
        # This function is primarily for results from the old DTI fitting (fit_dti_volume)
        # which stores results in self.parameter_maps.
        # For new class-based models, update_output_display is the primary way.
        if self.parameter_maps is None or self.map_combo.count() == 0:
            # Only clear if we are not currently viewing a class-based model output
            if self.fitted_model is None: # i.e. no class-based model is active
                self.scalar_slicer_widget.clear()
            return

        selected_map_name = self.map_combo.currentText()
        if not selected_map_name:
            if self.fitted_model is None: self.scalar_slicer_widget.clear()
            return

        if selected_map_name in self.parameter_maps:
            map_data_3d = self.parameter_maps[selected_map_name]
            if map_data_3d is not None and isinstance(map_data_3d, np.ndarray) and map_data_3d.ndim == 3:
                try:
                    transposed_map = np.transpose(map_data_3d, (2, 0, 1))
                    self.scalar_slicer_widget.setImage(transposed_map, autoRange=True, autoLevels=True)
                    self.status_text.append(f"Displaying DTI map: {selected_map_name}")
                    self.current_view_type = 'scalar'
                    # If a DTI legacy map is selected, make sure MT-CSD specific UI is hidden
                    # and fitted_model is None or a DTI legacy context is set.
                    # This part might need refinement based on how DTI (legacy) vs DTI (class) is handled.
                    if self.model_combo.currentText() != "MT-CSD": # A bit of a safety check
                        self.mt_csd_output_type_combo.setVisible(False)
                except Exception as e:
                    self.scalar_slicer_widget.clear()
                    self.status_text.append(f"Error displaying DTI map '{selected_map_name}': {e}")
            else:
                if self.fitted_model is None: self.scalar_slicer_widget.clear()
                self.status_text.append(f"Cannot display DTI map '{selected_map_name}'. Not 3D or is None.")
        else:
            if self.fitted_model is None: self.scalar_slicer_widget.clear()
            self.status_text.append(f"Selected DTI map '{selected_map_name}' not found.")
```

