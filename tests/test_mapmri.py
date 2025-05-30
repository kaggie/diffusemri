import pytest
import numpy as np
from dipy.core.gradients import gradient_table, GradientTable
from dipy.data import get_sphere, Sphere
from dipy.sims.voxel import multi_tensor
from dipy.reconst.mapmri import MapmriFit # For type checking fit object
from diffusemri.models.mapmri import MapmriModel

# --- Fixture for Test Data ---

@pytest.fixture(scope="module")
def mapmri_data_gtab():
    """
    Generates synthetic 4D dMRI data and a Dipy gtab for MAP-MRI model testing.
    Uses multi-shell data.
    """
    # Using 3 shells as suggested by Dipy examples for MAPMRI
    bvals = np.concatenate((np.zeros(10), 
                            np.ones(30) * 700, 
                            np.ones(30) * 1000, 
                            np.ones(30) * 2000))
    N = len(bvals) - 10 # Number of non-b0 volumes
    
    # Generate random b-vectors for the shells
    bvecs_shells = np.random.randn(N, 3)
    bvecs_shells /= np.linalg.norm(bvecs_shells, axis=1)[:, None]
    bvecs = np.concatenate((np.zeros((10, 3)), bvecs_shells)) # Add b0s
    
    gtab = gradient_table(bvals, bvecs, b0_threshold=50)

    # Simulate data for a single voxel with some anisotropy
    mevals = np.array([[0.0015, 0.0004, 0.0004]]) # Anisotropic tensor
    signal, _ = multi_tensor(gtab, mevals, S0=100, angles=[(0,0)], fractions=[100], snr=None)
    
    data_shape_spatial = (3, 3, 3) # Small spatial volume for speed
    data = np.tile(signal, data_shape_spatial + (1,)).astype(np.float32)
    
    return gtab, data

class TestMapmriModel:
    def test_mapmrimodel_init(self, mapmri_data_gtab):
        gtab, _ = mapmri_data_gtab
        
        # Test with default parameters
        model_default = MapmriModel(gtab)
        assert model_default.gtab is gtab
        assert isinstance(model_default.gtab, GradientTable)
        assert model_default.radial_order == 6
        assert model_default.laplacian_regularization is True
        assert model_default.laplacian_weighting == 0.2
        assert model_default._dipy_model_instance is None
        assert model_default._fit_object is None

        # Test with custom parameters
        custom_radial_order = 4
        custom_lap_weight = 0.1
        model_custom = MapmriModel(gtab, radial_order=custom_radial_order, laplacian_weighting=custom_lap_weight)
        assert model_custom.radial_order == custom_radial_order
        assert model_custom.laplacian_weighting == custom_lap_weight

    def test_mapmrimodel_init_invalid_gtab(self):
        with pytest.raises(ValueError, match="gtab must be an instance of Dipy's GradientTable."):
            MapmriModel(gtab="not_a_gtab_object")

    def test_mapmrimodel_fit(self, mapmri_data_gtab):
        gtab, data = mapmri_data_gtab
        model = MapmriModel(gtab, radial_order=4) # Lower radial_order for faster test fit
        
        model.fit(data)
        assert model._dipy_model_instance is not None, "Dipy model instance not created."
        assert model._fit_object is not None, "_fit_object should not be None after fitting."
        assert isinstance(model._fit_object, MapmriFit), "_fit_object is not an instance of MapmriFit."

    def test_mapmrimodel_fit_invalid_data(self, mapmri_data_gtab):
        gtab, data = mapmri_data_gtab
        model = MapmriModel(gtab)
        
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit(data[..., 0]) # 3D data
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit("not_numpy_data")

    def _test_scalar_parameter_property(self, model_instance, data_shape_spatial, property_name, 
                                        non_negativity_check=False, range_0_1_check=False):
        """Helper for testing scalar MAP-MRI parameter properties."""
        # Test access before fit
        fresh_model = MapmriModel(model_instance.gtab, radial_order=4) # Use a fresh model
        with pytest.raises(ValueError, match="Model has not been fitted yet. Call the 'fit' method first."):
            getattr(fresh_model, property_name)

        # Access after fit (using the already fitted model_instance)
        param_map = getattr(model_instance, property_name)
        
        assert isinstance(param_map, np.ndarray), f"{property_name} map is not a NumPy array."
        assert param_map.shape == data_shape_spatial, \
            f"{property_name} map shape is {param_map.shape}, expected {data_shape_spatial}."
        assert np.issubdtype(param_map.dtype, np.floating), \
            f"{property_name} map dtype is {param_map.dtype}, expected a float type."

        valid_values = param_map[~np.isnan(param_map)]
        if valid_values.size > 0:
            if non_negativity_check:
                assert np.all(valid_values >= -1e-6), f"Values for {property_name} should be non-negative." # Allow for float inaccuracies
            if range_0_1_check:
                assert np.all((valid_values >= -1e-6) & (valid_values <= 1 + 1e-6)), \
                    f"Values for {property_name} should be in range [0, 1]."
        # else:
        #     print(f"Warning: All values for {property_name} are NaN. Check data/fit.")


    def test_mapmri_parameters_after_fit(self, mapmri_data_gtab):
        gtab, data = mapmri_data_gtab
        model = MapmriModel(gtab, radial_order=4) # Lower radial order for faster test
        model.fit(data)
        data_shape_spatial = data.shape[:-1]

        # Test each scalar parameter
        self._test_scalar_parameter_property(model, data_shape_spatial, "rtop", non_negativity_check=True) # RTOP is probability like, usually 0-1
        self._test_scalar_parameter_property(model, data_shape_spatial, "rtap", non_negativity_check=True) # RTAP is probability like, usually 0-1
        self._test_scalar_parameter_property(model, data_shape_spatial, "rtpp", non_negativity_check=True) # RTPP is probability like, usually 0-1
        self._test_scalar_parameter_property(model, data_shape_spatial, "msd", non_negativity_check=True)
        self._test_scalar_parameter_property(model, data_shape_spatial, "qiv", non_negativity_check=True)
        self._test_scalar_parameter_property(model, data_shape_spatial, "ng", non_negativity_check=True) # NG can be positive or negative, but often positive in tissue
        self._test_scalar_parameter_property(model, data_shape_spatial, "gfa", range_0_1_check=True)


    def test_mapmrimodel_odf(self, mapmri_data_gtab):
        gtab, data = mapmri_data_gtab
        model = MapmriModel(gtab, radial_order=4) # Lower radial order for faster test
        
        default_sphere = get_sphere('repulsion724')
        custom_sphere = get_sphere('symmetric362')

        # Test ODF calculation before fitting
        fresh_model_for_odf_test = MapmriModel(gtab, radial_order=4)
        with pytest.raises(ValueError, match="Model has not been fitted yet. Call the 'fit' method first."):
            fresh_model_for_odf_test.odf(default_sphere)

        # Fit the model
        model.fit(data)

        # Test with default sphere (passed as None to odf method)
        odfs_default_sphere = model.odf() 
        assert isinstance(odfs_default_sphere, np.ndarray)
        assert odfs_default_sphere.shape == data.shape[:-1] + (default_sphere.vertices.shape[0],)
        assert np.issubdtype(odfs_default_sphere.dtype, np.floating)
        np.testing.assert_array_compare(lambda x,y: np.all(x >= y), odfs_default_sphere, -1e-6,
                                        err_msg="ODF values with default sphere should be non-negative.")

        # Test with a custom sphere
        odfs_custom_sphere = model.odf(sphere=custom_sphere)
        assert isinstance(odfs_custom_sphere, np.ndarray)
        assert odfs_custom_sphere.shape == data.shape[:-1] + (custom_sphere.vertices.shape[0],)
        assert np.issubdtype(odfs_custom_sphere.dtype, np.floating)
        np.testing.assert_array_compare(lambda x,y: np.all(x >= y), odfs_custom_sphere, -1e-6,
                                         err_msg="ODF values on custom sphere should be non-negative.")

    def test_mapmrimodel_odf_invalid_sphere(self, mapmri_data_gtab):
        gtab, data = mapmri_data_gtab
        model = MapmriModel(gtab, radial_order=4)
        model.fit(data)
        
        with pytest.raises((AttributeError, TypeError)): 
            model.odf(sphere="not_a_sphere_object")
        
        class BadSphere: pass
        with pytest.raises((AttributeError, TypeError)):
            model.odf(sphere=BadSphere())
