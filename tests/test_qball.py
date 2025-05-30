import pytest
import numpy as np
from dipy.core.gradients import gradient_table, GradientTable
from dipy.data import get_sphere, Sphere
from dipy.sims.voxel import multi_tensor
from dipy.reconst.qball import QballFit # For type checking fit object
from diffusemri.models.qball import QballModel

# --- Fixture for Test Data ---

@pytest.fixture(scope="module")
def qball_data_gtab():
    """
    Generates synthetic 4D dMRI data and a Dipy gtab for Q-ball model testing.
    Uses a single shell at b=1000.
    """
    bvals = np.concatenate((np.zeros(6), np.ones(40) * 1000)) # Single shell b=1000
    bvecs_rand = np.random.randn(40, 3)
    bvecs_rand /= np.linalg.norm(bvecs_rand, axis=1)[:, None]
    bvecs = np.concatenate((np.zeros((6, 3)), bvecs_rand))
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)

    mevals = np.array([[0.0017, 0.0002, 0.0002], [0.0017, 0.0002, 0.0002]]) # Axially symmetric tensors
    angles = [(0, 0), (60, 0)] # Two fibers at 60 degrees
    fractions = [60, 40]
    signal, _ = multi_tensor(gtab, mevals, S0=100, angles=angles, fractions=fractions, snr=None)
    
    data_shape_spatial = (4, 4, 4) # Small spatial volume
    data = np.tile(signal, data_shape_spatial + (1,)).astype(np.float32)
    
    return gtab, data

class TestQballModel:
    def test_qballmodel_init(self, qball_data_gtab):
        gtab, _ = qball_data_gtab
        
        # Test with default smooth
        model_default_smooth = QballModel(gtab)
        assert model_default_smooth.gtab is gtab
        assert isinstance(model_default_smooth.gtab, GradientTable)
        assert model_default_smooth.smooth == 0.006 # Default value
        assert model_default_smooth._fit_object is None
        assert model_default_smooth._dipy_model_instance is None

        # Test with custom smooth
        custom_smooth_val = 0.01
        model_custom_smooth = QballModel(gtab, smooth=custom_smooth_val)
        assert model_custom_smooth.smooth == custom_smooth_val

    def test_qballmodel_init_invalid_gtab(self):
        with pytest.raises(ValueError, match="gtab must be an instance of Dipy's GradientTable."):
            QballModel(gtab="not_a_gtab_object")

    def test_qballmodel_fit(self, qball_data_gtab):
        gtab, data = qball_data_gtab
        
        # Test with default sh_order_max
        model_default_sh = QballModel(gtab)
        model_default_sh.fit(data)
        assert model_default_sh._dipy_model_instance is not None, "Dipy model not instantiated (default sh)."
        assert model_default_sh._fit_object is not None, "_fit_object should not be None after fitting (default sh)."
        assert isinstance(model_default_sh._fit_object, QballFit), \
            "_fit_object is not an instance of QballFit (default sh)."

        # Test with custom sh_order_max
        custom_sh_order = 6
        model_custom_sh = QballModel(gtab)
        model_custom_sh.fit(data, sh_order_max=custom_sh_order)
        assert model_custom_sh._dipy_model_instance is not None, "Dipy model not instantiated (custom sh)."
        assert model_custom_sh._fit_object is not None, "_fit_object should not be None after fitting (custom sh)."
        assert isinstance(model_custom_sh._fit_object, QballFit), \
            "_fit_object is not an instance of QballFit (custom sh)."
        # Check if sh_order was used (indirectly, via the model instance)
        assert model_custom_sh._dipy_model_instance.sh_order_max == custom_sh_order, \
             "Custom sh_order_max not set in Dipy's model instance."


    def test_qballmodel_fit_invalid_data(self, qball_data_gtab):
        gtab, data = qball_data_gtab
        model = QballModel(gtab)
        
        # Test with 3D data (should be 4D)
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit(data[..., 0])
        
        # Test with non-NumPy data
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit("not_numpy_data")

    def test_qballmodel_odf(self, qball_data_gtab):
        gtab, data = qball_data_gtab
        model = QballModel(gtab) # Use default smooth for this test
        
        default_sphere = get_sphere('repulsion724')
        custom_sphere = get_sphere('symmetric362') # A different sphere for testing

        # Test ODF calculation before fitting
        fresh_model_for_odf_test = QballModel(gtab)
        with pytest.raises(ValueError, match="Model has not been fitted yet. Call the 'fit' method first."):
            fresh_model_for_odf_test.odf(default_sphere)

        # Fit the model (using default sh_order_max)
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

    def test_qballmodel_odf_invalid_sphere(self, qball_data_gtab):
        gtab, data = qball_data_gtab
        model = QballModel(gtab)
        model.fit(data)
        
        # Dipy's QballFit.odf method expects a Sphere object.
        # Passing something else should raise an error (likely AttributeError or TypeError).
        with pytest.raises((AttributeError, TypeError)): 
            model.odf(sphere="not_a_sphere_object")

        # Test with a Sphere object that might be malformed or not what Dipy expects
        # (e.g. if it doesn't have 'vertices' attribute)
        class BadSphere:
            pass
        with pytest.raises((AttributeError, TypeError)):
            model.odf(sphere=BadSphere())
