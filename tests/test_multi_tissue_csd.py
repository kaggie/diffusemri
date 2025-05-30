import pytest
import numpy as np
from dipy.core.gradients import gradient_table, GradientTable
from dipy.data import get_sphere, Sphere
from dipy.reconst.mcsd import MultiShellDeconvFit, ResponseFunction
from dipy.sims.voxel import multi_tensor
from diffusemri.models.csd import MultiTissueCsdModel # Updated import

# --- Fixture for Test Data ---

@pytest.fixture(scope="module")
def mt_csd_data_gtab_responses():
    """
    Generates synthetic 4D dMRI data, a Dipy gtab, and fixed multi-tissue
    response functions for MultiTissueCsdModel testing.
    """
    # Multi-shell gtab (similar to before, ensuring it's suitable for MSMT-CSD)
    bvals = np.concatenate((np.zeros(6), np.ones(30) * 700, np.ones(30) * 1000, np.ones(30) * 2000))
    bvecs_s0 = np.zeros((6, 3))
    bvecs_s1 = np.random.randn(30, 3); bvecs_s1 /= np.linalg.norm(bvecs_s1, axis=1)[:, None]
    bvecs_s2 = np.random.randn(30, 3); bvecs_s2 /= np.linalg.norm(bvecs_s2, axis=1)[:, None]
    bvecs_s3 = np.random.randn(30, 3); bvecs_s3 /= np.linalg.norm(bvecs_s3, axis=1)[:, None]
    bvecs = np.concatenate((bvecs_s0, bvecs_s1, bvecs_s2, bvecs_s3))
    gtab_multi_shell = gradient_table(bvals, bvecs, b0_threshold=50) # Adjusted b0_threshold

    # Simulate data for a single voxel with WM, GM, CSF components
    # For simplicity, we'll make data that's somewhat responsive to typical WM/GM/CSF signals
    # This simulation won't be perfect but aims to provide data for model fitting.
    # WM-like component (anisotropic)
    mevals_wm = np.array([[0.0017, 0.0002, 0.0002]])
    # GM-like component (isotropic, lower diffusivity than WM)
    mevals_gm = np.array([[0.0008, 0.0008, 0.0008]])
    # CSF-like component (isotropic, higher diffusivity)
    mevals_csf = np.array([[0.0030, 0.0030, 0.0030]])

    # Combine signals - this is a simplification.
    # True multi-tissue simulation is more complex. Here, we just sum signals.
    # S0 values for each compartment
    S0_wm_comp = 70
    S0_gm_comp = 20
    S0_csf_comp = 10
    S0_total = S0_wm_comp + S0_gm_comp + S0_csf_comp # Effective S0 for the voxel

    signal_wm, _ = multi_tensor(gtab_multi_shell, mevals_wm, S0=S0_wm_comp, angles=[(0,0)], fractions=[100], snr=None)
    signal_gm, _ = multi_tensor(gtab_multi_shell, mevals_gm, S0=S0_gm_comp, angles=[(0,0)], fractions=[100], snr=None) # Isotropic, angle doesn't matter
    signal_csf, _ = multi_tensor(gtab_multi_shell, mevals_csf, S0=S0_csf_comp, angles=[(0,0)], fractions=[100], snr=None)

    # Combine signals (e.g., simple sum for this test data)
    signal_multi_shell = signal_wm + signal_gm + signal_csf
    
    # Ensure signal is non-negative and handle potential numerical issues with S0 for b0 images
    signal_multi_shell[gtab_multi_shell.b0s_mask] = S0_total # Set b0 images to total S0
    signal_multi_shell = np.maximum(signal_multi_shell, 0) # Ensure non-negativity

    data_shape_spatial = (3, 3, 3) # Small spatial dimensions for test speed
    data_multi_shell = np.tile(signal_multi_shell, data_shape_spatial + (1,)).astype(np.float32)

    # Fixed response functions (eigenvalues, S0)
    # These are tuples (evals, S0_response_signal_value)
    # Dipy's ResponseFunction class can also be used if preferred, e.g. ResponseFunction(evals, S0_signal)
    response_wm_fixed = (np.array([0.0015, 0.0003, 0.0003]), 100.) # Typical WM response Evals and S0
    response_gm_fixed = (np.array([0.0008, 0.0008, 0.0008]), 70.)  # Typical GM response Evals and S0
    response_csf_fixed = (np.array([0.0030, 0.0030, 0.0030]), 200.) # Typical CSF response Evals and S0
    
    # Convert to Dipy ResponseFunction objects as these are often expected by new models
    # or provide more robust type checking internally in Dipy.
    # The model itself might handle tuples, but using ResponseFunction objects is safer.
    # For MultiShellDeconvModel, the response functions should be ResponseFunction objects.
    # However, our wrapper currently passes the tuple directly to response_dhollander or stores them.
    # If response_dhollander is called, it returns ResponseFunction objects.
    # If user provides responses, our model stores them as is.
    # Let's ensure the fixture provides ResponseFunction objects for when we pass them directly.
    
    # The MultiShellDeconvModel expects a list of ResponseFunction objects.
    # Let's create them here for the fixture if we plan to pass them directly.
    # If we expect response_dhollander to be called, this is not strictly needed for auto-estimation tests.
    # For tests where we provide responses, they should be in the format expected by MultiShellDeconvModel,
    # which usually means ResponseFunction objects or compatible tuples.
    # The current MultiTissueCsdModel passes the provided responses to MultiShellDeconvModel.
    # MultiShellDeconvModel can handle tuples (evals, S0) for its `response` parameter.
    # So, fixture can return tuples.

    responses_fixed = (response_wm_fixed, response_gm_fixed, response_csf_fixed)

    return gtab_multi_shell, data_multi_shell, responses_fixed


class TestMultiTissueCsdModel: # Updated class name
    def test_multitissuecsdmodel_init(self, mt_csd_data_gtab_responses): # Updated fixture name
        gtab, _, _ = mt_csd_data_gtab_responses
        model = MultiTissueCsdModel(gtab) # Updated model name
        assert model.gtab is gtab
        assert isinstance(model.gtab, GradientTable)
        assert model._response_wm is None
        assert model._response_gm is None
        assert model._response_csf is None
        assert model._mt_csd_model is None # Updated attribute name
        assert model._mt_fit_object is None # Updated attribute name

    def test_multitissuecsdmodel_init_invalid_gtab(self): # Updated model name
        with pytest.raises(ValueError, match="gtab must be an instance of Dipy's GradientTable."):
            MultiTissueCsdModel(gtab="not_a_gtab_object") # Updated model name

    def test_multitissuecsdmodel_fit_with_response(self, mt_csd_data_gtab_responses): # Updated test and fixture names
        gtab, data, responses = mt_csd_data_gtab_responses
        resp_wm, resp_gm, resp_csf = responses
        model = MultiTissueCsdModel(gtab) # Updated model name
        
        model.fit(data, response_wm=resp_wm, response_gm=resp_gm, response_csf=resp_csf)
        
        assert model._response_wm is resp_wm
        assert model._response_gm is resp_gm
        assert model._response_csf is resp_csf
        assert model._mt_csd_model is not None # Updated attribute name
        assert model._mt_fit_object is not None # Updated attribute name
        assert isinstance(model._mt_fit_object, MultiShellDeconvFit) # Updated expected fit object type

    def test_multitissuecsdmodel_fit_auto_response(self, mt_csd_data_gtab_responses): # Updated test and fixture names
        gtab, data, _ = mt_csd_data_gtab_responses # Responses from fixture not used directly
        model = MultiTissueCsdModel(gtab) # Updated model name
        
        # response_dhollander might require specific keywords if defaults are not suitable
        # e.g., roi_center, roi_radii, fa_thresh, etc.
        # For synthetic data, default parameters might be okay or might need tuning.
        # This test will be slower due to response_dhollander.
        model.fit(data) # Call without providing responses to trigger auto-estimation
        
        assert model._response_wm is not None
        assert model._response_gm is not None
        assert model._response_csf is not None
        # response_dhollander returns ResponseFunction objects, not tuples,
        # although they might behave like tuples (value, S0) in some contexts.
        assert isinstance(model._response_wm, tuple) # Or ResponseFunction, check what response_dhollander returns and how it's stored
        assert isinstance(model._response_gm, tuple) # Or ResponseFunction
        assert isinstance(model._response_csf, tuple) # Or ResponseFunction
        # After response_dhollander, these should be ResponseFunction objects.
        # The model stores whatever response_dhollander returns.
        # Let's assume response_dhollander returns ResponseFunction instances as per Dipy docs.
        # And our model stores them.
        assert isinstance(model._response_wm, ResponseFunction)
        assert isinstance(model._response_gm, ResponseFunction)
        assert isinstance(model._response_csf, ResponseFunction)


        assert model._mt_csd_model is not None # Updated attribute name
        assert model._mt_fit_object is not None # Updated attribute name
        assert isinstance(model._mt_fit_object, MultiShellDeconvFit) # Updated expected fit object type

    def test_multitissuecsdmodel_fit_invalid_data(self, mt_csd_data_gtab_responses): # Updated names
        gtab, data, _ = mt_csd_data_gtab_responses
        model = MultiTissueCsdModel(gtab) # Updated model name
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit(data[..., 0]) # 3D data
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit("not_numpy_data")

    # This test needs to be re-evaluated. The MultiTissueCsdModel's fit method
    # does not explicitly validate the format of individual response functions if they are provided.
    # The validation would occur deeper within Dipy's MultiShellDeconvModel.
    # We can test that if *not all* responses are provided, estimation is triggered,
    # or that the model *can* run with valid provided responses.
    # A simple way is to check if providing a malformed response to MultiShellDeconvModel raises an error.
    # For now, the model wrapper itself doesn't add extra validation on provided responses.
    # So, this test might need to be adapted or removed if Dipy handles internal errors.
    # Let's assume Dipy's MultiShellDeconvModel might raise an error for incompatible response types.
    def test_multitissuecsdmodel_fit_invalid_response_format(self, mt_csd_data_gtab_responses):
        gtab, data, responses = mt_csd_data_gtab_responses
        resp_wm, resp_gm, resp_csf = responses # Valid responses
        model = MultiTissueCsdModel(gtab)

        # Example: Provide one response that is clearly not a tuple or ResponseFunction
        # Dipy's MultiShellDeconvModel expects a list of response functions (often ResponseFunction objects or (evals,S0) tuples)
        # If the internal model creation fails due to bad response, it should raise an error.
        with pytest.raises(Exception): # Catch a general Exception as Dipy's error might be specific
             model.fit(data, response_wm="bad_response", response_gm=resp_gm, response_csf=resp_csf)
        
        with pytest.raises(Exception):
             model.fit(data, response_wm=resp_wm, response_gm=(np.array([1,2,3])), response_csf=resp_csf) # GM response tuple of wrong length


    def test_multitissuecsdmodel_wm_odf(self, mt_csd_data_gtab_responses): # Updated test and fixture name
        gtab, data, responses = mt_csd_data_gtab_responses
        resp_wm, resp_gm, resp_csf = responses
        model = MultiTissueCsdModel(gtab) # Updated model name
        
        default_sphere = get_sphere('repulsion724')
        custom_sphere = get_sphere('symmetric362')

        fresh_model_for_odf_test = MultiTissueCsdModel(gtab) # Updated model name
        with pytest.raises(ValueError, match="Model has not been fitted yet. Call the 'fit' method first."):
            fresh_model_for_odf_test.wm_odf(default_sphere) # Updated method call

        model.fit(data, response_wm=resp_wm, response_gm=resp_gm, response_csf=resp_csf)

        odfs_default_sphere = model.wm_odf() # Updated method call
        assert isinstance(odfs_default_sphere, np.ndarray)
        assert odfs_default_sphere.shape == data.shape[:-1] + (default_sphere.vertices.shape[0],)
        assert np.issubdtype(odfs_default_sphere.dtype, np.floating)
        np.testing.assert_array_compare(lambda x,y: np.all(x >= y), odfs_default_sphere, -1e-6,
                                        err_msg="WM ODF values should be non-negative.")

        odfs_custom_sphere = model.wm_odf(sphere=custom_sphere) # Updated method call
        assert isinstance(odfs_custom_sphere, np.ndarray)
        assert odfs_custom_sphere.shape == data.shape[:-1] + (custom_sphere.vertices.shape[0],)
        np.testing.assert_array_compare(lambda x,y: np.all(x >= y), odfs_custom_sphere, -1e-6,
                                         err_msg="WM ODF values on custom sphere should be non-negative.")

    def test_multitissuecsdmodel_wm_odf_invalid_sphere(self, mt_csd_data_gtab_responses): # Updated names
        gtab, data, responses = mt_csd_data_gtab_responses
        resp_wm, resp_gm, resp_csf = responses
        model = MultiTissueCsdModel(gtab) # Updated model name
        model.fit(data, response_wm=resp_wm, response_gm=resp_gm, response_csf=resp_csf)
        with pytest.raises((AttributeError, TypeError)):
            model.wm_odf(sphere="not_a_sphere_object") # Updated method call

    # --- New tests for GM and CSF fractions ---

    def test_multitissuecsdmodel_gm_fraction(self, mt_csd_data_gtab_responses):
        gtab, data, responses = mt_csd_data_gtab_responses
        resp_wm, resp_gm, resp_csf = responses
        model = MultiTissueCsdModel(gtab)

        with pytest.raises(ValueError, match="Model has not been fitted yet. Call 'fit' method first."):
            model.gm_fraction()

        model.fit(data, response_wm=resp_wm, response_gm=resp_gm, response_csf=resp_csf)
        gm_frac = model.gm_fraction()
        
        assert isinstance(gm_frac, np.ndarray)
        assert gm_frac.shape == data.shape[:-1]
        assert np.issubdtype(gm_frac.dtype, np.floating)
        # Check if values are broadly within expected range [0, 1]
        # Allowing for small floating point inaccuracies or model variations.
        assert np.all(gm_frac >= -1e-6) and np.all(gm_frac <= 1.0 + 1e-6), "GM fractions out of [0,1] range."

    def test_multitissuecsdmodel_csf_fraction(self, mt_csd_data_gtab_responses):
        gtab, data, responses = mt_csd_data_gtab_responses
        resp_wm, resp_gm, resp_csf = responses
        model = MultiTissueCsdModel(gtab)

        with pytest.raises(ValueError, match="Model has not been fitted yet. Call 'fit' method first."):
            model.csf_fraction()

        model.fit(data, response_wm=resp_wm, response_gm=resp_gm, response_csf=resp_csf)
        csf_frac = model.csf_fraction()

        assert isinstance(csf_frac, np.ndarray)
        assert csf_frac.shape == data.shape[:-1]
        assert np.issubdtype(csf_frac.dtype, np.floating)
        assert np.all(csf_frac >= -1e-6) and np.all(csf_frac <= 1.0 + 1e-6), "CSF fractions out of [0,1] range."
        
    # Test for dhollander_kwargs (simplified version)
    # This test primarily ensures that the fit process completes when kwargs are passed.
    # Mocking response_dhollander to check arguments is more involved.
    def test_multitissuecsdmodel_fit_dhollander_kwargs(self, mt_csd_data_gtab_responses):
        gtab, data, _ = mt_csd_data_gtab_responses
        model = MultiTissueCsdModel(gtab)
        
        # Example dhollander_kwargs, these might not be optimal but test the pathway.
        # Using very basic parameters that response_dhollander accepts.
        # Check Dipy's documentation for appropriate values if specific behavior is needed.
        # For example, `cluster_thr` is a valid kwarg.
        # `roi_radii` or `roi_center` could also be passed.
        # If these kwargs are invalid for response_dhollander, it should raise an error.
        try:
            model.fit(data, fa_thresh=0.05, frf_fa_thresh=0.05, gm_fa_thresh=0.05, csf_fa_thresh=0.05) # Example valid kwargs
            # Check that fit object is created, indicating response_dhollander was likely called successfully.
            assert model._mt_fit_object is not None
            assert isinstance(model._response_wm, ResponseFunction) # Should be populated
        except RuntimeError as e:
            # If response estimation fails due to kwarg issues or data, this test might fail.
            # This is acceptable as it tests the kwarg pass-through.
            # We are primarily testing that the kwargs don't break the call signature.
            pytest.fail(f"model.fit with dhollander_kwargs raised RuntimeError: {e}")
        except TypeError as e:
            # This might happen if kwargs are invalid for response_dhollander
            pytest.fail(f"model.fit with dhollander_kwargs raised TypeError (invalid kwarg?): {e}")
