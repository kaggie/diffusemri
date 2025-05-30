import pytest
import numpy as np
from dipy.core.gradients import gradient_table, GradientTable
from dipy.reconst.dki import DiffusionKurtosisFit
from diffusemri.models.dki import DkiModel

# --- Fixture for Test Data ---

@pytest.fixture(scope="module") # Use module scope for efficiency if data generation is slow
def gtab_and_data():
    """
    Generates synthetic 4D dMRI data and a Dipy gtab for DKI model testing.
    """
    # Define b-values: one b0, and two non-zero shells (e.g., 1000, 2000)
    # For DKI, at least 15 unique gradient directions are recommended for bvals > 0.
    # Using 6 directions for b=1000 and 6 for b=2000, plus one b0. (Total 13 volumes)
    # For robust testing, more directions would be better, but this is a start.
    bvals_shell1 = np.full(6, 1000)
    bvals_shell2 = np.full(6, 2000)
    bvals = np.concatenate(([0], bvals_shell1, bvals_shell2))

    # Generate some approximately unique b-vectors
    # For real data, these would come from the scanner sequence.
    # Using a simple approach for unit testing.
    bvecs = np.zeros((len(bvals), 3))
    # For shell 1
    bvecs[1:7] = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    # For shell 2 (can be same or different, here slightly varied for example)
    bvecs[7:13] = np.array([
        [1, 1, 0], [-1, -1, 0],
        [1, -1, 0], [-1, 1, 0],
        [0, 1, 1], [0, -1, -1]
    ])
    # Normalize b-vectors
    for i in range(len(bvals)):
        if bvals[i] > 0:
            norm = np.linalg.norm(bvecs[i])
            if norm > 1e-6: # Avoid division by zero for b0 or if a bvec is accidentally zero
                bvecs[i] /= norm
            else: # If a bvec for non-b0 is zero (bad), set to a default
                bvecs[i] = np.array([1,0,0])


    gtab = gradient_table(bvals, bvecs)

    # Simulate data: S = S0 * exp(-b * ADC + 1/6 * b^2 * ADC^2 * K)
    # For simplicity, we'll generate data that is not purely noise.
    # We'll use a simplified signal decay, as the exact signal form is complex.
    # The goal is to have data that dipy.reconst.dki.DiffusionKurtosisModel can process.
    data_shape = (3, 3, 3, len(bvals)) # Small spatial dimensions for speed
    data = np.zeros(data_shape, dtype=np.float64) # Use float64 for stability
    S0 = 150.0
    ADC_sim = 0.7e-3 # Simulated ADC
    K_sim = 0.8      # Simulated Kurtosis (positive)

    for x in range(data_shape[0]):
        for y in range(data_shape[1]):
            for z in range(data_shape[2]):
                # Make it slightly spatially varying for more realistic testing
                S0_voxel = S0 + (x-1)*5 
                ADC_voxel = ADC_sim * (1 + (y-1)*0.1)
                K_voxel = K_sim * (1 + (z-1)*0.1)

                signal = S0_voxel * np.exp(-bvals * ADC_voxel + (1/6) * (bvals**2) * (ADC_voxel**2) * K_voxel)
                data[x, y, z, :] = signal

    # Add a small amount of Rician noise (more realistic than Gaussian for magnitude MR data)
    # For simplicity in this test, Gaussian noise is often used.
    # Dipy's localpca for MP-PCA assumes Gaussian noise.
    noise_std = 5.0
    data += np.random.normal(0, noise_std, data_shape)
    data[data < 0] = 0 # Signal is non-negative

    return gtab, data.astype(np.float32) # Cast to float32 as often used in practice


class TestDkiModel:
    def test_dkimodel_init(self, gtab_and_data):
        gtab, _ = gtab_and_data
        model = DkiModel(gtab)
        assert model.gtab is gtab, "gtab not stored correctly."
        assert isinstance(model.gtab, GradientTable), "gtab is not a GradientTable instance."
        assert model._fit_object is None, "_fit_object should be None initially."

    def test_dkimodel_init_invalid_gtab(self):
        with pytest.raises(ValueError, match="gtab must be an instance of Dipy's GradientTable."):
            DkiModel(gtab="not_a_gtab_object")

    def test_dkimodel_fit(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        
        # Test fitting with valid data
        model.fit(data)
        assert model._fit_object is not None, "_fit_object should not be None after fitting."
        assert isinstance(model._fit_object, DiffusionKurtosisFit), \
            "_fit_object is not an instance of DiffusionKurtosisFit."

    def test_dkimodel_fit_invalid_data(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        
        # Test with 3D data (should be 4D)
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit(data[..., 0])
        
        # Test with non-NumPy data
        with pytest.raises(ValueError, match="Input data must be a 4D NumPy array."):
            model.fit("not_numpy_data")

    def _test_parameter_property(self, model, data, property_name, expected_dtype=np.floating, 
                                 value_range_check=None, can_be_negative=False):
        """Helper function to test parameter properties."""
        # Test access before fit
        with pytest.raises(ValueError, match="Model has not been fitted yet. Call the 'fit' method first."):
            getattr(model, property_name)

        # Fit the model
        model.fit(data)
        
        # Access the property
        param_map = getattr(model, property_name)

        assert isinstance(param_map, np.ndarray), f"{property_name} map is not a NumPy array."
        assert param_map.shape == data.shape[:-1], \
            f"{property_name} map shape is {param_map.shape}, expected {data.shape[:-1]}."
        assert np.issubdtype(param_map.dtype, expected_dtype), \
            f"{property_name} map dtype is {param_map.dtype}, expected {expected_dtype}."

        if value_range_check:
            # Handle NaNs safely before range check
            valid_values = param_map[~np.isnan(param_map)]
            if valid_values.size > 0: # Only check if there are non-NaN values
                 assert value_range_check(valid_values), f"Values for {property_name} are out of expected range."
            # else:
            #     print(f"Warning: All values for {property_name} are NaN. Check data/fit.")

    def test_dkimodel_fa_property(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        self._test_parameter_property(model, data, "fa", 
                                      value_range_check=lambda x: np.all((x >= 0) & (x <= 1)))

    def test_dkimodel_md_property(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        # MD should generally be positive. Small negative values can occur due to noise/fit issues.
        self._test_parameter_property(model, data, "md",
                                      value_range_check=lambda x: np.all(x >= 0 if not can_be_negative else True)
                                     ) # A more lenient check might be needed for very noisy data.
                                     # For this test, strict non-negativity for valid values.

    def test_dkimodel_mk_property(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        # MK can be positive or negative. No specific range check here without ground truth.
        self._test_parameter_property(model, data, "mk", can_be_negative=True)

    def test_dkimodel_ak_property(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        self._test_parameter_property(model, data, "ak", can_be_negative=True)

    def test_dkimodel_rk_property(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        self._test_parameter_property(model, data, "rk", can_be_negative=True)

    def test_dkimodel_ka_property(self, gtab_and_data):
        gtab, data = gtab_and_data
        model = DkiModel(gtab)
        # KA is typically non-negative.
        self._test_parameter_property(model, data, "ka",
                                      value_range_check=lambda x: np.all(x >= 0))

# Example of how one might run this with pytest:
# Ensure PYTHONPATH includes the root of the diffusemri project.
# Then, from the root directory:
# pytest tests/test_dki.py
#
# To make this file runnable by itself for quick checks (though pytest is preferred):
# if __name__ == "__main__":
#     # This part is optional and primarily for direct execution debugging
#     # Pytest handles test discovery and execution normally.
#     gtab_fixture, data_fixture = gtab_and_data()
#     test_suite = TestDkiModel()
#     test_suite.test_dkimodel_init((gtab_fixture, data_fixture))
#     test_suite.test_dkimodel_fit((gtab_fixture, data_fixture))
#     # ... and so on for other tests
#     print("Subset of tests completed via direct run (use pytest for full suite).")
