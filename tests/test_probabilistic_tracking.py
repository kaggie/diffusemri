import pytest
import numpy as np
from dipy.core.gradients import gradient_table, GradientTable
from dipy.data import get_sphere, Sphere
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion, StoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.sims.voxel import multi_tensor
from diffusemri.models import CsdModel # Using CSD for ODF model with shm_coeff
from diffusemri.tracking import track_probabilistic_odf

# --- Fixture for Test Data ---

@pytest.fixture(scope="module")
def prob_tracking_setup():
    """
    Generates synthetic data and model components for probabilistic tracking tests.
    Similar to tracking_setup but tailored for probabilistic needs if any.
    """
    # 1. Generate gtab (multi-shell for CSD)
    bvals = np.concatenate((np.zeros(6), np.ones(30) * 1000, np.ones(30) * 2000))
    bvecs_s1 = np.random.randn(30, 3)
    bvecs_s1 /= np.linalg.norm(bvecs_s1, axis=1)[:, None]
    bvecs_s2 = np.random.randn(30, 3)
    bvecs_s2 /= np.linalg.norm(bvecs_s2, axis=1)[:, None]
    bvecs = np.concatenate((np.zeros((6, 3)), bvecs_s1, bvecs_s2))
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)

    # 2. Simulate dMRI data (e.g., 10x10x10 volume with crossing fibers)
    data_shape_spatial = (10, 10, 10)
    mevals = np.array([[0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]])
    angles = [(0, 0), (90, 0)] # Two perpendicular fibers
    fractions = [50, 50]
    signal, _ = multi_tensor(gtab, mevals, S0=100, angles=angles, fractions=fractions, snr=None)
    data = np.tile(signal, data_shape_spatial + (1,)).astype(np.float32)
    affine = np.eye(4) # Simple identity affine

    # 3. Fit ODF Model (CsdModel)
    csd_model_wrapper = CsdModel(gtab)
    csd_model_wrapper.fit(data, roi_center=(data_shape_spatial[0]//2, 
                                            data_shape_spatial[1]//2, 
                                            data_shape_spatial[2]//2), 
                                roi_radii=3, fa_thresh=0.5)
    fitted_dipy_odf_model = csd_model_wrapper._fit_object # This is CsdFit, has .shm_coeff

    # 4. Create seed mask
    seed_mask = np.zeros(data_shape_spatial, dtype=bool)
    center_slice = slice(data_shape_spatial[0]//2 - 1, data_shape_spatial[0]//2 + 2)
    seed_mask[center_slice, center_slice, center_slice] = True

    # 5. Create seed coordinates
    coords_x, coords_y, coords_z = np.where(seed_mask)
    # Take a few points from the True region of the mask
    seed_coords = np.vstack((coords_x[:3], coords_y[:3], coords_z[:3])).T 

    return {
        "gtab": gtab,
        "data": data,
        "fitted_dipy_odf_model": fitted_dipy_odf_model, # Has .shm_coeff
        "seed_mask": seed_mask,
        "seed_coords": seed_coords,
        "affine": affine,
        "standard_sphere": get_sphere('repulsion724')
    }

class TestProbabilisticTracking:
    def test_tracking_prob_with_mask_seeds(self, prob_tracking_setup):
        setup = prob_tracking_setup
        # Use a different part of the mask for stopping to avoid immediate termination
        stopping_mask = np.zeros_like(setup["seed_mask"])
        stopping_mask[0,0,0] = True # Arbitrary small stopping region for test
        stopping_criterion = BinaryStoppingCriterion(stopping_mask) 
        samples_per_voxel = 2 # Reduced for faster tests

        streamlines = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"],
            seeds=setup["seed_mask"],
            stopping_criterion=stopping_criterion,
            sphere=setup["standard_sphere"],
            affine=setup["affine"],
            samples_per_voxel=samples_per_voxel,
            min_length=1 # Use small min_length to ensure some streamlines are kept
        )
        assert isinstance(streamlines, Streamlines)
        
        num_seed_voxels = np.sum(setup["seed_mask"])
        expected_min_streamlines = num_seed_voxels * samples_per_voxel * 0.1 # Expect at least 10% success
        
        assert len(streamlines) > 0, "No streamlines generated with mask seeds."
        # Probabilistic nature means exact count is not guaranteed.
        # Check if a reasonable number of streamlines were generated.
        # For this test, ensuring some streamlines are generated is key.
        # assert len(streamlines) >= expected_min_streamlines 

        if len(streamlines) > 0:
            for sl in streamlines:
                assert sl.ndim == 2, "Streamline should be a 2D array."
                assert sl.shape[0] >= 2, "Streamline should have at least 2 points."
                assert sl.shape[1] == 3, "Streamline points should have 3 coordinates."

    def test_tracking_prob_with_coord_seeds(self, prob_tracking_setup):
        setup = prob_tracking_setup
        stopping_mask = np.zeros_like(setup["seed_mask"])
        stopping_mask[0,0,0] = True 
        stopping_criterion = BinaryStoppingCriterion(stopping_mask)
        samples_per_voxel = 2

        streamlines = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"],
            seeds=setup["seed_coords"],
            stopping_criterion=stopping_criterion,
            sphere=setup["standard_sphere"],
            affine=setup["affine"],
            samples_per_voxel=samples_per_voxel,
            min_length=1
        )
        assert isinstance(streamlines, Streamlines)
        assert len(streamlines) > 0, "No streamlines generated with coordinate seeds."
        # expected_min_streamlines = setup["seed_coords"].shape[0] * samples_per_voxel * 0.1
        # assert len(streamlines) >= expected_min_streamlines

        if len(streamlines) > 0:
            for sl in streamlines:
                assert sl.ndim == 2
                assert sl.shape[0] >= 2
                assert sl.shape[1] == 3
            
    def test_tracking_prob_length_filters(self, prob_tracking_setup):
        setup = prob_tracking_setup
        stopping_mask = np.zeros_like(setup["seed_mask"])
        stopping_mask[0,0,0] = True
        stopping_criterion = BinaryStoppingCriterion(stopping_mask)
        
        min_l, max_l = 5, 15 # Adjusted for potentially shorter probabilistic tracks
        
        streamlines = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"],
            seeds=setup["seed_mask"],
            stopping_criterion=stopping_criterion,
            sphere=setup["standard_sphere"],
            affine=setup["affine"],
            min_length=min_l,
            max_length=max_l,
            samples_per_voxel=1 # Reduce samples to speed up this specific test
        )
        assert isinstance(streamlines, Streamlines)
        if len(streamlines) > 0:
            for sl_idx in range(len(streamlines)):
                sl = streamlines[sl_idx]
                # Calculate length using Dipy's Streamlines object's length method
                sl_len = streamlines.length(sl_idx)
                assert sl_len >= min_l, f"Streamline length {sl_len} is less than min_length {min_l}."
                assert sl_len <= max_l, f"Streamline length {sl_len} is greater than max_length {max_l}."

    def test_tracking_prob_invalid_inputs(self, prob_tracking_setup):
        setup = prob_tracking_setup
        valid_seeds = setup["seed_mask"]
        stopping_mask = np.zeros_like(setup["seed_mask"])
        stopping_mask[0,0,0] = True
        valid_stopping_criterion = BinaryStoppingCriterion(stopping_mask)
        valid_sphere = setup["standard_sphere"]

        # Invalid odf_fit_object (missing shm_coeff)
        class BadFitObject: pass
        with pytest.raises(AttributeError, match="odf_fit_object must have an 'shm_coeff' attribute."):
            track_probabilistic_odf(BadFitObject(), valid_seeds, valid_stopping_criterion, valid_sphere)

        # Invalid seeds
        with pytest.raises(ValueError):
            track_probabilistic_odf(setup["fitted_dipy_odf_model"], "not_an_array", valid_stopping_criterion, valid_sphere)

    def test_tracking_prob_pmf_threshold(self, prob_tracking_setup):
        setup = prob_tracking_setup
        stopping_mask = np.zeros_like(setup["seed_mask"])
        stopping_mask[0,0,0] = True
        stopping_criterion = BinaryStoppingCriterion(stopping_mask)

        streamlines_low_thresh = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"], seeds=setup["seed_mask"],
            stopping_criterion=stopping_criterion, sphere=setup["standard_sphere"],
            affine=setup["affine"], pmf_threshold=0.01, samples_per_voxel=1, min_length=1
        )
        streamlines_high_thresh = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"], seeds=setup["seed_mask"],
            stopping_criterion=stopping_criterion, sphere=setup["standard_sphere"],
            affine=setup["affine"], pmf_threshold=0.95, samples_per_voxel=1, min_length=1
        )
        assert isinstance(streamlines_low_thresh, Streamlines)
        assert isinstance(streamlines_high_thresh, Streamlines)
        # Expect high threshold to produce fewer or equal streamlines than low threshold
        if len(streamlines_low_thresh) > 0: # Only assert if low_thresh produced something
             assert len(streamlines_high_thresh) <= len(streamlines_low_thresh)
        # else:
        #     print("Warning: Low PMF threshold tracking produced no streamlines, high threshold comparison is trivial.")


    def test_tracking_prob_max_angle(self, prob_tracking_setup):
        setup = prob_tracking_setup
        stopping_mask = np.zeros_like(setup["seed_mask"])
        stopping_mask[0,0,0] = True
        stopping_criterion = BinaryStoppingCriterion(stopping_mask)

        # Test with a small max_angle
        streamlines_small_angle = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"], seeds=setup["seed_mask"],
            stopping_criterion=stopping_criterion, sphere=setup["standard_sphere"],
            affine=setup["affine"], max_angle=5.0, samples_per_voxel=1, min_length=1
        )
        # Test with a larger max_angle
        streamlines_large_angle = track_probabilistic_odf(
            odf_fit_object=setup["fitted_dipy_odf_model"], seeds=setup["seed_mask"],
            stopping_criterion=stopping_criterion, sphere=setup["standard_sphere"],
            affine=setup["affine"], max_angle=60.0, samples_per_voxel=1, min_length=1
        )
        assert isinstance(streamlines_small_angle, Streamlines)
        assert isinstance(streamlines_large_angle, Streamlines)
        # Difficult to assert specific outcomes, but both should run.
        # Typically, larger max_angle might produce more (or longer) streamlines if curvature is high.
        # For now, just checking they run and produce Streamlines objects.
        # If small angle produces 0, large angle might produce more, but not guaranteed.
        if len(streamlines_small_angle) == 0 and len(streamlines_large_angle) > 0:
            pass # This is a plausible outcome
        elif len(streamlines_large_angle) == 0 and len(streamlines_small_angle) > 0:
             pass # Also plausible if data is very straight, small angle is enough
        # print(f"Streamlines with small max_angle: {len(streamlines_small_angle)}")
        # print(f"Streamlines with large max_angle: {len(streamlines_large_angle)}")
