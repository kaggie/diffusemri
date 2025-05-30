import pytest
import numpy as np
from dipy.core.gradients import gradient_table # Retain for gtab in fixture
# from dipy.data import get_sphere, Sphere # get_sphere can be removed, Sphere not directly passed
from dipy.tracking.streamline import Streamlines 
from dipy.sims.voxel import multi_tensor
from diffusemri.models import CsdModel # Still used in fixture to generate GFA
from diffusemri.tracking import track_deterministic_oudf

# --- Fixture for Test Data ---

@pytest.fixture(scope="module")
def tracking_setup():
    """
    Generates synthetic data and model components for deterministic tracking tests.
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
    # For robust response function in test, use a small central ROI of the data
    # Assuming auto_response_ssst can find reasonable FA values in the synthetic data.
    # These kwargs are for auto_response_ssst called within CsdModel's fit
    csd_model_wrapper.fit(data, roi_center=(data_shape_spatial[0]//2, 
                                            data_shape_spatial[1]//2, 
                                            data_shape_spatial[2]//2), 
                                roi_radii=3, fa_thresh=0.5) # fa_thresh might need tuning
    fitted_dipy_odf_model = csd_model_wrapper._fit_object # This is the CsdFit object

    # 4. Create seed mask and stopping criterion mask
    seed_mask = np.zeros(data_shape_spatial, dtype=bool)
    # Seed from a small central region (e.g., a 3x3x3 cube in the center)
    center_slice = slice(data_shape_spatial[0]//2 - 1, data_shape_spatial[0]//2 + 2)
    seed_mask[center_slice, center_slice, center_slice] = True

    # 5. Create seed coordinates
    # Take a few points from the True region of the mask
    coords_x, coords_y, coords_z = np.where(seed_mask)
    seed_coords = np.vstack((coords_x[:5], coords_y[:5], coords_z[:5])).T 
    # Transform to world space if affine was not identity (here it is, so coords are same)
    # For non-identity affine: seed_coords = nib.affines.apply_affine(affine, seed_coords)


    # 6. Create GFA map
    gfa_map = csd_model_wrapper.gfa # Using the GFA property from our CsdModel

    return {
        "gtab": gtab,
        "data": data,
        # "fitted_dipy_odf_model": fitted_dipy_odf_model, # Removed
        "seed_mask": seed_mask,
        "seed_coords": seed_coords,
        "gfa_map": gfa_map, # Still needed for stopping criterion
        "affine": affine
        # "standard_sphere": get_sphere('repulsion724') # Removed
    }

class TestDeterministicTracking:
    def test_tracking_with_mask_seeds(self, tracking_setup):
        setup = tracking_setup

        streamlines = track_deterministic_oudf(
            dwi_data=setup["data"],                         # NEW
            gtab=setup["gtab"],                             # NEW
            seeds=setup["seed_mask"],
            metric_map_for_stopping=setup["gfa_map"],       
            stopping_threshold_value=0.05,                  
            affine=setup["affine"],
            # Add new PeaksFromModel params:
            sh_order=6, 
            model_max_peaks=3, 
            model_min_separation_angle=25, 
            model_peak_threshold=0.4 
        )
        assert isinstance(streamlines, Streamlines)
        assert len(streamlines) > 0, "No streamlines generated with mask seeds."
        for sl in streamlines:
            assert sl.ndim == 2, "Streamline should be a 2D array."
            assert sl.shape[0] >= 2, "Streamline should have at least 2 points."
            assert sl.shape[1] == 3, "Streamline points should have 3 coordinates."

    def test_tracking_with_coord_seeds(self, tracking_setup):
        setup = tracking_setup
        # For coord seeds, stopping criterion needs to be based on something sensible.
        # Using a GFA-based one or a dilated version of the seed_mask.
        # stopping_criterion = BinaryStoppingCriterion(setup["seed_mask"]) # Removed

        streamlines = track_deterministic_oudf(
            dwi_data=setup["data"],                         # NEW
            gtab=setup["gtab"],                             # NEW
            seeds=setup["seed_coords"],
            metric_map_for_stopping=setup["gfa_map"],       
            stopping_threshold_value=0.05,                  
            affine=setup["affine"],
            sh_order=6,
            model_max_peaks=3
        )
        assert isinstance(streamlines, Streamlines)
        # Using coordinates might generate fewer streamlines than a dense mask.
        # The number of seeds is small (5), so we expect some streamlines.
        assert len(streamlines) > 0, "No streamlines generated with coordinate seeds." 
        for sl in streamlines:
            assert sl.ndim == 2
            assert sl.shape[0] >= 2
            assert sl.shape[1] == 3
            
    def test_tracking_with_length_filters(self, tracking_setup):
        setup = tracking_setup
        # stopping_criterion = BinaryStoppingCriterion(setup["seed_mask"]) # Removed
        
        min_l, max_l = 20, 50 # Choose values that are likely to filter some default streamlines
        
        streamlines = track_deterministic_oudf(
            dwi_data=setup["data"],                         # NEW
            gtab=setup["gtab"],                             # NEW
            seeds=setup["seed_mask"], # Using mask for more streamlines to test filtering
            metric_map_for_stopping=setup["gfa_map"],       
            stopping_threshold_value=0.05,                  
            affine=setup["affine"],
            sh_order=6,
            model_max_peaks=3,
            min_length=min_l,
            max_length=max_l
        )
        assert isinstance(streamlines, Streamlines)
        if len(streamlines) > 0: # Only check lengths if streamlines were produced
            for sl in streamlines:
                sl_len = np.linalg.norm(np.diff(sl, axis=0), axis=1).sum()
                assert sl_len >= min_l, f"Streamline length {sl_len} is less than min_length {min_l}."
                assert sl_len <= max_l, f"Streamline length {sl_len} is greater than max_length {max_l}."
        # else:
        #     print(f"Warning: No streamlines produced with length filters [{min_l}, {max_l}]. Test might not be very informative.")


    def test_tracking_invalid_inputs(self, tracking_setup):
        setup = tracking_setup
        # valid_odf_model = setup["fitted_dipy_odf_model"] # Removed
        valid_dwi_data = setup["data"]
        valid_gtab = setup["gtab"]
        valid_seeds = setup["seed_mask"]
        valid_gfa_map = setup["gfa_map"]
        # valid_sphere = setup["standard_sphere"] # Removed
        valid_affine = setup["affine"]
        valid_threshold = 0.1
        default_sh_order = 6 # Matching new defaults in track_deterministic_oudf

        # Invalid dwi_data
        with pytest.raises(ValueError): # Not 4D
            track_deterministic_oudf(dwi_data=np.random.rand(10,10,10), gtab=valid_gtab, seeds=valid_seeds, 
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value=valid_threshold, 
                                     affine=valid_affine, sh_order=default_sh_order)
        with pytest.raises(ValueError): # Wrong type (TypeError might be caught by np.array if not an array-like)
                                       # Assuming track_deterministic_oudf expects np.ndarray for dwi_data
            track_deterministic_oudf(dwi_data="not_an_array", gtab=valid_gtab, seeds=valid_seeds,
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value=valid_threshold,
                                     affine=valid_affine, sh_order=default_sh_order)

        # Invalid gtab
        with pytest.raises(TypeError):
            track_deterministic_oudf(dwi_data=valid_dwi_data, gtab="not_a_gtab", seeds=valid_seeds,
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value=valid_threshold,
                                     affine=valid_affine, sh_order=default_sh_order)

        # Invalid seeds (copied from previous version, still valid)
        with pytest.raises(ValueError): 
            track_deterministic_oudf(dwi_data=valid_dwi_data, gtab=valid_gtab, seeds=np.array([1,2,3]), 
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value=valid_threshold, 
                                     affine=valid_affine, sh_order=default_sh_order)
        with pytest.raises(ValueError): 
             track_deterministic_oudf(dwi_data=valid_dwi_data, gtab=valid_gtab, seeds="not_an_array", 
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value=valid_threshold, 
                                     affine=valid_affine, sh_order=default_sh_order)

        # Invalid metric_map_for_stopping (copied from previous version, still valid)
        with pytest.raises(ValueError): 
            track_deterministic_oudf(dwi_data=valid_dwi_data, gtab=valid_gtab, seeds=valid_seeds, 
                                     metric_map_for_stopping="not_an_array", stopping_threshold_value=valid_threshold, 
                                     affine=valid_affine, sh_order=default_sh_order)
        with pytest.raises(ValueError): 
            track_deterministic_oudf(dwi_data=valid_dwi_data, gtab=valid_gtab, seeds=valid_seeds, 
                                     metric_map_for_stopping=np.random.rand(10,10), stopping_threshold_value=valid_threshold, 
                                     affine=valid_affine, sh_order=default_sh_order)

        # Invalid stopping_threshold_value (copied from previous version, still valid)
        with pytest.raises(ValueError): 
            track_deterministic_oudf(dwi_data=valid_dwi_data, gtab=valid_gtab, seeds=valid_seeds, 
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value="not_a_float", 
                                     affine=valid_affine, sh_order=default_sh_order)
        
        # Invalid sh_order (example)
        with pytest.raises(ValueError): # Assuming PeaksFromModel or track_deterministic_oudf might validate sh_order (e.g. must be even, positive)
             track_deterministic_oudf(dwi_data=valid_dwi_data, gtab=valid_gtab, seeds=valid_seeds, 
                                     metric_map_for_stopping=valid_gfa_map, stopping_threshold_value=valid_threshold, 
                                     affine=valid_affine, sh_order=-1) # Invalid sh_order


    def test_tracking_with_gfa_stopping(self, tracking_setup):
        setup = tracking_setup
        gfa_threshold = 0.05 
        # stopping_criterion = ThresholdStoppingCriterion(setup["gfa_map"], threshold=gfa_threshold) # Removed

        streamlines = track_deterministic_oudf(
            dwi_data=setup["data"],                         # NEW
            gtab=setup["gtab"],                             # NEW
            seeds=setup["seed_mask"],
            metric_map_for_stopping=setup["gfa_map"],       
            stopping_threshold_value=gfa_threshold,         
            affine=setup["affine"],
            sh_order=6,
            model_max_peaks=3
        )
        assert isinstance(streamlines, Streamlines)
        # It's possible no streamlines are generated if GFA is too low everywhere or seeds are in low GFA
        # For this test, we mainly check if it runs without error with this criterion.
        # A more robust test would require specific GFA pattern and seed placement.
        if len(streamlines) == 0:
            print(f"Warning: No streamlines generated with GFA stopping criterion (threshold={gfa_threshold}). "
                  "This might be due to GFA values in synthetic data or seed placement.")
        # else:
        #    print(f"Generated {len(streamlines)} streamlines with GFA stopping.")
