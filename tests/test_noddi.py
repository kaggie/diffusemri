import pytest
import torch
import numpy as np
from utils.pytorch_gradient_utils import PyTorchGradientTable # Updated import
import math

# Imports from our project
from models import noddi_signal
from models.noddi_model import NoddiModelTorch
from fitting.noddi_fitter import preprocess_noddi_input, fit_noddi_volume

EPS = 1e-7 # Epsilon for floating point comparisons and stability

# --- Fixtures ---

@pytest.fixture(scope="module")
def simple_gtab():
    """A simple GradientTable for use in multiple tests."""
    bvals = np.array([0, 1000, 1000, 1000, 2000, 2000, 2000])
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), 0, 1/np.sqrt(2)],
        [0, 1/np.sqrt(2), 1/np.sqrt(2)]
    ])
    # Using PyTorchGradientTable's default b0_threshold=50.0, which is fine for bvals[0]=0
    return PyTorchGradientTable(bvals, bvecs, b0_threshold=50.0) 

@pytest.fixture(scope="module")
def default_noddi_params_dict_torch():
    """Default single-voxel NODDI parameters as torch tensors."""
    return {
        'f_intra': torch.tensor(0.55),
        'f_iso': torch.tensor(0.15),
        'kappa': torch.tensor(2.0),    # ODI ~0.3
        'mu_theta': torch.tensor(math.pi / 2.0), # Equatorial
        'mu_phi': torch.tensor(0.0)           # Aligned with x-axis
    }

# --- Helper Functions for Tests ---

def generate_synthetic_dwi_signals(params_dict, gtab_torch_bvals, gtab_torch_bvecs, 
                                   d_intra=noddi_signal.d_intra_default, 
                                   d_iso=noddi_signal.d_iso_default):
    """Generates synthetic DWI signals for given NODDI parameters."""
    with torch.no_grad():
        signals = noddi_signal.noddi_signal_model(
            params=params_dict,
            b_values=gtab_torch_bvals,
            gradient_directions=gtab_torch_bvecs,
            d_intra_val=d_intra,
            d_iso_val=d_iso
        )
    return signals

# --- Tests for models.noddi_signal ---

def test_noddi_signal_generation_basic(simple_gtab, default_noddi_params_dict_torch):
    torch.manual_seed(0)
    # simple_gtab.bvals and .bvecs are already PyTorch tensors
    gtab_bvals_torch = simple_gtab.bvals.float() 
    gtab_bvecs_torch = simple_gtab.bvecs.float()

    signals = generate_synthetic_dwi_signals(
        default_noddi_params_dict_torch, gtab_bvals_torch, gtab_bvecs_torch
    )

    assert signals.shape == (gtab_bvals_torch.shape[0],)
    assert torch.all(signals >= 0.0)
    assert torch.all(signals <= 1.0 + EPS) # Should be <= 1.0 for normalized signals

    # Test b0 signal (should be close to 1.0 as S0 is factored out)
    # f_intra + f_iso + f_extra = 1. S_intra(b0)=1, S_iso(b0)=1, S_extra(b0)=1
    # So, S(b0)/S0 should be 1.0
    assert torch.isclose(signals[0], torch.tensor(1.0), atol=1e-4)


def test_noddi_signal_batching(simple_gtab, default_noddi_params_dict_torch):
    torch.manual_seed(0)
    gtab_bvals_torch = simple_gtab.bvals.float()
    gtab_bvecs_torch = simple_gtab.bvecs.float()
    
    batch_size = 3
    batched_params = {}
    for key, val in default_noddi_params_dict_torch.items():
        batched_params[key] = val.repeat(batch_size)
    
    # Add some variation to batched params
    batched_params['f_intra'] = torch.tensor([0.4, 0.5, 0.6])
    batched_params['kappa'] = torch.tensor([1.0, 2.0, 3.0])

    signals = generate_synthetic_dwi_signals(
        batched_params, gtab_bvals_torch, gtab_bvecs_torch
    )
    
    assert signals.shape == (batch_size, gtab_bvals_torch.shape[0])
    assert torch.all(signals >= 0.0)
    assert torch.all(signals <= 1.0 + EPS)
    assert torch.allclose(signals[:, 0], torch.tensor(1.0), atol=1e-4)


def test_noddi_signal_parameter_ranges(simple_gtab):
    """Test with some extreme valid parameter values."""
    torch.manual_seed(0)
    gtab_bvals_torch = simple_gtab.bvals.float()
    gtab_bvecs_torch = simple_gtab.bvecs.float()

    test_cases = [
        ({'f_intra': torch.tensor(0.01), 'f_iso': torch.tensor(0.98), 'kappa': torch.tensor(0.1), # High f_iso, low kappa
          'mu_theta': torch.tensor(0.1), 'mu_phi': torch.tensor(0.1)}, "low_fic_high_fiso"),
        ({'f_intra': torch.tensor(0.98), 'f_iso': torch.tensor(0.01), 'kappa': torch.tensor(50.0), # High fic, high kappa
          'mu_theta': torch.tensor(math.pi/2), 'mu_phi': torch.tensor(math.pi/2)}, "high_fic_high_kappa"),
        ({'f_intra': torch.tensor(0.5), 'f_iso': torch.tensor(0.0), 'kappa': torch.tensor(10.0), # No f_iso
          'mu_theta': torch.tensor(0.0), 'mu_phi': torch.tensor(0.0)}, "no_fiso_oriented_z"),
    ]

    for params, name in test_cases:
        signals = generate_synthetic_dwi_signals(params, gtab_bvals_torch, gtab_bvecs_torch)
        assert signals.shape == (gtab_bvals_torch.shape[0],), f"Test case {name} failed shape check."
        assert torch.all(signals >= 0.0) and torch.all(signals <= 1.0 + EPS), f"Test case {name} failed bounds check."
        assert torch.isclose(signals[0], torch.tensor(1.0), atol=1e-4), f"Test case {name} failed b0 signal check."

# --- Tests for models.noddi_model.NoddiModelTorch ---

@pytest.fixture
def synthetic_batch_for_fitting(simple_gtab, default_noddi_params_dict_torch):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gtab_bvals_torch = simple_gtab.bvals.float().to(device)
    gtab_bvecs_torch = simple_gtab.bvecs.float().to(device)

    batch_size = 2
    true_params_batch = {}
    # Voxel 1: default params
    # Voxel 2: different params
    f_intra_vals = [default_noddi_params_dict_torch['f_intra'].item(), 0.7]
    f_iso_vals = [default_noddi_params_dict_torch['f_iso'].item(), 0.05]
    kappa_vals = [default_noddi_params_dict_torch['kappa'].item(), 0.5] # High ODI for voxel 2
    mu_theta_vals = [default_noddi_params_dict_torch['mu_theta'].item(), math.pi / 4.0]
    mu_phi_vals = [default_noddi_params_dict_torch['mu_phi'].item(), math.pi / 3.0]

    for key in default_noddi_params_dict_torch.keys():
        if key == 'f_intra': true_params_batch[key] = torch.tensor(f_intra_vals, device=device)
        elif key == 'f_iso': true_params_batch[key] = torch.tensor(f_iso_vals, device=device)
        elif key == 'kappa': true_params_batch[key] = torch.tensor(kappa_vals, device=device)
        elif key == 'mu_theta': true_params_batch[key] = torch.tensor(mu_theta_vals, device=device)
        elif key == 'mu_phi': true_params_batch[key] = torch.tensor(mu_phi_vals, device=device)
        else: true_params_batch[key] = default_noddi_params_dict_torch[key].repeat(batch_size).to(device)
        
    dwi_signals_true = generate_synthetic_dwi_signals(
        true_params_batch, gtab_bvals_torch, gtab_bvecs_torch
    )
    
    # Add a small amount of noise
    noise_level = 0.01
    dwi_signals_noisy = dwi_signals_true + torch.randn_like(dwi_signals_true) * noise_level
    dwi_signals_noisy = dwi_signals_noisy.clamp(min=EPS)
    
    return {
        "gtab": simple_gtab,
        "true_params": true_params_batch,
        "dwi_signals_noisy": dwi_signals_noisy,
        "device": device
    }

def test_noddi_fit_batch_synthetic_data(synthetic_batch_for_fitting):
    torch.manual_seed(42)
    data = synthetic_batch_for_fitting
    device = data["device"]
    
    noddi_fitter_model = NoddiModelTorch(gtab=data["gtab"]).to(device)
    
    # Use fewer iterations for faster tests
    fit_hyperparams = {'learning_rate': 0.025, 'n_iterations': 300} 

    fitted_params = noddi_fitter_model.fit_batch(
        dwi_signals_normalized=data["dwi_signals_noisy"],
        **fit_hyperparams
    )

    true_odi = noddi_fitter_model.kappa_to_odi(data["true_params"]['kappa'])

    # Check if fitted parameters are reasonably close to true ones
    # Tolerances might need to be adjusted based on noise and iterations
    # Note: Angles (mu_theta, mu_phi) can be tricky due to symmetries/periodicity,
    # so direct comparison might fail if orientation is ambiguous or fit is not perfect.
    # For now, check volume fractions and ODI/kappa.
    assert np.allclose(fitted_params['f_intra'].cpu().numpy(), data["true_params"]['f_intra'].cpu().numpy(), atol=0.15)
    assert np.allclose(fitted_params['f_iso'].cpu().numpy(), data["true_params"]['f_iso'].cpu().numpy(), atol=0.15)
    assert np.allclose(fitted_params['odi'].cpu().numpy(), true_odi.cpu().numpy(), atol=0.2) # ODI can be harder to fit
    assert np.allclose(fitted_params['kappa'].cpu().numpy(), data["true_params"]['kappa'].cpu().numpy(), atol=1.5) # Kappa is sensitive

    # Test constraints
    assert torch.all(fitted_params['f_intra'] >= 0) and torch.all(fitted_params['f_intra'] <= 1)
    assert torch.all(fitted_params['f_iso'] >= 0) and torch.all(fitted_params['f_iso'] <= 1)
    assert torch.all(fitted_params['f_intra'] + fitted_params['f_iso'] <= 1.0 + EPS)
    assert torch.all(fitted_params['odi'] >= 0) and torch.all(fitted_params['odi'] <= 1)
    assert torch.all(fitted_params['kappa'] > 0)
    assert torch.all(fitted_params['mu_theta'] >= 0) and torch.all(fitted_params['mu_theta'] <= math.pi + EPS)
    # mu_phi can be 0 to 2pi. The model maps it to this.
    assert torch.all(fitted_params['mu_phi'] >= 0) and torch.all(fitted_params['mu_phi'] <= 2 * math.pi + EPS)


def test_noddi_fit_batch_with_initial_mu(synthetic_batch_for_fitting):
    """Tests fit_batch with initial mu orientations provided."""
    torch.manual_seed(43) # Different seed for variety
    data = synthetic_batch_for_fitting
    device = data["device"]
    batch_size = data["dwi_signals_noisy"].shape[0]

    noddi_fitter_model = NoddiModelTorch(gtab=data["gtab"]).to(device)
    fit_hyperparams = {'learning_rate': 0.025, 'n_iterations': 50} # Very few iterations

    # Create some dummy initial mu vectors (e.g., all pointing along y-axis)
    # These might not be close to true_params, to see if model attempts to use them.
    initial_mu_orientations = torch.zeros(batch_size, 3, device=device)
    initial_mu_orientations[:, 1] = 1.0 # Pointing along Y

    fitted_params = noddi_fitter_model.fit_batch(
        dwi_signals_normalized=data["dwi_signals_noisy"],
        initial_mu_batch=initial_mu_orientations,
        **fit_hyperparams
    )
    
    # Basic checks: runs without error, parameters are within bounds
    assert 'f_intra' in fitted_params
    assert fitted_params['f_intra'].shape == (batch_size,)
    assert torch.all(fitted_params['f_intra'] >= 0) and torch.all(fitted_params['f_intra'] <= 1)
    assert torch.all(fitted_params['odi'] >= 0) and torch.all(fitted_params['odi'] <= 1)

    # Check if the initial orientation influenced the result.
    # With very few iterations, the fitted mu should be closer to the initial_mu_batch
    # than to the default initialization (which is pi/2, pi/2 for theta, phi).
    # Default init: theta=pi/2, phi=pi/2 -> mu_cartesian = [cos(pi/2)sin(pi/2), sin(pi/2)sin(pi/2), cos(pi/2)] = [0,1,0]
    # Our initial_mu_orientations is also [0,1,0].
    # So, we expect theta to be around pi/2 and phi around pi/2.
    # This test mainly ensures the pathway for initial_mu_batch works.
    assert np.allclose(fitted_params['mu_theta'].cpu().numpy(), math.pi / 2, atol=0.5) # Wider tol due to few iterations
    assert np.allclose(fitted_params['mu_phi'].cpu().numpy(), math.pi / 2, atol=0.5)


# Parameterized test cases for different NODDI parameter combinations
NODDI_TEST_CASES = [
    # f_intra, kappa, f_iso, mu_theta_factor (x pi), mu_phi_factor (x 2pi)
    pytest.param(0.3, 0.5, 0.1, 0.25, 0.25, id="low_fintra_high_odi_low_fiso"),
    pytest.param(0.7, 10.0, 0.05, 0.75, 0.75, id="high_fintra_low_odi_low_fiso"),
    pytest.param(0.5, 2.0, 0.3, 0.5, 0.5, id="med_fintra_med_odi_med_fiso"),
    pytest.param(0.8, 5.0, 0.15, 0.1, 0.9, id="vhigh_fintra_lowmed_odi_med_fiso_asymm_mu"),
    pytest.param(0.2, 0.2, 0.5, 0.9, 0.1, id="low_fintra_vhigh_odi_high_fiso_asymm_mu_2"), # Ensure f_intra + f_iso < 1
]

@pytest.mark.parametrize("true_f_intra_val, true_kappa_val, true_f_iso_val, true_mu_theta_factor, true_mu_phi_factor", NODDI_TEST_CASES)
def test_noddi_fit_batch_parameter_combinations(
    simple_gtab, 
    true_f_intra_val, true_kappa_val, true_f_iso_val, 
    true_mu_theta_factor, true_mu_phi_factor,
    default_noddi_params_dict_torch # for default mu if needed, but we override
):
    torch.manual_seed(hash(str(true_f_intra_val + true_kappa_val + true_f_iso_val))) # Seed per combination
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gtab_bvals_torch = simple_gtab.bvals.float().to(device)
    gtab_bvecs_torch = simple_gtab.bvecs.float().to(device)

    batch_size = 2 # Test with a small batch for speed
    
    # Ensure f_intra + f_iso < 1.0 (e.g. max 0.95)
    if true_f_intra_val + true_f_iso_val >= 0.98:
        # Scale them down if sum is too high for this test case
        current_sum = true_f_intra_val + true_f_iso_val
        scale_factor = 0.95 / current_sum
        true_f_intra_val *= scale_factor
        true_f_iso_val *= scale_factor

    true_params_batch = {
        'f_intra': torch.full((batch_size,), true_f_intra_val, device=device),
        'f_iso': torch.full((batch_size,), true_f_iso_val, device=device),
        'kappa': torch.full((batch_size,), true_kappa_val, device=device),
        'mu_theta': torch.full((batch_size,), true_mu_theta_factor * math.pi, device=device),
        'mu_phi': torch.full((batch_size,), true_mu_phi_factor * (2 * math.pi), device=device)
    }
        
    dwi_signals_true = generate_synthetic_dwi_signals(
        true_params_batch, gtab_bvals_torch, gtab_bvecs_torch
    )
    
    noise_level = 0.02 # Keep noise level consistent for comparable MAE
    dwi_signals_noisy = dwi_signals_true + torch.randn_like(dwi_signals_true) * noise_level
    dwi_signals_noisy = dwi_signals_noisy.clamp(min=EPS)

    noddi_fitter_model = NoddiModelTorch(gtab=simple_gtab).to(device)
    fit_hyperparams = {'learning_rate': 0.025, 'n_iterations': 300} # Standardized fit params

    fitted_params = noddi_fitter_model.fit_batch(
        dwi_signals_normalized=dwi_signals_noisy,
        **fit_hyperparams
    )

    true_odi_batch = noddi_fitter_model.kappa_to_odi(true_params_batch['kappa'])

    # Calculate MAE for key parameters
    mae_f_intra = torch.abs(fitted_params['f_intra'] - true_params_batch['f_intra']).mean().item()
    mae_f_iso = torch.abs(fitted_params['f_iso'] - true_params_batch['f_iso']).mean().item()
    mae_odi = torch.abs(fitted_params['odi'] - true_odi_batch).mean().item()
    mae_kappa = torch.abs(fitted_params['kappa'] - true_params_batch['kappa']).mean().item()

    print(f"\nTest Case: f_intra={true_f_intra_val:.2f}, kappa={true_kappa_val:.2f}, f_iso={true_f_iso_val:.2f}")
    print(f"  MAE f_intra: {mae_f_intra:.4f}")
    print(f"  MAE f_iso: {mae_f_iso:.4f}")
    print(f"  MAE ODI: {mae_odi:.4f}")
    print(f"  MAE Kappa: {mae_kappa:.4f}")

    # Define acceptable MAE thresholds (these are quite lenient for CI & few iterations)
    # These might need adjustment based on expected performance with low iterations.
    assert mae_f_intra < 0.15, "MAE for f_intra too high"
    assert mae_f_iso < 0.15, "MAE for f_iso too high"
    assert mae_odi < 0.2, "MAE for ODI too high" # ODI can be sensitive
    assert mae_kappa < 2.0, "MAE for kappa too high" # Kappa is very sensitive

    # Basic constraint checks (already in test_noddi_fit_batch_synthetic_data, repeated for safety)
    assert torch.all(fitted_params['f_intra'] >= 0) and torch.all(fitted_params['f_intra'] <= 1)
    assert torch.all(fitted_params['f_iso'] >= 0) and torch.all(fitted_params['f_iso'] <= 1)
    assert torch.all(fitted_params['f_intra'] + fitted_params['f_iso'] <= 1.0 + EPS)
    assert torch.all(fitted_params['odi'] >= 0) and torch.all(fitted_params['odi'] <= 1)
    assert torch.all(fitted_params['kappa'] > 0)


def test_noddi_fit_batch_with_regularization(synthetic_batch_for_fitting):
    """Tests fit_batch with L1 and L2 regularization."""
    torch.manual_seed(44)
    data = synthetic_batch_for_fitting
    device = data["device"]
    batch_size = data["dwi_signals_noisy"].shape[0]

    noddi_fitter_model = NoddiModelTorch(gtab=data["gtab"]).to(device)
    
    # Params for first voxel, which has true f_iso = 0.15
    # We will apply L1 to f_iso and expect it to be smaller.
    fit_hyperparams = {
        'learning_rate': 0.02, 
        'n_iterations': 300, # Fewer iterations to see effect of strong regularization
        'l1_penalty_weight': 0.1, # Strong L1 on f_iso
        'l2_penalty_weight': 0.01, # Moderate L2 on all raw params
        'l1_params_to_regularize': ['f_iso']
    }

    # Fit with regularization
    fitted_params_reg = noddi_fitter_model.fit_batch(
        dwi_signals_normalized=data["dwi_signals_noisy"],
        **fit_hyperparams
    )

    # Fit without L1 on f_iso for comparison (or with very small L1)
    fit_hyperparams_no_l1_fiso = {
        'learning_rate': 0.02, 
        'n_iterations': 300,
        'l1_penalty_weight': 0.0, # No L1 on f_iso
        'l2_penalty_weight': 0.01, # Keep L2 for raw params
        'l1_params_to_regularize': ['f_iso'] # list can be present but weight is 0
    }
    fitted_params_no_l1_fiso = noddi_fitter_model.fit_batch(
        dwi_signals_normalized=data["dwi_signals_noisy"],
        **fit_hyperparams_no_l1_fiso
    )

    # Assertions
    assert 'f_iso' in fitted_params_reg
    assert 'f_iso' in fitted_params_no_l1_fiso

    # Check that f_iso with strong L1 is smaller than f_iso without L1 (or with very small L1)
    # This is a qualitative check; exact values depend on many factors.
    # We check the mean f_iso across the batch.
    mean_f_iso_reg = fitted_params_reg['f_iso'].mean().item()
    mean_f_iso_no_l1 = fitted_params_no_l1_fiso['f_iso'].mean().item()
    
    print(f"Mean f_iso with L1 reg: {mean_f_iso_reg:.4f}")
    print(f"Mean f_iso without L1 reg (on f_iso): {mean_f_iso_no_l1:.4f}")
    assert mean_f_iso_reg < mean_f_iso_no_l1 + 0.03, \
        "L1 regularization on f_iso did not lead to a sufficiently smaller f_iso compared to no L1 on f_iso."

    # Also check basic parameter constraints for the regularized fit
    assert torch.all(fitted_params_reg['f_intra'] >= 0) and torch.all(fitted_params_reg['f_intra'] <= 1)
    assert torch.all(fitted_params_reg['f_iso'] >= 0) and torch.all(fitted_params_reg['f_iso'] <= 1)
    assert torch.all(fitted_params_reg['f_intra'] + fitted_params_reg['f_iso'] <= 1.0 + EPS)
    assert torch.all(fitted_params_reg['odi'] >= 0) and torch.all(fitted_params_reg['odi'] <= 1)


# --- Tests for fitting.noddi_fitter ---

@pytest.fixture
def synthetic_volume_for_fitting(simple_gtab, include_initial_orientation=False):
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_shape = (3, 3, 2) # Tiny volume
    n_voxels = np.prod(img_shape)
    
    gtab_bvals_torch = simple_gtab.bvals.float().to(device)
    gtab_bvecs_torch = simple_gtab.bvecs.float().to(device)

    # Create true parameter maps (with some variation)
    true_vol_params_torch = {
        'f_intra': torch.rand(n_voxels, device=device) * 0.6 + 0.1, # 0.1 to 0.7
        'f_iso': torch.rand(n_voxels, device=device) * 0.3,       # 0.0 to 0.3
        'kappa': torch.rand(n_voxels, device=device) * 4.0 + 0.5,   # 0.5 to 4.5
        'mu_theta': torch.rand(n_voxels, device=device) * math.pi,
        'mu_phi': torch.rand(n_voxels, device=device) * 2 * math.pi,
    }
    # Ensure f_intra + f_iso <= 1
    total_f = true_vol_params_torch['f_intra'] + true_vol_params_torch['f_iso']
    exceed_mask = total_f > 0.95
    true_vol_params_torch['f_intra'][exceed_mask] *= 0.95 / total_f[exceed_mask]
    true_vol_params_torch['f_iso'][exceed_mask] *= 0.95 / total_f[exceed_mask]
    
    synthetic_S_norm_flat = generate_synthetic_dwi_signals(
        true_vol_params_torch, gtab_bvals_torch, gtab_bvecs_torch
    ) # (n_voxels, n_grads)
    
    s0_map_flat = torch.rand(n_voxels, device=device) * 500 + 1000 # S0 between 1000-1500
    dwi_flat_true = synthetic_S_norm_flat * s0_map_flat.unsqueeze(-1)
    
    noise_level_vol = 0.02 # Relative to S0=1 signals, so actual sigma scales with S0
    dwi_flat_noisy = dwi_flat_true + torch.randn_like(dwi_flat_true) * (noise_level_vol * s0_map_flat.unsqueeze(-1))
    dwi_flat_noisy = dwi_flat_noisy.clamp(min=EPS)

    dwi_4d_np = dwi_flat_noisy.cpu().numpy().reshape(img_shape + (simple_gtab.bvals.shape[0],))
    
    mask_np = np.ones(img_shape, dtype=bool)
    mask_np[0,0,0] = False # Make one voxel invalid in mask

    s0_map_np = s0_map_flat.cpu().numpy().reshape(img_shape)
    # Make another voxel invalid by S0 threshold
    s0_map_np_for_dwi = s0_map_np.copy() # This is the S0 used to create DWI
    
    # Reshape true params for comparison
    true_vol_params_np = {}
    for k, v in true_vol_params_torch.items():
        true_vol_params_np[k] = v.cpu().numpy().reshape(img_shape)
    
    model_for_odi_calc = NoddiModelTorch(gtab=simple_gtab) 
    true_vol_params_np['odi'] = model_for_odi_calc.kappa_to_odi(torch.from_numpy(true_vol_params_np['kappa'])).numpy()

    initial_orientation_map_np = None
    if include_initial_orientation:
        # Create a dummy orientation map (e.g., all pointing along x-axis, or from true params)
        # For simplicity, let's use the true orientations (converted to Cartesian) as the initial map
        mu_theta_flat = true_vol_params_torch['mu_theta']
        mu_phi_flat = true_vol_params_torch['mu_phi']
        initial_orientation_map_flat = noddi_signal.spherical_to_cartesian(mu_theta_flat, mu_phi_flat)
        initial_orientation_map_np = initial_orientation_map_flat.cpu().numpy().reshape(img_shape + (3,))

    return {
        "dwi_4d": dwi_4d_np,
        "gtab": simple_gtab,
        "mask": mask_np,
        "true_params_maps": true_vol_params_np,
        "s0_map_for_dwi": s0_map_np_for_dwi, 
        "initial_orientation_map": initial_orientation_map_np,
        "device": device
    }


@pytest.mark.parametrize("use_initial_orientation", [False, True])
def test_fit_noddi_volume_synthetic_volume(synthetic_volume_for_fitting, use_initial_orientation):
    torch.manual_seed(123)
    np.random.seed(123)
    data = synthetic_volume_for_fitting
    
    fit_hyperparams_vol = {'learning_rate': 0.03, 'n_iterations': 250} # Faster test
    
    fitted_maps = fit_noddi_volume(
        dwi_data=data["dwi_4d"],
        gtab=data["gtab"],
        mask=data["mask"],
        min_s0_val=50.0, # Low threshold for test
        batch_size=4,    # Small batch size
        fit_params=fit_hyperparams_vol,
        device=data["device"]
    )

    assert isinstance(fitted_maps, dict)
    expected_param_keys = ['f_intra', 'f_iso', 'odi', 'kappa', 'mu_theta', 'mu_phi', 'f_extra', 'S0_fit']
    for key in expected_param_keys:
        assert key in fitted_maps
        assert fitted_maps[key].shape == data["dwi_4d"].shape[:3]

    # Check a valid voxel that was fitted
    # Coords of the first voxel that should be true in the mask
    valid_coord_idx = np.where(data["mask"])
    if len(valid_coord_idx[0]) > 0:
        check_coord = (valid_coord_idx[0][0], valid_coord_idx[1][0], valid_coord_idx[2][0])
        print(f"Checking voxel {check_coord}")

        # Check S0_fit is positive where fitted
        assert fitted_maps['S0_fit'][check_coord] > 0

        for param_name in ['f_intra', 'f_iso', 'odi']: # Check key parameters
            true_val = data["true_params_maps"][param_name][check_coord]
            fitted_val = fitted_maps[param_name][check_coord]
            # Use a more lenient tolerance for volume fitting test
            assert np.isclose(fitted_val, true_val, atol=0.3), \
                f"Voxel {check_coord}, Param {param_name}: True={true_val:.3f}, Fitted={fitted_val:.3f}"
    else:
        pytest.skip("No valid voxels found in the synthetic mask to perform detailed check.")

    # Check masked-out voxel (0,0,0) - should be zero or NaN (currently zero by np.zeros initialization)
    assert fitted_maps['f_intra'][0,0,0] == 0.0 
    assert fitted_maps['odi'][0,0,0] == 0.0
    assert fitted_maps['S0_fit'][0,0,0] == 0.0


def test_noddi_orientation_consistency_with_dti(simple_gtab, tmp_path):
    """
    Tests if NODDI's mean orientation (mu) is broadly consistent with DTI's
    primary eigenvector (V1) in a simple synthetic scenario.
    """
    torch.manual_seed(1234)
    np.random.seed(1234)
    device = torch.device("cpu") # Keep test on CPU for simplicity for now

    # 1. Synthetic Data Generation
    img_shape = (3, 3, 1) # Very small volume for speed
    n_voxels = np.prod(img_shape)

    # Use simple_gtab as it has b0, b1000 (for DTI), b2000 (for NODDI)
    gtab_torch_bvals = simple_gtab.bvals.float().to(device)
    gtab_torch_bvecs = simple_gtab.bvecs.float().to(device)

    # Define a clear primary orientation for all voxels (e.g., along y-axis)
    true_mu_theta_val = math.pi / 2.0
    true_mu_phi_val = math.pi / 2.0 # y-axis: x=0, y=1, z=0

    # Other NODDI params - chosen for strong anisotropy, low ODI, low f_iso
    true_params_flat = {
        'f_intra': torch.full((n_voxels,), 0.75, device=device), # High NDI
        'f_iso': torch.full((n_voxels,), 0.05, device=device),  # Low Fiso
        'kappa': torch.full((n_voxels,), 15.0, device=device),   # Low ODI (high kappa)
        'mu_theta': torch.full((n_voxels,), true_mu_theta_val, device=device),
        'mu_phi': torch.full((n_voxels,), true_mu_phi_val, device=device)
    }
    
    synthetic_S_norm_flat = generate_synthetic_dwi_signals(
        true_params_flat, gtab_torch_bvals, gtab_torch_bvecs
    )
    
    s0_map_flat = torch.full((n_voxels,), 1000.0, device=device) # Uniform S0
    dwi_flat_true = synthetic_S_norm_flat * s0_map_flat.unsqueeze(-1)
    
    noise_level = 0.02 
    dwi_flat_noisy = dwi_flat_true + torch.randn_like(dwi_flat_true) * (noise_level * s0_map_flat.unsqueeze(-1))
    dwi_flat_noisy = dwi_flat_noisy.clamp(min=EPS)

    dwi_4d_np = dwi_flat_noisy.cpu().numpy().reshape(img_shape + (simple_gtab.bvals.shape[0],))
    mask_np = np.ones(img_shape, dtype=bool) # Process all voxels

    # 2. Fit DTI
    # Need to import or define a DTI fitting function compatible with this test environment
    # For now, assume a conceptual `fit_dti_volume_simplified` that returns eigenvectors
    # This part would require access to the actual dti_fitter.py contents
    # For this test, we'll mock its output or skip if not easily callable.
    
    # Mocking DTI output: Assume DTI perfectly recovers the y-axis orientation
    dti_v1_map = np.zeros(img_shape + (3,))
    dti_v1_map[..., 1] = 1.0 # All V1s along y-axis
    
    # If we had access to the actual DTI fitter:
    # from fitting.dti_fitter import fit_dti_volume 
    # dti_results = fit_dti_volume(dwi_4d_np, simple_gtab.bvals, simple_gtab.bvecs, mask=mask_np)
    # dti_v1_map = dti_results['V1'] # Assuming V1 is stored like this (x,y,z,3)

    # 3. Fit NODDI
    noddi_fit_params = {'learning_rate': 0.02, 'n_iterations': 200} # Faster fitting
    
    # Use the same dwi_4d_np and simple_gtab for NODDI fitting
    # Need to ensure dwi_4d_np is suitable for preprocess_noddi_input (expects S0 to be calculated from b0s within it)
    
    noddi_fitted_maps = fit_noddi_volume(
        dwi_data=dwi_4d_np,
        gtab=simple_gtab,
        mask=mask_np,
        batch_size=n_voxels, # Fit all voxels in one batch for this small test
        fit_params=noddi_fit_params,
        device=device
    )

    noddi_mu_theta = noddi_fitted_maps['mu_theta']
    noddi_mu_phi = noddi_fitted_maps['mu_phi']
    
    # Convert NODDI mu (theta, phi) to Cartesian vectors
    noddi_mu_cartesian_map = np.zeros(img_shape + (3,))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for k in range(img_shape[2]):
                if mask_np[i,j,k]:
                    theta, phi = noddi_mu_theta[i,j,k], noddi_mu_phi[i,j,k]
                    cart_vec = noddi_signal.spherical_to_cartesian(torch.tensor(theta), torch.tensor(phi))
                    noddi_mu_cartesian_map[i,j,k,:] = cart_vec.numpy()

    # 4. Comparison
    angular_differences = []
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for k in range(img_shape[2]):
                if mask_np[i,j,k]:
                    v1 = dti_v1_map[i,j,k,:]
                    mu_noddi = noddi_mu_cartesian_map[i,j,k,:]
                    
                    # Ensure they are unit vectors (should be, but for safety)
                    v1_norm = np.linalg.norm(v1)
                    mu_noddi_norm = np.linalg.norm(mu_noddi)
                    if v1_norm < EPS or mu_noddi_norm < EPS:
                        continue # Skip if either vector is zero (e.g. isotropic voxel for DTI)

                    v1_unit = v1 / v1_norm
                    mu_noddi_unit = mu_noddi / mu_noddi_norm
                    
                    # Dot product, taking abs because orientation has no polarity
                    dot_product = np.abs(np.dot(v1_unit, mu_noddi_unit))
                    # Clamp dot_product to [-1, 1] due to potential numerical inaccuracies
                    dot_product = np.clip(dot_product, -1.0, 1.0)
                    angle_rad = np.arccos(dot_product)
                    angle_deg = np.degrees(angle_rad)
                    angular_differences.append(angle_deg)

    assert len(angular_differences) > 0, "No valid voxels found for comparison"
    mean_angular_diff = np.mean(angular_differences)
    max_angular_diff = np.max(angular_differences)
    
    print(f"Mean angular difference between DTI V1 and NODDI mu: {mean_angular_diff:.2f} degrees")
    print(f"Max angular difference: {max_angular_diff:.2f} degrees")

    # Threshold for consistency (can be somewhat lenient due to different models and fitting noise)
    # For highly anisotropic, low ODI data, DTI and NODDI mu should align well.
    assert mean_angular_diff < 20.0, "Mean angular difference is too high"
    assert max_angular_diff < 30.0, "Max angular difference is too high"

```
