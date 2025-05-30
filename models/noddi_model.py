import torch
import torch.nn as nn
import torch.optim as optim
import math
from utils.pytorch_gradient_utils import PyTorchGradientTable # New import
from . import noddi_signal

# Default constants (can be overridden in __init__)
d_intra_default = 1.7e-3  # mm^2/s
d_iso_default = 3.0e-3    # mm^2/s
EPS = 1e-8 # Small epsilon for numerical stability

# Import the new conversion function
from .noddi_signal import cartesian_to_spherical_mu

class NoddiModelTorch(nn.Module):
    """PyTorch module for fitting the NODDI (Neurite Orientation Dispersion and Density Imaging) model.

    This class encapsulates the NODDI model parameters, their transformations for constrained
    optimization, and a batch-wise fitting procedure. It uses the signal generation functions
    from `models.noddi_signal` to predict diffusion-weighted signals and optimizes
    parameters by minimizing the Mean Squared Error (MSE) against observed signals.

    The model parameters that are fitted per voxel include:
    - f_intra (vic): Volume fraction of the intra-cellular compartment (neurites).
    - f_iso (viso): Volume fraction of the isotropic (CSF) compartment.
    - kappa: Concentration parameter of the Watson distribution, related to ODI.
    - mu_theta, mu_phi: Spherical angles defining the mean orientation of neurites.

    The extra-cellular volume fraction (f_extra) is derived as 1 - f_intra - f_iso.
    The Orientation Dispersion Index (ODI) is derived from kappa.

    Args:
        gtab (PyTorchGradientTable): PyTorchGradientTable object containing b-values and b-vectors
            for the diffusion acquisition scheme.
        d_intra (float, optional): Fixed intrinsic diffusivity for the intra-cellular
            compartment (d_parallel for sticks, e.g., 1.7e-3 mm^2/s).
            Defaults to `d_intra_default` from `noddi_signal` module.
        d_iso (float, optional): Fixed diffusivity for the isotropic compartment
            (e.g., 3.0e-3 mm^2/s for CSF). Defaults to `d_iso_default` from `noddi_signal` module.

    Attributes:
        gtab (gradient_table): The GradientTable object.
        b_values (torch.Tensor): Float tensor of b-values. Shape: (N_gradients,).
        b_vectors (torch.Tensor): Float tensor of b-vectors (normalized gradient directions).
            Shape: (N_gradients, 3).
        d_intra (float): Intra-cellular intrinsic diffusivity.
        d_iso (float): Isotropic diffusivity.
        b0_mask (torch.Tensor): Boolean tensor indicating which entries in `b_values`
            correspond to b0 images.
    """
    def __init__(self, gtab: PyTorchGradientTable, d_intra: float = d_intra_default, d_iso: float = d_iso_default):
        super().__init__()
        # self.gtab = gtab # No longer need to store the whole gtab object if not used elsewhere
        self.b_values = gtab.bvals.float()   # PyTorchGradientTable returns tensors
        self.b_vectors = gtab.bvecs.float()  # Ensure float type
        self.b0_mask = gtab.b0s_mask.bool()    # Ensure bool type
        self.d_intra = d_intra
        self.d_iso = d_iso

        # EPS is still used elsewhere, e.g. for clamping in _inv_sigmoid, _inv_softplus, kappa_to_odi

    # --- Parameter Transformation Helper Functions ---
    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid function to map real numbers to (0, 1)."""
        return torch.sigmoid(x)

    def _inv_sigmoid(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse sigmoid (logit) function to map (0, 1) to real numbers."""
        y = y.clamp(min=EPS, max=1.0 - EPS)  # Clamp input to avoid log(0) or log(inf)
        return torch.log(y / (1.0 - y))

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        """Softplus function to map real numbers to (0, inf)."""
        return nn.functional.softplus(x)

    def _inv_softplus(self, y: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
        """Inverse of softplus function to map (0, inf) to real numbers.

        y = (1/beta) * log(1 + exp(beta*x))
        x = (1/beta) * log(exp(beta*y) - 1)

        Args:
            y (torch.Tensor): Input tensor (values from softplus).
            beta (float, optional): Beta value used in softplus. Defaults to 1.0.
            threshold (float, optional): Threshold for numerical stability approximation.
                If beta*y > threshold, y is returned directly as x approx y for large y.
                Defaults to 20.0.

        Returns:
            torch.Tensor: Tensor in the unconstrained space.
        """
        y = y.clamp(min=EPS)  # Ensure y is positive
        val = beta * y
        # For large val, exp(val)-1 might be unstable or overflow if not handled.
        # If val is large, softplus(x) approx x, so inv_softplus(y) approx y.
        # Otherwise, use the direct inverse formula.
        return torch.where(val > threshold, y, (1.0 / beta) * torch.log(torch.exp(val) - 1.0 + EPS).clamp(min=-100))


    def _get_initial_unconstrained_params(
        self,
        batch_size: int,
        device: torch.device,
        initial_mu_batch: Optional[torch.Tensor] = None
    ) -> nn.ParameterDict:
        """Initializes unconstrained (raw) parameters for fitting a batch of voxels.

        These initial values are in the transformed (unconstrained) space. They are set
        to correspond to typical physical values, or can be guided by `initial_mu_batch`.

        Args:
            batch_size (int): Number of voxels in the batch.
            device (torch.device): PyTorch device for the parameters.
            initial_mu_batch (Optional[torch.Tensor], optional): A tensor of initial
                mean orientations (mu) in Cartesian coordinates (x,y,z) for each voxel
                in the batch. Shape: (batch_size, 3). If provided, these are used to
                initialize `raw_mu_theta` and `raw_mu_phi`. Defaults to None (standard initialization).

        Returns:
            nn.ParameterDict: Dictionary of initial unconstrained parameters.
        """
        # Initial guesses in constrained (physical) space (used if initial_mu_batch is None for orientations)
        init_f_intra_constrained = 0.5
        init_f_iso_norm_constrained = 0.2  # f_iso = 0.2 * (1 - f_intra) -> 0.1 if f_intra = 0.5
        init_kappa_constrained = 2.0
        
        default_mu_theta_constrained = torch.full((batch_size,), math.pi / 2.0, device=device)
        default_mu_phi_constrained = torch.full((batch_size,), math.pi / 2.0, device=device)

        params = nn.ParameterDict()
        params['raw_f_intra'] = nn.Parameter(
            torch.full((batch_size,), self._inv_sigmoid(torch.tensor(init_f_intra_constrained, device=device)), device=device)
        )
        params['raw_f_iso_norm'] = nn.Parameter(
            torch.full((batch_size,), self._inv_sigmoid(torch.tensor(init_f_iso_norm_constrained, device=device)), device=device)
        )
        params['raw_kappa'] = nn.Parameter(
            torch.full((batch_size,), self._inv_softplus(torch.tensor(init_kappa_constrained, device=device)), device=device)
        )

        if initial_mu_batch is not None:
            if initial_mu_batch.shape[0] != batch_size or initial_mu_batch.shape[1] != 3:
                raise ValueError(f"initial_mu_batch must have shape (batch_size, 3), "
                                 f"got {initial_mu_batch.shape}")
            
            # Normalize initial_mu_batch to ensure they are unit vectors
            norm_initial_mu = torch.linalg.norm(initial_mu_batch, dim=1, keepdim=True) + EPS
            initial_mu_unit_vecs = initial_mu_batch / norm_initial_mu
            
            # Convert Cartesian mu to spherical theta and phi
            # Ensure cartesian_to_spherical_mu is imported or defined in this class/module
            init_mu_theta_constrained, init_mu_phi_constrained = \
                cartesian_to_spherical_mu(initial_mu_unit_vecs.to(device))
        else:
            init_mu_theta_constrained = default_mu_theta_constrained
            init_mu_phi_constrained = default_mu_phi_constrained

        # Transform constrained orientation angles to unconstrained space for initialization
        # mu_theta is in [0, pi], mu_phi is in [0, 2*pi]
        # Sigmoid maps to (0,1). We need raw_param such that sigmoid(raw_param) = angle / range_max
        params['raw_mu_theta'] = nn.Parameter(
            self._inv_sigmoid(init_mu_theta_constrained / math.pi) # Normalize to (0,1) before inv_sigmoid
        )
        params['raw_mu_phi'] = nn.Parameter(
            self._inv_sigmoid(init_mu_phi_constrained / (2 * math.pi)) # Normalize to (0,1) before inv_sigmoid
        )
        return params

    def _unconstrained_to_constrained(self, raw_params: nn.ParameterDict) -> Dict[str, torch.Tensor]:
        """Converts unconstrained (raw) parameters to physically meaningful constrained NODDI parameters.

        This function applies transformations (sigmoid, softplus) to ensure that the
        parameters used in the signal model adhere to their physical bounds (e.g., volume
        fractions between 0 and 1, kappa > 0, angles within their ranges).

        Args:
            raw_params (nn.ParameterDict): A dictionary of raw (unconstrained) parameters
                from the optimizer. Expected keys: 'raw_f_intra', 'raw_f_iso_norm',
                'raw_kappa', 'raw_mu_theta', 'raw_mu_phi'.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of constrained NODDI parameters.
                Keys: 'f_intra', 'f_iso', 'kappa', 'mu_theta', 'mu_phi'.
        """
        # f_intra is transformed from unconstrained raw_f_intra to be in [0, 1]
        f_intra = self._sigmoid(raw_params['raw_f_intra'])
        
        # f_iso is modeled as a fraction of the remaining non-intra space (1 - f_intra)
        f_iso_norm = self._sigmoid(raw_params['raw_f_iso_norm'])
        f_iso = f_iso_norm * (1.0 - f_intra)
        
        kappa = self._softplus(raw_params['raw_kappa']).clamp(min=EPS)
        
        # mu_theta (polar angle) from raw_mu_theta to [0, pi]
        # Here, raw_mu_theta was initialized based on angle/pi.
        mu_theta = self._sigmoid(raw_params['raw_mu_theta']) * math.pi 
        mu_theta = mu_theta.clamp(min=EPS, max=math.pi - EPS)

        # mu_phi (azimuthal angle) from raw_mu_phi to [0, 2*pi]
        # Here, raw_mu_phi was initialized based on angle/(2*pi).
        mu_phi = self._sigmoid(raw_params['raw_mu_phi']) * (2 * math.pi)

        return {
            'f_intra': f_intra,
            'f_iso': f_iso,
            'kappa': kappa,
            'mu_theta': mu_theta,
            'mu_phi': mu_phi,
        }

    def kappa_to_odi(self, kappa: torch.Tensor) -> torch.Tensor:
        """Converts Watson concentration parameter kappa to Orientation Dispersion Index (ODI).

        The relationship used is ODI = (2/pi) * atan(1/kappa), which is a common
        approximation. ODI ranges from 0 (no dispersion, kappa->inf) to 1
        (isotropic dispersion, kappa->0).

        Args:
            kappa (torch.Tensor): Watson concentration parameter. Must be > 0.
                Shape: (...).

        Returns:
            torch.Tensor: Orientation Dispersion Index (ODI). Shape: (...).
        """
        kappa = kappa.clamp(min=EPS)  # Ensure kappa is positive to avoid division by zero or invalid atan input
        odi = (2.0 / math.pi) * torch.atan(1.0 / kappa)
        return odi


    def fit_batch(self,
                  dwi_signals_normalized: torch.Tensor,
                  learning_rate: float = 1e-2,
                  n_iterations: int = 500,
                  initial_params: Optional[nn.ParameterDict] = None,
                  initial_mu_batch: Optional[torch.Tensor] = None,
                  l1_penalty_weight: float = 0.0,
                  l2_penalty_weight: float = 0.0,
                  l1_params_to_regularize: Optional[List[str]] = None
                  ) -> Dict[str, torch.Tensor]:
        """Fits the NODDI model to a batch of normalized DWI signals (S_observed / S0).

        This method uses gradient descent (Adam optimizer) to find the NODDI parameters
        that best explain the observed normalized DWI signals for a batch of voxels.
        The loss function is the Mean Squared Error (MSE) between the predicted signals
        (from `noddi_signal.noddi_signal_model`) and the input `dwi_signals_normalized`.
        b0 images (where b-value < EPS based on `self.b0_mask`) are excluded from the loss calculation.
        Optional L1 and L2 regularization terms can be added to the loss.

        Args:
            dwi_signals_normalized (torch.Tensor): Batch of normalized DWI signals.
                Shape: (batch_size, N_gradients). These are S_observed/S0.
            learning_rate (float, optional): Learning rate for the Adam optimizer.
                Defaults to 1e-2.
            n_iterations (int, optional): Number of optimization iterations.
                Defaults to 500.
            initial_params (Optional[nn.ParameterDict], optional): Optional pre-initialized
                unconstrained parameters. If None, `_get_initial_unconstrained_params` is called.
                If provided, `initial_mu_batch` is ignored by `_get_initial_unconstrained_params`.
                Defaults to None.
            initial_mu_batch (Optional[torch.Tensor], optional): Tensor of initial mean
                orientations (mu) in Cartesian coordinates (x,y,z) for each voxel.
                Shape: (batch_size, 3). Used by `_get_initial_unconstrained_params` if
                `initial_params` is None. Defaults to None.
            l1_penalty_weight (float, optional): Weight for L1 regularization.
                Applied to parameters specified in `l1_params_to_regularize`. Defaults to 0.0.
            l2_penalty_weight (float, optional): Weight for L2 regularization.
                Applied to all *unconstrained* (raw) parameters. Defaults to 0.0.
            l1_params_to_regularize (Optional[List[str]], optional): List of *constrained*
                parameter names (e.g., ['f_iso', 'f_intra']) to which L1 regularization
                should be applied. If None or empty, L1 regularization is not applied to specific
                constrained parameters. Defaults to None.


        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the fitted constrained NODDI
                parameters for the batch. Each parameter tensor has shape (batch_size,).
                Includes derived 'odi' (Orientation Dispersion Index) and 'f_extra'
                (extra-cellular volume fraction). Parameters are detached from the
                computation graph.
        """
        batch_size = dwi_signals_normalized.shape[0]
        device = dwi_signals_normalized.device
        
        self.b_values = self.b_values.to(device)
        self.b_vectors = self.b_vectors.to(device)
        self.b0_mask = self.b0_mask.to(device)

        if initial_params is None:
            # Use initial_mu_batch if provided to _get_initial_unconstrained_params
            unconstrained_params = self._get_initial_unconstrained_params(
                batch_size, device, initial_mu_batch=initial_mu_batch
            )
        else:
            unconstrained_params = nn.ParameterDict(
                {k: nn.Parameter(v.clone().detach().to(device)) for k, v in initial_params.items()}
            )

        optimizer = optim.Adam(list(unconstrained_params.values()), lr=learning_rate)
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            constrained_noddi_params = self._unconstrained_to_constrained(unconstrained_params)
            
            predicted_signals = noddi_signal.noddi_signal_model(
                params=constrained_noddi_params,
                b_values=self.b_values,
                gradient_directions=self.b_vectors,
                d_intra_val=self.d_intra,
                d_iso_val=self.d_iso
            )
            
            if predicted_signals.shape != dwi_signals_normalized.shape:
                raise ValueError(f"Shape mismatch: predicted {predicted_signals.shape}, target {dwi_signals_normalized.shape}")

            # Primary loss: Mean Squared Error on non-b0 signals
            mse_loss = nn.functional.mse_loss(
                predicted_signals[:, ~self.b0_mask],
                dwi_signals_normalized[:, ~self.b0_mask]
            )
            total_loss = mse_loss

            # L1 Regularization on specified *constrained* parameters
            if l1_penalty_weight > 0 and l1_params_to_regularize:
                l1_reg_term = torch.tensor(0.0, device=device)
                for param_name in l1_params_to_regularize:
                    if param_name in constrained_noddi_params:
                        l1_reg_term += torch.abs(constrained_noddi_params[param_name]).sum()
                    else:
                        # This could also apply to raw_params if needed, but typically for constrained
                        logger.warning(f"L1 regularization specified for '{param_name}', but it's not "
                                       f"a recognized constrained parameter. Skipping.")
                total_loss += l1_penalty_weight * l1_reg_term

            # L2 Regularization on all *unconstrained* (raw) parameters
            if l2_penalty_weight > 0:
                l2_reg_term = torch.tensor(0.0, device=device)
                for raw_param_name in unconstrained_params:
                    l2_reg_term += torch.pow(unconstrained_params[raw_param_name], 2).sum()
                total_loss += l2_penalty_weight * l2_reg_term
            
            total_loss.backward()
            optimizer.step()
            
            # Optional: Print loss for monitoring
            # if (i + 1) % (n_iterations // 10) == 0:
            #     print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.6f}")

        # Get final constrained parameters
        with torch.no_grad():
            fitted_constrained_params = self._unconstrained_to_constrained(unconstrained_params)
            # Add ODI
            fitted_constrained_params['odi'] = self.kappa_to_odi(fitted_constrained_params['kappa'])
            # Add f_extra
            fitted_constrained_params['f_extra'] = (1.0 - fitted_constrained_params['f_intra'] - fitted_constrained_params['f_iso']).clamp(min=0.0, max=1.0)
        
        return {k: v.detach() for k, v in fitted_constrained_params.items()}


if __name__ == '__main__':
    import numpy as np
    # from dipy.core.gradients import gradient_table as create_gradient_table # Removed

    # 1. Create a synthetic GradientTable using PyTorchGradientTable
    n_grads = 32
    bvals_np = np.random.uniform(0, 2500, n_grads).astype(np.float32)
    bvals_np[0:3] = 0 # Add some b0s
    bvecs_np = np.random.randn(n_grads, 3).astype(np.float32)
    
    # Normalize b-vectors for DWIs and zero for b0s, as expected by PyTorchGradientTable
    b0_threshold_example = 50.0 # Using a consistent threshold
    dwi_mask_np = bvals_np > b0_threshold_example
    
    # Handle b-vectors for DWIs
    dwi_bvecs = bvecs_np[dwi_mask_np]
    if dwi_bvecs.size > 0: # Ensure there are DWI b-vectors to normalize
        norms = np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)
        # Avoid division by zero for zero-norm vectors (though ideally, input bvecs for DWIs shouldn't be zero)
        non_zero_norms_mask = (norms > EPS).squeeze()
        dwi_bvecs_normalized = dwi_bvecs.copy() # Work on a copy
        
        # Apply normalization only to vectors with non-zero norm
        if np.any(non_zero_norms_mask): # Check if there's anything to normalize
            dwi_bvecs_normalized[non_zero_norms_mask] = dwi_bvecs[non_zero_norms_mask] / norms[non_zero_norms_mask]
        
        bvecs_np[dwi_mask_np] = dwi_bvecs_normalized

    # Ensure b-vectors for b0s are [0,0,0]
    bvecs_np[~dwi_mask_np] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    gtab_torch_instance = PyTorchGradientTable(bvals_np, bvecs_np, b0_threshold=b0_threshold_example)
    print(f"Using PyTorchGradientTable with {gtab_torch_instance.bvals.shape[0]} gradients.")
    print(f"  Number of b0s: {torch.sum(gtab_torch_instance.b0s_mask).item()}")
    print(f"  Number of DWIs: {torch.sum(gtab_torch_instance.dwis_mask).item()}")

    # 2. Instantiate the model
    noddi_torch_model = NoddiModelTorch(gtab=gtab_torch_instance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noddi_torch_model.to(device)
    print(f"Model moved to {device}")

    # 3. Generate synthetic DWI data for a batch of voxels
    batch_s = 5 # Number of voxels to simulate and fit
    
    # True parameters (for generating synthetic data)
    true_f_intra = torch.tensor([0.6, 0.5, 0.7, 0.4, 0.65], device=device)
    true_f_iso = torch.tensor([0.1, 0.2, 0.05, 0.3, 0.15], device=device)
    true_kappa = torch.tensor([1.5, 3.0, 0.8, 5.0, 2.2], device=device) # ODI ~0.37, 0.2, 0.5, 0.12, 0.28
    true_mu_theta = torch.tensor([math.pi/2, math.pi/4, math.pi/3, math.pi/6, math.pi/2], device=device)
    true_mu_phi = torch.tensor([math.pi/2, 0.0, math.pi, math.pi/4, math.pi*1.5], device=device)

    true_noddi_params = {
        'f_intra': true_f_intra,
        'f_iso': true_f_iso,
        'kappa': true_kappa,
        'mu_theta': true_mu_theta,
        'mu_phi': true_mu_phi
    }

    # Generate signals using the signal model part
    with torch.no_grad():
        synthetic_S_norm = noddi_signal.noddi_signal_model(
            params=true_noddi_params,
            b_values=noddi_torch_model.b_values.to(device),
            gradient_directions=noddi_torch_model.b_vectors.to(device),
            d_intra_val=noddi_torch_model.d_intra,
            d_iso_val=noddi_torch_model.d_iso
        ) # Shape (batch_s, N_gradients)
    
    # Add some noise
    noise_level = 0.02
    synthetic_S_norm_noisy = synthetic_S_norm + torch.randn_like(synthetic_S_norm) * noise_level
    synthetic_S_norm_noisy = synthetic_S_norm_noisy.clamp(min=EPS)
    
    print(f"Generated synthetic DWI signals of shape: {synthetic_S_norm_noisy.shape}")

    # 4. Fit the model to the synthetic DWI data
    print("Starting batch fit...")
    fitted_params_batch = noddi_torch_model.fit_batch(
        dwi_signals_normalized=synthetic_S_norm_noisy,
        learning_rate=0.02, # Adjusted learning rate
        n_iterations=1000   # Increased iterations
    )
    print("Batch fit completed.")

    # 5. Display true and fitted parameters
    print("\n--- Comparison of True and Fitted Parameters (Batch) ---")
    for i in range(batch_s):
        print(f"\nVoxel {i+1}:")
        for param_name in ['f_intra', 'f_iso', 'odi', 'kappa', 'mu_theta', 'mu_phi', 'f_extra']:
            true_val_display = "N/A"
            if param_name == 'odi':
                true_val = noddi_torch_model.kappa_to_odi(true_noddi_params['kappa'][i])
                true_val_display = f"{true_val.item():.3f}"
            elif param_name == 'f_extra':
                 true_val = 1.0 - true_noddi_params['f_intra'][i] - true_noddi_params['f_iso'][i]
                 true_val_display = f"{true_val.item():.3f}"
            elif param_name in true_noddi_params:
                true_val_display = f"{true_noddi_params[param_name][i].item():.3f}"
            
            fitted_val = fitted_params_batch[param_name][i]
            print(f"  {param_name:<10}: True = {true_val_display}, Fitted = {fitted_val.item():.3f}")

    print("\nNote: mu_theta and mu_phi can have ambiguities (e.g. phi +/- 2pi, or (theta,phi) vs (pi-theta, phi+pi) for axial symmetry).")
    print("Kappa/ODI fitting can also be challenging.")

```
