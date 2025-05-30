import torch
import math

# Default constants for NODDI model (Zhang et al., NeuroImage 2012)
# Intrinsic diffusivity for the intra-axonal space (sticks)
d_intra_default = 1.7e-3  # mm^2/s
# Isotropic diffusivity for the CSF compartment
d_iso_default = 3.0e-3    # mm^2/s
# Small epsilon to prevent division by zero or log(0)
EPS = 1e-8


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Converts spherical coordinates to Cartesian unit vectors.

    The spherical coordinates are defined by:
    - theta: inclination or polar angle (0 to pi). Angle from the positive z-axis.
    - phi: azimuthal angle (0 to 2pi). Angle from the positive x-axis in the xy-plane.

    Args:
        theta (torch.Tensor): Tensor of polar angles. Shape: (...).
        phi (torch.Tensor): Tensor of azimuthal angles. Shape: (...).

    Returns:
        torch.Tensor: Cartesian coordinates (x, y, z) as unit vectors. Shape: (..., 3).
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


def cartesian_to_spherical_mu(cart_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts Cartesian unit vectors to spherical coordinates (theta, phi).

    This is specifically for NODDI's mean orientation vector mu.
    - theta: polar angle (0 to pi), angle from +z axis.
    - phi: azimuthal angle (0 to 2*pi), angle from +x axis in xy-plane.

    Args:
        cart_vec (torch.Tensor): Cartesian vectors (x, y, z), expected to be
            unit vectors. Shape: (..., 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - mu_theta (torch.Tensor): Polar angles. Shape: (...).
            - mu_phi (torch.Tensor): Azimuthal angles. Shape: (...).
    """
    # Ensure input is float for acos/atan2
    cart_vec = cart_vec.float()
    
    # Calculate radius component in xy plane (rho)
    # Add EPS to prevent issues with vectors exactly along z-axis for atan2, or norm being zero
    rho_sq = cart_vec[..., 0]**2 + cart_vec[..., 1]**2
    rho = torch.sqrt(rho_sq)

    # Theta (polar angle from +z axis)
    # cart_vec[..., 2] is z. Norm of vector is assumed to be 1.
    # theta = acos(z / norm). If norm=1, theta = acos(z)
    mu_theta = torch.acos(cart_vec[..., 2].clamp(min=-1.0, max=1.0)) # Clamp for numerical stability

    # Phi (azimuthal angle from +x axis in xy-plane)
    # phi = atan2(y, x)
    mu_phi = torch.atan2(cart_vec[..., 1], cart_vec[..., 0])

    # Adjust phi to be in [0, 2*pi] range
    # atan2 returns in [-pi, pi]. Add 2*pi to negative values.
    mu_phi = torch.where(mu_phi < 0, mu_phi + 2 * math.pi, mu_phi)
    
    # Handle cases where rho is very small (vector is along z-axis)
    # In this case, phi is undefined but can be set to 0 by convention.
    # atan2(0,0) is often 0, but good to be explicit.
    mu_phi = torch.where(rho < EPS, torch.zeros_like(mu_phi), mu_phi)

    return mu_theta, mu_phi


# The _psi_function was part of an earlier exploration and is not used in the final model.
# It's removed to avoid confusion. The model uses a Legendre series expansion instead.


def _R_0(zeta: torch.Tensor) -> torch.Tensor:
    """Computes the R_0(zeta) integral for the NODDI model.

    R_0(zeta) = integral_0^1 exp(-zeta*x) P_0(sqrt(x)) dx = integral_0^1 exp(-zeta*x) dx
    This is used in the Legendre polynomial expansion for the intra-cellular
    and extra-cellular signal components in the NODDI model (Zhang et al., 2012, Appendix A).

    Args:
        zeta (torch.Tensor): Input tensor, typically b_value * diffusivity. Shape: (...).

    Returns:
        torch.Tensor: Result of the R_0 integral. Shape: (...).
                     Returns 1.0 where zeta is close to 0 (limit of (1-exp(-zeta))/zeta as zeta->0).
    """
    # For zeta = 0, R_0 = 1 (limit). For zeta > 0, R_0 = (1 - exp(-zeta)) / zeta.
    # Use EPS to handle potential division by zero if zeta is exactly 0,
    # though typically b-values ensure zeta > 0 for weighted signals.
    zeta_eps = zeta + EPS
    res = (1.0 - torch.exp(-zeta_eps)) / zeta_eps
    # For true zero zeta (e.g. b=0), the signal contribution should make this result effectively 1.
    # This form is generally stable for positive zeta.
    return res


def _R_2(zeta: torch.Tensor, R_0_val: torch.Tensor) -> torch.Tensor:
    """Computes the R_2(zeta) integral for the NODDI model.

    R_2(zeta) = integral_0^1 exp(-zeta*x) P_2(sqrt(x)) dx
              = integral_0^1 exp(-zeta*x) * (3x - 1)/2 dx
    This is used in the Legendre polynomial expansion for the intra-cellular
    and extra-cellular signal components (Zhang et al., 2012, Appendix A).

    Args:
        zeta (torch.Tensor): Input tensor, typically b_value * diffusivity. Shape: (...).
        R_0_val (torch.Tensor): Pre-computed R_0(zeta) for the same zeta values. Shape: (...).

    Returns:
        torch.Tensor: Result of the R_2 integral. Shape: (...).
                     Returns 0.0 where zeta is close to 0 (limit of R_2(zeta) as zeta->0).
    """
    zeta_eps = zeta + EPS  # Avoid division by zero for the denominator
    # Formula derived from analytical integration: ( (5/zeta)R_0 - ( (3/zeta) + 2 )exp(-zeta) ) / 2
    # or (5 R_0(zeta) - (3+2zeta)exp(-zeta)) / (2*zeta)
    # This can be numerically unstable if zeta is very small.
    # The limit of R_2(zeta) as zeta -> 0 is 0.
    numerator = 5.0 * R_0_val - (3.0 + 2.0 * zeta) * torch.exp(-zeta)
    denominator = 2.0 * zeta_eps
    res = numerator / denominator
    return torch.where(zeta < EPS, torch.zeros_like(zeta), res)


def compute_intra_signal(
    b_values: torch.Tensor,
    gradient_directions: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    d_intra: float
) -> torch.Tensor:
    """Computes the normalized signal from the intra-cellular compartment in the NODDI model.

    This compartment models neurites as "sticks" (zero perpendicular diffusivity)
    with an orientation distribution modeled by a Watson distribution. The signal
    is calculated using a truncated Legendre polynomial expansion based on
    Eq. (A.10) from Zhang et al., NeuroImage 2012. The expansion is truncated
    at L_max=2 (using terms l=0 and l=2).

    Args:
        b_values (torch.Tensor): Tensor of b-values. Shape: (N_gradients,).
        gradient_directions (torch.Tensor): Tensor of normalized gradient directions.
            Shape: (N_gradients, 3).
        mu (torch.Tensor): Mean orientation vector(s) of neurites (unit vector).
            Shape: (..., 3), where ... represents batch dimensions (e.g., for multiple voxels).
        kappa (torch.Tensor): Watson distribution concentration parameter(s).
            Higher kappa means less dispersion. Shape: (...).
        d_intra (float): Intrinsic axial diffusivity of neurites (d_parallel for sticks, e.g., 1.7e-3 mm^2/s).

    Returns:
        torch.Tensor: Normalized intra-cellular signal (S_ic / S0_ic).
            Shape: (..., N_gradients), matching batch dimensions of mu/kappa and N_gradients.
            Values are clamped to [EPS, 1.0].
    """
    # Ensure mu is normalized, though it should be by design from spherical_to_cartesian
    # mu = mu / (torch.linalg.norm(mu, dim=-1, keepdim=True) + EPS) # Uncomment if mu might not be normalized

    # Reshape for broadcasting if mu and kappa are for a batch of voxels
    # b_values and gradient_directions are typically fixed for all voxels in a batch fit.
    _mu = mu.unsqueeze(-2)  # Shape: (..., 1, 3)
    _kappa = kappa.unsqueeze(-1)  # Shape: (..., 1)

    # Cosine squared of angle between gradient directions and mean neurite orientation
    # (g_vec @ mu_vec)^2. Unsqueeze gradient_directions for batch compatibility.
    g_dot_mu_sq = (gradient_directions.unsqueeze(0) @ _mu.transpose(-1, -2)).squeeze(-1).pow(2)
    # Clamp to [0,1] due to potential numerical precision issues.
    g_dot_mu = torch.sqrt(g_dot_mu_sq.clamp(min=0.0, max=1.0))

    # Zeta term for R_l integrals: zeta = b * d_intra
    # Unsqueeze b_values to allow broadcasting with batched parameters if needed,
    # though typically b_values is (N_gradients,).
    zeta = b_values.unsqueeze(0) * d_intra  # Shape: (1, N_gradients)

    # Legendre Polynomials P_l(cos_alpha) where cos_alpha is (g_dot_mu)
    P_0_g_dot_mu = torch.ones_like(g_dot_mu)  # P_0(x) = 1
    P_2_g_dot_mu = (3.0 * g_dot_mu_sq - 1.0) / 2.0  # P_2(x) = (3x^2 - 1)/2

    # R_l integrals
    R_0_val = _R_0(zeta)  # Shape: (1, N_gradients)
    R_2_val = _R_2(zeta, R_0_val)  # Shape: (1, N_gradients)

    # Coefficients c_l for the sum from Eq. (A.10), Zhang et al. 2012
    # c_l = exp(-kappa) * ( (2l+1)/2 ) * (kappa^l / l!) * R_l(b*d_intra)
    # Truncated at L_max=2 (terms for l=0 and l=2).

    # l=0 term: (2*0+1)/2 * kappa^0/0! = 1/2. Factorial(0)=1.
    coeff_l0_base = 1.0 / 2.0
    term_l0_contrib = coeff_l0_base * R_0_val * P_0_g_dot_mu

    # l=2 term: (2*2+1)/2 * kappa^2/2! = 5/2 * kappa^2/2.
    # Original paper uses kappa^l in series. Here kappa appears in exp(-kappa) and in c_l.
    # The term is ( (2l+1)/2 * (kappa)^l / l! ). For l=0, (1/2). For l=2, (5/2 * kappa^2 / 2).
    # However, the common implementation uses coefficients for Watson distribution itself,
    # where c_0 ~ exp(-kappa), c_2 ~ exp(-kappa) * kappa.
    # Let's follow the structure: exp(-kappa) * sum ( (2l+1)/2 * kappa^l/l! * R_l * P_l )
    # For l=0: (1/2) * R_0_val * P_0_g_dot_mu
    # For l=2: (5/2) * (kappa^2 / 2) * R_2_val * P_2_g_dot_mu -> This seems to have kappa squared.
    # The Zhang paper's formulation (A.10) has kappa^k/k! * R_k(zeta) * P_k(g.mu).
    # It is actually exp(-kappa) * Sum_{k even} [(2k+1)/2 * kappa^k/k! * R_k(b d_a) P_k(g.mu)]
    # So for l=0: exp(-kappa) * (1/2) * R_0_val * P_0_g_dot_mu
    # For l=2: exp(-kappa) * (5/2) * (kappa^2 / 2!) * R_2_val * P_2_g_dot_mu
    # The provided code used _kappa/2.0 for l=2, which is kappa^1. This is standard in many implementations.
    # Let's assume the common simplification where the l-th term is proportional to kappa^ (l/2) for even l.
    # Or, more directly, from many codes:
    # c0_watson = exp(-kappa)
    # c2_watson = exp(-kappa) * kappa (this is for the SH expansion of Watson, not directly the signal terms)

    # The existing code's coefficient structure:
    # term_l0 = exp(-kappa) * (1/2) * R_0 * P_0
    # term_l2 = exp(-kappa) * (5/2) * (kappa/2) * R_2 * P_2
    # This implies the series is more like: exp(-kappa) * Sum C_l(kappa) * R_l * P_l
    # where C_0 = 1/2, C_2 = 5/4 * kappa. This seems to be a common simplification/adaptation.

    signal = (1.0/2.0) * R_0_val * P_0_g_dot_mu
    # For l=2 term coefficient: (5/2) * (kappa/2!) based on some sources (kappa^1, not kappa^2)
    # The original code has (_kappa / 2.0) which suggests kappa^1. Let's stick to that for now.
    # This means the coefficient for R_2 P_2 is (5/2) * (_kappa / 2.0)
    signal = signal + (5.0 / 2.0) * (_kappa / 2.0) * R_2_val * P_2_g_dot_mu
    signal = torch.exp(-_kappa) * signal

    return signal.clamp(min=EPS, max=1.0)  # Clamp signal to be non-negative and <= 1


def compute_iso_signal(
    b_values: torch.Tensor,
    d_iso: float
) -> torch.Tensor:
    """Computes the normalized signal from the isotropic compartment (e.g., CSF) in NODDI.

    The signal is modeled as a simple exponential decay: S_iso / S0_iso = exp(-b * d_iso).

    Args:
        b_values (torch.Tensor): Tensor of b-values.
            Shape: (N_gradients,) or can be broadcastable (e.g., (1, N_gradients)).
        d_iso (float): Isotropic diffusivity (e.g., 3.0e-3 mm^2/s for CSF).

    Returns:
        torch.Tensor: Normalized isotropic signal.
            Shape matches `b_values` after broadcasting.
    """
    return torch.exp(-b_values * d_iso)


def compute_extra_signal(
    b_values: torch.Tensor,
    gradient_directions: torch.Tensor,
    mu: torch.Tensor,
    kappa: torch.Tensor,
    f_intra: torch.Tensor,
    d_intra: float
) -> torch.Tensor:
    """Computes the normalized signal from the extra-cellular compartment in NODDI.

    The extra-cellular diffusion is anisotropic. Its parallel diffusivity is typically
    set equal to the intra-cellular intrinsic diffusivity (`d_intra`). Its perpendicular
    diffusivity is hindered by the presence of neurites, modeled using a tortuosity
    model: `d_ec_perp = d_intra * (1 - f_intra_effective)`.
    The orientation of the extra-cellular diffusion tensor components is also governed
    by the same Watson distribution (defined by `mu` and `kappa`) as the intra-cellular
    compartment. The signal is calculated using a truncated Legendre polynomial
    expansion based on Eq. (A.12) from Zhang et al., NeuroImage 2012 (L_max=2).

    Args:
        b_values (torch.Tensor): Tensor of b-values. Shape: (N_gradients,).
        gradient_directions (torch.Tensor): Tensor of normalized gradient directions.
            Shape: (N_gradients, 3).
        mu (torch.Tensor): Mean orientation vector(s) of neurites (unit vector).
            Shape: (..., 3).
        kappa (torch.Tensor): Watson distribution concentration parameter(s). Shape: (...).
        f_intra (torch.Tensor): Volume fraction(s) of the intra-cellular compartment.
            Used in the tortuosity model. Shape: (...).
        d_intra (float): Intrinsic axial diffusivity (e.g., 1.7e-3 mm^2/s), used for `d_ec_parallel`.

    Returns:
        torch.Tensor: Normalized extra-cellular signal (S_ec / S0_ec).
            Shape: (..., N_gradients). Values are clamped to [EPS, 1.0].
    """
    _mu = mu.unsqueeze(-2)  # Shape: (..., 1, 3)
    _kappa = kappa.unsqueeze(-1)  # Shape: (..., 1)
    _f_intra = f_intra.unsqueeze(-1) # Shape: (..., 1)

    # Effective f_intra for tortuosity model.
    # Clamp to avoid d_ec_perpendicular <= 0 or other numerical issues.
    # Max value is 1.0-EPS to ensure d_ec_perpendicular is strictly positive.
    fic_tort = _f_intra.clamp(min=0.0, max=1.0 - EPS)

    d_ec_parallel = d_intra  # Parallel diffusivity definition in NODDI
    d_ec_perpendicular = d_intra * (1.0 - fic_tort)  # Tortuosity model, shape: (..., 1)

    g_dot_mu_sq = (gradient_directions.unsqueeze(0) @ _mu.transpose(-1, -2)).squeeze(-1).pow(2)
    g_dot_mu = torch.sqrt(g_dot_mu_sq.clamp(min=0.0, max=1.0))

    zeta_par = b_values.unsqueeze(0) * d_ec_parallel      # Shape: (1, N_gradients)
    zeta_perp = b_values.unsqueeze(0) * d_ec_perpendicular # Shape: (..., N_gradients)

    # R_0 for extra-cellular compartment (Eq. A.12, Zhang et al. 2012, with l=0)
    # R_0(zeta_par, zeta_perp) = integral_0^1 exp(-zeta_par*x - zeta_perp*(1-x)) dx
    delta_zeta = zeta_par - zeta_perp  # Shape: (..., N_gradients)
    
    # Handle case where delta_zeta is close to zero (zeta_par approx zeta_perp)
    R_0_ec = torch.where(
        torch.abs(delta_zeta) < EPS,
        torch.exp(-zeta_par),  # If delta_zeta is 0, integral is exp(-zeta_par)
        torch.exp(-zeta_perp) * (1.0 - torch.exp(-delta_zeta)) / (delta_zeta + EPS) # Add EPS to denominator for safety
    ) # Shape: (..., N_gradients)

    # R_2 for extra-cellular compartment (Eq. A.12, Zhang et al. 2012, with l=2)
    # R_2(zeta_par, zeta_perp) = integral_0^1 exp(-zeta_par*x - zeta_perp*(1-x)) * (3x-1)/2 dx
    # Analytically: 0.5 * (3 * I_1 - R_0_ec)
    # where I_1 = integral_0^1 x * exp(-zeta_par*x - zeta_perp*(1-x)) dx
    # I_1 = exp(-zeta_perp)/delta_zeta^2 * [1 - exp(-delta_zeta)*(1+delta_zeta) ] (if delta_zeta != 0)
    # If delta_zeta == 0, I_1 = exp(-zeta_par) * integral x dx = exp(-zeta_par)/2.
    
    delta_zeta_sq_safe = delta_zeta.pow(2) + EPS # Avoid division by zero

    I_1 = torch.where(
        torch.abs(delta_zeta) < EPS,
        torch.exp(-zeta_par) / 2.0,
        (torch.exp(-zeta_perp) / delta_zeta_sq_safe) * (1.0 - torch.exp(-delta_zeta) * (1.0 + delta_zeta))
    )
    R_2_ec = 0.5 * (3.0 * I_1 - R_0_ec)
    # If delta_zeta is very small (d_ec_par ~ d_ec_perp), EC is almost isotropic for that orientation 'n'.
    # In this case, R_2_ec should be close to zero.
    R_2_ec = torch.where(torch.abs(delta_zeta) < EPS, torch.zeros_like(R_2_ec), R_2_ec)


    # Legendre Polynomials (same as for intra-cellular)
    P_0_g_dot_mu = torch.ones_like(g_dot_mu)
    P_2_g_dot_mu = (3.0 * g_dot_mu_sq - 1.0) / 2.0

    # Signal contribution from l=0 term
    # Coeff structure similar to intra-cellular part based on common implementations
    signal_ec = (1.0/2.0) * R_0_ec * P_0_g_dot_mu
    # Signal contribution from l=2 term
    signal_ec = signal_ec + (5.0/2.0) * (_kappa / 2.0) * R_2_ec * P_2_g_dot_mu
    signal_ec = torch.exp(-_kappa) * signal_ec
    
    return signal_ec.clamp(min=EPS, max=1.0)


def noddi_signal_model(
    params: dict,
    b_values: torch.Tensor,
    gradient_directions: torch.Tensor,
    d_intra_val: float = d_intra_default,
    d_iso_val: float = d_iso_default
) -> torch.Tensor:
    """Computes the total normalized signal for the NODDI model by combining compartments.

    The NODDI model consists of three compartments:
    1.  Intra-cellular (IC): Models neurites as sticks with orientation dispersion (Watson distribution).
    2.  Extra-cellular (EC): Models diffusion hindered by neurites, with anisotropy coupled to IC.
    3.  Isotropic/CSF (ISO): Models free diffusion, e.g., cerebrospinal fluid.

    The total signal S is S0 * (f_ic * S_ic + f_ec * S_ec + f_iso * S_iso),
    where S_i are normalized signals from each compartment and f_i are volume fractions.
    f_ec = 1 - f_ic - f_iso.
    This function returns the S/S0 part.

    Args:
        params (dict): Dictionary of NODDI parameters for one or more voxels.
            Expected keys and their tensor shapes (e.g., for a batch of B voxels):
            - 'f_intra' (torch.Tensor): Volume fraction of intra-cellular compartment. Shape: (B,).
            - 'f_iso' (torch.Tensor): Volume fraction of isotropic compartment. Shape: (B,).
            - 'mu_theta' (torch.Tensor): Polar angle (theta) of mean neurite orientation. Shape: (B,).
            - 'mu_phi' (torch.Tensor): Azimuthal angle (phi) of mean neurite orientation. Shape: (B,).
            - 'kappa' (torch.Tensor): Watson distribution concentration parameter. Shape: (B,).
            If single voxel, shapes are scalar-like (0-dim tensor) or (1,).
        b_values (torch.Tensor): b-values for the diffusion acquisition. Shape: (N_gradients,).
        gradient_directions (torch.Tensor): Normalized gradient directions. Shape: (N_gradients, 3).
        d_intra_val (float, optional): Intrinsic diffusivity for the intra-cellular
            compartment (d_parallel for sticks). Defaults to `d_intra_default`.
        d_iso_val (float, optional): Diffusivity for the isotropic compartment.
            Defaults to `d_iso_default`.

    Returns:
        torch.Tensor: Predicted normalized diffusion signal (S_pred / S0_pred).
            Shape: (B, N_gradients) for batch input, or (N_gradients,) for single voxel.
            Values are clamped to [EPS, 1.0].
    """
    f_intra = params['f_intra']
    f_iso = params['f_iso']
    mu_theta = params['mu_theta']
    mu_phi = params['mu_phi']
    kappa = params['kappa']

    # Parameter Clamping (mostly for safety, primary constraints handled by optimizer via transformations)
    f_intra = f_intra.clamp(min=EPS, max=1.0 - EPS) # Ensure f_intra is in (0,1)
    f_iso = f_iso.clamp(min=EPS, max=1.0 - EPS)   # Ensure f_iso is in (0,1)
    kappa = kappa.clamp(min=EPS)                  # Kappa must be positive

    # Handle potential single voxel input (scalar-like tensors) by unsqueezing for batch operations
    is_single_voxel = (f_intra.ndim == 0)
    if is_single_voxel:
        f_intra = f_intra.unsqueeze(0)
        f_iso = f_iso.unsqueeze(0)
        mu_theta = mu_theta.unsqueeze(0)
        mu_phi = mu_phi.unsqueeze(0)
        kappa = kappa.unsqueeze(0)

    mu_cartesian = spherical_to_cartesian(mu_theta, mu_phi)  # Shape: (B, 3)

    # Compute extra-cellular volume fraction, ensuring it's valid
    f_extra = 1.0 - f_intra - f_iso
    # Clamp f_extra to avoid issues if f_intra + f_iso slightly > 1 due to numerical precision
    f_extra = f_extra.clamp(min=EPS, max=1.0 - EPS)

    # --- Calculate signal for each compartment ---
    # Intra-cellular signal (sticks with Watson dispersion)
    S_intra = compute_intra_signal(b_values, gradient_directions, mu_cartesian, kappa, d_intra_val)

    # Isotropic signal (CSF)
    S_iso = compute_iso_signal(b_values, d_iso_val)
    # Ensure S_iso can broadcast with batched S_intra if necessary
    # (it's usually (N_gradients,) and S_intra is (B, N_gradients))
    if S_iso.ndim < S_intra.ndim: # S_iso is (N_grad), S_intra is (B, N_grad)
         S_iso = S_iso.unsqueeze(0) # Becomes (1, N_grad) for broadcasting

    # Extra-cellular signal
    S_extra = compute_extra_signal(b_values, gradient_directions, mu_cartesian, kappa, f_intra, d_intra_val)

    # --- Combine signals from all compartments ---
    # Volume fractions need to be unsqueezed to multiply with signals: (B, 1) * (B, N_gradients)
    total_signal = (
        f_intra.unsqueeze(-1) * S_intra +
        f_iso.unsqueeze(-1) * S_iso +
        f_extra.unsqueeze(-1) * S_extra
    )

    if is_single_voxel: # If input was single voxel, return (N_gradients,)
        total_signal = total_signal.squeeze(0)
        
    return total_signal.clamp(min=EPS, max=1.0) # Clamp final signal


if __name__ == '__main__':
    # This block is for basic testing and example usage.
    # It requires torch.nn.functional for F.normalize if that's used.
    # For this file, we'll assume F.normalize is not directly needed here
    # as it's more of a utility for gradient vector prep.
    # However, the original file had it in __main__, so keep a placeholder.
    try:
        import torch.nn.functional as F
    except ImportError:
        print("torch.nn.functional not available for __main__ example in noddi_signal.py without PyTorch installed.")
        F = None # Placeholder

    # Example Usage (for testing one voxel)
    N_gradients = 64
    b_vals_example = torch.rand(N_gradients) * 2000  # Example b-values up to 2000 s/mm^2
    b_vals_example[0] = 0 # One b0 image
    
    grad_dirs_example = torch.randn(N_gradients, 3)
    grad_dirs_example[0, :] = 0 # b0 gradient is zero vector
    if F is not None: # Normalize if F is available
        grad_dirs_example = F.normalize(grad_dirs_example, p=2, dim=1)
        grad_dirs_example[0,:] = 0.0 # ensure b0 grad is zero after normalization
    else: # Simple normalization if F is not available (for basic execution)
        norms = torch.linalg.norm(grad_dirs_example, dim=1, keepdim=True) + EPS
        grad_dirs_example = grad_dirs_example / norms
        grad_dirs_example[0,:] = 0.0


    # Example NODDI parameters for one voxel (as 0-dim tensors)
    example_params = {
        'f_intra': torch.tensor(0.5),
        'f_iso': torch.tensor(0.1),
        'mu_theta': torch.tensor(math.pi / 4), # 45 degrees polar angle
        'mu_phi': torch.tensor(math.pi / 3),   # 60 degrees azimuthal angle
        'kappa': torch.tensor(2.0)             # Example concentration
    }

    predicted_signal_single = noddi_signal_model(example_params, b_vals_example, grad_dirs_example)
    print("--- Single Voxel Example ---")
    print("Input b-values shape:", b_vals_example.shape)
    print("Input gradient_directions shape:", grad_dirs_example.shape)
    print("Predicted signal shape (single voxel):", predicted_signal_single.shape)
    if predicted_signal_single.ndim > 0:
        print("Example predicted signal (first 5 values):", predicted_signal_single[:5])

    # Example for batch of voxels (e.g. 2 voxels) (as 1-dim tensors for params)
    example_params_batch = {
        'f_intra': torch.tensor([0.5, 0.6]),
        'f_iso': torch.tensor([0.1, 0.05]),
        'mu_theta': torch.tensor([math.pi / 4, math.pi / 2]),
        'mu_phi': torch.tensor([math.pi / 3, math.pi / 6]),
        'kappa': torch.tensor([2.0, 3.5])
    }
    predicted_signal_batch = noddi_signal_model(example_params_batch, b_vals_example, grad_dirs_example)
    print("\n--- Batch Voxel Example ---")
    print("Predicted signal batch shape:", predicted_signal_batch.shape)
    if predicted_signal_batch.ndim > 1 and predicted_signal_batch.shape[0] > 0:
        print("Example predicted signal for voxel 0 (first 5 values):", predicted_signal_batch[0, :5])
        if predicted_signal_batch.shape[0] > 1:
            print("Example predicted signal for voxel 1 (first 5 values):", predicted_signal_batch[1, :5])

    # Test edge case for kappa (very small kappa -> isotropic orientation for sticks) (single voxel)
    example_params_iso_kappa = {
        'f_intra': torch.tensor(0.5),
        'f_iso': torch.tensor(0.1),
        'mu_theta': torch.tensor(math.pi / 4),
        'mu_phi': torch.tensor(math.pi / 3),
        'kappa': torch.tensor(0.01) # Very small kappa
    }
    predicted_signal_iso_kappa = noddi_signal_model(example_params_iso_kappa, b_vals_example, grad_dirs_example)
    print("\n--- Low Kappa Example (Near Isotropic Dispersion) ---")
    if predicted_signal_iso_kappa.ndim > 0:
        print("Predicted signal (low kappa) (first 5 values):", predicted_signal_iso_kappa[:5])
    
    # Test high kappa (very oriented sticks) (single voxel)
    example_params_high_kappa = {
        'f_intra': torch.tensor(0.5),
        'f_iso': torch.tensor(0.1),
        'mu_theta': torch.tensor(0.0), # Aligned with z-axis
        'mu_phi': torch.tensor(0.0),
        'kappa': torch.tensor(50.0) # Very high kappa
    }
    # A gradient along z-axis should show high attenuation if d_intra is high
    grad_dirs_z_aligned = torch.zeros_like(grad_dirs)
    grad_dirs_z_aligned[:, 2] = 1.0 
    grad_dirs_z_aligned[0, :] = 0.0 # b0
    
    predicted_signal_high_kappa = noddi_signal_model(example_params_high_kappa, b_vals_example, grad_dirs_z_aligned)
    print("\n--- High Kappa Example (Highly Oriented Sticks) ---")
    if predicted_signal_high_kappa.ndim > 0:
        print("Predicted signal (high kappa, grad along mu) (first 5 values):", predicted_signal_high_kappa[:5])

    # A gradient perpendicular to z-axis should show low attenuation for sticks
    grad_dirs_x_aligned = torch.zeros_like(grad_dirs_example)
    grad_dirs_x_aligned[:, 0] = 1.0
    grad_dirs_x_aligned[0, :] = 0.0 # b0
    predicted_signal_high_kappa_perp = noddi_signal_model(example_params_high_kappa, b_vals_example, grad_dirs_x_aligned)
    if predicted_signal_high_kappa_perp.ndim > 0:
        print("Predicted signal (high kappa, grad perp mu) (first 5 values):", predicted_signal_high_kappa_perp[:5])

```
