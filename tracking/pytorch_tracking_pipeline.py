import torch
import numpy as np
from typing import Optional # For optional arguments, though not explicitly in user code, good practice
from abc import ABC, abstractmethod

from dipy.core.gradients import GradientTable
from dipy.data import get_sphere
from dipy.tracking.streamline import Streamlines # Dipy Streamlines for output
from dipy.tracking.utils import seeds_from_mask
from scipy.special import sph_harm

# For the if __name__ == "__main__": block
try:
    import nibabel as nib
    from dipy.io.image import load_data as load_dipy_data 
    from dipy.data import get_fnames as get_dipy_fnames
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_trk
    DIPY_EXAMPLES_AVAILABLE = True
except ImportError:
    DIPY_EXAMPLES_AVAILABLE = False

# --- PeaksFromModel Class (from models.pytorch_csd_peaks) ---
class PeaksFromModel:
    """PyTorch implementation of peaks_from_model for extracting fiber orientation peaks."""
    def __init__(self, data_tensor: torch.Tensor, gradient_table: GradientTable, 
                 sh_order: int = 8, max_peaks: int = 5, 
                 min_separation_angle: float = 25,
                 peak_threshold: float = 0.5, 
                 response: tuple = None, device: str ='cpu'):
        self.data_tensor = data_tensor.to(device)
        self.gtab = gradient_table 
        self.bvals = torch.tensor(self.gtab.bvals, dtype=torch.float32, device=device)
        self.bvecs = torch.tensor(self.gtab.bvecs, dtype=torch.float32, device=device)
        self.sh_order = sh_order
        self.max_peaks = max_peaks
        self.min_sep_angle_rad = torch.tensor(min_separation_angle * np.pi / 180.0, 
                                              dtype=torch.float32, device=device)
        self.cos_min_sep = torch.cos(self.min_sep_angle_rad)
        self.peak_threshold = peak_threshold
        self.device = device
        self.MPPCA_EPS = 1e-8 # Used in _fit_csd_sh_coeffs, from pytorch_denoising

        self.response = response if response else self._estimate_response_simple()
        self.sh_basis_matrix = self._compute_sh_basis(self.gtab, self.sh_order, self.device)
        self.sphere = get_sphere('repulsion724') 
        self.sphere_vertices = torch.tensor(self.sphere.vertices, dtype=torch.float32, device=device)
        self.sphere_sh_basis_matrix = self._compute_sh_basis_on_sphere(self.sphere, self.sh_order, self.device)

    def _estimate_response_simple(self):
        print("Warning: PeaksFromModel using a simplified, fixed response function for CSD.")
        response_eigenvals = np.array([0.0015, 0.0003, 0.0003], dtype=np.float32)
        response_s0 = 1.0 
        return (response_eigenvals, response_s0)

    @staticmethod
    def _compute_sh_basis(gtab: GradientTable, sh_order: int, device: str):
        bvecs_np = gtab.bvecs
        theta = np.arccos(bvecs_np[:, 2]) 
        phi = np.arctan2(bvecs_np[:, 1], bvecs_np[:, 0]) 
        sh_basis_list = []
        for l_order in range(0, sh_order + 1, 2): 
            for m_order in range(-l_order, l_order + 1):
                sh = sph_harm(m_order, l_order, phi, theta)
                sh_basis_list.append(torch.tensor(np.real(sh), dtype=torch.float32, device=device))
        return torch.stack(sh_basis_list, dim=0)

    @staticmethod
    def _compute_sh_basis_on_sphere(sphere, sh_order: int, device: str):
        vertices_np = sphere.vertices
        theta = np.arccos(vertices_np[:, 2])
        phi = np.arctan2(vertices_np[:, 1], vertices_np[:, 0])
        sh_basis_list = []
        for l_order in range(0, sh_order + 1, 2):
            for m_order in range(-l_order, l_order + 1):
                sh = sph_harm(m_order, l_order, phi, theta)
                sh_basis_list.append(torch.tensor(np.real(sh), dtype=torch.float32, device=device))
        return torch.stack(sh_basis_list, dim=0)

    def _fit_csd_sh_coeffs(self) -> torch.Tensor:
        dwi_mask = self.bvals > self.gtab.b0_threshold 
        s0_voxelwise = torch.mean(self.data_tensor[..., self.gtab.b0s_mask], dim=-1, keepdim=True)
        s0_voxelwise = s0_voxelwise.clamp(min=1e-6) 
        data_norm_s0 = self.data_tensor / s0_voxelwise
        dwi_signals = data_norm_s0[..., dwi_mask]
        sh_basis_dwi = self.sh_basis_matrix[:, dwi_mask] 
        pinv_sh_basis_dwi_T = torch.linalg.pinv(sh_basis_dwi.T)
        sh_coeffs = torch.einsum('...d,dn->...n', dwi_signals, pinv_sh_basis_dwi_T.T)
        return sh_coeffs

    def _evaluate_odf(self, sh_coeffs: torch.Tensor) -> torch.Tensor:
        odf = torch.einsum('...s,sv->...v', sh_coeffs, self.sphere_sh_basis_matrix)
        odf = torch.relu(odf) 
        return odf

    def _find_peaks_in_voxel_odf(self, odf_voxel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_odf_val = torch.max(odf_voxel)
        if max_odf_val < 1e-6: 
            return torch.zeros(self.max_peaks * 3, device=self.device, dtype=torch.float32), \
                   torch.zeros(self.max_peaks, device=self.device, dtype=torch.float32)
        odf_norm = odf_voxel / max_odf_val
        potential_peak_indices = torch.where(odf_norm >= self.peak_threshold)[0]
        if potential_peak_indices.numel() == 0:
            return torch.zeros(self.max_peaks * 3, device=self.device, dtype=torch.float32), \
                   torch.zeros(self.max_peaks, device=self.device, dtype=torch.float32)
        sorted_indices = torch.argsort(odf_voxel[potential_peak_indices], descending=True)
        potential_peak_indices = potential_peak_indices[sorted_indices]
        found_peaks_list = []
        found_values_list = []
        for idx in potential_peak_indices:
            if len(found_peaks_list) >= self.max_peaks: break
            current_peak_direction = self.sphere_vertices[idx]
            current_peak_value = odf_voxel[idx]
            is_separated = True
            for existing_peak_dir in found_peaks_list:
                cos_angle = torch.dot(current_peak_direction, existing_peak_dir)
                if cos_angle > self.cos_min_sep: 
                    is_separated = False; break
            if is_separated:
                found_peaks_list.append(current_peak_direction)
                found_values_list.append(current_peak_value)
        num_found = len(found_peaks_list)
        padded_peaks = torch.zeros((self.max_peaks, 3), device=self.device, dtype=torch.float32)
        padded_values = torch.zeros(self.max_peaks, device=self.device, dtype=torch.float32)
        if num_found > 0:
            padded_peaks[:num_found, :] = torch.stack(found_peaks_list)
            padded_values[:num_found] = torch.stack(found_values_list)
        return padded_peaks.reshape(-1), padded_values

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        sh_coeffs = self._fit_csd_sh_coeffs() 
        odf = self._evaluate_odf(sh_coeffs)   
        all_peaks = torch.zeros(self.data_tensor.shape[:3] + (self.max_peaks * 3,), device=self.device, dtype=torch.float32)
        all_peak_values = torch.zeros(self.data_tensor.shape[:3] + (self.max_peaks,), device=self.device, dtype=torch.float32)
        for x_idx in range(self.data_tensor.shape[0]):
            for y_idx in range(self.data_tensor.shape[1]):
                for z_idx in range(self.data_tensor.shape[2]):
                    voxel_odf = odf[x_idx, y_idx, z_idx, :]
                    peaks_flat, values = self._find_peaks_in_voxel_odf(voxel_odf)
                    all_peaks[x_idx, y_idx, z_idx, :] = peaks_flat
                    all_peak_values[x_idx, y_idx, z_idx, :] = values
        return all_peaks, all_peak_values

# --- PyTorch Local Tracking Components (from tracking.pytorch_local_tracking) ---
class DirectionGetter(ABC):
    @abstractmethod
    def get_directions(self, point): pass

class PeaksDirectionGetter(DirectionGetter):
    def __init__(self, peak_data: torch.Tensor, affine: torch.Tensor, device='cpu'):
        self.device = device
        self.peak_data = peak_data.to(self.device)
        if self.peak_data.ndim == 4 and self.peak_data.shape[-1] % 3 != 0:
            raise ValueError("Last dimension of peak_data must be divisible by 3 if 4D.")
        elif self.peak_data.ndim == 5 and self.peak_data.shape[-1] != 3:
            raise ValueError("Last dimension of 5D peak_data must be 3.")
        if self.peak_data.ndim == 4:
            self.n_peaks = self.peak_data.shape[-1] // 3
            self.peaks_reshaped_for_lookup = False
        elif self.peak_data.ndim == 5:
            self.n_peaks = self.peak_data.shape[-2]
            self.peaks_reshaped_for_lookup = True
        else:
            raise ValueError("peak_data must be 4D (X,Y,Z,N_peaks*3) or 5D (X,Y,Z,N_peaks,3)")
        self.affine = affine.to(self.device)

    def get_directions(self, point: torch.Tensor) -> torch.Tensor:
        point_int = torch.floor(point).long()
        if not (0 <= point_int[0] < self.peak_data.shape[0] and \
                0 <= point_int[1] < self.peak_data.shape[1] and \
                0 <= point_int[2] < self.peak_data.shape[2]):
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)
        if self.peaks_reshaped_for_lookup:
            peaks = self.peak_data[point_int[0], point_int[1], point_int[2]]
        else:
            voxel_peaks_flat = self.peak_data[point_int[0], point_int[1], point_int[2]]
            peaks = voxel_peaks_flat.reshape(self.n_peaks, 3)
        if peaks.numel() == 0: return torch.zeros((0,3), device=self.device, dtype=torch.float32)
        norm = torch.linalg.norm(peaks, dim=1)
        valid_mask = norm > 1e-6
        if not torch.any(valid_mask): return torch.zeros((0, 3), device=self.device, dtype=torch.float32)
        valid_peaks = peaks[valid_mask]
        return valid_peaks / torch.linalg.norm(valid_peaks, dim=1, keepdim=True)

class StoppingCriterion(ABC):
    @abstractmethod
    def evaluate(self, point: torch.Tensor) -> bool: pass

class ThresholdStoppingCriterion(StoppingCriterion):
    def __init__(self, metric_data: torch.Tensor, threshold: float, device='cpu'):
        self.metric_data = metric_data.to(device)
        self.threshold = threshold
        self.device = device

    def evaluate(self, point: torch.Tensor) -> bool:
        point_int = torch.floor(point).long()
        if not (0 <= point_int[0] < self.metric_data.shape[0] and \
                0 <= point_int[1] < self.metric_data.shape[1] and \
                0 <= point_int[2] < self.metric_data.shape[2]):
            return True
        metric_val = self.metric_data[point_int[0], point_int[1], point_int[2]]
        return metric_val < self.threshold

class LocalTracking:
    def __init__(self, direction_getter: DirectionGetter, stopping_criterion: StoppingCriterion, 
                 seeds: torch.Tensor, affine: torch.Tensor, step_size: float,
                 max_crossing_angle: Optional[float] = None, min_length: float = 0, 
                 max_length: float = float('inf'), max_steps: int = 1000,
                 device: str ='cpu'):
        self.direction_getter = direction_getter
        self.stopping_criterion = stopping_criterion
        self.seeds = seeds.to(device)
        self.affine = affine.to(device)
        self.step_size = step_size
        self.min_length = min_length
        self.max_length = max_length
        self.max_steps = max_steps
        self.device = device
        self.cos_max_angle = None
        if max_crossing_angle is not None:
            if not (0 < max_crossing_angle < 180):
                 raise ValueError("max_crossing_angle must be between 0 and 180 degrees.")
            self.cos_max_angle = torch.cos(torch.tensor(max_crossing_angle * np.pi / 180.0, 
                                                      dtype=torch.float32, device=self.device))

    def _apply_affine(self, point_vox: torch.Tensor) -> torch.Tensor:
        point_hom = torch.cat([point_vox.float(), torch.ones(1, device=self.device, dtype=point_vox.dtype)])
        return (self.affine @ point_hom)[:3]

    def _choose_direction(self, directions: torch.Tensor, previous_direction: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if directions.shape[0] == 0: return None
        if previous_direction is None: return directions[0] 
        cos_sim = torch.sum(directions * previous_direction.unsqueeze(0), dim=1)
        if self.cos_max_angle is not None:
            valid_indices = cos_sim >= self.cos_max_angle 
            if not torch.any(valid_indices): return None 
            directions, cos_sim = directions[valid_indices], cos_sim[valid_indices]
        if directions.shape[0] == 0: return None
        return directions[torch.argmax(cos_sim)]

    def track(self) -> list[np.ndarray]:
        streamlines_world_list = []
        for seed_vox in self.seeds:
            for i in range(2): 
                current_streamline_vox_points = [seed_vox.clone()]
                current_point_vox = seed_vox.clone()
                total_length_mm = 0.0
                step_sign = 1.0 if i == 0 else -1.0
                previous_step_orientation_vox = None
                for step_count in range(self.max_steps):
                    if self.stopping_criterion.evaluate(current_point_vox): break
                    available_directions_at_point = self.direction_getter.get_directions(current_point_vox)
                    if available_directions_at_point.shape[0] == 0: break 
                    chosen_orientation_vox = self._choose_direction(available_directions_at_point, previous_step_orientation_vox)
                    if chosen_orientation_vox is None: break 
                    actual_step_vector_vox = step_sign * self.step_size * chosen_orientation_vox
                    next_point_vox = current_point_vox + actual_step_vector_vox
                    total_length_mm += self.step_size 
                    if total_length_mm > self.max_length: break
                    current_streamline_vox_points.append(next_point_vox.clone())
                    current_point_vox = next_point_vox
                    previous_step_orientation_vox = chosen_orientation_vox
                if total_length_mm >= self.min_length and len(current_streamline_vox_points) > 1:
                    streamline_points_vox_tensor = torch.stack(current_streamline_vox_points)
                    streamline_world_points = torch.stack([self._apply_affine(p_vox) for p_vox in streamline_points_vox_tensor])
                    streamlines_world_list.append(streamline_world_points.cpu().numpy())
        return streamlines_world_list

# --- Refactored track_deterministic_oudf (from tracking.deterministic) ---
def track_deterministic_oudf(
    dwi_data: np.ndarray, 
    gtab: GradientTable,  
    affine: np.ndarray,
    stopping_metric_map: torch.Tensor, # Now a PyTorch tensor
    stopping_threshold_value: float,    
    seeds_input: Optional[torch.Tensor] = None, # Can be 3D mask or 2D (Nx3) coords in VOXEL space (PyTorch tensor)
    step_size: float = 0.5,
    # PeaksFromModel parameters
    sh_order: int = 8, # Default in PeaksFromModel
    response: Optional[tuple] = None, # For PeaksFromModel (renamed from csd_response)
    model_max_peaks: int = 5, # For PeaksFromModel
    model_min_separation_angle: float = 25,     
    model_peak_threshold: float = 0.5, 
    # PyTorchLocalTracking parameters
    max_crossing_angle: Optional[float] = 60, 
    min_length: float = 10.0,
    max_length: float = 250.0,
    max_steps: int = 1000, # Renamed from max_steps_per_streamline for LocalTracking
    device: Optional[str] = None # Allow device to be specified or auto-detected
) -> Streamlines:
    """
    Performs deterministic tractography using internally derived ODF peaks.

    This function takes raw dMRI data, fits a CSD-like model and extracts peaks
    using `PeaksFromModel`, then performs tracking using `LocalTracking`.
    All core computations (peak extraction, tracking) are PyTorch-based.

    Parameters
    ----------
    dwi_data : np.ndarray
        4D dMRI data array (X, Y, Z, N_gradients).
    gtab : dipy.core.gradients.GradientTable
        Dipy GradientTable object corresponding to the `dwi_data`.
    affine : np.ndarray
        NumPy array (4x4) representing the affine transformation from voxel
        space to world/scanner space. Used for seed generation from mask if applicable,
        and for transforming streamlines to world coordinates.
    stopping_metric_map : torch.Tensor
        A 3D PyTorch tensor (e.g., FA or GFA map) used for the stopping criterion.
        Tracking stops if the value in this map at the current point falls
        below `stopping_threshold_value`.
    stopping_threshold_value : float
        The threshold value for the `stopping_metric_map`.
    seeds_input : Optional[torch.Tensor], optional
        Seed points for tractography, as a PyTorch tensor.
        If a 3D tensor, treated as a binary mask in voxel space.
        If a 2D tensor (Nx3), treated as coordinates in voxel space.
        If None, seeds are generated from `stopping_metric_map > stopping_threshold_value`.
        Default is None.
    step_size : float, optional
        Tracking step size in mm. Default is 0.5.
    sh_order : int, optional
        Spherical harmonic order for the internal CSD-like model (`PeaksFromModel`). Default is 8.
    response : tuple, optional
        Response function (eigenvalues, S0_response) for `PeaksFromModel`.
        If None, `PeaksFromModel` uses a simplified estimate. Default is None.
    model_max_peaks : int, optional
        Maximum number of peaks to extract per voxel by `PeaksFromModel`. Default is 5.
    model_min_separation_angle : float, optional
        Minimum separation angle (degrees) between peaks for `PeaksFromModel`. Default is 25.
    model_peak_threshold : float, optional
        Relative threshold for peak detection in `PeaksFromModel`. Default is 0.5.
    max_crossing_angle : float, optional
        Maximum angle (degrees) between successive steps for `LocalTracking`. Default is 60.
    min_length : float, optional
        Minimum streamline length in mm. Default is 10.0.
    max_length : float, optional
        Maximum streamline length in mm. Default is 250.0.
    max_steps : int, optional
        Maximum steps per half-streamline for `LocalTracking`. Default is 1000.
    device : str, optional
        Computation device ('cpu' or 'cuda'). If None, auto-detects CUDA. Default is None.

    Returns
    -------
    Streamlines
        A Dipy `Streamlines` object containing the generated tracts in world coordinates.
    """
    if device is None:
        computed_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        computed_device = torch.device(device)

    # Input Validations
    if not isinstance(dwi_data, np.ndarray) or dwi_data.ndim != 4: raise ValueError("dwi_data must be a 4D NumPy array.")
    if not isinstance(gtab, GradientTable): raise TypeError("gtab must be a Dipy GradientTable object.")
    if dwi_data.shape[-1] != len(gtab.bvals): raise ValueError("Last dimension of dwi_data must match number of entries in gtab.")
    if not isinstance(affine, np.ndarray) or affine.shape != (4,4): raise ValueError("Affine must be a 4x4 NumPy array.")
    if not isinstance(stopping_metric_map, torch.Tensor) or stopping_metric_map.ndim !=3: raise ValueError("stopping_metric_map must be a 3D PyTorch Tensor.")
    if not isinstance(stopping_threshold_value, (float, int)): raise ValueError("stopping_threshold_value must be a number.")
    if seeds_input is not None and not isinstance(seeds_input, torch.Tensor): raise ValueError("seeds_input, if provided, must be a PyTorch Tensor.")
    if seeds_input is not None and seeds_input.ndim not in [2,3]: raise ValueError("seeds_input must be 2D (Nx3 coords) or 3D (mask).")
    if seeds_input is not None and seeds_input.ndim == 2 and seeds_input.shape[1] != 3: raise ValueError("2D seeds_input must be Nx3.")
    
    dwi_data_torch = torch.tensor(dwi_data, dtype=torch.float32, device=computed_device)
    affine_torch = torch.tensor(affine, dtype=torch.float32, device=computed_device)
    
    try:
        peak_finder = PeaksFromModel(
            data_tensor=dwi_data_torch, gradient_table=gtab, sh_order=sh_order,
            max_peaks=model_max_peaks, min_separation_angle=model_min_separation_angle,
            peak_threshold=model_peak_threshold, response=response, device=computed_device)
        peaks_tensor_torch, _ = peak_finder.compute()
    except Exception as e:
        raise RuntimeError(f"Failed during internal CSD modeling or peak extraction: {e}")

    # Seed generation
    if seeds_input is not None:
        if seeds_input.ndim == 3: # Mask (PyTorch tensor)
            seeds_np_from_dipy = seeds_from_mask(seeds_input.cpu().numpy().astype(bool), 
                                                 affine=affine, # Dipy utility needs affine_np
                                                 density=1) 
            if seeds_np_from_dipy.shape[0] == 0: return Streamlines([])
            actual_seeds_vox_torch = torch.tensor(seeds_np_from_dipy, dtype=torch.float32, device=computed_device)
        elif seeds_input.ndim == 2 and seeds_input.shape[1] == 3: # Coordinates (PyTorch tensor)
            actual_seeds_vox_torch = seeds_input.to(computed_device).float()
            if actual_seeds_vox_torch.shape[0] == 0: return Streamlines([])
        else: # Should be caught by initial validation
            raise ValueError("seeds_input format error after initial checks.")
    else: # Default seeding using the stopping_metric_map
        default_seed_mask_torch = stopping_metric_map > stopping_threshold_value
        seeds_np_from_dipy = seeds_from_mask(default_seed_mask_torch.cpu().numpy().astype(bool), 
                                             affine=affine, 
                                             density=1)
        if seeds_np_from_dipy.shape[0] == 0: return Streamlines([]) 
        actual_seeds_vox_torch = torch.tensor(seeds_np_from_dipy, dtype=torch.float32, device=computed_device)
    
    # Ensure stopping_metric_map is on the correct device
    stopping_metric_map_on_device = stopping_metric_map.to(computed_device)
    pt_stopping_criterion = ThresholdStoppingCriterion(stopping_metric_map_on_device, stopping_threshold_value, device=computed_device)
    pt_direction_getter = PeaksDirectionGetter(peak_data=peaks_tensor_torch, affine=affine_torch, device=computed_device)

    try:
        tracker = LocalTracking(
            direction_getter=pt_direction_getter, stopping_criterion=pt_stopping_criterion,
            seeds=actual_seeds_vox_torch, affine=affine_torch, step_size=step_size,
            max_crossing_angle=max_crossing_angle, min_length=min_length, max_length=max_length,
            max_steps=max_steps, device=computed_device) # Pass max_steps here
        streamlines_list_np = tracker.track()
    except Exception as e:
        raise RuntimeError(f"Failed during PyTorch-based streamline generation: {e}")

    if not streamlines_list_np: return Streamlines([])
    return Streamlines(streamlines_list_np)


# --- Example Usage ---
if __name__ == "__main__":
    if DIPY_EXAMPLES_AVAILABLE:
        print("Running PyTorch Unified Tracking Pipeline example using Dipy for data I/O...")
        try:
            hardi_fname, bval_fname, bvec_fname = get_dipy_fnames('stanford_hardi')
            fa_fname = get_dipy_fnames('stanford_fa') # For stopping criterion
            
            img_hardi_nib = nib.load(hardi_fname) 
            affine_np = img_hardi_nib.affine.astype(np.float32) # Ensure affine is float32
            dwi_data_np = img_hardi_nib.get_fdata().astype(np.float32)

            bvals_np, bvecs_np = read_bvals_bvecs(bval_fname, bvec_fname)
            gtab_dipy = GradientTable(bvals_np, bvecs_np, b0_threshold=50)

            fa_np, _ = load_dipy_data(fa_fname) # For stopping map
            
        except Exception as e:
            print(f"Could not load Dipy example data. Make sure you have run 'dipy_fetch_stanford_hardi'. Error: {e}")
            exit()

        # Determine device for PyTorch operations
        current_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {current_device_str}")
        
        # Prepare inputs as PyTorch tensors where appropriate for the new signature
        stopping_metric_map_torch = torch.tensor(fa_np.astype(np.float32), device=current_device_str)
        
        # Seeds can be a mask or coordinates. Let's use coordinates from FA map.
        seed_coords_np = np.array(np.where(fa_np > 0.35)).T 
        if seed_coords_np.shape[0] == 0:
            print("No suitable seed points found with FA > 0.35. Using a default voxel seed.")
            seeds_input_torch = torch.tensor([[40.0, 53.0, 38.0]], dtype=torch.float32, device=current_device_str) 
        else:
            num_seeds_to_use = min(200, seed_coords_np.shape[0]) 
            random_indices = np.random.choice(seed_coords_np.shape[0], size=num_seeds_to_use, replace=False)
            selected_seeds_np = seed_coords_np[random_indices].astype(np.float32)
            seeds_input_torch = torch.tensor(selected_seeds_np, dtype=torch.float32, device=current_device_str)
            print(f"Using {num_seeds_to_use} seeds with FA > 0.35. Example seed (voxel coords): {selected_seeds_np[0]}")

        print("Starting unified tracking pipeline...")
        streamlines_dipy_obj = track_deterministic_oudf(
            dwi_data=dwi_data_np, # NumPy array
            gtab=gtab_dipy,       # Dipy GradientTable
            affine=affine_np,     # NumPy array
            stopping_metric_map=stopping_metric_map_torch, # PyTorch Tensor
            stopping_threshold_value=0.10,
            seeds_input=seeds_input_torch, # PyTorch Tensor (coordinates)
            step_size=0.5,
            sh_order=6, 
            model_max_peaks=3,
            model_min_separation_angle=30,
            model_peak_threshold=0.4,
            min_length=20.0,
            max_length=250.0,
            max_steps=1000, # Renamed from max_steps_per_streamline
            max_crossing_angle=35.0,
            device=current_device_str # Pass the determined device
        )
        print(f"Tracking finished. Generated {len(streamlines_dipy_obj)} streamlines.")

        if streamlines_dipy_obj:
            sft = StatefulTractogram(streamlines_dipy_obj.streamlines, img_hardi_nib, Space.RASMM)
            save_trk(sft, "pytorch_pipeline_tracks.trk", bbox_valid_check=False)
            print("Saved streamlines to pytorch_pipeline_tracks.trk")
        else:
            print("No streamlines generated or kept after length filtering.")
    else:
        print("To run the example usage for the unified tracking pipeline, please ensure Dipy and Nibabel are installed.")

```
