import numpy as np
import torch
from typing import Optional

from dipy.core.gradients import GradientTable
from dipy.tracking.streamline import Streamlines as DipyStreamlines
# seeds_from_mask is now used inside the pytorch_tracking_pipeline version
# from dipy.tracking.utils import seeds_from_mask 

# Import the core pipeline function
from .pytorch_tracking_pipeline import track_deterministic_oudf as pytorch_pipeline_track_deterministic_oudf

def track_deterministic_oudf(
    dwi_data: np.ndarray,
    gtab: GradientTable, 
    seeds: np.ndarray, # Can be 3D mask or 2D (Nx3) coords, in VOXEL space
    affine: np.ndarray,
    metric_map_for_stopping: np.ndarray, # e.g., FA map, NumPy array
    stopping_threshold_value: float,
    step_size: float = 0.5,
    # Parameters for internal PeaksFromModel
    sh_order: int = 8,
    response: Optional[tuple] = None, # Renamed from csd_response to match PeaksFromModel
    model_max_peaks: int = 5, 
    model_min_separation_angle: float = 25,
    model_peak_threshold: float = 0.5,
    # Parameters for internal LocalTracking
    max_crossing_angle: Optional[float] = 60.0,
    min_length: float = 10.0,
    max_length: float = 250.0,
    max_steps: int = 1000, 
    device: Optional[str] = None # NEW parameter
) -> DipyStreamlines:
    """
    Performs deterministic tractography using a PyTorch-based pipeline.

    This function serves as a wrapper around a more comprehensive PyTorch-based
    tracking pipeline (`tracking.pytorch_tracking_pipeline.track_deterministic_oudf`).
    It handles CSD-like modeling, peak extraction, and streamline generation internally
    using PyTorch.

    Parameters
    ----------
    dwi_data : np.ndarray
        4D dMRI data array (X, Y, Z, N_gradients).
    gtab : dipy.core.gradients.GradientTable
        Dipy GradientTable object corresponding to the `dwi_data`.
    seeds : np.ndarray
        Seed points for tractography, as a NumPy array.
        If a 3D array, treated as a binary mask in voxel space.
        If a 2D array (Nx3), treated as coordinates in voxel space.
    affine : np.ndarray
        NumPy array (4x4) representing the affine transformation from voxel
        space to world/scanner space.
    metric_map_for_stopping : np.ndarray
        A 3D NumPy array (e.g., FA or GFA map) used for the stopping criterion.
    stopping_threshold_value : float
        The threshold value for the `metric_map_for_stopping`.
    step_size : float, optional
        Tracking step size in mm. Default is 0.5.
    sh_order : int, optional
        Spherical harmonic order for the internal CSD-like model. Default is 8.
    response : tuple, optional
        Response function (eigenvalues, S0_response) for the internal CSD-like model.
        If None, a simplified estimate is used internally. Default is None.
    model_max_peaks : int, optional
        Maximum number of peaks to extract per voxel by the internal model. Default is 5.
    model_min_separation_angle : float, optional
        Minimum separation angle (degrees) between peaks for the internal model. Default is 25.
    model_peak_threshold : float, optional
        Relative threshold for peak detection in the internal model. Default is 0.5.
    max_crossing_angle : float, optional
        Maximum angle (degrees) between successive steps. Default is 60.0.
    min_length : float, optional
        Minimum streamline length in mm. Default is 10.0.
    max_length : float, optional
        Maximum streamline length in mm. Default is 250.0.
    max_steps : int, optional
        Maximum steps per half-streamline. Default is 1000.
    device : str, optional
        Computation device for PyTorch operations, e.g., 'cpu', 'cuda', 'auto'.
        If None or 'auto', the pipeline will auto-detect CUDA availability. Default is None.


    Returns
    -------
    DipyStreamlines
        A Dipy `Streamlines` object containing the generated tracts in world coordinates.
    """
    
    # Determine the effective device string to be passed to the PyTorch pipeline
    # The pipeline's track_deterministic_oudf itself has a default device='cpu' if None is passed,
    # and can also handle 'auto'. So passing this 'device' string directly is fine.
    # If this wrapper's 'device' is None, the pipeline will auto-detect.
    # If 'cpu' or 'cuda' is specified, that will be used by the pipeline.
    
    # Convert NumPy inputs to PyTorch tensors.
    # The actual device placement will be handled by the pipeline function based on its 'device' arg.
    # So, we can convert to tensors here without specifying device, or use a temp 'cpu' device.
    # However, the pipeline function expects torch tensors as input for these.
    # Let's use a determined device for conversion here, which the pipeline might override/use.
    # This wrapper's 'device' param informs the 'device' param of the pipeline.
    
    # If 'device' is None, the pipeline's track_deterministic_oudf will auto-detect.
    # If 'device' is 'cpu' or 'cuda', it will be used by the pipeline.
    # The conversion to tensors here should happen on some device, e.g. CPU, then pipeline moves if needed.
    # Or, we can use the passed 'device' string to convert directly to target device if specified.
    
    temp_device_for_conversion = 'cpu' # Default for initial tensor conversion
    if device and device.lower() == 'cuda' and torch.cuda.is_available():
        temp_device_for_conversion = 'cuda'
    
    dwi_data_torch = torch.tensor(dwi_data, dtype=torch.float32, device=temp_device_for_conversion)
    affine_torch = torch.tensor(affine, dtype=torch.float32, device=temp_device_for_conversion)
    metric_map_torch = torch.tensor(metric_map_for_stopping, dtype=torch.float32, device=temp_device_for_conversion)
    
    seeds_input_torch: Optional[torch.Tensor] = None
    if seeds is not None: 
        seeds_input_torch = torch.tensor(seeds, dtype=torch.float32, device=temp_device_for_conversion)

    # Call the core PyTorch pipeline function
    return pytorch_pipeline_track_deterministic_oudf(
        dwi_data=dwi_data_torch, 
        gtab=gtab,               
        affine=affine_torch,     
        stopping_metric_map=metric_map_torch, 
        stopping_threshold_value=stopping_threshold_value,
        seeds_input=seeds_input_torch, 
        step_size=step_size,
        sh_order=sh_order,
        response=response, 
        model_max_peaks=model_max_peaks, 
        model_min_separation_angle=model_min_separation_angle, 
        model_peak_threshold=model_peak_threshold, 
        max_crossing_angle=max_crossing_angle,
        min_length=min_length,
        max_length=max_length,
        max_steps=max_steps, 
        device=device # Pass the device string from wrapper's argument
    )
```
