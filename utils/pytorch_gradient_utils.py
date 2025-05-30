import torch
import numpy as np

class PyTorchGradientTable:
    """
    A class to store and manage gradient table information (b-values and b-vectors)
    using PyTorch tensors, intended as a replacement for parts of dipy's GradientTable.
    """
    def __init__(self, bvals: np.ndarray, bvecs: np.ndarray, b0_threshold: float = 50.0):
        """
        Initializes the PyTorchGradientTable.

        Args:
            bvals (np.ndarray): 1D array of b-values.
            bvecs (np.ndarray): 2D array of b-vectors (N, 3). Assumed to be normalized
                                for b>0, and [0,0,0] for b0s.
            b0_threshold (float, optional): The threshold below which b-values are
                                            considered to be b0 images. Defaults to 50.0.
        """
        if not isinstance(bvals, np.ndarray) or bvals.ndim != 1:
            raise ValueError("bvals must be a 1D NumPy array.")
        if not isinstance(bvecs, np.ndarray) or bvecs.ndim != 2:
            raise ValueError("bvecs must be a 2D NumPy array.")
        if bvals.shape[0] != bvecs.shape[0]:
            raise ValueError("Number of bvals must match the number of bvecs.")
        if bvecs.shape[1] != 3:
            raise ValueError("bvecs must have 3 columns (bx, by, bz).")

        self._bvals: torch.Tensor = torch.from_numpy(bvals).float()
        self._bvecs: torch.Tensor = torch.from_numpy(bvecs).float()
        self._b0_threshold: float = b0_threshold

        self._b0s_mask: torch.Tensor = self._bvals <= self._b0_threshold
        self._dwis_mask: torch.Tensor = ~self._b0s_mask

    @property
    def bvals(self) -> torch.Tensor:
        """Returns the b-values as a PyTorch tensor."""
        return self._bvals

    @property
    def bvecs(self) -> torch.Tensor:
        """Returns the b-vectors as a PyTorch tensor."""
        return self._bvecs

    @property
    def b0s_mask(self) -> torch.Tensor:
        """
        Returns a boolean PyTorch tensor indicating which entries are b0 images
        (True for b0s, False for diffusion-weighted images).
        """
        return self._b0s_mask

    @property
    def dwis_mask(self) -> torch.Tensor:
        """
        Returns a boolean PyTorch tensor indicating which entries are diffusion-weighted images
        (True for DWIs, False for b0s).
        """
        return self._dwis_mask
        
    @property
    def gradients(self) -> torch.Tensor:
        """Returns the diffusion-weighted gradient vectors (bvecs for DWIs)."""
        return self._bvecs[self._dwis_mask]

    @property
    def b0_indices(self) -> torch.Tensor:
        """Returns the indices of b0 images."""
        return torch.where(self._b0s_mask)[0]

    @property
    def dwi_indices(self) -> torch.Tensor:
        """Returns the indices of diffusion-weighted images."""
        return torch.where(self._dwis_mask)[0]

    def __repr__(self) -> str:
        return (f"PyTorchGradientTable(bvals_shape={self.bvals.shape}, "
                f"bvecs_shape={self.bvecs.shape}, "
                f"num_b0s={torch.sum(self.b0s_mask).item()}, "
                f"num_dwis={torch.sum(self.dwis_mask).item()})")

if __name__ == '__main__':
    # Example Usage
    print("--- PyTorchGradientTable Example ---")
    example_bvals_np = np.array([0, 0, 1000, 1000, 2000, 0, 1000], dtype=np.float32)
    example_bvecs_np = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0] # Example, should be normalized for actual use if DWI
    ], dtype=np.float32)
    
    # Normalize the DWI bvecs for realistic example
    # Identify DWI indices based on b-values being greater than the threshold
    dwi_indices_example_mask = example_bvals_np > 50.0 # Using a hardcoded threshold for example clarity
    
    # Extract b-vectors that correspond to DWIs
    dwi_bvecs = example_bvecs_np[dwi_indices_example_mask]
    
    # Calculate norms for these DWI b-vectors
    # Add a small epsilon to prevent division by zero for zero vectors if they accidentally appear
    epsilon = 1e-8 
    bvecs_norms_example = np.linalg.norm(dwi_bvecs, axis=1, keepdims=True)
    
    # Perform normalization: divide by norm only where norm is not close to zero
    # Create a copy to modify
    normalized_dwi_bvecs = dwi_bvecs.copy()
    non_zero_norm_mask = (bvecs_norms_example > epsilon).squeeze() # Ensure it's a 1D mask
    
    # Apply normalization
    normalized_dwi_bvecs[non_zero_norm_mask] = dwi_bvecs[non_zero_norm_mask] / bvecs_norms_example[non_zero_norm_mask]
    
    # Place the normalized b-vectors back into the original array
    example_bvecs_np[dwi_indices_example_mask] = normalized_dwi_bvecs

    print("Input bvals (NumPy):", example_bvals_np)
    print("Input bvecs (NumPy) (normalized for DWIs):\n", example_bvecs_np)

    gtab_torch = PyTorchGradientTable(example_bvals_np, example_bvecs_np, b0_threshold=50.0)

    print("\nPyTorchGradientTable instance:", gtab_torch)
    print("bvals (Tensor):", gtab_torch.bvals)
    print("bvecs (Tensor):\n", gtab_torch.bvecs)
    print("b0s_mask (Tensor):", gtab_torch.b0s_mask)
    print("dwis_mask (Tensor):", gtab_torch.dwis_mask)
    print("Gradients (DWIs bvecs):\n", gtab_torch.gradients)
    print("B0 indices:", gtab_torch.b0_indices)
    print("DWI indices:", gtab_torch.dwi_indices)

    print("\nTesting with edge cases:")
    gtab_all_b0 = PyTorchGradientTable(np.array([0,0,10,0], dtype=np.float32), np.zeros((4,3), dtype=np.float32), b0_threshold=20)
    print("All b0s (effectively):", gtab_all_b0)
    print("  num_dwis should be 0:", torch.sum(gtab_all_b0.dwis_mask).item())
    print("  b0_indices:", gtab_all_b0.b0_indices)
    print("  dwi_indices:", gtab_all_b0.dwi_indices)


    # For all DWIs, b-vectors should be unit vectors
    all_dwi_bvecs = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
    gtab_all_dwi = PyTorchGradientTable(np.array([1000,1000,1000], dtype=np.float32), all_dwi_bvecs, b0_threshold=20)
    print("All DWIs:", gtab_all_dwi)
    print("  num_b0s should be 0:", torch.sum(gtab_all_dwi.b0s_mask).item())
    print("  b0_indices:", gtab_all_dwi.b0_indices)
    print("  dwi_indices:", gtab_all_dwi.dwi_indices)
    
    try:
        print("\nTesting validation (should raise ValueError for mismatched shapes):")
        PyTorchGradientTable(np.array([0,1000]), np.array([[0,0,0]])) 
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nTesting validation (should raise ValueError for wrong bvals dim):")
        PyTorchGradientTable(np.array([[0,1000]]), np.array([[0,0,0],[1,0,0]])) 
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nTesting validation (should raise ValueError for wrong bvecs dim):")
        PyTorchGradientTable(np.array([0,1000]), np.array([0,0,0,1,0,0]))
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        print("\nTesting validation (should raise ValueError for wrong bvecs shape[1]):")
        PyTorchGradientTable(np.array([0,1000]), np.array([[0,0],[1,0]]))
    except ValueError as e:
        print(f"Caught expected error: {e}")
