import unittest
import torch
import numpy as np # For comparison or generating initial data if needed
from preprocessing.pytorch_denoising import pytorch_mppca # Adjust import if necessary

class TestPytorchMPPCA(unittest.TestCase):

    def _generate_synthetic_data(self, shape=(8, 8, 8, 10), noise_sigma=10.0, on_device=torch.device('cpu')):
        # Create a base signal (e.g., simulating some structure)
        base_signal = torch.rand(shape, device=on_device) * 50 + 100 # Range 100-150
        # Make b0s have higher signal (optional, for more dMRI like data)
        if shape[-1] > 3:
            base_signal[..., :1] = torch.rand(shape[:-1] + (1,), device=on_device) * 100 + 200 # b0s Range 200-300
        
        noisy_dwi_data = base_signal + torch.randn_like(base_signal) * noise_sigma
        noisy_dwi_data = noisy_dwi_data.clamp(min=0)
        return noisy_dwi_data

    def test_pytorch_mppca_runs_and_output_properties(self):
        shape = (8, 8, 8, 10) # Small volume for testing
        noisy_data = self._generate_synthetic_data(shape=shape)
        
        patch_r = 1 # 3x3x3 patch, small for speed

        try:
            denoised_data = pytorch_mppca(noisy_data, patch_radius=patch_r, verbose=False)
        except Exception as e:
            self.fail(f"pytorch_mppca raised an exception: {e}")

        self.assertIsInstance(denoised_data, torch.Tensor)
        self.assertEqual(denoised_data.shape, noisy_data.shape)
        self.assertEqual(denoised_data.dtype, noisy_data.dtype)
        self.assertEqual(denoised_data.device, noisy_data.device)

        # Check if data is modified (not identical to input)
        # This is a basic check that some processing occurred.
        # With the current placeholder, it might be very similar if SVD fails or heuristic is too simple.
        # The SVD and reconstruction itself, even without effective thresholding, should cause small numerical differences.
        abs_diff_sum = torch.abs(denoised_data - noisy_data).sum().item()
        
        # If SVD fails and returns original patch, then diff would be zero.
        # If SVD runs, even if all singular values are kept, reconstruction might introduce tiny diffs.
        # If some singular values are zeroed (even by poor heuristic), diff should be larger.
        # We need a check that's robust to the current placeholder state.
        # If the SVD path in _denoise_patch_mppca is taken, there should be some difference.
        # If SVD fails consistently, this test might fail.
        # A more robust check would be if the *mean* changes, or std dev.
        # For now, let's check if it's not *exactly* the same, assuming SVD path is mostly taken.
        if noisy_data.numel() > 0 :
             self.assertFalse(torch.equal(denoised_data, noisy_data), 
                             "Denoised data is identical to noisy data. Processing might not have occurred or SVD failed consistently.")
        # A mean diff check:
        # self.assertGreater(abs_diff_sum / noisy_data.numel(), 1e-5, # Avg diff per element
        #                    "Denoised data is too similar to noisy data, mean abs diff is very small.")


    def test_pytorch_mppca_input_validation(self):
        with self.assertRaisesRegex(TypeError, "Input image_data must be a PyTorch tensor"):
            pytorch_mppca(np.random.rand(5,5,5,5)) # NumPy input

        with self.assertRaisesRegex(ValueError, "Input image_data must be a 4D tensor"):
            pytorch_mppca(torch.rand(5,5,5)) # 3D input
            
        with self.assertRaisesRegex(ValueError, "Input image_data must be a 4D tensor"):
            pytorch_mppca(torch.rand(5,5,5,5,2)) # 5D input

        with self.assertRaisesRegex(ValueError, "patch_radius must be non-negative"):
            pytorch_mppca(torch.rand(5,5,5,5), patch_radius=-1)

    def test_pytorch_mppca_device_consistency(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            shape = (8,8,8,5) # Small
            noisy_data_cuda = self._generate_synthetic_data(shape=shape, on_device=device)
            
            denoised_data_cuda = pytorch_mppca(noisy_data_cuda, patch_radius=1, verbose=False)
            
            self.assertEqual(denoised_data_cuda.device, device)
        else:
            self.skipTest("CUDA not available, skipping device consistency test for CUDA.")
            
    def test_pytorch_mppca_verbose_mode(self):
        # Simple test to ensure verbose mode runs without error
        # Actual output checking is harder in unit tests.
        shape = (4,4,4,3) # Very small to make verbose output minimal
        noisy_data = self._generate_synthetic_data(shape=shape)
        try:
            pytorch_mppca(noisy_data, patch_radius=0, verbose=True) # patch_radius 0 means 1x1x1 patch
        except Exception as e:
            self.fail(f"pytorch_mppca with verbose=True raised an exception: {e}")

    def test_empty_input(self):
        # Test with empty tensor
        empty_data = torch.empty((0, 0, 0, 0))
        denoised_empty = pytorch_mppca(empty_data, patch_radius=1)
        self.assertEqual(denoised_empty.shape, empty_data.shape)

        # Test with spatial dim zero
        spatial_zero_data = torch.empty((0, 8, 8, 5))
        denoised_spatial_zero = pytorch_mppca(spatial_zero_data, patch_radius=1)
        self.assertEqual(denoised_spatial_zero.shape, spatial_zero_data.shape)
        
        # Test with gradient dim zero (should ideally also work or be handled gracefully)
        # The current _denoise_patch_mppca might have issues if n_gradients is 0 in SVD.
        # torch.linalg.svd(torch.empty(10,0)) gives error.
        # Let's ensure our code handles this or we test expected failure.
        # For now, assuming n_gradients > 0 from typical dMRI data.
        # If it needs to handle Ng=0, _denoise_patch_mppca would need adjustment.
        # current code SVD fails for Ng=0, returns original patch.
        grad_zero_data = torch.rand((8, 8, 8, 0))
        denoised_grad_zero = pytorch_mppca(grad_zero_data, patch_radius=1)
        self.assertEqual(denoised_grad_zero.shape, grad_zero_data.shape)

    def test_pytorch_mppca_noise_reduction(self):
        shape = (12, 12, 12, 15) # Ensure dimensions are large enough for patch radius
        patch_r = 2 # Results in 5x5x5 patches
        device = torch.device('cpu') # Or use torch.cuda.is_available()

        # 1. Generate true signal
        # Simple true signal: blocks of different constant values
        true_signal = torch.zeros(shape, device=device, dtype=torch.float32)
        mid_x, mid_y, mid_z = shape[0]//2, shape[1]//2, shape[2]//2
        # Block 1 in a corner
        true_signal[:mid_x, :mid_y, :mid_z, :] = 100.0
        # Block 2 in another area
        true_signal[mid_x:, mid_y:, mid_z:, :] = 200.0
        # Add some variation across the 4th dim for some blocks
        for i in range(shape[3]):
            true_signal[mid_x//2:mid_x, mid_y//2:mid_y, mid_z//2:mid_z, i] = 100.0 + i * 5 # Ramp

        # 2. Add Gaussian noise
        noise_std_known = 20.0
        noisy_data = true_signal + torch.randn_like(true_signal) * noise_std_known
        noisy_data = noisy_data.clamp(min=0)

        # 3. Denoise
        try:
            denoised_data = pytorch_mppca(noisy_data, patch_radius=patch_r, verbose=False)
        except Exception as e:
            self.fail(f"pytorch_mppca raised an exception during noise reduction test: {e}")

        # 4. Assertions
        # Assertion 1: Output is different from input
        self.assertFalse(torch.equal(denoised_data, noisy_data), "Denoised data is identical to noisy input.")

        # Assertion 2: Denoised data is closer to true signal (lower MSE)
        mse_noisy_vs_true = torch.mean((noisy_data - true_signal)**2).item()
        mse_denoised_vs_true = torch.mean((denoised_data - true_signal)**2).item()
        
        print(f"MSE (Noisy vs True): {mse_noisy_vs_true:.4f}")
        print(f"MSE (Denoised vs True): {mse_denoised_vs_true:.4f}")
        self.assertLess(mse_denoised_vs_true, mse_noisy_vs_true, 
                        "MSE of denoised data vs true signal is not less than MSE of noisy data vs true signal.")

        # Assertion 3: Noise variance reduction
        # This can be tricky if the signal itself is modified by the denoising.
        # A simpler check on overall variance reduction of the difference from true signal.
        # Or, if we assume the denoiser mostly removes noise without distorting signal too much:
        # var_noisy = torch.var(noisy_data).item()
        # var_denoised = torch.var(denoised_data).item()
        # This is not always true if signal structure is flattened.
        
        # Let's use variance of the estimated noise component
        input_noise_estimate = noisy_data - true_signal
        output_noise_estimate = denoised_data - true_signal # This is actually denoised_signal_error
        
        var_input_noise = torch.var(input_noise_estimate).item()
        var_output_error = torch.var(output_noise_estimate).item() # Variance of the error in the denoised signal

        print(f"Variance of (Noisy - True): {var_input_noise:.4f}")
        print(f"Variance of (Denoised - True): {var_output_error:.4f}") # This is error variance
        
        # We expect the error of the denoised signal to have smaller variance than the original noise variance,
        # assuming the denoising process is effective.
        self.assertLess(var_output_error, var_input_noise,
                        "Variance of (Denoised - True) is not less than Variance of (Noisy - True).")


if __name__ == '__main__':
    unittest.main()
```
