import unittest
import torch
import numpy as np
from preprocessing.pytorch_masking import pytorch_median_otsu, pytorch_otsu_threshold # Adjust import if necessary

# Define EPS_OTSU if it's used for comparisons in tests, otherwise not strictly needed here.
# from preprocessing.pytorch_masking import EPS_OTSU # Or define locally if needed for a check

class TestPytorchMasking(unittest.TestCase):

    def test_pytorch_otsu_threshold_simple(self):
        # Simple bimodal image
        img = torch.zeros(100, 100, dtype=torch.float32)
        img[25:75, 25:75] = 200 # Foreground
        img[0:25, :] = 50       # Background area 1
        img[:, 0:25] = 50       # Background area 2
        
        # Expected threshold should be somewhere between 50 and 200
        threshold = pytorch_otsu_threshold(img)
        self.assertGreater(threshold, 40) # Allow some leeway from pure 50
        self.assertLess(threshold, 210) # Allow some leeway from pure 200
        # A more precise expectation would require knowing the exact histogram and variance calculation
        # For a clear bimodal distribution like this, it should be roughly in the middle.
        # Otsu tends to place it closer to the larger peak if classes are imbalanced in size.
        # Here, foreground is 50x50=2500, background is 10000-2500=7500. Background peak is larger.
        # So threshold might be closer to 50 than to 200. Let's say between 50 and 125.
        self.assertGreaterEqual(threshold, 50)
        self.assertLessEqual(threshold, 125, f"Threshold {threshold} not in expected range for simple bimodal.")

    def test_pytorch_otsu_threshold_flat_image(self):
        img_flat_zeros = torch.zeros(50, 50, dtype=torch.float32)
        self.assertEqual(pytorch_otsu_threshold(img_flat_zeros), 0)

        img_flat_ones = torch.ones(50, 50, dtype=torch.float32) * 150
        self.assertEqual(pytorch_otsu_threshold(img_flat_ones), 150)

    def test_pytorch_otsu_threshold_normalized_range(self):
        # Test with image already in 0-1 range
        img = torch.rand(64, 64, dtype=torch.float32) # Random values
        img_min = img.min().item()
        img_max = img.max().item()

        # Scale to a specific small range, e.g., 0.2 to 0.5
        # To ensure the de-normalization logic is tested, we need to know the exact min/max after scaling.
        # Let's create an image with a clear min and max for testing de-normalization.
        img_test = torch.linspace(0.2, 0.5, 100*100).reshape(100,100)

        threshold_scaled = pytorch_otsu_threshold(img_test)
        
        # Threshold should be within the new min/max of img_test (0 or 1 if using int conversion)
        # Since pytorch_otsu_threshold returns an int, and input is 0.2 to 0.5,
        # min_val will be 0, max_val will be 0 after int conversion in some parts of the logic
        # if not handled carefully. The internal normalization to 0-255 should handle this.
        # The final threshold is scaled back to the original image's min/max range.
        # So, if img_test is e.g. 0.2 to 0.5, the threshold should be an integer close to this range.
        # For linspace 0.2 to 0.5, the threshold should be around 0.35 * 255 (normalized) then scaled back.
        # It should be an int.
        # The `pytorch_otsu_threshold` scales the final threshold:
        # `int(min_val_original + (optimal_thresh_norm / 255.0) * (max_val_original - min_val_original))`
        # If min_val_original=0.2, max_val_original=0.5, the returned threshold will be an int.
        # For this specific linspace, it will likely be 0.
        
        # Let's test a more robust case for normalization:
        img_0_10 = torch.linspace(0, 10, 100*100).reshape(100,100)
        thresh_0_10 = pytorch_otsu_threshold(img_0_10)
        self.assertGreaterEqual(thresh_0_10, 0)
        self.assertLessEqual(thresh_0_10, 10)
        # For a linspace, it should be around the middle.
        self.assertGreaterEqual(thresh_0_10, 4) # Expect around 5
        self.assertLessEqual(thresh_0_10, 6)


    def test_pytorch_median_otsu_basic(self):
        # Create a synthetic 3D volume (e.g., sphere)
        shape = (32, 32, 32) # Smaller for faster test
        center = tuple(s // 2 for s in shape)
        radius = shape[0] // 4
        
        coords = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing='ij')
        sphere_mask_gt = sum([(coords[i] - center[i])**2 for i in range(3)]) < radius**2
        
        synthetic_volume = torch.zeros(shape, dtype=torch.float32)
        synthetic_volume[sphere_mask_gt] = 200.0 # Sphere intensity
        synthetic_volume[~sphere_mask_gt] = 50.0  # Background intensity
        
        # Add a little noise
        synthetic_volume_noisy = synthetic_volume + torch.randn_like(synthetic_volume) * 10.0
        synthetic_volume_noisy = synthetic_volume_noisy.clamp(0, 255)

        brain_mask, masked_volume = pytorch_median_otsu(
            synthetic_volume_noisy, 
            median_radius=2, # Smaller radius for test speed
            numpass=1        # Fewer passes for test speed
        )

        self.assertEqual(brain_mask.shape, shape)
        self.assertEqual(masked_volume.shape, shape)
        self.assertEqual(brain_mask.dtype, torch.bool)
        self.assertEqual(masked_volume.dtype, torch.float32) # Should match input data type

        # Mask should not be all True or all False
        num_masked_voxels = torch.sum(brain_mask).item()
        self.assertGreater(num_masked_voxels, 0, "Mask is all False")
        self.assertLess(num_masked_voxels, brain_mask.numel(), "Mask is all True")

        # For this simple case, the mask should roughly capture the sphere
        # Intersection over Union (IoU) or Dice score could be used for better validation
        # For now, a simpler check: at least some overlap with ground truth sphere,
        # and some non-overlap in the background.
        intersection = torch.sum(brain_mask & sphere_mask_gt).item()
        union = torch.sum(brain_mask | sphere_mask_gt).item()
        iou = intersection / (union + 1e-6) # Add epsilon to avoid div by zero if union is 0
        
        print(f"IoU for median_otsu mask: {iou:.3f}")
        self.assertGreater(iou, 0.5, "Mask does not sufficiently overlap with the synthetic sphere (IoU < 0.5).")

        # Check that masked_volume is zero where brain_mask is False (approximately)
        # The multiplication `synthetic_volume_noisy * brain_mask` ensures this.
        self.assertTrue(torch.allclose(masked_volume[~brain_mask], torch.zeros_like(masked_volume[~brain_mask]), atol=1e-5))


    def test_pytorch_median_otsu_input_type_error(self):
        with self.assertRaisesRegex(TypeError, "Input image_data must be a PyTorch tensor"):
            pytorch_median_otsu(np.random.rand(10,10,10))

    def test_pytorch_median_otsu_input_dim_error(self):
        with self.assertRaisesRegex(ValueError, "Input image_data must be a 3D tensor"):
            pytorch_median_otsu(torch.rand(10,10)) # 2D input
        with self.assertRaisesRegex(ValueError, "Input image_data must be a 3D tensor"):
            pytorch_median_otsu(torch.rand(10,10,10,2)) # 4D input
            
    def test_pytorch_median_otsu_device_consistency(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Create a simple volume on CUDA
            shape = (16, 16, 16)
            synthetic_volume_cuda = torch.rand(shape, dtype=torch.float32, device=device) * 100
            
            brain_mask_cuda, masked_volume_cuda = pytorch_median_otsu(
                synthetic_volume_cuda, 
                median_radius=1, 
                numpass=1
            )
            
            self.assertEqual(brain_mask_cuda.device, device)
            self.assertEqual(masked_volume_cuda.device, device)
        else:
            self.skipTest("CUDA not available, skipping device consistency test for CUDA.")


if __name__ == '__main__':
    unittest.main()

```
