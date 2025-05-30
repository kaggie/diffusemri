import unittest
import torch
import numpy as np
from utils.pytorch_gradient_utils import PyTorchGradientTable # Adjust import if utils is not directly in PYTHONPATH

class TestPyTorchGradientTable(unittest.TestCase):

    def setUp(self):
        self.bvals_np = np.array([0, 1000, 1000, 0, 2000, 1000], dtype=np.float32)
        self.bvecs_np = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0.577, 0.577, 0.577] # Approx 1/sqrt(3)
        ], dtype=np.float32)
        self.b0_thresh = 50.0
        self.gtab = PyTorchGradientTable(self.bvals_np, self.bvecs_np, b0_threshold=self.b0_thresh)

    def test_instantiation_and_properties(self):
        self.assertIsInstance(self.gtab.bvals, torch.Tensor)
        self.assertIsInstance(self.gtab.bvecs, torch.Tensor)
        self.assertIsInstance(self.gtab.b0s_mask, torch.Tensor)
        self.assertIsInstance(self.gtab.dwis_mask, torch.Tensor)

        self.assertTrue(torch.equal(self.gtab.bvals, torch.from_numpy(self.bvals_np).float()))
        self.assertTrue(torch.equal(self.gtab.bvecs, torch.from_numpy(self.bvecs_np).float()))
        
        expected_b0s_mask = torch.tensor([True, False, False, True, False, False])
        self.assertTrue(torch.equal(self.gtab.b0s_mask, expected_b0s_mask))
        
        expected_dwis_mask = torch.tensor([False, True, True, False, True, True])
        self.assertTrue(torch.equal(self.gtab.dwis_mask, expected_dwis_mask))

    def test_b0_dwis_indices(self):
        expected_b0_indices = torch.tensor([0, 3])
        self.assertTrue(torch.equal(self.gtab.b0_indices, expected_b0_indices))

        expected_dwi_indices = torch.tensor([1, 2, 4, 5])
        self.assertTrue(torch.equal(self.gtab.dwi_indices, expected_dwi_indices))

    def test_gradients_property(self):
        expected_gradients = torch.from_numpy(self.bvecs_np[self.bvals_np > self.b0_thresh]).float()
        self.assertTrue(torch.equal(self.gtab.gradients, expected_gradients))
        self.assertEqual(self.gtab.gradients.shape[0], 4) # Number of DWIs

    def test_repr_method(self):
        self.assertIn("PyTorchGradientTable", repr(self.gtab))
        self.assertIn(f"num_b0s=2", repr(self.gtab))
        self.assertIn(f"num_dwis=4", repr(self.gtab))

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "bvals must be a 1D NumPy array"):
            PyTorchGradientTable(np.array([[0,0]]), self.bvecs_np) # bvals not 1D
        
        with self.assertRaisesRegex(ValueError, "bvecs must be a 2D NumPy array"):
            PyTorchGradientTable(self.bvals_np, np.array([0,0,0])) # bvecs not 2D
            
        with self.assertRaisesRegex(ValueError, "Number of bvals must match the number of bvecs"):
            PyTorchGradientTable(self.bvals_np, self.bvecs_np[:-1]) # Mismatched number
            
        with self.assertRaisesRegex(ValueError, "bvecs must have 3 columns"):
            PyTorchGradientTable(self.bvals_np, np.random.rand(self.bvals_np.shape[0], 2)) # bvecs not Nx3

    def test_edge_case_all_b0s(self):
        bvals_all_b0 = np.array([0, 0, 10, 0], dtype=np.float32)
        bvecs_all_b0 = np.zeros((4, 3), dtype=np.float32)
        gtab_all_b0 = PyTorchGradientTable(bvals_all_b0, bvecs_all_b0, b0_threshold=20.0)
        
        self.assertEqual(torch.sum(gtab_all_b0.b0s_mask).item(), 4)
        self.assertEqual(torch.sum(gtab_all_b0.dwis_mask).item(), 0)
        self.assertEqual(gtab_all_b0.gradients.shape[0], 0)
        self.assertEqual(gtab_all_b0.dwi_indices.shape[0], 0)
        self.assertEqual(gtab_all_b0.b0_indices.shape[0], 4)

    def test_edge_case_all_dwis(self):
        bvals_all_dwi = np.array([1000, 1000, 2000], dtype=np.float32)
        bvecs_all_dwi = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float32)
        gtab_all_dwi = PyTorchGradientTable(bvals_all_dwi, bvecs_all_dwi, b0_threshold=50.0)
        
        self.assertEqual(torch.sum(gtab_all_dwi.b0s_mask).item(), 0)
        self.assertEqual(torch.sum(gtab_all_dwi.dwis_mask).item(), 3)
        self.assertEqual(gtab_all_dwi.gradients.shape[0], 3)
        self.assertTrue(torch.equal(gtab_all_dwi.gradients, torch.from_numpy(bvecs_all_dwi).float()))

if __name__ == '__main__':
    unittest.main()
