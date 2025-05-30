import unittest
import torch
import numpy as np
from dipy.core.gradients import gradient_table as create_dipy_gradient_table
from dipy.tracking.streamline import Streamlines as DipyStreamlines
from dipy.data import get_sphere
from dipy.sims.voxel import single_tensor # For realistic synthetic signal

from tracking.pytorch_tracking_pipeline import track_deterministic_oudf

class TestPyTorchTrackingPipeline(unittest.TestCase):

    def setUp(self):
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_str)

        # GradientTable with enough directions for SH order 4 or 6
        sphere = get_sphere('repulsion24') # 24 DWI directions
        bvals_np = np.concatenate([np.zeros(6), np.ones(sphere.vertices.shape[0]) * 1000])
        bvecs_np = np.vstack([np.zeros((6, 3)), sphere.vertices])
        self.gtab_dipy = create_dipy_gradient_table(bvals_np, bvecs_np, b0_threshold=50)

        self.affine_torch = torch.eye(4, dtype=torch.float32, device=self.device)
        self.affine_np = self.affine_torch.cpu().numpy()

        self.vol_shape = (10, 10, 10) 
        s0 = 100.0
        # Eigenvalues for a prolate tensor (fiber-like)
        evals = np.array([0.0017, 0.0003, 0.0003]) 
        # Generate signal for a single fiber population oriented along X-axis ([1,0,0])
        # For X-axis, phi=0, theta=pi/2. Or use Dipy's default random orientation for simplicity.
        # To make peaks predictable, let's try to make it oriented along X.
        # single_tensor orients along first b-vector if not specified.
        # Let's create a simple orientation vector.
        fiber_orientation = np.array([1.0, 0.0, 0.0]) # X-axis
        
        # Create a data array where each voxel has this single fiber orientation
        dwi_data_list_voxels = []
        for _ in range(np.prod(self.vol_shape)):
            # Forcing orientation by rotating gtab for each voxel is complex.
            # Easier: generate one voxel, then tile.
            # The orientation in single_tensor is random by default.
            # To control it, we can pass `eigvecs`.
            # e1=[1,0,0], e2=[0,1,0], e3=[0,0,1] for X-axis fiber
            eig_vecs = np.array([[1,0,0],[0,1,0],[0,0,1]]).T # Columns are eigenvectors
            voxel_signal = single_tensor(self.gtab_dipy, S0=s0, evals=evals, evecs=eig_vecs, snr=None)
            dwi_data_list_voxels.append(voxel_signal)
        
        dwi_data_np = np.array(dwi_data_list_voxels).reshape(self.vol_shape + (len(self.gtab_dipy.bvals),)).astype(np.float32)
        self.dwi_data_np = dwi_data_np # Keep a NumPy version for some potential Dipy utilities if needed in future tests
        self.dwi_data_torch = torch.tensor(dwi_data_np, device=self.device)

        self.metric_map_np = np.zeros(self.vol_shape, dtype=np.float32)
        self.metric_map_np[3:7, 3:7, 3:7] = 0.8 # Tracking allowed here
        self.metric_map_torch = torch.tensor(self.metric_map_np, device=self.device)
        self.stopping_thresh = 0.2

        self.seeds_coords_np = np.array([[4,4,4], [5,5,5]], dtype=np.float32)
        self.seeds_coords_torch = torch.tensor(self.seeds_coords_np, device=self.device)
        
        self.seed_mask_np = np.zeros(self.vol_shape, dtype=bool)
        self.seed_mask_np[4:6, 4:6, 4:6] = True
        self.seed_mask_torch = torch.tensor(self.seed_mask_np, device=self.device) # Bool tensor for mask

    def test_tracking_basic_run_coord_seeds(self):
        streamlines_obj = track_deterministic_oudf(
            dwi_data=self.dwi_data_np,
            gtab=self.gtab_dipy,
            affine=self.affine_np,
            stopping_metric_map=self.metric_map_torch,
            stopping_threshold_value=self.stopping_thresh,
            seeds_input=self.seeds_coords_torch, # Pass torch tensor directly
            step_size=0.5, sh_order=4, model_max_peaks=1, # Faster for test
            min_length=1.0, max_length=20.0, max_steps=30, # Shorter tracks for test
            device=self.device_str
        )
        self.assertIsInstance(streamlines_obj, DipyStreamlines)
        self.assertGreater(len(streamlines_obj), 0, "No streamlines generated with coordinate seeds.")
        for sl in streamlines_obj:
            self.assertIsInstance(sl, np.ndarray)
            self.assertTrue(sl.ndim == 2 and sl.shape[1] == 3)

    def test_tracking_mask_seeds(self):
        streamlines_obj = track_deterministic_oudf(
            dwi_data=self.dwi_data_np,
            gtab=self.gtab_dipy,
            affine=self.affine_np,
            stopping_metric_map=self.metric_map_torch,
            stopping_threshold_value=self.stopping_thresh,
            seeds_input=self.seed_mask_torch, # Pass bool tensor mask
            step_size=0.5, sh_order=4, model_max_peaks=1,
            min_length=1.0, max_length=20.0, max_steps=30,
            device=self.device_str
        )
        self.assertIsInstance(streamlines_obj, DipyStreamlines)
        self.assertGreater(len(streamlines_obj), 0, "No streamlines generated with mask seeds.")

    def test_tracking_no_seeds_default_seeding(self):
        streamlines_obj = track_deterministic_oudf(
            dwi_data=self.dwi_data_np,
            gtab=self.gtab_dipy,
            affine=self.affine_np,
            stopping_metric_map=self.metric_map_torch, 
            stopping_threshold_value=self.stopping_thresh, 
            seeds_input=None, # Trigger default seeding
            step_size=0.5, sh_order=4, model_max_peaks=1,
            min_length=1.0, max_length=20.0, max_steps=30,
            device=self.device_str
        )
        self.assertIsInstance(streamlines_obj, DipyStreamlines)
        # Default seeding uses metric_map > threshold. Our metric_map has a region > threshold.
        self.assertGreater(len(streamlines_obj), 0, "No streamlines generated with default seeding.")

    def test_tracking_length_filters(self):
        # Test min_length
        streamlines_min = track_deterministic_oudf(
            dwi_data=self.dwi_data_np, gtab=self.gtab_dipy, affine=self.affine_np,
            stopping_metric_map=self.metric_map_torch, stopping_threshold_value=0.05, # Allow longer tracks
            seeds_input=self.seeds_coords_torch, 
            step_size=1.0, min_length=10.0, max_steps=5, # 5 steps * 1.0 = 5mm length. Should be filtered.
            sh_order=4, model_max_peaks=1, device=self.device_str, max_length=20.0
        )
        self.assertEqual(len(streamlines_min), 0, "Min_length filter failed.")

        # Test max_length
        metric_all_go = torch.ones(self.vol_shape, device=self.device) * 0.8
        streamlines_max = track_deterministic_oudf(
            dwi_data=self.dwi_data_np, gtab=self.gtab_dipy, affine=self.affine_np,
            stopping_metric_map=metric_all_go, stopping_threshold_value=0.1,
            seeds_input=torch.tensor([[5.0,5.0,1.0]], device=self.device, dtype=torch.float32), 
            step_size=0.5, max_length=2.0, max_steps=20, 
            sh_order=4, model_max_peaks=1, device=self.device_str, min_length=0.1
        )
        if len(streamlines_max) > 0:
            for sl in streamlines_max:
                sl_len = np.sum(np.sqrt(np.sum(np.diff(sl, axis=0)**2, axis=1)))
                self.assertLessEqual(sl_len, 2.0 + 0.5, "Streamline exceeded max_length significantly.")
        else:
            print("Warning: Max_length test produced no streamlines. Check test setup.")


    def test_tracking_stopping_criterion(self):
        # Create a metric map where seeds are in a "stop" region
        metric_map_stop_at_seed = torch.ones(self.vol_shape, device=self.device) * 0.8
        # Seed at [4,4,4]. Set this region to be below threshold.
        metric_map_stop_at_seed[4,4,4] = 0.1 
        
        streamlines_obj = track_deterministic_oudf(
            dwi_data=self.dwi_data_np, gtab=self.gtab_dipy, affine=self.affine_np,
            stopping_metric_map=metric_map_stop_at_seed, 
            stopping_threshold_value=0.2, # Threshold is 0.2
            seeds_input=torch.tensor([[4.0,4.0,4.0]], device=self.device, dtype=torch.float32),
            step_size=0.5, sh_order=4, model_max_peaks=1,
            min_length=0.1, # Allow very short streamlines (or just the seed point)
            max_steps=5, device=self.device_str
        )
        # Expect 0 streamlines as seed is in stopping region, or streamlines with <=1 point if not filtered by min_length=0
        # The current LocalTracking returns list of arrays. Streamlines with 1 point (seed only) might be produced
        # if it stops immediately. min_length=0.1 means at least one step.
        self.assertEqual(len(streamlines_obj), 0, "Streamlines generated despite seed in stopping region.")

    def test_tracking_sh_order_param(self):
        # Test with a different SH order to ensure it runs
        try:
            track_deterministic_oudf(
                dwi_data=self.dwi_data_np, gtab=self.gtab_dipy, affine=self.affine_np,
                stopping_metric_map=self.metric_map_torch, stopping_threshold_value=self.stopping_thresh,
                seeds_input=self.seeds_coords_torch,
                sh_order=6, # Different from default/other tests
                device=self.device_str
            )
        except Exception as e:
            self.fail(f"track_deterministic_oudf failed with sh_order=6: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tracking_cuda_device(self):
        # Ensure data is on CUDA for this test
        dwi_cuda = self.dwi_data_torch.to('cuda')
        affine_cuda = self.affine_torch.to('cuda')
        metric_map_cuda = self.metric_map_torch.to('cuda')
        seeds_cuda = self.seeds_coords_torch.to('cuda')

        streamlines_obj = track_deterministic_oudf(
            dwi_data=dwi_cuda.cpu().numpy(), # Wrapper expects numpy
            gtab=self.gtab_dipy,
            affine=affine_cuda.cpu().numpy(), # Wrapper expects numpy
            stopping_metric_map=metric_map_cuda, # Pipeline expects torch tensor
            stopping_threshold_value=self.stopping_thresh,
            seeds_input=seeds_cuda, # Pipeline expects torch tensor
            device='cuda' # Explicitly request CUDA
        )
        self.assertIsInstance(streamlines_obj, DipyStreamlines)
        # Streamline data itself is NumPy, check that some were generated
        self.assertGreater(len(streamlines_obj), 0, "No streamlines on CUDA test.")


if __name__ == '__main__':
    unittest.main()
```
