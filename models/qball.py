import numpy as np
from dipy.reconst.qball import QballModel as DipyQballModel
from dipy.core.gradients import GradientTable
from dipy.data import get_sphere # Added for default sphere in odf method
from dipy.reconst.shm import gfa # Added for GFA calculation

class QballModel:
    """
    A wrapper class for Dipy's QballModel (Constant Solid Angle - CSA type).

    This class provides an interface to fit the Q-ball model to diffusion MRI data.
    Q-ball models are a type of spherical deconvolution that can resolve multiple
    fiber orientations in a voxel from single-shell, high angular resolution
    diffusion imaging (HARDI) data.
    """
    def __init__(self, gtab, smooth=0.006):
        """
        Initializes the QballModel.

        Parameters
        ----------
        gtab : GradientTable
            A Dipy GradientTable object containing gradient information.
            Typically, this should be single-shell HARDI data.
        smooth : float, optional
            Regularization parameter for the Q-ball model (Laplace-Beltrami).
            Default is 0.006, a common value used in examples.
        """
        if not isinstance(gtab, GradientTable):
            raise ValueError("gtab must be an instance of Dipy's GradientTable.")
        self.gtab = gtab
        self.smooth = smooth
        self._fit_object = None
        self._dipy_model_instance = None # To store the model instance itself

    def fit(self, data, sh_order_max=8):
        """
        Fits the Q-ball model to the provided diffusion-weighted image data.

        Parameters
        ----------
        data : ndarray
            A 4D NumPy array containing the diffusion-weighted image data.
            The dimensions should be (x, y, z, number_of_gradients).
        sh_order_max : int, optional
            Maximum spherical harmonics order to use for the Q-ball model.
            Default is 8. This determines the complexity and angular resolution
            of the model.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            raise ValueError("Input data must be a 4D NumPy array.")

        # Instantiate the Dipy QballModel
        # Dipy's QballModel is typically instantiated with gtab, sh_order_max, and regularization terms.
        # The 'data' is then passed to its 'fit' method.
        try:
            self._dipy_model_instance = DipyQballModel(
                gtab=self.gtab,
                sh_order_max=sh_order_max,
                smooth=self.smooth,
                # Additional parameters like `assume_normed=True` might be relevant
                # depending on data normalization, but we'll use Dipy's defaults here.
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Dipy's QballModel: {e}")

        # Fit the model to the data
        try:
            self._fit_object = self._dipy_model_instance.fit(data)
        except Exception as e:
            # Common issues here could be data/gtab mismatch, insufficient b-values/directions for sh_order, etc.
            raise RuntimeError(f"Failed to fit Dipy's QballModel to data: {e}")

    def odf(self, sphere=None):
        """
        Calculate Orientation Distribution Functions (ODFs) from the fitted Q-ball model.

        Parameters
        ----------
        sphere : dipy.data.Sphere, optional
            The sphere on which to sample the ODFs.
            If None (default), uses 'repulsion724' sphere from Dipy.

        Returns
        -------
        np.ndarray
            4D array containing the ODF values sampled on the sphere for each voxel.
            Shape will be (X, Y, Z, num_sphere_vertices).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")

        if sphere is None:
            sphere = get_sphere('repulsion724')
        
        # Dipy's QballFit object's odf method will handle the sphere.
        return self._fit_object.odf(sphere)

    @property
    def gfa(self):
        """Generalized Fractional Anisotropy (GFA) from the Q-Ball model.

        Requires the model to be fitted.
        Calculated from the spherical harmonic coefficients.

        Returns
        -------
        np.ndarray
            3D array of GFA values.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        AttributeError
            If the fit object does not have 'shm_coeff'.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        if not hasattr(self._fit_object, 'shm_coeff'):
            # Dipy's QballFit object should have shm_coeff
            raise AttributeError("The Q-Ball fit object does not have 'shm_coeff', cannot calculate GFA.")
        # Dipy's gfa function expects spherical harmonic coefficients
        return gfa(self._fit_object.shm_coeff)
