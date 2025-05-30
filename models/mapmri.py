import numpy as np
from dipy.reconst.mapmri import MapmriModel as DipyMapmriModel
from dipy.core.gradients import GradientTable
from dipy.data import get_sphere # For odf method

class MapmriModel:
    """
    A wrapper class for Dipy's MapmriModel.

    This class provides an interface to fit the MAP-MRI model to diffusion MRI data,
    allowing for the estimation of various microstructure-sensitive parameters.
    MAP-MRI requires multi-shell data for optimal performance.
    """
    def __init__(self, gtab, radial_order=6, laplacian_regularization=True, laplacian_weighting=0.2):
        """
        Initializes the MapmriModel.

        Parameters
        ----------
        gtab : GradientTable
            A Dipy GradientTable object containing gradient information.
            Multi-shell data is typically required for MAP-MRI.
        radial_order : int, optional
            The radial order of the basis functions. Default is 6.
        laplacian_regularization : bool, optional
            Whether to use Laplacian regularization. Default is True.
        laplacian_weighting : float or str, optional
            The Laplacian regularization weighting. If 'GCV', generalized
            cross-validation is used. Default is 0.2.
        """
        if not isinstance(gtab, GradientTable):
            raise ValueError("gtab must be an instance of Dipy's GradientTable.")
        
        self.gtab = gtab
        self.radial_order = radial_order
        self.laplacian_regularization = laplacian_regularization
        self.laplacian_weighting = laplacian_weighting
        
        self._dipy_model_instance = None # To store the configured Dipy model
        self._fit_object = None          # To store the Dipy fit object

    def fit(self, data):
        """
        Fits the MAP-MRI model to the provided diffusion-weighted image data.

        Parameters
        ----------
        data : ndarray
            A 4D NumPy array containing the diffusion-weighted image data.
            The dimensions should be (x, y, z, number_of_gradients).

        Raises
        ------
        ValueError
            If input data is not a 4D NumPy array.
        RuntimeError
            If Dipy model instantiation or fitting fails.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            raise ValueError("Input data must be a 4D NumPy array.")

        # Instantiate the Dipy MapmriModel
        try:
            self._dipy_model_instance = DipyMapmriModel(
                gtab=self.gtab,
                radial_order=self.radial_order,
                laplacian_regularization=self.laplacian_regularization,
                laplacian_weighting=self.laplacian_weighting
                # Other parameters like `small_delta`, `big_delta`, `anisotropic_scaling`,
                # `eigenvalue_threshold`, `bval_threshold`, `dti_regularization`,
                # `positivity_constraint`, `global_mat_norm` can be added if needed,
                # using Dipy's defaults for now.
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Dipy's MapmriModel: {e}")

        # Fit the model to the data
        try:
            self._fit_object = self._dipy_model_instance.fit(data)
        except Exception as e:
            # Common issues: data/gtab mismatch, insufficient data quality/shells for model parameters.
            raise RuntimeError(f"Failed to fit Dipy's MapmriModel to data: {e}")

    # Placeholder for future methods to extract MAP-MRI parameters (e.g., RTOP, RTAP, RTPP, QIV, MSD, NGD)
    @property
    def rtop(self):
        """Return to Origin Probability (RTOP).

        RTOP is a measure of the probability that water molecules have not moved
        from their origin over the diffusion time. Higher RTOP indicates more
        restricted diffusion.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.rtop()

    @property
    def rtap(self):
        """Return to Axis Probability (RTAP).

        RTAP quantifies the probability that water molecules diffuse primarily
        along a single axis. Higher RTAP suggests more anisotropic diffusion
        along a dominant direction.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.rtap()

    @property
    def rtpp(self):
        """Return to Plane Probability (RTPP).

        RTPP measures the probability that water molecules diffuse predominantly
        within a plane. Higher RTPP indicates planar anisotropy.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.rtpp()

    @property
    def msd(self):
        """Mean Squared Displacement (MSD).

        MSD provides an estimate of the average squared distance traveled by
        water molecules, reflecting the overall diffusivity.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.msd()

    @property
    def qiv(self):
        """Q-space Inverse Variance (QIV).

        QIV reflects the inverse variance of the diffusion signal in q-space,
        related to the sharpness of the diffusion probability density function.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.qiv()

    @property
    def ng(self):
        """Non-Gaussianity (NG).

        NG quantifies the deviation of the diffusion process from a Gaussian
        distribution. Higher NG indicates more complex microstructural environments.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.ng()

    def odf(self, sphere=None):
        """Orientation Distribution Functions (ODFs) from the MAP-MRI model.

        Parameters
        ----------
        sphere : dipy.data.Sphere, optional
            The sphere on which to sample the ODFs.
            If None (default), uses 'repulsion724' sphere from Dipy.

        Returns
        -------
        np.ndarray
            4D array containing the ODF values sampled on the sphere for each voxel.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        if sphere is None:
            sphere = get_sphere('repulsion724')
        return self._fit_object.odf(sphere)

    @property
    def gfa(self):
        """Generalized Fractional Anisotropy (GFA) derived from MAP-MRI parameters.

        This GFA is computed using the formula: sqrt(QIV) * RTAP, which is a known
        approximation for GFA in the context of MAP-MRI.
        Requires the model to be fitted.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        # Ensure qiv and rtap are available (they would raise error if not fitted)
        qiv_val = self._fit_object.qiv()
        rtap_val = self._fit_object.rtap()
        # GFA calculation: sqrt(QIV) * RTAP
        # Need to handle potential NaNs or negative values in qiv_val if they can occur
        # For simplicity, assuming qiv_val is non-negative as it represents variance.
        return np.sqrt(qiv_val) * rtap_val
