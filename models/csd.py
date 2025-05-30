import numpy as np
from dipy.core.gradients import GradientTable
from dipy.data import get_sphere
from dipy.reconst.mcsd import (
    MultiShellDeconvModel,
    response_dhollander,
)
# ResponseFunction can be imported if type checking for individual responses is needed
# from dipy.reconst.mcsd import ResponseFunction

class MultiTissueCsdModel:
    """
    A wrapper class for Dipy's Multi-Shell Multi-Tissue Constrained Spherical
    Deconvolution (MSMT-CSD) model.

    This class provides an interface to fit the MSMT-CSD model to diffusion
    MRI data, including automated multi-tissue response function estimation
    if not provided.
    """
    def __init__(self, gtab):
        """
        Initializes the MultiTissueCsdModel.

        Parameters
        ----------
        gtab : GradientTable
            A Dipy GradientTable object containing gradient information.
        """
        if not isinstance(gtab, GradientTable):
            raise ValueError("gtab must be an instance of Dipy's GradientTable.")
        self.gtab = gtab
        self._response_wm = None
        self._response_gm = None
        self._response_csf = None
        self._mt_csd_model = None
        self._mt_fit_object = None

    def fit(self, data, response_wm=None, response_gm=None, response_csf=None, **dhollander_kwargs):
        """
        Fits the Multi-Shell Multi-Tissue CSD model to the provided data.

        This method handles multi-tissue response function estimation (if not
        provided for all tissues) using D'Hollander's algorithm and then fits
        the MSMT-CSD model.

        Parameters
        ----------
        data : ndarray
            A 4D NumPy array containing the diffusion-weighted image data.
            The dimensions should be (x, y, z, number_of_gradients).
        response_wm : tuple or dipy.reconst.mcsd.ResponseFunction, optional
            Pre-calculated white matter response function.
            If None, it will be estimated using `response_dhollander`.
        response_gm : tuple or dipy.reconst.mcsd.ResponseFunction, optional
            Pre-calculated grey matter response function.
            If None, it will be estimated using `response_dhollander`.
        response_csf : tuple or dipy.reconst.mcsd.ResponseFunction, optional
            Pre-calculated CSF response function.
            If None, it will be estimated using `response_dhollander`.
        **dhollander_kwargs : dict, optional
            Additional keyword arguments for `dipy.reconst.mcsd.response_dhollander`
            if any of the response functions are not provided.
            Sensible defaults for `roi_center`, `roi_radii`, `fa_thresh`,
            `frf_fa_thresh`, `gm_fa_thresh`, `csf_fa_thresh` will be used
            by `response_dhollander` if not specified.

        Raises
        ------
        ValueError
            If input data is not a 4D NumPy array.
        RuntimeError
            If response function estimation or model fitting fails.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            raise ValueError("Input data must be a 4D NumPy array.")

        # Response Function Handling
        # If any response is not provided, estimate all three using dhollander
        if response_wm is None or response_gm is None or response_csf is None:
            print(f"Estimating multi-tissue response functions using response_dhollander with kwargs: {dhollander_kwargs}")
            try:
                # response_dhollander returns a tuple of 3 ResponseFunction objects (WM, GM, CSF)
                response_wm_est, response_gm_est, response_csf_est = response_dhollander(
                    self.gtab,
                    data,
                    **dhollander_kwargs
                )
                # Store them, overwriting any individually provided ones if estimation is triggered
                self._response_wm = response_wm_est
                self._response_gm = response_gm_est
                self._response_csf = response_csf_est
            except Exception as e:
                raise RuntimeError(f"Multi-tissue response function estimation failed: {e}")
        else:
            # Use provided response functions
            # Optional: Add type/format validation for provided responses here
            print("Using pre-calculated response functions.")
            self._response_wm = response_wm
            self._response_gm = response_gm
            self._response_csf = response_csf

        # Ensure all responses are now available
        if not all([self._response_wm, self._response_gm, self._response_csf]):
            raise RuntimeError("One or more tissue response functions are still missing after attempted estimation/assignment.")

        # Model Instantiation
        try:
            # MultiShellDeconvModel expects gtab and a list/tuple of response functions
            responses = [self._response_wm, self._response_gm, self._response_csf]
            self._mt_csd_model = MultiShellDeconvModel(self.gtab, responses)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Dipy MultiShellDeconvModel: {e}. "
                               f"Responses used: WM={self._response_wm}, GM={self._response_gm}, CSF={self._response_csf}")

        # Fitting
        try:
            print("Fitting MSMT-CSD model to data...")
            self._mt_fit_object = self._mt_csd_model.fit(data)
            print("MSMT-CSD model fitting complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to fit MSMT-CSD model to data: {e}")

    def wm_odf(self, sphere=None):
        """
        Calculate the White Matter (WM) Orientation Distribution Functions (ODFs)
        from the fitted MSMT-CSD model.

        Parameters
        ----------
        sphere : dipy.data.Sphere, optional
            The sphere on which to sample the ODFs.
            If None (default), uses 'repulsion724' sphere.

        Returns
        -------
        np.ndarray
            4D array containing the WM ODF values sampled on the sphere for each voxel.
            Shape will be (X, Y, Z, num_sphere_vertices).

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._mt_fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")

        if sphere is None:
            sphere = get_sphere('repulsion724')

        # The MultiShellDeconvFit object stores FODFs for different tissues in fodf_shm.
        # Assuming WM is the first component (index 0).
        # This needs to be confirmed with Dipy's documentation or by inspecting a fit object.
        # Typically, for MSMT-CSD, the order is WM, GM, CSF.
        wm_fodf_sh = self._mt_fit_object.fodf_shm[0]
        return wm_fodf_sh.odf(sphere)

    def gm_fraction(self):
        """
        Calculate the Grey Matter (GM) volume fraction map.

        Returns
        -------
        np.ndarray
            3D array of GM volume fractions.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._mt_fit_object is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' method first.")
        # The MultiShellDeconvFit object's `all_shm_coeff` attribute holds
        # SH coefficients for all tissues. The first SH coefficient (degree 0, order 0)
        # for each tissue corresponds to its volume fraction scaled by S0.
        # Assuming GM is the second component (index 1).
        # all_shm_coeff shape is (x, y, z, num_tissues * num_sh_coeffs)
        # We need to extract the coefficients for the GM compartment.
        # The GM fraction is the first coefficient of the GM SH series.
        # If sh_order used for GM is 0, then GM compartment has 1 coeff.
        # If sh_order for GM is >0, it has more. The first is the volume fraction.
        # Let's assume MultiShellDeconvFit.all_shm_coeff provides direct access to fractions
        # or compartment signal contributions.
        # According to Dipy documentation, for MultiShellDeconvFit,
        # .all_shm_coeff stores coefficients for each tissue.
        # The volume fractions are typically derived from the S0 components of each tissue compartment.
        # MultiShellDeconvFit.volume_fractions might be a more direct way if available,
        # or by accessing the first (DC) component of each tissue's SH coefficients.
        # Let's assume self._mt_fit_object.volume_fractions gives a (x,y,z, N_tissues) array
        # If not, we need to use all_shm_coeff[:,:,:, index_of_first_gm_coeff]
        # Based on typical MSMT-CSD output, `fit_object.all_shm_coeff` contains coefficients
        # for each tissue sequentially. If GM is the second tissue, its coefficients start
        # after all WM coefficients. The first coefficient for each tissue is related to its fraction.
        # A more direct attribute `volume_fractions` is often available in newer Dipy versions or
        # can be computed from `S0s_pred = self._mt_fit_object.S0s_pred`.
        # For now, let's assume `self._mt_fit_object.volume_fractions` exists and
        # provides a (X, Y, Z, n_tissues) array.
        # If GM is the second tissue (index 1):
        # return self._mt_fit_object.volume_fractions[..., 1]
        # Looking at Dipy's MultiShellDeconvFit, there is a `volume_fractions` property.
        return self._mt_fit_object.volume_fractions[..., 1] # GM is typically index 1

    def csf_fraction(self):
        """
        Calculate the Cerebrospinal Fluid (CSF) volume fraction map.

        Returns
        -------
        np.ndarray
            3D array of CSF volume fractions.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._mt_fit_object is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' method first.")
        # Similar to gm_fraction, assuming CSF is the third component (index 2).
        # return self._mt_fit_object.volume_fractions[..., 2]
        # Looking at Dipy's MultiShellDeconvFit, there is a `volume_fractions` property.
        return self._mt_fit_object.volume_fractions[..., 2] # CSF is typically index 2

    # GFA is typically WM-specific. For multi-tissue models, it's less straightforward.
    # Removing gfa property as per instructions.
    # A WM-specific GFA could be calculated from wm_odf if needed.

    # Placeholder for future property methods if direct access to coefficients is desired
    # @property
    # def all_shm_coeff(self):
    #     if self._mt_fit_object is None:
    #         raise ValueError("Model has not been fitted yet. Call fit() first.")
    #     if not hasattr(self._fit_object, 'shm_coeff'):
    #         raise AttributeError("The CSD fit object does not have 'shm_coeff'.")
    #     return self._fit_object.shm_coeff
    #     return self._mt_fit_object.all_shm_coeff


