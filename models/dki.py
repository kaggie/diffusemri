import numpy as np
from dipy.reconst.dki import DiffusionKurtosisModel as DipyDkiModel
from dipy.core.gradients import GradientTable

class DkiModel:
    """
    A wrapper class for Dipy's DiffusionKurtosisModel.

    This class provides a simplified interface to fit the DKI model
    to diffusion MRI data.
    """
    def __init__(self, gtab):
        """
        Initializes the DkiModel.

        Parameters
        ----------
        gtab : GradientTable
            A Dipy GradientTable object containing gradient information.
        """
        if not isinstance(gtab, GradientTable):
            raise ValueError("gtab must be an instance of Dipy's GradientTable.")
        self.gtab = gtab
        self._fit_object = None

    def fit(self, data):
        """
        Fits the Diffusion Kurtosis Imaging (DKI) model to the provided data.

        Parameters
        ----------
        data : ndarray
            A 4D NumPy array containing the diffusion-weighted image data.
            The dimensions should be (x, y, z, number_of_gradients).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            raise ValueError("Input data must be a 4D NumPy array.")

        dipy_model = DipyDkiModel(self.gtab)
        self._fit_object = dipy_model.fit(data)
        # The DiffusionKurtosisFit object is now stored in self._fit_object.

    @property
    def fa(self):
        """Fractional Anisotropy (FA) derived from the diffusion tensor part of the DKI model.

        Returns
        -------
        np.ndarray
            3D array of FA values.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.fa

    @property
    def md(self):
        """Mean Diffusivity (MD) derived from the diffusion tensor part of the DKI model.

        Returns
        -------
        np.ndarray
            3D array of MD values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.md

    @property
    def mk(self):
        """Mean Kurtosis (MK).

        Returns
        -------
        np.ndarray
            3D array of MK values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.mk()

    @property
    def ak(self):
        """Axial Kurtosis (AK).

        Returns
        -------
        np.ndarray
            3D array of AK values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.ak()

    @property
    def rk(self):
        """Radial Kurtosis (RK).

        Returns
        -------
        np.ndarray
            3D array of RK values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.rk()

    @property
    def ka(self):
        """Kurtosis Anisotropy (KA).

        Returns
        -------
        np.ndarray
            3D array of KA values.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._fit_object is None:
            raise ValueError("Model has not been fitted yet. Call the 'fit' method first.")
        return self._fit_object.ka()
