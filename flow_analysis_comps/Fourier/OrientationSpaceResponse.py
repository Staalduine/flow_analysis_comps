from enum import StrEnum, auto
from typing import Optional
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.axes
import numpy as np
import colorcet  # noqa: F401
from scipy import fftpack
import skimage as ski
from flow_analysis_comps.util.coord_space_util import cart2pol
import numpy.typing as npt


class ThresholdMethods(StrEnum):
    OTSU = auto()
    ROSIN = auto()


class OrientationSpaceResponse:
    """
    Class that holds the response of the orientation filters.
    From here, the array can be accessed, but it can also be plotted.
    """

    def __init__(
        self,
        response_array: np.ndarray,
        # angles: np.ndarray,
        mask: npt.NDArray[np.bool_] = None,
    ):
        """Create object

        Args:
            response_array (np.ndarray): Input array of image which is filtered.
            angles (np.ndarray): Array of angles along which the filter was taken
            mask (npt.NDArray[np.bool_], optional): Mask which is used for nlms. Defaults to None.
        """
        self.response_array = response_array
        self.a_hat: np.ndarray = fftpack.fft(self.response_array.real, axis=2)
        self.n = self.response_array.shape[-1]
        self.angles = np.arange(self.response_array.shape[-1]) / self.response_array.shape[-1] * np.pi
        self.mask = mask

    def get_resp(self):
        return self.response_array

    def set_resp(self, response: np.ndarray):
        self.response_array = response

    @property
    def mean_response(self) -> np.ndarray:
        mean_response = self.a_hat[:, :, 0] / self.a_hat.shape[-1]
        return mean_response

    response = property(get_resp, set_resp)

    def threshold_mean(self, method: Optional[ThresholdMethods] = None):
        match method:
            case "otsu":
                threshold_val = ski.filters.threshold_otsu(self.mean_response.real)
            case "rosin":
                threshold_val = ski.filters.threshold_triangle(self.mean_response.real)
            case _:
                threshold_val = 0

        return self.mean_response.real > threshold_val

    def nlms_mask(self, dilation_rad: int = 3, fill_holes: bool = False, thresh_method: Optional[ThresholdMethods] = None):
        mean_thresh_mask: npt.NDArray[np.bool_] = self.threshold_mean(thresh_method)
        if self.mask:
            mean_thresh_mask = np.logical_and(mean_thresh_mask, self.mask)
        mean_thresh_mask_dil = ski.morphology.binary_dilation(
            mean_thresh_mask, ski.morphology.disk(dilation_rad)
        )
        if fill_holes:
            mean_thresh_mask_dil = ski.morphology.remove_small_holes(
                mean_thresh_mask_dil
            )
        return mean_thresh_mask_dil

    ## Plotting functions
    def plot_mean_response(self):
        largest_value = max(
            abs(self.response_array.min()), abs(self.response_array.max())
        )
        meanresponse = self.mean_response
        fig, ax = plt.subplots(dpi=200)
        ax.imshow(
            meanresponse.__abs__(),
            cmap="cet_CET_D1A",
            vmin=-largest_value,
            vmax=largest_value,
        )
        return fig

    def plot_lineof_point(self, coord):
        """
        Plot the filter response of a point in space with orientation units

        Args:
            coord (tuple[int, int, Optional[int]]): input coordinate, can be 2d, or 3d (not implemented yets)
        """
        fig, ax = plt.subplots(1, 2)

        match len(coord):
            case 2:
                # Assume img comes in as array (so y,x or z, x axes), so flip the coords
                ax[0].plot(self.angles, self.response_array[coord[1], coord[0], :].real)
                ax[0].set_xlim(0, np.pi)
                ax[0].set_xticks(np.linspace(0, np.pi, 3, endpoint=True))
                ax[0].set_xticklabels(["0", r"$\pi$/2", r"$\pi$"])
                ax[1].plot(self.a_hat[coord[1], coord[0], :].real)

    def visualize_point_response(self, coord, mesh_size=128):
        match len(coord):
            case 2:
                line_plot = self.response_array[coord[1], coord[0], :].real
                xs, ys = np.meshgrid(
                    np.linspace(-1, 1, mesh_size, endpoint=True),
                    np.linspace(-1, 1, mesh_size, endpoint=True),
                )
                rs, thetas_2 = cart2pol(xs, ys)
                theta_response_2 = np.interp(
                    (thetas_2 + np.pi / 2) % np.pi, self.angles, line_plot, period=np.pi
                )
                theta_response_2 *= rs < 1
                theta_response_2 *= rs > 0.5

                fig, ax = plt.subplots()
                ax.imshow(theta_response_2)

    def visualize_orientation_wheel(self, mesh_size = 128, cmap = 'cet_CET_C3_r', ax: Optional[matplotlib.axes.Axes] = None):
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, mesh_size, endpoint=True),
            np.linspace(-1, 1, mesh_size, endpoint=True),
        )
        rs, thetas_2 = cart2pol(xs, ys)

        thetas_2 *= rs < 1
        thetas_2 *= rs > 0.5
        thetas_2 = (thetas_2 + np.pi/2) % np.pi
        if ax is None:
            fig, ax = plt.subplots()
        ax:matplotlib.axes.Axes
        ax.imshow(thetas_2, cmap= cmap)
        ax.set_xticks([])
        ax.set_yticks([])