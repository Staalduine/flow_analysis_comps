from enum import StrEnum, auto
from typing import Optional
import numpy as np
import colorcet  # noqa: F401
from scipy import fftpack
import skimage as ski
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
    ):
        """Create object

        Args:
            response_array (np.ndarray): Input array of image which is filtered.
            angles (np.ndarray): Array of angles along which the filter was taken
            mask (npt.NDArray[np.bool_], optional): Mask which is used for nlms. Defaults to None.
        """
        self.response_stack = response_array
        self.response_stack_fft: np.ndarray = fftpack.fft(self.response_stack.real, axis=2)
        self.number_of_angles = self.response_stack.shape[-1]
        self.range_of_angles = np.arange(self.response_stack.shape[-1]) / self.response_stack.shape[-1] * np.pi

    def get_resp(self):
        return self.response_stack

    def set_resp(self, response: np.ndarray):
        self.response_stack = response

    @property
    def mean_response(self) -> np.ndarray:
        mean_response = self.response_stack_fft[:, :, 0] / self.response_stack_fft.shape[-1]
        return mean_response

    response = property(get_resp, set_resp)

    def threshold_mean(self, method: Optional[ThresholdMethods] = None):
        match method:
            case "otsu":
                threshold_val = ski.filters.threshold_otsu(self.mean_response.real)
            case "rosin":
                threshold_val = ski.filters.threshold_triangle(self.mean_response.real)
            case None:
                threshold_val = 0
            case _:
                print("Unknown threshold method!")
                threshold_val = 0

        return self.mean_response.real > threshold_val

    def nlms_mask(self, dilation_rad: int = 3, fill_holes: bool = False, thresh_method: Optional[ThresholdMethods] = None):
        mean_thresh_mask: npt.NDArray[np.bool_] = self.threshold_mean(thresh_method)
        
        mean_thresh_mask_dil = ski.morphology.binary_dilation(
            mean_thresh_mask, ski.morphology.disk(dilation_rad)
        )
        if fill_holes:
            mean_thresh_mask_dil = ski.morphology.remove_small_holes(
                mean_thresh_mask_dil
            )
        return mean_thresh_mask_dil
