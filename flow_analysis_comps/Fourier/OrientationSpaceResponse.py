from enum import StrEnum, auto
from typing import Optional
import cv2
from matplotlib import pyplot as plt
import numpy as np
import colorcet
import skimage as ski

class ThresholdMethods(StrEnum):
    OTSU = auto()
    ROSIN = auto()

class OrientationSpaceResponse:
    """
    Class that holds the response of the orientation filters.
    From here, the array can be accessed, but it can also be plotted.
    """

    def __init__(self, response_array: np.ndarray):
        self.response_array = response_array
        self.n = self.response_array.shape[-1]

    def get_resp(self):
        return self.response_array

    def set_resp(self, response: np.ndarray):
        self.response_array = response
    
    @property
    def mean_response(self)->np.ndarray:
        return np.mean(self.response_array, axis=2)

    response = property(get_resp, set_resp)

    ## Plotting functions
    def plot_mean_response(self):
        largest_value = max(abs(self.response_array.min()), abs(self.response_array.max()))
        meanresponse = self.mean_response
        fig, ax = plt.subplots(dpi=200)
        ax.imshow(meanresponse.real, cmap="cet_CET_D1A", vmin=-largest_value, vmax = largest_value)
        return fig
    
    def threshold_mean(self, method:Optional[ThresholdMethods]=None):
        match ThresholdMethods:
            case "otsu":
                threshold_val = ski.filters.threshold_otsu(self.mean_response.real)
            case "rosin":
                threshold_val = ski.filters.threshold_triangle(self.mean_response.real)
            case _:
                threshold_val = ski.filters.threshold_triangle(self.mean_response.real)
        
        return self.mean_response.real > threshold_val
        