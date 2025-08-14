import numpy as np
from pydantic import BaseModel, computed_field
from numpydantic import NDArray, Shape
from flow_analysis_comps.data_structs.video_metadata_structs import videoInfo
from flow_analysis_comps.processing.GSTSpeedExtract.classic_image_util import speed_from_orientation_image
data_stack = NDArray[Shape['* x, * y, * z'], float]

class angle_filter_values(BaseModel):
    magnitude: float = 0.1
    first_derivative: float = 0.1
    second_derivative: float = 5.0


class multiOriParams(BaseModel):
    space_frequency_center: float  # Central frequency band, selects for object sizes
    orientation_accuracy: float = 5.0  # Width of orientation band
    sampling_factor: int = 1
    padding: int = 0
    x_spacing: float = 1.0
    y_spacing: float = 1.0
    z_spacing: float = 1.0
    multires_filter_params: angle_filter_values = angle_filter_values()

    @computed_field
    @property
    def space_frequency_width(self) -> float:
        return 1 / np.sqrt(2) * self.space_frequency_center

    @computed_field
    @property
    def nr_of_samples(self) -> int:
        return int(2 * self.sampling_factor * np.ceil(self.orientation_accuracy) + 1)

class multiOriOutput(BaseModel):
    metadata: videoInfo
    angles_maxima: data_stack
    angles_minima: data_stack
    values_maxima: data_stack
    values_minima: data_stack

    @computed_field
    @property
    def speeds_maxima(self) -> data_stack:
        return speed_from_orientation_image(self.angles_maxima, self.metadata.deltas)

    @property
    def speeds_minima(self) -> data_stack:
        return speed_from_orientation_image(self.angles_minima, self.metadata.deltas)
