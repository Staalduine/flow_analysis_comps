import numpy as np
from pydantic import BaseModel, computed_field


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
