from typing import Optional
import numpy as np
from pydantic import BaseModel, computed_field, model_validator


class angle_filter_values(BaseModel):
    magnitude: float = 0.1
    first_derivative: float = 0.1
    second_derivative: float = 5.0


class OSFilterParams(BaseModel):
    space_frequency_center: float  # Central frequency band, selects for object sizes
    # space_frequency_width: Optional[float] = (
    #     None  # Width of frequency torus, selects the range of sizes around freq_central
    # )
    orientation_accuracy: float = 5.0  # Width of orientation band
    sampling_factor: int = 1
    # nr_of_samples: Optional[int] = None  # Number of filter samplings to take.
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

    # @model_validator(mode="after")
    # def set_defaults(self):
    #     if self.space_frequency_width is None:

    #     if self.nr_of_samples is None:
    #         self.nr_of_samples = int(
    #             2 * self.sampling_factor * np.ceil(self.orientation_accuracy)
    #             + 1
    #         )
    #     return self
