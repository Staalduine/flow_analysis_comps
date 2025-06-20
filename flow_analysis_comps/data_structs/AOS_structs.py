from typing import Optional
import numpy as np
from pydantic import BaseModel, model_validator


class OSFilterParams(BaseModel):
    space_frequency_center: float  # Central frequency band, selects for object sizes
    space_frequency_width: Optional[float] = (
        None  # Width of frequency torus, selects the range of sizes around freq_central
    )
    orientation_accuracy: float = 5.0  # Width of orientation band
    sampling_factor: int = 1
    nr_of_samples: Optional[int] = None  # Number of filter samplings to take.
    x_spacing: float = 1.0
    y_spacing: float = 1.0
    z_spacing: float = 1.0

    @model_validator(mode="after")
    def set_defaults(self):
        if self.space_frequency_width is None:
            self.space_frequency_width = (
                1 / np.sqrt(2) * self.space_frequency_center
            )
        if self.nr_of_samples is None:
            self.nr_of_samples = int(
                2 * self.sampling_factor * np.ceil(self.orientation_accuracy)
                + 1
            )
        return self