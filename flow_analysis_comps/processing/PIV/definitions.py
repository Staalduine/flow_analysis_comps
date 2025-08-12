from enum import StrEnum
from pydantic import BaseModel, PositiveInt, field_validator
from pathlib import Path

from flow_analysis_comps.data_structs.video_metadata_structs import videoMode


class segmentMode(StrEnum):
    NONE =  "NONE"
    BRIGHT = "BRIGHTFIELD"
    FLUO = "FLUO"


class preProcessMode(StrEnum):
    HARM_THRESH = "HARMONIC_MEAN_THRESHOLD"


class PIV_params(BaseModel):
    root_path: Path
    segment_mode: videoMode
    window_size_start: PositiveInt
    number_of_passes: PositiveInt
    stn_threshold: float
    max_speed_px_per_frame: float
    
    @field_validator("window_size_start")
    @classmethod
    def check_power_of_two(cls, value):
        if value <= 0 or (value & (value - 1)) != 0:
            raise ValueError("The number must be a power of 2.")
        return value
