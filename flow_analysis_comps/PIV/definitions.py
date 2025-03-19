from enum import StrEnum
from pydantic import BaseModel, PositiveInt
from pathlib import Path

from flow_analysis_comps.data_structs.video_info import videoMode


class segmentMode(StrEnum):
    NONE : str =  "NONE"
    BRIGHT: str = "BRIGHTFIELD"
    FLUO: str = "FLUO"


class preProcessMode(StrEnum):
    HARM_THRESH: str = "HARMONIC_MEAN_THRESHOLD"


class PIV_params(BaseModel):
    video_path: Path
    segment_mode: videoMode
    fps: float
    window_size: PositiveInt
    search_size: PositiveInt
    overlap_size: PositiveInt
    stn_threshold: float
    px_per_mm: float
