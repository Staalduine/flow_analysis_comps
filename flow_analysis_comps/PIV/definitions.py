from enum import StrEnum
from pydantic import BaseModel, PositiveInt
from pathlib import Path


class segmentMode(StrEnum):
    BRIGHT: str = "BRIGHTFIELD"
    FLUO: str = "FLUO"


class preProcessMode(StrEnum):
    HARM_THRESH: str = "HARMONIC_MEAN_THRESHOLD"


class PIV_params(BaseModel):
    video_path: Path
    segment_mode: segmentMode
    fps: float
    window_size: PositiveInt
    search_size: PositiveInt
    overlap_size: PositiveInt
    stn_threshold: float
    px_per_mm: float
