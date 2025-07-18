from enum import StrEnum, auto
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from flow_analysis_comps.data_structs.plate_info import plateInfo


class videoMode(StrEnum):
    BRIGHTFIELD = auto()
    FLUORESCENCE = auto()
    NO_THRESHOLD = auto()
    OTHER = auto()


class cameraSettings(BaseModel):
    model: str
    exposure_us: float
    frame_rate: float
    frame_size: tuple[int, int]
    binning: int
    gain: float
    gamma: float
    pixel_size: float = 1.725


class cameraPosition(BaseModel):
    x: float
    y: float
    z: float


class videoInfo(BaseModel):
    date_time: Optional[datetime] = None
    storage_path: Path
    plate_info: Optional[plateInfo] = None
    run_nr: Optional[int] = None
    duration: timedelta
    frame_nr: int
    mode: videoMode
    magnification: float
    camera: cameraSettings
    position: cameraPosition
    magnification: float = 50.0
    imaging_mode: str = "brightfield"
    video_nr: int = 1
