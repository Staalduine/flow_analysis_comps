from enum import StrEnum, auto
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
from data_structs.plate_info import plateInfo


class videoMode(StrEnum):
    BRIGHTFIELD= auto()
    FLUORESCENCE= auto()
    OTHER= auto()


class cameraSettings(BaseModel):
    model: str
    exposure_us: float
    frame_rate: float
    frame_size: tuple[int, int]
    binning: int
    gain: float
    gamma: float


class cameraPosition(BaseModel):
    x: float
    y: float
    z: float


class videoInfo(BaseModel):
    date_time: Optional[datetime] = None
    storage_path: Optional[Path] = None
    plate_info: Optional[plateInfo] = None
    run_nr: Optional[int] = None
    duration: timedelta
    frame_nr: int
    mode: videoMode
    magnification: float
    camera_settings: cameraSettings
    position: cameraPosition
