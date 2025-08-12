from enum import StrEnum, auto
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, computed_field
from datetime import datetime, timedelta
from flow_analysis_comps.data_structs.plate_info_structs import plateInfo


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
    gain: float = 0.0
    gamma: float = 1.0
    pixel_size_um: float = 1.725


class cameraPosition(BaseModel):
    x: float
    y: float
    z: float


class videoDeltas(BaseModel):
    delta_t: float
    delta_x: float


class videoInfo(BaseModel):
    date_time: Optional[datetime] = None
    root_path: Path
    plate_info: Optional[plateInfo] = None
    run_nr: Optional[int] = None
    duration: timedelta
    frame_nr: int
    mode: videoMode = videoMode.BRIGHTFIELD
    camera: cameraSettings
    position: Optional[cameraPosition] = None
    magnification: float = 50.0

    @computed_field
    @property
    def deltas(self) -> videoDeltas:
        delta_t = 1 / self.camera.frame_rate
        delta_x = (
            self.camera.pixel_size_um * 2 / self.magnification * self.camera.binning
        )
        return videoDeltas(delta_t=delta_t, delta_x=delta_x)
