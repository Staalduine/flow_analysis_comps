from flow_analysis_comps.data_structs.graph_extraction_structs import VideoGraphEdge
from flow_analysis_comps.data_structs.video_metadata_structs import (
    videoDeltas,
)
import numpy as np
from pydantic import BaseModel, computed_field

from flow_analysis_comps.processing.GSTSpeedExtract.classic_image_util import (
    filter_kymo_right,
)
from flow_analysis_comps.data_structs.array_types import image_float


class KymoCoordinates(BaseModel):
    segment_coords: np.ndarray
    perp_lines: np.ndarray
    edge_info: VideoGraphEdge

    class Config:
        arbitrary_types_allowed = True


class kymoExtractConfig(BaseModel):
    resolution: int = 1  # Pixel distance between sampled points
    step: int = 15
    target_length: int = 70  # Pixel length of perpendicular lines


class kymoOutputs(BaseModel):
    deltas: videoDeltas
    name: str
    kymograph: image_float  # type: ignore # The kymograph image

    @computed_field
    @property
    def kymo_left(self) -> image_float:
        return filter_kymo_right(self.kymograph)  # type: ignore

    @computed_field
    @property
    def kymo_right(self) -> image_float:
        return np.flip(filter_kymo_right(np.flip(self.kymograph, axis=1)), axis=1)  # type: ignore

    @computed_field
    @property
    def kymo_no_static(self) -> image_float:
        return self.kymo_left + self.kymo_right  # type: ignore
