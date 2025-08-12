from flow_analysis_comps.data_structs.video_info import videoDeltas, videoInfo
from networkx import Graph
import numpy as np
import pandas as pd
from pydantic import BaseModel, computed_field
from numpydantic import NDArray, Shape

from flow_analysis_comps.processing.GSTSpeedExtract.classic_image_util import (
    filter_kymo_right,
)


class graphOutput(BaseModel):
    graph: Graph
    positions: dict

    class Config:
        arbitrary_types_allowed = True


class VideoGraphEdge(BaseModel):
    name: str
    edge: tuple[int, int]
    pixel_list: NDArray[Shape["* x, 2 y"], int]  # type: ignore # Coordinates of the edge in the video  # noqa: F722


class VideoGraphExtraction(BaseModel):
    io: videoInfo
    edges: list[VideoGraphEdge]


class KymoCoordinates(BaseModel):
    segment_coords: np.ndarray
    perp_lines: np.ndarray
    edge_info: VideoGraphEdge

    class Config:
        arbitrary_types_allowed = True


class kymoOutputs(BaseModel):
    deltas: videoDeltas
    name: str
    kymograph: np.ndarray

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def kymo_left(self) -> np.ndarray:
        return filter_kymo_right(self.kymograph)

    @computed_field
    @property
    def kymo_right(self) -> np.ndarray:
        return np.flip(filter_kymo_right(np.flip(self.kymograph, axis=1)), axis=1)

    @computed_field
    @property
    def kymo_no_static(self) -> np.ndarray:
        return self.kymo_left + self.kymo_right


class GSTSpeedOutputs(BaseModel):
    deltas: videoDeltas
    name: str
    speed_left: np.ndarray
    speed_right: np.ndarray
    speed_mean_time_series: pd.DataFrame
    speed_mean_overall: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
