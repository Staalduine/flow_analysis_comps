from flow_analysis_comps.io.video import videoIO
from networkx import Graph
import numpy as np
import pandas as pd
from pydantic import BaseModel


class graphExtractConfig(BaseModel):
    edge_length_threshold: int = 200
    segmentation_threshold: float = 1.15


class kymoExtractConfig(BaseModel):
    resolution: int = 1  # Pixel distance between sampled points
    step: int = 15
    target_length: int = 70  # Pixel length of perpendicular lines
    bounds: tuple[float, float] = (0.0, 1.0)
    graph_extraction: graphExtractConfig = graphExtractConfig()


class kymoDeltas(BaseModel):
    delta_x: float
    delta_t: float


class graphOutput(BaseModel):
    graph: Graph
    positions: dict

    class Config:
        arbitrary_types_allowed = True


class VideoGraphEdge(BaseModel):
    name: str
    edge: tuple[int, int]
    pixel_list: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class VideoGraphExtraction(BaseModel):
    io: videoIO
    edges: list[VideoGraphEdge]

    class Config:
        arbitrary_types_allowed = True


class KymoCoordinates(BaseModel):
    segment_coords: np.ndarray
    perp_lines: np.ndarray
    edge_info: VideoGraphEdge

    class Config:
        arbitrary_types_allowed = True


class kymoOutputs(BaseModel):
    deltas: kymoDeltas
    name: str
    kymograph: np.ndarray
    kymo_left: np.ndarray
    kymo_right: np.ndarray
    kymo_no_static: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class GSTSpeedOutputs(BaseModel):
    deltas: kymoDeltas
    name: str
    speed_left: np.ndarray
    speed_right: np.ndarray
    speed_mean_time_series: pd.DataFrame
    speed_mean_overall: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


class GST_params(BaseModel):
    window_start: int = 3
    window_amount: int = 15
    coherency_threshold: float = 0.95
    coherency_threshold_falloff: float = 0.05


class GSTConfig(BaseModel):
    gst_params: GST_params = GST_params()
    speed_limit: float = 10.0
