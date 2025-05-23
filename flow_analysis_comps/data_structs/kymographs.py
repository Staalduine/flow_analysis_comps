from typing import Optional
from flow_analysis_comps.io.video import videoIO
from networkx import Graph
import numpy as np
from pydantic import BaseModel


class graphExtractConfig(BaseModel):
    edge_length_threshold: int = 200
    segmentation_threshold: float = 1.15


class kymoExtractConfig(BaseModel):
    resolution: int = 1
    step: int = 15
    target_length: int = 150
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
