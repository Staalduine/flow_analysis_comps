from networkx import Graph
from pydantic import BaseModel
from numpydantic import NDArray, Shape
from flow_analysis_comps.data_structs.video_metadata_structs import videoInfo

"""
Data structures for configuring video processing functions. Handles default values. 
Important rule: ONLY use basic types, no function imports.
"""


class graphExtractConfig(BaseModel):
    edge_length_threshold: int = 200
    segmentation_threshold: float = 1.15


class VideoGraphEdge(BaseModel):
    name: str
    edge: tuple[int, int]
    pixel_list: NDArray[Shape["* x, 2 y"], int]  # type: ignore # Coordinates of the edge in the video  # noqa: F722


class VideoGraphExtraction(BaseModel):
    metadata: videoInfo
    edges: list[VideoGraphEdge]


class graphOutput(BaseModel):
    graph: Graph
    positions: dict

    class Config:
        arbitrary_types_allowed = True
