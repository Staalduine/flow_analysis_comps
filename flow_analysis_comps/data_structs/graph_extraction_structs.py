from networkx import Graph
from pydantic import BaseModel
from flow_analysis_comps.data_structs.kymograph_structs import VideoGraphEdge
from flow_analysis_comps.data_structs.video_metadata_structs import videoInfo

"""
Data structures for configuring video processing functions. Handles default values. 
Important rule: ONLY use basic types, no function imports.
"""


class graphExtractConfig(BaseModel):
    edge_length_threshold: int = 200
    segmentation_threshold: float = 1.15

class VideoGraphExtraction(BaseModel):
    metadata: videoInfo
    edges: list[VideoGraphEdge]


class graphOutput(BaseModel):
    graph: Graph
    positions: dict

    class Config:
        arbitrary_types_allowed = True
