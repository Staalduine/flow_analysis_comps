from networkx import Graph
from pydantic import BaseModel


class graphExtractConfig(BaseModel):
    edge_length_threshold: int = 200
    segmentation_threshold: float = 1.15


class kymoExtractProperties(BaseModel):
    resolution: int = 1
    step: int = 15
    target_length: int = 70
    bounds: tuple[float, float] = (0.0, 1.0)
    graph_extraction: graphExtractConfig = graphExtractConfig()


class graphOutput(BaseModel):
    graph: Graph
    positions: dict


class edgeOutput(BaseModel):
    name: str
    edge: tuple[int, int]
    pixel_list: list[tuple[int, int]]
