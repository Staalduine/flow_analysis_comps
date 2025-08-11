from pydantic import BaseModel

"""
Data structures for configuring video processing functions. Handles default values. 
Important rule: ONLY use basic types, no function imports.
"""

class graphExtractConfig(BaseModel):
    edge_length_threshold: int = 200
    segmentation_threshold: float = 1.15


class kymoExtractConfig(BaseModel):
    resolution: int = 1  # Pixel distance between sampled points
    step: int = 15
    target_length: int = 70  # Pixel length of perpendicular lines
    bounds: tuple[float, float] = (0.0, 1.0)
    graph_extraction: graphExtractConfig = graphExtractConfig()


class GST_params(BaseModel):
    window_start: int = 3
    window_amount: int = 15
    coherency_threshold: float = 0.95
    coherency_threshold_falloff: float = 0.05


class GSTConfig(BaseModel):
    gst_params: GST_params = GST_params()
    speed_limit: float = 10.0
