import pandas as pd
from pydantic import BaseModel
from flow_analysis_comps.data_structs.array_types import image_float
from flow_analysis_comps.data_structs.video_metadata_structs import videoDeltas

class GST_params(BaseModel):
    window_start: int = 3
    window_amount: int = 15
    coherency_threshold: float = 0.95
    coherency_threshold_falloff: float = 0.05

class GSTSpeedOutputs(BaseModel):
    deltas: videoDeltas
    name: str
    speed_left: image_float
    speed_right: image_float
    speed_mean_time_series: pd.DataFrame
    speed_mean_overall: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
