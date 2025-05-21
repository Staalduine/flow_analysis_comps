from pydantic import BaseModel


class GST_params(BaseModel):
    window_start: int = 3
    window_amount: int = 15
    coherency_threshold: float = 0.95
    coherency_threshold_falloff: float = 0.05


class videoDeltas(BaseModel):
    delta_x: float
    delta_t: float