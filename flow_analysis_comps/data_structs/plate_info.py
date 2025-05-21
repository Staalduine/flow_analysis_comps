from enum import StrEnum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from datetime import datetime, date


class rootTypes(StrEnum):
    CARROT = "Carrot"
    TOBY = "Toby"


class strainTypes(StrEnum):
    C2 = "C2"
    A5 = "A5"


class treatmentTypes(StrEnum):
    STANDARD = "001P100N100C"


class plateInfo(BaseModel):
    plate_nr: str
    root: rootTypes
    strain: strainTypes
    treatment: treatmentTypes
    crossing_date: Optional[date]
