# src/api/schemas.py
from pydantic import BaseModel

class ForecastPoint(BaseModel):
    date: str
    prod: float

class ForecastResponse(BaseModel):
    id_well: str
    data: list[ForecastPoint]

class WellInfo(BaseModel):
    id_well: str