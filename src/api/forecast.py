# src/api/routes/forecast.py
from fastapi import APIRouter, Query
from src.api import _state
from src.api.schemas import ForecastResponse

router = APIRouter()

@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(id_well: str = Query(..., description="str(idpozo)"),
                 date_start: str = Query(..., description="YYYY-MM-DD"),
                 date_end: str = Query(..., description="YYYY-MM-DD")):
    data = _state.forecast_service.predict(id_well, date_start, date_end)
    return ForecastResponse(id_well=id_well, data=data)