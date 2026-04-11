# src/api/routes/forecast.py
from fastapi import APIRouter, Query, Request
from src.api.schemas import ForecastResponse

router = APIRouter()

@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(request: Request,
                 id_well: str = Query(..., description="str(idpozo)"),
                 date_start: str = Query(..., description="YYYY-MM-DD"),
                 date_end: str = Query(..., description="YYYY-MM-DD")):
    data = request.app.state.forecast_service.predict(id_well, date_start, date_end)
    return ForecastResponse(id_well=id_well, data=data)