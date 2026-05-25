# src/api/main.py
import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from src.api import forecast, wells
from src.inference_pipeline.predict import ForecastService

RELOAD_SECRET = os.environ.get("RELOAD_SECRET", "changeme-reload")
FEATURES_PATH = "feature_store/data/well_features.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.forecast_service = ForecastService()
    app.state.features_df = pd.read_parquet(FEATURES_PATH)
    yield


app = FastAPI(title="Oil & Gas Forecast API", version="1.0.0", lifespan=lifespan)
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(wells.router, prefix="/api/v1")


@app.post("/admin/reload")
async def reload_model(x_reload_secret: str | None = Header(default=None)):
    """Recarga el modelo y el feature store desde disco sin reiniciar el contenedor."""
    if x_reload_secret != RELOAD_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        app.state.forecast_service = ForecastService()
        app.state.features_df = pd.read_parquet(FEATURES_PATH)
        return {"status": "ok", "message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))