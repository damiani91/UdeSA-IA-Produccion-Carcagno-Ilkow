# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api import forecast, monitoring, wells
from src.inference_pipeline.predict import ForecastService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.forecast_service = ForecastService()
    import pandas as pd
    app.state.features_df = pd.read_parquet("feature_store/data/well_features.parquet")
    yield

app = FastAPI(title="Oil & Gas Forecast API", version="1.0.0", lifespan=lifespan)
app.include_router(forecast.router, prefix="/api/v1")
app.include_router(wells.router, prefix="/api/v1")
app.include_router(monitoring.router, prefix="/api/v1")