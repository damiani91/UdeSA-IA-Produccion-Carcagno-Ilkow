# src/api/main.py
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from src.api import forecast, wells
from src.inference_pipeline.predict import ForecastService

RELOAD_SECRET = os.environ.get("RELOAD_SECRET", "changeme-reload")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
FEATURES_PATH = "feature_store/data/well_features.parquet"
MODEL_NAME = "well_production_model"


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


def _find_production_run_id() -> str:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod = [v for v in versions if v.current_stage == "Production"]
    if not prod:
        raise HTTPException(status_code=404, detail="No hay modelo en Production")
    return prod[0].run_id


@app.get("/admin/drift-report")
async def get_drift_report():
    """Devuelve el JSON de drift metrics del último modelo en Production."""
    import mlflow
    run_id = _find_production_run_id()
    try:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="drift", tracking_uri=MLFLOW_TRACKING_URI,
        )
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"El run {run_id} no tiene artefactos de drift (¿primer ciclo?)",
        )

    json_path = Path(local_dir) / "drift_metrics.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="drift_metrics.json no encontrado")

    with open(json_path) as f:
        return json.load(f)


@app.get("/admin/drift-report.html")
async def get_drift_report_html():
    """Devuelve el HTML completo de Evidently."""
    import mlflow
    run_id = _find_production_run_id()
    try:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="drift", tracking_uri=MLFLOW_TRACKING_URI,
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Sin reporte de drift disponible")

    html_path = Path(local_dir) / "drift_report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="drift_report.html no encontrado")

    return FileResponse(str(html_path), media_type="text/html")