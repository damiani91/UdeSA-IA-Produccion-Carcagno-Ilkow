# src/api/main.py
import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from ray import serve

from src.api import forecast, wells
from src.inference_pipeline.predict import ForecastService

RELOAD_SECRET = os.environ.get("RELOAD_SECRET", "changeme-reload")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
FEATURES_PATH = "feature_store/data/well_features.parquet"
MODEL_NAME = "well_production_model"


def _find_production_run_id() -> str:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient(MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod = [v for v in versions if v.current_stage == "Production"]
    if not prod:
        raise HTTPException(status_code=404, detail="No hay modelo en Production")
    return prod[0].run_id


def _build_fastapi() -> FastAPI:
    """Build the FastAPI app with all routers and endpoints.

    Called once per replica at import time. Ray Serve does not fire FastAPI's
    lifespan, so state initialization lives in ForecastDeployment.__init__.
    """
    _app = FastAPI(title="Oil & Gas Forecast API", version="1.0.0")
    _app.include_router(forecast.router, prefix="/api/v1")
    _app.include_router(wells.router, prefix="/api/v1")

    @_app.post("/admin/reload")
    async def reload_model(x_reload_secret: str | None = Header(default=None)):
        """Recarga el modelo en TODAS las réplicas vía Ray Serve rolling redeploy."""
        if x_reload_secret != RELOAD_SECRET:
            raise HTTPException(status_code=403, detail="Forbidden")
        try:
            # serve.run() triggers a rolling restart of all replicas with fresh state.
            # Each new replica will call ForecastDeployment.__init__ and load the
            # latest Production model from MLflow.
            serve.run(
                ForecastDeployment.bind(),
                name="forecast-api",
                route_prefix="/",
            )
            return {"status": "ok", "message": "Rolling model reload triggered across all replicas"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @_app.get("/admin/drift-report")
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

    @_app.get("/admin/drift-report.html")
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

    return _app


# Module-level app — each Ray Serve replica imports this module independently,
# so each gets its own app instance and its own app.state namespace.
app = _build_fastapi()


@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(app)
class ForecastDeployment:
    def __init__(self):
        # Runs once per replica. Ray Serve does not call FastAPI lifespan,
        # so we initialize shared state here instead.
        app.state.forecast_service = ForecastService()
        app.state.features_df = pd.read_parquet(FEATURES_PATH)
