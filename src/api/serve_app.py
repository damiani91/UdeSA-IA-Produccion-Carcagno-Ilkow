"""Deployment de Ray Serve que envuelve la app FastAPI existente.

Decisión (plan §5.1):
- Mantenemos la spec OpenAPI intacta: Ray Serve es solo el front HTTP y proxy.
- `num_replicas=2` para mostrar paralelismo real (ajustable por env).
- El estado pesado (modelo MLflow + parquet) se inicializa en `__init__`,
  por lo que cada réplica carga su propia copia (réplicas independientes).

Uso:
    serve run src.api.serve_app:forecast_app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

import pandas as pd
from fastapi import FastAPI
from ray import serve

from src.api import forecast, monitoring, wells
from src.inference_pipeline.predict import ForecastService

# Construimos una app FastAPI fresca (sin el `lifespan` del main, porque ahora la
# inicialización vive en el __init__ del deployment, ver plan §5.2 paso 2).
fastapi_app = FastAPI(title="Oil & Gas Forecast API", version="1.0.0")
fastapi_app.include_router(forecast.router, prefix="/api/v1")
fastapi_app.include_router(wells.router, prefix="/api/v1")
fastapi_app.include_router(monitoring.router, prefix="/api/v1")


NUM_REPLICAS = int(os.getenv("SERVE_NUM_REPLICAS", "2"))


@serve.deployment(
    num_replicas=NUM_REPLICAS,
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(fastapi_app)
class ForecastDeployment:
    """Cada réplica carga su propio modelo Production + features offline."""

    def __init__(self) -> None:
        fastapi_app.state.forecast_service = ForecastService()
        fastapi_app.state.features_df = pd.read_parquet(
            "feature_store/data/well_features.parquet"
        )


# Handle bindable por `serve run src.api.serve_app:forecast_app`.
forecast_app = ForecastDeployment.bind()
