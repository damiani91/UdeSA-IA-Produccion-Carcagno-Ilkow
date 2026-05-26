# src/api/_state.py
"""Process-level state singleton for Ray Serve replicas.

Each Ray Serve replica is a separate OS process with its own import of this
module. ForecastDeployment.__init__ populates these variables once per replica
startup; route handlers read from them directly, avoiding any dependency on
FastAPI app.state (whose object identity is unstable under Ray Serve ingress).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from src.inference_pipeline.predict import ForecastService

forecast_service: "ForecastService | None" = None
features_df: "pd.DataFrame | None" = None
