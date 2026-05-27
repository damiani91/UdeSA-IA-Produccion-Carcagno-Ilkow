# src/api/entrypoint.py
"""Container entrypoint: initializes a local Ray cluster and starts Ray Serve.

Replaces the plain `uvicorn` CMD. Ray Serve handles HTTP on port 8000 and
load-balances across num_replicas replicas of ForecastDeployment.
Ray dashboard is available on port 8265.
"""

import os
import ray
from ray import serve

from src.api.main import ForecastDeployment

NUM_CPUS = int(os.environ.get("RAY_NUM_CPUS", "4"))


def main() -> None:
    ray.init(
        num_cpus=NUM_CPUS,
        dashboard_host="0.0.0.0",
        dashboard_port=8265,
        ignore_reinit_error=True,
    )
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    # blocking=True keeps the process alive until SIGTERM/SIGINT
    serve.run(
        ForecastDeployment.bind(),
        name="forecast-api",
        route_prefix="/",
        blocking=True,
    )


if __name__ == "__main__":
    main()
