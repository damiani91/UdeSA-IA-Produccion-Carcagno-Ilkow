"""Model Registry — ciclo de vida del modelo.

Stages: None → Staging → Production → Archived
"""

import mlflow
from mlflow.tracking import MlflowClient


def promote_model_to_production(
    model_name: str = "well_production_model",
    run_id: str | None = None,
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    if run_id:
        versions = client.search_model_versions(f"name='{model_name}'")
        target = [v for v in versions if v.run_id == run_id]
        if not target:
            raise ValueError(f"No version for run_id={run_id}")
        version = target[0].version
    else:
        versions = client.get_latest_versions(model_name)
        version = versions[-1].version

    # Transicionar a Production, archivando versiones anteriores (clase 2: rollback)
    client.transition_model_version_stage(
        name=model_name, version=version,
        stage="Production", archive_existing_versions=True,
    )
    print(f"Modelo {model_name} v{version} → Production")