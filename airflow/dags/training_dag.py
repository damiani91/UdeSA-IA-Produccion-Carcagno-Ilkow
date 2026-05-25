"""DAG mensual: populate feature store -> train -> promote -> drift report.

Diseño (ver PLAN_ENTREGA_FINAL.md sección 3.1):
- LocalExecutor + PythonOperator (las funciones se importan directamente desde
  src.* y scripts.*; la imagen de Airflow ya tiene las dependencias del proyecto).
- run_id del entrenamiento se pasa por XCom al task de promoción.
- schedule_interval='@monthly' porque el dataset público se actualiza mensualmente.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Asegurar que se pueda importar `src.*`, `scripts.*` y `feature_store.*`.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

default_args = {
    "owner": "ia-produccion",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


def _training_date_from_context(ds: str) -> str:
    """ds = data_interval_end en formato YYYY-MM-DD. Lo usamos como training_date."""
    return ds


def _prepare_offline_store(ds: str, **_):
    from scripts.populate_feature_store import prepare_offline_store
    prepare_offline_store(up_to_date=_training_date_from_context(ds))


def _apply_feast(**_):
    # cwd dentro del repo de feast para que `feast apply` encuentre el yaml.
    from scripts.populate_feature_store import apply_feast
    apply_feast()


def _populate_online_store(**_):
    from scripts.populate_feature_store import populate_online_store
    populate_online_store()


def _train_model(ds: str, **_) -> str:
    from src.training_pipeline.train import train_model
    training_date = _training_date_from_context(ds)
    run_id = train_model(
        training_date=training_date,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )
    if not run_id:
        raise RuntimeError(f"Training falló para fecha {training_date}")
    return run_id  # se pushea automáticamente como XCom 'return_value'


def _promote_model(ti, **_):
    from src.training_pipeline.registry import promote_model_to_production
    run_id = ti.xcom_pull(task_ids="train_model")
    if not run_id:
        raise RuntimeError("No run_id en XCom desde train_model")
    promote_model_to_production(
        run_id=run_id,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )


def _run_drift_report(ds: str, **_):
    # Se ejecuta tras entrenar y promover, usando el último modelo Production
    # disponible. La función toma la fecha del DAG run como punto de evaluación.
    from src.monitoring.report import build_and_persist_report
    build_and_persist_report(
        evaluation_date=_training_date_from_context(ds),
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )


with DAG(
    dag_id="training_dag",
    description="Entrenamiento + promoción + reporte de drift mensual",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "training", "drift"],
) as dag:

    prepare_offline_store = PythonOperator(
        task_id="prepare_offline_store",
        python_callable=_prepare_offline_store,
    )

    apply_feast_task = PythonOperator(
        task_id="apply_feast",
        python_callable=_apply_feast,
    )

    populate_online_store = PythonOperator(
        task_id="populate_online_store",
        python_callable=_populate_online_store,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=_promote_model,
    )

    drift_report = PythonOperator(
        task_id="drift_report",
        python_callable=_run_drift_report,
    )

    (
        prepare_offline_store
        >> apply_feast_task
        >> populate_online_store
        >> train_model
        >> promote_model
        >> drift_report
    )
