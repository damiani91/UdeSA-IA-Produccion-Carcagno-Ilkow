"""
Monthly ML pipeline for oil & gas well production forecasting.

Schedule: 10th of each month at 06:00 UTC.
Training date: first day of the previous month (auto-computed from logical_date).

Task chain:
    download_data >> prepare_offline_store >> apply_feast >> populate_online_store
    >> train_model >> promote_model >> reload_api
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, date

from airflow import DAG
from airflow.operators.python import PythonOperator


# ─── helpers ──────────────────────────────────────────────────────────────────

def _first_day_of_prev_month(logical_date: date) -> str:
    """Return YYYY-MM-01 for the month before logical_date."""
    if logical_date.month == 1:
        return f"{logical_date.year - 1}-12-01"
    return f"{logical_date.year}-{logical_date.month - 1:02d}-01"


def _ensure_app_in_path():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


# ─── task callables ───────────────────────────────────────────────────────────

def task_download_data(**context):
    """Download fresh produccion.csv and pozos.csv from datos.gob.ar."""
    _ensure_app_in_path()
    from pathlib import Path
    import requests

    URLS = {
        "produccion": (
            "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c"
            "/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccion.csv"
        ),
        "pozos": (
            "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c"
            "/resource/cbfa4d79-ffb3-4096-bab5-eb0dde9a8385/download/pozos.csv"
        ),
    }
    raw_dir = Path("/app/data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        dest = raw_dir / f"{name}.csv"
        print(f"Downloading {url} → {dest}")
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_mb = dest.stat().st_size / 1e6
        print(f"  Saved {dest} ({size_mb:.1f} MB)")


def task_prepare_offline_store(**context):
    _ensure_app_in_path()
    logical_date = context["logical_date"].date()
    training_date = _first_day_of_prev_month(logical_date)
    print(f"prepare_offline_store up_to_date={training_date}")
    from scripts.populate_feature_store import prepare_offline_store
    prepare_offline_store(up_to_date=training_date)


def task_apply_feast(**context):
    _ensure_app_in_path()
    from scripts.populate_feature_store import apply_feast
    apply_feast()


def task_populate_online_store(**context):
    _ensure_app_in_path()
    from scripts.populate_feature_store import populate_online_store
    populate_online_store()


def task_train_model(**context):
    _ensure_app_in_path()
    logical_date = context["logical_date"].date()
    training_date = _first_day_of_prev_month(logical_date)
    print(f"train_model training_date={training_date}")

    from src.training_pipeline.train import train_model
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    run_id = train_model(training_date=training_date, mlflow_tracking_uri=mlflow_uri)

    if run_id is None:
        raise RuntimeError("train_model returned None — training failed.")

    context["ti"].xcom_push(key="run_id", value=run_id)
    print(f"Training completed. run_id={run_id}")


def task_promote_model(**context):
    _ensure_app_in_path()
    run_id = context["ti"].xcom_pull(task_ids="train_model", key="run_id")
    print(f"Promoting run_id={run_id} to Production")

    from src.training_pipeline.registry import promote_model_to_production
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    promote_model_to_production(run_id=run_id, mlflow_tracking_uri=mlflow_uri)


def task_reload_api(**context):
    """Call the API /admin/reload endpoint so it picks up the new model."""
    import requests
    api_url = os.environ.get("API_URL", "http://api:8000")
    reload_secret = os.environ.get("RELOAD_SECRET", "changeme-reload")
    resp = requests.post(
        f"{api_url}/admin/reload",
        headers={"X-Reload-Secret": reload_secret},
        timeout=60,
    )
    resp.raise_for_status()
    print(f"API reload: {resp.status_code} {resp.json()}")


# ─── DAG definition ───────────────────────────────────────────────────────────

with DAG(
    dag_id="monthly_well_production_pipeline",
    description=(
        "Pipeline mensual: descarga datos → features → entrena modelo → "
        "promueve a Production → recarga API"
    ),
    schedule_interval="0 6 10 * *",
    start_date=datetime(2025, 1, 10),
    catchup=False,
    max_active_runs=1,
    tags=["oil-gas", "ml", "monthly"],
) as dag:

    t1 = PythonOperator(
        task_id="download_data",
        python_callable=task_download_data,
    )

    t2 = PythonOperator(
        task_id="prepare_offline_store",
        python_callable=task_prepare_offline_store,
    )

    t3 = PythonOperator(
        task_id="apply_feast",
        python_callable=task_apply_feast,
    )

    t4 = PythonOperator(
        task_id="populate_online_store",
        python_callable=task_populate_online_store,
    )

    t5 = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    t6 = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )

    t7 = PythonOperator(
        task_id="reload_api",
        python_callable=task_reload_api,
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
