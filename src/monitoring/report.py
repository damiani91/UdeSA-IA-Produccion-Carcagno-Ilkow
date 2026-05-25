"""Reporte de drift + decay.

Compone PSI por feature + MAE móvil del último mes, lo persiste como JSON en
`feature_store/data/drift_reports/{date}.json` y lo loguea como artefacto en
MLflow (en un run separado al de entrenamiento).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

from src.monitoring.drift import compute_decay, compute_psi_table

# Mantener consistencia con training_pipeline / inference_pipeline.
FEATURE_COLS = [
    "tipoextraccion_encoded",
    "tef",
    "profundidad",
    "avg_prod_gas_10m",
    "avg_prod_pet_10m",
    "last_prod_gas",
    "last_prod_pet",
    "n_readings",
    "target_lag1",
    "target_lag2",
    "target_lag3",
    "target_rolling_mean_3",
]
TARGET = "target"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PARQUET_PATH = PROJECT_ROOT / "feature_store" / "data" / "well_features.parquet"
REPORTS_DIR = PROJECT_ROOT / "feature_store" / "data" / "drift_reports"

# Umbrales (ver plan §4.1).
PSI_DRIFT_THRESHOLD = 0.25
DECAY_RATIO_THRESHOLD = 1.5  # MAE_actual > 1.5 * MAE_validacion → decay


def _load_production_model_and_baseline(
    model_name: str = "well_production_model",
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> tuple[object, dict, Optional[str]]:
    """Carga el modelo en stage Production y su MAE de validación baseline."""
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # MLflow 3 deprecated stages pero sigue soportando models:/Name/Production.
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

    versions = client.get_latest_versions(model_name, stages=["Production"])
    baseline: dict = {}
    run_id: Optional[str] = None
    if versions:
        run_id = versions[0].run_id
        run = client.get_run(run_id)
        for key in ("val_mae", "val_rmse", "val_mape"):
            if key in run.data.metrics:
                baseline[key] = float(run.data.metrics[key])

    return model, baseline, run_id


def build_report(
    evaluation_date: str,
    reference_months: int = 12,
    current_months: int = 1,
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> dict:
    """Construye el reporte de drift + decay como dict puro.

    - Ventana de referencia: features hasta `evaluation_date - current_months`.
    - Ventana actual: último mes hasta `evaluation_date`.
    """
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el offline store en {PARQUET_PATH}. "
            "Corré primero el feature pipeline."
        )

    df = pd.read_parquet(PARQUET_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])

    eval_ts = pd.Timestamp(evaluation_date)
    current_start = eval_ts - pd.DateOffset(months=current_months)
    reference_start = current_start - pd.DateOffset(months=reference_months)

    reference_df = df[(df["fecha"] >= reference_start) & (df["fecha"] < current_start)]
    current_df = df[(df["fecha"] >= current_start) & (df["fecha"] <= eval_ts)]

    psi_table = compute_psi_table(reference_df, current_df, FEATURE_COLS)
    psi_values = [v for v in psi_table.values() if pd.notna(v)]
    psi_max = float(max(psi_values)) if psi_values else float("nan")
    data_drift_detected = any(v > PSI_DRIFT_THRESHOLD for v in psi_values)

    # Decay: usar el modelo Production sobre el último mes (que ya tiene `y` real).
    model, baseline, baseline_run_id = _load_production_model_and_baseline(
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    decay = compute_decay(model, current_df, FEATURE_COLS, TARGET)

    baseline_mae = baseline.get("val_mae", float("nan"))
    model_decay_detected = False
    if pd.notna(decay["mae"]) and pd.notna(baseline_mae) and baseline_mae > 0:
        model_decay_detected = decay["mae"] > DECAY_RATIO_THRESHOLD * baseline_mae

    return {
        "evaluation_date": evaluation_date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "windows": {
            "reference": {
                "from": reference_start.strftime("%Y-%m-%d"),
                "to": current_start.strftime("%Y-%m-%d"),
                "n_rows": int(len(reference_df)),
            },
            "current": {
                "from": current_start.strftime("%Y-%m-%d"),
                "to": eval_ts.strftime("%Y-%m-%d"),
                "n_rows": int(len(current_df)),
            },
        },
        "data_drift": {
            "psi_by_feature": psi_table,
            "psi_max": psi_max,
            "threshold": PSI_DRIFT_THRESHOLD,
            "detected": bool(data_drift_detected),
        },
        "model_decay": {
            "current": decay,
            "baseline": baseline,
            "baseline_run_id": baseline_run_id,
            "ratio_threshold": DECAY_RATIO_THRESHOLD,
            "detected": bool(model_decay_detected),
        },
    }


def persist_report(report: dict, evaluation_date: str) -> Path:
    """Guarda el reporte como JSON. Devuelve la ruta absoluta."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{evaluation_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return out_path


def log_report_to_mlflow(
    report: dict,
    evaluation_date: str,
    mlflow_tracking_uri: str = "http://mlflow:5000",
    experiment_name: str = "well_production_drift",
) -> None:
    """Loguea el reporte como artefacto + métricas resumidas en un run dedicado."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"drift_{evaluation_date}"):
        mlflow.log_param("evaluation_date", evaluation_date)
        mlflow.log_param("data_drift_detected", report["data_drift"]["detected"])
        mlflow.log_param("model_decay_detected", report["model_decay"]["detected"])

        if pd.notna(report["data_drift"]["psi_max"]):
            mlflow.log_metric("psi_max", report["data_drift"]["psi_max"])

        decay = report["model_decay"]["current"]
        for key in ("mae", "rmse", "mape"):
            value = decay.get(key)
            if value is not None and pd.notna(value):
                mlflow.log_metric(f"current_{key}", float(value))

        # Persistir el JSON en disco y subirlo como artefacto.
        path = persist_report(report, evaluation_date)
        mlflow.log_artifact(str(path), artifact_path="drift_reports")


def build_and_persist_report(
    evaluation_date: str,
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> dict:
    """Atajo usado por el DAG y por el endpoint cuando se quiere regenerar."""
    report = build_report(evaluation_date, mlflow_tracking_uri=mlflow_tracking_uri)
    persist_report(report, evaluation_date)
    try:
        log_report_to_mlflow(report, evaluation_date, mlflow_tracking_uri)
    except Exception as e:  # noqa: BLE001
        # No queremos que un MLflow caído rompa el DAG entero; el JSON ya quedó.
        print(f"[drift_report] No se pudo loguear a MLflow: {e}")
    return report


def load_latest_report() -> Optional[dict]:
    """Devuelve el reporte más reciente persistido en disco, o None."""
    if not REPORTS_DIR.exists():
        return None
    files = sorted(REPORTS_DIR.glob("*.json"))
    if not files:
        return None
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def load_report_by_date(date: str) -> Optional[dict]:
    path = REPORTS_DIR / f"{date}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
