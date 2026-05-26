"""Drift monitoring — genera reporte de data drift, concept drift y model decay.

Compara:
  - Reference: snapshot del dataset de training del modelo en Production actual
    (artefacto `reference/reference_dataset.parquet` del run del modelo viejo).
  - Current: slice del parquet del feature store con datos del último mes
    (entre el training_date anterior y el actual), con target real ya disponible.

Outputs (loggeados al run NUEVO recibido por XCom):
  - drift_report.html (Evidently)
  - drift_metrics.json (dict estructurado con todas las métricas)
  - mlflow.log_metric: decay_mae_current, decay_mape_current,
    drift_psi_avg, drift_n_features_drifted, concept_ks_residuals_pvalue

Modelo de skip gracioso: si no hay Production model previo con snapshot,
loggea warning y termina sin error (no rompe el DAG).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

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
PARQUET_PATH = "feature_store/data/well_features.parquet"
MODEL_NAME = "well_production_model"


class NoBaselineError(Exception):
    """No hay modelo previo en Production con reference snapshot — primer ciclo."""


def _previous_training_date(training_date: str) -> str:
    y, m, _ = training_date.split("-")
    y, m = int(y), int(m)
    return f"{y - 1}-12-01" if m == 1 else f"{y}-{m - 1:02d}-01"


def _compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    ref = reference.replace([np.inf, -np.inf], np.nan).dropna()
    cur = current.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = ref.quantile(quantiles).unique()
    if len(bins) < 3:
        return 0.0

    bins[0], bins[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _get_production_run(client: mlflow.MlflowClient) -> tuple[str, str]:
    """Devuelve (run_id, version) del modelo en Production. Tira NoBaselineError si no hay."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        raise NoBaselineError(f"No se pudo consultar el registry: {e}")

    prod = [v for v in versions if v.current_stage == "Production"]
    if not prod:
        raise NoBaselineError("No hay versión en stage Production todavía")

    return prod[0].run_id, prod[0].version


def _download_reference(client: mlflow.MlflowClient, run_id: str) -> pd.DataFrame:
    """Descarga el snapshot reference del run viejo. Tira NoBaselineError si no existe."""
    try:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="reference"
        )
    except Exception as e:
        raise NoBaselineError(f"Run {run_id} no tiene artefacto 'reference': {e}")

    parquet = Path(local_dir) / "reference_dataset.parquet"
    if not parquet.exists():
        raise NoBaselineError(f"Falta {parquet}")

    return pd.read_parquet(parquet)


def _slice_current(training_date: str, prev_training_date: str) -> pd.DataFrame:
    """Carga el parquet y filtra al período entre prev_training_date y training_date."""
    df = pd.read_parquet(PARQUET_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])
    mask = (
        (df["fecha"] > pd.Timestamp(prev_training_date))
        & (df["fecha"] <= pd.Timestamp(training_date))
        & df[TARGET].notna()
    )
    cur = df[mask].copy()
    cur = cur.dropna(subset=FEATURE_COLS)
    cur = cur.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    return cur


def _evidently_html(ref: pd.DataFrame, cur: pd.DataFrame, output_path: str) -> dict:
    """Genera HTML report con Evidently y devuelve el dict con drift por columna."""
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently import ColumnMapping

    mapping = ColumnMapping(
        target=TARGET,
        prediction="prediction",
        numerical_features=FEATURE_COLS,
    )

    report = Report(metrics=[DataDriftPreset()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report.run(reference_data=ref, current_data=cur, column_mapping=mapping)
    report.save_html(output_path)

    return report.as_dict()


def generate_drift_report(
    training_date: str,
    new_run_id: str,
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> dict | None:
    """Entry point. Devuelve dict con métricas o None si se hizo skip."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient(mlflow_tracking_uri)

    print(f"[drift] training_date={training_date} new_run_id={new_run_id}")

    try:
        old_run_id, old_version = _get_production_run(client)
        print(f"[drift] reference: Production v{old_version} (run_id={old_run_id})")
        ref_df = _download_reference(client, old_run_id)
    except NoBaselineError as e:
        print(f"[drift] skip — {e}")
        return None

    prev_date = _previous_training_date(training_date)
    cur_df = _slice_current(training_date, prev_date)
    print(f"[drift] reference rows={len(ref_df)} current rows={len(cur_df)} (window {prev_date}→{training_date})")

    if len(cur_df) < 30:
        print("[drift] ventana actual demasiado chica (<30 filas con target) — skip")
        return None

    # Predicciones del modelo VIEJO sobre datos actuales (decay)
    old_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    cur_pred = old_model.predict(cur_df[FEATURE_COLS])
    cur_df = cur_df.copy()
    cur_df["prediction"] = cur_pred

    # Predicciones del modelo viejo sobre el reference también (para residuos)
    ref_pred = old_model.predict(ref_df[FEATURE_COLS])
    ref_df = ref_df.copy()
    ref_df["prediction"] = ref_pred

    y_cur = cur_df[TARGET].values
    decay_mae = float(np.mean(np.abs(y_cur - cur_pred)))
    nonzero = y_cur != 0
    decay_mape = (
        float(np.mean(np.abs((y_cur[nonzero] - cur_pred[nonzero]) / y_cur[nonzero])) * 100)
        if nonzero.any() else float("nan")
    )

    # Data drift por feature: PSI + KS
    psi_per_feature, ks_per_feature = {}, {}
    for col in FEATURE_COLS:
        psi_per_feature[col] = _compute_psi(ref_df[col], cur_df[col])
        ks_stat, ks_p = stats.ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
        ks_per_feature[col] = {"statistic": float(ks_stat), "pvalue": float(ks_p)}

    psi_values = [v for v in psi_per_feature.values() if not np.isnan(v)]
    psi_avg = float(np.mean(psi_values)) if psi_values else float("nan")
    n_drifted = sum(1 for v in psi_values if v > 0.25)

    # Concept drift: KS sobre residuos
    ref_resid = ref_df[TARGET].values - ref_pred
    cur_resid = y_cur - cur_pred
    ks_resid_stat, ks_resid_p = stats.ks_2samp(ref_resid, cur_resid)

    # Concept drift: delta de feature importances entre modelo viejo y nuevo
    importance_delta = {}
    try:
        new_model = mlflow.sklearn.load_model(f"runs:/{new_run_id}/model")
        for col, old_imp, new_imp in zip(
            FEATURE_COLS, old_model.feature_importances_, new_model.feature_importances_
        ):
            importance_delta[col] = {
                "old": float(old_imp),
                "new": float(new_imp),
                "delta": float(new_imp - old_imp),
            }
    except Exception as e:
        print(f"[drift] no se pudo cargar el modelo nuevo para feature importances: {e}")

    # Evidently HTML
    html_path = "/tmp/drift_report.html"
    try:
        evidently_dict = _evidently_html(ref_df, cur_df, html_path)
        evidently_summary = evidently_dict.get("metrics", [{}])[0].get("result", {})
    except Exception as e:
        print(f"[drift] Evidently HTML falló ({e}), continúo con métricas custom")
        html_path = None
        evidently_summary = {}

    metrics = {
        "training_date": training_date,
        "previous_training_date": prev_date,
        "reference_run_id": old_run_id,
        "reference_model_version": old_version,
        "new_run_id": new_run_id,
        "n_reference": len(ref_df),
        "n_current": len(cur_df),
        "decay": {
            "mae_current": decay_mae,
            "mape_current": decay_mape,
        },
        "data_drift": {
            "psi_per_feature": psi_per_feature,
            "ks_per_feature": ks_per_feature,
            "psi_avg": psi_avg,
            "n_features_drifted_psi_gt_025": n_drifted,
        },
        "concept_drift": {
            "residuals_ks_statistic": float(ks_resid_stat),
            "residuals_ks_pvalue": float(ks_resid_p),
            "feature_importance_delta": importance_delta,
        },
        "evidently_summary": evidently_summary,
    }

    # Loggear todo al run NUEVO
    with mlflow.start_run(run_id=new_run_id):
        mlflow.log_metric("decay_mae_current", decay_mae)
        if not np.isnan(decay_mape):
            mlflow.log_metric("decay_mape_current", decay_mape)
        if not np.isnan(psi_avg):
            mlflow.log_metric("drift_psi_avg", psi_avg)
        mlflow.log_metric("drift_n_features_drifted", n_drifted)
        mlflow.log_metric("concept_ks_residuals_pvalue", float(ks_resid_p))

        json_path = "/tmp/drift_metrics.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        mlflow.log_artifact(json_path, artifact_path="drift")

        if html_path and os.path.exists(html_path):
            mlflow.log_artifact(html_path, artifact_path="drift")

    print(
        f"[drift] OK — MAE={decay_mae:.2f} MAPE={decay_mape:.2f}% "
        f"PSI_avg={psi_avg:.4f} drifted={n_drifted}/{len(FEATURE_COLS)} "
        f"KS_resid_p={ks_resid_p:.4f}"
    )
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-date", required=True)
    parser.add_argument("--new-run-id", required=True)
    parser.add_argument("--mlflow-uri", default="http://localhost:5001")
    args = parser.parse_args()

    generate_drift_report(args.training_date, args.new_run_id, args.mlflow_uri)
