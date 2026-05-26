"""Training pipeline con MLflow — siguiendo clase 2.

Usa mlflow.sklearn.autolog() para captura automática de métricas.
Lee features del offline store via get_historical_features() (clase 3).
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_STORE_REPO = "feature_store"
PARQUET_PATH = f"{FEATURE_STORE_REPO}/data/well_features.parquet"

# Features para el modelo (combinación clase 3 + EDA)
FEAST_FEATURES = [
    "well_stats:tipoextraccion_encoded",
    "well_stats:tef",
    "well_stats:profundidad",
    "well_stats:avg_prod_gas_10m",
    "well_stats:avg_prod_pet_10m",
    "well_stats:last_prod_gas",
    "well_stats:last_prod_pet",
    "well_stats:n_readings",
    "well_stats:target_lag1",
    "well_stats:target_lag2",
    "well_stats:target_lag3",
    "well_stats:target_rolling_mean_3",
]

FEATURE_COLS = [f.split(":")[1] for f in FEAST_FEATURES]
TARGET = "target"


def train_model(
    training_date: str,
    mlflow_tracking_uri: str = "http://mlflow:5000",
    experiment_name: str = "well_production_forecast_v2",
    model_type: str = "gradient_boosting",
    model_params: dict | None = None,
) -> str | None:
    """Entrena un modelo leyendo del Feature Store (offline).

    get_historical_features() para training data.
    mlflow.sklearn.autolog() + log_param/log_metric manual.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Ensure experiment uses HTTP artifact proxy (mlflow-artifacts:/ URI).
    # Without this, MLflow creates the experiment with a local filesystem path
    # as artifact_location, which is inaccessible from the Airflow container.
    _client = mlflow.MlflowClient(mlflow_tracking_uri)
    if _client.get_experiment_by_name(experiment_name) is None:
        _client.create_experiment(experiment_name, artifact_location="mlflow-artifacts:/")

    mlflow.set_experiment(experiment_name)

    try:
        # 1. Leer del Feature Store offline (parquet ya tiene todas las features computadas)
        raw_df = pd.read_parquet(PARQUET_PATH)

        # Filtrar hasta training_date
        raw_df = raw_df[raw_df["fecha"] <= pd.Timestamp(training_date)]

        print("Cargando features históricas desde el parquet...")
        training_df = raw_df[["idpozo", "fecha", TARGET] + FEATURE_COLS].copy()
        training_df = training_df.rename(columns={"fecha": "event_timestamp"})
        training_df["event_timestamp"] = pd.to_datetime(training_df["event_timestamp"], utc=True)

        # Limpiar
        training_df = training_df.dropna(subset=[TARGET])
        training_df = training_df.dropna(subset=FEATURE_COLS)
        training_df = training_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

        # 2. Split temporal
        cutoff = pd.Timestamp(training_date, tz="UTC") - pd.DateOffset(months=3)
        train_mask = training_df["event_timestamp"] <= cutoff
        val_mask = training_df["event_timestamp"] > cutoff

        X_train = training_df.loc[train_mask, FEATURE_COLS]
        y_train = training_df.loc[train_mask, TARGET]
        X_val = training_df.loc[val_mask, FEATURE_COLS]
        y_val = training_df.loc[val_mask, TARGET]

        # 3. Entrenar con MLflow (clase 2: autolog + params manuales)
        if model_params is None:
            model_params = {"n_estimators": 200, "max_depth": 5, "random_state": 42}

        # Autolog captura métricas, parámetros y modelo automáticamente (clase 2)
        mlflow.sklearn.autolog(log_models=True)

        with mlflow.start_run() as run:
            # Params manuales adicionales (clase 2: log_param)
            mlflow.log_param("training_date", training_date)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_wells", int(training_df["idpozo"].nunique()))
            mlflow.log_param("n_samples_train", len(X_train))
            mlflow.log_param("n_samples_val", len(X_val))
            mlflow.log_param("features", str(FEATURE_COLS))

            # Seleccionar modelo
            if model_type == "random_forest":
                model = RandomForestRegressor(**model_params)
            else:
                model = GradientBoostingRegressor(
                    learning_rate=0.1, **model_params
                )

            model.fit(X_train, y_train)

            # Métricas de validación (autolog captura training, agregamos val)
            y_val_pred = model.predict(X_val)
            mlflow.log_metric("val_mae", mean_absolute_error(y_val, y_val_pred))
            mlflow.log_metric("val_rmse", np.sqrt(mean_squared_error(y_val, y_val_pred)))

            nonzero = y_val != 0
            if nonzero.any():
                mape = float((abs(y_val[nonzero] - y_val_pred[nonzero]) / y_val[nonzero]).mean() * 100)
                mlflow.log_metric("val_mape", mape)

            # Registrar modelo en Model Registry (clase 2: ciclo de vida)
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                registered_model_name="well_production_model",
            )

            # Snapshot del dataset de training como artefacto.
            # Lo usa el próximo ciclo del DAG como reference para drift monitoring
            # (compara distribuciones de features y residuos contra este baseline).
            ref_path = "/tmp/reference_dataset.parquet"
            training_df.to_parquet(ref_path, index=False)
            mlflow.log_artifact(ref_path, artifact_path="reference")

            # Feature importances como artefacto
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values().plot.barh(ax=ax)
            ax.set_title("Feature Importances")
            fig.tight_layout()
            fig.savefig("/tmp/feature_importances.png", dpi=100)
            mlflow.log_artifact("/tmp/feature_importances.png")
            plt.close()

            print(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    except Exception as e:
        print(f"Error en training pipeline: {e}")
        # Terminar activamente cualquier run de MLflow abortado
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        return None