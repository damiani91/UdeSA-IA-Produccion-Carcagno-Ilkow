"""Inference pipeline — lee del online store (clase 3).

Patrón: get_online_features() para obtener features precomputados,
luego predecir con el modelo cargado desde MLflow Registry.
"""

import mlflow
import numpy as np
import pandas as pd
from feast import FeatureStore

FEATURE_STORE_REPO = "feature_store"

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


class ForecastService:
    def __init__(self, mlflow_tracking_uri="http://mlflow:5000"):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.model = mlflow.sklearn.load_model("models:/well_production_model/Production")
        self.store = FeatureStore(repo_path=FEATURE_STORE_REPO)
        # También cargar offline features para forecast autoregresivo
        self.features_df = pd.read_parquet(f"{FEATURE_STORE_REPO}/data/well_features.parquet")
        # Computar la mediana global de features para evitar sesgar la imputación artificialmente con 0.0
        self.feature_medians = self.features_df[FEATURE_COLS].median()

    def predict(self, id_well: str, date_start: str, date_end: str) -> list[dict]:
        """Genera forecast usando online features (clase 3) + autoregresivo."""
        idpozo = int(id_well)

        # Intentar leer features del online store (clase 3)
        online_features = self.store.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[{"idpozo": idpozo}],
        ).to_df()

        dates = pd.date_range(start=date_start, end=date_end, freq="MS")
        results = []

        for target_date in dates:
            # Buscar en offline features primero
            well_data = self.features_df[
                (self.features_df["idpozo"] == idpozo) &
                (self.features_df["fecha"] == target_date)
            ]

            if not well_data.empty:
                X_df = well_data[FEATURE_COLS]
            elif not online_features.empty:
                # Usar online features como fallback
                X_df = online_features[FEATURE_COLS]
            else:
                results.append({"date": target_date.strftime("%Y-%m-%d"), "prod": 0.0})
                continue

            # Imputar valores nulos con la mediana poblacional para no sesgar
            X_df = X_df.fillna(self.feature_medians)
            X = X_df.values

            # Reemplazar infinitos u otros casos residuales por las dudas
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            pred = float(self.model.predict(X)[0])
            results.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "prod": round(max(pred, 0), 2),
            })

        return results