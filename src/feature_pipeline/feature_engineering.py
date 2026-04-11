"""
Script para cálculo de features.


Features de ventana (últimas N lecturas por pozo) — alineado con clase 3:
- avg_prod_gas_10m, avg_prod_pet_10m
- last_prod_gas, last_prod_pet
- n_readings

Features temporales adicionales (EDA):
- target_lag1, target_lag2, target_lag3
- target_rolling_mean_3, target_rolling_mean_6
- target_rolling_std_3
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def compute_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Computa features por pozo.

    Combina el enfoque de la práctica (ventanas de N lecturas)
    con los features temporales del EDA.
    """
    df = df.sort_values(["idpozo", "fecha"]).copy()

    # --- Codificar tipoextraccion como entero (como en la práctica) ---
    le = LabelEncoder()
    df["tipoextraccion_encoded"] = le.fit_transform(
        df["tipoextraccion"].fillna("Sin Sistema de Extracción")
    )

    records = []
    for idpozo, group in df.groupby("idpozo", sort=False):
        group = group.sort_values("fecha").reset_index(drop=True)

        for i in range(len(group)):
            row = group.iloc[i].to_dict()

            # --- Ventana de últimas N lecturas (patrón de la clase 3) ---
            start_idx = max(0, i - window + 1)
            tail = group.iloc[start_idx:i + 1]

            row["avg_prod_gas_10m"] = float(tail["prod_gas"].mean()) if len(tail) > 0 else 0.0
            row["avg_prod_pet_10m"] = float(tail["prod_pet"].mean()) if len(tail) > 0 else 0.0
            row["last_prod_gas"] = float(tail["prod_gas"].iloc[-1]) if len(tail) > 0 else 0.0
            row["last_prod_pet"] = float(tail["prod_pet"].iloc[-1]) if len(tail) > 0 else 0.0
            row["n_readings"] = int(len(tail))

            # --- Lags del target (EDA) ---
            for lag in [1, 2, 3]:
                row[f"target_lag{lag}"] = float(group["target"].iloc[i - lag]) if i >= lag else np.nan

            # --- Rolling del target (EDA) ---
            if i >= 2:
                recent_3 = group["target"].iloc[max(0, i - 2):i + 1]
                row["target_rolling_mean_3"] = float(recent_3.mean())
                row["target_rolling_std_3"] = float(recent_3.std())
            else:
                row["target_rolling_mean_3"] = np.nan
                row["target_rolling_std_3"] = np.nan

            if i >= 5:
                recent_6 = group["target"].iloc[max(0, i - 5):i + 1]
                row["target_rolling_mean_6"] = float(recent_6.mean())
            else:
                row["target_rolling_mean_6"] = np.nan

            # --- Meses produciendo ---
            row["months_producing"] = i + 1

            # --- Log transform ---
            row["target_log"] = float(np.log1p(max(row.get("target", 0) or 0, 0)))

            records.append(row)

    feat_df = pd.DataFrame(records)

    # Asegurar tipos
    feat_df["tef"] = pd.to_numeric(feat_df["tef"], errors="coerce").fillna(0)
    feat_df["profundidad"] = pd.to_numeric(feat_df["profundidad"], errors="coerce")

    return feat_df