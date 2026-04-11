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


def compute_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Computa features por pozo.

    Combina el enfoque de la práctica (ventanas de N lecturas)
    con los features temporales del EDA.
    """
    df = df.sort_values(["fecha", "idpozo"]).copy()

    # --- Codificar tipoextraccion sin contaminar (data leakage) ---
    # pd.factorize en datos ordenados por fecha asigna los IDs en orden de aparición.
    # Esto asegura que las categorías futuras no desplacen a las ya aprendidas.
    df["tipoextraccion_encoded"] = pd.factorize(df["tipoextraccion"].fillna("Sin Sistema de Extracción"))[0]

    # Ahora ordenamos para las operaciones de ventana por pozo
    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)

    # --- Operaciones vectorizadas por idpozo (evita el loop O(n²)) ---
    grouped = df.groupby("idpozo", sort=False)
    
    # --- Ventana de últimas N lecturas (patrón de la clase 3) ---
    df["avg_prod_gas_10m"] = grouped["prod_gas"].transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0.0).astype(float)
    df["avg_prod_pet_10m"] = grouped["prod_pet"].transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0.0).astype(float)
    df["last_prod_gas"] = df["prod_gas"].fillna(0.0).astype(float)
    df["last_prod_pet"] = df["prod_pet"].fillna(0.0).astype(float)
    df["n_readings"] = grouped["prod_gas"].transform(lambda x: x.rolling(window, min_periods=1).count()).astype(int)

    # --- Lags del target (EDA) ---
    for lag in [1, 2, 3]:
        df[f"target_lag{lag}"] = grouped["target"].shift(lag).astype(float)

    # --- Rolling del target (EDA) ---
    df["target_rolling_mean_3"] = grouped["target"].transform(lambda x: x.rolling(3).mean()).astype(float)
    df["target_rolling_std_3"] = grouped["target"].transform(lambda x: x.rolling(3).std()).astype(float)
    df["target_rolling_mean_6"] = grouped["target"].transform(lambda x: x.rolling(6).mean()).astype(float)

    # --- Meses produciendo ---
    df["months_producing"] = grouped.cumcount() + 1

    # --- Log transform ---
    df["target_log"] = np.log1p(df["target"].fillna(0).clip(lower=0)).astype(float)

    # Asegurar tipos
    df["tef"] = pd.to_numeric(df["tef"], errors="coerce").fillna(0)
    df["profundidad"] = pd.to_numeric(df["profundidad"], errors="coerce")

    return df