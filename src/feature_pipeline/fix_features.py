import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Computa features por pozo de forma vectorizada.
    """
    # Ordenar chronologically first to avoid data leakage in encoding
    df = df.sort_values("fecha").copy()

    # --- Codificar tipoextraccion sin fittear en todo el dataset (evitar contaminación) ---
    # Usamos factorize sobre los datos ordenados por fecha, así el ID de una categoría
    # depende sólo de su primera aparición cronológica y el test no "contamina" al train.
    df["tipoextraccion_encoded"] = pd.factorize(df["tipoextraccion"].fillna("Sin Sistema de Extracción"))[0]

    # Ahora ordenamos para las operaciones de ventana por pozo
    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)

    # --- Operaciones vectorizadas por idpozo ---
    # Variables auxiliares para ventanas
    grouped = df.groupby("idpozo")
    
    # --- Ventana de últimas N lecturas (patrón de la clase 3) ---
    df["avg_prod_gas_10m"] = grouped["prod_gas"].transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0.0)
    df["avg_prod_pet_10m"] = grouped["prod_pet"].transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0.0)
    df["last_prod_gas"] = df["prod_gas"].fillna(0.0)
    df["last_prod_pet"] = df["prod_pet"].fillna(0.0)
    df["n_readings"] = grouped["prod_gas"].transform(lambda x: x.rolling(window, min_periods=1).count()).fillna(0).astype(int)

    # --- Lags del target (EDA) ---
    for lag in [1, 2, 3]:
        df[f"target_lag{lag}"] = grouped["target"].shift(lag)

    # --- Rolling del target (EDA) ---
    df["target_rolling_mean_3"] = grouped["target"].transform(lambda x: x.rolling(3).mean())
    df["target_rolling_std_3"] = grouped["target"].transform(lambda x: x.rolling(3).std())
    df["target_rolling_mean_6"] = grouped["target"].transform(lambda x: x.rolling(6).mean())

    # --- Meses produciendo ---
    df["months_producing"] = grouped.cumcount() + 1

    # --- Log transform ---
    # Manejar fillna y clip para evitar logs negativos o de NaN
    target_clean = df["target"].fillna(0).clip(lower=0)
    df["target_log"] = np.log1p(target_clean)

    # Asegurar tipos
    df["tef"] = pd.to_numeric(df["tef"], errors="coerce").fillna(0)
    df["profundidad"] = pd.to_numeric(df["profundidad"], errors="coerce")

    return df
