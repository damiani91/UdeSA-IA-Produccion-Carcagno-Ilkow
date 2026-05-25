"""Métricas puras para drift y decay.

- PSI (Population Stability Index): mide el shift de la distribución de una feature
  entre dos ventanas (referencia vs. actual). >0.25 indica drift fuerte.
- compute_decay: MAE / RMSE sobre un set reciente con `y` real, usando el modelo
  productivo.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calcula el Population Stability Index entre dos distribuciones.

    Usa bins construidos sobre los cuantiles de la serie de referencia para
    evitar bins vacíos por sesgo. Devuelve un float >= 0; por convención:
    <0.1 sin drift, 0.1-0.25 leve, >0.25 fuerte.
    """
    ref = pd.Series(reference).dropna().astype(float)
    cur = pd.Series(current).dropna().astype(float)

    if ref.empty or cur.empty:
        return float("nan")

    # Bins basados en cuantiles de la referencia.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref.values, quantiles))

    # Si la feature es casi constante, no hay drift mensurable.
    if len(edges) < 2:
        return 0.0

    # Extender los bordes para capturar valores fuera del rango original.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref.values, bins=edges)
    cur_counts, _ = np.histogram(cur.values, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Evitar log(0): sumar epsilon (Laplace smoothing) a cada bin.
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi_per_bin = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(np.sum(psi_per_bin))


def compute_psi_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Iterable[str],
    bins: int = 10,
) -> dict[str, float]:
    """PSI por cada feature. Devuelve {feature: psi}."""
    out: dict[str, float] = {}
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            out[col] = float("nan")
            continue
        out[col] = compute_psi(reference_df[col], current_df[col], bins=bins)
    return out


def compute_decay(
    model,
    df_recent: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict:
    """Calcula métricas de decay sobre un dataframe con `y` real disponible.

    Devuelve {"mae", "rmse", "mape", "n_samples"}. Si no hay filas válidas
    devuelve NaN en las métricas (en vez de explotar).
    """
    df = df_recent.dropna(subset=[target_col] + feature_cols).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

    n_samples = int(len(df))
    if n_samples == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n_samples": 0}

    X = df[feature_cols].values
    y_true = df[target_col].values.astype(float)
    y_pred = model.predict(X)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )

    return {"mae": mae, "rmse": rmse, "mape": mape, "n_samples": n_samples}
