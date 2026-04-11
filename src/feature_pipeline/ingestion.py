"""Carga y limpieza de datos crudos. Columnas reales del CSV."""

from pathlib import Path
import numpy as np
import pandas as pd

DROP_COLS = ["vida_util", "observaciones", "rectificado", "habilitado", "idusuario"]


def load_production_data(path: str | Path) -> pd.DataFrame:
    """
    Carga y limpieza de los datos de producción.
    """
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    # Fecha a partir de anio + mes
    df["fecha"] = pd.to_datetime(
        df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01"
    )

    # Descartar columnas con alta nulidad
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Target según tipopozo
    df["target"] = np.where(
        df["tipopozo"] == "Gasífero", df["prod_gas"],
        np.where(df["tipopozo"].isin(["Petrolífero", "Petrolero"]), df["prod_pet"], np.nan),
    )

    for col in ["prod_pet", "prod_gas", "prod_agua", "tef", "profundidad"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)
    return df


def load_wells_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)