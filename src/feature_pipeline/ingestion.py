"""Carga y limpieza de datos crudos. Columnas reales del CSV."""

from pathlib import Path
import numpy as np
import pandas as pd

DROP_COLS = ["vida_util", "observaciones", "rectificado", "habilitado", "idusuario"]


def load_production_data(path: str | Path, chunksize: int = 100000) -> pd.DataFrame:
    """
    Carga y limpieza de los datos de producción en chunks para evitar OOM.
    """
    chunks = []
    
    # Procesar el archivo en bloques (chunks) para reducir uso de RAM durante el parseo
    for chunk in pd.read_csv(path, encoding="utf-8", chunksize=chunksize, low_memory=False):
        
        # Descartar columnas innecesarias inmediatamente para liberar memoria
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], errors="ignore")

        # Fecha a partir de anio + mes
        chunk["fecha"] = pd.to_datetime(
            chunk["anio"].astype(str) + "-" + chunk["mes"].astype(str).str.zfill(2) + "-01"
        )
        
        # Target según tipopozo
        if "tipopozo" in chunk.columns:
            chunk["target"] = np.where(
                chunk["tipopozo"] == "Gasífero", chunk.get("prod_gas", np.nan),
                np.where(chunk["tipopozo"].isin(["Petrolífero", "Petrolero"]), chunk.get("prod_pet", np.nan), np.nan),
            )

        for col in ["prod_pet", "prod_gas", "prod_agua", "tef", "profundidad"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)
    return df


def load_wells_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", low_memory=False)