"""Poblar offline y online stores — patrón de la práctica de clase 3.

Flujo:
1. Cargar datos crudos
2. Computar features → guardar como parquet (offline store)
3. feast apply (registrar definiciones)
4. write_to_online_store (materializar features más recientes al online store)
"""

import subprocess
import pandas as pd
from feast import FeatureStore
from src.feature_pipeline.ingestion import load_production_data
from src.feature_pipeline.feature_engineering import compute_features

FEATURE_STORE_REPO = "feature_store"
PARQUET_PATH = f"{FEATURE_STORE_REPO}/data/well_features.parquet"
PROD_FILE = "data/raw/produccion.csv"


def prepare_offline_store(up_to_date: str | None = None):
    """Computa features y guarda como parquet (offline store)."""
    print("Cargando datos crudos...")
    df = load_production_data(PROD_FILE)

    # Filtrar solo pozos activos
    active = df[df["tipoestado"] == "Extracción Efectiva"]["idpozo"].unique()
    df = df[df["idpozo"].isin(active) & df["target"].notna()]

    if up_to_date:
        df = df[df["fecha"] <= pd.Timestamp(up_to_date)]

    print("Computando features...")
    feat_df = compute_features(df)

    feat_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Offline store: {len(feat_df)} filas, {feat_df['idpozo'].nunique()} pozos")


def apply_feast():
    """Registra las definiciones de Feast (feast apply)."""
    subprocess.run(["feast", "apply"], cwd=FEATURE_STORE_REPO, check=True)
    print("Feast apply completado.")


def populate_online_store():
    """Materializa los features más recientes al online store.

    Patrón de la clase 3: write_to_online_store con la última fila de cada pozo.
    """
    print("Materializando al online store...")
    feat_df = pd.read_parquet(PARQUET_PATH)

    # Última lectura de cada pozo (la más reciente)
    latest_df = feat_df.sort_values("fecha").groupby("idpozo").tail(1)

    store = FeatureStore(repo_path=FEATURE_STORE_REPO)
    store.write_to_online_store(
        feature_view_name="well_stats",
        df=latest_df,
    )
    print(f"Online store: {len(latest_df)} pozos materializados.")

    # Validación (como en la práctica)
    pozo_ejemplo = latest_df["idpozo"].iloc[0]
    features = store.get_online_features(
        features=["well_stats:avg_prod_gas_10m", "well_stats:n_readings"],
        entity_rows=[{"idpozo": pozo_ejemplo}],
    ).to_dict()
    print(f"Validación pozo {pozo_ejemplo}: {features}")


if __name__ == "__main__":
    prepare_offline_store()
    apply_feast()
    populate_online_store()