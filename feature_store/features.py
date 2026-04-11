"""Definiciones Feast

Definición del Entity, la Source offline y la Feature View en donde se establecen 
los atributos con los que se trabajará en el proyecto.

"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32

# Entidad: idpozo (como en la práctica, usando el campo directamente)
pozo = Entity(
    name="idpozo",
    description="Identificador único del pozo de extracción",
)

# Source offline
well_stats_source = FileSource(
    path="/app/feature_store/data/well_features.parquet",
    timestamp_field="fecha",
)

# Feature View
well_stats = FeatureView(
    name="well_stats",
    entities=[pozo],
    schema=[
        # Targets
        Field(name="prod_gas", dtype=Float32),
        Field(name="prod_pet", dtype=Float32),
        Field(name="target", dtype=Float32),

        # Features del dataset original
        Field(name="prod_agua", dtype=Float32),
        Field(name="tef", dtype=Float32),
        Field(name="profundidad", dtype=Float32),
        Field(name="tipoextraccion_encoded", dtype=Int32),

        # Features de ventana (clase 3: últimas 10 lecturas por pozo)
        Field(name="avg_prod_gas_10m", dtype=Float32),
        Field(name="avg_prod_pet_10m", dtype=Float32),
        Field(name="last_prod_gas", dtype=Float32),
        Field(name="last_prod_pet", dtype=Float32),
        Field(name="n_readings", dtype=Int32),

        # Features temporales (EDA)
        Field(name="target_lag1", dtype=Float32),
        Field(name="target_lag2", dtype=Float32),
        Field(name="target_lag3", dtype=Float32),
        Field(name="target_rolling_mean_3", dtype=Float32),
        Field(name="target_rolling_mean_6", dtype=Float32),
        Field(name="target_rolling_std_3", dtype=Float32),
        Field(name="months_producing", dtype=Int32),
        Field(name="target_log", dtype=Float32),
    ],
    source=well_stats_source,
)