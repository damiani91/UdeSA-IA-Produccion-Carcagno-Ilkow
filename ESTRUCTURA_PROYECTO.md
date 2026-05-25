# Estructura del Proyecto — UdeSA IA en Producción (Carcagno · Ilkow)

Documento explicativo de cada carpeta y archivo del proyecto. Para cada elemento se incluye:
- **Razón de existir** (justificación teórica dentro de la arquitectura MLOps).
- **Descripción del código / contenido** (qué hace internamente).

El proyecto implementa un sistema productivo end-to-end de *forecasting* de producción de pozos de petróleo y gas, siguiendo el patrón clásico de MLOps: **Feature Store (Feast)** + **Tracking & Registry (MLflow)** + **API de inferencia (FastAPI)**, todo orquestado con **Docker Compose**.

---

## Raíz del repositorio

- **`README.md`**
  - *Razón de existir*: punto de entrada documental del repositorio (convención de GitHub).
  - *Contenido*: actualmente sólo contiene el título del proyecto. Sirve como marcador del repo.

- **`.gitignore`**
  - *Razón de existir*: evitar versionar artefactos pesados o sensibles (datos crudos, modelos, secretos, cachés).
  - *Contenido*: excluye `data/raw/`, `data/processed/`, `*.csv`, `*.parquet` (datos), `__pycache__/`, `.venv/` (Python), `mlruns/`, `mlartifacts/` (MLflow), `.env` (secretos), `docker-volumes/` (volúmenes Docker), carpetas de IDE, `.DS_Store`, registros de Feast y checkpoints de notebooks.

- **`.env.example`**
  - *Razón de existir*: plantilla de variables de entorno (12-factor app). El `.env` real no se versiona.
  - *Contenido*: define `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` para el backend Postgres de MLflow.

- **`requirements.txt`**
  - *Razón de existir*: fijar versiones exactas (pinning) de dependencias Python para reproducibilidad.
  - *Contenido*: listado pinneado de librerías. Las más relevantes:
    - `fastapi`, `uvicorn`, `gunicorn`: servidor API de inferencia.
    - `mlflow==3.11.1`: tracking, registry y model serving.
    - `feast==0.62.0`: feature store offline/online.
    - `pandas`, `numpy`, `pyarrow`: manipulación de datos y soporte parquet.
    - `scikit-learn==1.8.0`: modelos (GradientBoosting, RandomForest).
    - `pydantic==2.12.5`: validación de schemas de la API.
    - `SQLAlchemy`, `dask`, `tenacity`, `tqdm`, `Jinja2`, etc.: utilidades auxiliares.

- **`docker-compose.yml`**
  - *Razón de existir*: orquestar los servicios del stack MLOps (Postgres + MLflow + API + Training) declarativamente. Permite levantar el sistema completo con un solo comando.
  - *Contenido*:
    - **`postgres`** (`postgres:16-alpine`): backend de metadatos para MLflow. Persiste en el volumen `postgres_data`. Expone `5432`. Healthcheck con `pg_isready`.
    - **`mlflow`**: construido desde `docker/mlflow.Dockerfile`. Depende de `postgres` (healthy). Expone `5001:5000`. Comando `mlflow server` con `--backend-store-uri` apuntando a Postgres y `--default-artifact-root /mlflow/artifacts`. Healthcheck contra `/health`.
    - **`api`**: construido desde `docker/api.Dockerfile`. Variable `MLFLOW_TRACKING_URI=http://mlflow:5000`. Monta `./feature_store` y el volumen de artefactos de MLflow. Expone `8000`.
    - **`training`**: construido desde `docker/training.Dockerfile`. Profile `training` (no se levanta por defecto, se invoca a demanda). Monta `./data`, `./feature_store`, `./.git:ro` y artefactos de MLflow.
    - **Volúmenes**: `postgres_data`, `mlflow_artifacts` (compartidos entre `mlflow`, `api` y `training` para que la API pueda cargar el modelo desde el mismo path donde MLflow lo guardó).

- **`Consigna.docx`**
  - *Razón de existir*: enunciado del trabajo práctico final de la materia. Material de referencia.
  - *Contenido*: documento Word con los requerimientos académicos del trabajo (no es parte del sistema).

- **`PLAN_ENTREGA_FINAL.md`**
  - *Razón de existir*: planificación interna del equipo para la entrega.
  - *Contenido*: notas de trabajo sobre tareas pendientes, decisiones de diseño y cronograma.

- **`ESTRUCTURA_PROYECTO.md`** (este archivo)
  - *Razón de existir*: documentar la arquitectura y propósito de cada componente.

---

## `docker/` — Imágenes Docker

Contiene los Dockerfiles que construyen las imágenes para cada servicio.

- **`docker/api.Dockerfile`**
  - *Razón de existir*: imagen reproducible para servir el modelo vía FastAPI.
  - *Contenido*:
    1. Base `python:3.11-slim`.
    2. `WORKDIR /app`.
    3. Copia e instala `requirements.txt` (`pip install --no-cache-dir`).
    4. Copia el código de `src/` y el repo de `feature_store/`.
    5. Setea `PYTHONPATH=/app` para que `src.*` sea importable.
    6. Lanza `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.

- **`docker/mlflow.Dockerfile`**
  - *Razón de existir*: imagen minimalista del servidor MLflow.
  - *Contenido*:
    1. Base `python:3.11-slim`.
    2. Instala `mlflow==3.11.1` y `psycopg2-binary` (driver Postgres).
    3. Expone `5000`.
    4. `ENTRYPOINT ["mlflow"]` (el comando concreto `server ...` viene del compose).

- **`docker/training.Dockerfile`**
  - *Razón de existir*: imagen para el job de entrenamiento (batch). Incluye `git` (necesario por las dependencias y para `mlflow` poder loggear el commit hash).
  - *Contenido*:
    1. Base `python:3.11-slim`.
    2. `apt-get install git`.
    3. Instala `requirements.txt`.
    4. Copia `src/`, `scripts/`, `feature_store/`.
    5. Setea `PYTHONPATH=/app`.
    6. Comando por defecto: `python -m scripts.train --date 2024-12-01`.

---

## `data/` — Datos crudos y scripts de descarga

Carpeta destinada a los datasets de producción de pozos. Los CSV crudos están en `.gitignore` por su tamaño.

- **`data/scripts/download_data.py`**
  - *Razón de existir*: automatizar la descarga de los datasets públicos del portal `datos.energia.gob.ar` para garantizar reproducibilidad.
  - *Contenido*:
    - Constante `URLS`: diccionario con los dos datasets (`produccion` y `pozos`) y sus URLs en `datos.energia.gob.ar`.
    - `RAW_DIR`: ruta absoluta calculada a `<repo>/data/raw/`.
    - `download_file(url, dest)`: hace `requests.get(stream=True, timeout=120)`, crea la carpeta destino, escribe el archivo en chunks de 8192 bytes y reporta el tamaño en MB.
    - `__main__`: itera sobre las URLs, omite los archivos que ya existen y captura excepciones para no abortar la descarga de un dataset si falla el otro.

- **`data/raw/`** (no versionada): destino de los CSVs descargados (`produccion.csv`, `pozos.csv`).
- **`data/processed/`** (no versionada): salidas intermedias procesadas.

---

## `feature_store/` — Repo de Feast

Implementa el **Feature Store** con Feast: define entidades, sources y feature views; mantiene el registry y la base online local.

- **`feature_store/feature_store.yaml`**
  - *Razón de existir*: archivo de configuración requerido por Feast. Declara el proyecto, el proveedor y los stores offline/online.
  - *Contenido*:
    - `project: oil_gas_production`.
    - `registry: data/registry.db` (registro de metadatos Feast).
    - `provider: local` (todo corre localmente, sin GCP/AWS).
    - `offline_store: file` (parquet en disco).
    - `online_store: sqlite` con `path: data/online.db` (BD ligera para servir features con baja latencia).

- **`feature_store/features.py`**
  - *Razón de existir*: declarar el esquema lógico del Feature Store (Entity + Source + FeatureView). Es el contrato entre el pipeline de features y los consumidores (training & inference).
  - *Contenido*:
    - `pozo = Entity(name="idpozo", ...)`: identifica unívocamente cada pozo.
    - `well_stats_source = FileSource(path=".../well_features.parquet", timestamp_field="fecha")`: source offline en parquet.
    - `well_stats = FeatureView(name="well_stats", entities=[pozo], source=well_stats_source, schema=[...])` con los siguientes campos:
      - Targets: `prod_gas`, `prod_pet`, `target` (Float32).
      - Atributos del pozo: `prod_agua`, `tef`, `profundidad`, `tipoextraccion_encoded`.
      - Features de ventana (últimas 10 lecturas): `avg_prod_gas_10m`, `avg_prod_pet_10m`, `last_prod_gas`, `last_prod_pet`, `n_readings`.
      - Features temporales (lags y rollings): `target_lag1..3`, `target_rolling_mean_3`, `target_rolling_mean_6`, `target_rolling_std_3`, `months_producing`, `target_log`.

- **`feature_store/data/online.db`** (SQLite): online store materializado. Generado por `feast materialize` / `write_to_online_store`. Sirve para lookups de baja latencia desde la API.
- **`feature_store/data/registry.db`** y **`feature_store/registry/registry.db`**: registry binario de Feast con los metadatos del proyecto. Generado por `feast apply`.

---

## `notebooks/` — Análisis exploratorio

- **`notebooks/01_eda.ipynb`**
  - *Razón de existir*: análisis exploratorio (EDA) inicial sobre los datos crudos. Documenta hallazgos que motivaron las decisiones de feature engineering (qué columnas eliminar, qué lags incluir, qué transformaciones aplicar como `log1p`, etc.).
  - *Contenido*: notebook Jupyter con celdas de carga del CSV de producción, estadísticas descriptivas, visualizaciones de la distribución temporal por pozo, análisis de missingness, y motivación de las features temporales.

---

## `scripts/` — Entrypoints batch

Scripts ejecutables que orquestan los pipelines completos. Aislados de `src/` para distinguir lógica reusable (en `src/`) de comandos (en `scripts/`).

- **`scripts/__init__.py`**: marcador de paquete Python.

- **`scripts/populate_feature_store.py`**
  - *Razón de existir*: pipeline de poblamiento del Feature Store. Ejecuta el patrón estándar de Feast: cómputo de features → escritura offline (parquet) → `feast apply` → materialización online.
  - *Contenido*:
    - Constantes de paths: `ROOT`, `FEATURE_STORE_REPO`, `PARQUET_PATH`, `PROD_FILE`.
    - `prepare_offline_store(up_to_date=None)`:
      1. Llama a `load_production_data(PROD_FILE)`.
      2. Filtra sólo pozos `tipoestado == "Extracción Efectiva"` y `target` no nulo.
      3. Si se pasa `up_to_date`, recorta el dataframe hasta esa fecha (para evitar leakage en training).
      4. Llama a `compute_features(df)` y guarda el resultado como parquet.
    - `apply_feast()`: ejecuta `subprocess.run(["feast", "apply"], cwd=FEATURE_STORE_REPO, check=True)`.
    - `populate_online_store()`:
      1. Lee el parquet generado.
      2. Toma la última lectura por pozo (`sort_values("fecha").groupby("idpozo").tail(1)`).
      3. Crea `FeatureStore(repo_path=...)` y llama `store.write_to_online_store(feature_view_name="well_stats", df=latest_df)`.
      4. Valida leyendo `get_online_features(...)` para un pozo de muestra.
    - `__main__`: encadena las tres funciones.

- **`scripts/train.py`**
  - *Razón de existir*: entrypoint del pipeline completo de entrenamiento. Encadena Feature Pipeline → Training Pipeline → promoción a Production en el Model Registry.
  - *Contenido*:
    - Inserta la raíz del proyecto en `sys.path`.
    - `argparse` con `--date` (obligatoria) y `--raw-data` (default `data/raw/produccion.csv`).
    - Fase 1 — Feature Pipeline: importa y ejecuta `prepare_offline_store(up_to_date=args.date)`, `apply_feast()`, `populate_online_store()`.
    - Fase 2 — Training Pipeline: llama `train_model(training_date=args.date)` y obtiene `run_id`.
    - Fase 3 — Registry: si hay `run_id` válido, llama `promote_model_to_production(run_id=run_id)`. Caso contrario reporta fallo.

---

## `src/` — Código fuente reutilizable

Paquete Python con la lógica de negocio organizada por pipeline.

- **`src/__init__.py`**: marcador de paquete.

### `src/feature_pipeline/` — Ingesta y feature engineering

- **`__init__.py`**: marcador.

- **`ingestion.py`**
  - *Razón de existir*: capa de I/O y limpieza inicial. Aísla el resto del pipeline de cambios en el formato del CSV.
  - *Contenido*:
    - Constante `DROP_COLS` con columnas irrelevantes (`vida_util`, `observaciones`, `rectificado`, `habilitado`, `idusuario`).
    - `load_production_data(path, chunksize=100000)`:
      - Lee el CSV con `pd.read_csv(..., chunksize=...)` para evitar OOM en datasets grandes.
      - Para cada chunk: descarta `DROP_COLS`, construye `fecha` a partir de `anio` + `mes` con padding `zfill(2)`, calcula `target` según `tipopozo` (gas para `Gasífero`, petróleo para `Petrolífero`/`Petrolero`, NaN en otro caso), y convierte columnas numéricas con `errors="coerce"`.
      - Concatena los chunks y ordena por `(idpozo, fecha)`.
    - `load_wells_data(path)`: lectura simple del CSV de pozos.

- **`feature_engineering.py`**
  - *Razón de existir*: cálculo determinístico de features. Toda la lógica de derivación de variables vive acá para mantener consistencia entre training (offline) e inference (online).
  - *Contenido*:
    - `compute_features(df, window=6)`:
      1. Ordena por `fecha` y `idpozo`.
      2. Codifica `tipoextraccion` con `pd.factorize` sobre el dataframe ya ordenado cronológicamente, así el ID depende sólo de la primera aparición (evita data leakage por categorías futuras).
      3. Reordena por `(idpozo, fecha)` para las ventanas por pozo.
      4. `grouped = df.groupby("idpozo", sort=False)` y aplica `transform` con `rolling(window, min_periods=1)` para `avg_prod_gas_10m`, `avg_prod_pet_10m`, y `count` para `n_readings`. Esto evita un loop O(n²).
      5. `last_prod_gas`, `last_prod_pet`: valores instantáneos.
      6. Lags 1, 2, 3 del `target` con `grouped["target"].shift(lag)`.
      7. Rollings del target: media 3, media 6, std 3.
      8. `months_producing = grouped.cumcount() + 1`.
      9. `target_log = log1p(clip(target, 0))`.
      10. Cast de `tef` y `profundidad` con `to_numeric(..., errors="coerce")`.

- **`fix_features.py`**
  - *Razón de existir*: variante previa / "patch" de `feature_engineering.py`. Conserva la misma lógica pero sin algunos `.astype(float)` y con `groupby("idpozo")` sin `sort=False`. Funcionalmente equivalente; histórico.
  - *Contenido*: prácticamente idéntico a `feature_engineering.compute_features` salvo detalles de casteo.

### `src/training_pipeline/` — Entrenamiento y registry

- **`__init__.py`**: marcador.

- **`train.py`**
  - *Razón de existir*: entrenamiento del modelo, integrado con MLflow (tracking + registry).
  - *Contenido*:
    - Constantes: `FEATURE_STORE_REPO`, `PARQUET_PATH`, `FEAST_FEATURES` (lista de referencias `well_stats:<feature>`), `FEATURE_COLS` (sin el prefijo), `TARGET = "target"`.
    - `train_model(training_date, mlflow_tracking_uri="http://mlflow:5000", experiment_name="well_production_forecast", model_type="gradient_boosting", model_params=None)`:
      1. Setea tracking URI y experimento.
      2. Lee `well_features.parquet`, filtra por `fecha <= training_date`.
      3. Construye `training_df` con `idpozo`, `event_timestamp` (renombre de `fecha`, UTC), `target` y `FEATURE_COLS`. Dropea NaN/inf.
      4. Split temporal: `cutoff = training_date - 3 meses`; antes → train, después → val.
      5. `mlflow.sklearn.autolog(log_models=True)`.
      6. Dentro de `with mlflow.start_run() as run:`:
         - `log_param`: `training_date`, `model_type`, `n_wells`, `n_samples_train`, `n_samples_val`, `features`.
         - Instancia `RandomForestRegressor` o `GradientBoostingRegressor(learning_rate=0.1, ...)` según `model_type`.
         - Defaults `{"n_estimators": 200, "max_depth": 5, "random_state": 42}`.
         - `model.fit(X_train, y_train)`.
         - `log_metric` para `val_mae`, `val_rmse`, y `val_mape` (excluyendo y_val == 0).
         - `mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="well_production_model")` (registra en Model Registry).
         - Genera y loggea `feature_importances.png` con `matplotlib` (backend `Agg`).
         - Retorna `run.info.run_id`.
      7. Manejo de excepciones: imprime el error y llama `mlflow.end_run(status="FAILED")` si hay run abierto.

- **`registry.py`**
  - *Razón de existir*: gestión del ciclo de vida del modelo en el MLflow Model Registry (stages `None → Staging → Production → Archived`).
  - *Contenido*:
    - `promote_model_to_production(model_name="well_production_model", run_id=None, mlflow_tracking_uri=...)`:
      1. Crea `MlflowClient`.
      2. Si se pasa `run_id`, busca la versión correspondiente con `search_model_versions`. Si no, toma la última con `get_latest_versions`.
      3. `client.transition_model_version_stage(name, version, stage="Production", archive_existing_versions=True)` — promueve archivando versiones anteriores (estrategia "one Production at a time" que permite rollback).

### `src/inference_pipeline/` — Servicio de predicción

- **`__init__.py`**: marcador.

- **`predict.py`**
  - *Razón de existir*: encapsula la lógica de inferencia en una clase reutilizable. Lee features del **online store** (latencia baja) con fallback al offline store para el caso autoregresivo (forecast de varias fechas futuras).
  - *Contenido*:
    - `FEAST_FEATURES` y `FEATURE_COLS`: mismo listado que en training (garantiza consistencia).
    - `class ForecastService`:
      - `__init__(mlflow_tracking_uri="http://mlflow:5000")`:
        - Setea tracking URI.
        - `self.model = mlflow.sklearn.load_model("models:/well_production_model/Production")` (carga la versión Production).
        - `self.store = FeatureStore(repo_path="feature_store")`.
        - Carga `well_features.parquet` como `self.features_df`.
        - Precalcula `self.feature_medians` (mediana global por columna) para imputar nulos sin sesgar a 0.
      - `predict(id_well, date_start, date_end) -> list[dict]`:
        1. Convierte `id_well` a int.
        2. Llama `self.store.get_online_features(features=FEAST_FEATURES, entity_rows=[{"idpozo": idpozo}])` y lo lleva a DataFrame.
        3. `dates = pd.date_range(date_start, date_end, freq="MS")` (inicio de cada mes).
        4. Para cada `target_date`:
           - Busca primero en el offline store (más completo, con la fecha exacta).
           - Si no hay, usa los features del online store como fallback.
           - Si tampoco, devuelve `prod=0.0`.
           - Imputa nulos con `self.feature_medians`, reemplaza inf/NaN con `np.nan_to_num`.
           - `pred = model.predict(X)[0]`, lo clipea a no-negativo y lo redondea a 2 decimales.
        5. Devuelve lista de `{"date": "YYYY-MM-DD", "prod": float}`.

### `src/api/` — Capa REST (FastAPI)

- **`main.py`**
  - *Razón de existir*: bootstrap de la aplicación FastAPI. Carga modelo y features una sola vez al arrancar (lifespan), no por request.
  - *Contenido*:
    - `lifespan` (asynccontextmanager): al startup instancia `app.state.forecast_service = ForecastService()` y carga `app.state.features_df = pd.read_parquet("feature_store/data/well_features.parquet")`. `yield` mantiene la app corriendo.
    - `app = FastAPI(title="Oil & Gas Forecast API", version="1.0.0", lifespan=lifespan)`.
    - `app.include_router(forecast.router, prefix="/api/v1")` y lo mismo para `wells.router`.

- **`forecast.py`**
  - *Razón de existir*: endpoint REST que expone el forecast del modelo.
  - *Contenido*:
    - `router = APIRouter()`.
    - `GET /forecast` con query params `id_well`, `date_start` (YYYY-MM-DD) y `date_end` (YYYY-MM-DD).
    - Llama `request.app.state.forecast_service.predict(id_well, date_start, date_end)` y devuelve un `ForecastResponse`.

- **`wells.py`**
  - *Razón de existir*: endpoint para listar pozos activos hasta una fecha dada (catálogo para la UI).
  - *Contenido*:
    - `GET /wells?date_query=...`. Filtra `features_df` por `fecha <= date_query`, toma `idpozo.unique()` ordenado, y devuelve `list[WellInfo]`.

- **`schemas.py`**
  - *Razón de existir*: contratos Pydantic para validación de entrada/salida y generación automática de OpenAPI.
  - *Contenido*:
    - `ForecastPoint(BaseModel)`: `date: str`, `prod: float`.
    - `ForecastResponse(BaseModel)`: `id_well: str`, `data: list[ForecastPoint]`.
    - `WellInfo(BaseModel)`: `id_well: str`.

---

## Flujo end-to-end (referencia)

1. **Descarga de datos** → `python data/scripts/download_data.py` → CSVs en `data/raw/`.
2. **Pipeline de entrenamiento** → `docker compose --profile training up training` (o `python -m scripts.train --date YYYY-MM-DD`):
   - `prepare_offline_store` lee el CSV, computa features y los guarda en `feature_store/data/well_features.parquet`.
   - `apply_feast` registra las definiciones en `registry.db`.
   - `populate_online_store` materializa las últimas lecturas al `online.db`.
   - `train_model` entrena con `GradientBoosting`, loggea en MLflow y registra el modelo.
   - `promote_model_to_production` lo lleva al stage Production.
3. **Inferencia** → `docker compose up api` → `GET /api/v1/forecast?id_well=...&date_start=...&date_end=...`:
   - `ForecastService` carga el modelo Production y consulta el online/offline store.
   - Devuelve la curva de producción mensual prevista.
