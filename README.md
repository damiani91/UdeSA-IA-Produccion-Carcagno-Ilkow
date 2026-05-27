# Proyecto IA en Producción — Carcagno · Ilkow

Sistema MLOps end-to-end para **forecasting de producción de petróleo y gas** en pozos argentinos, usando datos públicos de [datos.energia.gob.ar](http://datos.energia.gob.ar). El proyecto está pensado como demostración de un stack productivo completo: feature store, training reproducible, model registry, inferencia escalada, orquestación y monitoreo de drift.

> Materia: IA en Producción — Maestría en Inteligencia Artificial, UdeSA.

---

## Stack

| Componente | Tecnología | Puerto host |
|---|---|---|
| Orquestación | Apache Airflow 2.10 (LocalExecutor) | `8081` |
| Tracking / Registry | MLflow 3.11 + Postgres backend | `5001` |
| Feature Store | Feast (offline=parquet, online=SQLite) | — |
| Inferencia | FastAPI + Ray Serve (4 réplicas / 1 actual) | `8000` |
| Ray Dashboard | Ray | `8265` |
| Metadata DB | PostgreSQL 16 (`mlflow` + `airflow` DBs) | `5432` |
| Drift monitoring | Evidently + PSI/KS custom | — |

---

## Estructura del repositorio

```
.
├── docker-compose.yml          # Orquesta postgres + mlflow + api + airflow
├── docker/
│   ├── airflow.Dockerfile      # Airflow 2.10 + project-venv aislado (SQLAlchemy 2.x)
│   ├── api.Dockerfile          # FastAPI + Ray Serve
│   ├── mlflow.Dockerfile       # MLflow server con Postgres backend
│   ├── training.Dockerfile     # Imagen one-shot para entrenar localmente
│   └── postgres-init/          # Init script (crea la DB airflow)
├── dags/
│   └── monthly_pipeline.py     # DAG mensual (descarga → features → train → drift → promote → reload)
├── src/
│   ├── api/                    # FastAPI + endpoints /forecast, /wells, /admin/*
│   ├── feature_pipeline/       # Ingestión y feature engineering
│   ├── training_pipeline/      # Entrenamiento + registry (Staging→Production)
│   ├── inference_pipeline/     # ForecastService (predict)
│   └── monitoring/             # (espacio reservado)
├── scripts/
│   ├── populate_feature_store.py  # offline → feast apply → online
│   ├── train.py                   # entrypoint local (un solo run)
│   └── generate_drift_report.py   # data + concept drift + model decay
├── feature_store/
│   ├── feature_store.yaml      # Config Feast (provider=local)
│   ├── features.py             # Entity (idpozo) + FeatureView well_stats
│   ├── data/                   # parquet offline + sqlite online (se generan)
│   └── registry/               # Registry de Feast
├── data/
│   ├── raw/                    # produccion.csv + pozos.csv (descargados)
│   └── scripts/download_data.py
├── notebooks/01_eda.ipynb      # EDA inicial
├── requirements.txt
├── .env.example
└── README.md
```

---

## Cómo funciona

### 1. Datos
Dos CSV públicos de la Secretaría de Energía:
- **`produccion.csv`** (~140 MB) — producción mensual por pozo.
- **`pozos.csv`** (~27 MB) — metadata de pozos.

### 2. Feature pipeline
`src/feature_pipeline/` carga los datos en chunks, filtra pozos con `tipoestado == "Extracción Efectiva"` y computa features por pozo:
- Ventanas rolling (`avg_prod_gas_10m`, `avg_prod_pet_10m`, `n_readings`).
- Lags del target (`target_lag1..3`) y rolling means/std.
- Encoding de `tipoextraccion` por orden cronológico de aparición (evita data leakage).
- Salida → `feature_store/data/well_features.parquet` (offline store de Feast).
- `feast apply` registra las definiciones; `write_to_online_store()` materializa la última lectura de cada pozo en SQLite (online store).

### 3. Training pipeline
`src/training_pipeline/train.py`:
- Lee del offline store, hace split temporal (los últimos 3 meses → validación).
- Entrena `GradientBoostingRegressor` (o `RandomForestRegressor`).
- Usa `mlflow.sklearn.autolog()` + métricas custom (`val_mae`, `val_rmse`, `val_mape`).
- Loggea como artefacto un **snapshot del dataset de training** (`reference_dataset.parquet`) que el próximo ciclo usa como baseline para drift.
- Registra el modelo en MLflow Model Registry como `well_production_model`.

`src/training_pipeline/registry.py` promueve la nueva versión a `Production` y archiva la anterior.

### 4. Drift monitoring
`scripts/generate_drift_report.py` se corre **antes** de promover el nuevo modelo:
- **Reference**: snapshot del training del modelo en Production actual.
- **Current**: slice del último mes con target real ya disponible.
- Calcula **PSI + KS por feature** (data drift), **KS sobre residuos** (concept drift), **MAE/MAPE del modelo viejo sobre datos nuevos** (model decay) y delta de feature importances.
- Genera HTML con Evidently + JSON estructurado, y los loggea como artefactos del run nuevo en MLflow.
- **Skip gracioso** si no hay baseline previo (primer ciclo).

### 5. Inferencia
`src/api/`:
- FastAPI montado dentro de un `@serve.deployment` de **Ray Serve**.
- Cada réplica carga el modelo desde `models:/well_production_model/Production` y el parquet de features en memoria.
- Endpoint `POST /admin/reload` dispara un **rolling redeploy** (cada réplica vuelve a cargar el modelo desde MLflow sin downtime).
- Imputa NaNs con la **mediana global** de cada feature (no con 0) para no sesgar.
- Soporta forecast multi-mes con fallback al online store (Feast) si el offline no tiene la fecha pedida.

### 6. Orquestación
DAG `monthly_well_production_pipeline` (`dags/monthly_pipeline.py`) corre el día **10 de cada mes a las 06:00 UTC**:

```
download_data → prepare_offline_store → apply_feast → populate_online_store
              → train_model → generate_drift_report → promote_model → reload_api
```

Tarea crítica de diseño: los pasos de ML corren con `ExternalPythonOperator` en un venv aislado (`/home/airflow/project-venv`) porque el stack (Feast/MLflow/SQLAlchemy 2.x) entra en conflicto con la propia instalación de Airflow (SQLAlchemy 1.4.x).

---

## Quickstart

### Pre-requisitos
- Docker Desktop (con WSL2 en Windows) o Docker Engine en Linux/macOS.
- ~8 GB de RAM libres recomendados para correr todos los servicios.
- Puertos libres: `5432`, `5001`, `8000`, `8081`, `8265`.

### 1. Clonar y configurar `.env`
```bash
git clone <repo-url>
cd UdeSA-IA-Produccion-Carcagno-Ilkow
cp .env.example .env
```

Editá `.env` y generá las claves (no commitees el archivo):

```bash
# Linux/macOS/WSL
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # → AIRFLOW_FERNET_KEY
python -c "import secrets; print(secrets.token_hex(32))"                                   # → AIRFLOW_SECRET_KEY
python -c "import secrets; print(secrets.token_hex(16))"                                   # → RELOAD_SECRET
```

```powershell
# Windows PowerShell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
python -c "import secrets; print(secrets.token_hex(32))"
python -c "import secrets; print(secrets.token_hex(16))"
```

### 2. Build de imágenes
```bash
docker compose build
```

### 3. Inicializar la metadata DB de Airflow (una sola vez)
```bash
docker compose --profile airflow-init up airflow-init
```

Crear el usuario admin de Airflow:
```bash
docker compose run --rm airflow-webserver airflow users create \
    --username admin --password admin \
    --firstname Admin --lastname User --role Admin \
    --email admin@example.com
```

### 4. Levantar el stack
```bash
docker compose up -d postgres mlflow api airflow-webserver airflow-scheduler
```

Verificá que estén sanos:
```bash
docker compose ps
```

### 5. Acceder a las UIs
| Servicio | URL | Credenciales |
|---|---|---|
| API (Swagger) | http://localhost:8000/docs | — |
| Ray Dashboard | http://localhost:8265 | — |
| MLflow | http://localhost:5001 | — |
| Airflow | http://localhost:8081 | `admin` / `admin` |

---

## Primer entrenamiento

La API no responde hasta que exista una versión `Production` del modelo en MLflow. Dos formas:

### Opción A — Disparar el DAG en Airflow (recomendado)
1. Entrar a http://localhost:8081 → habilitar el DAG `monthly_well_production_pipeline`.
2. Trigger manual ("play" → "Trigger DAG").
3. Esperar ~10–20 min (el `download_data` baja ~170 MB).

### Opción B — Run local one-shot (más rápido para iterar)
```bash
docker compose --profile training run --rm training \
    python -m scripts.train --date 2024-12-01
```

Esto: descarga features → entrena → promueve a `Production`. Luego forzá el reload de la API:

```bash
curl -X POST http://localhost:8000/admin/reload \
    -H "X-Reload-Secret: <tu RELOAD_SECRET>"
```

---

## Uso de la API

### Listar pozos activos a una fecha
```bash
curl "http://localhost:8000/api/v1/wells?date_query=2024-12-01"
```

### Forecast por pozo
```bash
curl "http://localhost:8000/api/v1/forecast?id_well=12345&date_start=2025-01-01&date_end=2025-06-01"
```

Respuesta:
```json
{
  "id_well": "12345",
  "data": [
    {"date": "2025-01-01", "prod": 1234.56},
    {"date": "2025-02-01", "prod": 1198.30}
  ]
}
```

### Endpoints administrativos
| Endpoint | Descripción |
|---|---|
| `POST /admin/reload` | Rolling redeploy de las réplicas Ray Serve (requiere header `X-Reload-Secret`) |
| `GET /admin/drift-report` | JSON con métricas de drift del último modelo en Production |
| `GET /admin/drift-report.html` | HTML interactivo de Evidently |

---

## Operación

### Ver logs
```bash
docker compose logs -f api          # FastAPI + Ray Serve
docker compose logs -f airflow-scheduler
docker compose logs -f mlflow
```

### Reiniciar un servicio sin afectar al resto
```bash
docker compose restart api
```

### Apagar todo (preserva volúmenes)
```bash
docker compose down
```

### Reset completo (incluye datos y artefactos)
```bash
docker compose down -v
```

> Cuidado: `-v` borra los volúmenes `postgres_data` y `mlflow_artifacts`. Perdés runs históricos del registry.

### Re-build después de tocar `requirements.txt` o un Dockerfile
```bash
docker compose build --no-cache <servicio>
docker compose up -d <servicio>
```

---

## Variables de entorno (`.env`)

| Variable | Descripción |
|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Credenciales del Postgres compartido por MLflow y Airflow. |
| `AIRFLOW_FERNET_KEY` | Encriptación de conexiones/variables en Airflow. Generar con `Fernet.generate_key()`. |
| `AIRFLOW_SECRET_KEY` | Firma de cookies del webserver de Airflow. |
| `RELOAD_SECRET` | Secreto compartido entre Airflow (`task_reload_api`) y la API (`/admin/reload`). |

---

## Troubleshooting

- **`api` reinicia en loop con OOM** → reducir réplicas (`num_replicas=1` en `src/api/main.py`) o bajar `RAY_NUM_CPUS` en `docker-compose.yml`.
- **`No hay modelo en Production`** → todavía no corrió ningún training. Ver "Primer entrenamiento".
- **`drift-report` devuelve 404** → es el primer ciclo y no hay baseline previo (comportamiento esperado, el log del DAG dice `[drift] skip`).
- **Airflow no encuentra `feast`** → asegurate de haber buildeado la imagen después del último cambio: `docker compose build airflow-webserver airflow-scheduler`.
- **`download_data` falla** → datos.gob.ar suele caerse. Descargar manualmente los CSV y colocarlos en `data/raw/`.

---

## Histórico relevante (últimos commits)

- `1a08845` — fix: Ray Serve OOM crash loop (1 réplica + disable memory monitor)
- `ba6301b` — feat: scale inference with Ray Serve (4 réplicas + dashboard)
- `243c676` — feat: drift monitoring (data, concept, model decay)
- `8fa935f` — fix: force `mlflow-artifacts:/` URI for experiment artifact storage
- `0d0d964` — feat: Airflow orchestration

---

## Licencia y autoría

Trabajo académico — Leandro Carcagno e Ilkow. Datos públicos de la Secretaría de Energía de la Nación Argentina.
