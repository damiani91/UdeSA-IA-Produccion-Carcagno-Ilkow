# Proyecto IA en Producción — Carcagno · Ilkow

Pipeline de puesta en producción para un modelo de pronóstico de producción
mensual de pozos de hidrocarburos (gas y petróleo). El sistema cubre ingestión,
feature store, entrenamiento reproducible con tracking, registry de modelos,
orquestación recurrente, monitoreo de drift/decay y serving escalable.

> **Video demo (6'):** _TBD — pegar acá el link al video antes de la entrega._

---

## 1. Cómo levantar todo

Requisitos: Docker + Docker Compose, `data/raw/produccion.csv` presente.

```bash
cp .env.example .env
docker compose up -d --build
```

Puertos expuestos:

| Servicio          | URL                              | Descripción                          |
| ----------------- | -------------------------------- | ------------------------------------ |
| API (Ray Serve)   | http://localhost:8000            | FastAPI servido por Ray Serve        |
| Ray dashboard     | http://localhost:8265            | Réplicas, throughput, logs           |
| MLflow            | http://localhost:5001            | Experimentos + Model Registry        |
| Airflow Webserver | http://localhost:8080            | DAGs (user/pass: `admin`/`admin`)    |
| Postgres (MLflow) | localhost:5432                   | Metastore de MLflow                  |

Los servicios `airflow-init`, `airflow-webserver` y `airflow-scheduler`
arrancan automáticamente con `docker compose up`. La primera vez la migración
de la metastore de Airflow puede tardar ~1 minuto.

---

## 2. Cómo correr un entrenamiento manual

El comando único de la entrega parcial sigue funcionando como atajo (no
requiere Airflow):

```bash
# Desde el host (perfil training del compose)
docker compose --profile training up training

# O contra una fecha específica:
docker compose run --rm training python -m scripts.train --date 2024-12-01
```

Este flujo ejecuta `populate → train → promote a Production` en un solo paso.

---

## 3. Cómo correr el DAG de Airflow

1. Abrir http://localhost:8080 e ingresar con `admin` / `admin`.
2. Habilitar el DAG `training_dag` (debería aparecer activo por defecto).
3. Click en **Trigger DAG** (▶) para forzar una corrida.

El DAG corre `@monthly` (`catchup=False`). Sus tasks en orden son:

```
prepare_offline_store → apply_feast → populate_online_store
        → train_model → promote_model → drift_report
```

- `train_model` pushea su `run_id` por XCom; `promote_model` lo consume y
  transiciona esa versión a stage **Production** archivando las anteriores.
- `drift_report` genera el JSON de drift+decay y lo loguea como artefacto a
  MLflow (experimento `well_production_drift`).

---

## 4. Cómo consumir la API

Listar pozos activos a una fecha:
```bash
curl "http://localhost:8000/api/v1/wells?date_query=2024-12-01"
```

Pronóstico para un pozo y rango de meses:
```bash
curl "http://localhost:8000/api/v1/forecast?id_well=12345&date_start=2025-01-01&date_end=2025-06-01"
```

Reporte de drift + decay (último disponible, o por fecha):
```bash
curl http://localhost:8000/api/v1/monitoring/report
curl "http://localhost:8000/api/v1/monitoring/report?date=2024-12-01"
```

Spec OpenAPI: http://localhost:8000/docs

---

## 5. Aspectos de diseño

- **Feast (online + offline store):** el offline store en parquet alimenta el
  entrenamiento (`get_historical_features` análogo via lectura directa del
  parquet para reproducibilidad temporal); el online store en SQLite sirve
  features de baja latencia a la API. Esa separación permite reentrenar con
  cualquier `training_date` sin pisar la versión productiva del online store.

- **MLflow Model Registry con stages:** cada entrenamiento queda como un `run`
  con métricas y modelo; `promote_model_to_production` mueve la versión a
  *Production* y archiva las anteriores (rollback explícito disponible). La
  API carga siempre `models:/well_production_model/Production`, así un nuevo
  deploy es transparente.

- **Airflow `@monthly` + LocalExecutor:** el dataset público se actualiza
  mensualmente. Usamos `LocalExecutor` con Postgres como metastore para tener
  paralelismo real sin overhead de un cluster. Las tasks son `PythonOperator`
  que importan funciones de `src/` y `scripts/` (la imagen `docker/airflow.Dockerfile`
  ya trae las dependencias del proyecto), evitando Docker-in-Docker.

- **Detección de drift:**
  - **Data drift:** **PSI por feature** (Population Stability Index) con bins
    cuantílicos sobre la ventana de referencia (últimos 12 meses) y la actual
    (último mes). Umbral fuerte 0.25.
  - **Model decay:** MAE del modelo Production sobre el último mes con `y`
    real, comparado contra `val_mae` baseline del run productivo. Se marca
    decay si `MAE_actual > 1.5 × MAE_validación`.

- **Escalabilidad con Ray Serve:** el deployment `ForecastDeployment` envuelve
  la app FastAPI con `@serve.ingress` manteniendo intacta la spec OpenAPI.
  Configurable vía `SERVE_NUM_REPLICAS` (default 2). Cada réplica carga su
  propia copia del modelo Production y del parquet de features, por lo que
  escala horizontalmente sin estado compartido. El Ray dashboard
  (puerto 8265) expone réplicas activas, latencia y logs.

---

## 6. Trade-offs y limitaciones

- **Online store en SQLite**: simple para entrega académica, pero no soporta
  escritura concurrente desde múltiples writers. Para producción real se usaría
  Redis/DynamoDB.

- **Predicciones para drift report**: usamos el modelo Production sobre el
  parquet en lugar de predicciones reales servidas por la API. Es equivalente
  a "qué hubiera predicho el modelo para esos pozos", pero asume que la
  distribución de inputs a la API matchea la de los registros del feature
  store. Para producción habría que persistir las predicciones servidas
  (event log) y compararlas contra el `y` real cuando aparezca.

- **Cada réplica de Ray Serve carga su propio modelo + parquet** (~unos MB).
  No es óptimo en memoria, pero evita el bottleneck de un Object Store y es
  más fácil de razonar para esta entrega. Para modelos pesados conviene
  mover el modelo a un actor compartido con `serve.get_replica_context()`.

- **Airflow corriendo en el mismo host que la API**: ambos comparten el
  volumen `feature_store/`. Si el DAG escribe el parquet mientras la API lo
  está leyendo, puede haber lectura inconsistente puntual. Mitigación posible:
  escribir a un parquet temporal y renombrarlo atómicamente.

- **Sin autenticación en la API ni en Airflow** más allá del basic auth por
  default de Airflow. Para producción habría que poner un proxy con OAuth y
  rotar credenciales por secrets manager.

- **No hay tests automatizados de regresión** sobre las métricas del modelo
  (sólo umbrales de drift). Una mejora natural es agregar un task del DAG
  que bloquee la promoción si `val_mae` empeora más de un porcentaje fijado.
