# Plan de Implementación — Entrega Final (28/05)

> **Objetivo de este documento:** servir de hoja de ruta paso a paso para completar la
> segunda entrega del trabajo integrador de "IA en Producción". Está escrito de modo
> que pueda leerse de corrido, incluso por alguien que no participó en la primera
> entrega, e incluye explicaciones teóricas además de las tareas concretas.

---

## 0. Contexto: dónde estamos parados

### 0.1 ¿Qué pide la consigna?
El trabajo consiste en armar el pipeline de **puesta en producción** de un modelo de
ML que pronostica la producción de hidrocarburos (gas y petróleo) por pozo a partir
del dataset público de datos.gob.ar (producción de pozos no convencionales).

La consigna se divide en dos entregas:

**Entrega parcial (16/4)** — ya cumplida:
- Sistema levantable con `docker-compose`.
- API REST con la spec OpenAPI (endpoints `/forecast` y `/wells`).
- Tracking de experimentos con MLflow (parámetros, métricas, artefactos, registry).
- Feature Store (Feast) poblado por una pipeline.
- Entrenamiento reproducible con un solo comando para una fecha dada.

**Entrega final (28/5)** — lo que cubre este plan:
1. **Orquestación**: entrenamiento y despliegue **recurrente y automático** (Airflow).
2. **Monitoreo**: reporte de **model decay** + **data/concept drift** con ≥2 métricas.
3. **Escalabilidad de inferencia**: arquitectura escalable para la API (Ray Serve).
4. Además: trabajar con PRs prolijos, commits distribuidos, README actualizado y
   un video demo de 6 minutos.

### 0.2 ¿Qué hay ya implementado? (estado del repo)

Estructura actual:
```
src/
  api/                  → FastAPI con /forecast y /wells (main, schemas, routers)
  feature_pipeline/     → ingestion + feature_engineering (lags, rolling, etc.)
  training_pipeline/    → train.py (MLflow autolog + registry) + registry.py
  inference_pipeline/   → predict.py (carga modelo de MLflow "Production" + Feast online)
feature_store/
  feature_store.yaml    → Feast: offline=file, online=sqlite
  features.py           → Entity idpozo + FeatureView well_stats
  data/                 → parquet offline + sqlite online
scripts/
  populate_feature_store.py  → ingestion + feast apply + write_to_online_store
  train.py                   → comando único: features + train + promote a Production
docker/
  api.Dockerfile, mlflow.Dockerfile, training.Dockerfile
docker-compose.yml      → postgres + mlflow + api (+ training como profile)
```

**Flujo end-to-end actual** (entrega parcial):
1. `python -m scripts.train --date YYYY-MM-DD` o `docker compose --profile training up training`
2. Lee `produccion.csv` crudo, filtra hasta esa fecha, computa features, los persiste
   como parquet (offline store), corre `feast apply`, materializa al online store
   (sqlite).
3. Entrena un `GradientBoostingRegressor`, loguea todo en MLflow, registra el modelo
   y lo promueve a stage *Production*.
4. La API (`uvicorn`) carga el modelo "Production" desde MLflow + features online
   desde Feast y sirve `/api/v1/forecast` y `/api/v1/wells`.

### 0.3 ¿Qué hay que agregar?

| Bloque | Qué falta | Stack sugerido |
| --- | --- | --- |
| Orquestación | Correr el flujo `populate → train → promote` de forma recurrente | **Airflow** (la consigna lo sugiere) |
| Monitoreo / Drift | Job + endpoint que reporte degradación del modelo y drift de datos | **Evidently** o reportes propios |
| Escalabilidad inferencia | Servir la inferencia detrás de un cluster de workers | **Ray Serve** (la consigna lo sugiere) |
| Documentación | README detallado + video demo 6' | — |

---

## 1. Marco teórico mínimo (para no perderse)

### 1.1 ¿Qué es la orquestación y por qué Airflow?
En un sistema de ML productivo, no alcanza con poder correr el pipeline a mano. Hay
que correrlo **automáticamente, en horarios fijos, con reintentos, alertas y un
registro de qué corrió, cuándo y cómo terminó**. Eso es *orquestación*.

**Airflow** es la herramienta de facto: representa un pipeline como un **DAG** (grafo
dirigido acíclico) cuyos nodos son `tasks`. Cada DAG tiene un `schedule_interval`
(p.ej. `@monthly`) y mantiene el historial de cada ejecución (`DagRun`). Si una
tarea falla, Airflow puede reintentar, mandar email, etc.

El dataset de producción de pozos se actualiza con **frecuencia mensual** (chequear
en el portal), por lo cual el DAG va a correr `@monthly`.

### 1.2 ¿Qué es model decay, data drift y concept drift?
Un modelo entrenado con datos del pasado deja de funcionar bien cuando el mundo
cambia. Conviene distinguir tres conceptos:

- **Data drift** (también llamado *covariate shift*): cambia la distribución de las
  *features* `P(X)`. Ej.: aparecen pozos en yacimientos nuevos, cambian rangos de
  profundidad, etc. Se detecta comparando la distribución de las features actuales
  contra la distribución del set de entrenamiento. Métricas típicas:
  - **PSI (Population Stability Index)** — score continuo, >0.25 indica drift fuerte.
  - **KS test** (Kolmogorov–Smirnov) — test estadístico para variables continuas.
  - **Jensen–Shannon divergence** — divergencia entre distribuciones.

- **Concept drift**: cambia la relación entre features y target `P(Y|X)`. Las mismas
  features producen targets distintos (ej.: un pozo se acidifica y empieza a producir
  más con las mismas condiciones operativas). Sólo es observable cuando ya tenemos
  el `y` real.

- **Model decay**: la performance del modelo en producción se degrada con el tiempo
  (es la consecuencia de los dos anteriores). Se mide volviendo a calcular **MAE /
  RMSE / MAPE** sobre los datos reales del mes vencido contra las predicciones que
  el modelo había hecho.

Para este trabajo nos alcanza con reportar ≥2 métricas. Vamos a elegir:
1. **PSI por feature** (data drift).
2. **MAE en ventana móvil de los últimos N meses** comparado contra el MAE de
   validación (model decay).

### 1.3 ¿Qué es Ray Serve y por qué no alcanza con FastAPI?
FastAPI + uvicorn corre en un solo proceso (o N workers gunicorn) en una sola
máquina. Si el tráfico de inferencia crece, no puede *escalar horizontalmente* sin
reinvención. **Ray Serve** es un framework de serving sobre **Ray** (un runtime
distribuido). Permite:
- Definir el modelo como un *deployment* declarando `num_replicas` y recursos.
- Autoscaling y batching automático (junta varios pedidos para hacer una inferencia
  vectorizada y aumentar throughput).
- Integrarse con FastAPI (`@serve.ingress(app)`), por lo que mantenemos la misma API
  REST que ya valida la spec OpenAPI.

---

## 2. Plan de trabajo: cronograma sugerido

El orden está pensado para que cada bloque sea *independientemente verificable* y
los PRs sean chicos y revisables.

```
Sprint 1: Airflow (orquestación)        ── PR #A
Sprint 2: Monitoreo de drift / decay    ── PR #B
Sprint 3: Ray Serve (escalabilidad)     ── PR #C
Sprint 4: README + Video demo           ── PR #D
```

Cada sprint es un PR distinto. Asegurar que ambos integrantes hagan commits
visibles en distintos PRs (la consigna lo evalúa explícitamente).

---

## 3. Sprint 1 — Orquestación con Airflow

### 3.1 Decisiones de diseño
- **Modo de Airflow**: `LocalExecutor` con `postgres` como metastore. Es lo más
  liviano que sigue siendo realista; un `SequentialExecutor` con SQLite no soporta
  paralelismo y se ve "de juguete".
- **Cómo se ejecuta el código del pipeline**: dos opciones:
  - (a) **PythonOperator** importando las funciones de `scripts.train`.
  - (b) **DockerOperator / BashOperator** corriendo el contenedor `training`.
  Vamos a usar **(a)** porque ya tenemos el código modular y evita el problema de
  Docker-in-Docker. Pero significa que las dependencias del proyecto deben estar
  instaladas en la imagen de Airflow → vamos a construir una imagen custom.
- **Donde se monta el feature_store y mlflow**: volúmenes compartidos entre el
  worker de Airflow y el contenedor de MLflow (igual que hoy con `training`).

### 3.2 Pasos
1. **Crear el servicio Airflow en `docker-compose.yml`**
   - Agregar servicios: `airflow-postgres` (otra DB para metastore), `airflow-init`
     (corre `airflow db migrate` y crea usuario), `airflow-webserver`, `airflow-scheduler`.
   - Volúmenes: `./airflow/dags:/opt/airflow/dags`, `./feature_store`, `./data`,
     `mlflow_artifacts`.
   - Exponer webserver en `8080`.
   - Variables de entorno: `AIRFLOW__CORE__EXECUTOR=LocalExecutor`,
     `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://...`,
     `MLFLOW_TRACKING_URI=http://mlflow:5000`.

2. **Crear `docker/airflow.Dockerfile`**
   - `FROM apache/airflow:2.10.x-python3.11`.
   - Instalar `requirements.txt` del proyecto encima (para que el DAG pueda hacer
     `from src.training_pipeline.train import train_model`).
   - Copiar `src/` y `scripts/` a `/opt/airflow/project/` y agregarlo al `PYTHONPATH`.

3. **Crear el DAG `airflow/dags/training_dag.py`**
   - Schedule `@monthly`, `catchup=False`.
   - Tasks (con dependencias `>>`):
     1. `prepare_offline_store(execution_date)` — usa `data_interval_end` del DAG run.
     2. `apply_feast()`
     3. `populate_online_store()`
     4. `train_model(training_date=...)` → push del `run_id` por **XCom**.
     5. `promote_model_to_production(run_id=xcom_pull('train_model'))`.
     6. `run_drift_report(...)` (lo definimos en Sprint 2; por ahora dejarlo como
        placeholder `EmptyOperator`).
   - Manejo de fallas: `retries=1`, `retry_delay=timedelta(minutes=5)`.

4. **Probar localmente**
   - `docker compose up -d` levanta todo.
   - Entrar a `http://localhost:8080` → habilitar el DAG → "Trigger DAG".
   - Verificar: corrida verde, run aparece en MLflow, nueva versión promovida a
     `Production`, la API sigue respondiendo con el modelo nuevo.

### 3.3 Criterio de aceptación del Sprint 1
- [ ] `docker compose up -d` levanta postgres, mlflow, api, airflow-webserver,
      airflow-scheduler.
- [ ] El DAG `training_dag` aparece en la UI sin errores de import.
- [ ] Trigger manual completa OK y deja un modelo nuevo en stage `Production`.
- [ ] `schedule_interval='@monthly'` documentado en el README.

---

## 4. Sprint 2 — Monitoreo: data drift + model decay

### 4.1 Qué métricas, sobre qué datos
- **PSI por feature**: dividir en bins las features de entrenamiento (referencia) y
  comparar contra la distribución del último mes (current). Reportar PSI por
  feature + máximo.
- **MAE móvil**: para cada mes con `y` real disponible, comparar las predicciones
  guardadas contra el target real → MAE. Comparar con el MAE de validación del
  modelo actual (que ya está en MLflow).

> **Nota**: para tener MAE móvil necesitamos guardar las predicciones que hizo la
> API. Más simple para este TP: regenerar predicciones in-sample sobre el último
> mes del parquet (que ya tiene `target` real) usando el modelo "Production". Es
> equivalente a "qué hubiera predicho el modelo en producción para esos pozos".

### 4.2 Pasos
1. **Crear `src/monitoring/drift.py`** con dos funciones puras:
   - `compute_psi(reference: pd.Series, current: pd.Series, bins=10) -> float`
   - `compute_decay(model, df_recent, feature_cols, target_col) -> dict` que devuelve
     `{"mae": ..., "rmse": ..., "n_samples": ...}`.

2. **Crear `src/monitoring/report.py`**: arma un reporte (dict + plot opcional) con:
   - PSI por feature (tabla).
   - MAE/RMSE/MAPE actuales vs. baseline (las del run en MLflow).
   - Flags booleanos: `data_drift_detected` (cualquier PSI > 0.25),
     `model_decay_detected` (MAE_actual > 1.5 × MAE_val).
   - Persiste el reporte en JSON dentro de `feature_store/data/drift_reports/{date}.json`
     y como artefacto en MLflow (run separado del entrenamiento).

3. **Endpoint `GET /api/v1/monitoring/report`**
   - Parámetros: `date` (opcional, default = último reporte).
   - Devuelve el JSON del reporte.

4. **Task en Airflow `run_drift_report`** (ya estaba como placeholder en el Sprint 1)
   - Corre `report.py` después de `populate_online_store` y antes de `train_model`
     **usando el modelo "Production" actual** (es decir, el viejo). Así el reporte
     muestra el drift que justifica el reentrenamiento.

### 4.3 Criterio de aceptación del Sprint 2
- [ ] `GET /api/v1/monitoring/report` devuelve JSON válido con PSI y MAE.
- [ ] El reporte aparece como artefacto en MLflow.
- [ ] Si todo el dataset es igual al de entrenamiento, PSI ≈ 0 (sanity check).
- [ ] Si se filtra un subconjunto distinto (ej. sólo pozos petrolíferos), PSI > 0.25
      en al menos una feature (verifica que el detector funciona).

---

## 5. Sprint 3 — Escalabilidad de inferencia con Ray Serve

### 5.1 Diseño
La consigna pide "arquitectura escalable para responder la inferencia". El cambio
mínimo viable que mantiene la API tal cual:

- Convertir la app FastAPI en un *deployment* de Ray Serve con `@serve.deployment`
  + `@serve.ingress(app)`.
- Configurar `num_replicas=2` (o `autoscaling_config`) para mostrar paralelismo
  real.
- Levantar Ray como otro servicio en `docker-compose` (o `serve.run` en proceso).

### 5.2 Pasos
1. **Agregar `ray[serve]` a `requirements.txt`** (versión compatible con Python 3.11).

2. **Crear `src/api/serve_app.py`**:
   ```python
   from ray import serve
   from src.api.main import app as fastapi_app
   from src.inference_pipeline.predict import ForecastService

   @serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
   @serve.ingress(fastapi_app)
   class ForecastDeployment:
       def __init__(self):
           # estado por réplica
           fastapi_app.state.forecast_service = ForecastService()
           ...
   ```
   - Mover la inicialización del `lifespan` actual al `__init__` del deployment.

3. **Servicio `api` del compose**: cambiar el `CMD` a
   `serve run src.api.serve_app:ForecastDeployment --host 0.0.0.0 --port 8000`.

4. **Demostrar escalabilidad**: agregar a `README.md` un experimento simple con
   `hey` o `ab` mostrando throughput con `num_replicas=1` vs `num_replicas=4`.

### 5.3 Criterio de aceptación del Sprint 3
- [ ] `curl http://localhost:8000/api/v1/forecast?...` sigue funcionando idéntico.
- [ ] El dashboard de Ray (`http://localhost:8265`) muestra ≥2 réplicas activas.
- [ ] La validación OpenAPI sigue pasando (la firma de los endpoints no cambió).

---

## 6. Sprint 4 — README + Video demo

### 6.1 README mínimo a producir
Secciones a incluir (la consigna lo pide explícitamente):
1. **Cómo levantar todo**: `docker compose up -d`, qué puertos quedan expuestos.
2. **Cómo correr un entrenamiento manual**: el comando único de la entrega parcial.
3. **Cómo correr el DAG**: link a Airflow UI, cómo gatillarlo a mano.
4. **Cómo consumir la API**: ejemplos con `curl` para `/forecast`, `/wells`,
   `/monitoring/report`.
5. **Aspectos de diseño**:
   - Por qué Feast (online vs offline store).
   - Por qué MLflow Registry con stages.
   - Por qué Airflow `@monthly`.
   - Cómo se detecta drift (PSI) y decay (MAE móvil).
   - Cómo escala la API (Ray Serve, réplicas configurables).
6. **Trade-offs y limitaciones** asumidas.

### 6.2 Video demo (6 minutos)
Guión sugerido (con tiempos), ambos integrantes deben hablar:
- 0:00–0:45 — contexto del problema + arquitectura general (un slide).
- 0:45–2:00 — demo: `docker compose up`, mostrar Airflow UI, trigger del DAG,
  watching tasks pasar a verde.
- 2:00–3:30 — MLflow UI: experimentos, métricas, modelo en *Production*.
- 3:30–4:30 — reporte de drift: hit a `/monitoring/report`, mostrar PSI y MAE.
- 4:30–5:30 — Ray Serve: dashboard con réplicas + un benchmark rápido de throughput.
- 5:30–6:00 — cierre: limitaciones y posibles mejoras.

---

## 7. Riesgos conocidos y mitigaciones

| Riesgo | Mitigación |
| --- | --- |
| Imagen de Airflow muy pesada → builds lentos | Cachear capa de `requirements.txt` y no copiar `data/` |
| Conflicto de versiones (`mlflow`, `feast`, `ray`) en el mismo `requirements.txt` | Pinear versiones y testear `pip install -r` en CI local antes de mergear |
| `feast apply` desde Airflow puede no encontrar `feature_store.yaml` por `cwd` | Pasar `repo_path` absoluto y/o usar `subprocess.run(..., cwd=...)` |
| Ray Serve no encuentra el modelo de MLflow | Asegurar que el volumen `mlflow_artifacts` esté montado en el container que corre Serve |
| Predicciones para drift report sin `y` real disponible | Definir explícitamente la "fecha de evaluación" como `execution_date - 1 mes` (cuando ya hay target) |

---

## 8. Checklist final de la entrega

Antes del 28/05, verificar:

- [ ] `docker compose up -d` levanta postgres, mlflow, api (Ray Serve), airflow-*,
      sin errores.
- [ ] DAG `training_dag` corre completo en una ejecución manual.
- [ ] Modelo en stage `Production` en MLflow después del DAG.
- [ ] `GET /api/v1/forecast` y `/api/v1/wells` devuelven respuesta válida según la
      spec OpenAPI.
- [ ] `GET /api/v1/monitoring/report` devuelve PSI + MAE/decay.
- [ ] Ray dashboard accesible y con ≥2 réplicas.
- [ ] README cubre las 6 secciones del punto 6.1.
- [ ] Video de 6' subido (link en el README).
- [ ] Historial de commits con contribuciones balanceadas entre los dos integrantes.
- [ ] Cada bloque (Airflow, monitoring, Ray, docs) entró por su propio PR revisado.

---

## 9. Referencias rápidas

- Feast online vs offline store: https://docs.feast.dev/
- MLflow Model Registry stages: https://mlflow.org/docs/latest/model-registry.html
- Airflow LocalExecutor + Postgres: https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html
- Ray Serve + FastAPI: https://docs.ray.io/en/latest/serve/http-guide.html
- Population Stability Index: https://www.listendata.com/2015/05/population-stability-index.html
