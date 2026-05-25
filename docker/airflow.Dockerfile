FROM apache/airflow:2.10.3-python3.11

USER root

# Dependencias del sistema necesarias para algunas libs (gcc para compilaciones puntuales)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Instalar dependencias del proyecto encima de la base de Airflow para que
# los DAGs puedan importar src.* directamente (PythonOperator).
COPY requirements.txt /tmp/project-requirements.txt
RUN pip install --no-cache-dir -r /tmp/project-requirements.txt

# Copiar el código del proyecto a una ubicación conocida y agregarla al PYTHONPATH
# para que los DAGs puedan hacer `from src.training_pipeline.train import train_model`.
COPY --chown=airflow:root src/ /opt/airflow/project/src/
COPY --chown=airflow:root scripts/ /opt/airflow/project/scripts/
COPY --chown=airflow:root feature_store/feature_store.yaml /opt/airflow/project/feature_store/feature_store.yaml
COPY --chown=airflow:root feature_store/features.py /opt/airflow/project/feature_store/features.py

ENV PYTHONPATH=/opt/airflow/project:${PYTHONPATH}
