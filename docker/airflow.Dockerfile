FROM apache/airflow:2.10.4-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# dill is required by ExternalPythonOperator to serialize callables across
# Python interpreter boundaries (Airflow → project venv).
RUN pip install --no-cache-dir dill

# Create a separate venv for the project ML stack (feast, mlflow, sklearn, etc).
# This isolates SQLAlchemy 2.x (required by the project) from Airflow's own
# SQLAlchemy 1.4.x, preventing the MappedAnnotationError on TaskInstance.
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /home/airflow/project-venv \
    && /home/airflow/project-venv/bin/pip install --no-cache-dir dill \
    && /home/airflow/project-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY feature_store/ ./feature_store/
COPY data/scripts/ ./data/scripts/
