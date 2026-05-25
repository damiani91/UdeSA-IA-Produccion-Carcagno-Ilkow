-- Creates the Airflow metadata database on first PostgreSQL boot.
-- The existing POSTGRES_USER (mlflow) is reused as the owner to avoid
-- creating a separate user.
CREATE DATABASE airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow TO mlflow;
