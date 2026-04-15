FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.11.1 psycopg2-binary

EXPOSE 5000

ENTRYPOINT ["mlflow"]
