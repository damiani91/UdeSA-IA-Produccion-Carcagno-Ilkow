FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY feature_store/ ./feature_store/

ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8265

CMD ["python", "-m", "src.api.entrypoint"]
