FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY feature_store/ ./feature_store/

ENV PYTHONPATH=/app

# Sprint 3: la API se sirve detrás de Ray Serve (ver src/api/serve_app.py).
# El comando del compose lo sobrescribe; este CMD queda como fallback razonable.
CMD ["serve", "run", "--host", "0.0.0.0", "--port", "8000", "src.api.serve_app:forecast_app"]

