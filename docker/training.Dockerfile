FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY feature_store/ ./feature_store/

ENV PYTHONPATH=/app

CMD ["python", "-m", "scripts.train", "--date", "2024-12-01"]
