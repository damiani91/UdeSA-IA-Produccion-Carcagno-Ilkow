FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY feature_store/ ./feature_store/

ENV PYTHONPATH=/app

CMD ["python", "-m", "scripts.train", "--date", "2024-12-01"]
