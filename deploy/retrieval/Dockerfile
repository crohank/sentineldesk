FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /tmp/app-requirements.txt
COPY deploy/retrieval/requirements.txt /tmp/cloud-requirements.txt
RUN pip install --no-cache-dir -r /tmp/app-requirements.txt -r /tmp/cloud-requirements.txt
COPY sentineldesk/ sentineldesk/
COPY data/processed/kb_train.parquet data/processed/kb_partitions.parquet data/processed/
COPY artifacts/fast_path/ artifacts/fast_path/
ENV OPENBLAS_NUM_THREADS=1 HOST=0.0.0.0 SENTINELDESK_MODE=managed PYTHONUNBUFFERED=1
CMD ["python", "-m", "sentineldesk.api"]
