# ---------------------------------------------------------------------------
# Estágio 1 (descartável): torch CPU + ultralytics só para exportar yolov8n -> ONNX.
# Nada daqui vai para a imagem final além do .onnx e das imagens de exemplo.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS exporter

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Wheels CPU-only do PyTorch (~200 MB em vez de ~2 GB com CUDA)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics==8.3.40 onnx==1.17.0

WORKDIR /build
COPY scripts/export_model.py .
RUN python export_model.py

# ---------------------------------------------------------------------------
# Estágio 2 (runtime): Python + onnxruntime CPU + OpenTelemetry. Sem torch.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/srv/models/yolov8n.onnx \
    SAMPLES_DIR=/srv/samples

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=exporter /export/yolov8n.onnx /srv/models/yolov8n.onnx
COPY --from=exporter /export/samples /srv/samples
COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
