"""
API FastAPI de detecção de objetos com YOLOv8n + OpenTelemetry.

Endpoints:
  GET  /                      front-end de teste
  POST /detect                upload de imagem (multipart, campo "file")
  POST /detect/sample/{name}  roda numa imagem de exemplo embutida
  POST /simulate/error        gera um erro proposital (ver span vermelho no Tempo)
  POST /simulate/slow?ms=800  gera latência artificial (ver cauda no p99)
  GET  /health, /config, /samples
"""
from __future__ import annotations

import io
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode
from PIL import Image, UnidentifiedImageError

from .detector import YoloDetector
from .telemetry import setup_telemetry, shutdown_telemetry

# 1) Telemetria primeiro — antes de criar qualquer coisa que emita spans/métricas
tracer, inst = setup_telemetry()
log = logging.getLogger("yolo.api")

BASE = Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", "/srv/models/yolov8n.onnx")
SAMPLES_DIR = Path(os.getenv("SAMPLES_DIR", "/srv/samples"))
GRAFANA_URL = os.getenv("GRAFANA_PUBLIC_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_PUBLIC_URL", "http://localhost:9090")

detector = YoloDetector(MODEL_PATH, tracer, inst)
log.info("modelo carregado: %s (input %dpx)", MODEL_PATH, detector.input_size)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    shutdown_telemetry()


app = FastAPI(title="YOLO + OpenTelemetry lab", lifespan=lifespan)
# 2) Auto-instrumentação: cria um span SERVER por request + métricas http.server.*
FastAPIInstrumentor.instrument_app(app, excluded_urls="health,static,samples/.*")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def _trace_id() -> str:
    return format(trace.get_current_span().get_span_context().trace_id, "032x")


def _run_detection(data: bytes, conf: float, source: str) -> dict:
    span = trace.get_current_span()  # o span SERVER criado pela auto-instrumentação
    span.set_attribute("yolo.source", source)
    source_kind = "sample" if source.startswith("sample:") else "upload"

    inst.inflight.add(1)
    t0 = time.perf_counter()
    status = "ok"
    try:
        with tracer.start_as_current_span("yolo.decode_image") as s:
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
                img = img.convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                s.record_exception(e)
                s.set_status(Status(StatusCode.ERROR, "imagem inválida"))
                raise HTTPException(400, "O arquivo enviado não é uma imagem válida") from e
            s.set_attribute("image.width", img.width)
            s.set_attribute("image.height", img.height)
            s.set_attribute("image.bytes", len(data))

        result = detector.detect(img, conf)
        n = len(result["detections"])
        span.set_attribute("yolo.detections.count", n)
        log.info("detecção ok source=%s objetos=%d", source, n)
        return {
            **result,
            "trace_id": _trace_id(),
            "image_size": [img.width, img.height],
            "model": detector.model_name,
        }
    except Exception:
        status = "error"
        log.warning("detecção falhou source=%s", source)
        raise
    finally:
        elapsed = (time.perf_counter() - t0) * 1000
        # Registradas dentro do span ativo -> o SDK anexa um exemplar (trace_id)
        inst.request_duration.record(elapsed, {"status": status, "source": source_kind})
        inst.requests.add(1, {"status": status, "source": source_kind})
        inst.inflight.add(-1)


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config")
def config():
    return {"grafana_url": GRAFANA_URL, "prometheus_url": PROMETHEUS_URL,
            "model": detector.model_name}


@app.get("/samples")
def samples():
    return sorted(p.name for p in SAMPLES_DIR.glob("*.jpg"))


@app.get("/samples/{name}")
def sample_file(name: str):
    path = SAMPLES_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "amostra não encontrada")
    return FileResponse(path)


@app.post("/detect")
def detect(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.01, le=0.99)):
    return _run_detection(file.file.read(), conf, source=f"upload:{file.filename}")


@app.post("/detect/sample/{name}")
def detect_sample(name: str, conf: float = Query(0.25, ge=0.01, le=0.99)):
    path = SAMPLES_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(404, "amostra não encontrada")
    return _run_detection(path.read_bytes(), conf, source=f"sample:{path.name}")


@app.post("/simulate/error")
def simulate_error():
    """Exceção não tratada -> 500. O span fica com status ERROR e a exceção registrada."""
    with tracer.start_as_current_span("yolo.fake_dependency") as s:
        s.set_attribute("peer.service", "feature-store")
        inst.requests.add(1, {"status": "error", "source": "simulated"})
        raise RuntimeError("falha simulada: feature-store indisponível")


@app.post("/simulate/slow")
def simulate_slow(ms: int = Query(800, ge=50, le=10_000)):
    """Latência artificial — útil para ver o p99 e os exemplares subirem no Grafana."""
    t0 = time.perf_counter()
    with tracer.start_as_current_span("yolo.slow_io") as s:
        jitter = ms * random.uniform(0.8, 1.2)
        s.set_attribute("simulated.ms", round(jitter))
        time.sleep(jitter / 1000)
    elapsed = (time.perf_counter() - t0) * 1000
    inst.request_duration.record(elapsed, {"status": "ok", "source": "simulated"})
    inst.requests.add(1, {"status": "ok", "source": "simulated"})
    return {"slept_ms": round(elapsed), "trace_id": _trace_id()}
