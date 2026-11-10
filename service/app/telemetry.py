"""
Configuração do OpenTelemetry (traces + métricas) para a API.

Fluxo:  app (SDK OTel)  --OTLP/gRPC-->  otel-collector  --> Tempo (traces)
                                                        --> /metrics :8889 --> Prometheus (scrape)

Tudo é configurável por variáveis de ambiente padrão do OTel:
  OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_INSECURE
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Buckets pensados para inferência em CPU (ms). Os buckets padrão do SDK
# (0, 5, 10, 25 ... 10000) são genéricos demais para enxergar p95 direito.
LATENCY_BUCKETS_MS = [5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000]
CONFIDENCE_BUCKETS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class Instruments:
    """Todos os instrumentos de métrica da aplicação num lugar só.

    Nome OTel                 -> nome no Prometheus (o collector converte)
    yolo.request.duration (ms)-> yolo_request_duration_milliseconds_{bucket,sum,count}
    yolo.stage.duration (ms)  -> yolo_stage_duration_milliseconds_{bucket,sum,count}
    yolo.requests             -> yolo_requests_total
    yolo.detections           -> yolo_detections_total
    yolo.detection.confidence -> yolo_detection_confidence_{bucket,sum,count}
    yolo.inflight             -> yolo_inflight (gauge)
    """
    request_duration: metrics.Histogram
    stage_duration: metrics.Histogram
    requests: metrics.Counter
    detections: metrics.Counter
    confidence: metrics.Histogram
    inflight: metrics.UpDownCounter


_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def setup_telemetry() -> tuple[trace.Tracer, Instruments]:
    global _tracer_provider, _meter_provider

    # Resource = "quem" está emitindo a telemetria. Vira labels/atributos em tudo.
    # Resource.create também lê OTEL_RESOURCE_ATTRIBUTES do ambiente.
    resource = Resource.create(
        {"service.name": os.getenv("OTEL_SERVICE_NAME", "yolo-api")}
    )

    # ---------- Traces ----------
    _tracer_provider = TracerProvider(resource=resource)
    # BatchSpanProcessor agrupa spans e envia em background (não bloqueia o request).
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)

    # ---------- Métricas ----------
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(), export_interval_millis=5_000
    )
    views = [
        View(instrument_name="yolo.request.duration",
             aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS_MS)),
        View(instrument_name="yolo.stage.duration",
             aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS_MS)),
        View(instrument_name="yolo.detection.confidence",
             aggregation=ExplicitBucketHistogramAggregation(CONFIDENCE_BUCKETS)),
    ]
    _meter_provider = MeterProvider(resource=resource, metric_readers=[reader], views=views)
    metrics.set_meter_provider(_meter_provider)

    # ---------- Logs ----------
    # Injeta trace_id/span_id em cada linha de log -> dá pra correlacionar log <-> trace.
    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging.INFO)

    tracer = trace.get_tracer("yolo.api")
    meter = metrics.get_meter("yolo.api")

    instruments = Instruments(
        request_duration=meter.create_histogram(
            "yolo.request.duration", unit="ms",
            description="Tempo total de uma detecção (decode + preprocess + inferência + pós + render)"),
        stage_duration=meter.create_histogram(
            "yolo.stage.duration", unit="ms",
            description="Tempo de cada etapa do pipeline YOLO (atributo stage)"),
        requests=meter.create_counter(
            "yolo.requests", unit="{request}",
            description="Requisições de detecção por status (ok/error)"),
        detections=meter.create_counter(
            "yolo.detections", unit="{object}",
            description="Objetos detectados por classe"),
        confidence=meter.create_histogram(
            "yolo.detection.confidence",
            description="Distribuição da confiança das detecções"),
        inflight=meter.create_up_down_counter(
            "yolo.inflight", unit="{request}",
            description="Detecções em andamento agora"),
    )
    return tracer, instruments


def shutdown_telemetry() -> None:
    """Faz flush do que estiver em buffer antes do processo morrer."""
    if _tracer_provider:
        _tracer_provider.shutdown()
    if _meter_provider:
        _meter_provider.shutdown()
