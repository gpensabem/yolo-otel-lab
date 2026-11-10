# YOLO Telemetry Lab

Detecção de objetos com **YOLOv8n em CPU** (ONNX Runtime, sem torch em runtime) instrumentada com
**OpenTelemetry**, coletada por um **OTel Collector**, métricas no **Prometheus**, traces no **Tempo**
e tudo visualizado no **Grafana** com dashboard já provisionado.

```
 browser ──▶ yolo-api (FastAPI :8000)
               │  OpenTelemetry SDK (traces + métricas + trace_id nos logs)
               ▼  OTLP gRPC :4317
          otel-collector
            ├─ traces  ──▶ Tempo (:3200)
            └─ metrics ──▶ exporter Prometheus (:8889) ◀── scrape ── Prometheus (:9090)
                                                                         │
                                          Grafana (:3000) ◀──────────────┴── + Tempo
```

## Subir

```bash
docker compose up -d --build          # 1º build: ~3-5 min (baixa torch CPU só p/ exportar o ONNX)
docker compose --profile load up -d   # opcional: gerador de carga contínuo
```

| O quê | URL |
|---|---|
| Front da app (upload, exemplos, botões de carga/erro/lentidão) | http://localhost:8000 |
| Grafana (sem login, dashboard "YOLO OTel Lab" é a home) | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Métricas cruas que o collector expõe (formato Prometheus/OpenMetrics) | http://localhost:8889/metrics |
| Métricas internas do collector | http://localhost:8888/metrics |
| Tempo API | http://localhost:3200 |
| OTLP para mandar telemetria do host | `localhost:4317` (gRPC) / `localhost:4318` (HTTP) |

Parar e limpar: `docker compose --profile load down`

## Roteiro sugerido

1. Abra http://localhost:8000, clique em `bus.jpg`. Veja a barra "tempo por etapa" e clique no **trace_id**:
   abre o trace no Grafana com os spans `POST /detect/sample/{name}` → `yolo.decode_image`, `yolo.preprocess`,
   `yolo.inference`, `yolo.postprocess` (com eventos `detection`) e `yolo.render`.
2. Ligue **Carga contínua** (ou o profile `load`) e abra o dashboard. Em ~30s os painéis enchem.
3. No painel de latência, os **losangos ◆ são exemplares**: clique num deles → "Query with Tempo" e você cai
   exatamente no trace que gerou aquele ponto. É a ponte métrica → trace.
4. Clique em **Request lento** e veja o p99 subir e o exemplar correspondente aparecer lá em cima.
5. Clique em **Erro 500**: o span fica vermelho no Tempo com a exceção registrada; a taxa de erro sobe.
6. Abra http://localhost:8889/metrics e procure `yolo_` — é o que o Prometheus raspa. Repare nos `# {trace_id=...}`
   (exemplares em OpenMetrics) e nos labels vindos do resource (`service_name`, `deployment_environment`...).
7. `docker compose logs -f yolo-api`: cada log tem `trace_id=` — correlação log ↔ trace.

## Mapa do código (onde a telemetria acontece)

| Arquivo | O que mostra |
|---|---|
| `service/app/telemetry.py` | Setup do SDK: `Resource`, `TracerProvider` + `BatchSpanProcessor`, `MeterProvider` + `PeriodicExportingMetricReader`, **Views** com buckets customizados, instrumentação de logging, e a criação de todos os instrumentos (Counter, Histogram, UpDownCounter). |
| `service/app/main.py` | Auto-instrumentação (`FastAPIInstrumentor`), spans manuais, `record_exception` + `Status(ERROR)`, métricas registradas dentro do span ativo (→ exemplares). |
| `service/app/detector.py` | Um span por etapa do pipeline, atributos (`yolo.detections.count`…), **span events** por detecção, histograma por etapa com atributo `stage`. |
| `otel-collector/config.yaml` | receivers → processors (`memory_limiter`, `batch`) → exporters (Tempo, Prometheus, debug). |
| `prometheus/prometheus.yml` | scrape do collector (`honor_labels`), feature `exemplar-storage` está no compose. |
| `grafana/provisioning/*` | Datasources (Prometheus com exemplares → Tempo; Tempo → métricas) e dashboard como código. |
| `grafana/dashboards/yolo.json` | O dashboard. Cada painel tem descrição explicando a query. |

### Nomes: OTel → Prometheus

O exporter do collector converte nome + unidade + tipo:

| Instrumento OTel | Série no Prometheus |
|---|---|
| `yolo.request.duration` (Histogram, `ms`) | `yolo_request_duration_milliseconds_bucket/_sum/_count` |
| `yolo.stage.duration` (Histogram, `ms`, attr `stage`) | `yolo_stage_duration_milliseconds_*` |
| `yolo.requests` (Counter, attr `status`, `source`) | `yolo_requests_total` |
| `yolo.detections` (Counter, attr `class_name`) | `yolo_detections_total` |
| `yolo.detection.confidence` (Histogram) | `yolo_detection_confidence_*` |
| `yolo.inflight` (UpDownCounter) | `yolo_inflight` (gauge) |
| `http.server.duration` (auto, FastAPI) | `http_server_duration_milliseconds_*` |

## Queries para brincar

PromQL (Prometheus ou Grafana Explore):
```promql
# p95 total
histogram_quantile(0.95, sum by (le) (rate(yolo_request_duration_milliseconds_bucket[1m])))
# média por etapa
sum by (stage) (rate(yolo_stage_duration_milliseconds_sum[1m])) / sum by (stage) (rate(yolo_stage_duration_milliseconds_count[1m]))
# % das inferências abaixo de 100ms (SLI)
sum(rate(yolo_stage_duration_milliseconds_bucket{stage="inference",le="100"}[5m])) / sum(rate(yolo_stage_duration_milliseconds_count{stage="inference"}[5m]))
```

TraceQL (Grafana → Explore → Tempo):
```
{ resource.service.name = "yolo-api" && status = error }
{ span.yolo.detections.count > 5 }
{ name = "yolo.inference" && duration > 150ms }
{ span:name = "yolo.postprocess" } >> { }          # spans abaixo do postprocess
```

## Por que ONNX e não ultralytics em runtime?

O `Dockerfile` é multi-stage: o 1º estágio instala **torch CPU** + ultralytics só para exportar `yolov8n.pt → yolov8n.onnx`
e é descartado. A imagem final tem apenas onnxruntime + numpy + Pillow + OTel. Bônus para o aprendizado: como o
pré e o pós-processamento (letterbox, NMS) estão escritos à mão, cada etapa vira um span próprio e dá para ver
onde o tempo vai de verdade.

## Próximos passos (se quiser expandir)

- **Logs no Grafana**: adicionar Loki + exporter `otlphttp/loki` no collector e o `LoggingHandler` do SDK.
- **Métricas derivadas de traces**: connector `spanmetrics` no collector (RED automático por span).
- **Service graph**: `metrics_generator` do Tempo com remote-write no Prometheus.
- **Alertas**: regra no Prometheus/Grafana para p95 > 500ms ou taxa de erro > 5%.
- **Browser**: OTel JS no front para propagar `traceparent` e ter o trace começando no clique.

> Versões fixadas (collector 0.120, Tempo 2.7, Prometheus 3.2, Grafana 11.5, OTel Python 1.29) para o lab ser
> reprodutível. Se atualizar o collector, nomes de métricas internas (`otelcol_*`) podem mudar um pouco.
