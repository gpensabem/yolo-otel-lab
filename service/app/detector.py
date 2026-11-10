"""
Detector YOLOv8n rodando em ONNX Runtime (CPU puro — sem torch em runtime).

Cada etapa do pipeline vira um span filho e um ponto no histograma yolo.stage.duration:
  yolo.preprocess -> yolo.inference -> yolo.postprocess -> yolo.render
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import onnxruntime as ort
from opentelemetry import trace
from PIL import Image, ImageDraw, ImageFont

from .coco import COCO_CLASSES
from .telemetry import Instruments


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: list[float]  # x1, y1, x2, y2 em pixels da imagem original


class YoloDetector:
    def __init__(self, model_path: str, tracer: trace.Tracer, inst: Instruments,
                 iou_threshold: float = 0.45, max_det: int = 100):
        self.tracer = tracer
        self.inst = inst
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(model_path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2])  # [1, 3, 640, 640]

    # ------------------------------------------------------------------ pipeline
    def detect(self, img: Image.Image, conf: float) -> dict:
        timings: dict[str, float] = {}
        attrs = {"model": self.model_name}

        with self._stage("preprocess", timings) as span:
            tensor, ratio, pad = self._letterbox(img)
            span.set_attribute("yolo.input.shape", str(list(tensor.shape)))
            span.set_attribute("yolo.letterbox.ratio", round(ratio, 4))

        with self._stage("inference", timings) as span:
            output = self.session.run(None, {self.input_name: tensor})[0]
            span.set_attribute("yolo.model", self.model_name)
            span.set_attribute("yolo.runtime", "onnxruntime-cpu")

        with self._stage("postprocess", timings) as span:
            dets = self._postprocess(output, ratio, pad, conf, img.size)
            span.set_attribute("yolo.conf_threshold", conf)
            span.set_attribute("yolo.detections.count", len(dets))
            # Eventos no span: aparecem na linha do tempo do trace no Grafana/Tempo
            for d in dets[:10]:
                span.add_event("detection", {"class": d.class_name,
                                             "confidence": round(d.confidence, 3)})

        with self._stage("render", timings):
            image_b64 = self._render(img, dets)

        for stage, ms in timings.items():
            self.inst.stage_duration.record(ms, {**attrs, "stage": stage})
        for d in dets:
            self.inst.detections.add(1, {"class_name": d.class_name})
            self.inst.confidence.record(d.confidence)

        return {
            "detections": [asdict(d) for d in dets],
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
            "image_base64": image_b64,
        }

    # ------------------------------------------------------------------ helpers
    class _StageCtx:
        def __init__(self, outer: "YoloDetector", name: str, timings: dict):
            self.outer, self.name, self.timings = outer, name, timings

        def __enter__(self):
            self._cm = self.outer.tracer.start_as_current_span(f"yolo.{self.name}")
            self.span = self._cm.__enter__()
            self.t0 = time.perf_counter()
            return self.span

        def __exit__(self, *exc):
            self.timings[self.name] = (time.perf_counter() - self.t0) * 1000
            return self._cm.__exit__(*exc)

    def _stage(self, name: str, timings: dict) -> "_StageCtx":
        return self._StageCtx(self, name, timings)

    def _letterbox(self, img: Image.Image):
        """Redimensiona mantendo proporção e completa com cinza (114) até 640x640."""
        s = self.input_size
        w, h = img.size
        r = min(s / w, s / h)
        nw, nh = round(w * r), round(h * r)
        canvas = Image.new("RGB", (s, s), (114, 114, 114))
        padx, pady = (s - nw) // 2, (s - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR), (padx, pady))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None]  # HWC -> 1xCxHxW
        return np.ascontiguousarray(arr), r, (padx, pady)

    def _postprocess(self, output, ratio, pad, conf, orig_size) -> list[Detection]:
        # YOLOv8 output: (1, 84, 8400) -> 8400 candidatos x [cx, cy, w, h, 80 scores]
        pred = output[0].T
        scores_all = pred[:, 4:]
        cls_ids = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(pred)), cls_ids]
        keep = scores >= conf
        if not keep.any():
            return []
        boxes, scores, cls_ids = pred[keep, :4], scores[keep], cls_ids[keep]

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # NMS por classe: desloca as caixas de cada classe para não competirem entre si
        offset = cls_ids[:, None].astype(np.float32) * 4096.0
        idx = _nms(xyxy + offset, scores, self.iou_threshold)[: self.max_det]

        xyxy = xyxy[idx]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
        w, h = orig_size
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        return [
            Detection(int(c), COCO_CLASSES[int(c)], float(s), [round(float(v), 1) for v in b])
            for b, s, c in zip(xyxy, scores[idx], cls_ids[idx])
        ]

    def _render(self, img: Image.Image, dets: list[Detection]) -> str:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        font = ImageFont.load_default()
        lw = max(2, round(min(img.size) / 250))
        for d in dets:
            color = _color(d.class_id)
            x1, y1, x2, y2 = d.box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            label = f"{d.class_name} {d.confidence:.2f}"
            tb = draw.textbbox((x1, y1), label, font=font)
            th = tb[3] - tb[1] + 6
            ty = y1 - th if y1 - th > 0 else y1
            draw.rectangle([x1, ty, x1 + (tb[2] - tb[0]) + 8, ty + th], fill=color)
            draw.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def _color(class_id: int) -> tuple[int, int, int]:
    palette = [(230, 57, 70), (29, 120, 116), (244, 162, 97), (69, 123, 157),
               (131, 56, 236), (42, 157, 143), (231, 111, 81), (58, 134, 255)]
    return palette[class_id % len(palette)]
