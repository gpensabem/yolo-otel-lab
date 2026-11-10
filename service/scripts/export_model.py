"""Roda só no estágio de build: baixa yolov8n.pt, exporta para ONNX e copia imagens de exemplo.
O torch/ultralytics ficam nesse estágio descartável; a imagem final só leva o .onnx."""
import os
import shutil

import ultralytics
from ultralytics import YOLO

OUT = "/export"
os.makedirs(f"{OUT}/samples", exist_ok=True)

onnx_path = YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=12, simplify=False, dynamic=False)
shutil.copy(onnx_path, f"{OUT}/yolov8n.onnx")

assets = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
for name in os.listdir(assets):
    if name.lower().endswith(".jpg"):
        shutil.copy(os.path.join(assets, name), f"{OUT}/samples/{name}")

print("exportado:", os.listdir(OUT), os.listdir(f"{OUT}/samples"))
