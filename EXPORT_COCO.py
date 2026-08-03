"""
EXPORT_COCO.py — Exporta o YOLOv8n padrão (COCO) para ONNX
─────────────────────────────────────────────────────────────────
ZERO TREINO NECESSÁRIO.

O YOLOv8n padrão já foi treinado em milhões de fotos reais e
detecta de fábrica:
  - stop sign  (placa de PARE)
  - car        (carro)
  - person     (pessoa)
  - truck, bus, traffic light...

Este script:
  1. Baixa yolov8n.pt (se não existir)
  2. Exporta para ONNX
  3. Copia para models/sign_detector.onnx

O carro-autonomo.py detecta automaticamente que é um modelo
COCO (80 classes) e mapeia para as ações do carro.

USO:
  python EXPORT_COCO.py
"""

import shutil
import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, default=640,
                help="Tamanho de entrada do modelo. 640=padrão (mais preciso), "
                     "416=mais rápido na CPU (~2.4x), 320=máxima velocidade")
args = ap.parse_args()

print("=" * 60)
print("  EXPORT_COCO — YOLOv8n pré-treinado (sem treino)")
print(f"  imgsz={args.imgsz}")
print("=" * 60)

from ultralytics import YOLO

print("\n[1/3] Carregando yolov8n.pt (baixa automático se preciso)...")
model = YOLO("yolov8n.pt")

print(f"[2/3] Exportando para ONNX (imgsz={args.imgsz})...")
model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False)

onnx = Path("yolov8n.onnx")
if not onnx.exists():
    # ultralytics pode salvar ao lado do .pt
    candidates = list(BASE.rglob("yolov8n.onnx"))
    if candidates:
        onnx = candidates[0]
    else:
        print("[ERRO] yolov8n.onnx não encontrado após export")
        raise SystemExit(1)

print("[3/3] Copiando para models/sign_detector.onnx...")
MODELS.mkdir(exist_ok=True)
shutil.copy2(onnx, MODELS / "sign_detector.onnx")

mb = (MODELS / "sign_detector.onnx").stat().st_size / 1e6
print(f"\n[OK] models/sign_detector.onnx  ({mb:.1f} MB)")
print()
print("Classes COCO que o carro vai usar:")
print("  stop sign     → STOP")
print("  person        → OBSTACLE")
print("  car/truck/bus → OBSTACLE")
print()
print("Agora rode:")
print("  python carro-autonomo.py")