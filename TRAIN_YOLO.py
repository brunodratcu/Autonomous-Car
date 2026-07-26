"""
TRAIN_YOLO.py — Treino YOLOv8n + Export ONNX
─────────────────────────────────────────────────────────────────
USO:
  python TRAIN_YOLO.py                              ← dataset auto-gerado
  python TRAIN_YOLO.py --annotated ./frames_anotacao ← frames anotados no LabelImg
  python TRAIN_YOLO.py --annotated ./frames_anotacao --real-only

Modos:
  sem --annotated   → usa yolo_dataset/ (gerado pelo PREPARE_YOLO_DATASET.py)
  com --annotated   → mistura frames reais anotados + dataset existente
  com --real-only   → usa APENAS frames reais anotados (melhor qualidade)
"""

import os, sys, shutil, argparse, time, random
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
YOLO_DST   = BASE_DIR / "yolo_dataset"
MODELS_DIR = BASE_DIR / "models"
FINAL_ONNX = MODELS_DIR / "sign_detector.onnx"

CLASSES = ["Stop","Esquerda","Direita","SemRetorno",
           "Verde","Cone","Carro","Pessoa"]
EXTS    = {".jpg",".jpeg",".png",".bmp"}
SEED    = 42
random.seed(SEED)


# ================================================================
#  INGERE FRAMES ANOTADOS NO LABELIMG (formato YOLO txt)
# ================================================================

def ingerir_anotados(src: Path, dst_img: Path, dst_lbl: Path,
                     val_frac: float = 0.20) -> tuple[int, int]:
    """
    Lê src/ onde LabelImg salvou frame_XXXXXX.jpg + frame_XXXXXX.txt
    (formato YOLO: class_id cx cy w h, normalizado).
    Copia para dst_img/ e dst_lbl/ com split train/val.
    Retorna (n_train, n_val).
    """
    pares = []
    for img_p in sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")):
        lbl_p = img_p.with_suffix(".txt")
        if not lbl_p.exists():
            continue   # frame sem anotação = ignorado
        # Verifica se o txt tem conteúdo (não está vazio)
        txt = lbl_p.read_text().strip()
        if not txt:
            continue   # frame sem objetos = ignorado no treino
        pares.append((img_p, lbl_p))

    if not pares:
        print(f"  [WARN] Nenhum frame anotado encontrado em {src}")
        print(f"         Certifique-se de salvar no formato YOLO (.txt) no LabelImg")
        return 0, 0

    random.shuffle(pares)
    n_val   = max(1, int(len(pares) * val_frac))
    splits  = {"val": pares[:n_val], "train": pares[n_val:]}
    n_tr = n_va = 0

    for split, lista in splits.items():
        idir = dst_img / split; idir.mkdir(parents=True, exist_ok=True)
        ldir = dst_lbl / split; ldir.mkdir(parents=True, exist_ok=True)
        for img_p, lbl_p in lista:
            stem = f"real_{img_p.stem}"
            shutil.copy2(img_p, idir / f"{stem}.jpg")
            shutil.copy2(lbl_p, ldir / f"{stem}.txt")
            if split == "train": n_tr += 1
            else:                n_va += 1

    print(f"  Frames reais: {n_tr} train / {n_va} val")
    return n_tr, n_va


# ================================================================
#  GARANTE QUE YOLO_DATASET EXISTE (chama PREPARE se necessário)
# ================================================================

def garantir_dataset(real_only: bool) -> Path:
    yaml = YOLO_DST / "dataset.yaml"

    if real_only:
        # Recria do zero só com frames reais
        if YOLO_DST.exists():
            shutil.rmtree(YOLO_DST)

    if not yaml.exists():
        prep = BASE_DIR / "PREPARE_YOLO_DATASET.py"
        if not prep.exists():
            print("[ERRO] PREPARE_YOLO_DATASET.py não encontrado")
            sys.exit(1)
        src = prep.read_text(encoding="utf-8")
        cut = src.find("\nif __name__")
        ns  = {}
        exec(compile(src[:cut], str(prep), "exec"), ns)
        ns["converter"](str(BASE_DIR / "dataset_v3"), str(YOLO_DST))

    n_tr = len(list((YOLO_DST/"images"/"train").glob("*.jpg")))
    n_va = len(list((YOLO_DST/"images"/"val").glob("*.jpg")))
    print(f"  Dataset base: {n_tr} train / {n_va} val")
    return yaml


# ================================================================
#  GERA DATASET.YAML COM CAMINHO ABSOLUTO
# ================================================================

def gerar_yaml(dst: Path) -> Path:
    yaml_path = dst / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dst.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\nnc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )
    return yaml_path


# ================================================================
#  TREINO
# ================================================================

def treinar(yaml: Path, epochs: int, batch: int, patience: int) -> Path:
    from ultralytics import YOLO

    print(f"\n  epochs={epochs}  batch={batch}  patience={patience}")
    print(f"  data={yaml}")

    model   = YOLO("yolov8n.pt")
    results = model.train(
        data      = str(yaml),
        epochs    = epochs,
        imgsz     = 640,
        batch     = batch,
        patience  = patience,
        device    = "cpu",
        project   = str(BASE_DIR / "runs"),
        name      = "sign_detector",
        exist_ok  = True,
        workers   = 0,
        verbose   = True,
        plots     = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        cands = sorted((BASE_DIR/"runs").rglob("best.pt"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands: print("[ERRO] best.pt não encontrado"); sys.exit(1)
        best = cands[0]

    print(f"\n  [OK] best.pt: {best}")
    return best


def exportar(best_pt: Path) -> Path:
    from ultralytics import YOLO
    YOLO(str(best_pt)).export(format="onnx", imgsz=640, opset=12,
                               simplify=True, dynamic=False)
    onnx = best_pt.with_suffix(".onnx")
    if not onnx.exists(): print(f"[ERRO] .onnx não gerado"); sys.exit(1)
    return onnx


def copiar(onnx: Path):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, FINAL_ONNX)
    mb = FINAL_ONNX.stat().st_size / 1e6
    print(f"  [OK] {FINAL_ONNX}  ({mb:.1f} MB)")


def metricas(best_pt: Path):
    csv = best_pt.parent.parent / "results.csv"
    if not csv.exists(): return
    try:
        import csv as _csv
        rows = list(_csv.DictReader(open(csv)))
        if not rows: return
        last = {k.strip():v.strip() for k,v in rows[-1].items()}
        m50   = last.get("metrics/mAP50(B)","?")
        m5095 = last.get("metrics/mAP50-95(B)","?")
        print(f"\n  mAP50={m50}  mAP50-95={m5095}")
        try:
            v = float(m50)
            if   v >= 0.85: print("  → Pronto para uso")
            elif v >= 0.70: print("  → Funcional; mais dados melhora")
            else:           print("  → Insuficiente; precisa mais anotações")
        except ValueError: pass
    except Exception: pass


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated",  default=None,
                    help="Pasta com frames anotados pelo LabelImg")
    ap.add_argument("--real-only",  action="store_true",
                    help="Usa SOMENTE os frames anotados (ignora dataset auto-gerado)")
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--batch",      type=int, default=8)
    ap.add_argument("--patience",   type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  TRAIN_YOLO — YOLOv8n")
    modo = "REAL-ONLY" if args.real_only else \
           ("REAL+AUTO" if args.annotated else "AUTO")
    print(f"  Modo: {modo}")
    print("=" * 62)

    try:
        import ultralytics
        print(f"  ultralytics {ultralytics.__version__}")
    except ImportError:
        print("[ERRO] pip install ultralytics"); sys.exit(1)

    # Garante dataset base (ou cria do zero se real-only)
    garantir_dataset(args.real_only)

    # Injeta frames anotados reais
    if args.annotated:
        ann = Path(args.annotated)
        if not ann.exists():
            print(f"[ERRO] Pasta não encontrada: {ann}"); sys.exit(1)
        n_tr, n_va = ingerir_anotados(
            ann,
            YOLO_DST / "images",
            YOLO_DST / "labels",
        )
        if n_tr == 0 and args.real_only:
            print("[ERRO] Nenhum frame anotado válido encontrado.")
            print("       Verifique se o LabelImg salvou em formato YOLO (.txt)")
            sys.exit(1)

    # Regera yaml com caminho absoluto (evita FileNotFoundError no Windows)
    yaml = gerar_yaml(YOLO_DST)
    print(f"  YAML: {yaml}")

    n_total_tr = len(list((YOLO_DST/"images"/"train").glob("*.jpg")))
    n_total_va = len(list((YOLO_DST/"images"/"val").glob("*.jpg")))
    print(f"  Total: {n_total_tr} train / {n_total_va} val\n")

    best_pt  = treinar(yaml, args.epochs, args.batch, args.patience)
    onnx_p   = exportar(best_pt)
    copiar(onnx_p)
    metricas(best_pt)

    print(f"\n  [DONE] {(time.time()-t0)/60:.0f} min")
    print(f"  → python carro-autonomo.py --cam")