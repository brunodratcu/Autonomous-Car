"""
================================================================
  PREPARE_YOLO_DATASET.py
  ─────────────────────────────────────────────────────────────
  Converte dataset de CLASSIFICAÇÃO para formato de DETECÇÃO
  compatível com YOLOv8.

  POR QUÊ ISSO É NECESSÁRIO:
    - Seu dataset atual é de classificação:
        dataset_v3/train/Stop/img.jpg   (só a imagem, sem bbox)
    - YOLO precisa de detecção:
        yolo_dataset/images/train/img.jpg     (imagem)
        yolo_dataset/labels/train/img.txt     (classe cx cy w h)

  COMO FUNCIONA A AUTO-ANOTAÇÃO:
    Datasets de classificação são geralmente crops já centrados
    no objeto. Então uma bbox centralizada de 85% da imagem é
    uma aproximação válida para bootstrap.

    Label gerada: "class_id 0.50 0.50 0.85 0.85"
    (centro da imagem, 85% de largura e altura)

    O YOLO aprende a localizar mesmo com bbox ligeiramente
    maior que o objeto — o treinamento calibra isso.

  ENTRADA:
    dataset_v3/
      train/
        Stop/ Esquerda/ Direita/ ...  <- classificação

  SAÍDA:
    yolo_dataset/
      images/
        train/  *.jpg
        val/    *.jpg
      labels/
        train/  *.txt   ("class_id 0.5 0.5 0.85 0.85")
        val/    *.txt
      dataset.yaml

  USO:
    python PREPARE_YOLO_DATASET.py
    python PREPARE_YOLO_DATASET.py --src ./dataset_v3 --dst ./yolo_dataset
    python PREPARE_YOLO_DATASET.py --src ./dataset_v3 --dst ./yolo_dataset --verify

  DEPOIS DESSE SCRIPT:
    pip install ultralytics
    yolo train model=yolov8n.pt data=./yolo_dataset/dataset.yaml \
          epochs=100 imgsz=640 batch=16
    yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
    # Copiar:
    # runs/detect/train/weights/best.onnx → ./models/sign_detector.onnx

  DEPENDÊNCIAS:
    pip install opencv-python numpy scikit-learn
================================================================
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path

import numpy as np
import cv2

# ================================================================
#  CONFIGURAÇÃO
# ================================================================

# Classes — ordem DEVE ser idêntica a CNN_CLASSES em carro-autonomo.py
CLASSES = [
    "Stop", "Esquerda", "Direita", "SemRetorno",
    "Verde", "Cone", "Carro", "Pessoa",
]
NUM_CLASSES = len(CLASSES)
IDX         = {c: i for i, c in enumerate(CLASSES)}

EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Bbox auto-gerada: centro 0.5/0.5, cobertura de 85%
# Datasets de classificação costumam ser crops centrados no objeto.
# Ajuste se as suas imagens tiverem muito contexto ao redor.
BBOX_CX = 0.50
BBOX_CY = 0.50
BBOX_W  = 0.85
BBOX_H  = 0.85

# Val split quando não existe val/ separada
VAL_FRAC = 0.20
SEED     = 42

# ================================================================
#  DETECÇÃO DE ESTRUTURA DE ORIGEM
# ================================================================

def detectar_estrutura(src: Path) -> dict:
    """
    Detecta como o dataset de origem está organizado.
    Retorna dict com chaves 'modo', 'classes_encontradas', 'por_split'.
    """
    if not src.exists():
        return {"modo": "nao_existe"}

    subdirs = {d.name: d for d in src.iterdir() if d.is_dir()}

    # Caso A: tem train/ (com ou sem val/)
    if "train" in subdirs:
        train_dir  = subdirs["train"]
        val_dir    = subdirs.get("val")

        classes_train = _encontrar_classes(train_dir)
        classes_val   = _encontrar_classes(val_dir) if val_dir else {}

        return {
            "modo":              "split" if val_dir else "train_only",
            "train_dir":         train_dir,
            "val_dir":           val_dir,
            "classes_train":     classes_train,
            "classes_val":       classes_val,
            "classes_encontradas": sorted(
                set(classes_train) | set(classes_val)),
        }

    # Caso B: pastas de classe diretamente na raiz
    classes_root = _encontrar_classes(src)
    if classes_root:
        return {
            "modo":              "flat",
            "train_dir":         src,
            "val_dir":           None,
            "classes_train":     classes_root,
            "classes_val":       {},
            "classes_encontradas": sorted(classes_root),
        }

    return {"modo": "nao_reconhecido", "subdirs": list(subdirs)}


def _encontrar_classes(pasta: Path | None) -> dict:
    """
    Retorna {nome_classe: Path} para pastas que batem com CLASSES.
    Busca case-insensitive.
    """
    if pasta is None or not pasta.exists():
        return {}
    resultado = {}
    for d in pasta.iterdir():
        if not d.is_dir():
            continue
        for cls in CLASSES:
            if d.name.lower() == cls.lower():
                resultado[cls] = d
                break
    return resultado


def _imagens(pasta: Path) -> list[Path]:
    return [p for p in pasta.iterdir() if p.suffix.lower() in EXTS]

# ================================================================
#  CONVERSÃO
# ================================================================

def converter(src_dir: str, dst_dir: str, val_frac: float = VAL_FRAC, bbox_w: float = BBOX_W, bbox_h: float = BBOX_H):
    src = Path(src_dir)
    dst = Path(dst_dir)

    print(f"\n  Fonte : {src.resolve()}")
    print(f"  Destino: {dst.resolve()}")

    info = detectar_estrutura(src)
    modo = info.get("modo", "nao_reconhecido")

    print(f"  Estrutura detectada: {modo.upper()}")

    if modo == "nao_existe":
        print(f"\n[ERRO] Diretório não encontrado: {src}")
        sys.exit(1)

    if modo == "nao_reconhecido":
        print(f"\n[ERRO] Estrutura não reconhecida.")
        print(f"       Subdiretórios encontrados: {info.get('subdirs')}")
        print("       Esperado: pasta com subpastas Stop/ Esquerda/ ...")
        sys.exit(1)

    print()

    # ── Coleta imagens por classe e por split ─────────────────────
    if modo in ("split", "train_only"):
        # Usa train/ como fonte principal
        imgs_train: dict[str, list[Path]] = {}
        imgs_val:   dict[str, list[Path]] = {}

        for cls, pasta in info["classes_train"].items():
            imgs = _imagens(pasta)
            random.shuffle(imgs)

            if modo == "split" and cls in info.get("classes_val", {}):
                # val/ já existe — usa diretamente
                imgs_train[cls] = imgs
                imgs_val[cls]   = _imagens(info["classes_val"][cls])
            else:
                # Faz auto-split estratificado
                n_val = max(1, int(len(imgs) * val_frac))
                imgs_val[cls]   = imgs[:n_val]
                imgs_train[cls] = imgs[n_val:]

    else:  # flat
        imgs_train = {}
        imgs_val   = {}
        for cls, pasta in info["classes_train"].items():
            imgs = _imagens(pasta)
            random.shuffle(imgs)
            n_val = max(1, int(len(imgs) * val_frac))
            imgs_val[cls]   = imgs[:n_val]
            imgs_train[cls] = imgs[n_val:]

    # ── Resumo ────────────────────────────────────────────────────
    print(f"  {'Classe':<14s}  {'Train':>6s}  {'Val':>6s}  Status")
    print("  " + "-" * 42)
    total_tr = total_va = 0
    ausentes = []
    for cls in CLASSES:
        n_tr = len(imgs_train.get(cls, []))
        n_va = len(imgs_val.get(cls,   []))
        total_tr += n_tr; total_va += n_va
        status = "OK" if (n_tr + n_va) > 0 else "AUSENTE"
        if status == "AUSENTE":
            ausentes.append(cls)
        print(f"  {cls:<14s}  {n_tr:>6d}  {n_va:>6d}  {status}")
    print(f"\n  Total             {total_tr:>6d}  {total_va:>6d}")

    if ausentes:
        print(f"\n  [AVISO] Classes ausentes: {ausentes}")
        print("          O YOLO será treinado sem elas.")

    # ── Cria estrutura de diretórios ──────────────────────────────
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Copia imagens e gera labels ───────────────────────────────
    print("\n  Copiando imagens e gerando labels...")
    stats = {"train": 0, "val": 0, "erros": 0}

    for split, mapa in [("train", imgs_train), ("val", imgs_val)]:
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split

        for cls, caminhos in mapa.items():
            cls_id = IDX[cls]
            label_linha = f"{cls_id} {BBOX_CX:.4f} {BBOX_CY:.4f} {bbox_w:.4f} {bbox_h:.4f}\n"

            for src_path in caminhos:
                # Nome único: classe_stem_original.jpg
                stem     = f"{cls}_{src_path.stem}"
                img_dst  = img_dir / f"{stem}.jpg"
                lbl_dst  = lbl_dir / f"{stem}.txt"

                # Converte para JPEG se necessário (YOLO prefere jpg)
                if src_path.suffix.lower() == ".jpg":
                    try:
                        shutil.copy2(src_path, img_dst)
                    except Exception as e:
                        print(f"    [ERRO] {src_path}: {e}")
                        stats["erros"] += 1
                        continue
                else:
                    img = cv2.imread(str(src_path))
                    if img is None:
                        stats["erros"] += 1
                        continue
                    cv2.imwrite(str(img_dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Label YOLO
                lbl_dst.write_text(label_linha)
                stats[split] += 1

    print(f"    train: {stats['train']} imagens")
    print(f"    val:   {stats['val']} imagens")
    if stats["erros"]:
        print(f"    erros: {stats['erros']}")

    # ── Gera dataset.yaml ─────────────────────────────────────────
    yaml_path = dst / "dataset.yaml"
    yaml_content = f"""# YOLOv8 Dataset — gerado por PREPARE_YOLO_DATASET.py
path: {dst.resolve().as_posix()}
train: images/train
val:   images/val

nc: {NUM_CLASSES}
names: {CLASSES}
"""
    yaml_path.write_text(yaml_content)
    print(f"\n  [OK] {yaml_path}")
    print(f"  [OK] {dst}/images/  ({stats['train'] + stats['val']} imagens)")
    print(f"  [OK] {dst}/labels/  ({stats['train'] + stats['val']} labels)")

    return dst, stats

# ================================================================
#  VERIFICAÇÃO VISUAL (--verify)
#  Abre as primeiras N imagens de cada classe com a bbox desenhada
# ================================================================

def verificar(dst_dir: str, n_por_classe: int = 3):
    dst     = Path(dst_dir)
    img_dir = dst / "images" / "train"
    lbl_dir = dst / "labels" / "train"

    print(f"\n  Verificação visual — {n_por_classe} imagens por classe")
    print("  Pressione qualquer tecla para avançar, Q para sair.\n")

    for cls in CLASSES:
        cls_id  = IDX[cls]
        # Pega arquivos desta classe
        arquivos = sorted(img_dir.glob(f"{cls}_*.jpg"))[:n_por_classe]
        if not arquivos:
            print(f"  [SKIP] {cls}: nenhuma imagem em train/")
            continue

        for img_path in arquivos:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Lê label e desenha bbox
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                for linha in lbl_path.read_text().strip().splitlines():
                    parts = linha.split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, bw, bh = [float(x) for x in parts]
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{cls} ({cls_id})", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(f"Verificação — {cls}", img)
            k = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if k == ord("q"):
                print("  Verificação interrompida.")
                return

    print("  Verificação concluída.")

# ================================================================
#  MAIN
# ================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Converte dataset de classificação para formato YOLO")
    ap.add_argument("--src",    default="./dataset_v3",
                    help="Dataset de classificação (padrão: ./dataset_v3)")
    ap.add_argument("--dst",    default="./yolo_dataset",
                    help="Destino YOLO (padrão: ./yolo_dataset)")
    ap.add_argument("--val-split", type=float, default=VAL_FRAC,
                    help=f"Fração de val para auto-split (padrão: {VAL_FRAC})")
    ap.add_argument("--verify", action="store_true",
                    help="Abre janela de verificação visual após converter")
    ap.add_argument("--bbox-w", type=float, default=BBOX_W,
                    help=f"Largura da bbox auto-gerada, 0-1 (padrão: {BBOX_W})")
    ap.add_argument("--bbox-h", type=float, default=BBOX_H,
                    help=f"Altura da bbox auto-gerada, 0-1 (padrão: {BBOX_H})")
    args = ap.parse_args()

    random.seed(SEED)

    # Permite override das dimensoes da bbox via argumento
    bbox_w_eff = args.bbox_w
    bbox_h_eff = args.bbox_h

    print("=" * 62)
    print("  PREPARE_YOLO_DATASET")
    print(f"  Classes ({NUM_CLASSES}): {', '.join(CLASSES)}")
    print(f"  Bbox auto: cx={BBOX_CX}  cy={BBOX_CY}  w={bbox_w_eff}  h={bbox_h_eff}")
    print("=" * 62)

    dst, stats = converter(args.src, args.dst,
                           val_frac=args.val_split,
                           bbox_w=bbox_w_eff, bbox_h=bbox_h_eff)

    if args.verify:
        verificar(str(dst))

    total = stats["train"] + stats["val"]
    print("\n" + "=" * 62)
    print("  PRONTO — próximos passos:")
    print()
    print("  1) Instalar ultralytics (se ainda não tiver):")
    print("       pip install ultralytics")
    print()
    print("  2) Treinar YOLOv8n:")
    print(f"       yolo train model=yolov8n.pt \\")
    print(f"             data={Path(args.dst).resolve() / 'dataset.yaml'} \\")
    print(f"             epochs=100 imgsz=640 batch=16 patience=20")
    print()
    print("  3) Exportar para ONNX:")
    print("       yolo export model=runs/detect/train/weights/best.pt \\")
    print("              format=onnx imgsz=640 opset=12")
    print()
    print("  4) Copiar para ./models/:")
    print("       copy runs\\detect\\train\\weights\\best.onnx models\\sign_detector.onnx")
    print()
    print("  5) Executar o carro:")
    print("       python carro-autonomo.py --cam")
    print("=" * 62)


if __name__ == "__main__":
    main()