"""
================================================================
  INFER_SIGN_CNN.py v3.0 — Inferência com foco em símbolo
  ─────────────────────────────────────────────────────────────
  Integra PREP_SIGN: o mesmo pré-processamento do treino
  é aplicado em cada crop antes da inferência.

  USO NO TERMINAL:
    python INFER_SIGN_CNN.py foto.jpg
    python INFER_SIGN_CNN.py foto1.jpg foto2.jpg foto3.jpg
    python INFER_SIGN_CNN.py foto.jpg --top 9 --conf 0.0
    python INFER_SIGN_CNN.py foto.jpg --no-window
    python INFER_SIGN_CNN.py foto.jpg --show-prep   ← mostra símbolo extraído

  IMPORTADO POR carro-autonomo.py:
    from INFER_SIGN_CNN import load_model_and_labels, classify_sign_crop
================================================================
"""

import os, sys, cv2, numpy as np

_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_DIR, "sign_model.tflite")
LABELS_PATH = os.path.join(_DIR, "sign_labels.txt")

# Importa pré-processador
sys.path.insert(0, _DIR)
try:
    from PREP_SIGN import prep_sign, prep_sign_visual
    _PREP_OK = True
except ImportError:
    _PREP_OK = False
    def prep_sign(img_bgr, sz=96):
        img = cv2.resize(img_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    def prep_sign_visual(img_bgr, sz=96):
        return cv2.resize(img_bgr, (sz, sz))

LABEL_TO_ACTION = {
    "B": "SLOW_DOWN", "F": "SLOW_DOWN", "K": "STOP",
    "N": "STRAIGHT",  "P": "STRAIGHT",  "S": "STOP",
    "T": "STRAIGHT",  "U": "SPEED_UP",  "W": "SLOW_DOWN",
    "Y": "YIELD",
}

LABEL_NAMES = {
    "B": "Bump/Road_Work",    "F": "Speed_Limit_90",
    "K": "No_Stopping",       "N": "No_U-Turn",
    "P": "No_Parking",        "S": "Stop/Pare",
    "T": "Sentido_Obrig.",    "U": "Speed_Limit_120",
    "W": "Speed_Limit_40",    "Y": "Pedestrian/Pedestre",
}


# ================================================================
#  CARREGAR MODELO
# ================================================================

def load_model_and_labels(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[CNN] Modelo não encontrado: {model_path}\n"
            f"      Execute: python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado"
        )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"[CNN] Labels não encontrados: {labels_path}")

    with open(labels_path) as f:
        labels = [l.strip() for l in f if l.strip()]

    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)

    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    sz      = interp.get_input_details()[0]["shape"][1]

    prep_status = "PREP_SIGN ativo" if _PREP_OK else "fallback resize"
    print(f"[CNN] Modelo OK — {len(labels)} classes: {labels} | {sz}×{sz} | {prep_status}",
          flush=True)
    return interp, labels, in_idx, out_idx


# ================================================================
#  CLASSIFICAR CROP
# ================================================================

def classify_sign_crop(crop_bgr, interp, labels, in_idx, out_idx,
                        conf_min=0.60):
    """
    Classifica crop BGR usando o pipeline de símbolo.
    Retorna: (label, confiança, top3)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0, []

    sz     = interp.get_input_details()[0]["shape"][1]
    rgb    = prep_sign(crop_bgr, sz)          # foco no símbolo
    batch  = np.expand_dims(rgb, axis=0)

    interp.set_tensor(in_idx, batch)
    interp.invoke()
    probs = interp.get_tensor(out_idx)[0]

    top_idx  = np.argsort(probs)[::-1][:3]
    top3     = [(labels[i], float(probs[i])) for i in top_idx]
    best_lbl  = top3[0][0]
    best_conf = top3[0][1]

    if best_conf < conf_min:
        return "unknown", best_conf, top3

    return best_lbl, best_conf, top3


def label_to_action(label):
    return LABEL_TO_ACTION.get(label)


# ================================================================
#  TESTE NO TERMINAL
# ================================================================

def _testar_imagem(caminho, top_n, conf_min, interp, labels,
                   in_idx, out_idx, show_window, show_prep):
    img = cv2.imread(caminho)
    if img is None:
        print(f"[ERRO] Não encontrou: {caminho}"); return

    h, w  = img.shape[:2]
    sz    = interp.get_input_details()[0]["shape"][1]

    # Pré-processa e roda inferência
    rgb   = prep_sign(img, sz)
    batch = np.expand_dims(rgb, axis=0)
    interp.set_tensor(in_idx, batch)
    interp.invoke()
    probs = interp.get_tensor(out_idx)[0]

    top_n   = min(top_n, len(labels))
    top_idx = np.argsort(probs)[::-1][:top_n]

    print(f"\n{'='*58}")
    print(f"  {os.path.basename(caminho)}  ({w}×{h}px)")
    if not _PREP_OK:
        print(f"  [AVISO] PREP_SIGN não encontrado — resultado pode variar")
    print(f"{'='*58}")
    print(f"  {'#':>2}  {'Lbl':>4}  {'Conf':>7}  {'Classe':<22}  Ação")
    print(f"  {'-'*55}")

    for i, idx in enumerate(top_idx):
        lbl   = labels[idx]
        conf  = probs[idx]
        nome  = LABEL_NAMES.get(lbl, lbl)
        acao  = LABEL_TO_ACTION.get(lbl, "—")
        marca = "◄" if i == 0 else " "
        dim   = "" if conf >= conf_min else "  (< limiar)"
        print(f"  {i+1:>2}{marca} '{lbl}'  {conf*100:6.1f}%  {nome:<22}  {acao}{dim}")

    best_lbl  = labels[top_idx[0]]
    best_conf = probs[top_idx[0]]
    print(f"\n  Limiar : {conf_min*100:.0f}%")
    if best_conf >= conf_min:
        print(f"  AÇÃO   : '{best_lbl}' ({best_conf*100:.1f}%)  →  {LABEL_TO_ACTION.get(best_lbl,'—')}")
    else:
        print(f"  AÇÃO   : confiança baixa ({best_conf*100:.1f}%) — não dispara")
    print(f"{'='*58}\n")

    if show_window or show_prep:
        prep_vis = prep_sign_visual(img, sz)
        orig_vis = cv2.resize(img, (sz, sz))

        if show_prep:
            painel = np.hstack([orig_vis, prep_vis])
            cv2.putText(painel, "ORIGINAL", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,220,80), 1)
            cv2.putText(painel, "SIMBOLO",  (sz+2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,220,80), 1)
        else:
            painel = orig_vis.copy()

        cv2.putText(painel, f"'{best_lbl}' {best_conf*100:.0f}%",
                    (4, sz-6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,220,80), 1)
        cv2.imshow(f"CNN — {os.path.basename(caminho)}", painel)
        print("  [Pressione qualquer tecla]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Testa placas com CNN v3.0")
    ap.add_argument("imagens",     nargs="+", metavar="FOTO.JPG")
    ap.add_argument("--top",       type=int,   default=3)
    ap.add_argument("--conf",      type=float, default=0.60)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--show-prep", action="store_true",
                    help="Mostra símbolo extraído lado a lado com original")
    ap.add_argument("--model",     default=MODEL_PATH)
    ap.add_argument("--labels",    default=LABELS_PATH)
    args = ap.parse_args()

    try:
        interp, labels, in_idx, out_idx = load_model_and_labels(args.model, args.labels)
    except (FileNotFoundError, ImportError) as e:
        print(e); sys.exit(1)

    for caminho in args.imagens:
        _testar_imagem(caminho, args.top, args.conf,
                       interp, labels, in_idx, out_idx,
                       show_window=not args.no_window,
                       show_prep=args.show_prep)