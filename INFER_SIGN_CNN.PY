"""
================================================================
  INFER_SIGN_CNN.py — Módulo de Inferência TFLite
  ─────────────────────────────────────────────────────────────
  Carrega sign_model.tflite e expõe:
    • load_model_and_labels()  → interpreter, labels, in_idx, out_idx
    • classify_sign_crop()     → (label, conf, bbox_normalizado)
    • preprocess_crop()        → tensor float32 normalizado
  ─────────────────────────────────────────────────────────────
  INSTALAÇÃO (escolha um):
    pip install tflite-runtime       ← mais leve
    pip install tensorflow           ← fallback completo
================================================================
"""

import os
import cv2
import numpy as np

# ── Importação flexível TFLite ────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    _BACKEND = "tflite_runtime"
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        _BACKEND = "tensorflow"
    except ImportError:
        raise ImportError(
            "Nenhum backend TFLite encontrado.\n"
            "Execute: pip install tflite-runtime\n"
            "      ou pip install tensorflow"
        )

# ── Caminhos padrão ──────────────────────────────────────────────
_DEFAULT_MODEL  = os.path.join(os.path.dirname(__file__), "sign_model.tflite")
_DEFAULT_LABELS = os.path.join(os.path.dirname(__file__), "sign_labels.txt")

# ── Tamanho de entrada padrão (deve coincidir com o treino) ───────
_IMG_SIZE = 96   # pixels


# ================================================================
#  CARREGAMENTO
# ================================================================

def load_model_and_labels(
    model_path: str  = _DEFAULT_MODEL,
    labels_path: str = _DEFAULT_LABELS
):
    """
    Carrega o modelo TFLite e a lista de labels.

    Retorna:
        interpreter  — objeto TFLite Interpreter
        labels       — list[str] com nomes das classes
        input_idx    — índice do tensor de entrada
        output_idx   — índice do tensor de saída

    Lança:
        FileNotFoundError se os arquivos não existirem.
        RuntimeError      se o modelo for inválido.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            f"Execute TRAIN_SIGN_CNN.py para treinar e gerar o arquivo."
        )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"Labels não encontradas: {labels_path}\n"
            f"Execute TRAIN_SIGN_CNN.py para gerar sign_labels.txt."
        )

    # Carrega labels
    with open(labels_path, "r") as f:
        labels = [l.strip() for l in f if l.strip()]

    # Carrega interpreter
    if _BACKEND == "tflite_runtime":
        interp = tflite.Interpreter(model_path=model_path)
    else:
        interp = tflite.Interpreter(model_path=model_path)

    interp.allocate_tensors()
    in_details  = interp.get_input_details()
    out_details = interp.get_output_details()

    in_idx  = in_details[0]["index"]
    out_idx = out_details[0]["index"]

    # Valida shape de entrada
    expected_shape = in_details[0]["shape"]   # [1, H, W, C]
    if len(expected_shape) == 4:
        global _IMG_SIZE
        _IMG_SIZE = int(expected_shape[1])    # atualiza tamanho de entrada

    return interp, labels, in_idx, out_idx


# ================================================================
#  PRÉ-PROCESSAMENTO
# ================================================================

def preprocess_crop(crop_bgr: np.ndarray, img_size: int = None) -> np.ndarray:
    """
    Pré-processa um recorte BGR para tensor de entrada da CNN.

    Pipeline:
      1. Resize para img_size × img_size
      2. BGR → RGB
      3. Normalização [0, 1]  (float32)
      4. Adiciona dimensão batch: shape = [1, H, W, 3]

    Retorna tensor float32 ou None se crop inválido.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    sz = img_size or _IMG_SIZE

    # Resize
    resized = cv2.resize(crop_bgr, (sz, sz), interpolation=cv2.INTER_AREA)

    # BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalização
    tensor = rgb.astype(np.float32) / 255.0

    # Batch dim
    return np.expand_dims(tensor, axis=0)


# ================================================================
#  INFERÊNCIA
# ================================================================

def classify_sign_crop(
    crop_bgr: np.ndarray,
    interpreter,
    labels: list,
    input_idx: int,
    output_idx: int,
    conf_min: float = 0.75,
    img_size: int   = None
) -> tuple:
    """
    Classifica um recorte de placa usando o modelo TFLite.

    Parâmetros:
        crop_bgr    — imagem BGR do recorte da placa
        interpreter — objeto TFLite Interpreter
        labels      — lista de nomes de classe
        input_idx   — índice do tensor de entrada
        output_idx  — índice do tensor de saída
        conf_min    — confiança mínima para aceitar (padrão: 0.75)
        img_size    — tamanho de resize (padrão: automático do modelo)

    Retorna:
        (label: str, confidence: float, probs: np.ndarray)
        label = "unknown" se confiança < conf_min ou falha
    """
    tensor = preprocess_crop(crop_bgr, img_size)
    if tensor is None:
        return "unknown", 0.0, np.zeros(len(labels))

    try:
        interpreter.set_tensor(input_idx, tensor)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_idx)[0]   # shape [num_classes]
    except Exception as e:
        return "unknown", 0.0, np.zeros(len(labels))

    best_idx  = int(np.argmax(probs))
    best_conf = float(probs[best_idx])

    if best_conf < conf_min or best_idx >= len(labels):
        return "unknown", best_conf, probs

    return labels[best_idx], best_conf, probs


# ================================================================
#  TOP-K (útil para debug)
# ================================================================

def top_k_results(probs: np.ndarray, labels: list, k: int = 3) -> list:
    """
    Retorna os k melhores resultados como lista de (label, conf).

    Exemplo:
        >>> top_k_results(probs, labels, k=3)
        [("STOP", 0.91), ("RIGHT", 0.06), ("LEFT", 0.02)]
    """
    indices = np.argsort(probs)[::-1][:k]
    return [(labels[i], float(probs[i])) for i in indices if i < len(labels)]


# ================================================================
#  TESTE RÁPIDO (execução direta)
# ================================================================

if __name__ == "__main__":
    import sys

    print(f"[INFER] Backend: {_BACKEND}")

    try:
        interp, labels, in_idx, out_idx = load_model_and_labels()
        print(f"[INFER] Modelo carregado — {len(labels)} classes: {labels}")
        print(f"[INFER] Tamanho de entrada: {_IMG_SIZE}×{_IMG_SIZE}")
    except FileNotFoundError as e:
        print(f"[INFER] {e}")
        sys.exit(1)

    # Testa com uma imagem de arquivo (se passada como arg)
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img  = cv2.imread(path)
        if img is None:
            print(f"[INFER] Imagem não encontrada: {path}")
            sys.exit(1)

        label, conf, probs = classify_sign_crop(
            img, interp, labels, in_idx, out_idx, conf_min=0.0
        )
        print(f"\n[INFER] Arquivo: {path}")
        print(f"[INFER] Resultado: {label}  ({conf*100:.1f}%)")
        print(f"[INFER] Top-3: {top_k_results(probs, labels, 3)}")

    else:
        # Teste com imagem sintética
        img_dummy = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.rectangle(img_dummy, (10, 10), (86, 86), (50, 50, 200), -1)
        label, conf, probs = classify_sign_crop(
            img_dummy, interp, labels, in_idx, out_idx, conf_min=0.0
        )
        print(f"\n[INFER] Teste sintético: {label}  ({conf*100:.1f}%)")
        print(f"[INFER] Top-3: {top_k_results(probs, labels, 3)}")