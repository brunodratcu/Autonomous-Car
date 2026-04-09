# infer_sign_cnn.py — versão otimizada para CPU sem GPU
# Troca TensorFlow por TFLite: ~40MB RAM vs ~800MB do TF completo

import cv2
import numpy as np
import os

IMG_W = 64
IMG_H = 64
MODEL_PATH  = "./models/sign_cnn.tflite"   # modelo convertido (ver abaixo)
MODEL_H5    = "./models/sign_cnn.h5"       # fallback se tflite não existir
LABELS_PATH = "./labels.txt"


def load_labels(path=LABELS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _converter_para_tflite():
    """
    Converte sign_cnn.h5 → sign_cnn.tflite uma única vez.
    TFLite ocupa ~40MB de RAM vs ~800MB do TensorFlow completo.
    """
    import tensorflow as tf
    print("[CONV] Convertendo .h5 → .tflite (uma vez só)...")
    model     = tf.keras.models.load_model(MODEL_H5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Quantização INT8 opcional: reduz modelo de ~4MB para ~1MB
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"[CONV] Salvo em {MODEL_PATH}")
    del model, converter, tflite_model
    import gc; gc.collect()


def load_model_and_labels():
    """
    Carrega o interpreter TFLite (leve) em vez do Keras completo.
    Se .tflite não existir, converte automaticamente do .h5.
    """
    if not os.path.exists(MODEL_PATH):
        if os.path.exists(MODEL_H5):
            _converter_para_tflite()
        else:
            raise FileNotFoundError(
                f"Modelo não encontrado: {MODEL_PATH} nem {MODEL_H5}\n"
                "Treine primeiro com: python train_sign_cnn.py"
            )

    # TFLite interpreter — não carrega CUDA, não reserva GPU
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=2)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=2)

    interpreter.allocate_tensors()

    # Pré-busca índices de entrada/saída (evita lookup a cada frame)
    in_idx  = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    labels = load_labels()
    return interpreter, labels, in_idx, out_idx


# Buffer reutilizável: evita realocar np.float32 a cada frame
_INPUT_BUF = np.zeros((1, IMG_H, IMG_W, 3), dtype=np.float32)

def classify_sign_crop(crop_bgr, interpreter, labels, in_idx, out_idx, conf_min=0.82):
    """
    Inferência com TFLite reutilizando o buffer de entrada.
    ~5x menos RAM e ~2x mais rápido que model.predict() no Keras.

    Retorna: (label, conf, probs_array)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0, np.zeros(len(labels))

    # Redimensiona direto no buffer pré-alocado (sem alocação extra)
    resized = cv2.resize(crop_bgr, (IMG_W, IMG_H))
    np.divide(resized, 255.0, out=_INPUT_BUF[0], casting="unsafe")

    interpreter.set_tensor(in_idx, _INPUT_BUF)
    interpreter.invoke()

    probs = interpreter.get_tensor(out_idx)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = labels[idx] if idx < len(labels) else "unknown"

    if conf < conf_min:
        return "unknown", conf, probs
    return label, conf, probs


if __name__ == "__main__":
    interp, labels, in_idx, out_idx = load_model_and_labels()
    img = cv2.imread("./samples/test_crop.png")
    if img is None:
        print("Imagem de teste não encontrada.")
    else:
        label, conf, _ = classify_sign_crop(img, interp, labels, in_idx, out_idx)
        print("Classe:", label, "| Confiança:", f"{conf:.3f}")