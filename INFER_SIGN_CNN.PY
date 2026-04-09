import cv2
import numpy as np
import tensorflow as tf

IMG_W = 64
IMG_H = 64
MODEL_PATH = "./models/sign_cnn.h5"
LABELS_PATH = "./labels.txt"

def load_labels(labels_path=LABELS_PATH):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels()
    return model, labels

def preprocess_crop(crop_bgr):
    img = cv2.resize(crop_bgr, (IMG_W, IMG_H))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_sign_crop(crop_bgr, model, labels, conf_min=0.85):
    """
    Retorna:
        label: str
        conf: float
        probs: np.ndarray
    """
    x = preprocess_crop(crop_bgr)
    pred = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    label = labels[idx]

    if conf < conf_min:
        return "unknown", conf, pred

    return label, conf, pred

if __name__ == "__main__":
    model, labels = load_model_and_labels()
    img = cv2.imread("./samples/test_crop.png")

    if img is None:
        print("Imagem de teste não encontrada.")
    else:
        label, conf, _ = classify_sign_crop(img, model, labels)
        print("Classe:", label)
        print("Confiança:", conf)