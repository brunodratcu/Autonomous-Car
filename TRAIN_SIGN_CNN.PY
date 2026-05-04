"""
================================================================
  TREINAR CNN DE PLACAS — SPRINT 2
  ─────────────────────────────────────────────────────────────
  Treina uma CNN leve para classificar placas de trânsito.

  ESTRUTURA DO DATASET (crie antes de treinar):
    dataset/
      STOP/        → fotos da placa STOP
      LEFT/        → fotos da placa VIRE À ESQUERDA
      RIGHT/       → fotos da placa VIRE À DIREITA
      STRAIGHT/    → fotos da placa SIGA EM FRENTE
      DELIVERY/    → fotos da placa ENTREGA
      YIELD/       → fotos da placa DÊ PREFERÊNCIA

  RECOMENDADO: 150–300 imagens por classe
  Formatos aceitos: .jpg .jpeg .png .bmp

  INSTALAÇÃO:
    pip install tensorflow opencv-python numpy scikit-learn

  USO:
    python TRAIN_SIGN_CNN.py                  ← treina com defaults
    python TRAIN_SIGN_CNN.py --epochs 60      ← mais épocas
    python TRAIN_SIGN_CNN.py --dataset ./meu_dataset

  SAÍDA:
    sign_model.tflite   ← modelo leve para inferência no carro
    sign_labels.txt     ← uma classe por linha
    training_report.png ← curvas de loss e accuracy
================================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2

# ── Silencia logs TF antes do import ──────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[ERRO] TensorFlow não encontrado.")
    print("       pip install tensorflow scikit-learn")
    sys.exit(1)

# ================================================================
#  CONFIGURAÇÃO
# ================================================================

IMG_SIZE   = 96          # pixels (96×96)
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
VAL_SPLIT  = 0.20
SEED       = 42

DATASET_DIR  = "./dataset"
MODEL_PATH   = "./sign_model.tflite"
LABELS_PATH  = "./sign_labels.txt"
REPORT_PATH  = "./training_report.png"
KERAS_PATH   = "./sign_model.keras"   # salvo antes da conversão

# ── Augmentação de dados ──────────────────────────────────────────
AUG_BRIGHTNESS = 0.30   # ±30% brilho
AUG_CONTRAST   = 0.25
AUG_ROTATION   = 15     # graus
AUG_ZOOM       = 0.15


# ================================================================
#  1. CARREGAR DATASET
# ================================================================

def carregar_dataset(dataset_dir):
    """
    Lê imagens de subpastas (uma por classe).
    Retorna (X: float32 [0,1], y: int, classes: list[str])
    """
    if not os.path.isdir(dataset_dir):
        print(f"[ERRO] Dataset não encontrado: {dataset_dir}")
        print("       Crie a estrutura: dataset/STOP/ dataset/LEFT/ ...")
        sys.exit(1)

    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if len(classes) < 2:
        print(f"[ERRO] Precisa de ao menos 2 classes no dataset (encontrado: {classes})")
        sys.exit(1)

    print(f"[DATA] Classes detectadas: {classes}")

    X, y = [], []
    ext_validas = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for idx, classe in enumerate(classes):
        pasta = os.path.join(dataset_dir, classe)
        arquivos = [
            f for f in os.listdir(pasta)
            if os.path.splitext(f)[1].lower() in ext_validas
        ]
        print(f"  [{idx}] {classe}: {len(arquivos)} imagens")

        for fname in arquivos:
            path = os.path.join(pasta, fname)
            img  = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img.astype(np.float32) / 255.0)
            y.append(idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[DATA] Total: {len(X)} imagens  |  Shape: {X.shape}")
    return X, y, classes


# ================================================================
#  2. AUMENTAÇÃO (pipeline tf.data)
# ================================================================

def build_augmentation():
    """Camada de data augmentation aplicada só no treino."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(AUG_ROTATION / 360.0),
        layers.RandomZoom(AUG_ZOOM),
        layers.RandomBrightness(AUG_BRIGHTNESS),
        layers.RandomContrast(AUG_CONTRAST),
    ], name="augmentation")


# ================================================================
#  3. ARQUITETURA CNN
#  MobileNetV2-style com depthwise separable convolutions.
#  ~800 K parâmetros → roda bem em CPU.
# ================================================================

def build_model(num_classes):
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    aug = build_augmentation()(inp)   # só ativo no treino

    # ── Bloco 1 ──────────────────────────────────────────────────
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(aug)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.MaxPooling2D(2)(x)       # 48×48

    # ── Bloco 2: Depthwise Separable ─────────────────────────────
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.Conv2D(64, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.MaxPooling2D(2)(x)       # 24×24

    # ── Bloco 3 ──────────────────────────────────────────────────
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.Conv2D(128, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.MaxPooling2D(2)(x)       # 12×12

    # ── Bloco 4 ──────────────────────────────────────────────────
    x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.Conv2D(256, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
    x = layers.GlobalAveragePooling2D()(x)

    # ── Classificador ────────────────────────────────────────────
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SignCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ================================================================
#  4. TREINO
# ================================================================

def treinar(dataset_dir, epochs):
    print("\n" + "=" * 56)
    print("  TREINAMENTO CNN — CLASSIFICAÇÃO DE PLACAS")
    print("=" * 56)

    # 4.1 Carregar dados
    X, y, classes = carregar_dataset(dataset_dir)
    num_classes = len(classes)

    # 4.2 Split treino/validação
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y
    )
    print(f"[SPLIT] Treino: {len(X_tr)}  |  Val: {len(X_val)}")

    # 4.3 Class weights (compensa desbalanceamento)
    cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=y_tr)
    class_weight = dict(enumerate(cw))
    print(f"[CW] {class_weight}")

    # 4.4 tf.data pipeline
    def make_ds(Xd, yd, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((Xd, yd))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(Xd), seed=SEED)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_tr  = make_ds(X_tr, y_tr, shuffle=True)
    ds_val = make_ds(X_val, y_val)

    # 4.5 Modelo
    model = build_model(num_classes)
    model.summary()

    # 4.6 Callbacks
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=12,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=6,
            min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            KERAS_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
    ]

    # 4.7 Treinar
    print(f"\n[TRAIN] Iniciando — {epochs} épocas máx / batch {BATCH_SIZE}")
    hist = model.fit(
        ds_tr,
        validation_data=ds_val,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=cbs,
        verbose=1
    )

    # 4.8 Relatório final
    best_acc = max(hist.history["val_accuracy"])
    print(f"\n[RESULT] Melhor val_accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")

    # 4.9 Gráfico
    _salvar_grafico(hist)

    # 4.10 Converter para TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(MODEL_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"[TFLITE] Salvo: {MODEL_PATH}  ({len(tflite_model)//1024} KB)")

    # 4.11 Salvar labels
    with open(LABELS_PATH, "w") as f:
        f.write("\n".join(classes))
    print(f"[LABELS] Salvo: {LABELS_PATH}")
    print(f"[LABELS] Classes: {classes}")

    print("\n[OK] Treinamento concluído!")
    print(f"     Copie '{MODEL_PATH}' e '{LABELS_PATH}' para a pasta do carro.\n")

    return model, classes


# ================================================================
#  5. GRÁFICO DE TREINAMENTO
# ================================================================

def _salvar_grafico(hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("CNN — Treinamento de Placas", fontsize=14, fontweight="bold")

        # Accuracy
        axes[0].plot(hist.history["accuracy"],     label="Treino",   color="#2E75B6", lw=2)
        axes[0].plot(hist.history["val_accuracy"], label="Val",      color="#E74C3C", lw=2, linestyle="--")
        axes[0].set_title("Acurácia"); axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Acurácia"); axes[0].legend(); axes[0].grid(alpha=0.3)

        # Loss
        axes[1].plot(hist.history["loss"],     label="Treino",  color="#2E75B6", lw=2)
        axes[1].plot(hist.history["val_loss"], label="Val",     color="#E74C3C", lw=2, linestyle="--")
        axes[1].set_title("Loss"); axes[1].set_xlabel("Época")
        axes[1].set_ylabel("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(REPORT_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Gráfico salvo: {REPORT_PATH}")
    except Exception as e:
        print(f"[PLOT] Matplotlib não disponível ({e}) — gráfico ignorado")


# ================================================================
#  6. UTILITÁRIO: GERAR DATASET SINTÉTICO PARA TESTES
#  Cria imagens coloridas simples quando não há fotos reais.
# ================================================================

def gerar_dataset_sintetico(dst="./dataset", n_por_classe=80):
    """
    Gera imagens sintéticas para permitir testar o treinamento
    mesmo sem fotos reais. Substitua depois por imagens reais!
    """
    specs = {
        "STOP":     {"cor": (50, 50, 200),  "forma": "octogono"},
        "LEFT":     {"cor": (200, 100, 30), "forma": "seta_esq"},
        "RIGHT":    {"cor": (30, 100, 200), "forma": "seta_dir"},
        "STRAIGHT": {"cor": (30, 180, 80),  "forma": "seta_cima"},
        "DELIVERY": {"cor": (180, 60, 180), "forma": "quadrado"},
        "YIELD":    {"cor": (200, 200, 30), "forma": "triangulo"},
    }

    print(f"[SYN] Gerando dataset sintético em '{dst}' ...")
    np.random.seed(42)

    for classe, spec in specs.items():
        pasta = os.path.join(dst, classe)
        os.makedirs(pasta, exist_ok=True)
        cor = spec["cor"]

        for i in range(n_por_classe):
            img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 240
            # ruído leve
            img = img + np.random.randint(-20, 20, img.shape, dtype=np.int16).clip(0, 255).astype(np.uint8)

            cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
            sz = IMG_SIZE // 3 + np.random.randint(-8, 8)

            forma = spec["forma"]
            if forma == "octogono":
                pts = np.array([
                    [cx - sz//2, cy - sz], [cx + sz//2, cy - sz],
                    [cx + sz,    cy - sz//2], [cx + sz, cy + sz//2],
                    [cx + sz//2, cy + sz], [cx - sz//2, cy + sz],
                    [cx - sz,    cy + sz//2], [cx - sz, cy - sz//2],
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], cor)
            elif forma in ("seta_esq", "seta_dir", "seta_cima"):
                cv2.rectangle(img, (cx-sz, cy-sz), (cx+sz, cy+sz), cor, -1)
                # seta branca simples
                if forma == "seta_esq":
                    cv2.arrowedLine(img, (cx+sz//2, cy), (cx-sz//2, cy), (255,255,255), 3, tipLength=0.4)
                elif forma == "seta_dir":
                    cv2.arrowedLine(img, (cx-sz//2, cy), (cx+sz//2, cy), (255,255,255), 3, tipLength=0.4)
                else:
                    cv2.arrowedLine(img, (cx, cy+sz//2), (cx, cy-sz//2), (255,255,255), 3, tipLength=0.4)
            elif forma == "triangulo":
                pts = np.array([[cx, cy-sz], [cx-sz, cy+sz], [cx+sz, cy+sz]], dtype=np.int32)
                cv2.fillPoly(img, [pts], cor)
            else:
                cv2.rectangle(img, (cx-sz, cy-sz), (cx+sz, cy+sz), cor, -1)

            # variação de brilho
            fator = 0.7 + np.random.rand() * 0.6
            img = np.clip(img.astype(np.float32) * fator, 0, 255).astype(np.uint8)

            cv2.imwrite(os.path.join(pasta, f"{i:04d}.png"), img)

    print(f"[SYN] Dataset sintético criado: {n_por_classe} imagens × {len(specs)} classes")


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Treinar CNN de placas")
    p.add_argument("--dataset", default=DATASET_DIR, help="Pasta do dataset")
    p.add_argument("--epochs",  type=int, default=EPOCHS, help="Épocas máx")
    p.add_argument("--sintetico", action="store_true",
                   help="Gerar dataset sintético antes de treinar (para testes)")
    args = p.parse_args()

    if args.sintetico:
        gerar_dataset_sintetico(args.dataset)

    treinar(args.dataset, args.epochs)