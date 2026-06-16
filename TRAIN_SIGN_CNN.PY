"""
================================================================
  TRAIN_SIGN_CNN.py — Dual-pipeline CNN  v3
  ─────────────────────────────────────────────────────────────
  8 classes:
    Placas  → prep_sign()     (símbolo binário)
      Stop | Esquerda | Direita | SemRetorno | Verde
    Obstáculos → prep_obstacle()  (RGB+CLAHE)
      Cone | Carro | Pessoa

  Gera após treino:
    sign_model.tflite       ← modelo compacto
    sign_labels.txt         ← ordem das classes
    sign_thresholds.txt     ← threshold ótimo por classe (F1)
    training_report.png     ← curvas treino/validação
    confusion_matrix.png    ← onde a rede erra

  USO:
    python TRAIN_SIGN_CNN.py --dataset ./dataset_v3
    python TRAIN_SIGN_CNN.py --test foto.jpg
================================================================
"""

import os, sys, argparse, glob
import cv2, numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, applications
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    print("Execute: pip install tensorflow scikit-learn"); sys.exit(1)

# ================================================================
#  CONFIGURAÇÃO
# ================================================================

DATASET_DIR  = "./dataset_v3"
MODEL_PATH   = "./sign_model.tflite"
LABELS_PATH  = "./sign_labels.txt"
KERAS_PATH   = "./sign_model.keras"
REPORT_PATH  = "./training_report.png"
CONF_PATH    = "./confusion_matrix.png"
THRESH_PATH  = "./sign_thresholds.txt"

IMG_SIZE    = 96
BATCH_SIZE  = 32
EPOCHS_1    = 12
EPOCHS_2    = 60
LR_1        = 1e-3
LR_2        = 5e-5
SEED        = 42

# Obstáculos usam RGB; placas usam símbolo
OBSTACLE_CLASSES = {"Cone", "Carro", "Pessoa"}
SIGN_CLASSES     = {"Stop", "Esquerda", "Direita", "SemRetorno", "Verde"}

LABEL_TO_ACTION = {
    "Stop":       "STOP",
    "Esquerda":   "LEFT",
    "Direita":    "RIGHT",
    "SemRetorno": "STRAIGHT",
    "Verde":      "SPEED_UP",
    "Cone":       "OBSTACLE",
    "Carro":      "OBSTACLE",
    "Pessoa":     "OBSTACLE",
}

# ================================================================
#  DUAL PIPELINE
# ================================================================

def prep_sign(img_bgr, sz=96):
    """
    Placas — símbolo binário.
    Remove 8% de borda, grayscale, CLAHE, threshold adaptativo.
    CNN aprende a FORMA, não a cor.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((sz, sz, 3), np.float32)
    h, w = img_bgr.shape[:2]
    py, px = max(1, int(h*.08)), max(1, int(w*.08))
    roi = img_bgr[py:h-py, px:w-px]
    if roi.size == 0: roi = img_bgr
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hh, ww = gray.shape
    sc = sz / max(hh, ww)
    nh, nw = max(1, int(hh*sc)), max(1, int(ww*sc))
    rs = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    c  = np.zeros((sz, sz), np.uint8)
    c[(sz-nh)//2:(sz-nh)//2+nh, (sz-nw)//2:(sz-nw)//2+nw] = rs
    c  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(c)
    t  = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 13, 5)
    t  = cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    return cv2.cvtColor(t, cv2.COLOR_GRAY2RGB).astype(np.float32)


def prep_obstacle(img_bgr, sz=96):
    """
    Obstáculos — RGB completo preservado.
    Resize + CLAHE no canal L.
    CNN aprende cor e silhueta do cone/carro/pessoa.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((sz, sz, 3), np.float32)
    img_r = cv2.resize(img_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
    lab   = cv2.cvtColor(img_r, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(lab[:,:,0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32)


def preprocess(img_bgr, cls_name, sz=96):
    if cls_name in OBSTACLE_CLASSES:
        return prep_obstacle(img_bgr, sz)
    return prep_sign(img_bgr, sz)


# ================================================================
#  CARREGAR DATASET
# ================================================================

def carregar_dataset(dataset_dir, sz):
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.isdir(train_dir):
        train_dir = dataset_dir

    all_classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
        and d in (SIGN_CLASSES | OBSTACLE_CLASSES)
    ])
    if not all_classes:
        print(f"[ERRO] Nenhuma classe reconhecida em {train_dir}")
        print(f"       Esperadas: {sorted(SIGN_CLASSES|OBSTACLE_CLASSES)}")
        sys.exit(1)

    labels = sorted(all_classes)   # label = nome da pasta
    l2i    = {l: i for i, l in enumerate(labels)}

    print(f"\n[DATA] {len(labels)} classes: {labels}")
    print(f"\n  {'Classe':<14} {'Pipeline':<12} {'Ação'}")
    print("  " + "-"*40)
    for cls in labels:
        pipe = "RGB+CLAHE" if cls in OBSTACLE_CLASSES else "Símbolo"
        print(f"  {cls:<14} {pipe:<12} {LABEL_TO_ACTION.get(cls,'—')}")

    X_all, y_all = [], []
    contagem = {cls: 0 for cls in labels}

    for split in ["train", "valid", "test"]:
        sdir = os.path.join(dataset_dir, split)
        if not os.path.isdir(sdir):
            sdir = train_dir if split == "train" else None
        if not sdir or not os.path.isdir(sdir): continue

        for cls in labels:
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir): continue
            for f in (glob.glob(f"{cdir}/*.jpg") +
                      glob.glob(f"{cdir}/*.png") +
                      glob.glob(f"{cdir}/*.jpeg")):
                img = cv2.imread(f)
                if img is None: continue
                X_all.append(preprocess(img, cls, sz))
                y_all.append(l2i[cls])
                contagem[cls] += 1

    print(f"\n  {'Classe':<14} {'Imgs':>6}")
    for cls in labels:
        bar = "█" * (contagem[cls]//10)
        print(f"  {cls:<14} {contagem[cls]:>6}  {bar}")

    if not X_all:
        print("[ERRO] Nenhuma imagem carregada."); sys.exit(1)

    X = np.array(X_all, np.float32)
    y = np.array(y_all, np.int32)
    print(f"\n[DATA] Total: {len(X)} imagens")
    return X, y, labels


# ================================================================
#  MODELO
# ================================================================

def construir_modelo(n, sz):
    base = applications.MobileNetV2(
        input_shape=(sz, sz, 3), include_top=False,
        weights="imagenet", pooling="avg"
    )
    base.trainable = False
    inp = keras.Input(shape=(sz, sz, 3), name="imagem")
    x   = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.12),
        layers.RandomBrightness(0.20),
        layers.RandomContrast(0.15),
    ], name="aug")(inp)
    x   = layers.Lambda(
        lambda i: applications.mobilenet_v2.preprocess_input(i),
        name="preprocess")(x)
    x   = base(x, training=False)
    x   = layers.Dropout(0.35)(x)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.25)(x)
    out = layers.Dense(n, activation="softmax", name="saida")(x)
    return keras.Model(inp, out, name="SignObstCNN_v3")


# ================================================================
#  THRESHOLD ADAPTATIVO — maximiza F1 por classe
# ================================================================

def calcular_thresholds(modelo, X_v, y_v, labels):
    print("\n[THRESHOLD] Calculando threshold ótimo por classe (máximo F1)...")
    probs = modelo.predict(X_v, verbose=0)
    thresholds = {}
    for i, lbl in enumerate(labels):
        col  = probs[:, i]
        mask = (y_v == i).astype(int)
        best_t, best_f1 = 0.50, 0.0
        for t in np.arange(0.25, 0.96, 0.01):
            preds = (col >= t).astype(int)
            tp = ((preds==1)&(mask==1)).sum()
            fp = ((preds==1)&(mask==0)).sum()
            fn = ((preds==0)&(mask==1)).sum()
            prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
            f1   = 2*prec*rec/(prec+rec+1e-9)
            if f1 > best_f1: best_f1=f1; best_t=t
        thresholds[lbl] = round(float(best_t), 2)
        print(f"  '{lbl}'  t={best_t:.2f}  F1={best_f1:.3f}")
    with open(THRESH_PATH, "w") as f:
        for lbl, t in thresholds.items():
            f.write(f"{lbl},{t:.2f}\n")
    print(f"[OK] {THRESH_PATH}")
    return thresholds


# ================================================================
#  MATRIZ DE CONFUSÃO
# ================================================================

def plotar_confusao(modelo, X_v, y_v, labels):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        probs  = modelo.predict(X_v, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        cm     = confusion_matrix(y_v, y_pred, labels=list(range(len(labels))))
        cm_n   = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("Predito", fontsize=12); ax.set_ylabel("Real", fontsize=12)
        ax.set_title("Matriz de Confusão (recall por classe)\n"
                     "Diagonal = acertos | Fora = erros",
                     fontsize=12, fontweight="bold")
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = cm_n[i,j]; r = cm[i,j]
                c = "white" if v > 0.6 else "black"
                ax.text(j, i, f"{v:.0%}\n({r})",
                        ha="center", va="center", fontsize=9,
                        color=c, fontweight="bold" if i==j else "normal")
        plt.tight_layout()
        plt.savefig(CONF_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[CONFUSÃO] {CONF_PATH}")

        print("\n[CONFUSÃO] Análise — onde o modelo erra:")
        for i, lbl in enumerate(labels):
            total  = cm[i].sum()
            recall = cm[i,i] / max(total, 1)
            erros  = [(labels[j], cm[i,j]) for j in range(len(labels))
                      if j != i and cm[i,j] > 0]
            erros.sort(key=lambda x: -x[1])
            err_str = ", ".join([f"→'{e}' {n}x" for e,n in erros[:2]])
            status  = "✅" if recall>=.85 else ("⚠️" if recall>=.65 else "❌")
            print(f"  {status} '{lbl:<12}'  recall={recall*100:.1f}%  {err_str or 'sem erros'}")

        print("\n  ✅≥85%  ⚠️65-85%  ❌<65% → adicionar mais imagens dessa classe")
    except Exception as e:
        print(f"[CONFUSÃO] {e}")


# ================================================================
#  TREINO
# ================================================================

def treinar(source, sz):
    print("="*60)
    print("  SIGN+OBSTACLE CNN v3 — 8 classes")
    print(f"  {source}")
    print("="*60)

    X, y, labels = carregar_dataset(source, sz)
    n = len(labels)

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.18, stratify=y, random_state=SEED)
    print(f"\n[SPLIT] Treino: {len(X_tr)}  |  Validação: {len(X_v)}")

    pesos_vals = compute_class_weight("balanced", classes=np.arange(n), y=y_tr)
    pesos = dict(enumerate(pesos_vals))
    print(f"[PESOS] { {labels[k]: f'{v:.2f}' for k,v in pesos.items()} }")

    ds_tr = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
             .shuffle(len(X_tr), seed=SEED).batch(BATCH_SIZE)
             .prefetch(tf.data.AUTOTUNE))
    ds_v  = (tf.data.Dataset.from_tensor_slices((X_v, y_v))
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    modelo = construir_modelo(n, sz)
    modelo.summary()

    ckpt = keras.callbacks.ModelCheckpoint(
        KERAS_PATH, monitor="val_accuracy", save_best_only=True, verbose=0)

    # Fase 1 — head only
    print(f"\n[FASE 1] {EPOCHS_1} épocas — base congelada")
    modelo.compile(optimizer=keras.optimizers.Adam(LR_1),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    h1 = modelo.fit(ds_tr, validation_data=ds_v, epochs=EPOCHS_1,
                    class_weight=pesos, callbacks=[ckpt], verbose=1)
    print(f"[FASE 1] val_accuracy: {max(h1.history['val_accuracy'])*100:.1f}%")

    # Fase 2 — fine-tune
    print(f"\n[FASE 2] {EPOCHS_2} épocas — fine-tune últimas 30 camadas")
    base = modelo.get_layer("mobilenetv2_1.00_96")
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    modelo.compile(optimizer=keras.optimizers.Adam(LR_2),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    cbs2 = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                       restore_best_weights=True, verbose=1),
        ckpt,
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=5, min_lr=1e-7, verbose=1),
    ]
    h2 = modelo.fit(ds_tr, validation_data=ds_v, epochs=EPOCHS_2,
                    class_weight=pesos, callbacks=cbs2, verbose=1)
    best = max(h2.history["val_accuracy"])
    print(f"\n[RESULTADO] val_accuracy: {max(max(h1.history['val_accuracy']),best)*100:.1f}%")

    _grafico({
        "accuracy":     h1.history["accuracy"]     + h2.history["accuracy"],
        "val_accuracy": h1.history["val_accuracy"] + h2.history["val_accuracy"],
        "loss":         h1.history["loss"]          + h2.history["loss"],
        "val_loss":     h1.history["val_loss"]      + h2.history["val_loss"],
    }, labels, EPOCHS_1)

    plotar_confusao(modelo, X_v, y_v, labels)
    calcular_thresholds(modelo, X_v, y_v, labels)
    _exportar(modelo, labels)


# ================================================================
#  EXPORT
# ================================================================

def _exportar(modelo, labels):
    print("\n[EXPORT] Convertendo para TFLite...")
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    with open(MODEL_PATH, "wb") as f: f.write(tfl)
    with open(LABELS_PATH, "w")  as f: f.write("\n".join(labels))
    print(f"[OK] {MODEL_PATH}  ({len(tfl)//1024} KB)")
    print(f"[OK] {LABELS_PATH} → {labels}")
    print("\n[MAPA FINAL]")
    for lbl in labels:
        pipe = "RGB" if lbl in OBSTACLE_CLASSES else "símbolo"
        print(f"  '{lbl}'  [{pipe}]  →  {LABEL_TO_ACTION.get(lbl,'—')}")
    print(f"\n  Copie para a pasta do SIGN_DETECTOR.py:")
    print(f"  sign_model.tflite  sign_labels.txt  sign_thresholds.txt")


# ================================================================
#  GRÁFICO
# ================================================================

def _grafico(hist, labels, split_ep=0):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"Sign+Obstacle CNN v3 — {labels}",
                     fontsize=9, fontweight="bold")
        best_ep = int(np.argmax(hist["val_accuracy"]))
        best_ac = hist["val_accuracy"][best_ep]
        for i, (m, t) in enumerate([("accuracy","Acurácia"),("loss","Loss")]):
            ax  = axes[i]; eps = range(1, len(hist[m])+1)
            ax.plot(eps, hist[m],          label="treino",
                    color="#2E75B6", lw=2)
            ax.plot(eps, hist[f"val_{m}"], label="validação",
                    color="#E74C3C", lw=2, ls="--")
            if split_ep:
                ax.axvline(split_ep, color="gray", alpha=0.5, ls=":", lw=1.5)
                ax.text(split_ep+.3, ax.get_ylim()[0],
                        "fine-tune→", fontsize=7, color="gray")
            ax.set_title(t); ax.set_xlabel("Época")
            ax.legend(); ax.grid(alpha=0.3)
            if m == "accuracy":
                ax.set_ylim(0, 1)
                ax.axvline(best_ep+1, color="green", alpha=0.4, ls=":")
                ax.annotate(f"best {best_ac:.3f}",
                            xy=(best_ep+1, best_ac),
                            xytext=(best_ep+2, best_ac-.08),
                            fontsize=8, color="green")
        plt.tight_layout()
        plt.savefig(REPORT_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[GRÁFICO] {REPORT_PATH}")
    except Exception as e:
        print(f"[GRÁFICO] {e}")


# ================================================================
#  MODO --test  (localiza placa na cena antes de classificar)
# ================================================================

def testar(caminho):
    for arq in [MODEL_PATH, LABELS_PATH]:
        if not os.path.isfile(arq):
            print(f"[ERRO] {arq} — treine primeiro."); sys.exit(1)

    with open(LABELS_PATH) as f:
        labels = [l.strip() for l in f if l.strip()]

    thresholds = {}
    if os.path.isfile(THRESH_PATH):
        with open(THRESH_PATH) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p)==2: thresholds[p[0]] = float(p[1])
    else:
        thresholds = {l: 0.60 for l in labels}

    try:
        import tflite_runtime.interpreter as tr
        interp = tr.Interpreter(model_path=MODEL_PATH)
    except ImportError:
        interp = tf.lite.Interpreter(model_path=MODEL_PATH)

    interp.allocate_tensors()
    ind  = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    sz   = int(ind["shape"][1])

    img = cv2.imread(caminho)
    if img is None: print(f"[ERRO] {caminho}"); sys.exit(1)

    h, w = img.shape[:2]
    print(f"\n{'='*58}")
    print(f"  {os.path.basename(caminho)}  ({w}×{h}px)")

    # Tenta localizar placa via HSV na imagem
    crops = _localizar_placas(img)
    if crops:
        print(f"  [{len(crops)} região(ões) detectada(s) via HSV]")
    else:
        # Fallback: usa imagem inteira
        crops = [(img, "cena inteira")]
        print(f"  [Nenhuma região detectada — classificando cena inteira]")

    print(f"{'='*58}")

    for crop_img, desc in crops[:3]:
        print(f"\n  Região: {desc}")
        print(f"  {'-'*50}")

        # Testa com ambos os pipelines
        for pipe_name, tensor_img in [
            ("Símbolo", prep_sign(crop_img, sz)),
            ("RGB",     prep_obstacle(crop_img, sz)),
        ]:
            interp.set_tensor(ind["index"], np.expand_dims(tensor_img, 0))
            interp.invoke()
            p    = interp.get_tensor(outd["index"])[0]
            top3 = np.argsort(p)[::-1][:3]
            print(f"  [{pipe_name}]")
            for i, idx in enumerate(top3):
                lbl  = labels[idx]; conf = p[idx]
                t    = thresholds.get(lbl, 0.60)
                acao = LABEL_TO_ACTION.get(lbl, "—")
                mk   = "◄" if i==0 else " "
                dim  = "" if conf>=t else f"  (abaixo do threshold {t:.0%})"
                print(f"    {mk} '{lbl}'  {conf*100:5.1f}%  → {acao}{dim}")

    # Painel visual: original + símbolo + RGB
    orig_96 = cv2.resize(img, (sz*2, sz*2))
    if crops and crops[0][1] != "cena inteira":
        crop_img = crops[0][0]
    else:
        crop_img = img
    sym_v = cv2.cvtColor(prep_sign(crop_img,sz).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rgb_v = cv2.cvtColor(prep_obstacle(crop_img,sz).astype(np.uint8), cv2.COLOR_RGB2BGR)
    sym_v = cv2.resize(sym_v, (sz*2, sz*2))
    rgb_v = cv2.resize(rgb_v, (sz*2, sz*2))
    panel = np.hstack([orig_96, sym_v, rgb_v])
    for x, txt in [(2,"ORIGINAL"),(sz*2+2,"SÍMBOLO"),(sz*4+2,"RGB")]:
        cv2.putText(panel, txt, (x,14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,80), 1)
    cv2.imshow(f"TEST — {os.path.basename(caminho)}", panel)
    cv2.waitKey(0); cv2.destroyAllWindows()


def _localizar_placas(img):
    """
    Localiza regiões com cores de placa via HSV.
    Retorna lista de (crop_bgr, descricao).
    """
    h, w = img.shape[:2]
    # ROI: ignora rodapé e céu extremo
    y0, y1 = int(h*.05), int(h*.85)
    roi = img[y0:y1]

    hsv = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (5,5), 0)
    ranges = [
        (np.array([0,  70,50]), np.array([14, 255,255])),  # vermelho
        (np.array([162,70,50]), np.array([180,255,255])),  # vermelho wrap
        (np.array([90, 60,40]), np.array([130,255,255])),  # azul
        (np.array([15,100,70]), np.array([36, 255,255])),  # amarelo
        (np.array([5, 100,80]), np.array([20, 255,255])),  # laranja (cone)
    ]
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        area = cv2.contourArea(cnt)
        if area < 500: continue
        peri  = cv2.arcLength(cnt, True)
        circ  = 4*np.pi*area/max(peri**2,1)
        hull  = cv2.convexHull(cnt)
        solid = area/max(cv2.contourArea(hull),1)
        if circ < 0.35 or solid < 0.65: continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        prop = bw/max(bh,1)
        if not (0.25 <= prop <= 3.5): continue
        pad  = int(max(bw,bh)*.08)
        x1   = max(0, x-pad); y1_ = max(0, y+y0-pad)
        x2   = min(w, x+bw+pad); y2 = min(h, y+y0+bh+pad)
        crop = img[y1_:y2, x1:x2]
        if crop.size == 0: continue
        desc = f"bbox=({x1},{y1_},{x2},{y2})  area={area:.0f}  circ={circ:.2f}"
        crops.append((crop, desc))
    return crops


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="Pasta do dataset (ex: ./dataset_v3)")
    ap.add_argument("--img",  type=int, default=IMG_SIZE)
    ap.add_argument("--test", type=str, default=None, metavar="FOTO.JPG")
    a = ap.parse_args()

    if a.test:
        testar(a.test)
    elif a.dataset:
        treinar(a.dataset, a.img)
    elif os.path.isdir(DATASET_DIR):
        print(f"[AUTO] {DATASET_DIR}")
        treinar(DATASET_DIR, a.img)
    else:
        ap.print_help()
        print(f"\n[ERRO] Use: python TRAIN_SIGN_CNN.py --dataset ./dataset_v3")