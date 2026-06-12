"""
================================================================
  TRAIN_SIGN_CNN.py — Transfer Learning MobileNetV2
  ─────────────────────────────────────────────────────────────
  USO:
    python TRAIN_SIGN_CNN.py --dataset ./dataset_final
    python TRAIN_SIGN_CNN.py --test ./foto.jpg

  Dataset esperado (pasta train com subpastas por classe):
    dataset_final/train/S/       ← Stop/Pare
    dataset_final/train/L/       ← Vira Esquerda
    dataset_final/train/N/       ← Sem Retorno
    dataset_final/train/G/       ← Semáforo Verde
    dataset_final/train/Cone/    ← Cone de trânsito
    dataset_final/train/Car/     ← Carro na pista
    dataset_final/train/People/  ← Pessoa na pista
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

DATASET_DIR = "./dataset_final"
MODEL_PATH  = "./sign_model.tflite"
LABELS_PATH = "./sign_labels.txt"
KERAS_PATH  = "./sign_model.keras"
REPORT_PATH = "./training_report.png"

IMG_SIZE    = 96
BATCH_SIZE  = 32
EPOCHS_1    = 12
EPOCHS_2    = 60
LR_1        = 1e-3
LR_2        = 5e-5
SEED        = 42

# ── Mapeamento classe → label (cmd serial) ──────────────────────
CLASS_TO_CMD = {
    # Placas de trânsito
    "S":      "S",   # Stop/Pare        → STOP
    "L":      "L",   # Turn Left        → LEFT
    "N":      "N",   # No U-Turn        → STRAIGHT
    "G":      "G",   # Green Light      → SPEED_UP
    # Obstáculos na pista
    "Cone":   "O",   # Cone             → OBSTACLE
    "Car":    "O",   # Carro            → OBSTACLE
    "People": "O",   # Pessoa           → OBSTACLE
}

# ── Mapeamento label → ação do carro ────────────────────────────
LABEL_TO_ACTION = {
    "S": "STOP",
    "L": "LEFT",
    "N": "STRAIGHT",
    "G": "SPEED_UP",
    "O": "OBSTACLE",   # cone/carro/pessoa → para até sumir
}

LABEL_NAMES = {
    "S": "Stop/Pare",
    "L": "Vira Esq",
    "N": "Sem Retorno",
    "G": "Semáforo Verde",
    "O": "Obstáculo (Cone/Carro/Pessoa)",
}


# ================================================================
#  PRÉ-PROCESSADOR — idêntico ao SIGN_DETECTOR.py
# ================================================================

def prep_sign(img_bgr, sz=96):
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((sz,sz,3), np.float32)
    h, w = img_bgr.shape[:2]
    py, px = max(1,int(h*.12)), max(1,int(w*.12))
    roi = img_bgr[py:h-py, px:w-px]
    if roi.size == 0: roi = img_bgr
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hh, ww = gray.shape
    sc = sz/max(hh,ww)
    nh, nw = max(1,int(hh*sc)), max(1,int(ww*sc))
    rs = cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
    c  = np.zeros((sz,sz), np.uint8)
    c[(sz-nh)//2:(sz-nh)//2+nh,(sz-nw)//2:(sz-nw)//2+nw] = rs
    c  = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(4,4)).apply(c)
    t  = cv2.adaptiveThreshold(c,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,13,5)
    t  = cv2.morphologyEx(t,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    return cv2.cvtColor(t,cv2.COLOR_GRAY2RGB).astype(np.float32)


# ================================================================
#  CARREGAR DATASET
# ================================================================

def carregar_dataset(dataset_dir, sz):
    train_dir = os.path.join(dataset_dir, "train")
    # Aceita estrutura plana também (sem subpasta train/)
    if not os.path.isdir(train_dir):
        train_dir = dataset_dir

    classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and d in CLASS_TO_CMD
    ])
    if not classes:
        print(f"[ERRO] Nenhuma classe reconhecida em {train_dir}")
        print(f"       Classes esperadas: {list(CLASS_TO_CMD.keys())}")
        sys.exit(1)

    cmds    = sorted(set(CLASS_TO_CMD[c] for c in classes))
    c2i     = {c:i for i,c in enumerate(cmds)}

    print(f"\n[DATA] Classes: {classes}")
    print(f"[DATA] Labels únicos: {cmds}")
    print(f"\n  {'Classe':<12} {'Label':>6}  {'Ação':<12}  {'Imgs':>6}")
    print("  " + "-"*40)

    X_all, y_all = [], []
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            # Try flat structure
            split_dir = train_dir if split == "train" else None
        if not split_dir or not os.path.isdir(split_dir):
            continue

        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir): continue
            files = (glob.glob(f"{cls_dir}/*.jpg") +
                     glob.glob(f"{cls_dir}/*.png") +
                     glob.glob(f"{cls_dir}/*.jpeg"))
            for f in files:
                img = cv2.imread(f)
                if img is None: continue
                rgb = prep_sign(img, sz)
                X_all.append(rgb)
                y_all.append(c2i[CLASS_TO_CMD[cls]])

    # Print summary
    for cls in classes:
        cmd  = CLASS_TO_CMD[cls]
        acao = LABEL_TO_ACTION.get(cmd,"—")
        n    = sum(1 for y in y_all if y == c2i[cmd] and cls in CLASS_TO_CMD
                   and CLASS_TO_CMD[cls] == cmd)
        print(f"  {cls:<12} '{cmd}'   {acao:<12}")

    if not X_all:
        print("[ERRO] Nenhuma imagem carregada."); sys.exit(1)

    X = np.array(X_all, np.float32)
    y = np.array(y_all, np.int32)
    print(f"\n[DATA] Total: {len(X)} imagens | {len(cmds)} labels")
    return X, y, cmds


# ================================================================
#  MODELO — MobileNetV2
# ================================================================

def construir_modelo(n, sz):
    base = applications.MobileNetV2(
        input_shape=(sz,sz,3), include_top=False,
        weights="imagenet", pooling="avg"
    )
    base.trainable = False

    inp = keras.Input(shape=(sz,sz,3), name="imagem")
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
    return keras.Model(inp, out, name="SignCNN")


# ================================================================
#  TREINO EM DUAS FASES
# ================================================================

def treinar(source, sz):
    print("="*60)
    print(f"  SIGN CNN — Transfer Learning MobileNetV2")
    print(f"  {source}")
    print("="*60)

    X, y, cmds = carregar_dataset(source, sz)
    n = len(cmds)

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.18, stratify=y, random_state=SEED)
    print(f"\n[SPLIT] Treino: {len(X_tr)}  |  Validação: {len(X_v)}")

    pesos_vals = compute_class_weight("balanced", classes=np.arange(n), y=y_tr)
    pesos      = dict(enumerate(pesos_vals))
    print(f"[PESOS] { {cmds[k]: f'{v:.2f}' for k,v in pesos.items()} }")

    ds_tr = (tf.data.Dataset.from_tensor_slices((X_tr,y_tr))
             .shuffle(len(X_tr),seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    ds_v  = (tf.data.Dataset.from_tensor_slices((X_v,y_v))
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    modelo = construir_modelo(n, sz)
    modelo.summary()

    ckpt = keras.callbacks.ModelCheckpoint(
        KERAS_PATH, monitor="val_accuracy", save_best_only=True, verbose=0)

    # Fase 1 — head only
    print(f"\n[FASE 1] {EPOCHS_1} épocas — base congelada")
    modelo.compile(optimizer=keras.optimizers.Adam(LR_1),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = modelo.fit(ds_tr, validation_data=ds_v, epochs=EPOCHS_1,
                    class_weight=pesos, callbacks=[ckpt], verbose=1)
    print(f"[FASE 1] val_accuracy: {max(h1.history['val_accuracy'])*100:.1f}%")

    # Fase 2 — fine-tune últimas 30 camadas
    print(f"\n[FASE 2] {EPOCHS_2} épocas — fine-tune")
    base = modelo.get_layer("mobilenetv2_1.00_96")
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    modelo.compile(optimizer=keras.optimizers.Adam(LR_2),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    print(f"[FASE 2] val_accuracy: {best*100:.1f}%")
    print(f"\n[RESULTADO FINAL] {max(max(h1.history['val_accuracy']),best)*100:.1f}%")

    _grafico({
        "accuracy":     h1.history["accuracy"]     + h2.history["accuracy"],
        "val_accuracy": h1.history["val_accuracy"] + h2.history["val_accuracy"],
        "loss":         h1.history["loss"]          + h2.history["loss"],
        "val_loss":     h1.history["val_loss"]      + h2.history["val_loss"],
    }, cmds, EPOCHS_1)
    _exportar(modelo, cmds)


# ================================================================
#  EXPORT
# ================================================================

def _exportar(modelo, cmds):
    print("\n[EXPORT] Convertendo para TFLite...")
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    with open(MODEL_PATH,"wb") as f: f.write(tfl)
    with open(LABELS_PATH,"w")  as f: f.write("\n".join(cmds))
    print(f"[OK] {MODEL_PATH}  ({len(tfl)//1024} KB)")
    print(f"[OK] {LABELS_PATH} → {cmds}")
    print("\n[MAPA] Label → Ação:")
    for lbl in cmds:
        print(f"  '{lbl}'  {LABEL_TO_ACTION.get(lbl,'—'):<12}  {LABEL_NAMES.get(lbl,'')}")


# ================================================================
#  GRÁFICO
# ================================================================

def _grafico(hist, cmds, split_ep=0):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(13,4))
        fig.suptitle(f"Sign CNN — {cmds}", fontsize=10, fontweight="bold")
        best_ep = int(np.argmax(hist["val_accuracy"]))
        best_ac = hist["val_accuracy"][best_ep]
        for i,(m,t) in enumerate([("accuracy","Acurácia"),("loss","Loss")]):
            ax=axes[i]; eps=range(1,len(hist[m])+1)
            ax.plot(eps,hist[m],label="treino",color="#2E75B6",lw=2)
            ax.plot(eps,hist[f"val_{m}"],label="validação",color="#E74C3C",lw=2,ls="--")
            if split_ep: ax.axvline(split_ep,color="gray",alpha=0.5,ls=":",lw=1.5)
            ax.set_title(t); ax.set_xlabel("Época"); ax.legend(); ax.grid(alpha=0.3)
            if m=="accuracy":
                ax.set_ylim(0,1); ax.axvline(best_ep+1,color="green",alpha=0.4,ls=":")
                ax.annotate(f"best {best_ac:.3f}",xy=(best_ep+1,best_ac),
                            xytext=(best_ep+2,best_ac-.08),fontsize=8,color="green")
        plt.tight_layout(); plt.savefig(REPORT_PATH,dpi=120,bbox_inches="tight")
        plt.close(); print(f"[GRÁFICO] {REPORT_PATH}")
    except Exception as e: print(f"[GRÁFICO] {e}")


# ================================================================
#  TESTE COM FOTO
# ================================================================

def testar(caminho):
    for arq in [MODEL_PATH,LABELS_PATH]:
        if not os.path.isfile(arq):
            print(f"[ERRO] {arq} — treine primeiro."); sys.exit(1)
    with open(LABELS_PATH) as f:
        cmds=[l.strip() for l in f if l.strip()]
    try:
        import tflite_runtime.interpreter as tr
        interp=tr.Interpreter(model_path=MODEL_PATH)
    except ImportError:
        interp=tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    ind=interp.get_input_details()[0]; outd=interp.get_output_details()[0]
    sz=int(ind["shape"][1])
    img=cv2.imread(caminho)
    if img is None: print(f"[ERRO] {caminho}"); sys.exit(1)
    tensor=np.expand_dims(prep_sign(img,sz),0)
    interp.set_tensor(ind["index"],tensor); interp.invoke()
    p=interp.get_tensor(outd["index"])[0]
    top3=np.argsort(p)[::-1][:3]
    print(f"\n[TESTE] {caminho}")
    print("-"*50)
    for i,idx in enumerate(top3):
        lbl=cmds[idx]; mk="←" if i==0 else " "
        print(f"  {mk} '{lbl}'  {p[idx]*100:5.1f}%  "
              f"{LABEL_NAMES.get(lbl,lbl)}  →  {LABEL_TO_ACTION.get(lbl,'—')}")
    cv2.putText(img,f"'{cmds[top3[0]]}' {p[top3[0]]*100:.0f}%",
                (10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,220,80),2)
    cv2.imshow("Resultado",img); cv2.waitKey(0); cv2.destroyAllWindows()


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__=="__main__":
    ap=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset",default=None,help="Pasta do dataset")
    ap.add_argument("--img",type=int,default=IMG_SIZE)
    ap.add_argument("--test",type=str,default=None,metavar="FOTO.JPG")
    a=ap.parse_args()
    if a.test:
        testar(a.test)
    elif a.dataset:
        treinar(a.dataset,a.img)
    elif os.path.isdir(DATASET_DIR):
        print(f"[AUTO] {DATASET_DIR}"); treinar(DATASET_DIR,a.img)
    else:
        ap.print_help()
        print(f"\n[ERRO] Use: python TRAIN_SIGN_CNN.py --dataset ./dataset_novo")