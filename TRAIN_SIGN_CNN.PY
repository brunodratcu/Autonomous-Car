"""
================================================================
  TRAIN_SIGN_CNN.py v4.0 — Transfer Learning + Foco em Símbolo
  ─────────────────────────────────────────────────────────────
  DIFERENÇAS vs v3.0:
  ─────────────────────────────────────────────────────────────
  1. PRÉ-PROCESSAMENTO via PREP_SIGN
     Entrada: símbolo binário (grayscale → threshold adaptativo)
     O modelo aprende FORMA, não cor/borda.
     Funciona para qualquer placa independente de cor.

  2. SUPORTE A DOIS DATASETS SIMULTÂNEOS
     --dataset ./dataset_tratado     ← dataset original (9 classes)
     --circuito ./dataset_circuito   ← suas placas do circuito
     --ambos                         ← combina os dois
     Quando as placas chegarem: só adicionar --circuito

  3. CLASSES DINÂMICAS
     Detecta automaticamente todas as pastas presentes.
     Novas classes = novas pastas = retreino automático.

  4. MAPEAMENTO CONFIGURÁVEL
     CLASS_TO_CMD pode ser editado sem mexer no código.
     Placas do circuito mapeiam para ações do carro.

  USO:
    # Dataset original
    python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado

    # Suas placas do circuito (quando tiver)
    python TRAIN_SIGN_CNN.py --circuito ./dataset_circuito

    # Ambos combinados
    python TRAIN_SIGN_CNN.py --ambos

    # Testar foto
    python TRAIN_SIGN_CNN.py --test ./foto.jpg
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
    print("Execute: pip install tensorflow scikit-learn")
    sys.exit(1)

# Importa pré-processador
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from PREP_SIGN import prep_sign
    print("[PREP] PREP_SIGN.py carregado — foco em símbolo ativo")
except ImportError:
    print("[PREP] PREP_SIGN.py não encontrado — usando resize simples")
    def prep_sign(img_bgr, sz=96):
        img = cv2.resize(img_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        return rgb

# ================================================================
#  CONFIGURAÇÃO
# ================================================================

DATASET_DIR  = "./dataset_tratado"
CIRCUITO_DIR = "./dataset_circuito"
MODEL_PATH   = "./sign_model.tflite"
LABELS_PATH  = "./sign_labels.txt"
KERAS_PATH   = "./sign_model.keras"
REPORT_PATH  = "./training_report.png"

IMG_SIZE     = 96
BATCH_SIZE   = 32
EPOCHS_1     = 10
EPOCHS_2     = 60
LR_1         = 1e-3
LR_2         = 5e-5
SEED         = 42

# ── Mapeamento classe → comando serial ───────────────────────────
# Dataset original
# ── Dataset novo (DS2 tratado) ──────────────────────────────────
# Pasta: dataset_novo/train/<CMD>/  onde CMD é direto o label
# Não há mapeamento classe→cmd porque as pastas JÁ são os cmds
CLASS_TO_CMD = {
    # Dataset novo — pastas já nomeadas pelo CMD
    "S": "S",   # Stop / Red Traffic Light / No Stopping → STOP
    "L": "L",   # Turn Left                              → LEFT
    "R": "R",   # Turn Right                             → RIGHT
    "Y": "Y",   # Pedestrian Crossing                    → YIELD
    "N": "N",   # No U-Turn / Don't Turn                 → STRAIGHT
    "G": "G",   # Green Traffic Light                    → SPEED_UP
    "W": "W",   # Yellow Traffic Light                   → SLOW_DOWN
    # Dataset original (opcional, se quiser combinar)
    "Stop":            "S",
    "Pedestrian":      "Y",
    "Speed_Limit_40":  "W",
    "Speed_Limit_90":  "W",
    "Speed_Limit_120": "G",
    "No_U-Turn":       "N",
    "No_Stopping":     "S",
    "No_Parking":      "N",
    "Bump":            "W",
    "Road_Work":       "W",
}

# ── Mapeamento label → ação do carro ────────────────────────────
LABEL_TO_ACTION = {
    "S": "STOP",
    "L": "LEFT",
    "R": "RIGHT",
    "Y": "YIELD",
    "N": "STRAIGHT",
    "G": "SPEED_UP",
    "W": "SLOW_DOWN",
}


# ================================================================
#  CARREGAR DATASET
# ================================================================

def _carregar_pasta(dataset_dir: str, sz: int,
                    classes_presentes: list, cmd_to_idx: dict) -> tuple:
    X, y, contagem = [], [], {}

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in classes_presentes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            arquivos = (
                glob.glob(os.path.join(cls_dir, "*.jpg")) +
                glob.glob(os.path.join(cls_dir, "*.png")) +
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
            )
            for caminho in arquivos:
                img = cv2.imread(caminho)
                if img is None:
                    continue
                rgb = prep_sign(img, sz)
                X.append(rgb)
                y.append(cmd_to_idx[CLASS_TO_CMD[cls]])
                contagem[cls] = contagem.get(cls, 0) + 1

    return X, y, contagem


def carregar_dataset(fontes: list, sz: int):
    """
    Carrega um ou mais datasets.
    fontes: lista de caminhos de pasta raiz.
    """
    print(f"\n[DATA] Fontes: {fontes}")

    # Descobre todas as classes presentes em qualquer fonte
    classes_all = set()
    for fonte in fontes:
        train_dir = os.path.join(fonte, "train")
        if not os.path.isdir(train_dir):
            # Aceita estrutura plana (só subpastas direto)
            train_dir = fonte
        if os.path.isdir(train_dir):
            for d in os.listdir(train_dir):
                if os.path.isdir(os.path.join(train_dir, d)) and d in CLASS_TO_CMD:
                    classes_all.add(d)

    classes_presentes = sorted(classes_all)
    if not classes_presentes:
        print(f"[ERRO] Nenhuma classe reconhecida nas fontes: {fontes}")
        print(f"       Classes mapeadas: {list(CLASS_TO_CMD.keys())}")
        sys.exit(1)

    cmds_unicos = sorted(set(CLASS_TO_CMD[c] for c in classes_presentes))
    cmd_to_idx  = {c: i for i, c in enumerate(cmds_unicos)}

    print(f"[DATA] Classes: {classes_presentes}")
    print(f"[DATA] Comandos: {cmds_unicos}")
    print(f"\n  {'Classe':<22} {'Cmd':>4}  {'Ação':<12}")
    print("  " + "-" * 42)
    for cls in classes_presentes:
        cmd  = CLASS_TO_CMD[cls]
        acao = LABEL_TO_ACTION.get(cmd, "—")
        print(f"  {cls:<22} '{cmd}'  {acao}")

    X_all, y_all = [], []
    for fonte in fontes:
        # Tenta estrutura com subpastas train/valid/test
        if os.path.isdir(os.path.join(fonte, "train")):
            Xf, yf, cnt = _carregar_pasta(fonte, sz, classes_presentes, cmd_to_idx)
        else:
            # Estrutura plana: fonte/NomeClasse/*.jpg
            Xf, yf, cnt = _carregar_pasta_plana(fonte, sz, classes_presentes, cmd_to_idx)

        X_all.extend(Xf)
        y_all.extend(yf)

        for cls, n in sorted(cnt.items()):
            print(f"    {cls:<22}: {n}")

    if not X_all:
        print("[ERRO] Nenhuma imagem carregada.")
        sys.exit(1)

    X = np.array(X_all, np.float32)
    y = np.array(y_all, np.int32)
    print(f"\n[DATA] Total: {len(X)} imagens | {len(cmds_unicos)} comandos")
    return X, y, cmds_unicos


def _carregar_pasta_plana(dataset_dir: str, sz: int,
                           classes_presentes: list, cmd_to_idx: dict) -> tuple:
    X, y, contagem = [], [], {}
    for cls in classes_presentes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for caminho in glob.glob(os.path.join(cls_dir, "*.jpg")):
            img = cv2.imread(caminho)
            if img is None:
                continue
            rgb = prep_sign(img, sz)
            X.append(rgb)
            y.append(cmd_to_idx[CLASS_TO_CMD[cls]])
            contagem[cls] = contagem.get(cls, 0) + 1
    return X, y, contagem


# ================================================================
#  MODELO — MobileNetV2 com entrada de símbolo
# ================================================================

def construir_modelo(n_classes: int, sz: int) -> keras.Model:
    base = applications.MobileNetV2(
        input_shape=(sz, sz, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inp = keras.Input(shape=(sz, sz, 3), name="imagem")

    # Augmentação — só no treino
    x = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.12),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="aug")(inp)

    # preprocess_input: [0,255] → [-1,1]
    x = layers.Lambda(
        lambda img: applications.mobilenet_v2.preprocess_input(img),
        name="preprocess"
    )(x)

    x   = base(x, training=False)
    x   = layers.Dropout(0.35)(x)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.25)(x)
    out = layers.Dense(n_classes, activation="softmax", name="saida")(x)

    return keras.Model(inp, out, name="SignCNN_v4")


# ================================================================
#  TREINO EM DUAS FASES
# ================================================================

def treinar(fontes: list, sz: int):
    print("=" * 60)
    print("  SIGN CNN v4.0 — Transfer Learning + Símbolo")
    print(f"  Fontes: {fontes}")
    print("=" * 60)

    X, y, cmds = carregar_dataset(fontes, sz)
    n          = len(cmds)

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.18, stratify=y, random_state=SEED
    )
    print(f"\n[SPLIT] Treino: {len(X_tr)}  |  Validação: {len(X_v)}")

    pesos_vals = compute_class_weight("balanced", classes=np.arange(n), y=y_tr)
    pesos      = dict(enumerate(pesos_vals))

    ds_tr = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
             .shuffle(len(X_tr), seed=SEED)
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    ds_v  = (tf.data.Dataset.from_tensor_slices((X_v, y_v))
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    modelo = construir_modelo(n, sz)
    modelo.summary()

    ckpt = keras.callbacks.ModelCheckpoint(
        KERAS_PATH, monitor="val_accuracy", save_best_only=True, verbose=0
    )

    # ── Fase 1 ───────────────────────────────────────────────────
    print(f"\n[FASE 1] {EPOCHS_1} épocas — head only")
    modelo.compile(optimizer=keras.optimizers.Adam(LR_1),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    h1 = modelo.fit(ds_tr, validation_data=ds_v, epochs=EPOCHS_1,
                    class_weight=pesos, callbacks=[ckpt], verbose=1)
    best1 = max(h1.history["val_accuracy"])
    print(f"[FASE 1] val_accuracy: {best1*100:.1f}%")

    # ── Fase 2: fine-tune ────────────────────────────────────────
    print(f"\n[FASE 2] {EPOCHS_2} épocas — fine-tune últimas 30 camadas")
    base = modelo.get_layer("mobilenetv2_1.00_96")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

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
    best2 = max(h2.history["val_accuracy"])
    print(f"[FASE 2] val_accuracy: {best2*100:.1f}%")
    print(f"\n[RESULTADO] val_accuracy final: {max(best1,best2)*100:.1f}%")

    hist = {
        "accuracy":     h1.history["accuracy"]     + h2.history["accuracy"],
        "val_accuracy": h1.history["val_accuracy"] + h2.history["val_accuracy"],
        "loss":         h1.history["loss"]         + h2.history["loss"],
        "val_loss":     h1.history["val_loss"]     + h2.history["val_loss"],
    }
    _grafico(hist, cmds, split_epoch=EPOCHS_1)
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
    with open(MODEL_PATH, "wb") as f: f.write(tfl)
    with open(LABELS_PATH, "w")  as f: f.write("\n".join(cmds))
    print(f"[OK] {MODEL_PATH}  ({len(tfl)//1024} KB)")
    print(f"[OK] {LABELS_PATH} → {cmds}")

    rev = {}
    for k, v in CLASS_TO_CMD.items():
        rev.setdefault(v, []).append(k)
    print("\n[MAPA] Comando → Classes → Ação:")
    for cmd in sorted(set(CLASS_TO_CMD.values())):
        acao  = LABEL_TO_ACTION.get(cmd, "—")
        nomes = ", ".join(rev.get(cmd, []))
        print(f"  '{cmd}'  {acao:<12}  {nomes}")


# ================================================================
#  GRÁFICO
# ================================================================

def _grafico(hist, cmds, split_epoch=0):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"Sign CNN v4.0 MobileNetV2 — {cmds}", fontsize=10, fontweight="bold")
        best_ep = int(np.argmax(hist["val_accuracy"]))
        best_ac = hist["val_accuracy"][best_ep]
        for i, (m, t) in enumerate([("accuracy","Acurácia"),("loss","Loss")]):
            ax  = axes[i]
            eps = range(1, len(hist[m]) + 1)
            ax.plot(eps, hist[m],          label="treino",    color="#2E75B6", lw=2)
            ax.plot(eps, hist[f"val_{m}"], label="validação", color="#E74C3C", lw=2, ls="--")
            if split_epoch:
                ax.axvline(split_epoch, color="gray", alpha=0.5, ls=":", lw=1.5)
                ax.text(split_epoch+0.3, ax.get_ylim()[0], "fine-tune→", fontsize=7, color="gray")
            ax.set_title(t); ax.set_xlabel("Época"); ax.legend(); ax.grid(alpha=0.3)
            if m == "accuracy":
                ax.set_ylim(0, 1)
                ax.axvline(best_ep+1, color="green", alpha=0.4, ls=":")
                ax.annotate(f"best {best_ac:.3f}", xy=(best_ep+1, best_ac),
                            xytext=(best_ep+2, best_ac-0.08), fontsize=8, color="green")
        plt.tight_layout()
        plt.savefig(REPORT_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[GRÁFICO] {REPORT_PATH}")
    except Exception as e:
        print(f"[GRÁFICO] {e}")


# ================================================================
#  TESTE COM FOTO
# ================================================================

def testar(caminho: str):
    for arq in [MODEL_PATH, LABELS_PATH]:
        if not os.path.isfile(arq):
            print(f"[ERRO] {arq} não encontrado — treine primeiro."); sys.exit(1)

    with open(LABELS_PATH) as f:
        cmds = [l.strip() for l in f if l.strip()]

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
    if img is None:
        print(f"[ERRO] {caminho}"); sys.exit(1)

    tensor = np.expand_dims(prep_sign(img, sz), 0)
    interp.set_tensor(ind["index"], tensor)
    interp.invoke()
    p    = interp.get_tensor(outd["index"])[0]
    top3 = np.argsort(p)[::-1][:3]

    rev = {}
    for k, v in CLASS_TO_CMD.items():
        rev.setdefault(v, []).append(k)

    print(f"\n[TESTE] {caminho}")
    print("-" * 55)
    for i, idx in enumerate(top3):
        cmd   = cmds[idx]
        acao  = LABEL_TO_ACTION.get(cmd, "—")
        nomes = ", ".join(rev.get(cmd, [cmd]))
        marca = " ←" if i == 0 else "  "
        print(f"  {marca} '{cmd}'  {p[idx]*100:5.1f}%  {acao:<12}  {nomes}")

    cv2.putText(img, f"'{cmds[top3[0]]}' {p[top3[0]]*100:.0f}%",
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,220,80), 2)
    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Treina CNN de placas v4.0 — Transfer Learning + Símbolo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--dataset",  default=None,
                    help="Dataset original (ex: ./dataset_tratado)")
    ap.add_argument("--circuito", default=None,
                    help="Dataset do circuito (ex: ./dataset_circuito)")
    ap.add_argument("--ambos",    action="store_true",
                    help="Usa dataset_tratado + dataset_circuito juntos")
    ap.add_argument("--img",      type=int, default=IMG_SIZE,
                    help=f"Resolução px (padrão: {IMG_SIZE})")
    ap.add_argument("--test",     type=str, default=None,
                    help="Testa foto com modelo já treinado")
    a = ap.parse_args()

    if a.test:
        testar(a.test)
    else:
        fontes = []
        if a.ambos:
            fontes = [DATASET_DIR, CIRCUITO_DIR]
        elif a.dataset:
            fontes = [a.dataset]
        elif a.circuito:
            fontes = [a.circuito]
        elif os.path.isdir(DATASET_DIR):
            print(f"[AUTO] Usando: {DATASET_DIR}")
            fontes = [DATASET_DIR]
        else:
            ap.print_help()
            sys.exit(1)

        # Filtra fontes inexistentes
        fontes = [f for f in fontes if os.path.isdir(f)]
        if not fontes:
            print("[ERRO] Nenhuma pasta de dataset encontrada.")
            sys.exit(1)

        treinar(fontes, a.img)