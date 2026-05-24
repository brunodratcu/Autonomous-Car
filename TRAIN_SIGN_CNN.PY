"""
================================================================
  TREINAR CNN DE PLACAS — TENSORFLOW / KERAS
  ─────────────────────────────────────────────────────────────
  Aceita duas fontes de dados:
    --dataset ./dataset_tratado   ← pasta extraída (recomendado)
    --zip ./arquivo.zip           ← zip direto do Roboflow

  Estrutura esperada da pasta:
    dataset_tratado/
      train/
        Stop/        ← imagens .jpg
        Pedestrian/
        Speed_Limit_40/
        ... (uma pasta por classe)
      valid/
        Stop/
        ...
      test/          ← opcional
        Stop/
        ...

  USO:
    python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado
    python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado --epochs 80
    python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado --img 96
    python TRAIN_SIGN_CNN.py --test ./foto_placa.jpg
================================================================
"""

import os, sys, argparse, csv, io, zipfile, glob
import cv2, numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # silencia warnings oneDNN

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except ImportError:
    print("Execute: pip install tensorflow scikit-learn")
    sys.exit(1)

#  CONFIGURAÇÃO
DATASET_DIR = "./dataset_tratado"   # pasta padrão
MODEL_PATH  = "./sign_model.tflite"
LABELS_PATH = "./sign_labels.txt"
KERAS_PATH  = "./sign_model.keras"
REPORT_PATH = "./training_report.png"

IMG_SIZE    = 96
BATCH_SIZE  = 32
EPOCHS      = 80
LR          = 8e-4
SEED        = 42

# ── Mapeamento classe → comando serial ───────────────────────────
CLASS_TO_CMD = {
    "Stop":            "S",
    "Pedestrian":      "Y",
    "Speed_Limit_40":  "W",
    "Speed_Limit_90":  "F",
    "Speed_Limit_120": "U",
    "No_U-Turn":       "N",
    "No_Stopping":     "K",
    "No_Parking":      "P",
    "Bump":            "B",
    "Road_Work":       "W",
}

ESPERA_SINAL = {"S", "Y"}


#  CARREGAR DE PASTA  (dataset_tratado já extraído)
def carregar_de_pasta(dataset_dir: str, sz: int):
    """
    Lê imagens de dataset_tratado/train/NomeClasse/*.jpg
    As imagens já foram tratadas (CLAHE, bordas, sharp etc.)
    Só faz resize para garantir tamanho uniforme.
    """
    print(f"\n[DATA] Lendo pasta: {dataset_dir}")

    # Descobre classes presentes nas subpastas de train/
    train_dir  = os.path.join(dataset_dir, "train")
    if not os.path.isdir(train_dir):
        print(f"[ERRO] Pasta não encontrada: {train_dir}")
        print( "       Estrutura esperada: dataset_tratado/train/NomeClasse/*.jpg")
        sys.exit(1)

    classes_presentes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
        and d in CLASS_TO_CMD
    ])

    if not classes_presentes:
        print(f"[ERRO] Nenhuma classe reconhecida em {train_dir}")
        print(f"       Classes esperadas: {list(CLASS_TO_CMD.keys())}")
        sys.exit(1)

    # Monta lista de comandos únicos na ordem estável
    cmds_unicos = sorted(set(CLASS_TO_CMD[c] for c in classes_presentes))
    cmd_to_idx  = {c: i for i, c in enumerate(cmds_unicos)}

    print(f"[DATA] Classes encontradas ({len(classes_presentes)}): {classes_presentes}")
    print(f"[DATA] Comandos únicos ({len(cmds_unicos)}): {cmds_unicos}")

    X_all, y_all = [], []
    contagem     = {cls: 0 for cls in classes_presentes}

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue

        n_split = 0
        for cls in classes_presentes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue

            # Pega todos os .jpg e .png
            arquivos = (
                glob.glob(os.path.join(cls_dir, "*.jpg")) +
                glob.glob(os.path.join(cls_dir, "*.png")) +
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
            )

            for caminho in arquivos:
                img = cv2.imread(caminho)
                if img is None:
                    continue

                # As imagens já foram tratadas — só garante tamanho
                if img.shape[:2] != (sz, sz):
                    img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)

                # BGR → RGB, normaliza [0,1]
                rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = rgb.astype(np.float32) / 255.0

                X_all.append(tensor)
                y_all.append(cmd_to_idx[CLASS_TO_CMD[cls]])
                contagem[cls] += 1
                n_split += 1

        print(f"  [{split}] {n_split} imagens")

    if not X_all:
        print("[ERRO] Nenhuma imagem carregada.")
        sys.exit(1)

    X = np.array(X_all, np.float32)
    y = np.array(y_all, np.int32)

    print(f"\n[DATA] Total: {len(X)} imagens | {len(cmds_unicos)} comandos")
    print(f"\n  {'Classe':<22} {'Cmd':>5}  {'Imagens':>8}  Tratamento")
    print("  " + "-"*60)
    tratamentos = {
        "Stop":            "CLAHE forte + bordas + sharp",
        "Pedestrian":      "Upscale CUBIC + denoise + CLAHE",
        "Speed_Limit_40":  "CLAHE leve + normalização cor",
        "Speed_Limit_90":  "CLAHE leve + normalização cor",
        "Speed_Limit_120": "CLAHE leve + normalização cor",
        "No_U-Turn":       "CLAHE forte + bordas + sharp",
        "No_Stopping":     "CLAHE forte + bordas + sharp",
        "No_Parking":      "CLAHE forte + bordas + sharp",
        "Bump":            "CLAHE médio + sharp",
        "Road_Work":       "CLAHE médio + sharp",
    }
    for cls in classes_presentes:
        cmd = CLASS_TO_CMD[cls]
        n   = contagem[cls]
        trt = tratamentos.get(cls, "-")
        print(f"  {cls:<22} '{cmd}'  {n:>8}   {trt}")

    return X, y, cmds_unicos


#  CARREGAR DE ZIP  (compatibilidade com zip do Roboflow)
def _preprocess_zip(crop_bgr, cls, sz):
    """Pré-processamento inline para quando carrega direto do zip."""
    def clahe(img, clip):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        c   = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4,4))
        lab[:,:,0] = c.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    def sharp(img):
        b = cv2.GaussianBlur(img,(0,0),sigmaX=1.5)
        return cv2.addWeighted(img,1.6,b,-0.6,0)
    def normcor(img):
        f  = img.astype(np.float32)
        mx = f.max(axis=2,keepdims=True).clip(1)
        return np.clip(f/mx*200,0,255).astype(np.uint8)
    def edge(img):
        g = cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),40,120)
        return cv2.addWeighted(img,.85,cv2.cvtColor(g,cv2.COLOR_GRAY2BGR),.20,0)

    if "Speed_Limit" in cls:
        img = normcor(clahe(crop_bgr,1.5))
        return cv2.resize(img,(sz,sz),interpolation=cv2.INTER_AREA)
    elif cls in ("Stop","No_U-Turn","No_Stopping","No_Parking"):
        img = sharp(edge(clahe(crop_bgr,3.0)))
        return cv2.resize(img,(sz,sz),interpolation=cv2.INTER_AREA)
    elif cls in ("Bump","Road_Work"):
        return cv2.resize(sharp(clahe(crop_bgr,2.0)),(sz,sz),interpolation=cv2.INTER_AREA)
    elif cls == "Pedestrian":
        img = cv2.resize(crop_bgr,(sz,sz),interpolation=cv2.INTER_CUBIC)
        img = cv2.fastNlMeansDenoisingColored(img,None,6,6,7,21)
        return clahe(img,2.5)
    return cv2.resize(clahe(crop_bgr,2.0),(sz,sz),interpolation=cv2.INTER_AREA)


def carregar_de_zip(zip_path: str, sz: int):
    print(f"\n[ZIP] Lendo: {zip_path}")
    cmds_unicos = sorted(set(CLASS_TO_CMD.values()))
    cmd_to_idx  = {c: i for i, c in enumerate(cmds_unicos)}
    X_all, y_all = [], []

    with zipfile.ZipFile(zip_path, "r") as zf:
        nomes = set(zf.namelist())
        for split in ["train","valid","test"]:
            csv_key = f"{split}/_annotations.csv"
            if csv_key not in nomes:
                continue
            linhas = list(csv.DictReader(
                io.StringIO(zf.read(csv_key).decode("utf-8"))
            ))
            ok = skip = 0
            for row in linhas:
                cls = row["class"].strip()
                if cls in ("class","") or cls not in CLASS_TO_CMD:
                    skip+=1; continue
                img_key = f"{split}/{row['filename']}"
                if img_key not in nomes:
                    skip+=1; continue
                arr = np.frombuffer(zf.read(img_key), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    skip+=1; continue
                H,W = img.shape[:2]
                x1=max(0,int(row["xmin"])); y1=max(0,int(row["ymin"]))
                x2=min(W,int(row["xmax"])); y2=min(H,int(row["ymax"]))
                if x2<=x1 or y2<=y1:
                    skip+=1; continue
                crop   = img[y1:y2,x1:x2]
                tensor = _preprocess_zip(crop,cls,sz)
                rgb    = cv2.cvtColor(tensor,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
                X_all.append(rgb)
                y_all.append(cmd_to_idx[CLASS_TO_CMD[cls]])
                ok+=1
            print(f"  [{split}] {ok} carregados ({skip} pulados)")

    if not X_all:
        print("[ERRO] Nenhum crop carregado."); sys.exit(1)

    X = np.array(X_all, np.float32)
    y = np.array(y_all, np.int32)
    print(f"\n[DATA] Total: {len(X)} | Comandos: {cmds_unicos}")
    return X, y, cmds_unicos


#  ARQUITETURA CNN
def construir_modelo(n_classes: int, sz: int) -> keras.Model:
    inp = keras.Input(shape=(sz, sz, 3), name="imagem")
    l2  = keras.regularizers.l2(1e-4)

    # Augmentação — só ativa no treino
    x = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.15),
        layers.RandomBrightness(0.30),
        layers.RandomContrast(0.25),
        layers.RandomTranslation(0.10, 0.10),
    ], name="aug")(inp)

    # Bloco Conv inicial
    x = layers.Conv2D(32, 3, padding="same", use_bias=False,
                      kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6.0)(x)
    x = layers.MaxPooling2D(2)(x)

    # Blocos Depthwise Separable
    for filtros, pool in [(64,True),(128,True),(256,True),(512,False)]:
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                                   depthwise_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        x = layers.Conv2D(filtros, 1, use_bias=False,
                          kernel_regularizer=l2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(max_value=6.0)(x)
        if pool:
            x = layers.MaxPooling2D(2)(x)

    x     = layers.GlobalAveragePooling2D()(x)
    x     = layers.Dense(256, activation="relu", kernel_regularizer=l2)(x)
    x     = layers.Dropout(0.45)(x)
    x     = layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
    x     = layers.Dropout(0.30)(x)
    x     = layers.Dense(64,  activation="relu", kernel_regularizer=l2)(x)
    x     = layers.Dropout(0.15)(x)
    saida = layers.Dense(n_classes, activation="softmax", name="saida")(x)

    modelo = keras.Model(inp, saida, name="TrafficSign_CNN")
    modelo.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return modelo


#  TREINO
def treinar(source: str, epochs: int, sz: int):
    print("=" * 60)
    print("  CNN PLACAS — TENSORFLOW/KERAS")
    print(f"  {source}")
    print("=" * 60)

    # Detecta se é pasta ou zip
    if os.path.isdir(source):
        X, y, cmds = carregar_de_pasta(source, sz)
    elif os.path.isfile(source) and source.endswith(".zip"):
        X, y, cmds = carregar_de_zip(source, sz)
    else:
        print(f"[ERRO] '{source}' não é uma pasta nem um .zip válido.")
        print( "       Use --dataset ./dataset_tratado")
        print( "       Ou  --zip ./traffic_signs_v3i_tensorflow.zip")
        sys.exit(1)

    n = len(cmds)

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y, test_size=0.18, stratify=y, random_state=SEED
    )
    print(f"\n[SPLIT] Treino: {len(X_tr)}  |  Validação: {len(X_v)}")

    pesos_vals = compute_class_weight(
        "balanced", classes=np.arange(n), y=y_tr
    )
    pesos = dict(enumerate(pesos_vals))
    print(f"[PESOS] { {cmds[k]: f'{v:.2f}' for k,v in pesos.items()} }")

    ds_tr = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
             .shuffle(len(X_tr), seed=SEED)
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
    ds_v  = (tf.data.Dataset.from_tensor_slices((X_v, y_v))
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

    modelo = construir_modelo(n, sz)
    modelo.summary()

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=18,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.35,
            patience=8, min_lr=5e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            KERAS_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
    ]

    print(f"\n[TREINO] {epochs} épocas | batch={BATCH_SIZE} | img={sz}×{sz}")
    hist = modelo.fit(
        ds_tr, validation_data=ds_v,
        epochs=epochs, class_weight=pesos,
        callbacks=cbs, verbose=1
    )

    best = max(hist.history["val_accuracy"])
    print(f"\n[RESULTADO] Melhor acurácia de validação: {best*100:.1f}%")

    _grafico(hist, cmds)
    _exportar(modelo, cmds)


#  EXPORT
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
    print("\n[MAPA] Comando → Classes:")
    for cmd in sorted(rev):
        esp = " ← aguarda sinal externo" if cmd in ESPERA_SINAL else ""
        print(f"  '{cmd}'  {', '.join(rev[cmd])}{esp}")
    print(f"\n  Copie {MODEL_PATH} e {LABELS_PATH} para a pasta do sign_detector_ai.py")


#  GRÁFICO
def _grafico(hist, cmds):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(
            f"CNN Placas — comandos: {cmds}",
            fontsize=10, fontweight="bold"
        )
        best_ep = int(np.argmax(hist.history["val_accuracy"]))
        best_ac = hist.history["val_accuracy"][best_ep]

        for i, (m, t) in enumerate([("accuracy","Acurácia"),("loss","Loss")]):
            ax = axes[i]
            ax.plot(hist.history[m],           label="treino",    color="#2E75B6", lw=2)
            ax.plot(hist.history[f"val_{m}"],  label="validação", color="#E74C3C", lw=2, ls="--")
            ax.set_title(t); ax.set_xlabel("Época")
            ax.legend(); ax.grid(alpha=0.3)
            if m == "accuracy":
                ax.set_ylim(0, 1)
                ax.axvline(best_ep, color="green", alpha=0.4, ls=":")
                ax.annotate(f"best {best_ac:.3f}",
                            xy=(best_ep, best_ac),
                            xytext=(best_ep+1, best_ac-0.08),
                            fontsize=8, color="green")

        plt.tight_layout()
        plt.savefig(REPORT_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[GRAFICO] {REPORT_PATH}")
    except Exception as e:
        print(f"[GRAFICO] {e}")


#  TESTE COM FOTO
def testar(caminho: str):
    for arq in [MODEL_PATH, LABELS_PATH]:
        if not os.path.isfile(arq):
            print(f"[ERRO] {arq} não encontrado — treine primeiro.")
            sys.exit(1)

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
        print(f"[ERRO] Imagem não encontrada: {caminho}"); sys.exit(1)

    img_r  = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
    rgb    = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.expand_dims(rgb, 0)

    interp.set_tensor(ind["index"], tensor)
    interp.invoke()
    p    = interp.get_tensor(outd["index"])[0]
    top3 = np.argsort(p)[::-1][:3]

    rev = {}
    for k, v in CLASS_TO_CMD.items():
        rev.setdefault(v, []).append(k)

    print(f"\n[TESTE] {caminho}")
    print("-" * 50)
    for i, idx in enumerate(top3):
        cmd    = cmds[idx]
        cls    = ", ".join(rev.get(cmd, [cmd]))
        espera = " → aguarda sinal" if cmd in ESPERA_SINAL else ""
        marca  = " ←" if i == 0 else "  "
        print(f"  {marca} '{cmd}'  {p[idx]*100:5.1f}%  {cls}{espera}")

    cv2.putText(img,
                f"'{cmds[top3[0]]}' {p[top3[0]]*100:.0f}%",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 80), 2)
    cv2.imshow("Resultado", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#  ENTRY POINT
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Treina CNN de placas com TF/Keras",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "--dataset", default=None,
        metavar="PASTA",
        help="Pasta do dataset tratado (ex: ./dataset_tratado)\n"
             "Estrutura: pasta/train/NomeClasse/*.jpg"
    )
    ap.add_argument(
        "--zip", default=None,
        metavar="ARQUIVO.ZIP",
        help="Zip do Roboflow (ex: ./traffic_signs_v3i_tensorflow.zip)"
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"Épocas máximas (padrão: {EPOCHS})")
    ap.add_argument("--img",    type=int, default=IMG_SIZE,
                    help=f"Resolução de entrada px (padrão: {IMG_SIZE})")
    ap.add_argument("--test",   type=str, default=None,
                    metavar="FOTO.JPG",
                    help="Testa uma foto com o modelo já treinado")
    a = ap.parse_args()

    if a.test:
        testar(a.test)
    elif a.dataset:
        treinar(a.dataset, a.epochs, a.img)
    elif a.zip:
        treinar(a.zip, a.epochs, a.img)
    else:
        # Tenta o padrão automaticamente
        if os.path.isdir(DATASET_DIR):
            print(f"[AUTO] Usando pasta padrão: {DATASET_DIR}")
            treinar(DATASET_DIR, a.epochs, a.img)
        else:
            ap.print_help()
            print(f"\n[ERRO] Nenhuma fonte especificada e '{DATASET_DIR}' não existe.")
            print("  Use: python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado"