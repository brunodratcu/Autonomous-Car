"""
================================================================
  TREINO CNN — MobileNetV3Small 96x96
  Estruturas de dataset aceitas automaticamente:

  A) Flat (mais comum):
       dataset_v3/Stop/*.jpg
       dataset_v3/Esquerda/*.jpg  ...
     -> Script auto-divide 80/20 estratificado

  B) Ja dividido:
       dataset_v3/train/Stop/  dataset_v3/val/Stop/  ...
     -> Usado como esta

  USO:
    python TRAIN_SIGN_CNN.py
    python TRAIN_SIGN_CNN.py --dataset ./meu_dataset
    python TRAIN_SIGN_CNN.py --dataset ./dataset_v3 --val-split 0.20

  SAIDAS em ./models/:
    sign_classifier.tflite   <- para carro-autonomo.py
    sign_classifier.keras    <- backup completo
    ood_thresholds.json      <- calibrado no val set
    confusion_matrix.png     <- avaliacao visual
================================================================
"""
import os, sys, json, time, argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# ================================================================
#  [1] CONFIG
# ================================================================
OUTPUT_DIR    = "./models"
CNN_SIZE      = 64    # v5.1: 64x64 (2.2x mais rápido que 96) — o
                      # carro-autonomo.py lê o tamanho do .tflite
BATCH_SIZE    = 32
SEED          = 42
VAL_SPLIT     = 0.20
EPOCHS_1      = 15;   LR_1 = 1e-3
EPOCHS_2      = 20;   LR_2 = 2e-5
UNFREEZE_LAST = 50
OOD_PERCENTILE = 5;   OOD_MIN = 0.40;  OOD_MAX = 0.92

CLASSES = ["Stop","Esquerda","Direita","SemRetorno",
           "Verde","Cone","Carro","Pessoa","Fundo"]
NUM_CLASSES      = len(CLASSES)
IDX              = {c:i for i,c in enumerate(CLASSES)}
NO_FLIP_CLASSES  = {"Esquerda","Direita","SemRetorno"}

# Pastas com código-letra também são aceitas (ex.: dataset_final)
ALIASES = {"s":"Stop","l":"Esquerda","r":"Direita","n":"SemRetorno",
           "g":"Verde","car":"Carro","people":"Pessoa","cone":"Cone",
           "fundo":"Fundo","background":"Fundo","nada":"Fundo"}

def pasta_para_classe(nome: str) -> str | None:
    n = nome.lower()
    for cls in CLASSES:
        if n == cls.lower(): return cls
    return ALIASES.get(n)
EXTS             = {".jpg",".jpeg",".png",".bmp",".webp"}

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ================================================================
#  [2] DIAGNOSTICO
# ================================================================

def _scan_flat(base: Path) -> dict | None:
    """
    Tenta detectar estrutura flat em 'base': procura subpastas
    cujo nome bate (case-insensitive) com alguma classe.
    Retorna dict ou None se nao encontrou nenhuma classe.
    """
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    cls_dirs = {}
    for d in subdirs:
        cls = pasta_para_classe(d.name)
        if cls: cls_dirs[cls] = d

    if not cls_dirs:
        return None

    contagem, faltando = {}, []
    for cls in CLASSES:
        if cls in cls_dirs:
            n = len([f for f in cls_dirs[cls].iterdir()
                     if f.suffix.lower() in EXTS])
            if n > 0: contagem[cls] = n
            else:     faltando.append(cls)
        else:
            faltando.append(cls)
    return {
        "estrutura":     "flat",
        "raiz_efetiva":  str(base),
        "pastas":        sorted(d.name for d in subdirs),
        "contagem":      contagem,
        "faltando":      faltando,
    }


def diagnosticar(dataset_dir: str) -> dict:
    """
    Detecta estrutura real do dataset e conta imagens por classe.
    Casos suportados:
      1. dataset/ClassName/          -> flat, auto-split
      2. dataset/train/ + val/       -> split
      3. dataset/train/  (sem val/)  -> entra em train/, trata como flat
    """
    base = Path(dataset_dir)
    if not base.exists():
        return {"estrutura":"vazio","raiz_efetiva":str(base),
                "pastas":[],"contagem":{},"faltando":list(CLASSES)}

    subdirs = [d for d in base.iterdir() if d.is_dir()]
    nomes   = {d.name for d in subdirs}

    # ── Caso 1: split completo (train + val) ──────────────────────
    if "train" in nomes and "val" in nomes:
        contagem, faltando = {}, []
        for cls in CLASSES:
            n = 0
            for split in ("train","val"):
                p = base / split / cls
                if p.exists():
                    n += len([f for f in p.iterdir()
                               if f.suffix.lower() in EXTS])
            if n > 0: contagem[cls] = n
            else:     faltando.append(cls)
        return {"estrutura":"split","raiz_efetiva":str(base),
                "pastas":sorted(nomes),"contagem":contagem,"faltando":faltando}

    # ── Caso 2: flat direto (pastas de classe no root) ────────────
    flat = _scan_flat(base)
    if flat:
        return flat

    # ── Caso 3: so 'train/' existe sem 'val/' ─────────────────────
    #    Entra em train/ e trata como flat, fazendo auto-split
    if "train" in nomes:
        flat_in_train = _scan_flat(base / "train")
        if flat_in_train:
            flat_in_train["nota"] = (
                "Apenas train/ encontrado (sem val/). "
                "Usando train/ como raiz flat com auto-split."
            )
            return flat_in_train

        # train/ existe mas as pastas internas nao batem com CLASSES
        pastas_internas = sorted(
            d.name for d in (base/"train").iterdir() if d.is_dir()
        )
        return {"estrutura":"desconhecido","raiz_efetiva":str(base),
                "pastas":sorted(nomes),
                "pastas_em_train": pastas_internas,
                "contagem":{},"faltando":list(CLASSES)}

    # ── Desconhecido ──────────────────────────────────────────────
    return {"estrutura":"desconhecido","raiz_efetiva":str(base),
            "pastas":sorted(nomes),"contagem":{},"faltando":list(CLASSES)}


def imprimir_diagnostico(dataset_dir: str, info: dict):
    print("\n" + "-"*62)
    print(f"  DIAGNOSTICO")
    print(f"  Argumento    : {os.path.abspath(dataset_dir)}")
    raiz = info.get("raiz_efetiva", dataset_dir)
    if os.path.abspath(raiz) != os.path.abspath(dataset_dir):
        print(f"  Raiz efetiva : {os.path.abspath(raiz)}  <- ajustado automaticamente")
    print("-"*62)
    print(f"  Estrutura detectada : {info['estrutura'].upper()}")
    if "nota" in info:
        print(f"  Nota: {info['nota']}")
    print(f"  Pastas encontradas  : {info['pastas'] or '(nenhuma)'}")
    if "pastas_em_train" in info:
        print(f"  Pastas em train/    : {info['pastas_em_train']}")
    print()
    print(f"  {'Classe':<14s}  {'Imagens':>8s}  Status")
    print("  " + "-"*36)
    for cls in CLASSES:
        n      = info["contagem"].get(cls, 0)
        status = "OK" if n > 0 else "AUSENTE"
        print(f"  {cls:<14s}  {n:>8d}  {status}")
    total = sum(info["contagem"].values())
    print(f"\n  Total: {total} imagens")
    if info["faltando"]:
        print(f"  Ausentes: {info['faltando']}")
    print("-"*62)

# ================================================================
#  [3] PREPROCESSING MONO UNIFICADO
#
#  A webcam do carro entrega imagem em PRETO E BRANCO.
#  Regra de ouro: o preprocessing do TREINO deve ser IDÊNTICO ao
#  da INFERÊNCIA (carro-autonomo.py::prep_mono). Uma única função
#  para TODAS as classes:
#     resize 96 → gray → CLAHE → replica 3 canais → [0,1]
#  O dataset pode continuar colorido em disco — a conversão
#  acontece aqui, garantindo consistência automática.
# ================================================================

_CLAHE_TR = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def prep_mono(img_bgr):
    if img_bgr.ndim == 3:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(img_bgr, (CNN_SIZE, CNN_SIZE))
    g = _CLAHE_TR.apply(g)
    return np.stack([g,g,g], axis=-1).astype(np.float32)/255.0

def preprocess(img_bgr, cls_name):
    return prep_mono(img_bgr)

# ================================================================
#  [4] AUGMENTACAO
# ================================================================

def augmentar(img, cls_name):
    h, w = img.shape[:2]
    if np.random.rand() < 0.55:
        ang = np.random.uniform(-10,10)
        M   = cv2.getRotationMatrix2D((w/2,h/2),ang,1.0)
        img = cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
    if np.random.rand() < 0.40:
        f = np.random.uniform(0.90,1.10)
        nh,nw = int(h*f),int(w*f)
        rsz = cv2.resize(img,(nw,nh))
        if f>1:
            y0=(nh-h)//2; x0=(nw-w)//2
            img = rsz[y0:y0+h, x0:x0+w]
        else:
            tmp = np.full((h,w,3),128,dtype=img.dtype)
            y0=(h-nh)//2; x0=(w-nw)//2
            tmp[y0:y0+nh, x0:x0+nw]=rsz; img=tmp
    if np.random.rand() < 0.55:
        img = np.clip(img.astype(np.float32)*np.random.uniform(0.75,1.30)
                      + np.random.randint(-25,25),0,255).astype(np.uint8)
    if np.random.rand() < 0.30:
        img = np.clip(img.astype(np.float32)
                      +np.random.randn(*img.shape)*8,0,255).astype(np.uint8)
    # Emula a câmera do carro: contraste extremo / quase binário.
    # Fecha o gap de domínio dataset (fotos normais) → pista (P/B duro)
    if np.random.rand() < 0.35:
        g   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
        lim = np.random.randint(90, 170)
        _, b = cv2.threshold(g, lim, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    # Flip horizontal proibido para classes direcionais
    if cls_name not in NO_FLIP_CLASSES and np.random.rand() < 0.50:
        img = cv2.flip(img,1)
    return img

# ================================================================
#  [4b] CLASSE "FUNDO" — colheita automática de negativos
#
#  O detector geométrico gera candidatos que às vezes NÃO são
#  placas (móveis, sombras, blobs). A CNN precisa saber dizer
#  "isso não é nada" → classe Fundo.
#  --fundo video.mp4 extrai crops aleatórios do vídeo da pista
#  e salva em <dataset>/Fundo/ (uma vez; reutilizado depois).
# ================================================================

def colher_fundo(video_path: str, destino: Path, n_alvo: int = 300):
    destino.mkdir(parents=True, exist_ok=True)
    ja = len(list(destino.glob("*.jpg")))
    if ja >= n_alvo:
        print(f"  [FUNDO] {ja} crops ja existem em {destino} — ok")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [FUNDO][ERRO] video nao abriu: {video_path}"); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rng   = np.random.default_rng(SEED)
    salvos = ja
    while salvos < n_alvo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(rng.integers(0, total)))
        ret, f = cap.read()
        if not ret: continue
        h, w = f.shape[:2]
        tam  = int(rng.integers(40, min(h,w)//2))
        x    = int(rng.integers(0, w-tam))
        y    = int(rng.integers(0, h-tam))
        crop = f[y:y+tam, x:x+tam]
        cv2.imwrite(str(destino/f"fundo_{salvos:04d}.jpg"), crop)
        salvos += 1
    cap.release()
    print(f"  [FUNDO] {salvos} crops salvos em {destino}")

# ================================================================
#  [5] CARREGAMENTO
# ================================================================

def _ler_pasta(pasta):
    imgs=[]
    for p in Path(pasta).iterdir():
        if p.suffix.lower() in EXTS:
            img=cv2.imread(str(p))
            if img is not None: imgs.append(img)
    return imgs


def carregar_flat(dataset_dir, val_frac, aug_factor):
    from sklearn.model_selection import train_test_split
    base = Path(dataset_dir)
    print(f"\n  Modo FLAT — auto-split {int((1-val_frac)*100)}/{int(val_frac*100)}")
    raw_imgs, raw_labels = [],[]
    por_classe = {}
    for d in base.iterdir():
        if not d.is_dir(): continue
        cls = pasta_para_classe(d.name)
        if cls is None: continue
        por_classe.setdefault(cls, []).extend(_ler_pasta(d))
    # Direita ausente? Sintetiza espelhando Esquerda (flip horizontal
    # transforma seta ← em seta →). Barato e correto geometricamente.
    if not por_classe.get("Direita") and por_classe.get("Esquerda"):
        por_classe["Direita"] = [cv2.flip(i,1)
                                 for i in por_classe["Esquerda"]]
        print("    [SINT] Direita gerada espelhando Esquerda")
    for cls in CLASSES:
        imgs = por_classe.get(cls, [])
        if not imgs:
            print(f"    [SKIP] {cls}: pasta nao encontrada"); continue
        print(f"    {cls:<14s}: {len(imgs):>4d}")
        for img in imgs:
            raw_imgs.append(img); raw_labels.append(IDX[cls])
    if not raw_imgs: print("[ERRO] Nenhuma imagem carregada."); sys.exit(1)

    Xr_tr,Xr_va,yr_tr,yr_va = train_test_split(
        raw_imgs,raw_labels,test_size=val_frac,random_state=SEED,stratify=raw_labels)
    print(f"\n    Split: {len(Xr_tr)} treino / {len(Xr_va)} val")

    X_tr,y_tr=[],[]
    for img,lbl in zip(Xr_tr,yr_tr):
        cls=CLASSES[lbl]
        for k in range(aug_factor):
            src = augmentar(img.copy(),cls) if k>0 else img
            X_tr.append(preprocess(src,cls)); y_tr.append(lbl)

    X_va=[preprocess(img,CLASSES[lbl]) for img,lbl in zip(Xr_va,yr_va)]
    return (np.array(X_tr,dtype=np.float32),np.array(y_tr,dtype=np.int32),
            np.array(X_va,dtype=np.float32),np.array(yr_va,dtype=np.int32))


def carregar_split(dataset_dir, aug_factor):
    base=Path(dataset_dir)
    print("\n  Modo SPLIT — usando train/ e val/")
    def _load(split,augment):
        X,y=[],[]
        for cls in CLASSES:
            p = base/split/cls
            if not p.exists():
                # tenta alias (ex.: train/S para Stop)
                p = None
                for d in (base/split).iterdir():
                    if d.is_dir() and pasta_para_classe(d.name)==cls:
                        p = d; break
                if p is None:
                    print(f"    [SKIP] [{split}] {cls}"); continue
            imgs=_ler_pasta(p)
            print(f"    [{split}] {cls:<14s}: {len(imgs):>4d}")
            for img in imgs:
                if augment:
                    for k in range(aug_factor):
                        src=augmentar(img.copy(),cls) if k>0 else img
                        X.append(preprocess(src,cls)); y.append(IDX[cls])
                else:
                    X.append(preprocess(img,cls)); y.append(IDX[cls])
        return np.array(X,dtype=np.float32),np.array(y,dtype=np.int32)
    X_tr,y_tr=_load("train",True)
    X_va,y_va=_load("val",  False)
    return X_tr,y_tr,X_va,y_va

# ================================================================
#  [6] MODELO
# ================================================================

def build_model(objectness=False):
    """
    objectness=False (padrão): 1 saída softmax (compatível com o
    runtime atual). objectness=True: DUAS saídas —
      · output      : softmax das classes (ETAPA 2, "qual placa?")
      · objectness  : sigmoid binário (ETAPA 1, "é placa?"),
                       treinado com Fundo=0 e placas=1
    A separação em duas cabeças é a versão IDEAL da arquitetura
    de dois estágios: a rede aprende explicitamente a distinguir
    placa de não-placa, em vez de ser forçada a escolher classe.
    """
    base=keras.applications.MobileNetV3Small(
        input_shape=(CNN_SIZE,CNN_SIZE,3),include_top=False,
        weights="imagenet",include_preprocessing=False)
    base.trainable=False
    inp=keras.Input(shape=(CNN_SIZE,CNN_SIZE,3),name="input")
    x=base(inp,training=False)
    x=keras.layers.GlobalAveragePooling2D()(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Dropout(0.35)(x)
    x=keras.layers.Dense(128,activation="relu",
                          kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x=keras.layers.Dropout(0.20)(x)
    out=keras.layers.Dense(NUM_CLASSES,activation="softmax",name="output")(x)
    if not objectness:
        return keras.Model(inp,out,name="MobileNetV3Small_sign"),base
    obj=keras.layers.Dense(1,activation="sigmoid",name="objectness")(x)
    return keras.Model(inp,[out,obj],name="MobileNetV3Small_2head"),base

# ================================================================
#  [7] TREINO 2 FASES
# ================================================================

def treinar(X_tr,y_tr,X_va,y_va):
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr=compute_class_weight("balanced",classes=np.arange(NUM_CLASSES),y=y_tr)
    cw=dict(enumerate(cw_arr))
    pesos_str = {cls: round(v, 2) for cls, v in zip(CLASSES, cw_arr)}
    print(f"\n  Pesos: {pesos_str}")

    model,base=build_model()
    print(f"  Params totais: {model.count_params():,}")

    def compile_(lr):
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy",metrics=["accuracy"])

    print(f"\n  FASE 1 — Head only ({EPOCHS_1} epochs lr={LR_1})")
    compile_(LR_1)
    cb1=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True,
                                        monitor="val_accuracy"),
         keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,
                                           monitor="val_loss",verbose=1,min_lr=1e-6)]
    h1=model.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=EPOCHS_1,
                 batch_size=BATCH_SIZE,class_weight=cw,callbacks=cb1,verbose=1)
    print(f"  [F1] Melhor val_acc: {max(h1.history['val_accuracy'])*100:.2f}%")

    print(f"\n  FASE 2 — Fine-tune ({EPOCHS_2} epochs lr={LR_2} "
          f"ultimas {UNFREEZE_LAST} layers)")
    base.trainable=True
    for layer in base.layers[:-UNFREEZE_LAST]: layer.trainable=False
    for layer in base.layers:
        if isinstance(layer,keras.layers.BatchNormalization): layer.trainable=False

    compile_(LR_2)
    ck=os.path.join(OUTPUT_DIR,"_ck.keras")
    cb2=[keras.callbacks.EarlyStopping(patience=7,restore_best_weights=True,
                                        monitor="val_accuracy"),
         keras.callbacks.ModelCheckpoint(ck,save_best_only=True,
                                         monitor="val_accuracy",verbose=1),
         keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=4,
                                           monitor="val_loss",verbose=1,min_lr=1e-7)]
    h2=model.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=EPOCHS_2,
                 batch_size=BATCH_SIZE,class_weight=cw,callbacks=cb2,verbose=1)
    print(f"  [F2] Melhor val_acc: {max(h2.history['val_accuracy'])*100:.2f}%")

    loss,acc=model.evaluate(X_va,y_va,verbose=0)
    print(f"  [FINAL] val_loss={loss:.4f}  val_acc={acc*100:.2f}%")
    if os.path.exists(ck): os.remove(ck)
    return model

# ================================================================
#  [8] AVALIACAO
# ================================================================

def avaliar(model,X_va,y_va):
    from sklearn.metrics import confusion_matrix,classification_report
    probs=model.predict(X_va,verbose=0); y_pred=probs.argmax(axis=1)
    print("\n"+classification_report(y_va,y_pred,target_names=CLASSES,
                                     digits=3,zero_division=0))
    cm=confusion_matrix(y_va,y_pred)
    cm_n=cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-9)
    fig,axes=plt.subplots(1,2,figsize=(18,7))
    for ax,data,title,fmt in zip(axes,[cm,cm_n],
                                  ["Contagem","Recall normalizado"],["d",".2f"]):
        im=ax.imshow(data,cmap="Blues" if fmt=="d" else "YlOrRd",
                     vmin=0,vmax=None if fmt=="d" else 1)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES,rotation=40,ha="right",fontsize=9)
        ax.set_yticklabels(CLASSES,fontsize=9)
        ax.set_xlabel("Predito"); ax.set_ylabel("Real"); ax.set_title(title)
        plt.colorbar(im,ax=ax)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                v=data[i,j]; t=f"{v}" if fmt=="d" else f"{v:.2f}"
                ax.text(j,i,t,ha="center",va="center",fontsize=7,
                        color="white" if (v>cm.max()*0.6 if fmt=="d" else v>0.6) else "black")
    plt.suptitle("MobileNetV3Small 96x96 — Confusion Matrix")
    plt.tight_layout()
    out=os.path.join(OUTPUT_DIR,"confusion_matrix.png")
    plt.savefig(out,dpi=130); plt.close()
    print(f"  [OK] {out}")
    print("\n  Acuracia por classe:")
    for i,cls in enumerate(CLASSES):
        tp=cm[i,i]; total=cm[i].sum()
        print(f"    {cls:<14s}: {tp:3d}/{total:3d} = {tp/max(total,1)*100:5.1f}%")

# ================================================================
#  [9] OOD CALIBRACAO
# ================================================================

def calibrar_ood(model,X_va,y_va):
    print(f"\n  Calibrando OOD (percentil={OOD_PERCENTILE})...")
    probs=model.predict(X_va,verbose=0)
    max_sof=probs.max(axis=1); pred_cls=probs.argmax(axis=1)
    scores=defaultdict(list)
    for tc,pc,sc in zip(y_va,pred_cls,max_sof):
        if tc==pc: scores[CLASSES[tc]].append(float(sc))

    thresholds={}
    print(f"\n  {'Classe':<14s} {'n':>5s}  {'min':>5s}  {'med':>5s}  "
          f"p{OOD_PERCENTILE:02d} -> thr")
    for cls in CLASSES:
        sc=scores.get(cls,[])
        if len(sc)<5:
            thr=0.55; print(f"  {cls:<14s} {'<5':>5s}  (poucos dados) -> {thr:.3f}")
        else:
            raw=float(np.percentile(sc,OOD_PERCENTILE))
            thr=float(np.clip(raw,OOD_MIN,OOD_MAX))
            print(f"  {cls:<14s} {len(sc):>5d}  {min(sc):>5.3f}  "
                  f"{np.median(sc):>5.3f}  {raw:.3f} -> {thr:.3f}")
        thresholds[cls]=thr

    out=os.path.join(OUTPUT_DIR,"ood_thresholds.json")
    with open(out,"w") as f: json.dump(thresholds,f,indent=2)
    print(f"\n  [OK] {out}")
    return thresholds

# ================================================================
#  [10] EXPORT
# ================================================================

def exportar(model, X_rep=None, int8=False):
    kp=os.path.join(OUTPUT_DIR,"sign_classifier.keras")
    model.save(kp); print(f"  [OK] Keras  : {kp}")
    conv=tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]
    if int8 and X_rep is not None:
        # Quantização INT8 COMPLETA: pesos E ativações em 8 bits.
        # Precisa de amostras representativas p/ calibrar os ranges.
        # ~3x mais rápido em CPU; o runtime detecta o dtype sozinho.
        def rep():
            for i in range(min(200, len(X_rep))):
                yield [X_rep[i:i+1].astype(np.float32)]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type  = tf.uint8
        conv.inference_output_type = tf.uint8
        print("  [INT8] Quantizacao completa (uint8 in/out)")
    tflite=conv.convert()
    tp=os.path.join(OUTPUT_DIR,"sign_classifier.tflite")
    with open(tp,"wb") as f: f.write(tflite)
    print(f"  [OK] TFLite : {tp}  ({os.path.getsize(tp)//1024} KB)")
    cp=os.path.join(OUTPUT_DIR,"classes.txt")
    with open(cp,"w") as f: f.write("\n".join(CLASSES))
    print(f"  [OK] Classes: {cp}  (lido pelo carro-autonomo.py)")

# ================================================================
#  MAIN
# ================================================================

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="./dataset_v3")
    ap.add_argument("--val-split",type=float,default=VAL_SPLIT)
    ap.add_argument("--aug",      type=int,  default=3)
    ap.add_argument("--fundo",    default=None,
        help="video da pista p/ colher crops da classe Fundo")
    ap.add_argument("--size",     type=int, default=CNN_SIZE,
        help="lado da entrada da CNN (64 rapido / 96 preciso)")
    ap.add_argument("--int8",     action="store_true",
        help="exporta TFLite quantizado INT8 (~3x mais rapido)")
    args=ap.parse_args()

    t0=time.time()
    globals()["CNN_SIZE"] = args.size
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    print("="*62)
    print("  TREINO CNN — MobileNetV3Small 96x96")
    print(f"  Classes ({NUM_CLASSES}): {', '.join(CLASSES)}")
    print(f"  Dataset : {os.path.abspath(args.dataset)}")
    print(f"  Output  : {os.path.abspath(OUTPUT_DIR)}")
    print("="*62)

    gpus=tf.config.list_physical_devices("GPU")
    if gpus:
        try: tf.config.experimental.set_memory_growth(gpus[0],True)
        except Exception: pass
        print(f"[GPU] {gpus[0].name}")
    else:
        print("[CPU] Sem GPU — funcional mas mais lento")

    # Colheita da classe Fundo (se pedida)
    if args.fundo:
        base = Path(args.dataset)
        alvo = base/"train"/"Fundo" if (base/"train").exists() \
               else base/"Fundo"
        colher_fundo(args.fundo, alvo)

    # Diagnostico
    info=diagnosticar(args.dataset)
    imprimir_diagnostico(args.dataset,info)

    if info["estrutura"]=="vazio":
        print(f"\n[ERRO] Diretorio nao encontrado: {args.dataset}")
        print("       Use --dataset ./caminho/do/seu/dataset")
        sys.exit(1)

    if info["estrutura"]=="desconhecido":
        print("\n[ERRO] Estrutura nao reconhecida.")
        print(f"       Pastas encontradas: {info['pastas']}")
        if "pastas_em_train" in info:
            print(f"       Pastas dentro de train/: {info['pastas_em_train']}")
            print("\n       As pastas em train/ nao batem com os nomes das classes.")
            print("       Renomeie as pastas para os nomes exatos:")
            print(f"         {', '.join(CLASSES)}")
        else:
            print("       Esperado (flat):  dataset/Stop/  dataset/Esquerda/ ...")
            print("       Esperado (split): dataset/train/Stop/  dataset/val/Stop/ ...")
        sys.exit(1)

    if not info["contagem"]:
        print("\n[ERRO] Nenhuma classe com imagens encontrada.")
        sys.exit(1)

    if info["faltando"]:
        print(f"\n[AVISO] Classes ausentes: {info['faltando']}")
        print("        Treinando com o que esta disponivel.")

    # Usa raiz_efetiva (pode ser dataset_v3/train/ se so train/ existia)
    dataset_efetivo = info.get("raiz_efetiva", args.dataset)

    # Carregamento
    print("\n" + "="*62 + "\n  CARREGANDO\n" + "="*62)
    if info["estrutura"]=="split":
        X_tr,y_tr,X_va,y_va=carregar_split(dataset_efetivo,args.aug)
    else:
        X_tr,y_tr,X_va,y_va=carregar_flat(dataset_efetivo,args.val_split,args.aug)

    print(f"\n  Treino: {len(X_tr)}  Val: {len(X_va)}")

    # Pipeline
    print("\n"+"="*62+"\n  TREINANDO\n"+"="*62)
    model=treinar(X_tr,y_tr,X_va,y_va)

    print("\n"+"="*62+"\n  AVALIACAO\n"+"="*62)
    avaliar(model,X_va,y_va)

    print("\n"+"="*62+"\n  OOD\n"+"="*62)
    calibrar_ood(model,X_va,y_va)

    print("\n"+"="*62+"\n  EXPORTANDO\n"+"="*62)
    exportar(model, X_rep=X_va, int8=args.int8)

    print(f"\n  [DONE] {(time.time()-t0)/60:.1f} min")
    print(f"  CNN_MODEL = '{OUTPUT_DIR}/sign_classifier.tflite'")
    print(f"  OOD_FILE  = '{OUTPUT_DIR}/ood_thresholds.json'")