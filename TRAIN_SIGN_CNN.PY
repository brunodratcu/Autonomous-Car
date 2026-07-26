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
CNN_SIZE      = 96
BATCH_SIZE    = 32
SEED          = 42
VAL_SPLIT     = 0.20
EPOCHS_1      = 15;   LR_1 = 1e-3
EPOCHS_2      = 20;   LR_2 = 2e-5
UNFREEZE_LAST = 50
OOD_PERCENTILE = 5;   OOD_MIN = 0.40;  OOD_MAX = 0.92

CLASSES = ["Stop","Esquerda","Direita","SemRetorno",
           "Verde","Cone","Carro","Pessoa"]
NUM_CLASSES      = len(CLASSES)
IDX              = {c:i for i,c in enumerate(CLASSES)}
OBSTACLE_CLASSES = {"Cone","Carro","Pessoa"}
NO_FLIP_CLASSES  = {"Esquerda","Direita","SemRetorno"}
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
        for cls in CLASSES:
            if d.name.lower() == cls.lower():
                cls_dirs[cls] = d; break

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
#  [3] PREPROCESSING DUAL
# ================================================================

def prep_sign(img_bgr):
    img  = cv2.resize(img_bgr, (CNN_SIZE,CNN_SIZE))
    gray = cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(3,3),0)
    thr  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)
    return np.stack([thr,thr,thr],axis=-1).astype(np.float32)/255.0

def prep_obstacle(img_bgr):
    img = cv2.resize(img_bgr,(CNN_SIZE,CNN_SIZE))
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def preprocess(img_bgr, cls_name):
    return prep_obstacle(img_bgr) if cls_name in OBSTACLE_CLASSES else prep_sign(img_bgr)

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
    # Flip horizontal proibido para classes direcionais
    if cls_name not in NO_FLIP_CLASSES and np.random.rand() < 0.50:
        img = cv2.flip(img,1)
    return img

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
    for cls in CLASSES:
        pasta=None
        for d in base.iterdir():
            if d.is_dir() and d.name.lower()==cls.lower(): pasta=d; break
        if pasta is None: print(f"    [SKIP] {cls}: pasta nao encontrada"); continue
        imgs = _ler_pasta(pasta)
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
            if not p.exists(): print(f"    [SKIP] [{split}] {cls}"); continue
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

def build_model():
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
    return keras.Model(inp,out,name="MobileNetV3Small_sign"),base

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

def exportar(model):
    kp=os.path.join(OUTPUT_DIR,"sign_classifier.keras")
    model.save(kp); print(f"  [OK] Keras  : {kp}")
    conv=tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]
    tflite=conv.convert()
    tp=os.path.join(OUTPUT_DIR,"sign_classifier.tflite")
    with open(tp,"wb") as f: f.write(tflite)
    print(f"  [OK] TFLite : {tp}  ({os.path.getsize(tp)//1024} KB)")

# ================================================================
#  MAIN
# ================================================================

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="./dataset_v3")
    ap.add_argument("--val-split",type=float,default=VAL_SPLIT)
    ap.add_argument("--aug",      type=int,  default=3)
    args=ap.parse_args()

    t0=time.time()
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
    exportar(model)

    print(f"\n  [DONE] {(time.time()-t0)/60:.1f} min")
    print(f"  CNN_MODEL = '{OUTPUT_DIR}/sign_classifier.tflite'")
    print(f"  OOD_FILE  = '{OUTPUT_DIR}/ood_thresholds.json'")