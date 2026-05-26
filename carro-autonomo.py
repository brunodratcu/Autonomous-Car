"""
================================================================
  SIGN_DETECTOR.py — Detector de placas completo
  ─────────────────────────────────────────────────────────────
  Um único arquivo. Três modos:

  TESTAR FOTO (diagnóstico):
    python SIGN_DETECTOR.py --test foto.jpg
    python SIGN_DETECTOR.py --test f1.jpg f2.jpg --conf 0.0

  CÂMERA / VÍDEO (vida real no carro):
    python SIGN_DETECTOR.py               ← vídeo (videoplayback.mp4)
    python SIGN_DETECTOR.py --cam         ← câmera USB

  COLETAR AMOSTRAS (quando as placas chegarem):
    python SIGN_DETECTOR.py --collect Pare
    python SIGN_DETECTOR.py --collect Pedestre --n 200

  Arquivos necessários na mesma pasta:
    sign_model.tflite   ← gerado por TRAIN_SIGN_CNN.py
    sign_labels.txt     ← gerado por TRAIN_SIGN_CNN.py
================================================================
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import os

# ================================================================
#  [1] CONFIGURAÇÃO
# ================================================================

VIDEO        = "./videoplayback.mp4"
CAM_IDX      = 0
PROP         = 2
SERIAL_PORT  = "COM3"
BAUD         = 115200
MODEL_PATH   = "./sign_model.tflite"
LABELS_PATH  = "./sign_labels.txt"

# Detecção
ROI_Y0       = 0.05
ROI_Y1       = 0.72
BORDA_FRAC   = 0.18
AREA_MIN     = 300
AREA_MAX     = 80_000
CONFIRM_N    = 4
AREA_EXEC    = 400
COOLDOWN     = 55
CONF_MIN     = 0.60

# Obstáculo
OBST_Y0      = 0.50
OBST_AREA_MIN= 3000

# ================================================================
#  [2] AÇÕES DO CARRO
# ================================================================

ACOES = {
    "STOP":      dict(mot=0,  srv=127, buz=0, led=1, brk=1, dir=0, dur=2.5),
    "OBSTACLE":  dict(mot=0,  srv=127, buz=0, led=1, brk=1, dir=0, dur=0.0),
    "YIELD":     dict(mot=30, srv=127, buz=0, led=0, brk=0, dir=0, dur=1.5),
    "LEFT":      dict(mot=40, srv=50,  buz=0, led=0, brk=0, dir=1, dur=1.4),
    "RIGHT":     dict(mot=40, srv=204, buz=0, led=0, brk=0, dir=2, dur=1.4),
    "STRAIGHT":  dict(mot=62, srv=127, buz=0, led=0, brk=0, dir=3, dur=1.0),
    "SLOW_DOWN": dict(mot=30, srv=127, buz=0, led=0, brk=0, dir=0, dur=2.0),
    "SPEED_UP":  dict(mot=80, srv=127, buz=0, led=0, brk=0, dir=3, dur=2.0),
}

LABEL_TO_ACTION = {
    "B": "SLOW_DOWN", "F": "SLOW_DOWN", "K": "STOP",
    "N": "STRAIGHT",  "P": "STRAIGHT",  "S": "STOP",
    "T": "STRAIGHT",  "U": "SPEED_UP",  "W": "SLOW_DOWN",
    "Y": "YIELD",
}

LABEL_NAMES = {
    "B": "Bump/Road_Work",  "F": "Speed_Limit_90",
    "K": "No_Stopping",     "N": "No_U-Turn",
    "P": "No_Parking",      "S": "Stop/Pare",
    "T": "Sentido_Obrig.",  "U": "Speed_Limit_120",
    "W": "Speed_Limit_40",  "Y": "Pedestrian",
}

CORES_VIS = {
    "STOP":      (50,  50,  220), "OBSTACLE": (0,   0,   200),
    "YIELD":     (0,   200, 220), "LEFT":     (220, 120, 0),
    "RIGHT":     (0,   120, 220), "STRAIGHT": (50,  220, 50),
    "SLOW_DOWN": (0,   180, 255), "SPEED_UP": (0,   200, 100),
}

# ================================================================
#  [3] RANGES HSV
# ================================================================

HSV_RANGES = {
    "vermelho": [(np.array([0,  80, 50]), np.array([12, 255,255])),
                 (np.array([165,80, 50]), np.array([180,255,255]))],
    "azul":     [(np.array([90, 80, 50]), np.array([135,255,255]))],
    "amarelo":  [(np.array([15,120, 80]), np.array([38, 255,255]))],
    "verde":    [(np.array([38,120, 60]), np.array([88, 255,255]))],
    "laranja":  [(np.array([7, 110, 80]), np.array([22, 255,255]))],
    "roxo":     [(np.array([125,60, 40]), np.array([165,255,255]))],
}

K3 = np.ones((3,3), np.uint8)
K5 = np.ones((5,5), np.uint8)
K7 = np.ones((7,7), np.uint8)

# ================================================================
#  [4] PRÉ-PROCESSADOR DE SÍMBOLO  (era PREP_SIGN.py)
#  Grayscale + threshold adaptativo → foca no desenho, ignora cor
# ================================================================

def prep_sign(img_bgr: np.ndarray, sz: int = 96) -> np.ndarray:
    """
    Converte crop BGR em tensor focado no símbolo da placa.
    Saída: RGB float32 (sz,sz,3) em [0,255] para MobileNetV2.
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((sz, sz, 3), np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Padding proporcional
    hh, ww  = gray.shape
    scale   = sz / max(hh, ww)
    nh, nw  = max(1, int(hh*scale)), max(1, int(ww*scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((sz, sz), np.uint8)
    canvas[(sz-nh)//2:(sz-nh)//2+nh, (sz-nw)//2:(sz-nw)//2+nw] = resized

    # CLAHE leve + threshold adaptativo
    cl     = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
    canvas = cl.apply(canvas)
    thresh = cv2.adaptiveThreshold(
        canvas, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=13, C=5
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB).astype(np.float32)


# ================================================================
#  [5] MODELO TFLite
# ================================================================

def carregar_modelo(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[CNN] Modelo não encontrado: {model_path}\n"
            f"      Execute: python TRAIN_SIGN_CNN.py --dataset ./dataset_tratado"
        )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"[CNN] Labels não encontrados: {labels_path}")

    with open(labels_path) as f:
        labels = [l.strip() for l in f if l.strip()]

    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=model_path)

    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    sz      = interp.get_input_details()[0]["shape"][1]
    print(f"[CNN] {len(labels)} classes: {labels} | entrada {sz}×{sz}", flush=True)
    return interp, labels, in_idx, out_idx


def classificar(crop_bgr, interp, labels, in_idx, out_idx):
    """Retorna (label, confiança, probs_array)."""
    sz    = interp.get_input_details()[0]["shape"][1]
    batch = np.expand_dims(prep_sign(crop_bgr, sz), 0)
    interp.set_tensor(in_idx, batch)
    interp.invoke()
    probs = interp.get_tensor(out_idx)[0]
    best  = int(np.argmax(probs))
    return labels[best], float(probs[best]), probs


# ================================================================
#  [6] DETECÇÃO DE CANDIDATAS (HSV bilateral)
# ================================================================

def _mascara(hsv):
    m = np.zeros(hsv.shape[:2], np.uint8)
    for ranges in HSV_RANGES.values():
        for lo, hi in ranges:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
    return m


def localizar_candidatas(frame):
    h, w  = frame.shape[:2]
    y0, y1 = int(h*ROI_Y0), int(h*ROI_Y1)
    roi   = frame[y0:y1]
    hsv   = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (5,5), 0)
    mask  = _mascara(hsv)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K7)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx   = w // 2
    esq, dir_, rej = [], [], []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (AREA_MIN < area < AREA_MAX):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        crop = frame[y+y0:y+y0+bh, x:x+bw]
        if crop.size == 0:
            continue
        prop  = bw / max(bh, 1)
        compac = area / max(bw*bh, 1)
        if not (0.35 <= prop <= 2.2) or compac < 0.28:
            rej.append((x, y+y0, bw, bh, f"p={prop:.1f}"))
            continue
        entry = (x, y+y0, bw, bh, area, crop)
        (esq if x+bw//2 < cx else dir_).append(entry)

    esq.sort(key=lambda b: b[4], reverse=True)
    dir_.sort(key=lambda b: b[4], reverse=True)
    return esq[:2], dir_[:2], rej


def detectar_obstaculo(frame):
    h, w  = frame.shape[:2]
    roi   = frame[int(h*OBST_Y0):]
    gray  = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (15,15), 0)
    thresh = cv2.adaptiveThreshold(gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 10)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, K7)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  K5)
    for cnt in cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < OBST_AREA_MIN:
            continue
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bh > bw*0.4 and bw < w*0.85:
            return True
    return False


# ================================================================
#  [7] CONFIRMAÇÃO BILATERAL
# ================================================================

def _na_borda(x, bw, w):
    cx = x + bw//2
    lim = int(w * BORDA_FRAC)
    return cx < lim or cx > w - lim


def confirmar_lado(estado, label, area, x, bw, w):
    if _nav["cooldown"] > 0:
        estado.update(label=None, cnt=0); return None
    if label is None:
        estado.update(cnt=0, label=None); return None
    if label == estado["label"]:
        estado["cnt"] += 1; estado["area"] = area
    else:
        estado.update(label=label, cnt=1, area=area)
    borda = _na_borda(x, bw, w)
    if estado["cnt"] >= CONFIRM_N and (area >= AREA_EXEC or
       (borda and estado["cnt"] >= max(2, CONFIRM_N//2))):
        _nav["gatilho"] = "BORDA" if borda else "AREA"
        estado.update(cnt=0, label=None)
        return label
    return None


# ================================================================
#  [8] ESTADO / SERIAL / EXECUTOR
# ================================================================

CMD = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
_conf_esq = dict(label=None, cnt=0, area=0.0)
_conf_dir = dict(label=None, cnt=0, area=0.0)
_nav = dict(cooldown=0, ultimo=None, acao_label=None,
            acao_t=0.0, acao_dur=0.0, gatilho="")


def conectar_serial():
    kws = ["arduino","ch340","cp210","uart"]
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower() for k in kws):
            try:
                s = serial.Serial(p.device, BAUD, timeout=0, write_timeout=0)
                time.sleep(2); s.reset_input_buffer()
                print(f"[SER] {p.device}", flush=True); return s
            except Exception: pass
    try:
        s = serial.Serial(SERIAL_PORT, BAUD, timeout=0, write_timeout=0)
        time.sleep(2); s.reset_input_buffer()
        print(f"[SER] {SERIAL_PORT}", flush=True); return s
    except Exception as e:
        print(f"[SER] Simulação ({e})", flush=True); return None


def enviar(ser):
    j = (f'{{"mot":{CMD["mot"]},"srv":{CMD["srv"]},'
         f'"buz":{CMD["buz"]},"led":{CMD["led"]},'
         f'"brk":{CMD["brk"]},"dir":{CMD["dir"]},'
         f'"spd":{CMD["spd"]}}}')
    print(f"[CMD] {j}", flush=True)
    if ser:
        try: ser.write((j+"\n").encode())
        except Exception: pass


def executar(label, ser):
    acao = LABEL_TO_ACTION.get(label)
    if not acao or acao not in ACOES: return
    a = ACOES[acao]
    CMD.update(mot=a["mot"], srv=a["srv"], buz=a["buz"],
               led=a["led"], brk=a["brk"], dir=a["dir"],
               spd=0 if a["mot"]==0 else (1 if a["mot"]<50 else 2))
    enviar(ser)
    _nav.update(acao_label=acao, acao_t=time.monotonic(),
                acao_dur=a["dur"], cooldown=COOLDOWN, ultimo=label)
    print(f"[NAV] ▶ {acao} ← '{label}' [{_nav['gatilho']}]", flush=True)


def tick(ser):
    lbl = _nav["acao_label"]
    if lbl is None: return False
    if lbl == "OBSTACLE": return True
    if time.monotonic() - _nav["acao_t"] >= _nav["acao_dur"]:
        CMD.update(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
        enviar(ser)
        print(f"[NAV] ✅ {lbl}", flush=True)
        _nav["acao_label"] = None; return False
    return True


# ================================================================
#  [9] VISUALIZAÇÃO
# ================================================================

def desenhar(frame, esq, dir_, rej, lbl_e, lbl_d, conf_e, conf_d, fps, obst):
    out = frame.copy()
    h, w = out.shape[:2]
    cx   = w//2
    borda_px = int(w*BORDA_FRAC)

    cv2.rectangle(out, (0,int(h*ROI_Y0)), (w-1,int(h*ROI_Y1)), (0,200,255), 1)
    for yy in range(int(h*ROI_Y0), int(h*ROI_Y1), 8):
        cv2.line(out, (cx,yy), (cx,yy+4), (0,200,255), 1)
    for yy in range(int(h*ROI_Y0), int(h*ROI_Y1), 6):
        cv2.line(out, (borda_px,yy), (borda_px,yy+3), (0,100,255), 1)
        cv2.line(out, (w-borda_px,yy), (w-borda_px,yy+3), (0,100,255), 1)

    for x,y,bw,bh,mot in rej[:3]:
        cv2.rectangle(out,(x,y),(x+bw,y+bh),(60,60,150),1)

    def _box(lista, lbl, conf, estado):
        if not lista: return
        x,y,bw,bh,area,_ = lista[0]
        acao = LABEL_TO_ACTION.get(lbl,"") if lbl else ""
        cor  = CORES_VIS.get(acao,(160,160,160))
        thick = 3 if _na_borda(x,bw,w) else 2
        cv2.rectangle(out,(x,y),(x+bw,y+bh),cor,thick)
        if lbl:
            txt = f"{lbl} {conf:.0%} [{estado['cnt']}/{CONFIRM_N}]"
            cv2.rectangle(out,(x,y-18),(x+len(txt)*7,y),cor,-1)
            cv2.putText(out,txt,(x+2,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1)
            prog = int(bw*min(estado["cnt"],CONFIRM_N)/CONFIRM_N)
            cv2.rectangle(out,(x,y+bh+2),(x+bw,y+bh+6),(40,40,40),-1)
            cv2.rectangle(out,(x,y+bh+2),(x+prog,y+bh+6),cor,-1)

    _box(esq, lbl_e, conf_e, _conf_esq)
    _box(dir_, lbl_d, conf_d, _conf_dir)

    if obst:
        cv2.rectangle(out,(0,int(h*OBST_Y0)),(w-1,h-1),(0,0,255),2)
        cv2.putText(out,"OBSTACULO",(w//2-55,h-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    pw = 200
    pan = np.zeros((h,pw,3),np.uint8); pan[:]=(18,18,18)
    cv2.rectangle(pan,(0,0),(pw-1,h-1),(45,45,45),1)
    def t(s,l,cor=(190,190,190),sc=0.34):
        cv2.putText(pan,s,(5,14+l*16),cv2.FONT_HERSHEY_SIMPLEX,sc,cor,1)

    t(f"FPS:{fps}",0,(255,255,255),0.40)
    acao = _nav["acao_label"]
    if acao:
        cor_a = CORES_VIS.get(acao,(200,200,200))
        t(f"EXEC:{acao}",1,cor_a,0.40)
        if acao!="OBSTACLE":
            t(f" {time.monotonic()-_nav['acao_t']:.1f}s/{_nav['acao_dur']}s",2,cor_a)
    else:
        t("livre",1,(100,200,100))
        t(f"cd:{_nav['cooldown']}",2,(80,80,80))
    t("─ESQ─",4,(60,60,60))
    t(f"'{lbl_e or '-'}' {conf_e:.0%}",5,CORES_VIS.get(LABEL_TO_ACTION.get(lbl_e,""),  (140,140,140)))
    t(f"f:{_conf_esq['cnt']}/{CONFIRM_N}",6)
    t("─DIR─",8,(60,60,60))
    t(f"'{lbl_d or '-'}' {conf_d:.0%}",9,CORES_VIS.get(LABEL_TO_ACTION.get(lbl_d,""),  (140,140,140)))
    t(f"f:{_conf_dir['cnt']}/{CONFIRM_N}",10)
    t("─ÚLTIMO─",12,(60,60,60))
    t(f"  {_nav['ultimo'] or '-'}",13,CORES_VIS.get(LABEL_TO_ACTION.get(_nav['ultimo'],""), (160,160,160)))
    t("OBST:SIM" if obst else "obst:nao",15,(0,0,255) if obst else (80,180,80))

    return np.hstack([out,pan])


# ================================================================
#  [10] MODO --test (diagnóstico com foto)
# ================================================================

def modo_teste(caminhos, conf_min, top_n):
    try:
        interp, labels, in_idx, out_idx = carregar_modelo()
    except (FileNotFoundError, ImportError) as e:
        print(e); sys.exit(1)

    sz = interp.get_input_details()[0]["shape"][1]

    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None:
            print(f"[ERRO] {caminho}"); continue

        h, w = img.shape[:2]
        lbl, conf, probs = classificar(img, interp, labels, in_idx, out_idx)
        top_idx = np.argsort(probs)[::-1][:min(top_n, len(labels))]

        print(f"\n{'='*55}")
        print(f"  {os.path.basename(caminho)}  ({w}×{h}px)")
        print(f"{'='*55}")
        print(f"  #   Lbl   Conf    Classe               Ação")
        print(f"  {'-'*52}")
        for i, idx in enumerate(top_idx):
            l = labels[idx]; c = probs[idx]
            acao  = LABEL_TO_ACTION.get(l,"—")
            nome  = LABEL_NAMES.get(l,l)
            marca = "◄" if i==0 else " "
            dim   = "" if c >= conf_min else "  (< limiar)"
            print(f"  {i+1}{marca} '{l}'  {c*100:5.1f}%  {nome:<22} {acao}{dim}")

        best_lbl = labels[top_idx[0]]
        best_conf = probs[top_idx[0]]
        print(f"\n  {'AÇÃO' if best_conf>=conf_min else 'BAIXA CONF'}: "
              f"'{best_lbl}' {best_conf*100:.1f}% → "
              f"{LABEL_TO_ACTION.get(best_lbl,'—') if best_conf>=conf_min else 'não dispara'}")
        print(f"{'='*55}\n")

        # Painel visual: original | símbolo extraído
        orig_r   = cv2.resize(img, (sz, sz))
        prep_vis = cv2.cvtColor(prep_sign(img,sz).astype(np.uint8), cv2.COLOR_RGB2BGR)
        painel   = np.hstack([orig_r, prep_vis])
        cv2.putText(painel, "ORIGINAL", (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,(0,220,80),1)
        cv2.putText(painel, "SIMBOLO",  (sz+2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,(0,220,80),1)
        cv2.putText(painel, f"'{best_lbl}' {best_conf*100:.0f}%",
                    (4,sz-6), cv2.FONT_HERSHEY_SIMPLEX, 0.40,(0,220,80),1)
        cv2.imshow(f"SIGN_DETECTOR — {os.path.basename(caminho)}", painel)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ================================================================
#  [11] MODO --collect (coleta de amostras)
# ================================================================

def modo_collect(label, n_alvo, cam_idx, delay, manual, out_dir, sz):
    pasta = os.path.join(out_dir, "train", label)
    os.makedirs(pasta, exist_ok=True)
    existentes = len([f for f in os.listdir(pasta) if f.endswith(".jpg")])
    print(f"\n[COLLECT] '{label}' | alvo: {existentes+n_alvo} | pasta: {pasta}")
    print(f"[COLLECT] {'MANUAL (ESPAÇO)' if manual else f'AUTO ({delay}s)'} | Q=sair\n")

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    contador = existentes
    t_ultima = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        vis   = frame.copy()
        h, w  = frame.shape[:2]
        y0, y1 = int(h*0.05), int(h*0.90)
        roi   = frame[y0:y1]
        hsv   = cv2.GaussianBlur(cv2.cvtColor(roi,cv2.COLOR_BGR2HSV),(5,5),0)
        mask  = np.zeros(hsv.shape[:2],np.uint8)
        for ranges in HSV_RANGES.values():
            for lo,hi in ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv,lo,hi))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,K5)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,K3)
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                x,y,bw,bh = cv2.boundingRect(c)
                if 0.3 < bw/max(bh,1) < 2.5:
                    pad = int(max(bw,bh)*0.10)
                    bbox = (max(0,x-pad), max(0,y+y0-pad),
                            min(w,x+bw+pad), min(h,y+y0+bh+pad))
                    cv2.rectangle(vis,bbox[:2],bbox[2:],(0,220,80),2)

        agora = time.monotonic()
        total_alvo = existentes+n_alvo
        prog  = int((contador-existentes)/n_alvo*30) if n_alvo>0 else 0
        barra = "█"*prog + "░"*(30-prog)
        cv2.putText(vis,f"[{barra}] {contador}/{total_alvo}",(10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),1)
        cv2.putText(vis,f"Label: {label}",(10,45),
                    cv2.FONT_HERSHEY_SIMPLEX,0.50,(200,200,255),1)

        cv2.imshow(f"COLLECT — {label}", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break

        capturar = (bbox and not manual and (agora-t_ultima) >= delay) or \
                   (k == ord(' ') and bbox)
        if capturar and bbox:
            x1,y1_,x2,y2 = bbox
            crop = frame[y1_:y2, x1:x2]
            if crop.size == 0: continue
            img_s = cv2.cvtColor(prep_sign(crop,sz).astype(np.uint8), cv2.COLOR_RGB2BGR)
            caminho = os.path.join(pasta, f"{contador:05d}.jpg")
            cv2.imwrite(caminho, img_s)
            contador += 1; t_ultima = agora
            print(f"  [{contador}/{total_alvo}] {caminho}", flush=True)
            if contador >= total_alvo:
                print(f"[OK] {n_alvo} amostras coletadas"); break

    cap.release(); cv2.destroyAllWindows()
    print(f"[COLLECT] Total em '{pasta}': {contador}")


# ================================================================
#  [12] MODO CÂMERA/VÍDEO (vida real)
# ================================================================

def modo_camera(usar_camera):
    try:
        interp, labels, in_idx, out_idx = carregar_modelo()
    except (FileNotFoundError, ImportError) as e:
        print(e); sys.exit(1)

    src = CAM_IDX if usar_camera else VIDEO
    print(f"[CAM] {'Câmera' if usar_camera else 'Vídeo'}: {src}", flush=True)

    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERRO] Fonte de vídeo não abriu"); sys.exit(1)
    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ser = conectar_serial()
    fps_t = time.monotonic(); fps_n = fps = 0
    print("[OK] Rodando — Q para sair\n", flush=True)

    while True:
        ret, frame_big = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        h = round(frame_big.shape[0]/PROP)
        w = round(frame_big.shape[1]/PROP)
        frame = cv2.resize(frame_big, (w,h))

        if _nav["cooldown"] > 0: _nav["cooldown"] -= 1
        em_acao = tick(ser)

        esq, dir_, rej = localizar_candidatas(frame)
        lbl_e = lbl_d = None
        conf_e = conf_d = 0.0
        obst = False

        if _nav["cooldown"] == 0 and not em_acao:
            if esq:
                x,y,bw,bh,area,crop = esq[0]
                lbl_e, conf_e, _ = classificar(crop, interp, labels, in_idx, out_idx)
                if conf_e < CONF_MIN: lbl_e = None
                confirmado = confirmar_lado(_conf_esq, lbl_e, area, x, bw, w)
                if confirmado: executar(confirmado, ser)
            else:
                confirmar_lado(_conf_esq, None, 0, 0, 0, w)

            if dir_ and not em_acao and _nav["cooldown"] == 0:
                x,y,bw,bh,area,crop = dir_[0]
                lbl_d, conf_d, _ = classificar(crop, interp, labels, in_idx, out_idx)
                if conf_d < CONF_MIN: lbl_d = None
                confirmado = confirmar_lado(_conf_dir, lbl_d, area, x, bw, w)
                if confirmado: executar(confirmado, ser)
            else:
                confirmar_lado(_conf_dir, None, 0, 0, 0, w)

            if not esq and not dir_:
                obst = detectar_obstaculo(frame)
                if obst and _nav["acao_label"] is None:
                    executar("OBSTACLE", ser)

        if _nav["acao_label"] == "OBSTACLE":
            obst = detectar_obstaculo(frame)
            if not obst:
                CMD.update(mot=0,srv=127,buz=0,led=0,brk=0,dir=0,spd=0)
                enviar(ser)
                _nav["acao_label"] = None

        vis = desenhar(frame, esq, dir_, rej, lbl_e, lbl_d, conf_e, conf_d, fps, obst)
        cv2.imshow("SIGN_DETECTOR", vis)

        fps_n += 1
        if time.monotonic()-fps_t >= 1.0:
            fps=fps_n; fps_n=0; fps_t=time.monotonic()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CMD.update(mot=0,srv=127,buz=0,led=0,brk=1,dir=0,spd=0)
    enviar(ser)
    if ser: ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="SIGN_DETECTOR — detector de placas",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--test",    nargs="+", metavar="FOTO",
                    help="Testa uma ou mais fotos")
    ap.add_argument("--conf",    type=float, default=0.60,
                    help="Confiança mínima (padrão: 0.60)")
    ap.add_argument("--top",     type=int,   default=3,
                    help="Quantos resultados mostrar no --test")
    ap.add_argument("--cam",     action="store_true",
                    help="Câmera USB ao vivo")
    ap.add_argument("--collect", metavar="LABEL",
                    help="Coleta amostras: --collect Pare")
    ap.add_argument("--n",       type=int,   default=200,
                    help="Amostras a coletar (padrão: 200)")
    ap.add_argument("--delay",   type=float, default=0.2,
                    help="Segundos entre capturas automáticas")
    ap.add_argument("--manual",  action="store_true",
                    help="Captura só com ESPAÇO")
    ap.add_argument("--out",     default="./dataset_circuito",
                    help="Pasta do dataset para --collect")
    args = ap.parse_args()

    if args.test:
        modo_teste(args.test, args.conf, args.top)
    elif args.collect:
        modo_collect(args.collect, args.n, CAM_IDX, args.delay,
                     args.manual, args.out, 96)
    else:
        modo_camera(usar_camera=args.cam)