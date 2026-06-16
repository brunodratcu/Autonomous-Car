"""
================================================================
  SIGN_DETECTOR.py — Detector de placas
  ─────────────────────────────────────────────────────────────
  USO:
    python SIGN_DETECTOR.py            ← vídeo (videoplayback.mp4)
    python SIGN_DETECTOR.py --cam      ← câmera USB
    python SIGN_DETECTOR.py --test foto.jpg
    python SIGN_DETECTOR.py --collect Pare

  Necessário na mesma pasta:
    sign_model.tflite   ← gerado por TRAIN_SIGN_CNN.py
    sign_labels.txt     ← gerado por TRAIN_SIGN_CNN.py
================================================================
"""

import cv2, numpy as np, serial, serial.tools.list_ports
import time, sys, os, argparse

# ================================================================
#  CONFIGURAÇÃO
# ================================================================

VIDEO       = "./videoplayback.mp4"
CAM_IDX     = 0
PROP        = 2
SERIAL_PORT = "COM3"
BAUD        = 115200
MODEL_PATH  = "./sign_model.tflite"
LABELS_PATH = "./sign_labels.txt"

ROI_Y0      = 0.05
ROI_Y1      = 0.72
BORDA_FRAC  = 0.18

# Filtros geométricos
CIRC_MIN    = 0.50    # circularity — elimina vegetação/ruído
SOLID_MIN   = 0.78    # solidity    — forma convexa sólida
EXTENT_MIN  = 0.48    # extent      — preenche a bbox
PROP_MIN    = 0.25    # W/H mínimo  — aceita semáforo vertical
PROP_MAX    = 3.50    # W/H máximo
AREA_MIN    = 500
AREA_MAX    = 60_000

# CNN
CONFIRM_N   = 4
AREA_EXEC   = 500
COOLDOWN    = 55
CONF_MIN    = 0.65
CONF_BORDA  = 0.50    # limiar reduzido quando placa está na borda

# Obstáculo
OBST_Y0     = 0.55
OBST_MIN    = 3500

K3 = np.ones((3,3), np.uint8)
K5 = np.ones((5,5), np.uint8)
K7 = np.ones((7,7), np.uint8)

# ================================================================
#  AÇÕES E LABELS
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

# Labels = nomes completos (dataset_v3: 8 classes)
LABEL_TO_ACTION = {
    "Stop":       "STOP",
    "Esquerda":   "LEFT",
    "Direita":    "RIGHT",
    "SemRetorno": "STRAIGHT",
    "Verde":      "SPEED_UP",
    "Cone":       "OBSTACLE",
    "Carro":      "OBSTACLE",
    "Pessoa":     "OBSTACLE",
    # Compatibilidade legada
    "S":"STOP","L":"LEFT","R":"RIGHT","N":"STRAIGHT",
    "G":"SPEED_UP","W":"SLOW_DOWN","Y":"YIELD",
}
LABEL_NAMES = {
    "Stop":       "Pare/Stop",
    "Esquerda":   "Vira Esquerda",
    "Direita":    "Vira Direita",
    "SemRetorno": "Sem Retorno",
    "Verde":      "Semáforo Verde",
    "Cone":       "Cone na Pista",
    "Carro":      "Carro na Pista",
    "Pessoa":     "Pessoa na Pista",
}
_OBSTACLE_LABELS = {"Cone","Carro","Pessoa"}
# Thresholds adaptativos — carregados de sign_thresholds.txt
_THRESHOLDS = {}
def _carregar_thresh():
    global _THRESHOLDS
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),"sign_thresholds.txt")
    if os.path.isfile(p):
        with open(p) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts)==2:
                    _THRESHOLDS[parts[0]] = float(parts[1])
        print(f"[CNN] Thresholds: {_THRESHOLDS}", flush=True)
    else:
        print("[CNN] sign_thresholds.txt não encontrado — usando CONF_MIN padrão", flush=True)
CORES = {
    "STOP":(50,50,220),"OBSTACLE":(0,0,200),"YIELD":(0,200,220),
    "LEFT":(220,120,0),"RIGHT":(0,120,220),"STRAIGHT":(50,220,50),
    "SLOW_DOWN":(0,180,255),"SPEED_UP":(0,200,100),
}

# ================================================================
#  ESTADO GLOBAL
# ================================================================

CMD       = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
_esq      = dict(label=None, cnt=0, area=0.0)
_dir      = dict(label=None, cnt=0, area=0.0)
_nav      = dict(cooldown=0, ultimo=None, acao_label=None,
                 acao_t=0.0, acao_dur=0.0, gatilho="")

# ================================================================
#  SERIAL
# ================================================================

def conectar_serial():
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower()
               for k in ["arduino","ch340","cp210","uart"]):
            try:
                s = serial.Serial(p.device, BAUD, timeout=0, write_timeout=0)
                time.sleep(2); s.reset_input_buffer()
                print(f"[SER] {p.device}", flush=True); return s
            except: pass
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
        except: pass

# ================================================================
#  PRÉ-PROCESSADOR — foca no símbolo interno da placa
# ================================================================

def prep_sign(img, sz=96):
    if img is None or img.size == 0:
        return np.zeros((sz, sz, 3), np.float32)
    h, w = img.shape[:2]
    # Remove 12% de borda colorida — CNN vê só o símbolo
    py, px = max(1,int(h*.12)), max(1,int(w*.12))
    roi = img[py:h-py, px:w-px]
    if roi.size == 0: roi = img
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hh, ww = gray.shape
    sc = sz / max(hh, ww)
    nh, nw = max(1,int(hh*sc)), max(1,int(ww*sc))
    rs = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    c  = np.zeros((sz, sz), np.uint8)
    c[(sz-nh)//2:(sz-nh)//2+nh, (sz-nw)//2:(sz-nw)//2+nw] = rs
    c  = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4)).apply(c)
    t  = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 13, 5)
    t  = cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((2,2),np.uint8))
    return cv2.cvtColor(t, cv2.COLOR_GRAY2RGB).astype(np.float32)

# ================================================================
#  PRÉ-PROCESSADOR DE OBSTÁCULOS — RGB preservado
# ================================================================

def prep_obstacle(img, sz=96):
    """Obstáculos: RGB completo com CLAHE suave."""
    if img is None or img.size == 0:
        return np.zeros((sz,sz,3), np.float32)
    img_r = cv2.resize(img,(sz,sz),interpolation=cv2.INTER_AREA)
    lab   = cv2.cvtColor(img_r,cv2.COLOR_BGR2LAB)
    lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4)).apply(lab[:,:,0])
    out = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB).astype(np.float32)

_OBSTACLE_LABELS = {"Cone","Carro","Pessoa"}

def prep_auto(img, label=None, sz=96):
    """Roteia para o pipeline correto baseado no label classificado."""
    if label in _OBSTACLE_LABELS:
        return prep_obstacle(img, sz)
    return prep_sign(img, sz)

# ================================================================
#  MODELO
# ================================================================

def carregar_modelo():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"[CNN] {MODEL_PATH} não encontrado\n"
            "      Execute: python TRAIN_SIGN_CNN.py --dataset ./dataset_novo")
    with open(LABELS_PATH) as f:
        labels = [l.strip() for l in f if l.strip()]
    try:
        import tflite_runtime.interpreter as tfl
        interp = tfl.Interpreter(model_path=MODEL_PATH)
    except ImportError:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    ii  = interp.get_input_details()[0]["index"]
    oi  = interp.get_output_details()[0]["index"]
    sz  = interp.get_input_details()[0]["shape"][1]
    _carregar_thresh()
    print(f"[CNN] {len(labels)} classes: {labels} | {sz}×{sz}", flush=True)
    return interp, labels, ii, oi

def classificar(crop, interp, labels, ii, oi):
    """
    Inferência com threshold adaptativo por classe.
    Retorna (label, confiança, probs).
    Se confiança < threshold da classe → retorna None.
    """
    sz    = interp.get_input_details()[0]["shape"][1]
    # Primeira passagem: usa prep_sign para todos
    # (obstáculos serão reclassificados se necessário)
    interp.set_tensor(ii, np.expand_dims(prep_sign(crop, sz), 0))
    interp.invoke()
    p     = interp.get_tensor(oi)[0]
    b     = int(np.argmax(p))
    lbl   = labels[b]; conf = float(p[b])

    # Se top label é obstáculo, reprocessa com RGB
    if lbl in _OBSTACLE_LABELS:
        interp.set_tensor(ii, np.expand_dims(prep_obstacle(crop, sz), 0))
        interp.invoke()
        p2   = interp.get_tensor(oi)[0]
        b2   = int(np.argmax(p2)); conf2 = float(p2[b2])
        # Usa o resultado mais confiante
        if conf2 > conf:
            p = p2; b = b2; lbl = labels[b]; conf = conf2

    return lbl, conf, p

# ================================================================
#  DETECÇÃO — HSV + validação geométrica
#  Sem Canny: reduz falsos positivos de bordas de interface/janelas
# ================================================================

# Ranges HSV calibrados para placas de trânsito impressas
_HSV = [
    (np.array([0,   70, 50]),  np.array([14,  255,255])),  # vermelho
    (np.array([162, 70, 50]),  np.array([180, 255,255])),  # vermelho wrap
    (np.array([92,  70, 45]),  np.array([130, 255,255])),  # azul
    (np.array([16, 110, 75]),  np.array([36,  255,255])),  # amarelo
    (np.array([36, 110, 50]),  np.array([86,  255,255])),  # verde
    (np.array([8,  100, 65]),  np.array([21,  255,255])),  # laranja
]

def _mascara_hsv(roi):
    hsv = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (5,5), 0)
    m   = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in _HSV:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, lo, hi))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K7)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  K5)
    return m

def _validar(cnt):
    area = cv2.contourArea(cnt)
    if not (AREA_MIN < area < AREA_MAX): return False, {}
    peri = cv2.arcLength(cnt, True)
    if peri < 10: return False, {}
    circ  = 4 * np.pi * area / (peri**2)
    hull  = cv2.convexHull(cnt)
    solid = area / max(cv2.contourArea(hull), 1)
    x, y, bw, bh = cv2.boundingRect(cnt)
    extent = area / max(bw*bh, 1)
    prop   = bw / max(bh, 1)
    m = dict(circ=circ, solid=solid, extent=extent,
             prop=prop, area=area, x=x, y=y, bw=bw, bh=bh)
    if circ   < CIRC_MIN:  return False, m
    if solid  < SOLID_MIN: return False, m
    if extent < EXTENT_MIN: return False, m
    if not (PROP_MIN <= prop <= PROP_MAX): return False, m
    return True, m

def localizar(frame):
    h, w  = frame.shape[:2]
    y0, y1 = int(h*ROI_Y0), int(h*ROI_Y1)
    roi   = frame[y0:y1]
    cx    = w // 2
    cnts, _ = cv2.findContours(_mascara_hsv(roi),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    esq, dir_, rej = [], [], []
    for cnt in cnts:
        ok, m = _validar(cnt)
        if not ok:
            if m: rej.append((m["x"], m["y"]+y0, m["bw"], m["bh"],
                              f"c={m['circ']:.2f} s={m['solid']:.2f}"))
            continue
        x, y, bw, bh = m["x"], m["y"]+y0, m["bw"], m["bh"]
        crop = frame[y:y+bh, x:x+bw]
        if crop.size == 0: continue
        entry = (x, y, bw, bh, m["area"], crop, m)
        (esq if x+bw//2 < cx else dir_).append(entry)
    esq.sort(key=lambda b: b[4], reverse=True)
    dir_.sort(key=lambda b: b[4], reverse=True)
    return esq[:2], dir_[:2], rej

def detectar_obst(frame):
    h, w = frame.shape[:2]
    roi  = frame[int(h*OBST_Y0):]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (15,15), 0)
    t    = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 51, 10)
    t    = cv2.morphologyEx(t, cv2.MORPH_CLOSE, K7)
    t    = cv2.morphologyEx(t, cv2.MORPH_OPEN,  K5)
    for cnt in cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) < OBST_MIN: continue
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bh > bw*.4 and bw < w*.85: return True
    return False

# ================================================================
#  CONFIRMAÇÃO BILATERAL
# ================================================================

def _borda(x, bw, w):
    cx = x + bw//2
    lim = int(w * BORDA_FRAC)
    return cx < lim or cx > w - lim

def confirmar(estado, label, area, x, bw, w):
    if _nav["cooldown"] > 0:
        estado.update(label=None, cnt=0); return None
    if label is None:
        estado.update(cnt=0, label=None); return None
    if label == estado["label"]:
        estado["cnt"] += 1; estado["area"] = area
    else:
        estado.update(label=label, cnt=1, area=area)
    b  = _borda(x, bw, w)
    pa = estado["cnt"] >= CONFIRM_N and area >= AREA_EXEC
    pb = b and estado["cnt"] >= max(2, CONFIRM_N//2)
    if pa or pb:
        _nav["gatilho"] = "BORDA" if pb and not pa else "AREA"
        estado.update(cnt=0, label=None)
        return label
    return None

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
#  VISUALIZAÇÃO
# ================================================================

def desenhar(frame, esq, dir_, rej, le, ld, ce, cd, fps, obst):
    out = frame.copy(); h, w = out.shape[:2]
    cx  = w//2; bpx = int(w*BORDA_FRAC)

    cv2.rectangle(out,(0,int(h*ROI_Y0)),(w-1,int(h*ROI_Y1)),(0,200,255),1)
    for yy in range(int(h*ROI_Y0), int(h*ROI_Y1), 8):
        cv2.line(out,(cx,yy),(cx,yy+4),(0,200,255),1)
    for yy in range(int(h*ROI_Y0), int(h*ROI_Y1), 6):
        cv2.line(out,(bpx,yy),(bpx,yy+3),(0,80,200),1)
        cv2.line(out,(w-bpx,yy),(w-bpx,yy+3),(0,80,200),1)
    cv2.putText(out,"ESQ",(4,int(h*ROI_Y0)+12),
                cv2.FONT_HERSHEY_SIMPLEX,0.33,(0,200,255),1)
    cv2.putText(out,"DIR",(w-30,int(h*ROI_Y0)+12),
                cv2.FONT_HERSHEY_SIMPLEX,0.33,(0,200,255),1)

    for x,y,bw,bh,mot in rej[:3]:
        cv2.rectangle(out,(x,y),(x+bw,y+bh),(50,50,120),1)
        cv2.putText(out,mot,(x,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.22,(80,80,160),1)

    def _box(lista, lbl, conf, estado):
        if not lista: return
        x,y,bw,bh,area,_,m = lista[0]
        acao = LABEL_TO_ACTION.get(lbl,"") if lbl else ""
        cor  = CORES.get(acao,(160,160,160))
        cv2.rectangle(out,(x,y),(x+bw,y+bh),cor,3 if _borda(x,bw,w) else 2)
        cv2.putText(out,f"c={m['circ']:.2f} s={m['solid']:.2f}",
                    (x,y+bh+12),cv2.FONT_HERSHEY_SIMPLEX,0.26,(180,180,80),1)
        if lbl:
            txt = f"{lbl}:{LABEL_NAMES.get(lbl,lbl)} {conf:.0%} [{estado['cnt']}/{CONFIRM_N}]"
            cv2.rectangle(out,(x,y-18),(x+len(txt)*7,y),cor,-1)
            cv2.putText(out,txt,(x+2,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)
            p = int(bw*min(estado["cnt"],CONFIRM_N)/CONFIRM_N)
            cv2.rectangle(out,(x,y+bh+2),(x+bw,y+bh+6),(40,40,40),-1)
            cv2.rectangle(out,(x,y+bh+2),(x+p,y+bh+6),cor,-1)
        if _borda(x,bw,w):
            cv2.putText(out,"BORDA",(x,y-22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,255,255),1)

    _box(esq, le, ce, _esq); _box(dir_, ld, cd, _dir)

    if obst:
        cv2.rectangle(out,(0,int(h*OBST_Y0)),(w-1,h-1),(0,0,255),2)
        cv2.putText(out,"OBSTACULO",(w//2-55,h-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    pw  = 200; pan = np.zeros((h,pw,3),np.uint8); pan[:]=(18,18,18)
    cv2.rectangle(pan,(0,0),(pw-1,h-1),(45,45,45),1)
    def t(s,l,cor=(190,190,190),sc=0.33):
        cv2.putText(pan,s,(5,14+l*16),cv2.FONT_HERSHEY_SIMPLEX,sc,cor,1)

    t(f"FPS:{fps}",0,(255,255,255),0.40)
    acao = _nav["acao_label"]
    if acao:
        ca = CORES.get(acao,(200,200,200))
        t(f"EXEC:{acao}",1,ca,0.40)
        if acao != "OBSTACLE":
            t(f" {time.monotonic()-_nav['acao_t']:.1f}s/{_nav['acao_dur']}s",2,ca)
        t(f" [{_nav['gatilho']}]",3,(200,200,0),0.28)
    else:
        t("livre",1,(100,200,100)); t(f"cd:{_nav['cooldown']}",2,(80,80,80))
    t("─ESQ─",5,(60,60,60))
    t(f"'{le or'-'}' {ce:.0%}",6,CORES.get(LABEL_TO_ACTION.get(le,""),(140,140,140)))
    t(f"f:{_esq['cnt']}/{CONFIRM_N}",7)
    t("─DIR─",9,(60,60,60))
    t(f"'{ld or'-'}' {cd:.0%}",10,CORES.get(LABEL_TO_ACTION.get(ld,""),(140,140,140)))
    t(f"f:{_dir['cnt']}/{CONFIRM_N}",11)
    t("─ÚLTIMO─",13,(60,60,60))
    ult = _nav["ultimo"] or "-"
    t(f"  {ult}",14,CORES.get(LABEL_TO_ACTION.get(ult,""),(160,160,160)))
    t("OBST:SIM" if obst else "obst:não",16,(0,0,255) if obst else(80,180,80))

    return np.hstack([out, pan])

# ================================================================
#  MODO --test
# ================================================================

def modo_teste(caminhos, conf_min, top_n, interp, labels, ii, oi):
    sz = interp.get_input_details()[0]["shape"][1]
    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None: print(f"[ERRO] {caminho}"); continue
        h, w = img.shape[:2]
        lbl, conf, probs = classificar(img, interp, labels, ii, oi)
        top = np.argsort(probs)[::-1][:min(top_n, len(labels))]

        # Métricas geométricas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gstr = "—"
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            a = cv2.contourArea(c); p = cv2.arcLength(c, True)
            if p > 0:
                ci = 4*np.pi*a/p**2
                so = a/max(cv2.contourArea(cv2.convexHull(c)),1)
                x,y,bw,bh = cv2.boundingRect(c)
                ex = a/max(bw*bh,1); pr = bw/max(bh,1)
                gstr=(f"circ={ci:.2f}{'✅' if ci>=CIRC_MIN else '❌'}  "
                      f"solid={so:.2f}{'✅' if so>=SOLID_MIN else '❌'}  "
                      f"ext={ex:.2f}{'✅' if ex>=EXTENT_MIN else '❌'}  "
                      f"W/H={pr:.2f}{'✅' if PROP_MIN<=pr<=PROP_MAX else '❌'}")

        print(f"\n{'='*58}")
        print(f"  {os.path.basename(caminho)}  ({w}×{h}px)")
        print(f"  Geometria: {gstr}")
        print(f"{'='*58}")
        for i, idx in enumerate(top):
            l=labels[idx]; c=probs[idx]
            mk="◄" if i==0 else " "
            dim="" if c>=conf_min else "  (<limiar)"
            print(f"  {i+1}{mk} '{l}'  {c*100:5.1f}%  "
                  f"{LABEL_NAMES.get(l,l):<18} {LABEL_TO_ACTION.get(l,'—')}{dim}")
        best = labels[top[0]]; bc = probs[top[0]]
        print(f"\n  {'AÇÃO' if bc>=conf_min else 'BAIXA CONF'}: "
              f"'{best}' {bc*100:.1f}% → "
              f"{LABEL_TO_ACTION.get(best,'—') if bc>=conf_min else 'não dispara'}")
        print(f"{'='*58}\n")

        orig = cv2.resize(img,(sz,sz))
        symb = cv2.cvtColor(prep_sign(img,sz).astype(np.uint8), cv2.COLOR_RGB2BGR)
        pan  = np.hstack([orig,symb])
        cv2.putText(pan,"ORIGINAL",(2,12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,220,80),1)
        cv2.putText(pan,"SÍMBOLO",(sz+2,12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,220,80),1)
        cv2.putText(pan,f"'{best}' {bc*100:.0f}%",
                    (4,sz-6),cv2.FONT_HERSHEY_SIMPLEX,0.40,(0,220,80),1)
        cv2.imshow(f"TEST — {os.path.basename(caminho)}", pan)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================================================================
#  MODO --collect
# ================================================================

def modo_collect(label, n_alvo, cam_idx, delay, manual, out_dir):
    pasta = os.path.join(out_dir, "train", label)
    os.makedirs(pasta, exist_ok=True)
    exist = len([f for f in os.listdir(pasta) if f.endswith(".jpg")])
    print(f"\n[COLLECT] '{label}' | alvo: {exist+n_alvo} | Q=sair\n")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    cnt = exist; t_ul = 0.0
    while True:
        ret, frame = cap.read()
        if not ret: break
        vis = frame.copy(); h, w = frame.shape[:2]
        roi = frame[int(h*.05):int(h*.90)]
        cnts,_ = cv2.findContours(_mascara_hsv(roi),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = None; ba = 0
        for c in cnts:
            ok, m = _validar(c)
            if not ok or m.get("area",0) <= ba: continue
            pad = int(max(m["bw"],m["bh"])*.08)
            x1=max(0,m["x"]-pad); y1=max(0,m["y"]+int(h*.05)-pad)
            x2=min(w,m["x"]+m["bw"]+pad); y2=min(h,m["y"]+int(h*.05)+m["bh"]+pad)
            bbox=(x1,y1,x2,y2); ba=m["area"]
        if bbox:
            cv2.rectangle(vis,bbox[:2],bbox[2:],(0,220,80),2)
        agora = time.monotonic()
        total = exist+n_alvo
        prog  = int((cnt-exist)/n_alvo*30) if n_alvo>0 else 0
        cv2.putText(vis,f"[{'█'*prog+'░'*(30-prog)}] {cnt}/{total}",
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
        cv2.putText(vis,f"Label: {label}",
                    (10,45),cv2.FONT_HERSHEY_SIMPLEX,0.48,(200,200,255),1)
        cv2.imshow(f"COLLECT — {label}", vis)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        capturar = (bbox and not manual and (agora-t_ul)>=delay) or (k==32 and bbox)
        if capturar and bbox:
            x1,y1,x2,y2=bbox; crop=frame[y1:y2,x1:x2]
            if crop.size==0: continue
            img_s=cv2.cvtColor(prep_sign(crop,96).astype(np.uint8),cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(pasta,f"{cnt:05d}.jpg"),img_s)
            cnt+=1; t_ul=agora; print(f"  [{cnt}/{total}]",flush=True)
            if cnt>=total: print(f"[OK] {n_alvo} amostras"); break
    cap.release(); cv2.destroyAllWindows()
    print(f"[COLLECT] {cnt} amostras em {pasta}")

# ================================================================
#  LOOP PRINCIPAL — câmera ou vídeo
# ================================================================

def main(usar_camera):
    try:
        interp, labels, ii, oi = carregar_modelo()
    except (FileNotFoundError, ImportError) as e:
        print(e); sys.exit(1)

    src = CAM_IDX if usar_camera else VIDEO
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[ERRO] Fonte não abriu: {src}"); sys.exit(1)
    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cap.set(cv2.CAP_PROP_FPS,30); cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

    ser   = conectar_serial()
    fps_t = time.monotonic(); fps_n = fps = 0
    print(f"[OK] {'Câmera' if usar_camera else 'Vídeo'}: {src} — Q para sair\n",flush=True)

    while True:
        ret, fb = cap.read()
        if not ret:
            if not usar_camera: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
            break

        h=round(fb.shape[0]/PROP); w=round(fb.shape[1]/PROP)
        frame = cv2.resize(fb,(w,h))

        if _nav["cooldown"]>0: _nav["cooldown"]-=1
        em_acao = tick(ser)

        esq,dir_,rej = localizar(frame)
        le=ld=None; ce=cd=0.0; obst=False

        if _nav["cooldown"]==0 and not em_acao:
            if esq:
                x,y,bw,bh,area,crop,_ = esq[0]
                le,ce,_ = classificar(crop,interp,labels,ii,oi)
                thr_e = _THRESHOLDS.get(le, CONF_BORDA if _borda(x,bw,w) else CONF_MIN)
                if ce < thr_e: le=None
                conf = confirmar(_esq,le,area,x,bw,w)
                if conf: executar(conf,ser)
            else:
                confirmar(_esq,None,0,0,0,w)

            if dir_ and not em_acao and _nav["cooldown"]==0:
                x,y,bw,bh,area,crop,_ = dir_[0]
                ld,cd,_ = classificar(crop,interp,labels,ii,oi)
                thr_d = _THRESHOLDS.get(ld, CONF_BORDA if _borda(x,bw,w) else CONF_MIN)
                if cd < thr_d: ld=None
                conf = confirmar(_dir,ld,area,x,bw,w)
                if conf: executar(conf,ser)
            else:
                confirmar(_dir,None,0,0,0,w)

            if not esq and not dir_:
                obst = detectar_obst(frame)
                if obst and _nav["acao_label"] is None:
                    executar("OBSTACLE",ser)

        if _nav["acao_label"]=="OBSTACLE":
            if not detectar_obst(frame):
                CMD.update(mot=0,srv=127,buz=0,led=0,brk=0,dir=0,spd=0)
                enviar(ser); _nav["acao_label"]=None

        vis = desenhar(frame,esq,dir_,rej,le,ld,ce,cd,fps,obst)
        cv2.imshow("SIGN_DETECTOR",vis)

        fps_n+=1
        if time.monotonic()-fps_t>=1.0:
            fps=fps_n; fps_n=0; fps_t=time.monotonic()

        if cv2.waitKey(1)&0xFF==ord('q'): break

    CMD.update(mot=0,srv=127,buz=0,led=0,brk=1,dir=0,spd=0)
    enviar(ser)
    if ser: ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] Encerrado.",flush=True)

# ================================================================
#  ENTRY POINT
# ================================================================

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="SIGN_DETECTOR",
                               formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cam",     action="store_true", help="Câmera USB ao vivo")
    ap.add_argument("--test",    nargs="+", metavar="FOTO")
    ap.add_argument("--conf",    type=float, default=CONF_MIN)
    ap.add_argument("--top",     type=int,   default=3)
    ap.add_argument("--collect", metavar="LABEL")
    ap.add_argument("--n",       type=int,   default=200)
    ap.add_argument("--delay",   type=float, default=0.2)
    ap.add_argument("--manual",  action="store_true")
    ap.add_argument("--out",     default="./dataset_circuito")
    args=ap.parse_args()

    if args.test:
        try: interp,labels,ii,oi=carregar_modelo()
        except (FileNotFoundError,ImportError) as e: print(e); sys.exit(1)
        modo_teste(args.test,args.conf,args.top,interp,labels,ii,oi)
    elif args.collect:
        modo_collect(args.collect,args.n,CAM_IDX,args.delay,args.manual,args.out)
    else:
        main(usar_camera=args.cam)