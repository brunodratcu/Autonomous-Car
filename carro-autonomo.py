"""
================================================================
  SIGN_DETECTOR.py  v4.0 — YOLOv8n + ByteTrack + ONNX
  ─────────────────────────────────────────────────────────────
  Pipeline:
    Câmera/Vídeo
       ↓
    YOLOv8n (ONNX Runtime — CPU otimizado)
       ↓
    Bounding Boxes + Classe + Confiança
       ↓
    ByteTrack  (ID persistente por objeto)
       ↓
    Threshold por classe (sign_thresholds.txt)
       ↓
    Filtro temporal  (N frames consecutivos)
       ↓
    Test Time Augmentation  (flip + brilho → média)
       ↓
    LABEL_TO_ACTION
       ↓
    Arduino (JSON serial)

  INSTALAÇÃO:
    pip install ultralytics onnxruntime opencv-python pyserial

  USO:
    python SIGN_DETECTOR.py --export        ← exporta YOLOv8n→ONNX (1x)
    python SIGN_DETECTOR.py                 ← vídeo (videoplayback.mp4)
    python SIGN_DETECTOR.py --cam           ← câmera USB
    python SIGN_DETECTOR.py --test foto.jpg ← diagnóstico

  ARQUIVOS NECESSÁRIOS:
    yolov8n_signs.onnx      ← gerado por --export (ou treinado)
    sign_thresholds.txt     ← gerado por TRAIN_SIGN_CNN.py
================================================================
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import os
import argparse
import collections

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ================================================================
#  [1] CONFIGURAÇÃO
# ================================================================

VIDEO        = "./videoplayback.mp4"
CAM_IDX      = 0
PROP         = 1          # sem redução — YOLO precisa de resolução
SERIAL_PORT  = "COM3"
BAUD         = 115200

# Modelo ONNX
ONNX_PATH    = "./yolov8n_signs.onnx"
PT_PATH      = "./yolov8n_signs.pt"   # fallback se ONNX não existir
THRESH_PATH  = "./sign_thresholds.txt"

# Detecção
CONF_DEFAULT = 0.45    # confiança mínima YOLO
IOU_THRESH   = 0.45    # NMS IoU
IMGSZ        = 640     # tamanho de entrada YOLO

# Filtro temporal — exige N detecções consecutivas do mesmo ID
CONFIRM_N    = 4       # frames para confirmar placa
CONFIRM_OBS  = 2       # frames para confirmar obstáculo (mais urgente)
COOLDOWN     = 60      # frames de espera após executar ação

# TTA — Test Time Augmentation
TTA_ENABLED  = True    # média sobre variantes do frame
TTA_FLIP     = True    # inclui flip horizontal
TTA_BRIGHT   = [0.8, 1.2]  # fatores de brilho extras

# Obstáculo: distância mínima (bbox ocupa % da largura do frame)
OBST_MIN_W_FRAC = 0.08  # bbox largura >= 8% do frame → relevante

# ================================================================
#  [2] CLASSES E AÇÕES
# ================================================================

# Estas classes devem corresponder ao modelo treinado
# Se usar yolov8n padrão COCO: mapeia classes COCO para ações
# Se usar modelo treinado: classes são as do dataset_v3

LABEL_TO_ACTION = {
    # Placas
    "Stop":       "STOP",
    "Esquerda":   "LEFT",
    "Direita":    "RIGHT",
    "SemRetorno": "STRAIGHT",
    "Verde":      "SPEED_UP",
    # Obstáculos
    "Cone":       "OBSTACLE",
    "Carro":      "OBSTACLE",
    "Pessoa":     "OBSTACLE",
    # Classes COCO (se usar yolov8n padrão)
    "stop sign":  "STOP",
    "person":     "OBSTACLE",
    "car":        "OBSTACLE",
    "truck":      "OBSTACLE",
    "bicycle":    "OBSTACLE",
    "motorcycle": "OBSTACLE",
    "traffic light": "SPEED_UP",
}

LABEL_NAMES = {
    "Stop":"Pare",           "Esquerda":"Vira Esq",
    "Direita":"Vira Dir",    "SemRetorno":"Sem Retorno",
    "Verde":"Semáforo Verde","Cone":"Cone",
    "Carro":"Carro",         "Pessoa":"Pessoa",
    "stop sign":"Placa PARE","person":"Pessoa",
    "car":"Carro",           "truck":"Caminhão",
    "traffic light":"Semáforo",
}

OBSTACLE_CLASSES = {
    "Cone","Carro","Pessoa","person","car","truck","bicycle","motorcycle"
}

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

CORES = {
    "STOP":(50,50,220),      "OBSTACLE":(0,0,200),
    "YIELD":(0,200,220),     "LEFT":(220,120,0),
    "RIGHT":(0,120,220),     "STRAIGHT":(50,220,50),
    "SLOW_DOWN":(0,180,255), "SPEED_UP":(0,200,100),
}

# ================================================================
#  [3] THRESHOLDS ADAPTATIVOS
# ================================================================

_THRESHOLDS: dict[str, float] = {}

def carregar_thresholds():
    global _THRESHOLDS
    if os.path.isfile(THRESH_PATH):
        with open(THRESH_PATH) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) == 2:
                    try: _THRESHOLDS[p[0]] = float(p[1])
                    except: pass
        print(f"[THRESH] {_THRESHOLDS}", flush=True)
    else:
        print("[THRESH] sign_thresholds.txt não encontrado — usando padrão 0.45",
              flush=True)

def thresh(label: str) -> float:
    return _THRESHOLDS.get(label, CONF_DEFAULT)

# ================================================================
#  [4] ESTADO GLOBAL
# ================================================================

CMD = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)

_nav = dict(
    cooldown    = 0,
    ultimo      = None,
    acao_label  = None,
    acao_t      = 0.0,
    acao_dur    = 0.0,
)

# Filtro temporal por track_id
# _temporal[track_id] = deque de labels dos últimos N frames
_temporal: dict[int, collections.deque] = {}
_DEQUE_LEN = max(CONFIRM_N, CONFIRM_OBS) + 2

# ================================================================
#  [5] SERIAL
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
        try: ser.write((j + "\n").encode())
        except: pass

# ================================================================
#  [6] MODELO YOLO — ONNX Runtime
# ================================================================

class YOLODetector:
    """
    YOLOv8n via ONNX Runtime (CPU).
    Suporta ByteTrack quando usado via ultralytics.YOLO.track().
    """

    def __init__(self):
        self.model_onnx  = None   # onnxruntime session
        self.model_ultra = None   # ultralytics YOLO (tracking)
        self.class_names = []
        self.using_onnx  = False

    def carregar(self):
        """Tenta ONNX primeiro, fallback para ultralytics .pt"""
        # Tentativa 1: ONNX Runtime direto
        if os.path.isfile(ONNX_PATH):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 4
                opts.intra_op_num_threads = 4
                self.model_onnx = ort.InferenceSession(
                    ONNX_PATH,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                # Lê nomes das classes do arquivo auxiliar
                names_path = ONNX_PATH.replace(".onnx", "_classes.txt")
                if os.path.isfile(names_path):
                    with open(names_path) as f:
                        self.class_names = [l.strip() for l in f if l.strip()]
                self.using_onnx = True
                print(f"[YOLO] ONNX carregado: {ONNX_PATH}", flush=True)
                print(f"[YOLO] Classes: {self.class_names}", flush=True)
                return True
            except Exception as e:
                print(f"[YOLO] ONNX falhou ({e}) — tentando ultralytics",
                      flush=True)

        # Tentativa 2: ultralytics (com tracking nativo)
        try:
            from ultralytics import YOLO
            pt = PT_PATH if os.path.isfile(PT_PATH) else "yolov8n.pt"
            self.model_ultra = YOLO(pt)
            self.class_names = list(self.model_ultra.names.values())
            self.using_onnx  = False
            print(f"[YOLO] ultralytics carregado: {pt}", flush=True)
            print(f"[YOLO] {len(self.class_names)} classes", flush=True)
            return True
        except ImportError:
            print("[YOLO] ultralytics não instalado.", flush=True)
            print("       Execute: pip install ultralytics", flush=True)
            return False

    def detectar(self, frame: np.ndarray) -> list[dict]:
        """
        Retorna lista de dicts:
          {label, conf, bbox:(x1,y1,x2,y2), track_id}
        """
        if self.using_onnx and self.model_onnx:
            return self._detectar_onnx(frame)
        elif self.model_ultra:
            return self._detectar_ultra(frame)
        return []

    # ── ONNX Runtime ─────────────────────────────────────────────
    def _pre_onnx(self, frame):
        img  = cv2.resize(frame, (IMGSZ, IMGSZ))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(img.transpose(2,0,1), 0)  # NCHW

    def _detectar_onnx(self, frame) -> list[dict]:
        h, w = frame.shape[:2]
        inp  = self._pre_onnx(frame)
        out  = self.model_onnx.run(None, {
            self.model_onnx.get_inputs()[0].name: inp
        })[0]  # (1, 84, 8400) for YOLOv8

        # Decode YOLOv8 output
        out = out[0].T  # (8400, 84)
        boxes  = out[:, :4]
        scores = out[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs     = scores[np.arange(len(scores)), class_ids]

        # Scale boxes to frame size
        cx, cy, bw, bh = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = ((cx - bw/2) / IMGSZ * w).astype(int)
        y1 = ((cy - bh/2) / IMGSZ * h).astype(int)
        x2 = ((cx + bw/2) / IMGSZ * w).astype(int)
        y2 = ((cy + bh/2) / IMGSZ * h).astype(int)

        results = []
        for i in range(len(confs)):
            cid  = int(class_ids[i])
            conf = float(confs[i])
            lbl  = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            t    = thresh(lbl)
            if conf < t: continue
            # NMS handled externally — basic area filter
            bx1,by1,bx2,by2 = int(x1[i]),int(y1[i]),int(x2[i]),int(y2[i])
            if bx2<=bx1 or by2<=by1: continue
            results.append(dict(
                label=lbl, conf=conf,
                bbox=(max(0,bx1),max(0,by1),min(w,bx2),min(h,by2)),
                track_id=-1  # ONNX sem tracker
            ))

        # Apply NMS manually
        return _nms(results, IOU_THRESH)

    # ── Ultralytics com ByteTrack ─────────────────────────────────
    def _detectar_ultra(self, frame) -> list[dict]:
        results = self.model_ultra.track(
            frame,
            tracker   = "bytetrack.yaml",
            persist   = True,
            conf      = CONF_DEFAULT,
            iou       = IOU_THRESH,
            imgsz     = IMGSZ,
            verbose   = False,
        )
        dets = []
        if not results or results[0].boxes is None:
            return dets
        boxes = results[0].boxes
        for box in boxes:
            cid      = int(box.cls[0])
            conf     = float(box.conf[0])
            lbl      = self.model_ultra.names[cid]
            t        = thresh(lbl)
            if conf < t: continue
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            tid      = int(box.id[0]) if box.id is not None else -1
            dets.append(dict(
                label=lbl, conf=conf,
                bbox=(int(x1),int(y1),int(x2),int(y2)),
                track_id=tid
            ))
        return dets

    # ── TTA — Test Time Augmentation ─────────────────────────────
    def detectar_tta(self, frame: np.ndarray) -> list[dict]:
        """
        Roda inferência em múltiplas variantes do frame e agrega.
        Aumenta robustez em condições de iluminação variável.
        """
        if not TTA_ENABLED:
            return self.detectar(frame)

        variants = [frame]
        if TTA_FLIP:
            variants.append(cv2.flip(frame, 1))
        for f in TTA_BRIGHT:
            v = np.clip(frame.astype(np.float32) * f, 0, 255).astype(np.uint8)
            variants.append(v)

        all_dets: list[dict] = []
        for i, v in enumerate(variants):
            dets = self.detectar(v)
            if i == 1 and TTA_FLIP:
                # Espelha bboxes de volta
                w = frame.shape[1]
                for d in dets:
                    x1,y1,x2,y2 = d["bbox"]
                    d["bbox"] = (w-x2, y1, w-x1, y2)
            all_dets.extend(dets)

        # Agrega: para cada grupo de bboxes sobrepostas, usa conf média
        return _nms_merge(all_dets, IOU_THRESH)


# ── Instância global ─────────────────────────────────────────────
_yolo = YOLODetector()


# ================================================================
#  [7] NMS HELPERS
# ================================================================

def _iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2<=ix1 or iy2<=iy1: return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    ua    = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(ua, 1e-6)

def _nms(dets: list[dict], iou_thr: float) -> list[dict]:
    if not dets: return []
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    used = [False] * len(dets)
    for i, d in enumerate(dets):
        if used[i]: continue
        keep.append(d)
        for j in range(i+1, len(dets)):
            if not used[j] and _iou(d["bbox"], dets[j]["bbox"]) > iou_thr:
                used[j] = True
    return keep

def _nms_merge(dets: list[dict], iou_thr: float) -> list[dict]:
    """NMS com média de confiança para TTA."""
    if not dets: return []
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    used = [False] * len(dets)
    for i, d in enumerate(dets):
        if used[i]: continue
        group  = [d]
        for j in range(i+1, len(dets)):
            if not used[j] and _iou(d["bbox"], dets[j]["bbox"]) > iou_thr:
                group.append(dets[j]); used[j] = True
        # Média de confiança do grupo
        avg_conf = float(np.mean([g["conf"] for g in group]))
        best = max(group, key=lambda g: g["conf"])
        best["conf"] = avg_conf
        keep.append(best)
    return keep


# ================================================================
#  [8] FILTRO TEMPORAL (ByteTrack ID → deque de labels)
# ================================================================

def atualizar_temporal(dets: list[dict], frame_id: int) -> list[dict]:
    """
    Para cada detecção com track_id, mantém histórico de labels.
    Só confirma se o label aparece N vezes consecutivas.
    Retorna lista de detecções confirmadas.
    """
    global _temporal

    # Marca IDs ativos neste frame
    ids_ativos = set()
    for d in dets:
        tid = d["track_id"]
        if tid < 0:
            # Sem tracker (ONNX direto): usa posição como proxy de ID
            tid = hash((d["label"], d["bbox"][0]//30, d["bbox"][1]//30)) % 10000
            d["track_id"] = tid
        ids_ativos.add(tid)
        if tid not in _temporal:
            _temporal[tid] = collections.deque(maxlen=_DEQUE_LEN)
        _temporal[tid].append(d["label"])

    # Remove IDs que sumiram há mais de 2x DEQUE_LEN frames
    stale = [k for k in list(_temporal) if k not in ids_ativos]
    for k in stale[:10]:  # limita remoções por frame
        del _temporal[k]

    # Verifica quais passam no filtro temporal
    confirmadas = []
    for d in dets:
        tid   = d["track_id"]
        hist  = list(_temporal.get(tid, []))
        lbl   = d["label"]
        n_req = CONFIRM_OBS if lbl in OBSTACLE_CLASSES else CONFIRM_N

        if len(hist) >= n_req:
            recente = hist[-n_req:]
            if recente.count(lbl) >= n_req:
                d["confirmado"] = True
            else:
                d["confirmado"] = False
        else:
            d["confirmado"] = False
        confirmadas.append(d)

    return confirmadas


# ================================================================
#  [9] NAVEGAÇÃO — EXECUTOR + TICK
# ================================================================

def executar(label: str, ser):
    if _nav["cooldown"] > 0:
        return
    acao = LABEL_TO_ACTION.get(label)
    if not acao or acao not in ACOES:
        return
    a = ACOES[acao]
    CMD.update(mot=a["mot"], srv=a["srv"], buz=a["buz"],
               led=a["led"], brk=a["brk"], dir=a["dir"],
               spd=0 if a["mot"]==0 else (1 if a["mot"]<50 else 2))
    enviar(ser)
    _nav.update(acao_label=acao, acao_t=time.monotonic(),
                acao_dur=a["dur"], cooldown=COOLDOWN, ultimo=label)
    print(f"[NAV] ▶ {acao} ← '{label}'", flush=True)

def tick(ser) -> bool:
    lbl = _nav["acao_label"]
    if lbl is None: return False
    if lbl == "OBSTACLE": return True
    if time.monotonic() - _nav["acao_t"] >= _nav["acao_dur"]:
        CMD.update(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
        enviar(ser)
        print(f"[NAV] ✅ {lbl}", flush=True)
        _nav["acao_label"] = None
        return False
    return True

def decrementar_cooldown():
    if _nav["cooldown"] > 0:
        _nav["cooldown"] -= 1


# ================================================================
#  [10] DECISÃO — prioridade entre detecções confirmadas
# ================================================================

def decidir(dets_confirmadas: list[dict], ser, frame_w: int):
    """
    Prioridade:
      1. OBSTACLE (qualquer obstáculo próximo → para imediato)
      2. STOP
      3. Outras placas (LEFT/RIGHT/STRAIGHT/SPEED_UP)
    """
    if _nav["cooldown"] > 0:
        return

    em_acao = tick(ser)
    if em_acao and _nav["acao_label"] != "OBSTACLE":
        return

    # Verifica se obstáculo ACTIVE sumiu
    if _nav["acao_label"] == "OBSTACLE":
        obstaculos = [d for d in dets_confirmadas
                      if LABEL_TO_ACTION.get(d["label"]) == "OBSTACLE"]
        if not obstaculos:
            CMD.update(mot=0,srv=127,buz=0,led=0,brk=0,dir=0,spd=0)
            enviar(ser)
            print("[NAV] ✅ Obstáculo removido", flush=True)
            _nav["acao_label"] = None
        return

    # Filtra só confirmadas
    conf_list = [d for d in dets_confirmadas if d.get("confirmado")]
    if not conf_list:
        return

    # Ordena: obstáculos primeiro, depois por confiança
    def prioridade(d):
        acao = LABEL_TO_ACTION.get(d["label"], "")
        order = {"OBSTACLE": 0, "STOP": 1}.get(acao, 2)
        return (order, -d["conf"])

    conf_list.sort(key=prioridade)
    melhor = conf_list[0]

    # Obstáculo: valida tamanho da bbox (evita detecções distantes)
    acao = LABEL_TO_ACTION.get(melhor["label"])
    if acao == "OBSTACLE":
        x1,y1,x2,y2 = melhor["bbox"]
        bw = x2 - x1
        if bw < frame_w * OBST_MIN_W_FRAC:
            return  # muito longe

    executar(melhor["label"], ser)


# ================================================================
#  [11] VISUALIZAÇÃO
# ================================================================

def desenhar(frame, dets, fps):
    out  = frame.copy()
    h, w = out.shape[:2]

    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        lbl    = d["label"]
        conf   = d["conf"]
        tid    = d["track_id"]
        conf_n = d.get("confirmado", False)
        acao   = LABEL_TO_ACTION.get(lbl, "")
        cor    = CORES.get(acao, (160,160,160))

        thick  = 3 if conf_n else 1
        cv2.rectangle(out, (x1,y1), (x2,y2), cor, thick)

        # Label com nome, confiança e track_id
        nome = LABEL_NAMES.get(lbl, lbl)
        txt  = f"#{tid} {nome} {conf:.0%}"
        if conf_n:
            txt += " ✓"
        tw   = len(txt) * 7
        cv2.rectangle(out, (x1, y1-18), (x1+tw, y1), cor, -1)
        cv2.putText(out, txt, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1)

        # Barra de confirmação temporal
        tid_ = d["track_id"]
        hist = list(_temporal.get(tid_, []))
        n_req = CONFIRM_OBS if lbl in OBSTACLE_CLASSES else CONFIRM_N
        cnt   = min(hist.count(lbl), n_req) if hist else 0
        prog  = int((x2-x1) * cnt / max(n_req,1))
        cv2.rectangle(out, (x1,y2+2), (x2,y2+6), (40,40,40), -1)
        cv2.rectangle(out, (x1,y2+2), (x1+prog,y2+6), cor, -1)

    # Painel lateral
    pw  = 210
    pan = np.zeros((h, pw, 3), np.uint8); pan[:] = (18,18,18)
    cv2.rectangle(pan, (0,0), (pw-1,h-1), (45,45,45), 1)

    def t(s, l, cor=(190,190,190), sc=0.33):
        cv2.putText(pan, s, (5, 14+l*16),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, cor, 1)

    t(f"FPS:{fps}",  0, (255,255,255), 0.40)
    t(f"Dets:{len(dets)}", 1, (200,200,200), 0.36)
    t(f"Conf:{len([d for d in dets if d.get('confirmado')])}",
      2, (100,220,100), 0.36)

    acao = _nav["acao_label"]
    if acao:
        ca = CORES.get(acao, (200,200,200))
        t(f"EXEC:{acao}", 4, ca, 0.40)
        if acao != "OBSTACLE":
            dt = time.monotonic() - _nav["acao_t"]
            t(f" {dt:.1f}s/{_nav['acao_dur']}s", 5, ca)
    else:
        t("livre", 4, (100,200,100))
        t(f"cd:{_nav['cooldown']}", 5, (80,80,80))

    t("─ ÚLTIMO ─", 7, (60,60,60))
    ult = _nav["ultimo"] or "—"
    t(f" {ult}", 8, CORES.get(LABEL_TO_ACTION.get(ult,""), (160,160,160)))

    t("─ TTA ─", 10, (60,60,60))
    t(f" {'ON' if TTA_ENABLED else 'OFF'}", 11,
      (100,220,100) if TTA_ENABLED else (150,150,150))

    t("─ THRESH ─", 13, (60,60,60))
    for i, (k,v) in enumerate(list(_THRESHOLDS.items())[:5]):
        t(f" {k[:8]}:{v:.0%}", 14+i, (120,120,180), 0.28)

    return np.hstack([out, pan])


# ================================================================
#  [12] MODO --export (YOLOv8n pt → ONNX)
# ================================================================

def exportar_onnx():
    print("[EXPORT] Exportando YOLOv8n → ONNX...")
    try:
        from ultralytics import YOLO
        pt = PT_PATH if os.path.isfile(PT_PATH) else "yolov8n.pt"
        model = YOLO(pt)
        path  = model.export(format="onnx", imgsz=IMGSZ, simplify=True)
        print(f"[EXPORT] Salvo: {path}")

        # Salva lista de classes
        names_path = ONNX_PATH.replace(".onnx", "_classes.txt")
        with open(names_path, "w") as f:
            for name in model.names.values():
                f.write(name + "\n")
        print(f"[EXPORT] Classes: {names_path}")
        print(f"\n  Renomeie {path} para {ONNX_PATH}")
    except ImportError:
        print("[EXPORT] ultralytics não instalado.")
        print("         pip install ultralytics")
    except Exception as e:
        print(f"[EXPORT] Erro: {e}")


# ================================================================
#  [13] MODO --test
# ================================================================

def modo_teste(caminhos: list[str]):
    if not _yolo.carregar():
        print("[ERRO] Nenhum modelo disponível."); sys.exit(1)

    for caminho in caminhos:
        img = cv2.imread(caminho)
        if img is None:
            print(f"[ERRO] {caminho}"); continue

        h, w = img.shape[:2]
        print(f"\n{'='*60}")
        print(f"  {os.path.basename(caminho)}  ({w}×{h}px)")

        # TTA
        dets = _yolo.detectar_tta(img)
        dets = atualizar_temporal(dets, 0)

        print(f"  {len(dets)} detecção(ões):")
        print(f"  {'Label':<14} {'Conf':>6}  {'Ação':<12}  BBox")
        print(f"  {'-'*55}")
        for d in sorted(dets, key=lambda x: -x["conf"]):
            lbl  = d["label"]
            acao = LABEL_TO_ACTION.get(lbl, "—")
            print(f"  {lbl:<14} {d['conf']*100:5.1f}%  {acao:<12}  {d['bbox']}")

        if not dets:
            print("  Nenhuma detecção acima do threshold.")

        print(f"{'='*60}\n")

        vis = desenhar(img, dets, fps=0)
        cv2.imshow(f"TEST — {os.path.basename(caminho)}", vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ================================================================
#  [14] LOOP PRINCIPAL
# ================================================================

def main(usar_camera: bool):
    if not _yolo.carregar():
        print("[ERRO] Modelo não carregado."); sys.exit(1)

    src = CAM_IDX if usar_camera else VIDEO
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[ERRO] Fonte não abriu: {src}"); sys.exit(1)

    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    ser    = conectar_serial()
    fps_t  = time.monotonic(); fps_n = fps = 0
    frame_id = 0

    print(f"[OK] {'Câmera' if usar_camera else 'Vídeo'}: {src}", flush=True)
    print(f"[OK] TTA: {'ON' if TTA_ENABLED else 'OFF'}", flush=True)
    print("[OK] Q para sair\n", flush=True)

    while True:
        ret, fb = cap.read()
        if not ret:
            if not usar_camera:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break

        frame    = fb if PROP == 1 else cv2.resize(
            fb, (round(fb.shape[1]/PROP), round(fb.shape[0]/PROP)))
        frame_id += 1
        h, w     = frame.shape[:2]

        decrementar_cooldown()

        # Detecção com TTA
        dets = _yolo.detectar_tta(frame)

        # Filtro temporal com ByteTrack IDs
        dets = atualizar_temporal(dets, frame_id)

        # Decisão e envio ao Arduino
        decidir(dets, ser, w)

        # Visualização
        vis = desenhar(frame, dets, fps)
        cv2.imshow("SIGN_DETECTOR v4.0", vis)

        fps_n += 1
        if time.monotonic() - fps_t >= 1.0:
            fps = fps_n; fps_n = 0; fps_t = time.monotonic()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CMD.update(mot=0, srv=127, buz=0, led=0, brk=1, dir=0, spd=0)
    enviar(ser)
    if ser: ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)


# ================================================================
#  ENTRY POINT
# ================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="SIGN_DETECTOR v4.0 — YOLOv8n + ByteTrack + ONNX",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--cam",    action="store_true",
                    help="Câmera USB ao vivo")
    ap.add_argument("--test",   nargs="+", metavar="FOTO",
                    help="Testa imagens")
    ap.add_argument("--export", action="store_true",
                    help="Exporta YOLOv8n → ONNX (precisa ultralytics)")
    ap.add_argument("--no-tta", action="store_true",
                    help="Desativa TTA (mais rápido, menos preciso)")
    args = ap.parse_args()

    if args.no_tta:
        TTA_ENABLED = False

    carregar_thresholds()

    if args.export:
        exportar_onnx()
    elif args.test:
        modo_teste(args.test)
    else:
        main(usar_camera=args.cam)