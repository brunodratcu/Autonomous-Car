"""
================================================================
  CARRO AUTÔNOMO v4.0 — PIPELINE OTIMIZADO
  ─────────────────────────────────────────────────────────────
  Frame BGR
    ↓ Resize 640px
    ↓ CLAHE (LAB-L)          contraste adaptativo
    ↓ Bilateral Filter        deruído com preservação de bordas
    ↓ Sharpen leve            realce de bordas para YOLO
    ↓ YOLOv8n ONNX            localização principal
    ↓ [fallback se vazio]     Contornos + Adaptive Threshold
    ↓ Tracker (ByteTrack-lite) associação cross-frame
    ↓ Crop → CNN TFLite       MobileNetV3Small 96×96
    ↓ Temporal Vote 10 frames ≥60% de consenso
    ↓ Serial JSON             Arduino Portenta H7

  Classes (8): Stop · Esquerda · Direita · SemRetorno
               Verde · Cone · Carro · Pessoa

  USO:
    python carro-autonomo.py           ← vídeo
    python carro-autonomo.py --cam     ← webcam
    python carro-autonomo.py --cal     ← calibrar preprocessing
    python carro-autonomo.py --debug   ← visualização intermediária
    python carro-autonomo.py --cam --auto --fast
"""

import cv2
import numpy as np
import serial, serial.tools.list_ports
import time, sys, os, json
from collections import deque, Counter

# ================================================================
#  [1] CONFIG
# ================================================================

VIDEO       = "./videoplayback.mp4"
CAM_IDX     = 0
IMG_W       = 640        # largura alvo antes de entrar no YOLO

SERIAL_PORT = "COM3"
BAUD        = 115200

YOLO_MODEL  = "./models/sign_detector.onnx"
CNN_MODEL   = "./models/sign_classifier.tflite"
OOD_FILE    = "./models/ood_thresholds.json"

CNN_SIZE    = 96
OOD_DEFAULT = 0.55
YOLO_CONF   = 0.35
YOLO_NMS    = 0.45
YOLO_SIZE   = 640

CLASSES = ["Stop","Esquerda","Direita","SemRetorno",
           "Verde","Cone","Carro","Pessoa"]
NUM_CLASSES      = len(CLASSES)
OBSTACLE_CLASSES = {"Cone","Carro","Pessoa"}

# ── COCO (modelo pré-treinado, 80 classes) ──────────────────────
# Se o ONNX carregado tiver 80 classes, o sistema entra em modo
# COCO automaticamente: detecção direta sem CNN, mapeando as
# classes COCO relevantes para as classes do projeto.
COCO_MAP = {
    11: "Stop",     # stop sign
    0:  "Pessoa",   # person
    2:  "Carro",    # car
    7:  "Carro",    # truck
    5:  "Carro",    # bus
    3:  "Carro",    # motorcycle
    9:  "Semaforo", # traffic light — cor analisada dentro da bbox
}

# Tracker
TRACK_IOU_HIGH   = 0.30   # IoU mínimo — associação de alta confiança
TRACK_IOU_LOW    = 0.15   # IoU mínimo — segunda passagem (dets de baixa conf)
TRACK_MAX_AGE    = 10     # frames sem match antes de remover
CONF_HIGH_THR    = 0.45   # threshold de "alta confiança" YOLO

# Temporal voting
VOTE_BUFFER      = 10     # frames no buffer por track
VOTE_MIN_DETS    = 5      # mínimo de amostras no buffer para votar
VOTE_FRAC        = 0.60   # consenso mínimo (60% = 6/10)

# Execução
COOLDOWN_F       = 60
AREA_MIN_EXEC    = 800    # px² mínimo para executar ação

# ROI vertical (fração do frame)
ROI_Y0 = 0.05
ROI_Y1 = 0.92

# Fallback contornos
AREA_CNT_MIN = 500
AREA_CNT_MAX = 55_000

# ================================================================
#  [2] KERNELS  (do código de referência, ajustados para YOLO)
# ================================================================

# Sharpen leve: realça bordas sem saturar pixel values
# Kernel Laplaciano ajustado para não distorcer a entrada do YOLO
K_SHARP = np.array([[ 0, -1,  0],
                    [-1,  5, -1],
                    [ 0, -1,  0]], dtype=np.float32)

K3 = np.ones((3,3), np.uint8)
K5 = np.ones((5,5), np.uint8)

# CLAHE criado uma vez (reutilizado todo frame)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ================================================================
#  [3] MAPEAMENTO E AÇÕES
# ================================================================

CLASS_TO_ACTION = {
    "Stop":"STOP","SemRetorno":"STOP",
    "Esquerda":"LEFT","Direita":"RIGHT",
    "Verde":"STRAIGHT",
    "Cone":"OBSTACLE","Carro":"OBSTACLE","Pessoa":"OBSTACLE",
}
ACOES = {
    "STOP":     dict(mot=0,  srv=127, buz=1, led=1, brk=1, dir=0, dur=2.5),
    "OBSTACLE": dict(mot=0,  srv=127, buz=0, led=1, brk=1, dir=0, dur=0.0),
    "LEFT":     dict(mot=40, srv=50,  buz=0, led=0, brk=0, dir=1, dur=1.4),
    "RIGHT":    dict(mot=40, srv=204, buz=0, led=0, brk=0, dir=2, dur=1.4),
    "STRAIGHT": dict(mot=62, srv=127, buz=0, led=0, brk=0, dir=3, dur=1.0),
}
COR_CLASSE = {
    "Stop":(50,50,220),"SemRetorno":(20,20,180),
    "Esquerda":(220,120,0),"Direita":(0,120,220),
    "Verde":(50,220,50),"Cone":(0,165,255),
    "Carro":(200,0,200),"Pessoa":(0,0,255),
    "Semaforo":(0,220,220),
}

# ================================================================
#  [3b] SEMÁFORO — análise de cor dentro da bbox
#  O COCO detecta "traffic light" mas não diz a cor.
#  Analisamos a fração de pixels verde/vermelho no crop.
# ================================================================

def cor_semaforo(crop_bgr: np.ndarray) -> str | None:
    """Retorna 'verde', 'vermelho', 'amarelo' ou None."""
    if crop_bgr is None or crop_bgr.size == 0: return None
    hsv = cv2.cvtColor(cv2.resize(crop_bgr,(40,80)), cv2.COLOR_BGR2HSV)
    total = hsv.shape[0]*hsv.shape[1]

    # Máscaras de cor (S e V altos = luz acesa)
    verde    = cv2.inRange(hsv, (40, 80,120), (90,255,255))
    vermelho = cv2.inRange(hsv, (0,  80,120), (10,255,255)) | \
               cv2.inRange(hsv, (170,80,120), (180,255,255))
    amarelo  = cv2.inRange(hsv, (18, 80,120), (35,255,255))

    fr_v = verde.sum()/255/total
    fr_r = vermelho.sum()/255/total
    fr_a = amarelo.sum()/255/total

    melhor = max([("verde",fr_v),("vermelho",fr_r),("amarelo",fr_a)],
                 key=lambda x: x[1])
    return melhor[0] if melhor[1] > 0.04 else None

# ================================================================
#  [3c] MÁQUINA DE ESTADOS DA MISSÃO
#
#  AGUARDANDO  → carro parado; espera sinal de partida:
#                 · semáforo verde detectado, OU
#                 · tecla 'g', OU
#                 · Arduino envia {"btn":"start"}
#  RODANDO     → pipeline normal comanda o carro
#  PARADO_SEM  → semáforo vermelho à frente; espera ficar verde
#  ENTREGANDO  → chegou ao destino; para e sinaliza (buzzer)
#  FINALIZADO  → missão completa
#
#  Destino A/B/C: teclas '1','2','3' antes da partida.
#  A rota é uma sequência de decisões: a cada placa de direção
#  confirmada, a missão registra o progresso.
# ================================================================

ROTAS = {
    # destino: sequência esperada de comandos até o ponto de entrega
    "A": ["STRAIGHT", "LEFT"],
    "B": ["STRAIGHT", "STRAIGHT"],
    "C": ["STRAIGHT", "RIGHT"],
}

class Missao:
    def __init__(self):
        self.estado    = "AGUARDANDO"
        self.destino   = "A"
        self.progresso = 0           # quantos passos da rota já executados
        self.t_estado  = time.monotonic()

    def rota(self):
        return ROTAS.get(self.destino, [])

    def set_estado(self, novo):
        if novo != self.estado:
            print(f"[MISSAO] {self.estado} → {novo}", flush=True)
            self.estado   = novo
            self.t_estado = time.monotonic()

    def registrar_comando(self, acao):
        """Chamado quando uma ação de movimento é executada."""
        rota = self.rota()
        if self.progresso < len(rota) and acao == rota[self.progresso]:
            self.progresso += 1
            print(f"[MISSAO] Progresso {self.progresso}/{len(rota)} "
                  f"rumo a {self.destino}", flush=True)
            if self.progresso >= len(rota):
                self.set_estado("ENTREGANDO")

MISSAO = Missao()
COR_ACAO = {
    "STOP":(50,50,220),"OBSTACLE":(0,0,200),
    "LEFT":(220,120,0),"RIGHT":(0,120,220),"STRAIGHT":(50,220,50),
}

CMD  = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
_nav = dict(cooldown=0, ultimo=None, acao_label=None,
            acao_t=0.0, acao_dur=0.0)
_ser = None

# ================================================================
#  [4] SERIAL
# ================================================================

def conectar_serial():
    kws = ["arduino","ch340","cp210","uart","portenta"]
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
        print(f"[SER] Simulação — {e}", flush=True); return None

_seq = dict(n=0, pendente=None, t_envio=0.0)

def enviar(cmd, ser):
    """Envia comando com número de sequência. Arduino responde {"ack":N}."""
    _seq["n"] += 1
    spd = 0 if cmd["mot"]==0 else (1 if cmd["mot"]<50 else 2)
    j = (f'{{"seq":{_seq["n"]},"mot":{cmd["mot"]},"srv":{cmd["srv"]},'
         f'"buz":{cmd["buz"]},"led":{cmd["led"]},'
         f'"brk":{cmd["brk"]},"dir":{cmd["dir"]},'
         f'"spd":{spd}}}')
    print(f"[CMD] {j}", flush=True)
    if ser:
        try:
            ser.write((j+"\n").encode())
            _seq["pendente"] = _seq["n"]
            _seq["t_envio"]  = time.monotonic()
        except Exception: pass


def ler_serial(ser) -> list[dict]:
    """
    Lê mensagens do Arduino (não-bloqueante).
    Retorna lista de dicts. Trata ACKs e retransmite se necessário.
    Mensagens possíveis do Arduino:
      {"ack": N}          confirmação de comando
      {"btn": "start"}    botão físico de partida pressionado
      {"btn": "D"}        botão de entrega pressionado
      {"dist": 25.4}      leitura do sensor TOF (cm)
    """
    msgs = []
    if not ser: return msgs
    try:
        while ser.in_waiting:
            linha = ser.readline().decode(errors="ignore").strip()
            if not linha: continue
            try:
                m = json.loads(linha)
                msgs.append(m)
                if "ack" in m and m["ack"] == _seq["pendente"]:
                    _seq["pendente"] = None
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # Retransmissão: comando sem ACK após 200ms
    if (_seq["pendente"] is not None
            and time.monotonic() - _seq["t_envio"] > 0.2):
        print(f"[SER] Retransmitindo seq={_seq['pendente']}", flush=True)
        _seq["pendente"] = None   # evita loop infinito; reenvia estado atual
        enviar(CMD, ser)

    return msgs

# ================================================================
#  [5] PREPROCESSING PRINCIPAL
#
#  Aplicado ao frame ANTES do YOLO.
#  OBJETIVO: melhorar qualidade visual sem destruir informação.
#
#  Etapas:
#    1. Resize → largura fixa (640px) — dimensão ideal para YOLO
#    2. CLAHE no canal L (LAB) → equaliza contraste localmente
#       Resolve: câmera com auto-exposição, sombras, overexposure
#    3. Bilateral filter → denoising com preservação de bordas
#       d=5 (leve, CPU-friendly); sigma=40 (moderado)
#    4. Sharpen leve → realça bordas das placas
#       Kernel: [[0,-1,0],[-1,5,-1],[0,-1,0]]
#
#  NÃO usa threshold aqui — preserva informação visual para YOLO.
#  Threshold/morphology ficam no fallback de contornos.
# ================================================================

def preprocessar(frame_bgr: np.ndarray) -> np.ndarray:
    """Retorna frame melhorado (BGR) pronto para YOLO e CNN."""

    # 1. Resize
    h0, w0 = frame_bgr.shape[:2]
    if w0 != IMG_W:
        scale = IMG_W / w0
        frame_bgr = cv2.resize(frame_bgr, (IMG_W, int(h0*scale)),
                               interpolation=cv2.INTER_LINEAR)

    # 2. CLAHE no canal L
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = _CLAHE.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 3. Bilateral filter — denoising leve (CPU-friendly)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=40, sigmaSpace=40)

    # 4. Sharpen leve
    sharpened = cv2.filter2D(denoised, -1, K_SHARP)

    return sharpened


# ================================================================
#  [6] FALLBACK: DETECÇÃO POR CONTORNOS
#
#  Usado SOMENTE quando YOLO não retorna nenhuma detecção.
#  Aqui sim usamos adaptive threshold + morphological ops,
#  porque o objetivo é extrair bordas para contornos —
#  não precisamos preservar informação visual.
# ================================================================

def detectar_contornos_fallback(frame_enhanced: np.ndarray) -> list:
    """
    Pipeline de fallback:
      Grayscale → medianBlur → Adaptive Threshold → Close → Open → Contornos
    Retorna lista de dicts no mesmo formato do YOLO.
    """
    h, w = frame_enhanced.shape[:2]

    gray = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thr  = cv2.adaptiveThreshold(blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=15, C=4)
    thr  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, K5)
    thr  = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  K3)

    # Aplica ROI
    y0 = int(h * ROI_Y0); y1 = int(h * ROI_Y1)
    thr[:y0,:] = 0; thr[y1:,:] = 0

    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets   = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (AREA_CNT_MIN < area < AREA_CNT_MAX): continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        prop   = bw / max(bh, 1)
        if not (0.30 < prop < 2.50): continue
        solidez = area / max(bw*bh, 1)
        if solidez < 0.18: continue
        dets.append({
            "bbox":       (x, y, x+bw, y+bh),
            "class_name": "?",
            "class_id":   -1,
            "conf":       float(solidez),
        })
    dets.sort(key=lambda d:(d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]),
              reverse=True)
    return dets[:6]

# ================================================================
#  [7] YOLO DETECTOR — YOLOv8n ONNX Runtime
# ================================================================

class YOLODetector:

    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(
            path, opts,
            providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.in_n  = self.sess.get_inputs()[0].name
        self.out_n = self.sess.get_outputs()[0].name

        # Lê o tamanho de entrada REAL exigido pelo modelo (fixo no export).
        # Nunca usar YOLO_SIZE cego aqui — se o ONNX foi exportado com
        # dynamic=False, qualquer tamanho diferente quebra o InferenceSession.
        in_shape = self.sess.get_inputs()[0].shape   # [1,3,H,W]
        h_model, w_model = in_shape[2], in_shape[3]
        if isinstance(h_model, int) and isinstance(w_model, int):
            self.input_size = h_model   # assume quadrado (padrão YOLO)
        else:
            self.input_size = YOLO_SIZE  # modelo dinâmico → usa o configurado

        # Detecta automaticamente: COCO (80 classes) ou customizado (8)
        out_shape = self.sess.get_outputs()[0].shape   # [1, 4+NC, 8400]
        n_cls = out_shape[1] - 4 if isinstance(out_shape[1], int) else NUM_CLASSES
        self.coco_mode = (n_cls == 80)
        modo = "COCO (pré-treinado)" if self.coco_mode else f"CUSTOM ({n_cls} classes)"
        print(f"[YOLO] {path} | {self.sess.get_providers()[0]} | {modo} | "
              f"input={self.input_size}px", flush=True)

    @staticmethod
    def _letterbox(img, size=640):
        h, w = img.shape[:2]
        sc = size/max(h,w); nh,nw = int(h*sc),int(w*sc)
        canvas = np.full((size,size,3),114,np.uint8)
        py,px  = (size-nh)//2,(size-nw)//2
        canvas[py:py+nh, px:px+nw] = cv2.resize(img,(nw,nh))
        return canvas, sc, px, py

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        if not len(boxes): return []
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep=[]
        while len(order):
            i=order[0]; keep.append(i)
            if len(order)==1: break
            xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
            inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
            iou=inter/(areas[i]+areas[order[1:]]-inter+1e-9)
            order=order[1:][iou<iou_thr]
        return keep

    def detectar(self, frame_enhanced: np.ndarray) -> list:
        h0,w0 = frame_enhanced.shape[:2]
        canvas,sc,px,py = self._letterbox(frame_enhanced, self.input_size)
        inp = canvas[:,:,::-1].astype(np.float32)/255.0
        inp = inp.transpose(2,0,1)[np.newaxis]
        raw = self.sess.run([self.out_n],{self.in_n:inp})[0]
        preds = raw[0].T                     # [8400, 4+NC]
        cls_sc = preds[:,4:]; max_c = cls_sc.max(axis=1)
        mask   = max_c >= YOLO_CONF
        if not mask.any(): return []
        preds,cls_sc,max_c = preds[mask],cls_sc[mask],max_c[mask]
        cx,cy,bw,bh = preds[:,0],preds[:,1],preds[:,2],preds[:,3]
        cls_ids = cls_sc.argmax(axis=1)
        bxs = np.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],axis=1)
        results = []
        for cid in np.unique(cls_ids):
            # Em modo COCO: só aceita classes mapeadas (stop sign, car, person...)
            if self.coco_mode:
                if int(cid) not in COCO_MAP:
                    continue
                cls_name = COCO_MAP[int(cid)]
            else:
                cls_name = CLASSES[int(cid)] if int(cid)<NUM_CLASSES else "?"
            idx  = np.where(cls_ids==cid)[0]
            keep = self._nms(bxs[idx], max_c[idx], YOLO_NMS)
            for k in keep:
                i   = idx[k]
                x1  = int(np.clip((bxs[i,0]-px)/sc, 0, w0-1))
                y1  = int(np.clip((bxs[i,1]-py)/sc, 0, h0-1))
                x2  = int(np.clip((bxs[i,2]-px)/sc, 0, w0-1))
                y2  = int(np.clip((bxs[i,3]-py)/sc, 0, h0-1))
                # Filtra ROI vertical
                if y1 < h0*ROI_Y0 or y2 > h0*ROI_Y1: continue
                results.append({"bbox":(x1,y1,x2,y2),"class_name":cls_name,
                                 "class_id":int(cid),"conf":float(max_c[i]),
                                 "coco":self.coco_mode})
        return results

# ================================================================
#  [8] CNN + PREPROCESSING DE CROP
# ================================================================

class CNNClassifier:
    def __init__(self, path):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        d = interp.get_input_details()[0]
        self._interp = interp
        self._in  = d["index"]
        self._out = interp.get_output_details()[0]["index"]
        self._q   = d["dtype"] == np.uint8
        print(f"[CNN] {path}", flush=True)

    def predict(self, img_96: np.ndarray) -> np.ndarray:
        inp = img_96[np.newaxis].astype(np.float32)
        if self._q: inp = (inp*255).astype(np.uint8)
        self._interp.set_tensor(self._in, inp)
        self._interp.invoke()
        out = self._interp.get_tensor(self._out)[0]
        return out.astype(np.float32)/255.0 if self._q else out


def crop_para_cnn(crop_bgr: np.ndarray, cls_hint: str) -> np.ndarray:
    """
    Preprocessing do crop antes da CNN (dual, por tipo de classe).
    Placa    → grayscale + adaptive threshold (shape/bordas)
    Obstáculo→ CLAHE + RGB normalizado (cor + silhueta)
    """
    img = cv2.resize(crop_bgr, (CNN_SIZE, CNN_SIZE))
    if cls_hint in OBSTACLE_CLASSES:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = _CLAHE.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    gray = cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(3,3),0)
    thr  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,2)
    return np.stack([thr,thr,thr],axis=-1).astype(np.float32)/255.0


class OODRejector:
    def __init__(self, path):
        self._t = {}
        if os.path.exists(path):
            with open(path) as f: self._t = json.load(f)
            print(f"[OOD] thresholds: {self._t}", flush=True)
        else:
            print(f"[OOD] usando default={OOD_DEFAULT}", flush=True)
    def aceitar(self, cls, score):
        return score >= self._t.get(cls, OOD_DEFAULT)

# ================================================================
#  [9] BYTETRACK-LITE
#
#  Inspirado no ByteTrack (Zhang et al. 2022) sem dependências
#  externas. Dois estágios de matching por IoU:
#
#  Estágio 1: Detecções de ALTA confiança (conf > CONF_HIGH_THR)
#             → associadas a tracks existentes por IoU ≥ TRACK_IOU_HIGH
#
#  Estágio 2: Detecções de BAIXA confiança (conf < CONF_HIGH_THR)
#             → associadas a tracks NÃO matched no estágio 1
#             → IoU threshold menor (TRACK_IOU_LOW)
#
#  Tracks não matchadas: incrementa missed, removidas após TRACK_MAX_AGE
# ================================================================

class Track:
    _nid = 0

    def __init__(self, bbox: tuple, hint: str, conf: float):
        self.id         = Track._nid; Track._nid += 1
        self.bbox       = bbox
        self.class_hint = hint
        self.conf       = conf
        self.buf: deque = deque(maxlen=VOTE_BUFFER)
        self.age        = 0
        self.missed     = 0
        self.state      = "tentative"  # tentative → confirmed → lost

    def atualizar(self, bbox, hint, conf, cnn_lbl, cnn_conf):
        self.bbox       = bbox
        self.class_hint = hint
        self.conf       = conf
        self.missed     = 0
        self.age       += 1
        if cnn_lbl: self.buf.append((cnn_lbl, cnn_conf))
        if self.age >= 3: self.state = "confirmed"

    def votar(self) -> tuple:
        """Majority vote. Retorna (label, conf_media) ou (None, 0)."""
        if len(self.buf) < VOTE_MIN_DETS: return None, 0.0
        cnt = Counter(lb for lb,_ in self.buf)
        top, n = cnt.most_common(1)[0]
        frac = n / len(self.buf)
        conf_m = np.mean([c for lb,c in self.buf if lb==top])
        return (top, float(conf_m)) if frac >= VOTE_FRAC else (None, 0.0)

    @property
    def area(self):
        x1,y1,x2,y2 = self.bbox; return max(0,(x2-x1)*(y2-y1))


class ByteTrackLite:

    def __init__(self):
        self.tracks: list[Track] = []

    @staticmethod
    def _iou(a, b) -> float:
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
        ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if not inter: return 0.0
        aA=(a[2]-a[0])*(a[3]-a[1]); aB=(b[2]-b[0])*(b[3]-b[1])
        return inter/(aA+aB-inter+1e-9)

    def _match_greedy(self, tracks_idx, det_idx, dets, iou_min):
        """Greedy matching por IoU decrescente."""
        pairs = []
        for ti in tracks_idx:
            for di in det_idx:
                iou = self._iou(self.tracks[ti].bbox, dets[di]["bbox"])
                if iou >= iou_min:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        mt, md = set(), set()
        matched = []
        for iou, ti, di in pairs:
            if ti in mt or di in md: continue
            mt.add(ti); md.add(di)
            matched.append((ti, di))
        unmatched_t = [i for i in tracks_idx if i not in mt]
        unmatched_d = [i for i in det_idx  if i not in md]
        return matched, unmatched_t, unmatched_d

    def update(self, dets: list, frame_enhanced: np.ndarray,
               cnn: "CNNClassifier | None", ood: "OODRejector | None"
               ) -> list:
        """Atualiza tracks com novas detecções. Retorna tracks vivos."""

        # Separa dets por confiança
        hi_idx = [i for i,d in enumerate(dets) if d["conf"] >= CONF_HIGH_THR]
        lo_idx = [i for i,d in enumerate(dets) if d["conf"] <  CONF_HIGH_THR]
        all_t  = list(range(len(self.tracks)))

        # ── Estágio 1: alta confiança → todos os tracks ──────────
        matched1, unm_t1, unm_d_hi = self._match_greedy(
            all_t, hi_idx, dets, TRACK_IOU_HIGH)

        # ── Estágio 2: baixa confiança → tracks não matchados ────
        matched2, unm_t2, _ = self._match_greedy(
            unm_t1, lo_idx, dets, TRACK_IOU_LOW)

        # ── Atualiza tracks matchados ────────────────────────────
        for ti, di in matched1 + matched2:
            d   = dets[di]
            lbl, conf_cnn = self._classificar(d, frame_enhanced, cnn, ood)
            self.tracks[ti].atualizar(
                d["bbox"], d["class_name"], d["conf"], lbl, conf_cnn)

        # ── Incrementa missed nos não matchados ──────────────────
        for ti in unm_t2:
            self.tracks[ti].missed += 1

        # ── Cria novos tracks para dets de alta conf não matchadas
        for di in unm_d_hi:
            d   = dets[di]
            lbl, conf_cnn = self._classificar(d, frame_enhanced, cnn, ood)
            t = Track(d["bbox"], d["class_name"], d["conf"])
            if lbl: t.buf.append((lbl, conf_cnn))
            self.tracks.append(t)

        # ── Remove tracks mortas ─────────────────────────────────
        self.tracks = [t for t in self.tracks if t.missed <= TRACK_MAX_AGE]
        return self.tracks

    @staticmethod
    def _classificar(det, frame_enhanced, cnn, ood):
        # Modo COCO: a classe do YOLO já é confiável — usa direto no voto,
        # sem passar pela CNN (que foi treinada em outro domínio).
        if det.get("coco", False):
            return det["class_name"], det["conf"]
        if cnn is None: return None, 0.0
        x1,y1,x2,y2 = det["bbox"]
        crop = frame_enhanced[y1:y2, x1:x2]
        if crop.size==0 or (x2-x1)<8 or (y2-y1)<8: return None, 0.0
        img_in = crop_para_cnn(crop, det["class_name"])
        scores = cnn.predict(img_in)
        max_s  = float(scores.max())
        cls_nm = CLASSES[int(scores.argmax())]
        if ood and not ood.aceitar(cls_nm, max_s): return None, 0.0
        return cls_nm, max_s

# ================================================================
#  [10] AÇÕES
# ================================================================

def executar(acao, ser):
    if acao not in ACOES: return
    a = ACOES[acao]
    CMD.update(mot=a["mot"],srv=a["srv"],buz=a["buz"],
               led=a["led"],brk=a["brk"],dir=a["dir"])
    enviar(CMD, ser)
    _nav.update(acao_label=acao, acao_t=time.monotonic(),
                acao_dur=a["dur"], cooldown=COOLDOWN_F, ultimo=acao)
    print(f"[NAV] ▶ {acao}", flush=True)


def tick(ser) -> bool:
    lbl = _nav["acao_label"]
    if not lbl: return False
    if lbl == "OBSTACLE": return True
    if time.monotonic()-_nav["acao_t"] >= _nav["acao_dur"]:
        _nav["acao_label"]=None
        print(f"[NAV] ✓ {lbl}", flush=True)
        # JORNADA ININTERRUPTA: ação terminou → retoma cruzeiro
        if MISSAO.estado == "RODANDO":
            CMD.update(mot=62,srv=127,buz=0,led=0,brk=0,dir=3)
            enviar(CMD,ser)
            print("[NAV] → cruzeiro retomado", flush=True)
        else:
            CMD.update(mot=0,srv=127,buz=0,led=0,brk=0,dir=0)
            enviar(CMD,ser)
        return False
    return True


def liberar_obstaculo(ser):
    _nav["acao_label"]=None
    print("[NAV] Obstáculo removido", flush=True)
    # Retoma cruzeiro se a missão está em curso
    if MISSAO.estado in ("RODANDO","PARADO_SEM"):
        CMD.update(mot=62,srv=127,buz=0,led=0,brk=0,dir=3)
        enviar(CMD,ser)
        print("[NAV] → cruzeiro retomado", flush=True)
    else:
        CMD.update(mot=0,srv=127,buz=0,led=0,brk=0,dir=0)
        enviar(CMD,ser)

# ================================================================
#  [11] VISUALIZAÇÃO
# ================================================================

def desenhar(frame_e, tracks, fps, modo, debug_thr):
    out = frame_e.copy(); h,w = out.shape[:2]; PW = 225

    cv2.rectangle(out,(0,int(h*ROI_Y0)),(w-1,int(h*ROI_Y1)),(0,200,255),1)

    for trk in tracks:
        x1,y1,x2,y2 = trk.bbox
        cls  = trk.class_hint
        cor  = COR_CLASSE.get(cls,(180,180,180))
        lbl, cm = trk.votar()
        thick = 3 if trk.state=="confirmed" and lbl else 1
        cv2.rectangle(out,(x1,y1),(x2,y2),cor,thick)

        buf_n = len(trk.buf)
        hdr   = f"#{trk.id} {cls}"
        if lbl: hdr += f" → {lbl} {cm*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(hdr,cv2.FONT_HERSHEY_SIMPLEX,0.37,1)
        cv2.rectangle(out,(x1,y1-th-6),(x1+tw+4,y1),cor,-1)
        cv2.putText(out,hdr,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.37,(255,255,255),1)

        bw = x2-x1
        prog = int(bw*buf_n/VOTE_BUFFER)
        cv2.rectangle(out,(x1,y2+2),(x2,y2+6),(40,40,40),-1)
        cv2.rectangle(out,(x1,y2+2),(x1+prog,y2+6),cor,-1)

    pan = np.full((h,PW,3),(18,18,18),dtype=np.uint8)
    cv2.rectangle(pan,(0,0),(PW-1,h-1),(45,45,45),1)
    def t(s,ln,cor=(190,190,190),sc=0.33):
        cv2.putText(pan,s,(5,14+ln*16),cv2.FONT_HERSHEY_SIMPLEX,sc,cor,1)

    t(f"FPS:{fps:.0f} [{modo}]",0,(255,255,255),0.38)
    cor_missao = {"AGUARDANDO":(0,220,220),"RODANDO":(100,220,100),
                  "PARADO_SEM":(50,50,220),"ENTREGANDO":(220,180,0),
                  "FINALIZADO":(180,180,180)}.get(MISSAO.estado,(190,190,190))
    t(f"MISSAO: {MISSAO.estado}",1,cor_missao,0.36)
    t(f"Dest:{MISSAO.destino} {MISSAO.progresso}/{len(MISSAO.rota())} "
      f"Trk:{len(tracks)}",2)

    acao = _nav["acao_label"]
    if acao:
        ca = COR_ACAO.get(acao,(200,200,200))
        t(f"EXEC: {acao}",3,ca,0.40)
        if acao!="OBSTACLE":
            t(f" {time.monotonic()-_nav['acao_t']:.1f}/{_nav['acao_dur']}s",4,ca)
    else:
        t("livre",3,(100,200,100))
        t(f"cd:{_nav['cooldown']}",4,(80,80,80))

    t("── CONFIRMADOS ──",6,(60,60,60))
    ln = 7
    for trk in tracks[:5]:
        lbl, cm = trk.votar()
        if lbl:
            t(f"#{trk.id} {lbl} {cm*100:.0f}%",ln,COR_CLASSE.get(lbl,(180,180,180)))
            ln += 1

    t("── VOTE BUFFER ──",13,(60,60,60))
    for i,trk in enumerate(tracks[:3]):
        buf_n = len(trk.buf)
        t(f"#{trk.id} {buf_n}/{VOTE_BUFFER} {trk.state[:4]}",14+i,
          COR_CLASSE.get(trk.class_hint,(120,120,120)))

    t("── ÚLTIMO ──",18,(60,60,60))
    ult = _nav["ultimo"] or "-"
    t(f" {ult}",19,COR_ACAO.get(ult,(160,160,160)))

    vis = np.empty((h,w+PW,3),np.uint8)
    vis[:,:w]=out; vis[:,w:]=pan

    if debug_thr is not None:
        thr_bgr = cv2.cvtColor(debug_thr,cv2.COLOR_GRAY2BGR)
        thr_bgr = cv2.resize(thr_bgr,(w+PW,h))
        vis = np.vstack([vis, thr_bgr])

    return vis

# ================================================================
#  [12] CALIBRAÇÃO (trackbars em tempo real)
# ================================================================

def calibrar(usar_camera):
    src = CAM_IDX if usar_camera else VIDEO
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened(): print("[ERRO] Fonte não abriu"); return

    win = "Calibração — Q=sair  S=salvar"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 520)
    def n(x): pass
    cv2.createTrackbar("CLAHE clip x10", win, 20, 50, n)
    cv2.createTrackbar("Bilateral d",    win,  5, 15, n)
    cv2.createTrackbar("Bilateral sigma",win, 40,150, n)
    cv2.createTrackbar("Thresh block",   win, 15, 51, n)
    cv2.createTrackbar("Thresh C",       win,  4, 25, n)
    cv2.createTrackbar("Sharpen ON",     win,  1,  1, n)
    print("[CAL] Ajuste os sliders. S=salvar. Q=sair.", flush=True)

    while True:
        ret,frm = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        h0,w0 = frm.shape[:2]
        scale  = IMG_W/w0
        frm    = cv2.resize(frm,(IMG_W,int(h0*scale)))

        clip  = max(0.5, cv2.getTrackbarPos("CLAHE clip x10",win)/10)
        bd    = max(1,   cv2.getTrackbarPos("Bilateral d",win))
        bs    = max(1,   cv2.getTrackbarPos("Bilateral sigma",win))
        blk   = cv2.getTrackbarPos("Thresh block",win)
        blk   = blk if blk%2==1 and blk>=3 else max(3,blk|1)
        tc    = cv2.getTrackbarPos("Thresh C",win)
        sh    = cv2.getTrackbarPos("Sharpen ON",win)==1

        cl    = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        lab   = cv2.cvtColor(frm, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cl.apply(lab[:,:,0])
        enh   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enh   = cv2.bilateralFilter(enh, bd, bs, bs)
        if sh: enh = cv2.filter2D(enh,-1,K_SHARP)

        gray  = cv2.medianBlur(cv2.cvtColor(enh,cv2.COLOR_BGR2GRAY),5)
        thr   = cv2.adaptiveThreshold(gray,255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, blk, tc)
        thr   = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,K5)
        thr   = cv2.morphologyEx(thr,cv2.MORPH_OPEN, K3)
        thr_c = cv2.cvtColor(thr,cv2.COLOR_GRAY2BGR)
        cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            a=cv2.contourArea(cnt)
            if AREA_CNT_MIN<a<AREA_CNT_MAX:
                x,y,bw,bh=cv2.boundingRect(cnt)
                cv2.rectangle(thr_c,(x,y),(x+bw,y+bh),(0,255,0),1)

        info = f"clip={clip:.1f} bil={bd}/{bs} thr={blk}/{tc} sharp={'Y' if sh else 'N'}"
        cv2.putText(enh,info,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,180),1)

        h,w = frm.shape[:2]
        vis = np.hstack([frm,enh,thr_c])
        cv2.imshow(win, vis)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('s'):
            cfg = dict(clahe_clip=clip,bilateral_d=bd,bilateral_s=bs,
                       thresh_block=blk,thresh_c=tc,usar_sharp=sh)
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"pre_config.json")
            with open(path,"w") as f: json.dump(cfg,f,indent=2)
            print(f"[CAL] Salvo: {path}\n{cfg}", flush=True)
    cap.release(); cv2.destroyAllWindows()

# ================================================================
#  [13] LOOP PRINCIPAL
# ================================================================

def main(usar_camera=False, debug=False, auto=False, fast=False):
    global _ser, _CLAHE

    # ── Modo FAST: inferência menor + sem pular frames ────────────
    # YOLO 416px é ~2.4x mais rápido que 640px com perda mínima de
    # acurácia em objetos médios/grandes. Essencial para alta velocidade.
    if fast:
        # NOTA: o tamanho de entrada do YOLO é fixo no arquivo .onnx
        # (definido no export, não pode mudar em runtime). O ganho de
        # velocidade real do --fast vem de processar mais frames e
        # confirmar votos mais rápido — não de reduzir o imgsz aqui.
        print("[FAST] Processando todo frame (sem pular) + voting rápido",
              flush=True)
        # Voting mais responsivo: confirma com 4 de 6 frames.
        # A alta velocidade, o objeto fica poucos frames na tela —
        # buffer de 10 nunca fecharia antes do carro passar.
        global VOTE_BUFFER, VOTE_MIN_DETS, VOTE_FRAC
        VOTE_BUFFER   = 6
        VOTE_MIN_DETS = 3
        VOTE_FRAC     = 0.60
        print("[FAST] Voting 6 frames, confirma com 4/6", flush=True)

    # Carrega config de calibração
    cfg_p = os.path.join(os.path.dirname(os.path.abspath(__file__)),"pre_config.json")
    if os.path.exists(cfg_p):
        with open(cfg_p) as f: _cfg = json.load(f)
        _CLAHE = cv2.createCLAHE(
            clipLimit=_cfg.get("clahe_clip",2.0), tileGridSize=(8,8))
        print(f"[PRE] Config: {_cfg}", flush=True)

    # CNN (obrigatória para classificar)
    cnn = ood = None
    if os.path.exists(CNN_MODEL):
        try:   cnn=CNNClassifier(CNN_MODEL); ood=OODRejector(OOD_FILE)
        except Exception as e: print(f"[WARN] CNN: {e}", flush=True)
    else:
        print(f"[WARN] CNN não encontrada → execute: python TRAIN_SIGN_CNN.py",
              flush=True)

    # YOLO (opcional — fallback para contornos se ausente)
    yolo = None; modo = "CONTORNO"
    if os.path.exists(YOLO_MODEL):
        try:   yolo=YOLODetector(YOLO_MODEL); modo="YOLO"
        except Exception as e: print(f"[WARN] YOLO: {e}", flush=True)
    else:
        print(f"[INFO] YOLO não encontrado → detectando por contornos\n"
              f"       Para treinar: python TRAIN_YOLO.py", flush=True)

    tracker = ByteTrackLite()

    src = CAM_IDX if usar_camera else VIDEO
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened(): print("[ERRO] Fonte de vídeo"); sys.exit(1)
    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    fps_vid  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1,int(1000/fps_vid)) if not usar_camera else 1

    _ser = conectar_serial()
    fps_t=time.monotonic(); fps_n=0; fps=0.0; fn=0
    SKIP = 1 if fast else 2   # fast: processa todo frame

    # ── AUTO-START: começa em RODANDO sem esperar sinal ───────────
    if auto:
        print("[MISSAO] Auto-start ativo — carro em cruzeiro", flush=True)
        MISSAO.set_estado("RODANDO")
        executar("STRAIGHT", _ser)
        MISSAO.registrar_comando("STRAIGHT")

    frame_e = None; debug_thr = None

    print("[OK] Q=sair  SPACE=pausa  +=acelera  -=desacelera", flush=True)

    while True:
        ret, frame_raw = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        fn += 1

        if _nav["cooldown"]>0: _nav["cooldown"]-=1
        em_acao = tick(_ser)

        if fn % SKIP == 0:
            # ── Preprocessing principal ────────────────────────────
            frame_e = preprocessar(frame_raw)
            h, w    = frame_e.shape[:2]
            debug_thr = None

            # ── Localização ────────────────────────────────────────
            if yolo:
                dets = yolo.detectar(frame_e)
                det_modo = "YOLO"
            else:
                dets = []
                det_modo = "CONTORNO"

            # Fallback: só roda se YOLO encontrou nada
            if not dets:
                dets = detectar_contornos_fallback(frame_e)
                det_modo = "CONTORNO"
                if debug:
                    # Gera imagem de threshold para debug
                    gray = cv2.medianBlur(cv2.cvtColor(frame_e,cv2.COLOR_BGR2GRAY),5)
                    debug_thr = cv2.adaptiveThreshold(
                        gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,15,4)

            # Filtra área mínima
            dets = [d for d in dets
                    if (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])
                    >= AREA_MIN_EXEC]

            # ── Tracker + CNN ──────────────────────────────────────
            tracks = tracker.update(dets, frame_e, cnn, ood)

            # ── Lê mensagens do Arduino (ACKs, botões, sensores) ──
            for m in ler_serial(_ser):
                if m.get("btn") == "start" and MISSAO.estado == "AGUARDANDO":
                    MISSAO.set_estado("RODANDO")
                    executar("STRAIGHT", _ser)
                    MISSAO.registrar_comando("STRAIGHT")

            # ── Semáforo: verifica cor nos tracks confirmados ─────
            sem_cor = None
            for trk in tracks:
                if trk.class_hint == "Semaforo" and trk.votar()[0]:
                    x1,y1,x2,y2 = trk.bbox
                    sem_cor = cor_semaforo(frame_e[y1:y2, x1:x2])
                    break

            # ── MÁQUINA DE ESTADOS DA MISSÃO ──────────────────────
            if MISSAO.estado == "AGUARDANDO":
                # Partida liberada por: semáforo verde OU botão do Arduino
                if sem_cor == "verde":
                    print("[MISSAO] 🟢 Semáforo verde — PARTIDA!", flush=True)
                    MISSAO.set_estado("RODANDO")
                    executar("STRAIGHT", _ser)
                    MISSAO.registrar_comando("STRAIGHT")

            elif MISSAO.estado == "RODANDO":
                # Semáforo vermelho à frente → para e espera
                if sem_cor == "vermelho" and _nav["acao_label"] is None:
                    MISSAO.set_estado("PARADO_SEM")
                    executar("OBSTACLE", _ser)   # para sem duração fixa

                # Decisão normal de ações (placas, obstáculos)
                elif _nav["cooldown"]==0 and not em_acao:
                    candidatas = sorted(
                        [(t,*t.votar()) for t in tracks
                         if t.state=="confirmed" and t.votar()[0]],
                        key=lambda x: -x[0].area)
                    for trk, lbl, cm in candidatas:
                        if trk.area < AREA_MIN_EXEC: continue
                        if lbl == "Semaforo": continue   # tratado acima
                        acao = CLASS_TO_ACTION.get(lbl)
                        if acao:
                            executar(acao, _ser)
                            if acao in ("LEFT","RIGHT","STRAIGHT"):
                                MISSAO.registrar_comando(acao)
                            break

            elif MISSAO.estado == "PARADO_SEM":
                # Espera o semáforo abrir
                if sem_cor == "verde":
                    print("[MISSAO] 🟢 Verde — retomando", flush=True)
                    liberar_obstaculo(_ser)
                    MISSAO.set_estado("RODANDO")
                    executar("STRAIGHT", _ser)

            elif MISSAO.estado == "ENTREGANDO":
                # Chegou ao destino: para, buzzer, aguarda confirmação
                if _nav["acao_label"] is None:
                    CMD.update(mot=0,srv=127,buz=1,led=1,brk=1,dir=0)
                    enviar(CMD,_ser)
                    print(f"[MISSAO] 📦 Entrega no ponto {MISSAO.destino}!",
                          flush=True)
                    MISSAO.set_estado("FINALIZADO")

            # ── Libera OBSTACLE (só fora de PARADO_SEM) ───────────
            if (_nav["acao_label"]=="OBSTACLE"
                    and MISSAO.estado != "PARADO_SEM"):
                obs_vivo = any(
                    CLASS_TO_ACTION.get(t.class_hint)=="OBSTACLE"
                    and t.votar()[0] and t.state=="confirmed"
                    for t in tracks)
                if not obs_vivo: liberar_obstaculo(_ser)

            modo = det_modo

        # ── Visualização ───────────────────────────────────────────
        disp = frame_e if frame_e is not None else \
               cv2.resize(frame_raw,(IMG_W,int(frame_raw.shape[0]*IMG_W/frame_raw.shape[1])))
        vis = desenhar(disp, tracker.tracks, fps, modo, debug_thr)
        cv2.imshow("Carro Autônomo v4.0", vis)

        fps_n += 1
        if time.monotonic()-fps_t >= 1.0:
            fps=fps_n; fps_n=0; fps_t=time.monotonic()

        k = cv2.waitKey(delay_ms)&0xFF
        if   k==ord('q'): break
        elif k==ord(' '): cv2.waitKey(0)
        elif k==ord('+') and not usar_camera: delay_ms=max(1,delay_ms-5)
        elif k==ord('-') and not usar_camera: delay_ms=min(200,delay_ms+5)
        elif k==ord('g') and MISSAO.estado=="AGUARDANDO":
            print("[MISSAO] Partida manual (tecla G)", flush=True)
            MISSAO.set_estado("RODANDO")
            executar("STRAIGHT", _ser)
            MISSAO.registrar_comando("STRAIGHT")
        elif k in (ord('1'),ord('2'),ord('3')) and MISSAO.estado=="AGUARDANDO":
            MISSAO.destino = {ord('1'):"A",ord('2'):"B",ord('3'):"C"}[k]
            print(f"[MISSAO] Destino: {MISSAO.destino} — rota: "
                  f"{' → '.join(MISSAO.rota())}", flush=True)

    CMD.update(mot=0,srv=127,buz=0,led=0,brk=1,dir=0)
    enviar(CMD,_ser)
    if _ser: _ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)


if __name__=="__main__":
    args = sys.argv[1:]
    if "--cal" in args: calibrar("--cam" in args)
    else: main(usar_camera="--cam" in args,
               debug="--debug" in args,
               auto="--auto" in args,
               fast="--fast" in args)