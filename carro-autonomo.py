"""
================================================================
  CARRO AUTÔNOMO v5.1 — PIPELINE MONO GEOMÉTRICO DE ALTO FPS
  ─────────────────────────────────────────────────────────────
  Webcam (backend nativo + MJPG)
    ↓ Grayscale
    ↓ CLAHE
    ↓ Gaussian Blur
    ↓ Adaptive Threshold (2 polaridades)
    ↓ Morphological Open/Close
    ↓ [--canny opcional — ver nota em segmentar()]
    ↓ Contornos → ApproxPolyDP
    ↓ Filtro geométrico: SÓ octógono/8 lados (PARE) e semáforo
    │  (retângulo vertical com 3 faróis) — nada mais é alvo
    ↓ ROI DINÂMICO (histórico da última posição — olha onde
    │  as placas costumam aparecer; varredura completa periódica)
    ↓ Recorte das regiões candidatas
    ↓ [--yolo auxiliar 320×320 a cada N frames, só no ROI]
    ↓ PGOM — Persistent Geometric Object Manager: cada candidato
    │  mantém uma FICHA de evidências (forma 20%% · convexidade
    │  10%% · aspect 10%% · símbolo interno 15%% · persistência
    │  5-8 frames 20%% · estabilidade centro/área 15%% ·
    │  fingerprint geométrico 10%%); promoção à CNN só com ≥85%%
    ↓ CNN 64×64 INT8 — CONFIRMADORA de alta precisão para poucos
    │  candidatos de excelente qualidade
    ↓ Score de decisão: 0.85 evidências + 0.15 CNN
    ↓ Votação temporal + missão
    ↓ Controle do veículo (Serial JSON)

  Sem dependência de cor: semáforo lido pela POSIÇÃO do farol
  aceso (cima=vermelho, baixo=verde).

  Câmera: CAP_DSHOW no Windows, CAP_V4L2 no Linux (pista),
  FOURCC MJPG + buffer 1 → captura na taxa máxima do sensor.
  (v4.0 abaixo — histórico)
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
CAM_IDX     = 1        # câmera externa da pista (0 = webcam embutida)
# Espelhamento da imagem da câmera (cv2.flip):
#    None = sem flip |  1 = horizontal (efeito espelho) |
#    0 = vertical    | -1 = ambos (gira 180°)
CAM_FLIP    = 1        # padrão: corrige espelho horizontal
IMG_W       = 640        # largura alvo antes de entrar no YOLO

SERIAL_PORT = "COM3"
BAUD        = 115200

YOLO_MODEL  = "./models/sign_detector.onnx"
CNN_MODEL   = "./models/sign_classifier.tflite"
OOD_FILE    = "./models/ood_thresholds.json"

CNN_SIZE    = 96
OOD_DEFAULT = 0.55
YOLO_CONF   = 0.25    # piso global (filtro fino é por classe, abaixo)

# Confiança mínima POR CLASSE (modo COCO) — implementa a prioridade:
#   placas = muito relevante  → threshold baixo (não perder)
#   semáforo = relevante      → threshold médio
#   pessoa/carro = menos rel. → threshold alto (evitar falso positivo)
COCO_CONF_MIN = {
    "Stop":     0.28,
    "Semaforo": 0.40,
    "Carro":    0.55,
    "Pessoa":   0.62,
}
# Área mínima por classe (px²) — pessoa "fantasma" costuma ser pequena
COCO_AREA_MIN = {
    "Pessoa": 2000,
    "Carro":  1500,
}
YOLO_NMS    = 0.45
YOLO_SIZE   = 640     # fallback p/ ONNX dinâmico (export ideal: 320)
YOLO_EVERY  = 5       # YOLO auxiliar roda 1x a cada N frames

CLASSES = ["Stop","Esquerda","Direita","SemRetorno",
           "Verde","Cone","Carro","Pessoa","Fundo"]
# CNN_CLASSES é sobrescrito em runtime por models/classes.txt
CNN_CLASSES = CLASSES
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

def estado_semaforo_mono(crop_gray: np.ndarray) -> str | None:
    """
    Estado do semáforo SEM COR: posição vertical do farol ACESO.
    Layout FÍSICO da pista (3 faróis, de cima p/ baixo):
        ┌─────┐
        │  ●  │  topo  → VERMELHO  → PARA
        │  ○  │  meio  → AMARELO   → atenção
        │  ○  │  base  → VERDE     → ANDA
        └─────┘
    O farol aceso satura (quase branco) na imagem mono, então
    basta achar em qual dos 3 terços verticais está a zona mais
    clara. Retorna 'vermelho' | 'amarelo' | 'verde' | None.
    """
    if crop_gray is None or crop_gray.size == 0: return None
    if crop_gray.ndim == 3:
        crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(crop_gray, (30, 90))
    tercos = [float(g[0:30].mean()),   # topo → VERMELHO (para)
              float(g[30:60].mean()),  # meio → AMARELO
              float(g[60:90].mean())]  # base → VERDE (anda)
    i = int(np.argmax(tercos))
    contraste = max(tercos) - sorted(tercos)[1]
    if contraste < 8: return None      # nenhum farol claramente aceso
    return ["vermelho", "amarelo", "verde"][i]


def analisar_semaforo(crop) -> str | None:
    """Roteia conforme a situação da fonte:
    crop com cor → máscaras HSV | mono/cinza → posição do farol.
    Como o pipeline v5 trabalha em cinza, o método por posição é
    o padrão; o de cor fica disponível para fontes coloridas."""
    if crop is None or crop.size == 0: return None
    if crop.ndim == 2 or _sat_chk["mono"]:
        return estado_semaforo_mono(crop)
    return cor_semaforo(crop)


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
#  [3d] CÂMERA — backend nativo por plataforma + MJPG
#
#  cv2.VideoCapture(0) sem backend cai no autodetect (lento e,
#  no Windows, às vezes MSMF com latência alta). Forçamos:
#    Windows → CAP_DSHOW   |   Linux (pista) → CAP_V4L2
#  FOURCC MJPG: a webcam envia JPEG comprimido em vez de YUY2
#  bruto → USB deixa de ser gargalo → 30/60 fps reais em 640p.
#  BUFFERSIZE 1: sempre o frame mais recente (menor latência).
# ================================================================

def _abrir_idx(idx: int):
    """Tenta abrir UM índice com o backend nativo da plataforma."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") \
              else cv2.CAP_V4L2
    nome = "DSHOW" if backend == cv2.CAP_DSHOW else "V4L2"
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():                      # fallback autodetect
        cap = cv2.VideoCapture(idx); nome = "AUTO"
    if not cap.isOpened():
        return None, nome
    ret, _ = cap.read()                         # confirma que ENTREGA frame
    if not ret:
        cap.release(); return None, nome
    return cap, nome


def abrir_camera(idx: int) -> cv2.VideoCapture:
    """
    Abre a câmera do índice pedido. Se ela não responder, VARRE
    os índices 0..5 e usa a primeira que entregar imagem — assim
    o sistema nunca cai calado na webcam errada, e avisa qual
    índice acabou usando.
    """
    cap, nome = _abrir_idx(idx)
    if cap is None:
        print(f"[CAM] índice {idx} não respondeu — varrendo 0..5...",
              flush=True)
        for alt in range(6):
            if alt == idx: continue
            cap, nome = _abrir_idx(alt)
            if cap is not None:
                print(f"[CAM] usando índice {alt} (troque CAM_IDX ou "
                      f"passe --cam-idx {alt})", flush=True)
                idx = alt; break
    if cap is None:
        print("[CAM][ERRO] Nenhuma câmera respondeu. Rode "
              "SCAN_CAMERAS.py e verifique cabo / se outro app "
              "está usando a câmera.", flush=True)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fstr = "".join(chr((fourcc >> 8*i) & 0xFF) for i in range(4))
    print(f"[CAM] idx={idx} backend={nome} fourcc={fstr} "
          f"{cap.get(3):.0f}x{cap.get(4):.0f}@{cap.get(5):.0f}fps",
          flush=True)
    return cap

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

_sat_chk = dict(fn=0, avisado=False, mono=False)

def checar_camera(frame_bgr: np.ndarray):
    """
    Detecta fonte de vídeo SEM COR (mono/binarizada).
    Um frame com saturação ~0 quebra todo o pipeline:
      · COCO não reconhece a placa PARE (sem vermelho, sem texto)
      · cor_semaforo() nunca retorna verde/vermelho (máscaras HSV vazias)
      · blobs binários geram "pessoas" fantasmas
    Verifica 1x a cada 90 frames (barato).
    """
    _sat_chk["fn"] += 1
    if _sat_chk["fn"] % 90 != 1: return
    hsv = cv2.cvtColor(cv2.resize(frame_bgr,(160,90)), cv2.COLOR_BGR2HSV)
    sat = float(hsv[:,:,1].mean())
    _sat_chk["mono"] = sat < 5.0
    if _sat_chk["mono"] and not _sat_chk["avisado"]:
        _sat_chk["avisado"] = True
        print("="*60, flush=True)
        print("[CAMERA] ⚠ VÍDEO SEM COR (saturação ~0)!", flush=True)
        print("[CAMERA] A fonte está em modo mono/binarizado.", flush=True)
        print("[CAMERA] Verifique: modo da webcam, filtro de captura,", flush=True)
        print("[CAMERA] exposição/contraste. PARE e semáforo NÃO serão", flush=True)
        print("[CAMERA] detectados corretamente sem imagem colorida.", flush=True)
        print("="*60, flush=True)


def preprocessar(frame_bgr: np.ndarray) -> np.ndarray:
    """
    v5 MONO: retorna frame em ESCALA DE CINZA (2D) pronto para o
    detector geométrico e para a CNN.
    Removidos (custo alto, ganho nulo em imagem mono):
      · conversão LAB  · bilateral filter  · sharpen
    Mantido: resize + CLAHE (barato e essencial p/ contraste).
    ~5x mais rápido que o preprocessing v4.
    """
    h0, w0 = frame_bgr.shape[:2]
    if w0 != IMG_W:
        scale = IMG_W / w0
        frame_bgr = cv2.resize(frame_bgr, (IMG_W, int(h0*scale)),
                               interpolation=cv2.INTER_LINEAR)
    gray = frame_bgr if frame_bgr.ndim == 2 else \
           cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)


# ================================================================
#  [6] DETECTOR GEOMÉTRICO (principal no modo mono)
#
#  "verificar bordas e formatos geométricos":
#  binariza nas DUAS polaridades (objetos escuros E claros),
#  extrai contornos e filtra por forma:
#    · ~quadrado + 7-9 vértices + circularidade alta → octógono (PARE)
#    · ~quadrado + circularidade > 0.82            → círculo (placa)
#    · retângulo vertical alto                      → semáforo
#  Custa ~3-5 ms/frame em CPU → FPS alto.
#  A confirmação fina fica com a CNN (etapa seguinte).
# ================================================================

GEO_AREA_MIN   = 400
GEO_AREA_FRAC  = 0.25      # área máx = 25% do frame
GEO_MAX_CANDS  = 10        # YOLO/CNN olham no máx. 10 regiões
ROI_FULL_EVERY = 15        # varredura completa a cada N frames
ROI_EXPAND     = 1.8       # expansão da janela ao redor do histórico

def segmentar(gray: np.ndarray, usar_canny: bool = False) -> tuple:
    """
    Gaussian Blur → Adaptive Threshold (2 polaridades) →
    Morphological Open/Close.

    NOTA SOBRE O CANNY (--canny, desligado por padrão):
    após o adaptive threshold a imagem já é BINÁRIA — o
    findContours extrai exatamente as mesmas bordas que o Canny
    extrairia, de graça. Rodar Canny aqui duplica custo e, como
    já observado neste projeto, reintroduz falsos positivos nas
    bordas do frame/interface. O flag existe para demonstração
    e comparação em aula.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr_esc = cv2.adaptiveThreshold(          # objetos ESCUROS
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5)
    thr_cla = cv2.adaptiveThreshold(          # objetos CLAROS
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5)
    # Adaptive esvazia regiões grandes e uniformes (só bordas
    # locais sobrevivem) — ótimo p/ placas, ruim p/ estruturas
    # grandes como a carcaça do semáforo. O Otsu GLOBAL cobre
    # essas: sólidos escuros inteiros viram um contorno só.
    _, otsu_esc = cv2.threshold(blur, 0, 255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    saidas = []
    for t in (thr_esc, thr_cla, otsu_esc):
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN,  K3)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, K5)
        if usar_canny:
            t = cv2.Canny(t, 50, 150)
            t = cv2.dilate(t, K3)   # religa bordas p/ contorno fechado
        saidas.append(t)
    return saidas


class ROIDinamico:
    """
    ROI móvel com histórico: guarda as últimas posições de
    detecções confirmadas e concentra a busca onde as placas
    costumam aparecer. Varredura COMPLETA periódica (e sempre
    que o histórico esvazia) para captar objetos novos.
    """
    def __init__(self):
        self.hist: deque = deque(maxlen=12)   # últimos bboxes
        self.fn = 0

    def registrar(self, bbox): self.hist.append(bbox)

    def janela(self, h: int, w: int) -> tuple:
        """Retorna (x0,y0,x1,y1) da região de busca deste frame."""
        self.fn += 1
        if not self.hist or self.fn % ROI_FULL_EVERY == 0:
            return (0, int(h*ROI_Y0), w, int(h*ROI_Y1))   # completa
        xs0 = min(b[0] for b in self.hist)
        ys0 = min(b[1] for b in self.hist)
        xs1 = max(b[2] for b in self.hist)
        ys1 = max(b[3] for b in self.hist)
        cx, cy = (xs0+xs1)/2, (ys0+ys1)/2
        bw = max(xs1-xs0, 80) * ROI_EXPAND
        bh = max(ys1-ys0, 80) * ROI_EXPAND
        x0 = int(max(0, cx-bw)); x1 = int(min(w, cx+bw))
        y0 = int(max(h*ROI_Y0, cy-bh)); y1 = int(min(h*ROI_Y1, cy+bh))
        if x1-x0 < 64 or y1-y0 < 64:
            return (0, int(h*ROI_Y0), w, int(h*ROI_Y1))
        return (x0, y0, x1, y1)

ROI_DIN = ROIDinamico()


def analisar_farois(crop_gray: np.ndarray) -> tuple:
    """
    Validação ESTRUTURAL do semáforo:
      Retângulo → existem círculos internos? → 2? 3? →
      alinhados VERTICALMENTE? → candidato a semáforo.
    Retorna (n_circulos_empilhados, alinhados: bool).
    Postes/pernas de cadeira falham aqui: não têm 2-3 círculos
    na mesma coluna vertical.
    """
    if crop_gray.size == 0: return 0, False
    g = cv2.resize(crop_gray, (36, 100))
    circulos = []          # (cx, cy) de cada blob circular
    for thr_img in (cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
                    cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]):
        cnts,_ = cv2.findContours(thr_img, cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 40 or a > 1200: continue
            x,y,w,h = cv2.boundingRect(c)
            if not (0.6 < w/max(h,1) < 1.7): continue
            circ = 4*np.pi*a/max(cv2.arcLength(c,True)**2,1)
            if circ > 0.55:
                circulos.append((x + w/2, y + h/2))
    if len(circulos) < 2: return len(circulos), False
    circulos.sort(key=lambda p: p[1])
    # empilhados: separação vertical mínima entre faróis vizinhos
    empilhados = [circulos[0]]
    for p in circulos[1:]:
        if p[1] - empilhados[-1][1] > 15:
            empilhados.append(p)
    n = len(empilhados)
    if n < 2: return n, False
    # alinhados verticalmente: desvio horizontal dos centros pequeno
    xs = [p[0] for p in empilhados]
    alinhado = (max(xs) - min(xs)) < 12    # em px do crop 36 de largura
    return n, alinhado


def _tem_farois(crop_gray: np.ndarray) -> bool:
    # Semáforo da pista tem 3 faróis (vermelho/amarelo/verde).
    # Aceita 2 como tolerância: se um farol estiver apagado e não
    # gerar contorno, ainda reconhece a estrutura. Nunca aceita
    # 1 (poste) nem 4+ (padrão de fundo aleatório).
    n, alinhado = analisar_farois(crop_gray)
    return 2 <= n <= 3 and alinhado


def _tem_simbolo(crop_gray: np.ndarray) -> bool:
    """
    Existe SÍMBOLO/texto interno? Evidência para o PGOM: o PARE
    tem 'PARE'/'STOP' escrito dentro do octógono. Mede a fração
    da fase minoritária no miolo — superfície lisa (parede, placa
    em branco, disco vazio) dá ~0; texto real dá 8-50%%.
    """
    if crop_gray.size == 0: return False
    g = cv2.resize(crop_gray, (48, 48))
    miolo = g[10:38, 10:38]      # ignora a borda/anel externo
    _, t = cv2.threshold(miolo, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f = float(t.mean()) / 255.0
    # Fase MINORITÁRIA: independe da polaridade — funciona tanto
    # p/ seta escura em placa clara quanto p/ texto claro no PARE
    # escuro. Disco vazio → minoria ~0; símbolo real → 8-50%.
    minoria = min(f, 1.0 - f)
    return minoria > 0.08


def detectar_geometrico(gray: np.ndarray,
                        usar_canny: bool = False) -> list:
    """
    Contornos → ApproxPolyDP → filtro por número de lados:
      3       → triângulo  (placa de advertência/preferência)
      4-5     → retângulo  (vertical alto → teste de semáforo)
      6+      → círculo/octógono (7-9 vértices + circularidade
                 alta = octógono → prioridade PARE)
    Busca apenas dentro do ROI DINÂMICO (histórico de posições).
    """
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = ROI_DIN.janela(h, w)
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0: return []

    cands = []
    for m in segmentar(sub, usar_canny):
        cnts, _ = cv2.findContours(m, cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < GEO_AREA_MIN or a > GEO_AREA_FRAC*h*w: continue
            x, y, bw, bh = cv2.boundingRect(c)
            ar    = bw / max(bh, 1)
            solid = a / max(bw*bh, 1)
            if solid < 0.35: continue          # contorno irregular
            peri  = cv2.arcLength(c, True)
            circ  = 4*np.pi*a / max(peri*peri, 1)
            nv    = len(cv2.approxPolyDP(c, 0.02*peri, True))
            # Convexidade: área / área do fecho convexo.
            # Placas são convexas (~1.0); galhos/pessoas não.
            hull  = cv2.convexHull(c)
            convx = a / max(cv2.contourArea(hull), 1)

            hint = None
            # ── APENAS DOIS ALVOS NA PISTA ──────────────────────
            #  1) PARE   = OCTÓGONO (8 lados)
            #  2) SEMÁFORO = retângulo vertical com 3 faróis
            # Círculos soltos, triângulos e retângulos NÃO são
            # alvos → não viram candidatos (eliminam falso positivo).
            if 0.25 < ar < 0.62 and bh > 50:
                # Semáforo: aspecto vertical + faróis empilhados
                if _tem_farois(sub[y:y+bh, x:x+bw]):
                    hint = "Semaforo"
            elif 7 <= nv <= 9 and circ > 0.62 and 0.70 < ar < 1.40:
                # Octógono do PARE: 8 lados (tolera 7-9 por ruído),
                # circularidade alta e proporção ~quadrada
                hint = "Stop"
            if hint is None: continue
            gx, gy = x + x0, y + y0            # volta p/ coord. global
            # Hierarquia de contornos: símbolo/estrutura interna
            # (texto do PARE, seta da placa, faróis do semáforo)
            simb = _tem_simbolo(sub[y:y+bh, x:x+bw]) \
                   if hint != "Semaforo" else True
            cands.append({"bbox": (gx, gy, gx+bw, gy+bh),
                          "class_name": hint, "class_id": -1,
                          "conf": float(solid), "geo": True,
                          "lados": nv, "circ": float(circ),
                          "ar": float(ar), "area": float(a),
                          "convex": float(convx),
                          "simbolo": bool(simb)})

    cands.sort(key=lambda d: -(d["bbox"][2]-d["bbox"][0])
                              *(d["bbox"][3]-d["bbox"][1]))
    keep = []
    for d in cands:
        dup = False
        for k in keep:
            ax1,ay1,ax2,ay2 = d["bbox"]; bx1,by1,bx2,by2 = k["bbox"]
            ix = max(0, min(ax2,bx2)-max(ax1,bx1))
            iy = max(0, min(ay2,by2)-max(ay1,by1))
            if ix*iy > 0.6*min((ax2-ax1)*(ay2-ay1),
                               (bx2-bx1)*(by2-by1)):
                dup = True; break
        if not dup: keep.append(d)
    return keep[:GEO_MAX_CANDS]

# ================================================================
#  [6b] PGOM — PERSISTENT GEOMETRIC OBJECT MANAGER
#
#  Sistema de MÚLTIPLAS EVIDÊNCIAS. Nenhuma etapa isolada tem
#  permissão para promover um objeto à CNN: cada candidato mantém
#  uma FICHA que acumula evidências geométricas e temporais.
#  A promoção ocorre apenas quando a soma ponderada ultrapassa
#  PGOM_PROMOVE. A CNN deixa de ser filtro de centenas de
#  candidatos e vira CONFIRMADORA de alta precisão para poucos
#  candidatos de excelente qualidade.
#
#  Ficha de evidências (perfil PLACA):
#    Forma compatível ................. 20%
#    Convexidade elevada .............. 10%
#    Aspect ratio esperado ............ 10%
#    Hierarquia de contornos (símbolo)  15%
#    Persistência por 5-8 frames ...... 20%
#    Estabilidade de centro e área .... 15%
#    Fingerprint geométrico ........... 10%
#
#  Perfil SEMÁFORO: mesmo princípio, pesos ajustados à física do
#  objeto — pequeno e trêmulo, então a persistência pesa menos e
#  a ESTRUTURA interna (2-3 faróis alinhados, já validada no
#  detector) pesa mais. Nenhum objeto é promovido por uma
#  evidência só, em nenhum perfil.
#
#  Árvores, postes, nuvens e reflexos raramente sobrevivem: não
#  mantêm forma + centro + proporção + estrutura interna ao mesmo
#  tempo por vários frames. Placas e semáforos mantêm.
# ================================================================

PGOM_PROMOVE     = 0.85  # limiar de promoção — perfil PLACA
PGOM_PROMOVE_SEM = 0.72  # perfil SEMÁFORO: objeto pequeno e
                         # intermitente; a evidência estrutural
                         # (faróis alinhados, 35%) é forte por
                         # frame, mas persistência/estabilidade
                         # rendem menos — o teto da soma é menor
PGOM_MAX_MISS  = 3      # frames sumido → ficha zera
PGOM_MATCH_D   = 60     # px máx. entre centros p/ ser o mesmo objeto
PGOM_HIST      = 10     # janela de histórico da ficha
PGOM_PERSIST_N = 8      # persistência satura em N frames (rampa 0→1)

PESOS_PLACA = dict(forma=0.20, convex=0.10, aspecto=0.10,
                   simbolo=0.15, persist=0.20, estab=0.15,
                   fingerprint=0.10)
PESOS_SEMAFORO = dict(forma=0.15, convex=0.05, aspecto=0.15,
                      simbolo=0.35, persist=0.05, estab=0.15,
                      fingerprint=0.10)

class Ficha:
    """Dossiê de um candidato: acumula evidências frame a frame."""
    _nid = 0

    def __init__(self, det):
        self.id = Ficha._nid; Ficha._nid += 1
        self.centros  = deque(maxlen=PGOM_HIST)
        self.areas    = deque(maxlen=PGOM_HIST)
        self.formas   = deque(maxlen=PGOM_HIST)  # score forma/frame
        self.convexs  = deque(maxlen=PGOM_HIST)
        self.aspectos = deque(maxlen=PGOM_HIST)
        self.simbolos = deque(maxlen=PGOM_HIST)
        self.fingerps = deque(maxlen=PGOM_HIST)  # assinatura quantizada
        self.vistos   = 0
        self.missed   = 0
        self.det      = det
        self._push(det)

    @staticmethod
    def _score_forma(det) -> float:
        """Forma compatível com placa/sinal, graduada."""
        nv, circ = det.get("lados",0), det.get("circ",0)
        if det["class_name"] == "Semaforo":       return 1.0
        if 7 <= nv <= 9 and circ > 0.62:          return 1.0   # octógono
        if nv == 3:                               return 0.9   # triângulo
        if nv >= 6 and circ > 0.70:               return 0.9   # círculo
        if nv in (4,5):                           return 0.7   # retângulo
        return 0.3

    @staticmethod
    def _fingerprint(det) -> tuple:
        """Assinatura geométrica quantizada: (lados, circ, aspecto)
        em buckets. Objetos reais mudam de assinatura entre frames;
        placas rígidas mantêm a mesma."""
        nv = det.get("lados",0)
        # agrupa nv (4≈5, 8≈9) e usa buckets grossos p/ circ e ar —
        # buckets finos oscilam na fronteira e punem objetos rígidos
        grupo_nv = 3 if nv==3 else 4 if nv in (4,5) else \
                   7 if nv in (6,7) else 9
        return (grupo_nv,
                int(det.get("circ",0)*3),
                int(det.get("ar",1.0)*3))

    def _push(self, det):
        x1,y1,x2,y2 = det["bbox"]
        self.centros.append(((x1+x2)/2, (y1+y2)/2))
        self.areas.append(max(1.0,(x2-x1)*(y2-y1)))
        self.formas.append(self._score_forma(det))
        self.convexs.append(float(det.get("convex", 0.5)))
        ideal = 0.40 if det["class_name"]=="Semaforo" else 1.00
        ar = float(det.get("ar",1.0))
        self.aspectos.append(float(np.clip(1.0-abs(ar-ideal)/ideal,0,1)))
        self.simbolos.append(1.0 if det.get("simbolo") else 0.0)
        self.fingerps.append(self._fingerprint(det))
        self.vistos += 1
        self.missed  = 0
        self.det     = det

    def perto_de(self, det) -> float:
        x1,y1,x2,y2 = det["bbox"]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        px, py = self.centros[-1]
        return ((cx-px)**2 + (cy-py)**2) ** 0.5

    def evidencias(self) -> dict:
        """Calcula cada evidência da ficha, todas em [0,1]."""
        # Persistência: rampa 0→1 até PGOM_PERSIST_N frames
        e_persist = float(np.clip(self.vistos/PGOM_PERSIST_N, 0, 1))
        # Estabilidade = SUAVIDADE, não constância. O carro se
        # APROXIMA da placa: área cresce e centro deriva — isso é
        # esperado. O que denuncia ruído é variação ERRÁTICA:
        # passos de tamanho inconstante, área pulsando.
        if len(self.centros) >= 3:
            passos = [((self.centros[i+1][0]-self.centros[i][0])**2 +
                       (self.centros[i+1][1]-self.centros[i][1])**2)**0.5
                      for i in range(len(self.centros)-1)]
            e_c = float(np.clip(1.0 - np.std(passos)/12.0, 0, 1))
            razoes = [self.areas[i+1]/self.areas[i]
                      for i in range(len(self.areas)-1)]
            e_a = float(np.clip(1.0 - np.std(razoes)/0.25, 0, 1))
            e_estab = (e_c+e_a)/2
        else:
            e_estab = 0.4
        # Fingerprint: fração dos frames com a assinatura dominante
        cnt = Counter(self.fingerps)
        e_fp = cnt.most_common(1)[0][1]/len(self.fingerps)
        return dict(
            forma   = float(np.mean(self.formas)),
            convex  = float(np.mean(self.convexs)),
            aspecto = float(np.mean(self.aspectos)),
            simbolo = float(np.mean(self.simbolos)),
            persist = e_persist,
            estab   = float(e_estab),
            fingerprint = float(e_fp),
        )

    def total(self) -> float:
        pesos = PESOS_SEMAFORO if self.det["class_name"]=="Semaforo" \
                else PESOS_PLACA
        ev = self.evidencias()
        return sum(pesos[k]*ev[k] for k in pesos)


class PGOM:
    """Mantém as fichas e decide PROMOÇÃO por soma de evidências."""
    def __init__(self):
        self.fichas: list[Ficha] = []
        self.stats = dict(vistos=0, promovidos=0)

    def update(self, dets: list) -> list:
        geo    = [d for d in dets if d.get("geo")]
        outros = [d for d in dets if not d.get("geo")]   # YOLO/COCO
        self.stats["vistos"] += len(geo)

        usados = set()
        casadas = {}
        for d in geo:
            melhor, dist_m = None, PGOM_MATCH_D
            for f in self.fichas:
                if id(f) in usados: continue
                dist = f.perto_de(d)
                if dist < dist_m: melhor, dist_m = f, dist
            if melhor is not None:
                melhor._push(d); usados.add(id(melhor))
                casadas[id(melhor)] = d
            else:
                nova = Ficha(d)
                self.fichas.append(nova)
                # Ficha nova TAMBÉM é avaliada já neste frame: quem
                # decide é a SOMA de evidências contra o limiar do
                # perfil — placas não passam na 1ª visita (persistência
                # pesa 20%), mas um semáforo estruturalmente válido sim.
                casadas[id(nova)] = d
                usados.add(id(nova))

        for f in self.fichas:
            if id(f) not in usados and f.vistos > 1:
                f.missed += 1
        # sumiu → ficha zera (o mundo real não some por 3 frames)
        self.fichas = [f for f in self.fichas
                       if f.missed <= PGOM_MAX_MISS]

        promovidos = []
        for f in self.fichas:
            d = casadas.get(id(f))
            if d is None: continue             # não visto neste frame
            tot = f.total()
            limiar = PGOM_PROMOVE_SEM \
                     if f.det["class_name"] == "Semaforo" \
                     else PGOM_PROMOVE
            if tot >= limiar:
                d = dict(d)
                d["evid"]  = f.evidencias()
                d["evtot"] = float(tot)
                promovidos.append(d)
                self.stats["promovidos"] += 1
        return promovidos + outros

PGOM_M = PGOM()


def score_final(det: dict, cnn_conf: float) -> float:
    """
    Score de decisão: a soma de evidências geométricas/temporais
    responde por 85%% e a CNN — confirmadora final — pelos 15%%.
    """
    ev = float(det.get("evtot", det.get("conf", 0.5)))
    return 0.85*ev + 0.15*float(cnn_conf)

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
        if frame_enhanced.ndim == 2:   # mono → 3 canais p/ o modelo
            frame_enhanced = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2BGR)
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
                # Filtros por classe (só em modo COCO)
                if self.coco_mode:
                    if max_c[i] < COCO_CONF_MIN.get(cls_name, YOLO_CONF):
                        continue
                    if (x2-x1)*(y2-y1) < COCO_AREA_MIN.get(cls_name, 0):
                        continue
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
        # Lê o tamanho de entrada DO MODELO (64, 96...) — mesma
        # lição do ONNX: nunca confiar em constante hardcoded.
        self.size = int(d["shape"][1])
        global CNN_SIZE
        CNN_SIZE = self.size
        q = "INT8" if self._q else "FP32"
        print(f"[CNN] {path} | {self.size}x{self.size} | {q}",
              flush=True)

    def predict(self, img_96: np.ndarray) -> np.ndarray:
        inp = img_96[np.newaxis].astype(np.float32)
        if self._q: inp = (inp*255).astype(np.uint8)
        self._interp.set_tensor(self._in, inp)
        self._interp.invoke()
        out = self._interp.get_tensor(self._out)[0]
        return out.astype(np.float32)/255.0 if self._q else out


def prep_mono(crop: np.ndarray) -> np.ndarray:
    """
    Preprocessing ÚNICO do crop — IDÊNTICO ao usado no treino
    (TRAIN_SIGN_CNN.py::prep_mono). Consistência treino↔inferência
    é o que garante a acurácia:
      resize 96 → gray → CLAHE → replica 3 canais → [0,1]
    """
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(crop, (CNN_SIZE, CNN_SIZE))
    g = _CLAHE.apply(g)
    return np.stack([g, g, g], axis=-1).astype(np.float32) / 255.0


def carregar_classes() -> list:
    """Lê models/classes.txt (gerado no treino) para nunca haver
    divergência entre índices do treino e do runtime."""
    p = os.path.join(os.path.dirname(YOLO_MODEL), "classes.txt")
    if os.path.exists(p):
        with open(p) as f:
            cls = [l.strip() for l in f if l.strip()]
        if cls:
            print(f"[CNN] classes.txt: {cls}", flush=True)
            return cls
    return CLASSES


class VerificadorPlaca:
    """
    ETAPA 1 — "Isso é REALMENTE uma placa?" (verificação binária,
    independente de qual classe seria). Separada da ETAPA 2
    ("qual placa?", que é a CNN). Três testes que uma placa real
    passa e um objeto qualquer de forma parecida NÃO passa:

      1. MARGEM de decisão da CNN (top1 - top2): uma placa real
         gera um pico claro; ruído gera distribuição achatada,
         com 1º e 2º lugares empatados. Isso é MUITO mais robusto
         que o MaxSoftmax sozinho — uma rede confia alto até em
         lixo, mas raramente confia alto em UMA classe só.
      2. ENTROPIA da distribuição: baixa = decisão limpa (placa);
         alta = incerteza espalhada (não-placa).
      3. Coerência com a GEOMETRIA: a forma detectada (octógono,
         triângulo...) tem que ser compatível com a classe que a
         CNN escolheu. Um octógono classificado como "Esquerda"
         é contradição → rejeita.
    """
    # Forma geométrica esperada por classe (compatibilidade)
    # Na pista só existem PARE (octógono) e semáforo. As demais
    # classes ficam listadas por segurança, mas não aparecem.
    FORMA_CLASSE = {
        "Stop": {"octogono"},
    }

    def __init__(self, margem=0.25, entropia_max=1.30):
        self.margem_min   = margem        # top1 - top2 mínimo
        self.entropia_max = entropia_max  # nats; ln(9)=2.20 é o máx.

    @staticmethod
    def _forma_geo(det) -> str:
        nv, circ = det.get("lados",0), det.get("circ",0)
        if 7 <= nv <= 9 and circ > 0.62: return "octogono"
        if nv == 3:                      return "triangulo"
        if nv >= 6 and circ > 0.70:      return "circulo"
        if nv in (4,5):                  return "retangulo"
        return "?"

    def e_placa(self, scores: np.ndarray, cls_nm: str,
                det: dict) -> tuple:
        """Retorna (aceita: bool, motivo: str, margem: float)."""
        ordenado = np.sort(scores)[::-1]
        top1 = float(ordenado[0])
        top2 = float(ordenado[1]) if len(ordenado) > 1 else 0.0
        margem = top1 - top2
        # entropia da distribuição
        p = np.clip(scores, 1e-9, 1.0)
        entropia = float(-(p*np.log(p)).sum())

        if margem < self.margem_min:
            return False, f"margem baixa {margem:.2f}", margem
        if entropia > self.entropia_max:
            return False, f"entropia alta {entropia:.2f}", margem
        # coerência forma×classe (só p/ classes com forma definida)
        esperadas = self.FORMA_CLASSE.get(cls_nm)
        if esperadas is not None:
            fg = self._forma_geo(det)
            if fg != "?" and fg not in esperadas:
                return False, f"forma {fg}!={cls_nm}", margem
        return True, "ok", margem


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
               cnn: "CNNClassifier | None", ood: "OODRejector | None",
               verif: "VerificadorPlaca | None" = None) -> list:
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
            lbl, conf_cnn = self._classificar(d, frame_enhanced, cnn, ood, verif)
            self.tracks[ti].atualizar(
                d["bbox"], d["class_name"], d["conf"], lbl, conf_cnn)

        # ── Incrementa missed nos não matchados ──────────────────
        for ti in unm_t2:
            self.tracks[ti].missed += 1

        # ── Cria novos tracks para dets de alta conf não matchadas
        for di in unm_d_hi:
            d   = dets[di]
            lbl, conf_cnn = self._classificar(d, frame_enhanced, cnn, ood, verif)
            t = Track(d["bbox"], d["class_name"], d["conf"])
            if lbl: t.buf.append((lbl, conf_cnn))
            self.tracks.append(t)

        # ── Remove tracks mortas ─────────────────────────────────
        self.tracks = [t for t in self.tracks if t.missed <= TRACK_MAX_AGE]
        return self.tracks

    @staticmethod
    def _classificar(det, frame_enhanced, cnn, ood, verif=None):
        # COCO: classe do YOLO já é confiável — voto direto
        if det.get("coco", False):
            return det["class_name"], det["conf"]
        # Semáforo geométrico: forma basta; o ESTADO é lido depois
        # (posição do farol aceso) — não passa pela CNN
        if det.get("geo") and det["class_name"] == "Semaforo":
            return "Semaforo", score_final(det, det["conf"])
        if cnn is None: return None, 0.0
        x1,y1,x2,y2 = det["bbox"]
        crop = frame_enhanced[y1:y2, x1:x2]
        if crop.size==0 or (x2-x1)<8 or (y2-y1)<8: return None, 0.0
        scores = cnn.predict(prep_mono(crop))
        max_s  = float(scores.max())
        cls_nm = CNN_CLASSES[int(scores.argmax())] \
                 if int(scores.argmax()) < len(CNN_CLASSES) else None

        # ══ ETAPA 2 responde "qual placa?"; mas ANTES a ETAPA 1
        #    tem poder de veto: "isso é REALMENTE uma placa?" ══

        # 1a) A própria rede tem uma saída para "não é placa"
        #     (classe Fundo). Se venceu, encerra.
        if cls_nm in (None, "Fundo"): return None, 0.0

        # 1b) Verificador binário (margem + entropia + coerência
        #     forma×classe) — independente de qual classe seria.
        if verif is not None:
            ok, motivo, _ = verif.e_placa(scores, cls_nm, det)
            if not ok:
                if det.get("_dbg"):
                    print(f"[VERIF] rejeitado {cls_nm}: {motivo}",
                          flush=True)
                return None, 0.0

        # 1c) Limiar de confiança por classe (OOD clássico)
        if ood and not ood.aceitar(cls_nm, max_s): return None, 0.0

        # ══ Passou nas 3 barreiras → É placa. Score final combina
        #    evidência geométrica/temporal (PGOM) + CNN ══
        return cls_nm, score_final(det, max_s)

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
    out = cv2.cvtColor(frame_e, cv2.COLOR_GRAY2BGR) \
          if frame_e.ndim == 2 else frame_e.copy()
    h,w = out.shape[:2]; PW = 225

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
    if _sat_chk["mono"]:
        cv2.putText(pan,"CAM S/COR!",(120,14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,255),1)
    cor_missao = {"AGUARDANDO":(0,220,220),"RODANDO":(100,220,100),
                  "PARADO_SEM":(50,50,220),"ENTREGANDO":(220,180,0),
                  "FINALIZADO":(180,180,180)}.get(MISSAO.estado,(190,190,190))
    t(f"MISSAO: {MISSAO.estado}",1,cor_missao,0.36)
    t(f"Dest:{MISSAO.destino} {MISSAO.progresso}/{len(MISSAO.rota())} "
      f"Trk:{len(tracks)} Fichas:{len(PGOM_M.fichas)}",2)

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

def main(usar_camera=False, debug=False, auto=False, fast=False,
         usar_yolo=False, usar_canny=False, cam_idx=None):
    global _ser, _CLAHE, CNN_CLASSES, CAM_FLIP

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
    cnn = ood = verif = None
    if os.path.exists(CNN_MODEL):
        try:
            cnn=CNNClassifier(CNN_MODEL); ood=OODRejector(OOD_FILE)
            verif=VerificadorPlaca()
            CNN_CLASSES = carregar_classes()
        except Exception as e: print(f"[WARN] CNN: {e}", flush=True)
    else:
        print(f"[WARN] CNN não encontrada → execute: python TRAIN_SIGN_CNN.py",
              flush=True)

    # YOLO agora é OPCIONAL (--yolo). No feed mono, o detector
    # geométrico é o principal: mais rápido e sem alucinações.
    yolo = None; modo = "GEO"
    if usar_yolo and os.path.exists(YOLO_MODEL):
        try:   yolo=YOLODetector(YOLO_MODEL); modo="YOLO"
        except Exception as e: print(f"[WARN] YOLO: {e}", flush=True)
    elif usar_yolo:
        print(f"[WARN] --yolo pedido mas {YOLO_MODEL} não existe → GEO",
              flush=True)
    print(f"[DET] Detector principal: {modo}", flush=True)

    tracker = ByteTrackLite()

    if usar_camera:
        cap = abrir_camera(CAM_IDX if cam_idx is None else cam_idx)
    else:
        cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened(): print("[ERRO] Fonte de vídeo"); sys.exit(1)

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
        if ret and usar_camera and CAM_FLIP is not None:
            frame_raw = cv2.flip(frame_raw, CAM_FLIP)
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        fn += 1

        if _nav["cooldown"]>0: _nav["cooldown"]-=1
        em_acao = tick(_ser)

        if fn % SKIP == 0:
            # ── Sanity check da câmera (cor presente?) ────────────
            checar_camera(frame_raw)

            # ── Preprocessing principal ────────────────────────────
            frame_e = preprocessar(frame_raw)
            h, w    = frame_e.shape[:2]
            debug_thr = None

            # ── Localização ────────────────────────────────────────
            # GEOMÉTRICO é o principal (todo frame, ~ms).
            # YOLO é AUXILIAR: com --yolo, roda a cada YOLO_EVERY
            # frames e SÓ dentro do ROI dinâmico → registra o que
            # a geometria perder, sem derrubar o FPS.
            dets = detectar_geometrico(frame_e, usar_canny)
            det_modo = "GEO"
            if yolo and fn % YOLO_EVERY == 0:
                rx0, ry0, rx1, ry1 = ROI_DIN.janela(h, w)
                sub = frame_e[ry0:ry1, rx0:rx1]
                if sub.size:
                    for d in yolo.detectar(sub):
                        x1,y1,x2,y2 = d["bbox"]
                        d["bbox"] = (x1+rx0, y1+ry0, x2+rx0, y2+ry0)
                        dets.append(d)
                    det_modo = "GEO+YOLO"
            debug_thr = None
            if debug:
                _, debug_thr = cv2.threshold(
                    frame_e, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Filtra área mínima
            dets = [d for d in dets
                    if (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])
                    >= AREA_MIN_EXEC]

            # ── PGOM: promoção por múltiplas evidências ───────────
            # Nenhuma etapa isolada promove; só a SOMA ponderada
            # das evidências da ficha ≥ PGOM_PROMOVE chega à CNN.
            dets = PGOM_M.update(dets)

            # ── Tracker + CNN (apenas candidatos estáveis) ────────
            tracks = tracker.update(dets, frame_e, cnn, ood, verif)

            # Histórico do ROI dinâmico: posições dos tracks vivos
            for trk in tracks:
                if trk.state == "confirmed" or trk.age >= 2:
                    ROI_DIN.registrar(trk.bbox)

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
                    sem_cor = analisar_semaforo(frame_e[y1:y2, x1:x2])
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
        elif k==ord('f'):
            _ciclo = [None, 1, 0, -1]
            CAM_FLIP = _ciclo[(_ciclo.index(CAM_FLIP)+1) % 4]
            print(f"[CAM] flip = {CAM_FLIP} "
                  f"(None/1=horiz/0=vert/-1=180)", flush=True)
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
    else:
        cam_idx = None
        if "--cam-idx" in args:
            try:    cam_idx = int(args[args.index("--cam-idx")+1])
            except (IndexError, ValueError):
                print("[ERRO] use: --cam-idx N  (ex.: --cam-idx 1)")
                sys.exit(1)
        main(usar_camera="--cam" in args,
             debug="--debug" in args,
             auto="--auto" in args,
             fast="--fast" in args,
             usar_yolo="--yolo" in args,
             usar_canny="--canny" in args,
             cam_idx=cam_idx)