"""
================================================================
  CARRO AUTÔNOMO v5.1 — PIPELINE MONO GEOMÉTRICO DE ALTO FPS
================================================================
"""

import cv2
import numpy as np
import serial, serial.tools.list_ports
import time, sys, os, json
from collections import deque, Counter

VIDEO       = "./videoplayback.mp4"
CAM_IDX     = 1
CAM_FLIP    = 1
IMG_W       = 640

SERIAL_PORT = "COM3"
BAUD        = 115200

YOLO_MODEL  = "./models/sign_detector.onnx"
CNN_MODEL   = "./models/sign_classifier.tflite"
OOD_FILE    = "./models/ood_thresholds.json"

CNN_SIZE    = 96
OOD_DEFAULT = 0.55
YOLO_CONF   = 0.25

COCO_CONF_MIN = {"Stop": 0.28, "Semaforo": 0.40, "Carro": 0.55, "Pessoa": 0.62}
COCO_AREA_MIN = {"Pessoa": 2000, "Carro": 1500}
YOLO_NMS    = 0.45
YOLO_SIZE   = 640
YOLO_EVERY  = 5

CLASSES = ["Stop","Esquerda","Direita","SemRetorno","Verde","Cone","Carro","Pessoa","Fundo"]
CNN_CLASSES = CLASSES
NUM_CLASSES = len(CLASSES)
OBSTACLE_CLASSES = {"Cone","Carro","Pessoa"}

COCO_MAP = {11: "Stop", 0: "Pessoa", 2: "Carro", 7: "Carro", 5: "Carro", 3: "Carro", 9: "Semaforo"}

TRACK_IOU_HIGH = 0.30
TRACK_IOU_LOW  = 0.15
TRACK_MAX_AGE  = 10
CONF_HIGH_THR  = 0.45

VOTE_BUFFER   = 10
VOTE_MIN_DETS = 5
VOTE_FRAC     = 0.60

COOLDOWN_F    = 60
AREA_MIN_EXEC = 800

ROI_Y0 = 0.05
ROI_Y1 = 0.92

AREA_CNT_MIN = 500
AREA_CNT_MAX = 55_000

# ── PAINEL LOCAL — abre só de escanear o QR, sem instalar nada ──
# O laptop (ou o roteador de viagem) cria uma rede Wi-Fi local e o
# script sobe um servidor HTTP nela. Não usa internet — o QR guarda
# a URL local; o celular escaneia, o navegador abre sozinho.
PAINEL_PORTA = 8000
CAR_ID       = "171"

# ── PERFIL DE REDE — troque SÓ ESTA LINHA para alternar entre os
# dois cenários. Nada mais no código muda.
#
#   "notebook" → Hotspot Móvel do próprio Windows.
#                Configurações → Rede e Internet → Ponto de acesso
#                móvel → Editar → mesmo SSID/senha daqui embaixo.
#                Gateway sempre 192.168.137.1, em qualquer notebook.
#
#   "tplink"   → roteador de viagem TP-Link TL-MR3020 (ou similar),
#                interruptor físico no modo AP. Configure o SSID/
#                senha pela página 192.168.0.254 (ou tplinkwifi.net)
#                uma vez; gateway de fábrica é 192.168.0.1.
#
# Nos dois casos o script descobre o IP real sozinho a cada boot —
# só o "gateway_chute" abaixo muda pra achar a interface certa mais
# rápido quando o notebook tem mais de uma rede ativa ao mesmo tempo.
REDE_PERFIL = "notebook"

_PERFIS_REDE = {
    "tplink":   dict(ssid="171Garage", senha="garagem171",
                     gateway_chute="192.168.0.1"),
}
if REDE_PERFIL not in _PERFIS_REDE:
    raise SystemExit(f"[CFG] REDE_PERFIL={REDE_PERFIL!r} inválido — "
                     f"use um de {list(_PERFIS_REDE)}")
WIFI_SSID = _PERFIS_REDE[REDE_PERFIL]["ssid"]
WIFI_PASS = _PERFIS_REDE[REDE_PERFIL]["senha"]



ESPERA_QR_S      = 7.0    # segundos parado antes de mostrar o QR
ESPERA_ENTREGA_S = 7.0    # segundos parado entregando o pacote
BUZ_PARTIDA_S    = 0.4    # beep curto ao sair
BUZ_ENTREGA_S    = 1.2    # beep longo ao entregar

# ── PLACAS DE LETRA A/B/C (pontos de entrega) ──────────────────
# Placa CIRCULAR branca, anel preto, letra preta ao centro.
# Reconhecimento por TEMPLATE MATCHING: os moldes são gerados com
# a MESMA fonte usada na impressão, então o casamento é direto.
# Sem OCR: Tesseract acertou 73% e levou 123ms; template acerta
# ~100% em 0.4ms (medido com desfoque, rotação e escala da pista).
LETRA_FONTE      = cv2.FONT_HERSHEY_SIMPLEX
LETRA_ESCALA     = 4.5     # usada na geração do molde e da impressão
LETRA_ESPESSURA  = 12
LETRAS_PONTOS    = ["A", "B", "C"]

# ── PONTOS A/B/C ───────────────────────────────────────────────
# "aruco": marcadores binários ArUco (IDs 0/1/2 → A/B/C) — modo atual
# "letras": placa circular com letra (template matching) — para depois
PONTO_MODO        = "letras"
ARUCO_DICT        = "DICT_4X4_50"
ARUCO_ID_TO_PONTO = {0: "A", 1: "B", 2: "C"}

AREA_MIN_LEITURA = 400     # contorno: a partir daqui já tenta LER a letra
AREA_MIN_CHEGADA = 9000    # bbox: só aqui é "cheguei no ponto" (calibrar)
# Leitura da letra (o portão de forma está em FORMA_* mais abaixo).
# Limiares MEDIDOS (placa real x distrator circular aleatório):
#   score: real ≥0.720 · distrator ≤0.675  ·  tinta: real 0.33–0.47
LETRA_SCORE_MIN  = 0.50    # casamento mínimo com o molde NORMALIZADO
LETRA_SCORE_FORTE= 0.62    # acima disto a letra vence o veto de forma
LETRA_MARGEM_MIN = 0.06    # vantagem sobre a 2ª letra (o "C" fica em 0.08)
LETRA_TINTA_MIN  = 0.10    # fração de preto no miolo — uma letra ocupa
LETRA_TINTA_MAX  = 0.55    # ~1/3; disco liso fica fora da faixa
LETRA_MAX_CIRC   = 8       # teto de leituras por frame em círculos
LETRA_MAX_DUVIDA = 4       # teto de leituras em formas duvidosas ("?")
LETRA_VOTOS_N    = 5       # janela de votação temporal
LETRA_VOTOS_MIN  = 3       # confirmações necessárias dentro da janela
# Peneiras de forma da LETRA recortada (não da placa):
LETRA_AR_MIN     = 0.45    # largura/altura da letra
LETRA_AR_MAX     = 1.25
LETRA_PREENCH_MIN= 0.30    # fração de tinta dentro da bbox da letra
LETRA_PREENCH_MAX= 0.85

K_SHARP = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
K3 = np.ones((3,3), np.uint8)
K5 = np.ones((5,5), np.uint8)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Só existem obstáculos e alvos de parada — nada de direcionais.
CLASS_TO_ACTION = {"Cone":"OBSTACLE","Carro":"OBSTACLE","Pessoa":"OBSTACLE"}

# CRUZEIRO alto = "o carro deve ser veloz". As paradas têm dur=0.0:
# quem decide quando sair é a MISSÃO, não um timer da ação.
MOT_CRUZEIRO = 90
ACOES = {
    "STOP":     dict(mot=0,            srv=127, buz=0, led=1, brk=1, dir=0, dur=0.0),
    "OBSTACLE": dict(mot=0,            srv=127, buz=0, led=1, brk=1, dir=0, dur=0.0),
    "STRAIGHT": dict(mot=MOT_CRUZEIRO, srv=127, buz=0, led=0, brk=0, dir=3, dur=0.0),
}
COR_CLASSE = {
    "Stop":(50,50,220),"SemRetorno":(20,20,180),"Esquerda":(220,120,0),
    "Direita":(0,120,220),"Verde":(50,220,50),"Cone":(0,165,255),
    "Carro":(200,0,200),"Pessoa":(0,0,255),"Semaforo":(0,220,220),
}

def estado_semaforo_mono(crop_gray):
    if crop_gray is None or crop_gray.size == 0: return None
    if crop_gray.ndim == 3: crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(crop_gray, (30, 90))
    tercos = [float(g[0:30].mean()), float(g[30:60].mean()), float(g[60:90].mean())]
    i = int(np.argmax(tercos))
    contraste = max(tercos) - sorted(tercos)[1]
    if contraste < 8: return None
    return ["vermelho", "amarelo", "verde"][i]

def analisar_semaforo(crop):
    if crop is None or crop.size == 0: return None
    if crop.ndim == 2 or _sat_chk["mono"]: return estado_semaforo_mono(crop)
    return cor_semaforo(crop)

def cor_semaforo(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0: return None
    hsv = cv2.cvtColor(cv2.resize(crop_bgr,(40,80)), cv2.COLOR_BGR2HSV)
    total = hsv.shape[0]*hsv.shape[1]
    verde = cv2.inRange(hsv, (40,80,120), (90,255,255))
    vermelho = cv2.inRange(hsv, (0,80,120), (10,255,255)) | cv2.inRange(hsv, (170,80,120), (180,255,255))
    amarelo = cv2.inRange(hsv, (18,80,120), (35,255,255))
    fr_v, fr_r, fr_a = verde.sum()/255/total, vermelho.sum()/255/total, amarelo.sum()/255/total
    melhor = max([("verde",fr_v),("vermelho",fr_r),("amarelo",fr_a)], key=lambda x: x[1])
    return melhor[0] if melhor[1] > 0.04 else None

# ================================================================
#  MISSION ENGINE — "o que isso significa?"
#  ─────────────────────────────────────────────────────────────
#  Separação estrita de responsabilidades:
#    VISÃO    = o que existe no frame       (detectar_*, PGOM, tracker)
#    MISSÃO   = o que isso significa agora  (esta classe)
#    CONTROLE = o que o motor deve fazer    (executar/enviar/tick)
#
#  REGRAS ABSOLUTAS (têm prioridade sobre qualquer rota):
#      semáforo vermelho → PARAR
#      placa PARE        → PARAR
#      obstáculo         → PARAR
#
#  REGRAS DE MISSÃO (marcador do ponto visto):
#      A: se destino=A → entregar; senão → continuar
#      B: se destino=B → entregar; senão → continuar
#      C: se destino=C → entregar; senão → continuar
#
#  CICLO DE ENTREGA:
#      parou → 7s → QR na tela → app externo publica destino (MQTT)
#      → segue (respeitando prioridades de parada) → vê marcador do
#      destino → entrega: buzzer longo + 7s parado → buzzer curto de
#      saída → volta a rodar. Telemetria enviada em cada transição.
# ================================================================

class Missao:
    """
    ESTADOS:
      AGUARDANDO       parado, QR no canto, sem retirada definida
      IND_RETIRADA     rodando até o ponto de retirada
      RETIRANDO        parado 7s pegando o pacote
      AGUARDA_ENTREGA  parado na retirada, esperando escolha da entrega
                       (o MESMO QR/sessão continua valendo)
      IND_ENTREGA      rodando até o ponto de entrega
      ENTREGANDO       parado 7s entregando
      LIVRE            missão cumprida — CONTINUA ANDANDO em cruzeiro,
                       aceitando nova missão a qualquer momento
      PARADO_PARE / PARADO_SEM / PARADO_OBST   regras absolutas;
                       guardam o estado de retorno para retomar a rota
      DESLIGADO        motores cortados por comando (LIGAR religa)
    """
    RODANDO_STATES = ("IND_RETIRADA", "IND_ENTREGA", "LIVRE")

    def __init__(self):
        self.estado    = "AGUARDANDO"
        self.retirada  = None
        self.entrega   = None
        self.fase      = "retirada"      # o que falta cumprir
        self.retorno   = "LIVRE"         # p/ onde voltar após parada absoluta
        self.t_estado  = time.monotonic()
        self.entregas  = 0
        self.qr_img    = None      # QR 2: abre o painel (URL local)
        self.qr_wifi   = None      # QR 1: entra na rede (sem senha digitada)
        self.sessao    = None

    def nova_sessao(self):
        """UMA sessão por ligada do carro. Dois QRs ficam no canto do
        display o tempo todo:
          1) Wi-Fi  — a Câmera do celular reconhece o formato WIFI:
             nativamente e oferece 'Entrar na rede?' sem app nenhum.
          2) Painel — depois de conectado, abre a URL local com os
             botões de retirada/entrega.
        Não existe um único QR que faça as duas coisas numa ação só
        (são padrões diferentes); dois QRs pequenos lado a lado é o
        caminho mais confiável nos dois sistemas (iOS e Android)."""
        import secrets
        self.sessao = secrets.token_urlsafe(4)
        url = BT.url if (BT is not None and getattr(BT, "url", None)) \
              else f"http://{PainelLocal._meu_ip()}:{PAINEL_PORTA}/"
        self.qr_wifi = gerar_qr_wifi(WIFI_SSID, WIFI_PASS, lado=120)
        self.qr_img  = gerar_qr(url, lado=120)
        print(f"[QR] sessão {self.sessao} — Wi-Fi '{WIFI_SSID}' → painel {url}",
              flush=True)

    def set_estado(self, novo):
        if novo != self.estado:
            print(f"[MISSAO] {self.estado} → {novo}", flush=True)
            self.estado = novo
            self.t_estado = time.monotonic()

    def tempo_no_estado(self):
        return time.monotonic() - self.t_estado

    def parado_por_regra(self):
        return self.estado in ("PARADO_PARE", "PARADO_SEM", "PARADO_OBST")

    def alvo(self):
        """Qual marcador interessa AGORA (o resto é ignorado)."""
        if self.fase == "retirada" and self.retirada: return self.retirada
        if self.fase == "entrega"  and self.entrega:  return self.entrega
        return None

    def rota_txt(self):
        r = self.retirada or "?"
        e = self.entrega  or "?"
        return f"{r}→{e}" + (f" [falta {self.fase}]" if self.alvo() or self.fase else "")

    @staticmethod
    def regra_absoluta(percep):
        """obstáculo > vermelho > PARE — prioridade sobre a rota."""
        if percep.get("obstaculo"):              return "PARADO_OBST"
        if percep.get("semaforo") == "vermelho": return "PARADO_SEM"
        if percep.get("pare"):                   return "PARADO_PARE"
        return None

    def status(self):
        """As cinco perguntas do protocolo, numa linha só."""
        posso   = "sim" if self.retirada else "nao (envie R:<ponto>)"
        chegou  = "sim" if self.estado in ("RETIRANDO","ENTREGANDO") else "nao"
        parado  = "sim" if (self.parado_por_regra() or
                            self.estado in ("AGUARDANDO","AGUARDA_ENTREGA",
                                            "RETIRANDO","ENTREGANDO","DESLIGADO")) else "nao"
        return (f"ST;estado={self.estado};rota={self.retirada or '-'}"
                f">{self.entrega or '-'};fase={self.fase};posso_iniciar={posso};"
                f"cheguei={chegou};parado={parado};entregas={self.entregas}")

MISSAO = Missao()
COR_ACAO = {"STOP":(50,50,220),"OBSTACLE":(0,0,200),"STRAIGHT":(50,220,50)}
CMD = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
_nav = dict(cooldown=0, ultimo=None, acao_label=None, acao_t=0.0, acao_dur=0.0)
_ser = None

def _abrir_idx(idx):
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
    nome = "DSHOW" if backend == cv2.CAP_DSHOW else "V4L2"
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened(): cap = cv2.VideoCapture(idx); nome = "AUTO"
    if not cap.isOpened(): return None, nome
    ret, _ = cap.read()
    if not ret: cap.release(); return None, nome
    return cap, nome

def abrir_camera(idx):
    cap, nome = _abrir_idx(idx)
    if cap is None:
        print(f"[CAM] índice {idx} não respondeu — varrendo 0..5...", flush=True)
        for alt in range(6):
            if alt == idx: continue
            cap, nome = _abrir_idx(alt)
            if cap is not None:
                print(f"[CAM] usando índice {alt}", flush=True); idx = alt; break
    if cap is None:
        print("[CAM][ERRO] Nenhuma câmera respondeu.", flush=True); sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fstr = "".join(chr((fourcc >> 8*i) & 0xFF) for i in range(4))
    print(f"[CAM] idx={idx} backend={nome} fourcc={fstr} {cap.get(3):.0f}x{cap.get(4):.0f}@{cap.get(5):.0f}fps", flush=True)
    return cap

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
_ultimo_ponto = [None]   # última placa logada (evita repetir a linha)

def enviar(cmd, ser):
    _seq["n"] += 1
    spd = 0 if cmd["mot"]==0 else (1 if cmd["mot"]<50 else 2)
    j = (f'{{"seq":{_seq["n"]},"mot":{cmd["mot"]},"srv":{cmd["srv"]},'
         f'"buz":{cmd["buz"]},"led":{cmd["led"]},"brk":{cmd["brk"]},"dir":{cmd["dir"]},"spd":{spd}}}')
    print(f"[CMD] {j}", flush=True)
    if ser:
        try:
            ser.write((j+"\n").encode())
            _seq["pendente"] = _seq["n"]; _seq["t_envio"] = time.monotonic()
        except Exception: pass

def ler_serial(ser):
    msgs = []
    if not ser: return msgs
    try:
        while ser.in_waiting:
            linha = ser.readline().decode(errors="ignore").strip()
            if not linha: continue
            try:
                m = json.loads(linha); msgs.append(m)
                if "ack" in m and m["ack"] == _seq["pendente"]: _seq["pendente"] = None
            except json.JSONDecodeError: pass
    except Exception: pass
    if (_seq["pendente"] is not None and time.monotonic() - _seq["t_envio"] > 0.2):
        print(f"[SER] Retransmitindo seq={_seq['pendente']}", flush=True)
        _seq["pendente"] = None; enviar(CMD, ser)
    return msgs

_sat_chk = dict(fn=0, avisado=False, mono=False)

def checar_camera(frame_bgr):
    _sat_chk["fn"] += 1
    if _sat_chk["fn"] % 90 != 1: return
    hsv = cv2.cvtColor(cv2.resize(frame_bgr,(160,90)), cv2.COLOR_BGR2HSV)
    sat = float(hsv[:,:,1].mean())
    _sat_chk["mono"] = sat < 5.0
    if _sat_chk["mono"] and not _sat_chk["avisado"]:
        _sat_chk["avisado"] = True
        print("[CAMERA] ⚠ VÍDEO SEM COR (saturação ~0)!", flush=True)

def preprocessar(frame_bgr):
    h0, w0 = frame_bgr.shape[:2]
    if w0 != IMG_W:
        scale = IMG_W / w0
        frame_bgr = cv2.resize(frame_bgr, (IMG_W, int(h0*scale)), interpolation=cv2.INTER_LINEAR)
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)

GEO_AREA_MIN = 400
GEO_AREA_FRAC = 0.25
GEO_MAX_CANDS = 10
ROI_FULL_EVERY = 15
ROI_EXPAND = 1.8

def segmentar(gray, usar_canny=False):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr_esc = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    thr_cla = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    _, otsu_esc = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    saidas = []
    for t in (thr_esc, thr_cla, otsu_esc):
        t = cv2.morphologyEx(t, cv2.MORPH_OPEN, K3)
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, K5)
        if usar_canny:
            t = cv2.Canny(t, 50, 150); t = cv2.dilate(t, K3)
        saidas.append(t)
    return saidas

class ROIDinamico:
    def __init__(self):
        self.hist = deque(maxlen=12); self.fn = 0
    def registrar(self, bbox): self.hist.append(bbox)
    def janela(self, h, w):
        self.fn += 1
        if not self.hist or self.fn % ROI_FULL_EVERY == 0:
            return (0, int(h*ROI_Y0), w, int(h*ROI_Y1))
        xs0 = min(b[0] for b in self.hist); ys0 = min(b[1] for b in self.hist)
        xs1 = max(b[2] for b in self.hist); ys1 = max(b[3] for b in self.hist)
        cx, cy = (xs0+xs1)/2, (ys0+ys1)/2
        bw = max(xs1-xs0, 80) * ROI_EXPAND; bh = max(ys1-ys0, 80) * ROI_EXPAND
        x0 = int(max(0, cx-bw)); x1 = int(min(w, cx+bw))
        y0 = int(max(h*ROI_Y0, cy-bh)); y1 = int(min(h*ROI_Y1, cy+bh))
        if x1-x0 < 64 or y1-y0 < 64: return (0, int(h*ROI_Y0), w, int(h*ROI_Y1))
        return (x0, y0, x1, y1)

ROI_DIN = ROIDinamico()

def analisar_farois(crop_gray):
    if crop_gray.size == 0: return 0, False
    g = cv2.resize(crop_gray, (36, 100))
    circulos = []
    for thr_img in (cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
                    cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]):
        cnts,_ = cv2.findContours(thr_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 40 or a > 1200: continue
            x,y,w,h = cv2.boundingRect(c)
            if not (0.6 < w/max(h,1) < 1.7): continue
            circ = 4*np.pi*a/max(cv2.arcLength(c,True)**2,1)
            if circ > 0.55: circulos.append((x + w/2, y + h/2))
    if len(circulos) < 2: return len(circulos), False
    circulos.sort(key=lambda p: p[1])
    empilhados = [circulos[0]]
    for p in circulos[1:]:
        if p[1] - empilhados[-1][1] > 15: empilhados.append(p)
    n = len(empilhados)
    if n < 2: return n, False
    xs = [p[0] for p in empilhados]
    alinhado = (max(xs) - min(xs)) < 12
    return n, alinhado

def _tem_farois(crop_gray):
    n, alinhado = analisar_farois(crop_gray)
    return 2 <= n <= 3 and alinhado

def _tem_simbolo(crop_gray):
    if crop_gray.size == 0: return False
    g = cv2.resize(crop_gray, (48, 48))
    miolo = g[10:38, 10:38]
    _, t = cv2.threshold(miolo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f = float(t.mean()) / 255.0
    minoria = min(f, 1.0 - f)
    return minoria > 0.08

ROI_ESCURA_MIN = 12
ROI_ESCURA_MAX = 248
ROI_FOCO_MIN = 25.0
ROI_BORDA_MIN = 0.03
ROI_TEXTURA_MAX = 0.72
ROI_AREA_FRAC_MIN = 0.0008
ROI_AREA_FRAC_MAX = 0.25
_dbg_roi = dict(on=False, rej={})

def peneira_roi(crop_gray, area_rel, exige_simbolo=True):
    if crop_gray is None or crop_gray.size == 0: return False, "vazia"
    h, w = crop_gray.shape[:2]
    if h < 10 or w < 10: return False, "muito pequena (px)"
    if area_rel < ROI_AREA_FRAC_MIN: return False, "muito pequena"
    if area_rel > ROI_AREA_FRAC_MAX: return False, "muito grande"
    m = float(crop_gray.mean())
    if m < ROI_ESCURA_MIN: return False, "muito escura"
    if m > ROI_ESCURA_MAX: return False, "muito clara/estourada"
    g = cv2.resize(crop_gray, (48, 48))
    foco = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if foco < ROI_FOCO_MIN: return False, f"desfocada ({foco:.0f})"
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    dens = float((mag > 60).mean())
    if dens < ROI_BORDA_MIN: return False, "sem bordas fortes"
    if dens > ROI_TEXTURA_MAX: return False, f"textura complexa ({dens:.2f})"
    if exige_simbolo and not _tem_simbolo(crop_gray): return False, "sem símbolo interno"
    return True, "ok"

# ── TESTES DE ROBUSTEZ GEOMÉTRICA (evidências novas do PGOM) ───
#  Perspectiva: os vértices do contorno realmente cabem num
#  octógono regular sob ALGUMA transformação de perspectiva? Um
#  blob de ruído com 7-9 vértices por acaso não cabe; um octógono
#  real visto de qualquer ângulo cabe (erro de reprojeção baixo).
#  Simetria: uma placa real é simétrica ao espelhar horizontalmente
#  (mesmo sob leve perspectiva); sombra/reflexo/objeto aleatório não.

def score_perspectiva_octogono(poly) -> float:
    """
    Ordena os vértices do contorno e de um octógono-molde por
    ângulo em torno do centróide, ajusta uma homografia entre eles
    e mede o erro de reprojeção. Erro baixo = forma explicável por
    um octógono real visto em perspectiva; erro alto = coincidência
    geométrica (ruído que só parcialmente lembra a forma).
    """
    if poly is None or len(poly) < 6: return 0.5
    pts = np.array(poly, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    pts_ord = pts[np.argsort(ang)]
    n = len(pts_ord)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/8
    template = (np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)*100 + 100)
    try:
        H, _ = cv2.findHomography(template, pts_ord, method=0)
        if H is None: return 0.2
        proj = cv2.perspectiveTransform(template.reshape(-1,1,2), H).reshape(-1,2)
    except Exception:
        return 0.2
    erro = float(np.mean(np.linalg.norm(proj-pts_ord, axis=1)))
    escala = float(np.linalg.norm(pts_ord.max(axis=0)-pts_ord.min(axis=0))) + 1e-6
    erro_rel = erro/escala
    return float(np.clip(1.0 - erro_rel*4.0, 0, 1))


def score_simetria(crop_gray) -> float:
    """Compara o crop com seu espelho horizontal — placas reais
    são simétricas; ruído/sombra/objeto aleatório não."""
    if crop_gray is None or crop_gray.size == 0: return 0.5
    g = cv2.resize(crop_gray, (48, 48)).astype(np.float32)
    flip = cv2.flip(g, 1)
    diff = float(np.mean(np.abs(g-flip))) / 255.0
    return float(np.clip(1.0 - diff*2.2, 0, 1))


# ================================================================
#  [2] PLACAS A/B/C — geometria + template matching, ZERO OCR
#      Cada ponto de entrega tem uma placa quadrada impressa com
#      a letra A, B ou C. O reconhecimento tem três estágios:
#        1. acha o quadrado (contorno, 4 vértices, aspecto ~1)
#        2. casa o miolo com os moldes A/B/C (matchTemplate)
#        3. confirma por votação temporal (3 de 5 frames)
#      Sem CNN e sem OCR — determinístico e ~0.4ms por frame.
# ================================================================

# ================================================================
#  CLASSIFICADOR DE FORMA — árbitro único de toda a visão
#  ─────────────────────────────────────────────────────────────
#      CÍRCULO  → placa de delivery (A/B/C)
#      OCTÓGONO → placa PARE
#
#  Por que não basta contar vértices: um círculo aproximado por
#  approxPolyDP com eps=0.02*perímetro devolve EXATAMENTE 8 lados,
#  igual ao octógono (medido). Circularidade e preenchimento também
#  se sobrepõem sob desfoque. O que separa de verdade é o 8º
#  HARMÔNICO do raio: o octógono tem 8 lóbulos periódicos, o
#  círculo não tem nenhum.
#      medido:  círculo  0.0033–0.0212
#               octógono 0.0266–0.0335   (estável até 22° de giro)
#  Entre as duas faixas há uma ZONA MORTA: ali a função devolve
#  "?" em vez de chutar — melhor não classificar do que errar.
# ================================================================

# Portão de entrada LARGO: a placa chega inclinada, borrada, lavada.
FORMA_AR_MIN     = 0.55    # inclinada de lado achata a bbox
FORMA_AR_MAX     = 1.80
FORMA_CIRC_MIN   = 0.68    # redondo "o suficiente"
FORMA_CONVEX_MIN = 0.85    # descarta contorno recortado/irregular
# Decisão entre as duas classes (8º harmônico do raio):
FORMA_H8_CIRCULO = 0.022   # ≤ isto → círculo
FORMA_H8_OCTOG   = 0.025   # ≥ isto → octógono   (entre os dois: "?")
FORMA_CIRC_OCTOG = 0.85    # octógono só é declarado com contorno bem
                           # formado (senão folha quadrada vira PARE)


def harmonico_8(c, N=64):
    """Força do 8º harmônico do raio r(θ). Exige contorno obtido com
    CHAIN_APPROX_NONE — o SIMPLE guarda só os cantos e a medida perde
    o sentido."""
    if c is None or len(c) < 16: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    p  = c.reshape(-1, 2)
    if len(p) > 360:                    # subamostra: 360 pontos bastam
        p = p[::max(1, len(p)//360)]
    p  = p.astype(np.float64)
    th = np.arctan2(p[:,1]-cy, p[:,0]-cx)
    r  = np.hypot(p[:,0]-cx, p[:,1]-cy)
    o  = np.argsort(th); th, r = th[o], r[o]
    grade = np.linspace(-np.pi, np.pi, N, endpoint=False)
    rr = np.interp(grade, th, r, period=2*np.pi)
    med = rr.mean()
    if med <= 1e-6: return None
    F = np.abs(np.fft.rfft(rr - med))
    return float(F[8] / (med * N / 2))


def classificar_forma(c, bw, bh, peri=None, area=None, circ=None):
    """-> 'circulo' | 'octogono' | '?'   ÁRBITRO ÚNICO da visão.

    O portão de entrada é LARGO de propósito: a placa pode chegar
    inclinada (vira elipse), desfocada, lavada ou meio tapada. Quem
    decide entre as duas classes é o 8º HARMÔNICO do raio, não a
    contagem de vértices — um círculo também devolve 8 lados no
    approxPolyDP, e era essa a origem do conflito PARE↔delivery.

    Entre as duas faixas há uma ZONA MORTA que devolve '?'. Nela o
    candidato ainda pode virar delivery, desde que a LETRA case com
    folga: conteúdo vence forma duvidosa.
    """
    if peri is None: peri = cv2.arcLength(c, True)
    if peri <= 0: return "?"
    if area is None: area = cv2.contourArea(c)
    ar = bw / max(bh, 1)
    if not (FORMA_AR_MIN <= ar <= FORMA_AR_MAX): return "?"
    if circ is None: circ = 4*np.pi*area / (peri*peri)
    if circ < FORMA_CIRC_MIN: return "?"
    convex = area / max(cv2.contourArea(cv2.convexHull(c)), 1)
    if convex < FORMA_CONVEX_MIN: return "?"
    h8 = harmonico_8(c)
    if h8 is None: return "?"
    # o octógono só é declarado com contorno bem formado — senão uma
    # folha quadrada meio arredondada viraria PARE
    if h8 >= FORMA_H8_OCTOG and circ >= FORMA_CIRC_OCTOG: return "octogono"
    if h8 <= FORMA_H8_CIRCULO: return "circulo"
    return "?"


def detectar_geometrico(gray, usar_canny=False):
    """VARREDURA ÚNICA:

        segmentação → contornos → classificar_forma
                            │
              ┌─────────────┴─────────────┐
           OCTÓGONO                    CÍRCULO
              │                           │
          candidato STOP            procurar letra
        (perspectiva, simetria)      no miolo → A/B/C
              │                           │
              └──────► PGOM / votação ◄───┘

    Devolve candidatos com class_name em {'Stop','Semaforo','Delivery'}.
    Os de Delivery trazem também 'ponto' (A/B/C) e 'score'.
    """
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = ROI_DIN.janela(h, w)
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0: return []

    cands = []
    n_letra = [0, 0]     # teto de leituras por frame: [circulo, "?"]
    for m in segmentar(sub, usar_canny):
        # CHAIN_APPROX_NONE: classificar_forma() precisa da borda inteira
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < GEO_AREA_MIN or a > GEO_AREA_FRAC*h*w: continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 20 or bh < 20: continue
            ar = bw / max(bh, 1); solid = a / max(bw*bh, 1)
            if solid < 0.35: continue
            peri = cv2.arcLength(c, True)
            circ = 4*np.pi*a / max(peri*peri, 1)
            crop_roi = sub[y:y+bh, x:x+bw]

            # ── SEMÁFORO: retângulo vertical com faróis empilhados ──
            if 0.25 < ar < 0.62 and bh > 50 and _tem_farois(crop_roi):
                hint, forma, ponto, score = "Semaforo", "?", None, 0.0

            else:
                forma = classificar_forma(c, bw, bh, peri, a, circ)

                # ── OCTÓGONO → PARE ────────────────────────────────
                if forma == "octogono":
                    hint, ponto, score = "Stop", None, 0.0

                # ── CÍRCULO (ou forma duvidosa) → DELIVERY ─────────
                elif forma in ("circulo", "?") and a >= AREA_MIN_LEITURA:
                    # ORÇAMENTOS SEPARADOS: o círculo é raro e é o alvo
                    # de verdade; a forma duvidosa ("?") é abundante num
                    # frame sujo e, com orçamento único, consumia todas
                    # as vagas ANTES de o círculo real ser lido.
                    i_or = 0 if forma == "circulo" else 1
                    n_letra[i_or] += 1
                    if n_letra[i_or] > (LETRA_MAX_CIRC if i_or == 0
                                        else LETRA_MAX_DUVIDA): continue
                    ponto, score = LEITOR.ler(sub, c, (x, y, bw, bh))
                    if ponto is None: continue
                    # forma duvidosa só passa com letra FORTE
                    if forma == "?" and score < LETRA_SCORE_FORTE: continue
                    hint, forma = "Delivery", "circulo"
                else:
                    continue

            # ── peneira comum (foco, brilho, bordas, símbolo) ──────
            area_rel = (bw*bh) / float(h*w)
            ok, motivo = peneira_roi(crop_roi, area_rel,
                                     exige_simbolo=(hint == "Stop"))
            if not ok:
                if usar_canny is False and _dbg_roi["on"]:
                    _dbg_roi["rej"][motivo] = _dbg_roi["rej"].get(motivo,0)+1
                continue

            gx, gy = x + x0, y + y0
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            nv = len(approx)
            convx = a / max(cv2.contourArea(cv2.convexHull(c)), 1)
            poly_l = approx.reshape(-1,2).tolist()
            if hint == "Stop":
                s_persp = score_perspectiva_octogono(poly_l)
                s_simet = score_simetria(crop_roi)
            else:
                s_persp = s_simet = 1.0
            d = {"bbox": (gx, gy, gx+bw, gy+bh), "class_name": hint, "class_id": -1,
                 "conf": float(solid), "geo": True, "lados": nv, "circ": float(circ),
                 "ar": float(ar), "area": float(a), "convex": float(convx),
                 "simbolo": True, "poly": poly_l, "forma": forma,
                 "persp": float(s_persp), "simet": float(s_simet)}
            if hint == "Delivery":
                d["ponto"] = ponto; d["score"] = float(score)
            cands.append(d)

    # dedup por IoU sobre a UNIÃO — nunca por contenção, senão a folha
    # branca (maior) engoliria o disco que está dentro dela
    cands.sort(key=lambda d: -d["circ"])
    keep = []
    for d in cands:
        ax1,ay1,ax2,ay2 = d["bbox"]
        dup = False
        for k in keep:
            bx1,by1,bx2,by2 = k["bbox"]
            ix = max(0, min(ax2,bx2)-max(ax1,bx1))
            iy = max(0, min(ay2,by2)-max(ay1,by1))
            inter = ix*iy
            uniao = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            if inter > 0.70*max(uniao,1): dup = True; break
        if not dup: keep.append(d)
    return keep[:GEO_MAX_CANDS]


# ================================================================
#  [2] PLACAS A/B/C — geometria + template matching, ZERO OCR
#      Cada ponto de entrega tem uma placa quadrada impressa com
#      a letra A, B ou C. O reconhecimento tem três estágios:
#        1. acha o quadrado (contorno, 4 vértices, aspecto ~1)
#        2. casa o miolo com os moldes A/B/C (matchTemplate)
#        3. confirma por votação temporal (3 de 5 frames)
#      Sem CNN e sem OCR — determinístico e ~0.4ms por frame.
# ================================================================

# ================================================================
#  CLASSIFICADOR DE FORMA — árbitro único de toda a visão
#  ─────────────────────────────────────────────────────────────
#      CÍRCULO  → placa de delivery (A/B/C)
#      OCTÓGONO → placa PARE
#
#  Por que não basta contar vértices: um círculo aproximado por
#  approxPolyDP com eps=0.02*perímetro devolve EXATAMENTE 8 lados,
#  igual ao octógono (medido). Circularidade e preenchimento também
#  se sobrepõem sob desfoque. O que separa de verdade é o 8º
#  HARMÔNICO do raio: o octógono tem 8 lóbulos periódicos, o
#  círculo não tem nenhum.
#      medido:  círculo  0.0033–0.0212
#               octógono 0.0266–0.0335   (estável até 22° de giro)
#  Entre as duas faixas há uma ZONA MORTA: ali a função devolve
#  "?" em vez de chutar — melhor não classificar do que errar.
# ================================================================

# Portão de entrada LARGO: a placa chega inclinada, borrada, lavada.
FORMA_AR_MIN     = 0.55    # inclinada de lado achata a bbox
FORMA_AR_MAX     = 1.80
FORMA_CIRC_MIN   = 0.68    # redondo "o suficiente"
FORMA_CONVEX_MIN = 0.85    # descarta contorno recortado/irregular
# Decisão entre as duas classes (8º harmônico do raio):
FORMA_H8_CIRCULO = 0.022   # ≤ isto → círculo
FORMA_H8_OCTOG   = 0.025   # ≥ isto → octógono   (entre os dois: "?")
FORMA_CIRC_OCTOG = 0.85    # octógono só é declarado com contorno bem
                           # formado (senão folha quadrada vira PARE)


def harmonico_8(c, N=64):
    """Força do 8º harmônico do raio r(θ). Exige contorno obtido com
    CHAIN_APPROX_NONE — o SIMPLE guarda só os cantos e a medida perde
    o sentido."""
    if c is None or len(c) < 16: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    p  = c.reshape(-1, 2)
    if len(p) > 360:                    # subamostra: 360 pontos bastam
        p = p[::max(1, len(p)//360)]
    p  = p.astype(np.float64)
    th = np.arctan2(p[:,1]-cy, p[:,0]-cx)
    r  = np.hypot(p[:,0]-cx, p[:,1]-cy)
    o  = np.argsort(th); th, r = th[o], r[o]
    grade = np.linspace(-np.pi, np.pi, N, endpoint=False)
    rr = np.interp(grade, th, r, period=2*np.pi)
    med = rr.mean()
    if med <= 1e-6: return None
    F = np.abs(np.fft.rfft(rr - med))
    return float(F[8] / (med * N / 2))


def _mascara_circular(n=64, raio_frac=0.62):
    """Máscara do miolo do disco — descarta o anel, que é igual
    nas três placas e só polui a leitura."""
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (n//2, n//2), int(n*raio_frac/2), 255, -1)
    return m


_MASC_CIRC = None

def _binarizar_miolo(crop, n=96):
    """Recorte da placa -> (binário do miolo, fração de tinta).
    Trata placa invertida (disco escuro, letra clara) sozinho."""
    global _MASC_CIRC
    if _MASC_CIRC is None or _MASC_CIRC.shape[0] != n:
        _MASC_CIRC = _mascara_circular(n)
    g = cv2.resize(crop, (n, n))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b[_MASC_CIRC == 0] = 255
    tinta = float((b[_MASC_CIRC > 0] == 0).mean())
    if tinta > 0.55:                     # placa invertida: letra clara
        b = cv2.bitwise_not(b); b[_MASC_CIRC == 0] = 255
        tinta = float((b[_MASC_CIRC > 0] == 0).mean())
    return b, tinta


def _recorte_tinta(b, n=48):
    """Recorta a letra na PRÓPRIA bounding box, recentra e estica
    num quadrado n x n. É isto que torna a leitura indiferente à
    fonte, ao tamanho da letra dentro do disco, à espessura do anel
    e a um desalinhamento do centro. -> (normalizado, recorte_cru)"""
    ys, xs = np.where(b == 0)
    if len(xs) < 30: return None, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1-x0) < 8 or (y1-y0) < 8: return None, None
    rec  = b[y0:y1+1, x0:x1+1]
    hh, ww = rec.shape
    lado = max(hh, ww)
    quad = np.full((lado, lado), 255, np.uint8)
    quad[(lado-hh)//2:(lado-hh)//2+hh, (lado-ww)//2:(lado-ww)//2+ww] = rec
    return cv2.resize(quad, (n, n), interpolation=cv2.INTER_AREA), rec


def _contar_buracos(rec):
    """Buracos fechados dentro da letra. É a assinatura mais estável
    que existe para este alfabeto, porque não depende de fonte:
        A = 1 buraco   ·   B = 2 buracos   ·   C = nenhum
    -> (letra_pela_topologia, n_buracos)"""
    r = cv2.copyMakeBorder(rec, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    cnts, hier = cv2.findContours(cv2.bitwise_not(r), cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not cnts: return None, -1
    hier = hier[0]
    a_letra = max(cv2.contourArea(c) for c in cnts)
    nb = sum(1 for i, c in enumerate(cnts)
             if hier[i][3] != -1 and cv2.contourArea(c) > 0.02*a_letra)
    return {0: "C", 1: "A", 2: "B"}.get(nb), nb


def _molde_letra(letra, lado=200, com_moldura=True):
    """Placa CIRCULAR branca com anel preto e a letra ao centro.
    Usada para gerar o molde e o PNG de impressão."""
    img = np.full((lado, lado), 255, np.uint8)
    k    = lado / 200.0
    c    = lado // 2
    if com_moldura:
        cv2.circle(img, (c, c), int(c*0.94), 0, max(2, int(round(5*k))))
    esc  = LETRA_ESCALA * k
    espl = max(1, int(round(LETRA_ESPESSURA * k)))
    (w, h), _ = cv2.getTextSize(letra, LETRA_FONTE, esc, espl)
    cv2.putText(img, letra, ((lado-w)//2, (lado+h)//2),
                LETRA_FONTE, esc, 0, espl)
    return img


class LeitorLetras:
    """Lê a letra DENTRO de um candidato circular já encontrado pelo
    detector geométrico. Não procura contornos — essa varredura é
    única e vive em detectar_geometrico().

        círculo → miolo → recorte da tinta → DOIS JUÍZES → persistência

    Os dois juízes precisam CONCORDAR:
      1. TOPOLOGIA  — quantos buracos a letra tem (A=1, B=2, C=0).
                      Não depende de fonte nenhuma.
      2. MOLDE      — casamento com o molde, mas sobre o recorte
                      NORMALIZADO, não sobre a placa inteira.
    Um sozinho não basta: a topologia confunde 'O' com 'A' (ambos
    têm 1 buraco) e o molde sozinho é refém da fonte impressa.
    Juntos, medido: 108/108 acertos nas placas reais (90–320px,
    desfoque até k=13, perspectiva até 40°) e 11 de 13 distratores
    rejeitados.
    """

    def __init__(self, debug=False):
        self.moldes = {}
        for L in LETRAS_PONTOS:
            b, _ = _binarizar_miolo(_molde_letra(L, 200))
            self.moldes[L] = _recorte_tinta(b)[0]
        self.votos = deque(maxlen=LETRA_VOTOS_N)
        self.debug = debug
        self._fn   = 0
        print(f"[ABC] Leitor de placas circulares — letras {LETRAS_PONTOS} "
              f"(topologia + molde normalizado)", flush=True)

    def ler(self, sub, c, bbox):
        """-> (letra, score). Devolve (None, 0.0) e diz o porquê no
        debug — inclusive quando reprova no filtro barato de tinta,
        que era exatamente onde o debug antigo ficava mudo."""
        x, y, bw, bh = bbox
        cru = sub[y:y+bh, x:x+bw]
        if cru.size == 0: return None, 0.0

        b, tinta = _binarizar_miolo(cru)
        letra = None; s1 = mg = 0.0; nb = -1

        if not (LETRA_TINTA_MIN < tinta < LETRA_TINTA_MAX):
            motivo = f"tinta {tinta:.2f} fora de {LETRA_TINTA_MIN}-{LETRA_TINTA_MAX}"
        else:
            norm, rec = _recorte_tinta(b)
            if norm is None:
                motivo = "tinta insuficiente"
            else:
                hh, ww = rec.shape
                ar  = ww / max(hh, 1)
                pre = float((rec == 0).mean())
                if not (LETRA_AR_MIN < ar < LETRA_AR_MAX):
                    motivo = f"proporção da letra {ar:.2f}"
                elif not (LETRA_PREENCH_MIN < pre < LETRA_PREENCH_MAX):
                    motivo = f"preenchimento {pre:.2f}"
                else:
                    L_topo, nb = _contar_buracos(rec)
                    notas = sorted(((float(cv2.matchTemplate(
                                norm, mo, cv2.TM_CCOEFF_NORMED).max()), L)
                                for L, mo in self.moldes.items()), reverse=True)
                    (s1, L_mol), (s2, _) = notas[0], notas[1]
                    mg = s1 - s2
                    if L_topo is None:        motivo = f"{nb} buracos — não é A/B/C"
                    elif L_topo != L_mol:     motivo = f"juízes discordam (topo={L_topo} molde={L_mol})"
                    elif s1 < LETRA_SCORE_MIN:   motivo = f"score {s1:.2f}"
                    elif mg < LETRA_MARGEM_MIN:  motivo = f"margem {mg:.2f}"
                    else:                     letra, motivo = L_topo, "ok"

        if self.debug:
            self._fn += 1
            if self._fn % 12 == 1:
                print(f"[ABC?] {bw}x{bh} tinta={tinta:.2f} buracos={nb} "
                      f"score={s1:.2f} margem={mg:.2f} -> {letra or motivo}",
                      flush=True)
        return letra, s1

    def votar(self, entregas):
        """Persistência: confirma a letra com LETRA_VOTOS_MIN de
        LETRA_VOTOS_N frames. -> dict do candidato ou None."""
        melhor = max(entregas, key=lambda d: d["area"]) if entregas else None
        self.votos.append(melhor["ponto"] if melhor else None)
        if melhor is None: return None
        if list(self.votos).count(melhor["ponto"]) < LETRA_VOTOS_MIN: return None
        return melhor


LEITOR = LeitorLetras.__new__(LeitorLetras)   # substituído em main()


class DetectorABC:
    """Marcadores ArUco (IDs 0/1/2 → A/B/C) — modo binário atual.
    Zero CNN, detecção determinística. Interface:
    detectar(gray) -> [{'ponto','bbox','centro','area'}]"""

    def __init__(self):
        self.ok = False
        try:
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
            try:   # OpenCV >= 4.7
                self._det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
                self._novo = True
            except AttributeError:
                self._dict = d
                self._par  = cv2.aruco.DetectorParameters_create()
                self._novo = False
            self.ok = True
            print(f"[ABC] ArUco {ARUCO_DICT} — IDs {ARUCO_ID_TO_PONTO}", flush=True)
        except Exception as e:
            print(f"[ABC][WARN] ArUco indisponível ({e})", flush=True)

    def detectar(self, gray):
        if not self.ok: return []
        try:
            if self._novo: corners, ids, _ = self._det.detectMarkers(gray)
            else: corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._par)
        except Exception:
            return []
        if ids is None: return []
        saida = []
        for c, i in zip(corners, ids.flatten()):
            ponto = ARUCO_ID_TO_PONTO.get(int(i))
            if ponto is None: continue
            pts = c.reshape(-1, 2)
            x1, y1 = pts.min(axis=0); x2, y2 = pts.max(axis=0)
            saida.append({"ponto": ponto,
                          "bbox": (int(x1), int(y1), int(x2), int(y2)),
                          "centro": (float(pts[:,0].mean()), float(pts[:,1].mean())),
                          "area": float((x2-x1)*(y2-y1))})
        return saida


def gerar_marcadores_abc(pasta="./marcadores_abc", px=700):
    """python carro-autonomo.py --gerar-aruco → PNGs para imprimir."""
    os.makedirs(pasta, exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    for mid, ponto in ARUCO_ID_TO_PONTO.items():
        try:   img = cv2.aruco.generateImageMarker(d, mid, px)
        except AttributeError: img = cv2.aruco.drawMarker(d, mid, px)
        borda = px // 8
        canvas = np.full((px+2*borda, px+2*borda), 255, np.uint8)
        canvas[borda:borda+px, borda:borda+px] = img
        canvas = cv2.copyMakeBorder(canvas, 0, 90, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(canvas, f"PONTO {ponto}  (ID {mid})",
                    (borda, canvas.shape[0]-28), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
        cv2.imwrite(os.path.join(pasta, f"ponto_{ponto}_id{mid}.png"), canvas)
        print(f"[ABC] {pasta}/ponto_{ponto}_id{mid}.png", flush=True)
    print("[ABC] margem branca faz parte do marcador — não corte.", flush=True)

ABC = None   # instanciado em main()

def gerar_placas_abc(pasta="./placas_abc", px=700):
    """python carro-autonomo.py --gerar-abc → PNGs para imprimir."""
    os.makedirs(pasta, exist_ok=True)
    for L in LETRAS_PONTOS:
        img = _molde_letra(L, lado=px)
        canvas = cv2.copyMakeBorder(img, 40, 110, 40, 40,
                                    cv2.BORDER_CONSTANT, value=255)
        cv2.putText(canvas, f"PONTO {L}", (46, canvas.shape[0]-34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        caminho = os.path.join(pasta, f"ponto_{L}.png")
        cv2.imwrite(caminho, canvas)
        print(f"[ABC] {caminho}", flush=True)
    print("\n[ABC] Imprima em ~15cm de DIÂMETRO, papel FOSCO.\n"
          "      Recorte no QUADRADO (fundo branco em volta) ou deixe\n"
          "      a folha inteira — NÃO recorte rente ao anel preto.\n"
          "      Fixe na altura da câmera, perpendicular à pista.", flush=True)

PGOM_PROMOVE = 0.85
PGOM_PROMOVE_SEM = 0.72
PGOM_MAX_MISS = 3
PGOM_MATCH_D = 60
PGOM_HIST = 10
PGOM_PERSIST_N = 8

# Pesos do PARE agora incluem perspectiva (a forma é explicável por
# um octógono real?) e simetria (placa real é espelho-simétrica).
# O peso saiu de convexidade e fingerprint, que eram parcialmente
# redundantes com esses dois testes mais fortes. Soma = 1.00.
PESOS_PLACA = dict(forma=0.18, convex=0.05, aspecto=0.08, simbolo=0.12,
                   persist=0.18, estab=0.12, fingerprint=0.05,
                   persp=0.12, simet=0.10)
# Semáforo é julgado pelos faróis (símbolo) — perspectiva/simetria
# não se aplicam, entram neutros com peso zero.
PESOS_SEMAFORO = dict(forma=0.15, convex=0.05, aspecto=0.15, simbolo=0.35,
                      persist=0.05, estab=0.15, fingerprint=0.10,
                      persp=0.00, simet=0.00)

class Ficha:
    _nid = 0
    def __init__(self, det):
        self.id = Ficha._nid; Ficha._nid += 1
        self.centros = deque(maxlen=PGOM_HIST); self.areas = deque(maxlen=PGOM_HIST)
        self.formas = deque(maxlen=PGOM_HIST); self.convexs = deque(maxlen=PGOM_HIST)
        self.aspectos = deque(maxlen=PGOM_HIST); self.simbolos = deque(maxlen=PGOM_HIST)
        self.fingerps = deque(maxlen=PGOM_HIST)
        self.persps = deque(maxlen=PGOM_HIST); self.simets = deque(maxlen=PGOM_HIST)
        self.vistos = 0; self.missed = 0; self.det = det
        self._push(det)
    @staticmethod
    def _score_forma(det):
        nv, circ = det.get("lados",0), det.get("circ",0)
        if det["class_name"] == "Semaforo": return 1.0
        if 7 <= nv <= 9 and circ > 0.62: return 1.0
        if nv == 3: return 0.9
        if nv >= 6 and circ > 0.70: return 0.9
        if nv in (4,5): return 0.7
        return 0.3
    @staticmethod
    def _fingerprint(det):
        nv = det.get("lados",0)
        grupo_nv = 3 if nv==3 else 4 if nv in (4,5) else 7 if nv in (6,7) else 9
        return (grupo_nv, int(det.get("circ",0)*3), int(det.get("ar",1.0)*3))
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
        self.persps.append(float(det.get("persp", 0.5)))
        self.simets.append(float(det.get("simet", 0.5)))
        self.vistos += 1; self.missed = 0; self.det = det
    def perto_de(self, det):
        x1,y1,x2,y2 = det["bbox"]; cx, cy = (x1+x2)/2, (y1+y2)/2
        px, py = self.centros[-1]
        return ((cx-px)**2 + (cy-py)**2) ** 0.5
    def evidencias(self):
        e_persist = float(np.clip(self.vistos/PGOM_PERSIST_N, 0, 1))
        if len(self.centros) >= 3:
            passos = [((self.centros[i+1][0]-self.centros[i][0])**2 +
                       (self.centros[i+1][1]-self.centros[i][1])**2)**0.5
                      for i in range(len(self.centros)-1)]
            e_c = float(np.clip(1.0 - np.std(passos)/12.0, 0, 1))
            razoes = [self.areas[i+1]/self.areas[i] for i in range(len(self.areas)-1)]
            e_a = float(np.clip(1.0 - np.std(razoes)/0.25, 0, 1))
            e_estab = (e_c+e_a)/2
        else:
            e_estab = 0.4
        cnt = Counter(self.fingerps)
        e_fp = cnt.most_common(1)[0][1]/len(self.fingerps)
        return dict(forma=float(np.mean(self.formas)), convex=float(np.mean(self.convexs)),
                    aspecto=float(np.mean(self.aspectos)), simbolo=float(np.mean(self.simbolos)),
                    persist=e_persist, estab=float(e_estab), fingerprint=float(e_fp),
                    persp=float(np.mean(self.persps)), simet=float(np.mean(self.simets)))
    def total(self):
        pesos = PESOS_SEMAFORO if self.det["class_name"]=="Semaforo" else PESOS_PLACA
        ev = self.evidencias()
        return sum(pesos[k]*ev[k] for k in pesos)

class PGOM:
    def __init__(self): self.fichas = []; self.stats = dict(vistos=0, promovidos=0)
    def update(self, dets):
        geo = [d for d in dets if d.get("geo")]; outros = [d for d in dets if not d.get("geo")]
        self.stats["vistos"] += len(geo)
        usados = set(); casadas = {}
        for d in geo:
            melhor, dist_m = None, PGOM_MATCH_D
            for f in self.fichas:
                if id(f) in usados: continue
                dist = f.perto_de(d)
                if dist < dist_m: melhor, dist_m = f, dist
            if melhor is not None:
                melhor._push(d); usados.add(id(melhor)); casadas[id(melhor)] = d
            else:
                nova = Ficha(d); self.fichas.append(nova)
                casadas[id(nova)] = d; usados.add(id(nova))
        for f in self.fichas:
            if id(f) not in usados and f.vistos > 1: f.missed += 1
        self.fichas = [f for f in self.fichas if f.missed <= PGOM_MAX_MISS]
        promovidos = []
        for f in self.fichas:
            d = casadas.get(id(f))
            if d is None: continue
            tot = f.total()
            limiar = PGOM_PROMOVE_SEM if f.det["class_name"] == "Semaforo" else PGOM_PROMOVE
            if tot >= limiar:
                d = dict(d); d["evid"] = f.evidencias(); d["evtot"] = float(tot)
                promovidos.append(d); self.stats["promovidos"] += 1
        return promovidos + outros

PGOM_M = PGOM()

def score_final(det, cnn_conf):
    ev = float(det.get("evtot", det.get("conf", 0.5)))
    return 0.85*ev + 0.15*float(cnn_conf)

class YOLODetector:
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(path, opts, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.in_n = self.sess.get_inputs()[0].name; self.out_n = self.sess.get_outputs()[0].name
        in_shape = self.sess.get_inputs()[0].shape
        h_model, w_model = in_shape[2], in_shape[3]
        self.input_size = h_model if isinstance(h_model, int) and isinstance(w_model, int) else YOLO_SIZE
        out_shape = self.sess.get_outputs()[0].shape
        n_cls = out_shape[1] - 4 if isinstance(out_shape[1], int) else NUM_CLASSES
        self.coco_mode = (n_cls == 80)
        modo = "COCO (pré-treinado)" if self.coco_mode else f"CUSTOM ({n_cls} classes)"
        print(f"[YOLO] {path} | {self.sess.get_providers()[0]} | {modo} | input={self.input_size}px", flush=True)
    @staticmethod
    def _letterbox(img, size=640):
        h, w = img.shape[:2]; sc = size/max(h,w); nh,nw = int(h*sc),int(w*sc)
        canvas = np.full((size,size,3),114,np.uint8)
        py,px = (size-nh)//2,(size-nw)//2
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
    def detectar(self, frame_enhanced):
        if frame_enhanced.ndim == 2: frame_enhanced = cv2.cvtColor(frame_enhanced, cv2.COLOR_GRAY2BGR)
        h0,w0 = frame_enhanced.shape[:2]
        canvas,sc,px,py = self._letterbox(frame_enhanced, self.input_size)
        inp = canvas[:,:,::-1].astype(np.float32)/255.0
        inp = inp.transpose(2,0,1)[np.newaxis]
        raw = self.sess.run([self.out_n],{self.in_n:inp})[0]
        preds = raw[0].T
        cls_sc = preds[:,4:]; max_c = cls_sc.max(axis=1)
        mask = max_c >= YOLO_CONF
        if not mask.any(): return []
        preds,cls_sc,max_c = preds[mask],cls_sc[mask],max_c[mask]
        cx,cy,bw,bh = preds[:,0],preds[:,1],preds[:,2],preds[:,3]
        cls_ids = cls_sc.argmax(axis=1)
        bxs = np.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],axis=1)
        results = []
        for cid in np.unique(cls_ids):
            if self.coco_mode:
                if int(cid) not in COCO_MAP: continue
                cls_name = COCO_MAP[int(cid)]
            else:
                cls_name = CLASSES[int(cid)] if int(cid)<NUM_CLASSES else "?"
            idx = np.where(cls_ids==cid)[0]
            keep = self._nms(bxs[idx], max_c[idx], YOLO_NMS)
            for k in keep:
                i = idx[k]
                x1 = int(np.clip((bxs[i,0]-px)/sc, 0, w0-1)); y1 = int(np.clip((bxs[i,1]-py)/sc, 0, h0-1))
                x2 = int(np.clip((bxs[i,2]-px)/sc, 0, w0-1)); y2 = int(np.clip((bxs[i,3]-py)/sc, 0, h0-1))
                if y1 < h0*ROI_Y0 or y2 > h0*ROI_Y1: continue
                if self.coco_mode:
                    if max_c[i] < COCO_CONF_MIN.get(cls_name, YOLO_CONF): continue
                    if (x2-x1)*(y2-y1) < COCO_AREA_MIN.get(cls_name, 0): continue
                results.append({"bbox":(x1,y1,x2,y2),"class_name":cls_name,"class_id":int(cid),
                                 "conf":float(max_c[i]),"coco":self.coco_mode})
        return results

class CNNClassifier:
    def __init__(self, path):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path); interp.allocate_tensors()
        d = interp.get_input_details()[0]
        self._interp = interp; self._in = d["index"]; self._out = interp.get_output_details()[0]["index"]
        self._q = d["dtype"] == np.uint8
        self.size = int(d["shape"][1])
        global CNN_SIZE; CNN_SIZE = self.size
        q = "INT8" if self._q else "FP32"
        print(f"[CNN] {path} | {self.size}x{self.size} | {q}", flush=True)
    def predict(self, img_96):
        inp = img_96[np.newaxis].astype(np.float32)
        if self._q: inp = (inp*255).astype(np.uint8)
        self._interp.set_tensor(self._in, inp); self._interp.invoke()
        out = self._interp.get_tensor(self._out)[0]
        return out.astype(np.float32)/255.0 if self._q else out

def prep_mono(crop):
    if crop.ndim == 3: crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(crop, (CNN_SIZE, CNN_SIZE)); g = _CLAHE.apply(g)
    return np.stack([g, g, g], axis=-1).astype(np.float32) / 255.0

def carregar_classes():
    p = os.path.join(os.path.dirname(YOLO_MODEL), "classes.txt")
    if os.path.exists(p):
        with open(p) as f: cls = [l.strip() for l in f if l.strip()]
        if cls: print(f"[CNN] classes.txt: {cls}", flush=True); return cls
    return CLASSES

class VerificadorPlaca:
    # Contrato forma ↔ classe. Uma classe só é aceita se a geometria
    # concordar. Delivery (A/B/C) não passa pela CNN — está aqui como
    # documentação do contrato e para validar candidatos do YOLO.
    FORMA_CLASSE = {
        "Stop": {"octogono"},
        "A":    {"circulo"},
        "B":    {"circulo"},
        "C":    {"circulo"},
    }

    def __init__(self, margem=0.25, entropia_max=1.30):
        self.margem_min = margem; self.entropia_max = entropia_max

    @staticmethod
    def _forma_geo(det):
        """A forma já foi decidida por classificar_forma() na detecção.
        Não reclassifica aqui — árbitro único."""
        return det.get("forma", "?")
    def e_placa(self, scores, cls_nm, det):
        ordenado = np.sort(scores)[::-1]
        top1 = float(ordenado[0]); top2 = float(ordenado[1]) if len(ordenado) > 1 else 0.0
        margem = top1 - top2
        p = np.clip(scores, 1e-9, 1.0)
        entropia = float(-(p*np.log(p)).sum())
        if margem < self.margem_min: return False, f"margem baixa {margem:.2f}", margem
        if entropia > self.entropia_max: return False, f"entropia alta {entropia:.2f}", margem
        esperadas = self.FORMA_CLASSE.get(cls_nm)
        if esperadas is not None:
            fg = self._forma_geo(det)
            if fg != "?" and fg not in esperadas: return False, f"forma {fg}!={cls_nm}", margem
        return True, "ok", margem

def geometria_concorda(gray, det) -> bool:
    """YOLO auxiliar: a geometria valida ou descarta o candidato.

        YOLO diz STOP → é octógono? NÃO → descarta · SIM → aceita

    Classes sem contrato de forma (Carro, Pessoa, Cone, Semaforo)
    passam direto — não são placas e não têm forma esperada."""
    esperadas = VerificadorPlaca.FORMA_CLASSE.get(det.get("class_name"))
    if esperadas is None: return True
    x1, y1, x2, y2 = det["bbox"]
    m = 4                                  # folga: o box do YOLO corta a borda
    x1 = max(0, x1-m); y1 = max(0, y1-m)
    x2 = min(gray.shape[1], x2+m); y2 = min(gray.shape[0], y2+m)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16: return False
    _, b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, K5)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return False
    c = max(cnts, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(c)
    forma = classificar_forma(c, bw, bh)
    det["forma"] = forma
    return forma in esperadas


class OODRejector:
    def __init__(self, path):
        self._t = {}
        if os.path.exists(path):
            with open(path) as f: self._t = json.load(f)
            print(f"[OOD] thresholds: {self._t}", flush=True)
        else: print(f"[OOD] usando default={OOD_DEFAULT}", flush=True)
    def aceitar(self, cls, score): return score >= self._t.get(cls, OOD_DEFAULT)

class Track:
    _nid = 0
    def __init__(self, bbox, hint, conf):
        self.id = Track._nid; Track._nid += 1
        self.bbox = bbox; self.class_hint = hint; self.conf = conf
        self.buf = deque(maxlen=VOTE_BUFFER); self.age = 0; self.missed = 0
        self.state = "tentative"
    def atualizar(self, bbox, hint, conf, cnn_lbl, cnn_conf):
        self.bbox = bbox; self.class_hint = hint; self.conf = conf
        self.missed = 0; self.age += 1
        if cnn_lbl: self.buf.append((cnn_lbl, cnn_conf))
        if self.age >= 3: self.state = "confirmed"
    def votar(self):
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
    def __init__(self): self.tracks = []
    @staticmethod
    def _iou(a, b):
        ix1=max(a[0],b[0]); iy1=max(a[1],b[1]); ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        if not inter: return 0.0
        aA=(a[2]-a[0])*(a[3]-a[1]); aB=(b[2]-b[0])*(b[3]-b[1])
        return inter/(aA+aB-inter+1e-9)
    def _match_greedy(self, tracks_idx, det_idx, dets, iou_min):
        pairs = []
        for ti in tracks_idx:
            for di in det_idx:
                iou = self._iou(self.tracks[ti].bbox, dets[di]["bbox"])
                if iou >= iou_min: pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        mt, md = set(), set(); matched = []
        for iou, ti, di in pairs:
            if ti in mt or di in md: continue
            mt.add(ti); md.add(di); matched.append((ti, di))
        unmatched_t = [i for i in tracks_idx if i not in mt]
        unmatched_d = [i for i in det_idx if i not in md]
        return matched, unmatched_t, unmatched_d
    def update(self, dets, frame_enhanced, cnn, ood, verif=None):
        hi_idx = [i for i,d in enumerate(dets) if d["conf"] >= CONF_HIGH_THR]
        lo_idx = [i for i,d in enumerate(dets) if d["conf"] < CONF_HIGH_THR]
        all_t = list(range(len(self.tracks)))
        matched1, unm_t1, unm_d_hi = self._match_greedy(all_t, hi_idx, dets, TRACK_IOU_HIGH)
        matched2, unm_t2, _ = self._match_greedy(unm_t1, lo_idx, dets, TRACK_IOU_LOW)
        for ti, di in matched1 + matched2:
            d = dets[di]
            lbl, conf_cnn = self._classificar(d, frame_enhanced, cnn, ood, verif)
            self.tracks[ti].atualizar(d["bbox"], d["class_name"], d["conf"], lbl, conf_cnn)
        for ti in unm_t2: self.tracks[ti].missed += 1
        for di in unm_d_hi:
            d = dets[di]
            lbl, conf_cnn = self._classificar(d, frame_enhanced, cnn, ood, verif)
            t = Track(d["bbox"], d["class_name"], d["conf"])
            if lbl: t.buf.append((lbl, conf_cnn))
            self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= TRACK_MAX_AGE]
        return self.tracks
    @staticmethod
    def _classificar(det, frame_enhanced, cnn, ood, verif=None):
        if det.get("coco", False): return det["class_name"], det["conf"]
        if det.get("geo") and det["class_name"] == "Semaforo":
            return "Semaforo", score_final(det, det["conf"])
        if cnn is None: return None, 0.0
        x1,y1,x2,y2 = det["bbox"]
        crop = frame_enhanced[y1:y2, x1:x2]
        if crop.size==0 or (x2-x1)<8 or (y2-y1)<8: return None, 0.0
        scores = cnn.predict(prep_mono(crop))
        max_s = float(scores.max())
        cls_nm = CNN_CLASSES[int(scores.argmax())] if int(scores.argmax()) < len(CNN_CLASSES) else None
        if cls_nm in (None, "Fundo"): return None, 0.0
        if verif is not None:
            ok, motivo, _ = verif.e_placa(scores, cls_nm, det)
            if not ok: return None, 0.0
        if ood and not ood.aceitar(cls_nm, max_s): return None, 0.0
        return cls_nm, score_final(det, max_s)

# ================================================================
#  [3] COMUNICAÇÃO — QR · MQTT (destino) · TELEMETRIA (REST)
#      Tudo best-effort e não-bloqueante: se a rede cair, o carro
#      continua operando normalmente (fallback: teclas 1/2/3).
# ================================================================

PAGINA = """<!doctype html><html lang=pt-BR><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>171 Garage</title><style>
body{background:#141414;color:#f2f2f2;font-family:system-ui,sans-serif;
     margin:0;padding:24px 18px;max-width:420px;margin:auto}
h2{font-size:15px;color:#8a8a8a;font-weight:600;margin:26px 0 10px;
   letter-spacing:.04em;text-transform:uppercase}
.eq{text-align:center;font-size:12px;letter-spacing:.2em;color:#8a8a8a;
    text-transform:uppercase;padding-bottom:14px;border-bottom:1px solid #333}
.g{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
button{background:#1e1e1e;color:#f2f2f2;border:1px solid #333;border-radius:12px;
   padding:22px 0;font-size:26px;font-weight:700;font-family:inherit;cursor:pointer}
button:active{background:#262626}
button.on{background:#22a45d;border-color:#22a45d}
.off{grid-column:1/-1;font-size:15px;padding:15px 0;border-color:#c4341c;color:#c4341c}
#st{margin-top:22px;padding-top:14px;border-top:1px solid #333;
    font-size:13px;color:#8a8a8a;line-height:1.7;white-space:pre-line}
</style>
<div class=eq>171 Garage &middot; Carro CAR_ID_AQUI</div>
<h2>Retirar o pacote em</h2>
<div class=g id=r></div>
<h2>Entregar em</h2>
<div class=g id=e></div>
<div class=g style="margin-top:18px">
  <button class=off onclick="cmd('off')">Desligar motores</button>
</div>
<div id=st>carregando...</div>
<script>
for (const [id,tipo] of [['r','R'],['e','E']])
  for (const L of ['A','B','C']) {
    const b=document.createElement('button');
    b.textContent=L; b.id=tipo+L;
    b.onclick=()=>cmd(tipo.toLowerCase(),L);
    document.getElementById(id).appendChild(b);
  }
async function cmd(a,p){
  try{ pinta(await (await fetch(`/cmd?a=${a}&p=${p||''}`)).json()); }
  catch(e){ document.getElementById('st').textContent='sem conexao com o carro'; }
}
function pinta(d){
  for(const b of document.querySelectorAll('button')) b.classList.remove('on');
  if(d.retirada) document.getElementById('R'+d.retirada)?.classList.add('on');
  if(d.entrega)  document.getElementById('E'+d.entrega)?.classList.add('on');
  document.getElementById('st').textContent =
    `estado: ${d.estado}\nrota: ${d.retirada||'-'} para ${d.entrega||'-'}`+
    `\nfalta: ${d.fase}\nentregas: ${d.entregas}`;
}
setInterval(()=>cmd('status'),1500); cmd('status');
</script></html>"""


class PainelLocal:
    """Servidor HTTP na rede do próprio laptop (ponto de acesso, sem
    internet). O celular escaneia o QR, o navegador abre a página
    sozinho — zero app, zero configuração do lado do usuário.

    Roda em thread separada; NUNCA bloqueia o loop de visão."""

    def __init__(self, porta=PAINEL_PORTA):
        import threading, queue
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import urlparse, parse_qs
        self.fila = queue.Queue()
        self.ip   = self._meu_ip()
        self.url  = f"http://{self.ip}:{porta}/"
        painel = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass          # silencia o log HTTP
            def _envia(self, corpo, tipo="text/html; charset=utf-8"):
                b = corpo.encode()
                self.send_response(200)
                self.send_header("Content-Type", tipo)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers(); self.wfile.write(b)
            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/cmd":
                    q = parse_qs(u.query)
                    a = (q.get("a", [""])[0] or "").lower()
                    p = (q.get("p", [""])[0] or "").upper()[:1]
                    if a in ("r", "e") and p in ("A", "B", "C"):
                        painel.fila.put((a.upper(), p))
                    elif a == "off":
                        painel.fila.put(("DESLIGAR", None))
                    self._envia(json.dumps(dict(estado=MISSAO.estado,
                        retirada=MISSAO.retirada, entrega=MISSAO.entrega,
                        fase=MISSAO.fase, entregas=MISSAO.entregas)),
                        "application/json")
                else:
                    self._envia(PAGINA.replace("CAR_ID_AQUI", CAR_ID))

        try:
            self.srv = ThreadingHTTPServer(("0.0.0.0", porta), H)
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            print(f"[PAINEL] {self.url}  (o QR já aponta pra cá)", flush=True)
        except Exception as e:
            self.srv = None
            print(f"[PAINEL][WARN] não subiu ({e}) — use as teclas 1/2/3", flush=True)

    @staticmethod
    def _meu_ip():
        """IP do notebook NA REDE ATIVA — é esse endereço que vai no QR,
        nunca 127.0.0.1 (que só o próprio notebook alcança).

        Truque: perguntamos ao SO qual interface seria usada para
        alcançar um IP candidato — o socket UDP não chega a enviar
        nada, só consulta a tabela de rotas. Funciona mesmo que o
        candidato não seja alcançável de verdade.

        Tenta primeiro o gateway típico do REDE_PERFIL ativo (evita
        escolher a interface errada quando o notebook tem mais de
        uma rede up ao mesmo tempo, ex.: Wi-Fi do TP-Link + Ethernet
        de uma rede antiga); os outros perfis entram como reserva."""
        import socket
        candidatos = [_PERFIS_REDE[REDE_PERFIL]["gateway_chute"]]
        candidatos += [p["gateway_chute"] for p in _PERFIS_REDE.values()
                       if p["gateway_chute"] not in candidatos]
        candidatos.append("192.168.1.1")   # gateway doméstico comum, último recurso
        for alvo in candidatos:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((alvo, 80))
                ip = s.getsockname()[0]; s.close()
                if ip and ip != "0.0.0.0":
                    return ip
            except Exception:
                continue
        return "127.0.0.1"   # nenhuma rede ativa — o QR vai apontar errado

    def poll(self):
        """-> lista de comandos pendentes, sem bloquear."""
        import queue
        cmds = []
        while True:
            try: cmds.append(self.fila.get_nowait())
            except queue.Empty: break
        return cmds

    def responder(self, txt):
        print(f"[EVT] {txt}", flush=True)

    def fechar(self):
        try:
            if self.srv: self.srv.shutdown()
        except Exception: pass


def enviar_telemetria(destino, evento):
    """Sem internet: o evento vai só para o console."""
    if BT: BT.responder(f"{evento};{destino or '-'}")

BT = None   # painel; instanciado em main()


def _wifi_escape(txt):
    """Escapa barra invertida, ponto-e-vírgula, vírgula, dois-pontos
    e aspas — exigência do payload WIFI:. Sem isso, SSID/senha com
    esses caracteres quebram o parser do celular."""
    import re
    return re.sub(r'([\\;,:"])', r'\\\1', txt)


def gerar_qr_wifi(ssid, senha, lado=150):
    """QR no formato que iOS/Android reconhecem NATIVAMENTE pela
    Câmera — sem app nenhum. Ao escanear, o sistema mostra 'Entrar
    na rede X?'; o usuário toca em aceitar e já está conectado."""
    payload = f"WIFI:T:WPA;S:{_wifi_escape(ssid)};P:{_wifi_escape(senha)};H:false;;"
    return gerar_qr(payload, lado=lado)


def gerar_qr(url, lado=260):
    """QR do link de seleção. Usa qrcode se instalado; senão devolve
    um cartão com a URL legível (o operador digita/escaneia manual)."""
    try:
        import qrcode
        img = qrcode.make(url).convert("L")
        q = cv2.resize(np.array(img), (lado, lado), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    except Exception:
        card = np.full((lado, lado, 3), 255, np.uint8)
        cv2.putText(card, "QR indisponivel", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
        cv2.putText(card, "pip install qrcode", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90,90,90), 1)
        for i in range(0, len(url), 26):
            cv2.putText(card, url[i:i+26], (10, 110 + (i//26)*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
        return card


def buzinar(ser, dur_s):
    """Aciona o buzzer do Portenta por dur_s, sem travar o loop."""
    CMD.update(buz=1); enviar(CMD, ser)
    _buz["ate"] = time.monotonic() + dur_s
    print(f"[BUZ] ♪ {dur_s:.1f}s", flush=True)

_buz = dict(ate=0.0)

def tick_buzzer(ser):
    if _buz["ate"] and time.monotonic() >= _buz["ate"]:
        _buz["ate"] = 0.0
        CMD.update(buz=0); enviar(CMD, ser)


def parar(ser, motivo=""):
    """CONTROLE: freia e mantém parado. Quem decide voltar é a MISSÃO."""
    if _nav["ultimo"] == "STOP" and CMD["mot"] == 0: return
    CMD.update(mot=0, srv=127, led=1, brk=1, dir=0); enviar(CMD, ser)
    _nav.update(acao_label="STOP", acao_t=time.monotonic(), acao_dur=0.0, ultimo="STOP")
    print(f"[NAV] ■ PARADO {motivo}", flush=True)


def seguir(ser, com_buzzer=True):
    """CONTROLE: retoma cruzeiro. Buzzer curto sinaliza a saída."""
    if com_buzzer: buzinar(ser, BUZ_PARTIDA_S)
    CMD.update(mot=MOT_CRUZEIRO, srv=127, led=0, brk=0, dir=3); enviar(CMD, ser)
    _nav.update(acao_label=None, acao_t=time.monotonic(), acao_dur=0.0, ultimo="STRAIGHT")
    print(f"[NAV] ▶ CRUZEIRO (mot={MOT_CRUZEIRO})", flush=True)

def desenhar(frame_e, tracks, fps, modo, debug_thr):
    out = cv2.cvtColor(frame_e, cv2.COLOR_GRAY2BGR) if frame_e.ndim == 2 else frame_e.copy()
    h,w = out.shape[:2]; PW = 225
    cv2.rectangle(out,(0,int(h*ROI_Y0)),(w-1,int(h*ROI_Y1)),(0,200,255),1)
    for trk in tracks:
        x1,y1,x2,y2 = trk.bbox
        cls = trk.class_hint; cor = COR_CLASSE.get(cls,(180,180,180))
        lbl, cm = trk.votar()
        thick = 3 if trk.state=="confirmed" and lbl else 1
        cv2.rectangle(out,(x1,y1),(x2,y2),cor,thick)
        buf_n = len(trk.buf); hdr = f"#{trk.id} {cls}"
        if lbl: hdr += f" → {lbl} {cm*100:.0f}%"
        (tw,th),_ = cv2.getTextSize(hdr,cv2.FONT_HERSHEY_SIMPLEX,0.37,1)
        cv2.rectangle(out,(x1,y1-th-6),(x1+tw+4,y1),cor,-1)
        cv2.putText(out,hdr,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.37,(255,255,255),1)
        bw = x2-x1; prog = int(bw*buf_n/VOTE_BUFFER)
        cv2.rectangle(out,(x1,y2+2),(x2,y2+6),(40,40,40),-1)
        cv2.rectangle(out,(x1,y2+2),(x1+prog,y2+6),cor,-1)
    pan = np.full((h,PW,3),(18,18,18),dtype=np.uint8)
    cv2.rectangle(pan,(0,0),(PW-1,h-1),(45,45,45),1)
    def t(s,ln,cor=(190,190,190),sc=0.33):
        cv2.putText(pan,s,(5,14+ln*16),cv2.FONT_HERSHEY_SIMPLEX,sc,cor,1)
    t(f"FPS:{fps:.0f} [{modo}]",0,(255,255,255),0.38)
    if _sat_chk["mono"]:
        cv2.putText(pan,"CAM S/COR!",(120,14), cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,255),1)
    cor_missao = {"AGUARDANDO":(0,220,220),"AGUARDA_ENTREGA":(220,220,0),
                  "IND_RETIRADA":(100,220,100),"IND_ENTREGA":(100,220,100),
                  "LIVRE":(160,220,160),"PARADO_SEM":(50,50,220),"PARADO_PARE":(50,50,220),
                  "PARADO_OBST":(0,0,200),"RETIRANDO":(220,180,0),
                  "ENTREGANDO":(220,180,0),"DESLIGADO":(120,120,120)}.get(MISSAO.estado,(190,190,190))
    t(f"MISSAO: {MISSAO.estado}",1,cor_missao,0.36)
    t(f"Rota:{MISSAO.retirada or '-'}>{MISSAO.entrega or '-'}  "
      f"fase:{MISSAO.fase[:4]}  Entregas:{MISSAO.entregas}",2,(200,200,120))
    t(f"Trk:{len(tracks)} Fichas:{len(PGOM_M.fichas)}",3)
    if MISSAO.estado in ("RETIRANDO","ENTREGANDO"):
        t(f"  {MISSAO.tempo_no_estado():.1f}/{ESPERA_ENTREGA_S:.0f}s",4,cor_missao,0.40)
    elif MISSAO.parado_por_regra():
        t("  PARADO (regra absoluta)",4,(50,50,220),0.36)
    else:
        t(f"  mot={CMD['mot']}",4,(100,200,100))
    t("── CONFIRMADOS ──",6,(60,60,60)); ln = 7
    for trk in tracks[:5]:
        lbl, cm = trk.votar()
        if lbl: t(f"#{trk.id} {lbl} {cm*100:.0f}%",ln,COR_CLASSE.get(lbl,(180,180,180))); ln += 1
    t("── VOTE BUFFER ──",13,(60,60,60))
    for i,trk in enumerate(tracks[:3]):
        buf_n = len(trk.buf)
        t(f"#{trk.id} {buf_n}/{VOTE_BUFFER} {trk.state[:4]}",14+i,COR_CLASSE.get(trk.class_hint,(120,120,120)))
    t("── ÚLTIMO ──",18,(60,60,60))
    ult = _nav["ultimo"] or "-"
    t(f" {ult}",19,COR_ACAO.get(ult,(160,160,160)))
    # DOIS QRs NO CANTO, sempre visíveis — a sessão fica aberta a
    # missão inteira, porque a entrega pode ser escolhida a qualquer
    # momento (inclusive depois que o carro já retirou o pacote).
    #   1) Wi-Fi   — toque em "Entrar na rede" (sem senha, sem app)
    #   2) Painel  — abre a tela de retirada/entrega
    if MISSAO.qr_wifi is not None and MISSAO.qr_img is not None:
        q1, q2 = MISSAO.qr_wifi, MISSAO.qr_img
        qh, qw = q1.shape[:2]
        gap = 14
        qx2, qy = w - qw - 10, h - qh - 20
        qx1 = qx2 - qw - gap
        if qx1 > 0 and qy > 0:
            cv2.rectangle(out,(qx1-6,qy-20),(qx2+qw+6,qy+qh+6),(255,255,255),-1)
            cv2.putText(out,"1. Wi-Fi",(qx1,qy-6),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1)
            cv2.putText(out,"2. Painel",(qx2,qy-6),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),1)
            out[qy:qy+qh, qx1:qx1+qw] = q1
            out[qy:qy+qh, qx2:qx2+qw] = q2
    vis = np.empty((h,w+PW,3),np.uint8)
    vis[:,:w]=out; vis[:,w:]=pan
    if debug_thr is not None:
        thr_bgr = cv2.cvtColor(debug_thr,cv2.COLOR_GRAY2BGR)
        thr_bgr = cv2.resize(thr_bgr,(w+PW,h))
        vis = np.vstack([vis, thr_bgr])
    return vis

def calibrar(usar_camera):
    src = CAM_IDX if usar_camera else VIDEO
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened(): print("[ERRO] Fonte não abriu"); return
    win = "Calibração — Q=sair  S=salvar"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1200, 520)
    def n(x): pass
    cv2.createTrackbar("CLAHE clip x10", win, 20, 50, n)
    cv2.createTrackbar("Bilateral d", win, 5, 15, n)
    cv2.createTrackbar("Bilateral sigma", win, 40,150, n)
    cv2.createTrackbar("Thresh block", win, 15, 51, n)
    cv2.createTrackbar("Thresh C", win, 4, 25, n)
    cv2.createTrackbar("Sharpen ON", win, 1, 1, n)
    print("[CAL] Ajuste os sliders. S=salvar. Q=sair.", flush=True)
    while True:
        ret,frm = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        h0,w0 = frm.shape[:2]; scale = IMG_W/w0
        frm = cv2.resize(frm,(IMG_W,int(h0*scale)))
        clip = max(0.5, cv2.getTrackbarPos("CLAHE clip x10",win)/10)
        bd = max(1, cv2.getTrackbarPos("Bilateral d",win)); bs = max(1, cv2.getTrackbarPos("Bilateral sigma",win))
        blk = cv2.getTrackbarPos("Thresh block",win); blk = blk if blk%2==1 and blk>=3 else max(3,blk|1)
        tc = cv2.getTrackbarPos("Thresh C",win); sh = cv2.getTrackbarPos("Sharpen ON",win)==1
        cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        lab = cv2.cvtColor(frm, cv2.COLOR_BGR2LAB); lab[:,:,0] = cl.apply(lab[:,:,0])
        enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR); enh = cv2.bilateralFilter(enh, bd, bs, bs)
        if sh: enh = cv2.filter2D(enh,-1,K_SHARP)
        gray = cv2.medianBlur(cv2.cvtColor(enh,cv2.COLOR_BGR2GRAY),5)
        thr = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, tc)
        thr = cv2.morphologyEx(thr,cv2.MORPH_CLOSE,K5); thr = cv2.morphologyEx(thr,cv2.MORPH_OPEN, K3)
        thr_c = cv2.cvtColor(thr,cv2.COLOR_GRAY2BGR)
        cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            a=cv2.contourArea(cnt)
            if AREA_CNT_MIN<a<AREA_CNT_MAX:
                x,y,bw,bh=cv2.boundingRect(cnt)
                cv2.rectangle(thr_c,(x,y),(x+bw,y+bh),(0,255,0),1)
        info = f"clip={clip:.1f} bil={bd}/{bs} thr={blk}/{tc} sharp={'Y' if sh else 'N'}"
        cv2.putText(enh,info,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,255,180),1)
        h,w = frm.shape[:2]; vis = np.hstack([frm,enh,thr_c])
        cv2.imshow(win, vis)
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'): break
        if k==ord('s'):
            cfg = dict(clahe_clip=clip,bilateral_d=bd,bilateral_s=bs,thresh_block=blk,thresh_c=tc,usar_sharp=sh)
            path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"pre_config.json")
            with open(path,"w") as f: json.dump(cfg,f,indent=2)
            print(f"[CAL] Salvo: {path}\n{cfg}", flush=True)
    cap.release(); cv2.destroyAllWindows()

def main(usar_camera=False, debug=False, auto=False, fast=False, usar_yolo=False, usar_canny=False, cam_idx=None, debug_abc=False):
    global _ser, _CLAHE, CNN_CLASSES, CAM_FLIP
    if fast:
        print("[FAST] Processando todo frame (sem pular) + voting rápido", flush=True)
        global VOTE_BUFFER, VOTE_MIN_DETS, VOTE_FRAC
        VOTE_BUFFER = 6; VOTE_MIN_DETS = 3; VOTE_FRAC = 0.60
        print("[FAST] Voting 6 frames, confirma com 4/6", flush=True)
    cfg_p = os.path.join(os.path.dirname(os.path.abspath(__file__)),"pre_config.json")
    if os.path.exists(cfg_p):
        with open(cfg_p) as f: _cfg = json.load(f)
        _CLAHE = cv2.createCLAHE(clipLimit=_cfg.get("clahe_clip",2.0), tileGridSize=(8,8))
        print(f"[PRE] Config: {_cfg}", flush=True)
    cnn = ood = verif = None
    if os.path.exists(CNN_MODEL):
        try:
            cnn=CNNClassifier(CNN_MODEL); ood=OODRejector(OOD_FILE); verif=VerificadorPlaca()
            CNN_CLASSES = carregar_classes()
        except Exception as e: print(f"[WARN] CNN: {e}", flush=True)
    else: print(f"[WARN] CNN não encontrada → execute: python TRAIN_SIGN_CNN.py", flush=True)
    yolo = None; modo = "GEO"
    if usar_yolo and os.path.exists(YOLO_MODEL):
        try: yolo=YOLODetector(YOLO_MODEL); modo="YOLO"
        except Exception as e: print(f"[WARN] YOLO: {e}", flush=True)
    elif usar_yolo: print(f"[WARN] --yolo pedido mas {YOLO_MODEL} não existe → GEO", flush=True)
    print(f"[DET] Detector principal: {modo}", flush=True)
    _dica_rede = ("ative o Hotspot Móvel do Windows antes de ligar"
                  if REDE_PERFIL == "notebook" else
                  "ligue o roteador TP-Link no modo AP antes de ligar")
    print(f"[CFG] Painel   : porta {PAINEL_PORTA} — perfil='{REDE_PERFIL}' "
          f"SSID='{WIFI_SSID}' ({_dica_rede})", flush=True)
    print(f"[CFG] Pontos   : {PONTO_MODO}", flush=True)
    global LEITOR, ABC, BT
    LEITOR = LeitorLetras(debug=debug_abc)
    ABC    = DetectorABC() if PONTO_MODO == "aruco" else None
    BT     = PainelLocal()
    tracker = ByteTrackLite()
    if usar_camera: cap = abrir_camera(CAM_IDX if cam_idx is None else cam_idx)
    else: cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened(): print("[ERRO] Fonte de vídeo"); sys.exit(1)
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1,int(1000/fps_vid)) if not usar_camera else 1
    _ser = conectar_serial()
    fps_t=time.monotonic(); fps_n=0; fps=0.0; fn=0
    SKIP = 1 if fast else 2
    if auto:
        print("[MISSAO] Auto-start — pula a espera do QR", flush=True)
        MISSAO.retirada = "A"; MISSAO.entrega = "B"; MISSAO.fase = "retirada"
        MISSAO.set_estado("IND_RETIRADA"); seguir(_ser, com_buzzer=False)
    MISSAO.nova_sessao()   # UMA sessão por ligada — QR fica no canto
    frame_e = None; debug_thr = None
    print("[OK] Q=sair · SPACE=pausa · 1/2/3=retirada, depois entrega · "
          "S=status · D=desliga/liga · R=reset · F=flip", flush=True)
    while True:
        ret, frame_raw = cap.read()
        if ret and usar_camera and CAM_FLIP is not None: frame_raw = cv2.flip(frame_raw, CAM_FLIP)
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        fn += 1
        if _nav["cooldown"]>0: _nav["cooldown"]-=1
        tick_buzzer(_ser)
        if fn % SKIP == 0:
            checar_camera(frame_raw)
            frame_e = preprocessar(frame_raw)
            h, w = frame_e.shape[:2]; debug_thr = None
            dets = detectar_geometrico(frame_e, usar_canny); det_modo = "GEO"
            if yolo and fn % YOLO_EVERY == 0:
                rx0, ry0, rx1, ry1 = ROI_DIN.janela(h, w)
                sub = frame_e[ry0:ry1, rx0:rx1]
                if sub.size:
                    for d in yolo.detectar(sub):
                        x1,y1,x2,y2 = d["bbox"]
                        d["bbox"] = (x1+rx0, y1+ry0, x2+rx0, y2+ry0)
                        # YOLO é AUXILIAR: a geometria tem a palavra final.
                        # Se o YOLO diz "Stop" mas a forma não é octógono,
                        # o candidato é descartado.
                        if not geometria_concorda(frame_e, d):
                            continue
                        dets.append(d)
                    det_modo = "GEO+YOLO"
            debug_thr = None
            if debug: _, debug_thr = cv2.threshold(frame_e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Delivery NÃO passa por PGOM/CNN — é decidido por geometria
            # + template + votação. Só Stop e Semáforo seguem adiante.
            entregas = [d for d in dets if d["class_name"] == "Delivery"]
            dets     = [d for d in dets if d["class_name"] != "Delivery"]
            dets = [d for d in dets if (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]) >= AREA_MIN_EXEC]
            dets = PGOM_M.update(dets)
            tracks = tracker.update(dets, frame_e, cnn, ood, verif)
            for trk in tracks:
                if trk.state == "confirmed" or trk.age >= 2: ROI_DIN.registrar(trk.bbox)
            # ─── VISÃO: "o que existe?" ──────────────────────────
            # Consolida tudo o que a percepção viu num único dict.
            # Nenhuma decisão aqui — só constatação de fatos.
            percep = dict(pare=False, semaforo=None, obstaculo=False, ponto=None)
            for trk in tracks:
                lbl, _cm = trk.votar()
                if not lbl or trk.state != "confirmed": continue
                if trk.area < AREA_MIN_EXEC: continue
                if lbl == "Semaforo":
                    x1,y1,x2,y2 = trk.bbox
                    percep["semaforo"] = analisar_semaforo(frame_e[y1:y2, x1:x2])
                elif lbl == "Stop":
                    percep["pare"] = True
                elif CLASS_TO_ACTION.get(lbl) == "OBSTACLE":
                    percep["obstaculo"] = True
            # ─── PONTO A/B/C visto neste frame ───────────────────
            if PONTO_MODO == "aruco":
                marcadores = ABC.detectar(frame_e) if ABC else []
                m0 = max(marcadores, key=lambda m: m["area"]) if marcadores else None
            else:
                m0 = LEITOR.votar(entregas)
            if m0 is not None:
                bx1,by1,bx2,by2 = m0["bbox"]
                area_bbox = (bx2-bx1)*(by2-by1)
                if area_bbox >= AREA_MIN_CHEGADA:
                    percep["ponto"] = m0["ponto"]
                if m0["ponto"] != _ultimo_ponto[0]:
                    _ultimo_ponto[0] = m0["ponto"]
                    perto = area_bbox >= AREA_MIN_CHEGADA
                    print(f"[ABC] ponto '{m0['ponto']}' visto — área={area_bbox:.0f} "
                          f"({'perto' if perto else 'longe, min='+str(AREA_MIN_CHEGADA)})"
                          f"  alvo={MISSAO.alvo()}", flush=True)
            elif _ultimo_ponto[0] is not None:
                _ultimo_ponto[0] = None

            # ─── COMANDOS BLUETOOTH (a qualquer momento) ─────────
            for cmd, arg in BT.poll():
                if cmd == "R":
                    MISSAO.retirada = arg
                    MISSAO.fase = "retirada"
                    if MISSAO.estado == "DESLIGADO":
                        MISSAO.set_estado("AGUARDANDO")   # novo pedido religa
                    enviar_telemetria(arg, "retirada_definida")
                    if MISSAO.estado in ("AGUARDANDO", "LIVRE"):
                        MISSAO.set_estado("IND_RETIRADA"); seguir(_ser)
                elif cmd == "E":
                    MISSAO.entrega = arg
                    enviar_telemetria(arg, "entrega_definida")
                    # se estava esperando na retirada, já parte
                    if MISSAO.estado == "AGUARDA_ENTREGA":
                        MISSAO.fase = "entrega"
                        MISSAO.set_estado("IND_ENTREGA"); seguir(_ser)
                elif cmd == "DESLIGAR":
                    MISSAO.set_estado("DESLIGADO")
                    parar(_ser, "(comando DESLIGAR)")

            # ─── MISSÃO: "o que isso significa?" ─────────────────
            est   = MISSAO.estado
            alvo  = MISSAO.alvo()
            chegou = (alvo is not None and percep["ponto"] == alvo)

            def _chegada(ser):
                """Transição comum de chegada ao alvo da fase atual."""
                if MISSAO.fase == "retirada":
                    MISSAO.set_estado("RETIRANDO")
                    parar(ser, f"(retirada {MISSAO.retirada})")
                    buzinar(ser, BUZ_ENTREGA_S)
                    enviar_telemetria(MISSAO.retirada, "chegou_retirada")
                else:
                    MISSAO.set_estado("ENTREGANDO")
                    parar(ser, f"(entrega {MISSAO.entrega})")
                    buzinar(ser, BUZ_ENTREGA_S)
                    enviar_telemetria(MISSAO.entrega, "chegou_entrega")

            if est == "AGUARDANDO":
                parar(_ser, "(aguardando pedido)")

            elif est in ("IND_RETIRADA", "IND_ENTREGA", "LIVRE"):
                motivo = MISSAO.regra_absoluta(percep)   # PRIORIDADE MÁXIMA
                if motivo:
                    MISSAO.retorno = est                 # p/ retomar a rota
                    MISSAO.set_estado(motivo); parar(_ser, f"({motivo})")
                    enviar_telemetria(alvo, motivo.lower())
                elif chegou:
                    _chegada(_ser)

            elif MISSAO.parado_por_regra():
                # chegada vale mesmo parado por regra — o ponto costuma
                # ficar junto de um cruzamento
                if chegou:
                    print(f"[MISSAO] alvo {alvo} avistado durante parada", flush=True)
                    _chegada(_ser)
                elif est == "PARADO_SEM" and percep["semaforo"] == "verde":
                    MISSAO.set_estado(MISSAO.retorno); seguir(_ser)
                    enviar_telemetria(alvo, "retomou_verde")
                elif est == "PARADO_PARE" and (MISSAO.tempo_no_estado() >= 2.5
                        and not percep["obstaculo"]
                        and percep["semaforo"] != "vermelho"):
                    MISSAO.set_estado(MISSAO.retorno); seguir(_ser)
                    enviar_telemetria(alvo, "retomou_pare")
                elif est == "PARADO_OBST" and (not percep["obstaculo"]
                        and percep["semaforo"] != "vermelho"):
                    MISSAO.set_estado(MISSAO.retorno); seguir(_ser)
                    enviar_telemetria(alvo, "retomou_obstaculo")

            elif est == "RETIRANDO":
                if MISSAO.tempo_no_estado() >= ESPERA_ENTREGA_S:
                    enviar_telemetria(MISSAO.retirada, "pacote_retirado")
                    print(f"[MISSAO] 📦 Pacote retirado em {MISSAO.retirada}", flush=True)
                    MISSAO.fase = "entrega"
                    if MISSAO.entrega:
                        if MISSAO.entrega == MISSAO.retirada:
                            _chegada(_ser)          # entrega no mesmo ponto
                        else:
                            MISSAO.set_estado("IND_ENTREGA"); seguir(_ser)
                    else:
                        # entrega ainda não escolhida — espera aqui;
                        # o MESMO QR/sessão continua valendo
                        MISSAO.set_estado("AGUARDA_ENTREGA")
                        print("[MISSAO] pacote a bordo — aguardando escolha da entrega", flush=True)

            elif est == "AGUARDA_ENTREGA":
                pass    # sai pelo comando E: (tratado acima)

            elif est == "ENTREGANDO":
                if MISSAO.tempo_no_estado() >= ESPERA_ENTREGA_S:
                    MISSAO.entregas += 1
                    enviar_telemetria(MISSAO.entrega, "entregue")
                    print(f"[MISSAO] ✅ Entregue em {MISSAO.entrega} "
                          f"(total: {MISSAO.entregas})", flush=True)
                    # missão cumprida — o carro CONTINUA ANDANDO até
                    # nova missão (R:) ou comando DESLIGAR
                    MISSAO.retirada = None
                    MISSAO.entrega  = None
                    MISSAO.fase     = "retirada"
                    MISSAO.retorno  = "LIVRE"
                    MISSAO.set_estado("LIVRE"); seguir(_ser)

            elif est == "DESLIGADO":
                pass    # só sai pelo comando LIGAR
            modo = det_modo
        disp = frame_e if frame_e is not None else cv2.resize(frame_raw,(IMG_W,int(frame_raw.shape[0]*IMG_W/frame_raw.shape[1])))
        vis = desenhar(disp, tracker.tracks, fps, modo, debug_thr)
        cv2.imshow("Carro Autônomo v5.1", vis)
        fps_n += 1
        if time.monotonic()-fps_t >= 1.0: fps=fps_n; fps_n=0; fps_t=time.monotonic()
        k = cv2.waitKey(delay_ms)&0xFF
        if k==ord('q'): break
        elif k==ord(' '): cv2.waitKey(0)
        elif k==ord('+') and not usar_camera: delay_ms=max(1,delay_ms-5)
        elif k==ord('-') and not usar_camera: delay_ms=min(200,delay_ms+5)
        elif k==ord('f'):
            _ciclo = [None, 1, 0, -1]; CAM_FLIP = _ciclo[(_ciclo.index(CAM_FLIP)+1) % 4]
            print(f"[CAM] flip = {CAM_FLIP}", flush=True)
        elif k==ord('r'):
            MISSAO.retirada = MISSAO.entrega = None
            MISSAO.fase = "retirada"; MISSAO.retorno = "LIVRE"
            MISSAO.set_estado("AGUARDANDO"); parar(_ser, "(reset)")
        elif k==ord('s'):
            print(MISSAO.status(), flush=True)
        elif k==ord('d'):
            if MISSAO.estado == "DESLIGADO":
                MISSAO.set_estado("AGUARDANDO"); print("[MISSAO] religado", flush=True)
            else:
                MISSAO.set_estado("DESLIGADO"); parar(_ser, "(tecla D)")
        elif k in (ord('1'),ord('2'),ord('3')):
            # Fallback sem Bluetooth: 1/2/3 define a RETIRADA; se ela já
            # existe, define a ENTREGA. Mesma semântica de R: e E:.
            p = {ord('1'):"A",ord('2'):"B",ord('3'):"C"}[k]
            if MISSAO.retirada is None:
                MISSAO.retirada = p; MISSAO.fase = "retirada"
                print(f"[MISSAO] retirada = {p} (tecla)", flush=True)
                if MISSAO.estado in ("AGUARDANDO","LIVRE"):
                    MISSAO.set_estado("IND_RETIRADA"); seguir(_ser)
            else:
                MISSAO.entrega = p
                print(f"[MISSAO] entrega = {p} (tecla)", flush=True)
                if MISSAO.estado == "AGUARDA_ENTREGA":
                    MISSAO.fase = "entrega"
                    MISSAO.set_estado("IND_ENTREGA"); seguir(_ser)
    CMD.update(mot=0,srv=127,buz=0,led=0,brk=1,dir=0); enviar(CMD,_ser)
    BT.fechar()
    if _ser: _ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)

if __name__=="__main__":
    args = sys.argv[1:]
    if "--gerar-abc" in args: gerar_placas_abc()
    elif "--gerar-aruco" in args: gerar_marcadores_abc()
    elif "--cal" in args: calibrar("--cam" in args)
    else:
        cam_idx = None
        if "--cam-idx" in args:
            try: cam_idx = int(args[args.index("--cam-idx")+1])
            except (IndexError, ValueError):
                print("[ERRO] use: --cam-idx N  (ex.: --cam-idx 1)"); sys.exit(1)
        main(usar_camera="--cam" in args, debug="--debug" in args, auto="--auto" in args,
             fast="--fast" in args, usar_yolo="--yolo" in args, usar_canny="--canny" in args,
             cam_idx=cam_idx, debug_abc="--debug-abc" in args)