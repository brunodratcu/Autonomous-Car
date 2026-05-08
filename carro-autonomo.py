"""
================================================================
  DETECTOR DE PLACAS — VISÃO COMPUTACIONAL PURA
  Sem modelo treinado. Funciona com qualquer câmera USB.
  Envia comandos JSON ao Arduino via serial.

    USO:
    python carro_autonomo.py              ← vídeo (pista_01.mov)
    python carro_autonomo.py --cam        ← câmera USB
    python carro_autonomo.py --debug      ← janela de diagnóstico
    python carro_autonomo.py --cal        ← calibrar HSV
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
#  [1] CONFIGURAÇÃO — edite aqui
# ================================================================

VIDEO       = "./videoplayback.mp4"   # arquivo de vídeo (modo padrão)
CAM_IDX     = 0                  # índice da câmera USB
PROP        = 2                  # divisor de resolução (1=full, 2=metade)

SERIAL_PORT = "COM3"             # porta serial (Windows: COM3, Linux: /dev/ttyUSB0)
BAUD        = 115200

# ROI de placas: fração da altura do frame
ROI_Y0 = 0.05    # topo da faixa (5% — quase borda superior)
ROI_Y1 = 0.80    # base da faixa (80% — exclui chão próximo)

# Área mínima e máxima do contorno candidato (px²)
# Aumente AREA_MIN se estiver detectando ruído
# Diminua AREA_MIN se as placas aparecem pequenas (longe)
AREA_MIN = 800
AREA_MAX  = 120_000

# Confirmação: quantos frames consecutivos com o mesmo label
# Mais frames = mais seguro, mas mais lento para reagir
CONFIRM_N = 5

# Cooldown após executar uma ação (frames)
# Evita reexecutar a mesma placa logo em seguida
COOLDOWN = 50

# Distância mínima para executar (área mínima da placa no momento da confirmação)
# Serve como proxy de distância: placa grande = perto
AREA_EXEC = 2500   # px² — só executa se a placa tiver pelo menos esta área

# ================================================================
#  [2] MAPA DE AÇÕES
#  Cada label → comando JSON enviado ao Arduino
# ================================================================

ACOES = {
    #  label        mot  srv  buz  led  brk  dir  duracao(s)
    "STOP":     dict(mot=0,   srv=127, buz=0, led=1, brk=1, dir=0, dur=2.5),
    "YIELD":    dict(mot=35,  srv=127, buz=0, led=0, brk=0, dir=0, dur=1.5),
    "LEFT":     dict(mot=40,  srv=50,  buz=0, led=0, brk=0, dir=1, dur=1.4),
    "RIGHT":    dict(mot=40,  srv=204, buz=0, led=0, brk=0, dir=2, dur=1.4),
    "STRAIGHT": dict(mot=62,  srv=127, buz=0, led=0, brk=0, dir=3, dur=1.0),
    "DELIVERY": dict(mot=0,   srv=127, buz=1, led=1, brk=1, dir=0, dur=3.0),
    "PARKING":  dict(mot=0,   srv=127, buz=1, led=1, brk=1, dir=0, dur=4.0),
    "SPEED_UP": dict(mot=80,  srv=127, buz=0, led=0, brk=0, dir=3, dur=2.0),
    "SLOW_DOWN":dict(mot=30,  srv=127, buz=0, led=0, brk=0, dir=0, dur=2.0),
}

# ================================================================
#  [3] RANGES HSV DAS CORES DE PLACA
#  Cada cor tem uma lista de (lower, upper).
#  Vermelho usa dois ranges por ser wrap-around em HSV.
# ================================================================

HSV = {
    "vermelho": [
        (np.array([0,   90,  70]),  np.array([12,  255, 255])),
        (np.array([168, 90,  70]),  np.array([180, 255, 255])),
    ],
    "azul": [
        (np.array([95,  80,  60]),  np.array([135, 255, 255])),
    ],
    "amarelo": [
        (np.array([18,  100, 100]), np.array([35,  255, 255])),
    ],
    "verde": [
        (np.array([40,  70,  60]),  np.array([85,  255, 255])),
    ],
    "laranja": [
        (np.array([8,   130, 100]), np.array([20,  255, 255])),
    ],
    "branco": [
        (np.array([0,   0,   190]), np.array([180, 40,  255])),
    ],
}

# ================================================================
#  [4] KERNELS PRÉ-ALOCADOS
# ================================================================

K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)
K7 = np.ones((7, 7), np.uint8)

# ================================================================
#  [5] ESTADO GLOBAL
# ================================================================

CMD = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)

_conf = dict(
    label=None,   # label sendo acumulado
    cnt=0,        # frames consecutivos
    area=0.0,     # área no momento atual
)

_nav = dict(
    modo="IDLE",
    t_inicio=time.monotonic(),
    cooldown=0,
    ultimo_executado=None,
)

_ser = None   # objeto serial (global para enviar no encerramento)

# ================================================================
#  [6] SERIAL
# ================================================================

def conectar_serial():
    """Auto-detecta Arduino ou usa SERIAL_PORT fixo."""
    kws = ["arduino", "ch340", "cp210", "uart", "usb serial"]
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower() for k in kws):
            try:
                s = serial.Serial(p.device, BAUD, timeout=0, write_timeout=0)
                time.sleep(2)
                s.reset_input_buffer()
                print(f"[SER] Conectado auto: {p.device}", flush=True)
                return s
            except Exception:
                pass
    try:
        s = serial.Serial(SERIAL_PORT, BAUD, timeout=0, write_timeout=0)
        time.sleep(2)
        s.reset_input_buffer()
        print(f"[SER] Conectado manual: {SERIAL_PORT}", flush=True)
        return s
    except Exception as e:
        print(f"[SER] Simulação serial ({e})", flush=True)
        return None


def enviar(cmd: dict, ser):
    """Monta JSON e envia. Imprime no terminal sempre."""
    j = (f'{{"mot":{cmd["mot"]},"srv":{cmd["srv"]},'
         f'"buz":{cmd["buz"]},"led":{cmd["led"]},'
         f'"brk":{cmd["brk"]},"dir":{cmd["dir"]},'
         f'"spd":{cmd["spd"]}}}')
    print(f"[CMD] {j}", flush=True)
    if ser:
        try:
            ser.write((j + "\n").encode())
        except Exception:
            pass


# ================================================================
#  [7] UTILITÁRIOS DE VISÃO
# ================================================================

def mascara_cor(hsv_img: np.ndarray, nome: str) -> np.ndarray:
    """Une todos os ranges da cor em uma máscara binária."""
    m = np.zeros(hsv_img.shape[:2], np.uint8)
    for lo, hi in HSV[nome]:
        m = cv2.bitwise_or(m, cv2.inRange(hsv_img, lo, hi))
    return m


def proporcao_cor(hsv_crop: np.ndarray, nome: str) -> float:
    """Retorna fração [0,1] de pixels da cor no crop."""
    m = mascara_cor(hsv_crop, nome)
    return float(m.sum()) / (m.size * 255 + 1e-9)


def analisar_forma(gray_crop: np.ndarray):
    """
    Analisa o contorno externo do crop.
    Retorna (vertices, circularidade, area_ratio_hull)
    """
    blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 4, 0.0, 0.0

    c    = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    ap   = cv2.approxPolyDP(c, 0.04 * peri, True)
    area = cv2.contourArea(c)
    circ = (4 * np.pi * area) / (peri ** 2 + 1e-9)
    hull = cv2.contourArea(cv2.convexHull(c)) + 1e-9
    return len(ap), float(circ), float(area / hull)


def detectar_seta(gray_crop: np.ndarray) -> str | None:
    """
    Detecta direção de seta pelo centróide do símbolo interno.
    Retorna 'left' | 'right' | 'up' | 'down' | None
    """
    H, W = gray_crop.shape
    # Inverte para pegar símbolo escuro no fundo claro
    _, thr = cv2.threshold(gray_crop, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, K3)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    rx = (cx - W / 2) / (W / 2 + 1e-9)   # -1=esq +1=dir
    ry = (cy - H / 2) / (H / 2 + 1e-9)   # -1=topo +1=base

    x, y, bw, bh = cv2.boundingRect(c)
    asp = bw / max(bh, 1)

    if asp > 1.2:   # símbolo mais largo que alto → horizontal
        if rx < -0.12: return "left"
        if rx >  0.12: return "right"
    # Vertical ou ambíguo
    if ry < -0.12: return "up"
    if ry >  0.12: return "down"
    return "up"   # default: para frente


def numero_visivel(gray_crop: np.ndarray) -> int | None:
    """
    Tenta ler um dígito grande no centro da placa via contornos.
    Útil para placas de velocidade (30, 40, 50...).
    Retorna o dígito estimado ou None.
    """
    H, W = gray_crop.shape
    roi  = gray_crop[H//4: 3*H//4, W//4: 3*W//4]
    _, thr = cv2.threshold(roi, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Heurística simples: conta grandes contornos fechados
    grandes = [c for c in cnts if cv2.contourArea(c) > 80]
    n = len(grandes)
    if n == 1: return 1   # possivelmente "1" ou traço
    if n == 2: return 2   # dois elementos: "30", "50"
    return None


# ================================================================
#  [8] CLASSIFICAÇÃO COR + FORMA
#  Lógica hierárquica sem modelo treinado.
# ================================================================

def classificar(crop_bgr: np.ndarray, area_px: float):
    """
    Classifica uma placa a partir do crop BGR.

    Hierarquia de decisão:
      1. Cor dominante
      2. Forma do contorno externo (vértices + circularidade)
      3. Direção da seta interna (quando aplicável)
      4. Símbolo numérico (quando aplicável)

    Retorna (label: str | None, confiança: float, cor: str)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, "?"

    # Resize uniforme para análise
    crop = cv2.resize(crop_bgr, (80, 80), interpolation=cv2.INTER_AREA)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # ── Proporções de cor ─────────────────────────────────────────
    pc = {nome: proporcao_cor(hsv, nome) for nome in HSV}

    # Cor dominante (ignora branco como dominante — é fundo)
    cores_validas = {k: v for k, v in pc.items() if k != "branco"}
    cor_dom, val_dom = max(cores_validas.items(), key=lambda x: x[1])

    # Sem cor expressiva → não é placa conhecida
    if val_dom < 0.06:
        return None, 0.0, "?"

    # ── Forma ────────────────────────────────────────────────────
    vertices, circ, area_ratio = analisar_forma(gray)
    seta = detectar_seta(gray)

    label = None
    conf  = 0.0

    # ┌─────────────────────────────────────────────────────────┐
    # │  VERMELHO                                               │
    # └─────────────────────────────────────────────────────────┘
    if cor_dom == "vermelho":
        # Octógono → STOP
        if vertices >= 7 and circ > 0.60:
            label, conf = "STOP", 0.90
        # Triângulo pontudo → YIELD
        elif vertices == 3 and area_ratio < 0.75:
            label, conf = "YIELD", 0.85
        # Círculo com barra → PROIBIDO (mapeia para STOP)
        elif circ > 0.72:
            label, conf = "STOP", 0.78
        # Seta vermelha (borda proibida)
        elif seta == "left":
            label, conf = "LEFT",  0.72
        elif seta == "right":
            label, conf = "RIGHT", 0.72
        # Retângulo vermelho genérico → STOP (conservador)
        elif vertices <= 6:
            label, conf = "STOP", 0.65

    # ┌─────────────────────────────────────────────────────────┐
    # │  AZUL                                                   │
    # └─────────────────────────────────────────────────────────┘
    elif cor_dom == "azul":
        if seta == "left":
            label, conf = "LEFT",     0.88
        elif seta == "right":
            label, conf = "RIGHT",    0.88
        elif seta == "up":
            label, conf = "STRAIGHT", 0.85
        elif seta == "down":
            label, conf = "SLOW_DOWN",0.78
        else:
            # Azul sem seta clara → provavelmente indicação
            label, conf = "STRAIGHT", 0.60

    # ┌─────────────────────────────────────────────────────────┐
    # │  AMARELO                                                │
    # └─────────────────────────────────────────────────────────┘
    elif cor_dom == "amarelo":
        if vertices == 3:
            label, conf = "YIELD",     0.83
        elif vertices == 4 and seta == "left":
            label, conf = "LEFT",      0.76
        elif vertices == 4 and seta == "right":
            label, conf = "RIGHT",     0.76
        elif vertices == 4:
            label, conf = "SLOW_DOWN", 0.70
        else:
            label, conf = "YIELD",     0.65

    # ┌─────────────────────────────────────────────────────────┐
    # │  VERDE                                                  │
    # └─────────────────────────────────────────────────────────┘
    elif cor_dom == "verde":
        if seta == "left":
            label, conf = "LEFT",     0.82
        elif seta == "right":
            label, conf = "RIGHT",    0.82
        else:
            label, conf = "DELIVERY", 0.80

    # ┌─────────────────────────────────────────────────────────┐
    # │  LARANJA                                                │
    # └─────────────────────────────────────────────────────────┘
    elif cor_dom == "laranja":
        label, conf = "SLOW_DOWN", 0.75

    return label, conf, cor_dom


# ================================================================
#  [9] LOCALIZAÇÃO DE CANDIDATAS NA ROI
# ================================================================

def localizar_candidatas(frame: np.ndarray):
    """
    Segmenta todas as cores de placa na ROI vertical definida.

    Pipeline:
      1. Recorta ROI (ROI_Y0–ROI_Y1)
      2. Converte para HSV + Gaussian Blur
      3. Une máscaras de todas as cores
      4. Morfologia para limpar ruído
      5. Encontra contornos externos
      6. Filtra por área e proporção
      7. Retorna lista de (x, y, w, h) em coords do frame completo

    Quanto mais limpa a iluminação, melhor o resultado.
    """
    h, w = frame.shape[:2]
    y0   = int(h * ROI_Y0)
    y1   = int(h * ROI_Y1)
    roi  = frame[y0:y1, :]

    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv  = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Une todas as cores
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for nome in HSV:
        mask = cv2.bitwise_or(mask, mascara_cor(hsv, nome))

    # Morfologia: fecha buracos, remove pontos isolados
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (AREA_MIN < area < AREA_MAX):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        razao = bw / max(bh, 1)
        if not (0.35 < razao < 2.0):   # proporção plausível de placa
            continue
        boxes.append((x, y + y0, bw, bh, area))

    # Ordena: maior área primeiro (mais próxima)
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes[:4]   # máximo 4 candidatas por frame


# ================================================================
#  [10] CONFIRMAÇÃO (N frames consecutivos)
# ================================================================

def atualizar_confirmacao(label_raw: str | None,
                          area: float) -> str | None:
    """
    Acumula frames com mesmo label.
    Retorna label quando atingir CONFIRM_N frames E área >= AREA_EXEC.
    Retorna None enquanto não confirmado.
    """
    if _nav["cooldown"] > 0:
        _conf["label"] = None
        _conf["cnt"]   = 0
        return None

    if label_raw is None:
        _conf["cnt"]   = 0
        _conf["label"] = None
        return None

    if label_raw == _conf["label"]:
        _conf["cnt"]  += 1
        _conf["area"]  = area
    else:
        _conf["label"] = label_raw
        _conf["cnt"]   = 1
        _conf["area"]  = area

    if _conf["cnt"] >= CONFIRM_N and area >= AREA_EXEC:
        _conf["cnt"]   = 0
        _conf["label"] = None
        return label_raw

    return None


# ================================================================
#  [11] EXECUTOR DE AÇÃO
# ================================================================

_acao_ativa = dict(label=None, t=0.0, dur=0.0)

def executar_acao(label: str, ser):
    """Inicia a ação e envia primeiro comando ao Arduino."""
    if label not in ACOES:
        return
    a = ACOES[label]
    CMD.update(mot=a["mot"], srv=a["srv"], buz=a["buz"],
               led=a["led"], brk=a["brk"], dir=a["dir"],
               spd=0 if a["mot"]==0 else (1 if a["mot"]<50 else 2))
    enviar(CMD, ser)

    _acao_ativa["label"] = label
    _acao_ativa["t"]     = time.monotonic()
    _acao_ativa["dur"]   = a["dur"]
    _nav["cooldown"]     = COOLDOWN
    _nav["ultimo_executado"] = label
    print(f"[NAV] ▶ Executando: {label}  (dur={a['dur']}s)", flush=True)


def tick_acao(ser) -> bool:
    """
    Mantém o comando ativo durante a duração da ação.
    Retorna True enquanto ação ativa, False quando termina.
    """
    if _acao_ativa["label"] is None:
        return False

    dt = time.monotonic() - _acao_ativa["t"]
    if dt >= _acao_ativa["dur"]:
        # Finaliza: para o carro e zera sinais
        CMD.update(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
        enviar(CMD, ser)
        print(f"[NAV] ✅ Fim: {_acao_ativa['label']}", flush=True)
        _acao_ativa["label"] = None
        return False

    return True   # ainda em execução


# ================================================================
#  [12] VISUALIZAÇÃO
# ================================================================

# Cor BGR para cada label
LABEL_BGR = {
    "STOP":      (50,  50,  220),
    "YIELD":     (0,   200, 220),
    "LEFT":      (220, 120, 0  ),
    "RIGHT":     (0,   120, 220),
    "STRAIGHT":  (50,  220, 50 ),
    "DELIVERY":  (180, 60,  180),
    "PARKING":   (180, 180, 0  ),
    "SPEED_UP":  (0,   200, 100),
    "SLOW_DOWN": (0,   180, 255),
}

def desenhar(frame: np.ndarray,
             boxes: list,
             label_raw: str | None,
             conf_raw: float,
             best_bbox: tuple | None,
             fps: int) -> np.ndarray:
    """
    Renderiza:
      • Retângulo da ROI
      • Contornos candidatos (cinza)
      • Bounding box da melhor candidata (colorido por label)
      • Barra de confirmação
      • Painel de status lateral
    """
    out = frame.copy()
    h, w = out.shape[:2]
    y0 = int(h * ROI_Y0)
    y1 = int(h * ROI_Y1)

    # ── ROI ──────────────────────────────────────────────────────
    cv2.rectangle(out, (0, y0), (w-1, y1), (0, 200, 255), 1)
    cv2.putText(out, "ROI PLACAS", (4, y0+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,200,255), 1)

    # ── Candidatas (cinza) ────────────────────────────────────────
    for x, y, bw, bh, area in boxes:
        cv2.rectangle(out, (x,y), (x+bw, y+bh), (160,160,160), 1)
        cv2.putText(out, f"{int(area)}", (x, y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160,160,160), 1)

    # ── Melhor candidata ─────────────────────────────────────────
    if best_bbox and label_raw:
        x, y, bw, bh = best_bbox
        cor = LABEL_BGR.get(label_raw, (200, 200, 200))

        # Retângulo colorido por label
        thick = 3 if _conf["cnt"] >= CONFIRM_N-1 else 2
        cv2.rectangle(out, (x,y), (x+bw, y+bh), cor, thick)

        # Label + confiança
        txt = f"{label_raw}  {conf_raw:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(out, (x, y-th-10), (x+tw+6, y), cor, -1)
        cv2.putText(out, txt, (x+3, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1)

        # Barra de confirmação abaixo do bbox
        bar_w  = bw
        bar_h  = 7
        prog   = int(bar_w * min(_conf["cnt"], CONFIRM_N) / CONFIRM_N)
        cv2.rectangle(out, (x, y+bh+2), (x+bar_w, y+bh+bar_h+2), (60,60,60), -1)
        cv2.rectangle(out, (x, y+bh+2), (x+prog,  y+bh+bar_h+2), cor, -1)

    # ── Painel lateral direito ────────────────────────────────────
    pw = 210
    panel = np.zeros((h, pw, 3), np.uint8)
    panel[:] = (20, 20, 20)
    cv2.rectangle(panel, (0,0), (pw-1, h-1), (50,50,50), 1)

    def txt(s, linha, cor=(200,200,200), sc=0.38):
        cv2.putText(panel, s, (6, 16+linha*16),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, cor, 1)

    txt(f"FPS: {fps}", 0, (255,255,255), 0.42)

    # Ação ativa
    if _acao_ativa["label"]:
        dt  = time.monotonic() - _acao_ativa["t"]
        cor_a = LABEL_BGR.get(_acao_ativa["label"], (200,200,200))
        txt(f"EXEC: {_acao_ativa['label']}", 1, cor_a, 0.42)
        txt(f"  {dt:.1f}s / {_acao_ativa['dur']}s", 2, cor_a)
    else:
        txt("AGUARDANDO", 1, (120,200,120), 0.40)
        txt(f"cooldown: {_nav['cooldown']}", 2, (120,120,120))

    txt("─ DETECCAO ─", 4, (70,70,70))
    lbl = label_raw or "-"
    cor_l = LABEL_BGR.get(label_raw, (160,160,160)) if label_raw else (120,120,120)
    txt(f"label: {lbl}", 5, cor_l)
    txt(f"conf:  {conf_raw:.2f}", 6)
    txt(f"frames:{_conf['cnt']}/{CONFIRM_N}", 7,
        (0,255,180) if _conf["cnt"] >= CONFIRM_N-1 else (160,160,160))
    txt(f"area:  {int(_conf['area'])}px", 8)
    txt(f"exec>= {AREA_EXEC}px", 9, (80,80,80))

    txt("─ ULTIMO CMD ─", 11, (70,70,70))
    ult = _nav["ultimo_executado"] or "-"
    txt(f"{ult}", 12, LABEL_BGR.get(ult, (160,160,160)))
    txt(f"mot:{CMD['mot']}  srv:{CMD['srv']}", 13)
    txt(f"brk:{CMD['brk']} led:{CMD['led']} buz:{CMD['buz']}", 14)

    # Legenda de labels
    txt("─ LEGENDA ─", 16, (70,70,70))
    for i, (lbl_k, bgr) in enumerate(LABEL_BGR.items()):
        txt(f"  {lbl_k}", 17+i, bgr, 0.34)

    # Junta frame + painel
    out_full = np.hstack([out, panel])
    return out_full


# ================================================================
#  [13] LOOP PRINCIPAL
# ================================================================

def main(usar_camera=False):
    global _ser

    src = CAM_IDX if usar_camera else VIDEO
    print(f"[CAM] {'Camera' if usar_camera else 'Video'}: {src}", flush=True)

    cap = cv2.VideoCapture(
        src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY
    )
    if not cap.isOpened():
        print("[ERRO] Não abriu fonte de vídeo.")
        sys.exit(1)

    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    _ser = conectar_serial()

    fps_timer = time.monotonic()
    fps_cnt = fps = 0

    print("[OK] Rodando — Q para sair\n", flush=True)

    while True:
        ret, frame_big = cap.read()
        if not ret:
            # Vídeo chegou ao fim → reinicia
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Redimensionar
        h = round(frame_big.shape[0] / PROP)
        w = round(frame_big.shape[1] / PROP)
        frame = cv2.resize(frame_big, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Decrement cooldown ────────────────────────────────────
        if _nav["cooldown"] > 0:
            _nav["cooldown"] -= 1

        # ── Tick da ação ativa ────────────────────────────────────
        em_acao = tick_acao(_ser)

        # ── Localiza candidatas ───────────────────────────────────
        boxes = localizar_candidatas(frame)

        label_raw = None
        conf_raw  = 0.0
        best_bbox = None

        if boxes and not em_acao:
            # Testa apenas a maior candidata (mais próxima)
            x, y, bw, bh, area = boxes[0]
            crop = frame[y:y+bh, x:x+bw]
            label_raw, conf_raw, _ = classificar(crop, area)
            best_bbox = (x, y, bw, bh)

            # ── Confirmação ───────────────────────────────────────
            confirmado = atualizar_confirmacao(label_raw, area)
            if confirmado:
                executar_acao(confirmado, _ser)
        else:
            atualizar_confirmacao(None, 0.0)

        # ── Debug visual ──────────────────────────────────────────
        vis = desenhar(frame, boxes, label_raw, conf_raw, best_bbox, fps)
        cv2.imshow("Detector de Placas", vis)

        # ── FPS ───────────────────────────────────────────────────
        fps_cnt += 1
        if time.monotonic() - fps_timer >= 1.0:
            fps = fps_cnt; fps_cnt = 0
            fps_timer = time.monotonic()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Encerramento ──────────────────────────────────────────────
    CMD.update(mot=0, srv=127, buz=0, led=0, brk=1, dir=0, spd=0)
    enviar(CMD, _ser)
    if _ser: _ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)


# ================================================================
#  MODO CALIBRAÇÃO HSV  (python sign_detector.py --cal)
# ================================================================

def calibrar():
    """Ferramenta interativa para ajustar ranges HSV por cor."""
    cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("[CAL] Sem fonte."); return

    win = "Calibrar HSV — S=salvar  Q=sair"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 540)
    defaults = [("H_min",0),("S_min",80),("V_min",70),
                ("H_max",15),("S_max",255),("V_max",255)]
    for n, v in defaults:
        cv2.createTrackbar(n, win, v, 180 if "H" in n else 255,
                           lambda x: None)
    print("[CAL] Aponte para a placa. S=salvar  Q=sair")

    while True:
        ret, f = cap.read()
        if not ret: break
        f   = cv2.resize(f, (f.shape[1]//PROP, f.shape[0]//PROP))
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lo  = np.array([cv2.getTrackbarPos(n, win)
                        for n in ("H_min","S_min","V_min")])
        hi  = np.array([cv2.getTrackbarPos(n, win)
                        for n in ("H_max","S_max","V_max")])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(f, f, mask=mask)
        cv2.putText(result, f"lo={lo.tolist()} hi={hi.tolist()}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,180), 1)
        cv2.imshow(win, np.hstack([f, result]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            print(f"\n[CAL] Resultado:")
            print(f'      np.array({lo.tolist()}), np.array({hi.tolist()})')
            print("      Cole em HSV[] no topo do arquivo.\n")
        elif k == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


# ================================================================

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if   arg == "--cam": main(usar_camera=True)
    elif arg == "--cal": calibrar()
    else:                main(usar_camera=False)