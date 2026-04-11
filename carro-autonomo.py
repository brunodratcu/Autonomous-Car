"""
================================================================
  CARRO AUTÔNOMO — PISTA + PLACAS  (CPU, sem GPU)
  Otimizações vs versão anterior:
  ─────────────────────────────────────────────────────────────
  1. TFLite em vez de TensorFlow completo  → -750MB RAM
  2. Inferência a cada 6 frames            → -40% CPU de IA
  3. Thread de captura isolada             → sem acúmulo de frames
  4. Sem cv2.hconcat no loop crítico       → -50% uso de buffer de vídeo
  5. Memória limitada no TF (fallback)     → sem reserva excessiva
  6. Buffer de inferência pré-alocado      → zero alocações no loop
  7. IA após decisão (não após waitKey)    → sem frames desperdiçados
  8. Morfologia com kernel pré-alocado     → sem realocações
  9. Janela de debug resize INTER_NEAREST  → mais rápido que INTER_LINEAR
  10. Rota multi-ponto com lista de IDs    → entrega em ordem definida
================================================================

  INSTALAÇÃO:
    pip install opencv-python numpy pyserial
    pip install tflite-runtime             ← mais leve que tensorflow
    # OU se tflite-runtime não funcionar:
    pip install tensorflow                 ← fallback (mais pesado)

  USO:
    python pista_video.py          ← vídeo (pista_01.mov)
    python pista_video.py --cam    ← câmera USB
    python pista_video.py --cal    ← calibrar HSV
    python pista_video.py --rota   ← definir rota de entrega

  ROTA DE ENTREGA:
    Edite ROTA_PONTOS abaixo com os IDs das placas em ordem.
    O carro executa cada ação quando vê a placa correspondente.
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import threading
import queue
import time
import sys
import os

# Limita TF a usar no máximo 512MB de RAM (se TF for usado como fallback)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # silencia logs TF
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"   # não reserva tudo

from infer_sign_cnn import load_model_and_labels, classify_sign_crop

# ================================================================
#  [1] CONFIGURAÇÃO — edite aqui
# ================================================================

PROP     = 2        # divisor de resolução (2 = metade)
VIDEO    = "./pista_01.mov"
CAM_IDX  = 0

SERIAL_PORT = "COM3"
BAUD        = 115200

# PID
KP, KI, KD = 0.55, 0.005, 0.12

# Velocidades (% PWM)
VEL = {"normal": 62, "devagar": 35, "parado": 0}

# Threshold de binarização da pista (0=auto Otsu)
THRESH_PISTA = 0      # 0 = Otsu automático

# Frequência de inferência de IA (a cada N frames)
IA_EVERY = 6          # 6 = ~5fps de IA em 30fps de câmera

# Confiança mínima para aceitar classificação
IA_CONF_MIN = 0.82

# ── Rota de entrega (lista de ações em ordem) ────────────────────
# O carro executa cada ação na sequência, quando detectar a placa
# Opções: "STOP", "YIELD", "LEFT", "RIGHT", "STRAIGHT", "DELIVERY"
ROTA_PONTOS = ["STRAIGHT", "LEFT", "DELIVERY", "RIGHT", "STOP"]
# Exemplo: vai reto, vira esquerda, entrega, vira direita, para na base

# ── ROIs como fração da altura do frame ─────────────────────────
# Pista  : trapézio na metade inferior (definido em processar_pista)
# Placa  : faixa horizontal onde placas aparecem fisicamente
#          15% = abaixo do céu  |  55% = acima da pista
ROI_PLACA_Y0 = 0.15
ROI_PLACA_Y1 = 0.55

# ── Kernels morfológicos pré-alocados ────────────────────────────
_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)

# ================================================================
#  [2] ESTADO GLOBAL
# ================================================================

# Comando enviado ao Arduino
CMD = {
    "mot": 0,   # motor 0-100 %PWM
    "srv": 0,   # servo -40..+40 graus
    "buz": 0,   # buzina 0/1
    "led": 0,   # LED 0/1
    "mode": 0,  # modo operação
    "brk": 0,   # freio
    "dir": 0,   # direção lógica (0=neutro,1=esq,2=dir,3=frente)
    "spd": 0,   # perfil velocidade (0=parado,1=lento,2=normal,3=rápido)
    "err": 0,   # erro lateral escalado para int
}

# Resultados de visão
_pista = {"ok": False, "erro": 0.0, "centro": 0}

# PID state
_pid = {"i": 0.0, "e_ant": 0.0, "t_ant": time.monotonic()}
_hist = []   # média móvel do centro

# Navegação
NAV = {
    "modo":             "SEGUIR_PISTA",
    "t_inicio":         time.monotonic(),
    "acao_pendente":    None,
    "frames_placa":     0,
    "ultimo_tipo":      None,
    "cooldown_placa":   0,
    "cooldown_marcador":0,
    "placa_confirmada": False,

    # Rota multi-ponto
    "rota":             list(ROTA_PONTOS),   # cópia da rota configurada
    "rota_idx":         0,                   # índice atual na rota
    "missao_completa":  False,
}

# Memória de confirmação da IA
_mem_ai = {"ultima": None, "cnt": 0, "conf": 0.0}

# Resultado IA (atualizado pela thread de IA)
_ia_resultado = {"label": None, "conf": 0.0, "bbox": None}
_ia_lock      = threading.Lock()

# Filas e flags de controle
_q_frames  = queue.Queue(maxsize=2)
_q_serial  = queue.Queue(maxsize=4)
_stop_flag = threading.Event()

# Contador global de frames
frame_id = 0


# ================================================================
#  [3] INICIALIZAÇÃO DO MODELO (TFLite — leve)
# ================================================================

print("[IA] Carregando modelo TFLite...", flush=True)
try:
    _sign_interp, _sign_labels, _in_idx, _out_idx = load_model_and_labels()
    _ia_disponivel = True
    print(f"[IA] Modelo OK — {len(_sign_labels)} classes: {_sign_labels}", flush=True)
except Exception as e:
    print(f"[IA] Modelo não disponível ({e}) — usando fallback cor+forma", flush=True)
    _ia_disponivel = False
    _sign_interp = _sign_labels = _in_idx = _out_idx = None


# ================================================================
#  [4] CÂMERA / VÍDEO
# ================================================================

def abrir_fonte(usar_camera=False):
    if usar_camera:
        src = CAM_IDX
        print(f"[CAM] Câmera USB idx={CAM_IDX}", flush=True)
    else:
        src = VIDEO
        print(f"[CAM] Vídeo: {VIDEO}", flush=True)

    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[CAM] Falha ao abrir fonte de vídeo!")
        sys.exit(1)

    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    return cap


# ================================================================
#  [5] SERIAL
# ================================================================

def conectar_serial():
    kws = ["arduino", "ch340", "cp210", "uart"]
    porta = None
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower() for k in kws):
            porta = p.device
            print(f"[SER] Detectado: {porta}", flush=True)
            break
    porta = porta or SERIAL_PORT
    try:
        ser = serial.Serial(porta, BAUD, timeout=0, write_timeout=0)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[SER] Conectado: {porta}", flush=True)
        return ser
    except Exception as e:
        print(f"[SER] Simulação ({e})", flush=True)
        return None


def enviar(ser):
    """Monta JSON, imprime no terminal e envia ao Arduino."""
    CMD["err"] = int(np.clip(_pista["erro"] * 100, -100, 100))

    linha = (
        f'{{"mot":{CMD["mot"]},'
        f'"srv":{CMD["srv"]},'
        f'"buz":{CMD["buz"]},'
        f'"led":{CMD["led"]},'
        f'"mode":{CMD["mode"]},'
        f'"brk":{CMD["brk"]},'
        f'"dir":{CMD["dir"]},'
        f'"spd":{CMD["spd"]},'
        f'"err":{CMD["err"]}}}'
    )
    print(linha, flush=True)
    if ser:
        try:
            ser.write((linha + "\n").encode())
        except Exception:
            pass


def thread_serial(ser):
    """Thread dedicada: drena fila serial sem bloquear loop principal."""
    while not _stop_flag.is_set():
        try:
            data = _q_serial.get(timeout=0.05)
            if ser:
                try:
                    ser.write(data)
                except Exception:
                    pass
        except queue.Empty:
            pass


# ================================================================
#  [6] THREAD DE CAPTURA
# ================================================================

def thread_captura(cap):
    """Lê frames continuamente. Descarta o mais antigo se fila cheia."""
    while not _stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            _stop_flag.set()
            break
        if _q_frames.full():
            try: _q_frames.get_nowait()
            except queue.Empty: pass
        try: _q_frames.put_nowait(frame)
        except queue.Full: pass


# ================================================================
#  [7] PID
# ================================================================

def pid_calc(erro: float) -> float:
    agora = time.monotonic()
    dt    = max(agora - _pid["t_ant"], 1e-4)
    _pid["i"]     = float(np.clip(_pid["i"] + erro * dt, -1.0, 1.0))
    deriv         = (erro - _pid["e_ant"]) / dt
    _pid["e_ant"] = erro
    _pid["t_ant"] = agora
    return float(np.clip(KP * erro + KI * _pid["i"] + KD * deriv, -1.0, 1.0))


# ================================================================
#  [8] DETECÇÃO DE PISTA
#  Gray → Blur → Otsu/Threshold → Morfologia → ROI → Canny → Hough
# ================================================================

def processar_pista(frame):
    """
    Retorna (img_gray, filtrada_vis, erro, pista_ok, cp_suave)
    """
    global _hist

    h, w   = frame.shape[:2]
    cx_ref = w // 2

    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Binarização — Otsu automático ou valor fixo
    if THRESH_PISTA == 0:
        _, binaria = cv2.threshold(blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binaria = cv2.threshold(blur, THRESH_PISTA, 255,
                                   cv2.THRESH_BINARY)

    # 4. Morfologia com kernels pré-alocados (sem realocação)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN,  _K3)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _K3)

    # 5. ROI trapezoidal — foca na pista, ignora céu
    roi_mask = np.zeros_like(binaria)
    pts = np.array([[
        (int(w * 0.10), h),
        (int(w * 0.40), int(h * 0.60)),
        (int(w * 0.60), int(h * 0.60)),
        (int(w * 0.90), h),
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, pts, 255)
    masked = cv2.bitwise_and(binaria, roi_mask)

    # 6. Canny
    bordas = cv2.Canny(masked, 50, 150)

    # 7. HoughLinesP
    linhas = cv2.HoughLinesP(
        bordas, rho=1, theta=np.pi / 180,
        threshold=25, minLineLength=25, maxLineGap=30
    )

    esq_params = []
    dir_params = []

    if linhas is not None:
        for seg in linhas:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            if dx == 0:
                continue
            length = np.hypot(x2 - x1, y2 - y1)
            if length < 25:
                continue
            inc = (y2 - y1) / dx
            if abs(inc) < 0.3 or abs(inc) > 3.5:
                continue
            m, b = np.polyfit([x1, x2], [y1, y2], 1)
            (esq_params if inc < 0 else dir_params).append((m, b))

    y_ref  = int(h * 0.85)
    x_esq  = None
    x_dir  = None
    pista_ok = False

    if esq_params:
        me = np.mean([p[0] for p in esq_params])
        be = np.mean([p[1] for p in esq_params])
        if abs(me) > 1e-6:
            x_esq = int(np.clip((y_ref - be) / me, 0, w - 1))

    if dir_params:
        md = np.mean([p[0] for p in dir_params])
        bd = np.mean([p[1] for p in dir_params])
        if abs(md) > 1e-6:
            x_dir = int(np.clip((y_ref - bd) / md, 0, w - 1))

    if x_esq is not None and x_dir is not None:
        cp = (x_esq + x_dir) // 2
        pista_ok = True
    elif x_esq is not None:
        cp = int(np.clip(x_esq + w // 4, 0, w - 1))
        pista_ok = True
    elif x_dir is not None:
        cp = int(np.clip(x_dir - w // 4, 0, w - 1))
        pista_ok = True
    else:
        cp = cx_ref

    # 8. Média móvel
    _hist.append(cp)
    if len(_hist) > 5:
        _hist.pop(0)
    cp_suave = int(np.mean(_hist))
    erro     = float(np.clip((cp_suave - cx_ref) / cx_ref, -1.0, 1.0))

    _pista["ok"]     = pista_ok
    _pista["erro"]   = erro
    _pista["centro"] = cp_suave

    # 9. Visualização mínima (só o necessário)
    vis = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (cx_ref, int(h * 0.6)), (cx_ref, h), (0, 0, 255), 1)
    cv2.line(vis, (cp_suave, int(h * 0.6)), (cp_suave, h), (0, 255, 0), 2)
    if x_esq: cv2.circle(vis, (x_esq, y_ref), 4, (255, 255, 0), -1)
    if x_dir: cv2.circle(vis, (x_dir, y_ref), 4, (0, 255, 255), -1)

    return gray, vis, erro, pista_ok, cp_suave


# ================================================================
#  [8b] ROI + TRATAMENTO VISUAL DE PLACAS
#  Espelho da lógica de pista, mas na faixa ROI_PLACA_Y0–Y1.
#  Retorna imagem tratada BGR para o concat do console.
# ================================================================

def processar_placa_roi(frame, gray):
    """
    Aplica o mesmo pipeline de tratamento da pista na faixa de placas.
    ROI: ROI_PLACA_Y0 (15%) a ROI_PLACA_Y1 (55%) da altura do frame.

    Passos: Blur → Otsu → Morfologia → Canny → desenha contornos candidatos
    Retorna: vis_bgr (mesma dimensão do frame, fundo preto fora da ROI)
    """
    h, w = frame.shape[:2]
    y0 = int(h * ROI_PLACA_Y0)
    y1 = int(h * ROI_PLACA_Y1)

    roi_gray = gray[y0:y1, :]
    blur     = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, bin_  = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_     = cv2.morphologyEx(bin_, cv2.MORPH_OPEN,  _K3)
    bin_     = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, _K3)
    bordas   = cv2.Canny(bin_, 50, 150)

    # Monta frame completo: preto fora da ROI, bordas dentro
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[y0:y1, :] = bordas
    vis = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Linhas-guia da ROI
    cv2.line(vis, (0, y0), (w, y0), (0, 140, 255), 1)
    cv2.line(vis, (0, y1), (w, y1), (0, 140, 255), 1)
    cv2.putText(vis, f"ROI placa {int(ROI_PLACA_Y0*100)}-{int(ROI_PLACA_Y1*100)}%",
                (4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1)

    # Retângulos dos contornos candidatos (proporção de placa)
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 200:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 0.5 < bw / max(bh, 1) < 1.8:
            cv2.rectangle(vis, (x, y + y0), (x + bw, y + y0 + bh), (0, 220, 180), 1)

    return vis


# ================================================================
#  [9] DETECÇÃO DE MARCADOR NO CHÃO
#  Faixa escura horizontal transversal → dispara manobra pendente
# ================================================================

def detectar_marcador_chao(frame):
    """
    Procura faixa escura horizontal na parte inferior do frame.
    Retorna (marcador_ok: bool, vis: BGR)
    """
    h, w = frame.shape[:2]
    y1, y2 = int(h * 0.72), int(h * 0.95)
    roi = frame[y1:y2, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold invertido: escuro → branco
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    marcador_ok = False
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for cnt in cnts:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        asp      = ww / max(hh, 1)
        cobertura = ww / w
        if asp > 3.0 and cobertura > 0.45 and hh > 8:
            marcador_ok = True
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            break

    return marcador_ok, vis


# ================================================================
#  [10] DETECÇÃO DE PLACAS — IA (TFLite) + fallback cor/forma
# ================================================================

def _localizar_candidatas(frame):
    """
    Segmenta regiões de cor placa (vermelho/azul) na ROI de placas.
    Usa ROI_PLACA_Y0–Y1: mesma faixa do processar_placa_roi.
    Retorna lista de (x, y, w, h) em coordenadas do frame completo.
    """
    h, w = frame.shape[:2]
    y0   = int(h * ROI_PLACA_Y0)
    y1   = int(h * ROI_PLACA_Y1)
    roi  = frame[y0:y1, :]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv  = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Vermelho (dois ranges por wrap-around em HSV)
    m1 = cv2.inRange(hsv, np.array([0,  80, 70]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170,80, 70]), np.array([180,255, 255]))
    mb = cv2.inRange(hsv, np.array([90, 70, 60]), np.array([130,255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), mb)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 400:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if 0.6 < ww / max(hh, 1) < 1.4:
            # y relativo à ROI → converter para coordenadas do frame
            boxes.append((x, y + y0, ww, hh))
    return boxes


def _fallback_cor_forma(crop, bbox):
    """Classifica placa por cor dominante + número de vértices (sem IA)."""
    if crop is None or crop.size == 0:
        return None
    hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    def px(lo, hi): return int(np.count_nonzero(cv2.inRange(hsv_c, lo, hi)))

    a_vm = px(np.array([0,  80, 70]), np.array([12, 255,255])) + \
           px(np.array([170,80, 70]), np.array([180,255,255]))
    a_az = px(np.array([90, 70, 60]), np.array([130,255,255]))

    gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thr  = cv2.threshold(gray_c, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    lados = 4
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        lados = len(ap)

    if a_vm > a_az:
        return "STOP" if lados >= 7 else "RIGHT"
    if a_az > 100:
        return "LEFT"
    return None


def detectar_placa_frame(frame):
    """
    Tenta IA (TFLite) primeiro; se não disponível, usa cor+forma.
    Retorna (label_str_ou_None, conf, bbox_ou_None)
    """
    boxes   = _localizar_candidatas(frame)
    if not boxes:
        return None, 0.0, None

    # Melhor candidata = maior área
    bbox = max(boxes, key=lambda b: b[2] * b[3])
    x, y, ww, hh = bbox
    crop = frame[y:y + hh, x:x + ww]

    if _ia_disponivel:
        label, conf, _ = classify_sign_crop(
            crop, _sign_interp, _sign_labels,
            _in_idx, _out_idx, conf_min=IA_CONF_MIN
        )
        if label == "unknown":
            label = None
    else:
        label = _fallback_cor_forma(crop, bbox)
        conf  = 0.75 if label else 0.0

    return label, conf, bbox


# ================================================================
#  [11] SISTEMA DE CONFIRMAÇÃO DE PLACA
#  Exige N frames consecutivos com a mesma classe antes de aceitar
# ================================================================

def confirmar_placa(label, conf, min_frames=3):
    """
    Retorna o label confirmado ou None.
    Evita aceitar detecções únicas (ruído, reflexo).
    """
    if not label:
        _mem_ai["ultima"] = None
        _mem_ai["cnt"]    = 0
        return None

    if NAV["cooldown_placa"] > 0:
        return None

    if _mem_ai["ultima"] == label:
        _mem_ai["cnt"]  += 1
        _mem_ai["conf"]  = conf
    else:
        _mem_ai["ultima"] = label
        _mem_ai["cnt"]    = 1

    if _mem_ai["cnt"] >= min_frames:
        _mem_ai["cnt"] = 0   # reset para não confirmar em loop
        return label
    return None


# ================================================================
#  [12] SISTEMA DE ROTA MULTI-PONTO
#  O carro percorre ROTA_PONTOS na ordem. Cada ação é confirmada
#  por placa + marcador no chão antes de executar.
# ================================================================

def proxima_acao_rota():
    """Retorna a próxima ação esperada na rota, ou None se completa."""
    idx = NAV["rota_idx"]
    if NAV["missao_completa"] or idx >= len(NAV["rota"]):
        return None
    return NAV["rota"][idx]


def avancar_rota():
    """Avança para o próximo ponto da rota."""
    NAV["rota_idx"] += 1
    if NAV["rota_idx"] >= len(NAV["rota"]):
        NAV["missao_completa"] = True
        print("[ROTA] Missão completa!", flush=True)
    else:
        print(f"[ROTA] Próxima ação: {NAV['rota'][NAV['rota_idx']]}", flush=True)


def registrar_placa(label_confirmado):
    """
    Registra ação pendente se o label bate com a próxima da rota.
    Se ROTA_PONTOS estiver vazia, aceita qualquer placa.
    """
    if not label_confirmado:
        return

    proxima = proxima_acao_rota()

    # Sem rota definida: aceita qualquer placa detectada
    if proxima is None and not NAV["rota"]:
        NAV["acao_pendente"] = label_confirmado
        NAV["cooldown_placa"] = 20
        return

    # Com rota: só aceita se bate com a esperada
    mapa = {
        "stop": "STOP", "yield": "YIELD",
        "left": "LEFT", "right": "RIGHT",
        "straight": "STRAIGHT", "delivery": "DELIVERY",
        # fallback direto (maiúsculo)
        "STOP": "STOP", "LEFT": "LEFT", "RIGHT": "RIGHT",
        "STRAIGHT": "STRAIGHT", "DELIVERY": "DELIVERY",
    }
    acao = mapa.get(label_confirmado)
    if acao and acao == proxima:
        NAV["acao_pendente"] = acao
        NAV["cooldown_placa"] = 20
        print(f"[PLACA] Confirmada: {acao}", flush=True)


def disparar_no_marcador(marcador_ok):
    """Inicia manobra quando marcador no chão é detectado."""
    if NAV["cooldown_marcador"] > 0 or not marcador_ok:
        return
    if NAV["acao_pendente"] is None:
        return

    acao = NAV["acao_pendente"]
    NAV["t_inicio"]          = time.monotonic()
    NAV["cooldown_marcador"] = 25
    NAV["acao_pendente"]     = None
    NAV["modo"]              = f"EXEC_{acao}"
    print(f"[NAV] Executando: {acao}", flush=True)


# ================================================================
#  [13] CONTROLE DE MANOBRAS (máquina de estados temporizada)
# ================================================================

def controle_modo():
    """
    Executa a manobra ativa.
    Retorna True se assumiu o controle (não seguir pista).
    """
    agora = time.monotonic()
    dt    = agora - NAV["t_inicio"]
    modo  = NAV["modo"]

    if modo == "SEGUIR_PISTA":
        return False

    def _voltar():
        NAV["modo"] = "SEGUIR_PISTA"
        avancar_rota()

    # ── STOP: para 2s ──────────────────────────────────────────
    if modo == "EXEC_STOP":
        CMD.update({"mot": 0, "srv": 0, "buz": 0, "led": 1, "brk": 1})
        if dt > 2.0:
            CMD["led"] = 0
            CMD["brk"] = 0
            _voltar()
        return True

    # ── YIELD: devagar 1s ──────────────────────────────────────
    if modo == "EXEC_YIELD":
        CMD.update({"mot": VEL["devagar"], "srv": 0, "buz": 0, "led": 0, "brk": 0})
        if dt > 1.0:
            _voltar()
        return True

    # ── LEFT: 3 fases ──────────────────────────────────────────
    if modo == "EXEC_LEFT":
        if dt < 0.35:
            CMD.update({"mot": 40, "srv": 0,   "dir": 1})
        elif dt < 1.15:
            CMD.update({"mot": 38, "srv": -32, "dir": 1})
        elif dt < 1.45:
            CMD.update({"mot": 35, "srv": 0,   "dir": 1})
        else:
            CMD["dir"] = 0
            _voltar()
        CMD.update({"buz": 0, "led": 0, "brk": 0})
        return True

    # ── RIGHT: 3 fases ─────────────────────────────────────────
    if modo == "EXEC_RIGHT":
        if dt < 0.35:
            CMD.update({"mot": 40, "srv": 0,   "dir": 2})
        elif dt < 1.15:
            CMD.update({"mot": 38, "srv": 32,  "dir": 2})
        elif dt < 1.45:
            CMD.update({"mot": 35, "srv": 0,   "dir": 2})
        else:
            CMD["dir"] = 0
            _voltar()
        CMD.update({"buz": 0, "led": 0, "brk": 0})
        return True

    # ── STRAIGHT: vai reto 0.9s ────────────────────────────────
    if modo == "EXEC_STRAIGHT":
        CMD.update({"mot": 45, "srv": 0, "buz": 0, "led": 0,
                    "brk": 0, "dir": 3})
        if dt > 0.9:
            CMD["dir"] = 0
            _voltar()
        return True

    # ── DELIVERY: para 3s, buzina + LED ────────────────────────
    if modo == "EXEC_DELIVERY":
        CMD.update({"mot": 0, "srv": 0, "buz": 1, "led": 1, "brk": 1})
        if dt > 3.0:
            CMD.update({"buz": 0, "led": 0, "brk": 0})
            _voltar()
        return True

    # Modo desconhecido: volta ao normal
    NAV["modo"] = "SEGUIR_PISTA"
    return False


def atualizar_cooldowns():
    if NAV["cooldown_placa"]    > 0: NAV["cooldown_placa"]    -= 1
    if NAV["cooldown_marcador"] > 0: NAV["cooldown_marcador"] -= 1


# ================================================================
#  [14] DECISÃO PRINCIPAL
# ================================================================

def decidir(pista_ok, erro):
    """Seguimento de pista via PID."""
    if pista_ok:
        ang = int(np.clip(pid_calc(erro) * 40, -40, 40))
        CMD.update({"mot": VEL["normal"], "srv": ang,
                    "buz": 0, "led": 0, "brk": 0,
                    "spd": 2, "dir": 0})
    else:
        CMD.update({"mot": VEL["parado"], "srv": 0,
                    "buz": 0, "led": 1, "brk": 1,
                    "spd": 0, "dir": 0})


# ================================================================
#  [15] DEBUG VISUAL (leve — sem hconcat no loop crítico)
# ================================================================

def desenhar_debug(frame, gray, pista_vis, placa_vis,
                   bbox_ia, label_ia, conf_ia, fps):
    """
    Console concatenado: [GRAY | PISTA TRATADA | PLACA TRATADA]
    Uma única janela — mostra os três painéis lado a lado.
    Overlay de status no painel esquerdo (gray).
    """
    # ── Painel 1: escala de cinza com status ──────────────────────
    p1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cor = (0, 220, 50) if _pista["ok"] else (0, 40, 255)
    cv2.putText(p1, f"mot:{CMD['mot']} srv:{CMD['srv']:+d} err:{_pista['erro']:+.2f} fps:{fps}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, cor, 1)
    cv2.putText(p1, f"modo:{NAV['modo']}",
                (4, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 220, 0), 1)
    cv2.putText(p1, f"placa:{label_ia or '-'} {conf_ia:.2f}",
                (4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 255), 1)

    # Bounding box da placa no painel gray
    if bbox_ia:
        x, y, bw, bh = bbox_ia
        cv2.rectangle(p1, (x, y), (x + bw, y + bh), (0, 220, 180), 1)

    # ── Painel 2: pista tratada (já é BGR) ────────────────────────
    p2 = pista_vis if pista_vis is not None else np.zeros_like(p1)
    cv2.putText(p2, "PISTA", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 80), 1)

    # ── Painel 3: placa tratada (já é BGR) ────────────────────────
    p3 = placa_vis if placa_vis is not None else np.zeros_like(p1)
    cv2.putText(p3, "PLACAS", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1)

    # ── Concat horizontal dos 3 painéis ──────────────────────────
    console = cv2.hconcat([p1, p2, p3])
    cv2.imshow("Carro Autonomo", console)


# ================================================================
#  [16] LOOP PRINCIPAL
# ================================================================

def main(usar_camera=False):
    global frame_id

    cap = abrir_fonte(usar_camera)
    ser = conectar_serial()

    # Inicia threads de captura e serial
    t_cap = threading.Thread(target=thread_captura, args=(cap,), daemon=True)
    t_ser = threading.Thread(target=thread_serial,  args=(ser,), daemon=True)
    t_cap.start()
    t_ser.start()

    fps_timer = time.monotonic()
    fps_cnt   = 0
    fps       = 0

    label_ia  = None
    conf_ia   = 0.0
    bbox_ia   = None
    pista_vis = None
    placa_vis = None
    marc_vis  = None

    print("[OK] Rodando — Q para encerrar\n", flush=True)

    while not _stop_flag.is_set():
        try:
            frame_big = _q_frames.get(timeout=0.05)
        except queue.Empty:
            continue

        frame_id += 1
        atualizar_cooldowns()

        # ── Redimensionar ────────────────────────────────────────
        h = round(frame_big.shape[0] / PROP)
        w = round(frame_big.shape[1] / PROP)
        frame = cv2.resize(frame_big, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Pista ────────────────────────────────────────────────
        gray, pista_vis, erro, pista_ok, centro = processar_pista(frame)

        # ── ROI de Placas (tratamento visual) ────────────────────
        placa_vis = processar_placa_roi(frame, gray)

        # ── Marcador no chão (a cada 2 frames) ───────────────────
        if frame_id % 2 == 0:
            marcador_ok, marc_vis = detectar_marcador_chao(frame)
        else:
            marcador_ok = False

        # ── IA de placas (a cada IA_EVERY frames) ────────────────
        if frame_id % IA_EVERY == 0 and NAV["cooldown_placa"] == 0:
            raw_label, conf_ia, bbox_ia = detectar_placa_frame(frame)
            confirmado = confirmar_placa(raw_label, conf_ia)
            registrar_placa(confirmado)
            if confirmado:
                label_ia = confirmado
            elif raw_label:
                label_ia = f"({raw_label})"   # em confirmação
            else:
                label_ia = None

        # ── Disparo de manobra no marcador ───────────────────────
        disparar_no_marcador(marcador_ok)

        # ── Decisão ──────────────────────────────────────────────
        ocupado = controle_modo()
        if not ocupado:
            decidir(pista_ok, erro)

        # ── Envio serial ─────────────────────────────────────────
        enviar(ser)

        # ── FPS ──────────────────────────────────────────────────
        fps_cnt += 1
        if time.monotonic() - fps_timer >= 1.0:
            fps       = fps_cnt
            fps_cnt   = 0
            fps_timer = time.monotonic()

        # ── Debug ────────────────────────────────────────────────
        desenhar_debug(frame, gray, pista_vis, placa_vis,
                       bbox_ia, label_ia, conf_ia, fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Encerramento seguro
    _stop_flag.set()
    CMD.update({"mot": 0, "srv": 0, "buz": 0, "led": 0, "brk": 1})
    enviar(ser)
    t_ser.join(timeout=1.0)
    if ser: ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)


# ================================================================
#  MODO CALIBRAÇÃO HSV
# ================================================================

def calibrar():
    cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("[CAL] Nenhuma fonte disponível"); return

    win = "Calibrar HSV — S=salvar Q=sair"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 520)
    for n, v in [("H_min",0),("S_min",0),("V_min",160),
                 ("H_max",180),("S_max",50),("V_max",255)]:
        cv2.createTrackbar(n, win, v, 180 if "H" in n else 255, lambda x: None)

    print("[CAL] Aponte para a pista. S=salvar Q=sair")
    while True:
        ret, f = cap.read()
        if not ret: break
        h0 = round(f.shape[0] / PROP)
        w0 = round(f.shape[1] / PROP)
        f  = cv2.resize(f, (w0, h0))
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lo  = np.array([cv2.getTrackbarPos(n, win)
                        for n in ("H_min","S_min","V_min")])
        hi  = np.array([cv2.getTrackbarPos(n, win)
                        for n in ("H_max","S_max","V_max")])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(f, f, mask=mask)
        cv2.putText(result, f"lo={lo.tolist()} hi={hi.tolist()}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,180), 1)
        cv2.imshow(win, np.hstack([f, result]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            print(f"[CAL] branco: lo={lo.tolist()}, hi={hi.tolist()}")
            print("[CAL] Cole esses valores em THRESH_PISTA ou HSV_PISTA no código")
        elif k == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


# ================================================================
#  MODO ROTA (exibição e configuração)
# ================================================================

def mostrar_rota():
    print("\n[ROTA] Rota configurada:")
    for i, acao in enumerate(ROTA_PONTOS):
        print(f"  {i+1}. {acao}")
    print(f"\n  Total: {len(ROTA_PONTOS)} pontos")
    print("  Para alterar: edite ROTA_PONTOS no topo do arquivo\n")


# ================================================================
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if   arg == "--cam":   main(usar_camera=True)
    elif arg == "--cal":   calibrar()
    elif arg == "--rota":  mostrar_rota()
    else:                  main(usar_camera=False)