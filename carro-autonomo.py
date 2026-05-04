"""
================================================================
  CARRO AUTÔNOMO — SPRINT 1 — PERCEPÇÃO + NAVEGAÇÃO BÁSICA
  ─────────────────────────────────────────────────────────────
  Arquitetura:
    • ROI dupla: pista (trapézio inferior) + placas (faixa central)
    • Pipeline visual com 8 filtros por canal
    • Análise lateral (esq/dir) independente → fusão central
    • Detecção de cantos (Harris + Shi-Tomasi) em ambas as ROIs
    • CNN TFLite para classificação de placas
    • PID suave para seguimento de pista
    • Máquina de estados + rota multi-ponto
    • Envio serial JSON ao Arduino
  ─────────────────────────────────────────────────────────────
  INSTALAÇÃO:
    pip install opencv-python numpy pyserial
    pip install tflite-runtime   (ou tensorflow como fallback)

  USO:
    python pista_video.py           → vídeo (pista_01.mov)
    python pista_video.py --cam     → câmera USB
    python pista_video.py --cal     → calibrar HSV
    python pista_video.py --rota    → listar rota
    python pista_video.py --debug   → janela de debug expandida
================================================================
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Tenta importar módulo de inferência
try:
    from INFER_SIGN_CNN import load_model_and_labels, classify_sign_crop
    _modulo_ia_ok = True
except ImportError:
    _modulo_ia_ok = False

# ================================================================
#  [1] CONFIGURAÇÃO GLOBAL
# ================================================================

PROP      = 2          # divisor de resolução (2 = metade)
VIDEO     = "videoplayback.mp4"
CAM_IDX   = 0

SERIAL_PORT = "COM3"
BAUD        = 115200

# ── PID ──────────────────────────────────────────────────────────
KP, KI, KD = 0.55, 0.005, 0.12

# ── Velocidades (%PWM) ────────────────────────────────────────────
VEL = {"normal": 62, "devagar": 35, "parado": 0}

# ── Thresholds ────────────────────────────────────────────────────
THRESH_PISTA    = 0     # 0 = Otsu automático
IA_EVERY        = 6     # inferência a cada N frames
IA_CONF_MIN     = 0.82  # confiança mínima CNN

# ── ROIs (frações da altura do frame) ────────────────────────────
#   Pista  : 60%–100% (trapézio, definido em pts abaixo)
#   Placas : 15%–55%  (faixa retangular)
ROI_PISTA_Y0    = 0.60
ROI_PLACA_Y0    = 0.15
ROI_PLACA_Y1    = 0.55

# ── Cantos (Harris) ───────────────────────────────────────────────
HARRIS_K        = 0.04
HARRIS_THRESH   = 0.01   # fração do máximo
CORNER_DILATE   = 3      # pixels de dilatação para visualização

# ── Kernels pré-alocados ─────────────────────────────────────────
_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)
_K7 = np.ones((7, 7), np.uint8)

# ── Rota multi-ponto ─────────────────────────────────────────────
ROTA_PONTOS = ["STRAIGHT", "LEFT", "DELIVERY", "RIGHT", "STOP"]

# ================================================================
#  [2] ESTADO GLOBAL
# ================================================================

CMD = {
    "mot": 0,    # motor 0–100 %PWM
    "srv": 127,  # servo 0–254 (127 = centro)
    "buz": 0,
    "led": 0,
    "mode": 0,
    "brk": 0,
    "dir": 0,    # 0=neutro 1=esq 2=dir 3=frente
    "spd": 0,
    "err": 0,
}

# Resultados da pipeline de pista
_pista = {
    "ok":       False,
    "erro":     0.0,
    "centro":   0,
    # análise lateral separada
    "x_esq":    None,   # posição X da borda esquerda
    "x_dir":    None,   # posição X da borda direita
    "ang_esq":  0.0,    # inclinação média (linhas esq)
    "ang_dir":  0.0,    # inclinação média (linhas dir)
    "cantos_pista": 0,  # qtd cantos Harris detectados na ROI pista
}

# Resultados de placas
_placa = {
    "label":    None,
    "conf":     0.0,
    "bbox":     None,
    "cantos":   0,      # qtd cantos na ROI placa
}

# PID
_pid  = {"i": 0.0, "e_ant": 0.0, "t_ant": time.monotonic()}
_hist = []   # média móvel do centro

# Navegação / máquina de estados
NAV = {
    "modo":              "SEGUIR_PISTA",
    "t_inicio":          time.monotonic(),
    "acao_pendente":     None,
    "cooldown_placa":    0,
    "cooldown_marcador": 0,
    "rota":              list(ROTA_PONTOS),
    "rota_idx":          0,
    "missao_completa":   False,
}

# Memória de confirmação IA
_mem_ai = {"ultima": None, "cnt": 0, "conf": 0.0}

# Filas e flags de controle
_q_frames  = queue.Queue(maxsize=2)
_stop_flag = threading.Event()
frame_id   = 0

# ================================================================
#  [3] CARGA DO MODELO
# ================================================================

print("[IA] Carregando modelo TFLite...", flush=True)
if _modulo_ia_ok:
    try:
        _sign_interp, _sign_labels, _in_idx, _out_idx = load_model_and_labels()
        _ia_disponivel = True
        print(f"[IA] OK — {len(_sign_labels)} classes: {_sign_labels}", flush=True)
    except Exception as e:
        print(f"[IA] Falha ({e}) — fallback cor+forma", flush=True)
        _ia_disponivel = False
        _sign_interp = _sign_labels = _in_idx = _out_idx = None
else:
    print("[IA] INFER_SIGN_CNN não encontrado — fallback ativo", flush=True)
    _ia_disponivel = False
    _sign_interp = _sign_labels = _in_idx = _out_idx = None

# ================================================================
#  [4] CÂMERA / VÍDEO
# ================================================================

def abrir_fonte(usar_camera=False):
    src = CAM_IDX if usar_camera else VIDEO
    print(f"[CAM] {'Câmera' if usar_camera else 'Vídeo'}: {src}", flush=True)
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
    """Monta JSON e envia ao Arduino."""
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


# ================================================================
#  [6] THREAD DE CAPTURA
# ================================================================

def thread_captura(cap):
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
#  [8] UTILITÁRIO DE FILTROS
#  Aplica sequência de aprimoramento sobre uma imagem grayscale.
#  Retorna (filtrada, canny, binaria)
# ================================================================

def pipeline_filtros(gray_roi, nome=""):
    """
    8 estágios de filtragem:
      1. CLAHE         — equalização adaptativa de contraste
      2. Gauss 5×5     — suavização (remove ruído de alta frequência)
      3. Otsu/Fixed    — binarização global adaptativa
      4. Morph Open    — remove pequenas manchas
      5. Morph Close   — fecha buracos na linha
      6. Dilatação     — engrossa bordas para Canny
      7. Canny         — extração de bordas
      8. Sobel X+Y     — gradiente de intensidade (auxiliar)
    """
    # 1. CLAHE
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq     = clahe.apply(gray_roi)

    # 2. Gaussian Blur
    blur   = cv2.GaussianBlur(eq, (5, 5), 0)

    # 3. Binarização Otsu
    _, bin_ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4 + 5. Morfologia
    aberto = cv2.morphologyEx(bin_, cv2.MORPH_OPEN,  _K3)
    fechado = cv2.morphologyEx(aberto, cv2.MORPH_CLOSE, _K3)

    # 6. Dilatação leve
    dil   = cv2.dilate(fechado, _K3, iterations=1)

    # 7. Canny
    canny  = cv2.Canny(dil, 40, 120)

    # 8. Sobel (combinado)
    sx    = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sy    = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sob   = np.uint8(np.clip(np.sqrt(sx**2 + sy**2) / 4, 0, 255))

    return fechado, canny, sob


# ================================================================
#  [9] DETECÇÃO DE CANTOS
#  Harris Corner Detector aplicado sobre a ROI.
#  Retorna (lista de pontos (x,y) em coords do frame, visualização)
# ================================================================

def detectar_cantos(gray_roi, y_offset=0, x_offset=0,
                    max_cantos=80, metodo="harris"):
    """
    metodo: 'harris' ou 'shi'
    Retorna (pts_frame, vis_roi_bgr)
    """
    vis = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

    if metodo == "shi":
        # Shi-Tomasi (mais estável, menos sensível)
        pts = cv2.goodFeaturesToTrack(
            gray_roi, maxCorners=max_cantos,
            qualityLevel=0.01, minDistance=5
        )
        corners = []
        if pts is not None:
            for p in pts:
                x, y = int(p[0][0]), int(p[0][1])
                corners.append((x + x_offset, y + y_offset))
                cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)
        return corners, vis

    # Harris
    gray_f = np.float32(gray_roi)
    dst    = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=HARRIS_K)
    dst    = cv2.dilate(dst, None)

    thresh_val = HARRIS_THRESH * dst.max() if dst.max() > 0 else 1
    corners    = []
    ys, xs     = np.where(dst > thresh_val)

    # Amostra até max_cantos
    if len(xs) > max_cantos:
        idx = np.random.choice(len(xs), max_cantos, replace=False)
        xs, ys = xs[idx], ys[idx]

    for x, y in zip(xs, ys):
        corners.append((int(x) + x_offset, int(y) + y_offset))
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Marca hotspots (cantos de alta resposta)
    strong = np.column_stack(np.where(dst > 0.1 * dst.max()))
    for sy_c, sx_c in strong[:20]:
        cv2.circle(vis, (int(sx_c), int(sy_c)), 4, (255, 50, 50), 1)

    return corners, vis


# ================================================================
#  [10] PIPELINE ROI PISTA
#  Análise bilateral (esq/dir) + fusão + PID
# ================================================================

def processar_pista(frame):
    """
    Pipeline completo da pista:
      • Aplica pipeline_filtros na ROI trapezoidal
      • Separa linhas em esquerda e direita (inclinação)
      • Calcula x_esq e x_dir independentemente
      • Detecta cantos Harris nos dois lados
      • Funde os dois lados para calcular o centro e o erro
    Retorna (gray, vis_bgr, erro, pista_ok, centro_suave)
    """
    global _hist

    h, w   = frame.shape[:2]
    cx_ref = w // 2

    # ── Região trapezoidal da pista ──────────────────────────────
    pts_trap = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.38), int(h * ROI_PISTA_Y0)),
        (int(w * 0.62), int(h * ROI_PISTA_Y0)),
        (int(w * 0.95), h),
    ]], dtype=np.int32)

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Máscara trapezoidal
    trap_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(trap_mask, pts_trap, 255)
    gray_roi  = cv2.bitwise_and(gray_full, trap_mask)

    # ── 8 filtros ────────────────────────────────────────────────
    binaria, canny, sobel = pipeline_filtros(gray_roi, nome="pista")

    # ── Separação esq/dir por metade vertical ────────────────────
    esq_mask = np.zeros((h, w), np.uint8); esq_mask[:, :cx_ref] = 255
    dir_mask  = np.zeros((h, w), np.uint8); dir_mask[:, cx_ref:] = 255

    canny_esq = cv2.bitwise_and(canny, esq_mask)
    canny_dir  = cv2.bitwise_and(canny, dir_mask)

    # ── HoughLinesP bilateral ─────────────────────────────────────
    def hough(img):
        return cv2.HoughLinesP(
            img, rho=1, theta=np.pi / 180,
            threshold=20, minLineLength=20, maxLineGap=35
        )

    linhas_esq = hough(canny_esq)
    linhas_dir  = hough(canny_dir)

    def extrair_params(linhas, lado):
        """Filtra e retorna lista de (m, b) para o lado dado."""
        params = []
        if linhas is None:
            return params
        for seg in linhas:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            if dx == 0:
                continue
            length = np.hypot(dx, y2 - y1)
            if length < 20:
                continue
            inc = (y2 - y1) / dx
            # Esquerda: inclinação negativa; Direita: positiva
            if lado == "esq" and not (-3.5 < inc < -0.25):
                continue
            if lado == "dir" and not (0.25 < inc < 3.5):
                continue
            m, b = np.polyfit([x1, x2], [y1, y2], 1)
            params.append((m, b))
        return params

    params_esq = extrair_params(linhas_esq, "esq")
    params_dir  = extrair_params(linhas_dir,  "dir")

    y_ref    = int(h * 0.88)
    x_esq    = None
    x_dir    = None
    ang_esq  = 0.0
    ang_dir  = 0.0
    pista_ok = False

    if params_esq:
        me   = float(np.mean([p[0] for p in params_esq]))
        be   = float(np.mean([p[1] for p in params_esq]))
        ang_esq = float(np.degrees(np.arctan(me)))
        if abs(me) > 1e-6:
            x_esq = int(np.clip((y_ref - be) / me, 0, w - 1))

    if params_dir:
        md   = float(np.mean([p[0] for p in params_dir]))
        bd   = float(np.mean([p[1] for p in params_dir]))
        ang_dir = float(np.degrees(np.arctan(md)))
        if abs(md) > 1e-6:
            x_dir = int(np.clip((y_ref - bd) / md, 0, w - 1))

    # ── Fusão bilateral → centro ──────────────────────────────────
    if x_esq is not None and x_dir is not None:
        # Ambos os lados visíveis: centro real
        cp       = (x_esq + x_dir) // 2
        pista_ok = True
    elif x_esq is not None:
        # Só borda esquerda: estima centro com deslocamento
        cp       = int(np.clip(x_esq + int(w * 0.28), 0, w - 1))
        pista_ok = True
    elif x_dir is not None:
        # Só borda direita
        cp       = int(np.clip(x_dir - int(w * 0.28), 0, w - 1))
        pista_ok = True
    else:
        cp = cx_ref   # fallback: sem pista → fica no centro

    # ── Cantos Harris na ROI pista ────────────────────────────────
    y_roi0 = int(h * ROI_PISTA_Y0)
    gray_strip = gray_full[y_roi0:, :]
    corners_pista, _ = detectar_cantos(gray_strip, y_offset=y_roi0, max_cantos=60)

    # ── Média móvel ───────────────────────────────────────────────
    _hist.append(cp)
    if len(_hist) > 6:
        _hist.pop(0)
    cp_suave = int(np.mean(_hist))
    erro     = float(np.clip((cp_suave - cx_ref) / (cx_ref + 1e-6), -1.0, 1.0))

    # ── Atualiza estado global ────────────────────────────────────
    _pista.update({
        "ok":           pista_ok,
        "erro":         erro,
        "centro":       cp_suave,
        "x_esq":        x_esq,
        "x_dir":        x_dir,
        "ang_esq":      ang_esq,
        "ang_dir":      ang_dir,
        "cantos_pista": len(corners_pista),
    })

    # ── Visualização ──────────────────────────────────────────────
    vis = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

    # Trapézio da ROI
    cv2.polylines(vis, pts_trap, isClosed=True, color=(80, 80, 80), thickness=1)

    # Linha de referência central
    cv2.line(vis, (cx_ref, int(h * 0.55)), (cx_ref, h), (0, 0, 180), 1)

    # Centro calculado (linha verde)
    cv2.line(vis, (cp_suave, int(h * 0.55)), (cp_suave, h), (0, 255, 60), 2)

    # Pontos de interseção bilateral
    if x_esq is not None:
        cv2.circle(vis, (x_esq, y_ref), 7, (255, 220, 0), -1)
        cv2.putText(vis, f"E:{x_esq}", (x_esq - 30, y_ref - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1)
    if x_dir is not None:
        cv2.circle(vis, (x_dir, y_ref), 7, (0, 220, 255), -1)
        cv2.putText(vis, f"D:{x_dir}", (x_dir + 4, y_ref - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 255), 1)

    # Linhas de Hough projetadas
    for m, b in params_esq[:3]:
        y1_l, y2_l = int(h * 0.55), h
        x1_l = int(np.clip((y1_l - b) / (m + 1e-9), 0, w))
        x2_l = int(np.clip((y2_l - b) / (m + 1e-9), 0, w))
        cv2.line(vis, (x1_l, y1_l), (x2_l, y2_l), (255, 200, 0), 1)
    for m, b in params_dir[:3]:
        y1_l, y2_l = int(h * 0.55), h
        x1_l = int(np.clip((y1_l - b) / (m + 1e-9), 0, w))
        x2_l = int(np.clip((y2_l - b) / (m + 1e-9), 0, w))
        cv2.line(vis, (x1_l, y1_l), (x2_l, y2_l), (0, 200, 255), 1)

    # Cantos Harris na strip
    for cx_c, cy_c in corners_pista[:40]:
        cv2.circle(vis, (cx_c, cy_c), 2, (0, 0, 255), -1)

    # Erro + ângulos
    cv2.putText(vis, f"err:{erro:+.3f} esq:{ang_esq:+.1f}° dir:{ang_dir:+.1f}°",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1)
    cv2.putText(vis, f"cantos:{len(corners_pista)}  {'OK' if pista_ok else 'SEM PISTA'}",
                (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 255, 80) if pista_ok else (0, 60, 255), 1)

    return gray_full, vis, erro, pista_ok, cp_suave


# ================================================================
#  [11] PIPELINE ROI PLACAS
#  Mesmos 8 filtros aplicados na faixa 15%–55%.
#  Detecção de cantos + contornos candidatos.
# ================================================================

def processar_placa_roi(frame, gray_full):
    """
    Aplica pipeline_filtros na faixa de placas (ROI_PLACA_Y0–Y1).
    Detecta cantos (Shi-Tomasi, mais estável para placas).
    Retorna vis_bgr (frame completo, preto fora da ROI).
    """
    h, w = frame.shape[:2]
    y0 = int(h * ROI_PLACA_Y0)
    y1 = int(h * ROI_PLACA_Y1)

    roi_gray = gray_full[y0:y1, :]

    # 8 filtros
    binaria, canny, sobel = pipeline_filtros(roi_gray, nome="placa")

    # Cantos Shi-Tomasi (bom para contornos de placas quadradas)
    corners_shi, vis_corners = detectar_cantos(
        roi_gray, y_offset=0, metodo="shi", max_cantos=60
    )

    # Monta canvas completo
    canvas_bin = np.zeros((h, w), dtype=np.uint8)
    canvas_bin[y0:y1, :] = binaria
    canvas_can = np.zeros((h, w), dtype=np.uint8)
    canvas_can[y0:y1, :] = canny

    vis = cv2.cvtColor(canvas_bin, cv2.COLOR_GRAY2BGR)

    # Sobrepõe Canny em azul
    canny_bgr = cv2.cvtColor(canvas_can, cv2.COLOR_GRAY2BGR)
    canny_bgr[:, :, 1] = 0   # zera G
    canny_bgr[:, :, 2] = 0   # zera R → só canal B
    vis = cv2.addWeighted(vis, 0.7, canny_bgr, 0.5, 0)

    # Cantos
    for p in corners_shi:
        cx_c, cy_c = p
        cv2.circle(vis, (cx_c, cy_c + y0), 3, (0, 255, 255), -1)

    # Linhas de limite da ROI
    cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 140, 255), 1)
    cv2.putText(vis, f"ROI placa {int(ROI_PLACA_Y0*100)}-{int(ROI_PLACA_Y1*100)}%",
                (4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1)

    # Contornos candidatos a placa
    cnts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 250:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        razao = bw / max(bh, 1)
        if 0.5 < razao < 1.6:
            cv2.rectangle(vis, (x, y + y0), (x + bw, y + y0 + bh),
                          (0, 220, 140), 1)
            # Indicativo de proporção
            cv2.putText(vis, f"{razao:.1f}", (x, y + y0 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 220, 140), 1)

    # Gradiente Sobel sobreposto (roxo)
    sob_full = np.zeros((h, w), dtype=np.uint8)
    sob_full[y0:y1, :] = sobel
    vis[:, :, 0] = cv2.addWeighted(vis[:, :, 0], 1.0, sob_full, 0.4, 0)

    _placa["cantos"] = len(corners_shi)
    cv2.putText(vis, f"cantos:{len(corners_shi)}",
                (4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

    return vis


# ================================================================
#  [12] DETECÇÃO DE MARCADOR NO CHÃO
# ================================================================

def detectar_marcador_chao(frame):
    h, w = frame.shape[:2]
    y1_r, y2_r = int(h * 0.72), int(h * 0.95)
    roi = frame[y1_r:y2_r, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marcador_ok = False
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for cnt in cnts:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        asp       = ww / max(hh, 1)
        cobertura = ww / w
        if asp > 3.0 and cobertura > 0.45 and hh > 8:
            marcador_ok = True
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            break

    return marcador_ok, vis


# ================================================================
#  [13] LOCALIZAÇÃO E CLASSIFICAÇÃO DE PLACAS
# ================================================================

def _localizar_candidatas(frame):
    """Segmenta regiões de cor placa (vermelho/azul) na ROI."""
    h, w = frame.shape[:2]
    y0   = int(h * ROI_PLACA_Y0)
    y1   = int(h * ROI_PLACA_Y1)
    roi  = frame[y0:y1, :]
    hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv  = cv2.GaussianBlur(hsv, (5, 5), 0)

    m1 = cv2.inRange(hsv, np.array([0,  80, 70]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 80, 70]), np.array([180, 255, 255]))
    mb = cv2.inRange(hsv, np.array([90, 70, 60]),  np.array([130, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), mb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _K5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < 400:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        if 0.5 < ww / max(hh, 1) < 1.5:
            boxes.append((x, y + y0, ww, hh))
    return boxes


def _fallback_cor_forma(crop, bbox):
    if crop is None or crop.size == 0:
        return None
    hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    def px(lo, hi): return int(np.count_nonzero(cv2.inRange(hsv_c, lo, hi)))

    a_vm = px(np.array([0,  80, 70]), np.array([12, 255, 255])) + \
           px(np.array([170, 80, 70]), np.array([180, 255, 255]))
    a_az = px(np.array([90, 70, 60]), np.array([130, 255, 255]))

    gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thr  = cv2.threshold(gray_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lados = 4
    if cnts:
        c  = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        lados = len(ap)

    if a_vm > a_az:
        return "STOP" if lados >= 7 else "RIGHT"
    if a_az > 100:
        return "LEFT"
    return None


def detectar_placa_frame(frame):
    boxes = _localizar_candidatas(frame)
    if not boxes:
        return None, 0.0, None

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
#  [14] CONFIRMAÇÃO DE PLACA
# ================================================================

def confirmar_placa(label, conf, min_frames=3):
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
        _mem_ai["cnt"] = 0
        return label
    return None


# ================================================================
#  [15] ROTA MULTI-PONTO
# ================================================================

def proxima_acao_rota():
    idx = NAV["rota_idx"]
    if NAV["missao_completa"] or idx >= len(NAV["rota"]):
        return None
    return NAV["rota"][idx]


def avancar_rota():
    NAV["rota_idx"] += 1
    if NAV["rota_idx"] >= len(NAV["rota"]):
        NAV["missao_completa"] = True
        print("[ROTA] Missão completa!", flush=True)
    else:
        print(f"[ROTA] Próxima: {NAV['rota'][NAV['rota_idx']]}", flush=True)


def registrar_placa(label_confirmado):
    if not label_confirmado:
        return
    proxima = proxima_acao_rota()
    mapa = {
        "stop": "STOP", "yield": "YIELD",
        "left": "LEFT", "right": "RIGHT",
        "straight": "STRAIGHT", "delivery": "DELIVERY",
        "STOP": "STOP", "LEFT": "LEFT", "RIGHT": "RIGHT",
        "STRAIGHT": "STRAIGHT", "DELIVERY": "DELIVERY",
    }
    acao = mapa.get(label_confirmado)
    if not proxima:
        if acao:
            NAV["acao_pendente"]  = acao
            NAV["cooldown_placa"] = 20
        return
    if acao and acao == proxima:
        NAV["acao_pendente"]  = acao
        NAV["cooldown_placa"] = 20
        print(f"[PLACA] Confirmada: {acao}", flush=True)


def disparar_no_marcador(marcador_ok):
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


def atualizar_cooldowns():
    if NAV["cooldown_placa"]    > 0: NAV["cooldown_placa"]    -= 1
    if NAV["cooldown_marcador"] > 0: NAV["cooldown_marcador"] -= 1


# ================================================================
#  [16] CONVERSÃO SERVO
# ================================================================

def servo_to_254(valor, in_min=-40, in_max=40):
    valor  = float(np.clip(valor, in_min, in_max))
    escala = (valor - in_min) / (in_max - in_min)
    return int(round(escala * 254))


# ================================================================
#  [17] MÁQUINA DE ESTADOS (MANOBRAS)
# ================================================================

def controle_modo():
    agora = time.monotonic()
    dt    = agora - NAV["t_inicio"]
    modo  = NAV["modo"]

    if modo == "SEGUIR_PISTA":
        return False

    def _voltar():
        NAV["modo"] = "SEGUIR_PISTA"
        avancar_rota()

    if modo == "EXEC_STOP":
        CMD.update({"mot": 0, "srv": servo_to_254(0), "buz": 0, "led": 1, "brk": 1, "spd": 0})
        if dt > 2.0:
            CMD.update({"led": 0, "brk": 0})
            _voltar()
        return True

    if modo == "EXEC_YIELD":
        CMD.update({"mot": VEL["devagar"], "srv": servo_to_254(0),
                    "buz": 0, "led": 0, "brk": 0, "spd": 1})
        if dt > 1.0:
            _voltar()
        return True

    if modo == "EXEC_LEFT":
        if   dt < 0.35: CMD.update({"mot": 40, "srv": servo_to_254(0),   "dir": 1})
        elif dt < 1.15: CMD.update({"mot": 38, "srv": servo_to_254(-32), "dir": 1})
        elif dt < 1.45: CMD.update({"mot": 35, "srv": servo_to_254(0),   "dir": 1})
        else:           CMD["dir"] = 0; _voltar()
        CMD.update({"buz": 0, "led": 0, "brk": 0, "spd": 1})
        return True

    if modo == "EXEC_RIGHT":
        if   dt < 0.35: CMD.update({"mot": 40, "srv": servo_to_254(0),   "dir": 2})
        elif dt < 1.15: CMD.update({"mot": 38, "srv": servo_to_254(32),  "dir": 2})
        elif dt < 1.45: CMD.update({"mot": 35, "srv": servo_to_254(0),   "dir": 2})
        else:           CMD["dir"] = 0; _voltar()
        CMD.update({"buz": 0, "led": 0, "brk": 0, "spd": 1})
        return True

    if modo == "EXEC_STRAIGHT":
        CMD.update({"mot": 45, "srv": servo_to_254(0),
                    "buz": 0, "led": 0, "brk": 0, "dir": 3, "spd": 2})
        if dt > 0.9:
            CMD["dir"] = 0
            _voltar()
        return True

    if modo == "EXEC_DELIVERY":
        CMD.update({"mot": 0, "srv": servo_to_254(0),
                    "buz": 1, "led": 1, "brk": 1, "spd": 0})
        if dt > 3.0:
            CMD.update({"buz": 0, "led": 0, "brk": 0})
            _voltar()
        return True

    NAV["modo"] = "SEGUIR_PISTA"
    return False


# ================================================================
#  [18] DECISÃO PRINCIPAL (PID)
# ================================================================

def decidir(pista_ok, erro):
    if pista_ok:
        # Compensação assimétrica: se só um lado visível, reduz velocidade
        vel = VEL["normal"]
        if _pista["x_esq"] is None or _pista["x_dir"] is None:
            vel = int(VEL["normal"] * 0.85)   # um lado ausente → mais devagar

        ang     = int(np.clip(pid_calc(erro) * 40, -40, 40))
        srv_254 = servo_to_254(ang)
        CMD.update({"mot": vel, "srv": srv_254,
                    "buz": 0, "led": 0, "brk": 0, "spd": 2, "dir": 0})
    else:
        CMD.update({"mot": VEL["parado"], "srv": servo_to_254(0),
                    "buz": 0, "led": 1, "brk": 1, "spd": 0, "dir": 0})


# ================================================================
#  [19] DEBUG VISUAL
#  4 painéis: Frame original | ROI Pista | ROI Placas | Status
# ================================================================

def _make_status_panel(w, h, fps):
    """Painel de telemetria (4º painel)."""
    p = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(p, (0, 0), (w - 1, h - 1), (40, 40, 40), 1)

    def txt(texto, linha, cor=(200, 200, 200), escala=0.38):
        cv2.putText(p, texto, (6, 16 + linha * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1)

    cor_pista = (0, 255, 80) if _pista["ok"] else (0, 50, 255)
    txt(f"FPS: {fps}", 0, (255, 255, 255), 0.42)
    txt(f"MODO: {NAV['modo']}", 1, (255, 220, 50), 0.42)
    txt("─── PISTA ───", 3, (120, 120, 120))
    txt(f"ok: {'SIM' if _pista['ok'] else 'NAO'}", 4, cor_pista)
    txt(f"erro: {_pista['erro']:+.3f}", 5)
    txt(f"centro: {_pista['centro']}px", 6)
    txt(f"x_esq: {_pista['x_esq'] or '-'}", 7, (255, 220, 0))
    txt(f"x_dir:  {_pista['x_dir']  or '-'}", 8, (0, 220, 255))
    txt(f"ang_e: {_pista['ang_esq']:+.1f}°", 9, (255, 200, 0))
    txt(f"ang_d: {_pista['ang_dir']:+.1f}°",10, (0, 200, 255))
    txt(f"cantos_pista: {_pista['cantos_pista']}",11)
    txt("─── PLACA ───", 13, (120, 120, 120))
    txt(f"label: {_placa['label'] or '-'}", 14, (0, 220, 180))
    txt(f"conf:  {_placa['conf']:.2f}", 15)
    txt(f"cantos: {_placa['cantos']}", 16)
    txt("─── CMD ───", 18, (120, 120, 120))
    txt(f"mot: {CMD['mot']}%", 19)
    txt(f"srv: {CMD['srv']} ({CMD['srv']-127:+d})", 20)
    txt(f"err: {CMD['err']:+d}", 21)
    txt(f"brk: {CMD['brk']}  led: {CMD['led']}", 22)
    txt(f"buz: {CMD['buz']}  dir: {CMD['dir']}", 23)

    # Barra de erro lateral
    bw  = w - 12
    cx  = w // 2
    by  = h - 30
    cv2.rectangle(p, (6, by), (w - 6, by + 14), (60, 60, 60), -1)
    err_px = int(_pista["erro"] * bw / 2)
    x0 = cx; x1 = cx + err_px
    cor_bar = (0, 180, 255) if err_px >= 0 else (255, 80, 0)
    cv2.rectangle(p, (min(x0, x1), by + 2), (max(x0, x1), by + 12), cor_bar, -1)
    cv2.line(p, (cx, by), (cx, by + 14), (200, 200, 200), 1)
    cv2.putText(p, "ERRO LATERAL", (6, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1)

    # Rota
    txt("─── ROTA ───", 25, (120, 120, 120))
    for i, r in enumerate(NAV["rota"]):
        ativo = (i == NAV["rota_idx"])
        cor_r = (0, 255, 140) if ativo else (100, 100, 100)
        prefixo = "▶ " if ativo else "  "
        txt(f"{prefixo}{i+1}. {r}", 26 + i, cor_r)

    return p


def desenhar_debug(frame, pista_vis, placa_vis, fps):
    h, w = frame.shape[:2]

    # Painel 1: frame original com overlay
    p1 = frame.copy()
    cor_ov = (0, 220, 50) if _pista["ok"] else (0, 40, 255)
    cv2.putText(p1, f"mot:{CMD['mot']} srv:{CMD['srv']} err:{_pista['erro']:+.2f}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, cor_ov, 1)

    # BBox placa no frame original
    if _placa["bbox"]:
        x, y, bw, bh = _placa["bbox"]
        cv2.rectangle(p1, (x, y), (x + bw, y + bh), (0, 220, 180), 2)
        cv2.putText(p1, f"{_placa['label']} {_placa['conf']:.2f}",
                    (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 180), 1)

    # Linhas de ROI no frame original
    cv2.line(p1, (0, int(h * ROI_PLACA_Y0)), (w, int(h * ROI_PLACA_Y0)), (0, 100, 255), 1)
    cv2.line(p1, (0, int(h * ROI_PLACA_Y1)), (w, int(h * ROI_PLACA_Y1)), (0, 100, 255), 1)
    cv2.line(p1, (0, int(h * ROI_PISTA_Y0)), (w, int(h * ROI_PISTA_Y0)), (0, 255, 100), 1)
    cv2.putText(p1, "ROI placa", (4, int(h * ROI_PLACA_Y0) + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 120, 255), 1)
    cv2.putText(p1, "ROI pista", (4, int(h * ROI_PISTA_Y0) + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 120), 1)

    # Painel 2: ROI pista
    p2 = pista_vis if pista_vis is not None else np.zeros_like(p1)
    cv2.putText(p2, "ROI PISTA", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 80), 1)

    # Painel 3: ROI placas
    p3 = placa_vis if placa_vis is not None else np.zeros_like(p1)
    cv2.putText(p3, "ROI PLACAS", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 140, 255), 1)

    # Painel 4: telemetria
    p4 = _make_status_panel(w, h, fps)

    # Concat 2×2
    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    console = np.vstack([row1, row2])

    # Resize para caber na tela (máx 1280px de largura)
    if console.shape[1] > 1280:
        scale  = 1280 / console.shape[1]
        nh     = int(console.shape[0] * scale)
        console = cv2.resize(console, (1280, nh), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Carro Autonomo — Sprint 1", console)


# ================================================================
#  [20] LOOP PRINCIPAL
# ================================================================

def main(usar_camera=False):
    global frame_id

    cap = abrir_fonte(usar_camera)
    ser = conectar_serial()

    t_cap = threading.Thread(target=thread_captura, args=(cap,), daemon=True)
    t_cap.start()

    fps_timer = time.monotonic()
    fps_cnt   = 0
    fps       = 0

    pista_vis = None
    placa_vis = None

    print("[OK] Rodando — Q para encerrar\n", flush=True)

    while not _stop_flag.is_set():
        try:
            frame_big = _q_frames.get(timeout=0.05)
        except queue.Empty:
            continue

        frame_id += 1
        atualizar_cooldowns()

        # Redimensionar
        h = round(frame_big.shape[0] / PROP)
        w = round(frame_big.shape[1] / PROP)
        frame = cv2.resize(frame_big, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Pipeline Pista (análise bilateral) ───────────────────
        gray_full, pista_vis, erro, pista_ok, centro = processar_pista(frame)

        # ── Pipeline ROI Placas ───────────────────────────────────
        placa_vis = processar_placa_roi(frame, gray_full)

        # ── Marcador no chão (a cada 2 frames) ───────────────────
        if frame_id % 2 == 0:
            marcador_ok, _ = detectar_marcador_chao(frame)
        else:
            marcador_ok = False

        # ── IA de placas (a cada IA_EVERY frames) ────────────────
        if frame_id % IA_EVERY == 0 and NAV["cooldown_placa"] == 0:
            raw_label, conf, bbox = detectar_placa_frame(frame)
            confirmado = confirmar_placa(raw_label, conf)
            registrar_placa(confirmado)
            _placa["label"] = confirmado or (f"({raw_label})" if raw_label else None)
            _placa["conf"]  = conf
            _placa["bbox"]  = bbox

        # ── Disparo no marcador ───────────────────────────────────
        disparar_no_marcador(marcador_ok)

        # ── Decisão ──────────────────────────────────────────────
        if not controle_modo():
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
        desenhar_debug(frame, pista_vis, placa_vis, fps)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Encerramento seguro
    _stop_flag.set()
    CMD.update({"mot": 0, "srv": 127, "buz": 0, "led": 0, "brk": 1})
    enviar(ser)
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
    for n, v in [("H_min", 0), ("S_min", 0), ("V_min", 160),
                 ("H_max", 180), ("S_max", 50), ("V_max", 255)]:
        cv2.createTrackbar(n, win, v, 180 if "H" in n else 255, lambda x: None)

    print("[CAL] S=salvar Q=sair")
    while True:
        ret, f = cap.read()
        if not ret: break
        f   = cv2.resize(f, (f.shape[1] // PROP, f.shape[0] // PROP))
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lo  = np.array([cv2.getTrackbarPos(n, win) for n in ("H_min", "S_min", "V_min")])
        hi  = np.array([cv2.getTrackbarPos(n, win) for n in ("H_max", "S_max", "V_max")])
        mask   = cv2.inRange(hsv, lo, hi)
        result = cv2.bitwise_and(f, f, mask=mask)
        cv2.putText(result, f"lo={lo.tolist()} hi={hi.tolist()}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
        cv2.imshow(win, np.hstack([f, result]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'):
            print(f"[CAL] lo={lo.tolist()}, hi={hi.tolist()}")
        elif k == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


# ================================================================
#  MODO ROTA
# ================================================================

def mostrar_rota():
    print("\n[ROTA] Rota configurada:")
    for i, acao in enumerate(ROTA_PONTOS):
        marcador = " ◀ PRÓXIMA" if i == NAV["rota_idx"] else ""
        print(f"  {i+1}. {acao}{marcador}")
    print(f"\n  Total: {len(ROTA_PONTOS)} pontos")
    print("  Edite ROTA_PONTOS no topo do arquivo para alterar.\n")


# ================================================================
if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if   arg == "--cam":  main(usar_camera=True)
    elif arg == "--cal":  calibrar()
    elif arg == "--rota": mostrar_rota()
    else:                 main(usar_camera=False)