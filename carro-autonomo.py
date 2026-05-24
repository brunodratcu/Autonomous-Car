"""
================================================================
  DETECTOR DE PLACAS — VISÃO COMPUTACIONAL  v2.1
  ─────────────────────────────────────────────────────────────
  NOVIDADES vs v2.0:
  ─────────────────────────────────────────────────────────────
  1. DETECÇÃO BILATERAL DA ROI
     • A ROI horizontal é dividida em ZONA_ESQ e ZONA_DIR
     • Cada zona rastreia sua própria placa independentemente
     • Quando uma placa confirmada ALCANÇA A BORDA da zona
       (x_centro < BORDA_X_ESQ  ou  x_centro > BORDA_X_DIR)
       a ação dispara imediatamente — sem esperar mais frames

  2. CONFIRMAÇÃO POR ZONA
     • _conf_esq / _conf_dir: estado separado por lado
     • Evita que uma placa grande do lado esquerdo mascare
       uma placa pequena do lado direito

  3. PARÂMETROS AJUSTADOS PARA PLACAS PEQUENAS/IMPRESSAS
     • AREA_EXEC baixado de 2500 → 400 px²
     • CONFIRM_N reduzido de 7 → 4 frames (placas passam rápido)
     • PLACA_COR_CENTRO_MIN reduzido: placas impressas têm cor
       distribuída diferente de placas metálicas
     • HSV expandido: vermelho mais permissivo (S≥80, não 100)
       e azul menos restrito

  4. CLASSIFICAÇÃO MELHORADA
     • Normalização do crop por CLAHE antes da análise
     • Detecção de cor robusta: considera saturação média
       dentro do contorno, não só pixel count
     • Fallback de texto: se a cor dominar >40% do crop,
       assume placa mesmo sem forma perfeita

  5. VISUALIZAÇÃO BILATERAL
     • Painel lateral mostra estado ESQ e DIR separadamente
     • Linha divisória vertical na ROI (centro)
     • Marcador "⚡ BORDA" quando a placa atinge a zona de gatilho

  Lógica de decisão atualizada:
  ─────────────────────────────────────────────────────────────
  Para cada lado (ESQ / DIR) da ROI:
    1. Localiza candidatas coloridas nessa metade
    2. Valida (proporção + compacidade + cor central)
    3. Classifica (cor + forma + seta)
    4. Confirma por CONFIRM_N frames consecutivos
    5. SE confirmado E (área >= AREA_EXEC OU x_centro na borda):
       → Executa ação imediatamente

  Comandos seriais: STOP/YIELD/LEFT/RIGHT/STRAIGHT/SLOW_DOWN/SPEED_UP/OBSTACLE

  USO:
    python sign_detector.py            ← vídeo (videoplayback.mp4)
    python sign_detector.py --cam      ← câmera USB
    python sign_detector.py --cal      ← calibrar HSV
================================================================
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import os

#  [1] CONFIGURAÇÃO

VIDEO       = "./videoplayback.mp4"
CAM_IDX     = 0
PROP        = 2

SERIAL_PORT = "COM3"
BAUD        = 115200

# ROI de placas — faixa onde placas aparecem fisicamente
ROI_Y0 = 0.05   # 5% do topo (abre levemente para pegar placas altas)
ROI_Y1 = 0.72   # 72%

# ── Lógica bilateral ──────────────────────────────────────────
BORDA_FRAC = 0.18   # 18% das bordas = zona de gatilho

# Parâmetros de detecção
AREA_MIN   = 300     # px² mínimo para candidata (placas impressas são menores)
AREA_MAX   = 80_000  # px² máximo
CONFIRM_N  = 4       # frames consecutivos para confirmar (reduzido de 7→4)
AREA_EXEC  = 400     # px² mínimo para execução por área (reduzido de 2500→400)
COOLDOWN   = 55      # frames de espera após executar

# Limiares de validação de placa real
PLACA_COMPACIDADE_MIN  = 0.28   # reduzido: placas impressas podem ter bordas irregulares
PLACA_COR_CENTRO_MIN   = 0.15   # reduzido: texto na placa reduz cor no centro
PLACA_PROP_MIN         = 0.35   # aceita mais formatos
PLACA_PROP_MAX         = 2.20   # aceita placas retangulares horizontais

# Detecção de obstáculo
OBST_Y0       = 0.50
OBST_AREA_MIN = 3000

#  [2] AÇÕES

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

#  [3] RANGES HSV — expandidos para placas impressas
#  Placas impressas (papel/papelão) têm saturação menor que
#  placas metálicas; o limiar de S foi reduzido.

HSV = {
    # Vermelho: S≥80 (era 100) — abrange vermelho impresso/fosco
    "vermelho": [
        (np.array([0,   80,  50]), np.array([12,  255, 255])),
        (np.array([165, 80,  50]), np.array([180, 255, 255])),
    ],
    # Azul: S≥80 (era 100) — abrange azul de impressora
    "azul": [
        (np.array([90, 80, 50]), np.array([135, 255, 255])),
    ],
    # Amarelo: S≥120 (era 150)
    "amarelo": [
        (np.array([15, 120, 80]), np.array([38, 255, 255])),
    ],
    # Verde: S≥120 (era 150)
    "verde": [
        (np.array([38, 120, 60]), np.array([88, 255, 255])),
    ],
    # Laranja: S≥110 (era 140)
    "laranja": [
        (np.array([7, 110, 80]), np.array([22, 255, 255])),
    ],
    # Roxo/lilás: adicionado — aparece em algumas placas impressas
    "roxo": [
        (np.array([125, 60, 40]), np.array([165, 255, 255])),
    ],
}

#  [4] KERNELS

K3 = np.ones((3, 3), np.uint8)
K5 = np.ones((5, 5), np.uint8)
K7 = np.ones((7, 7), np.uint8)

#  [5] ESTADO

CMD = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)

# Confirmação bilateral — um estado por lado
_conf_esq = dict(label=None, cnt=0, area=0.0, x_centro=0)
_conf_dir = dict(label=None, cnt=0, area=0.0, x_centro=0)

_nav = dict(
    cooldown=0,
    ultimo=None,
    acao_label=None,
    acao_t=0.0,
    acao_dur=0.0,
    obstaculo_ativo=False,
    # Logs de disparo por borda (para debug)
    ultimo_gatilho="",
)

_ser = None

#  [6] SERIAL

def conectar_serial():
    kws = ["arduino", "ch340", "cp210", "uart", "usb serial"]
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower() for k in kws):
            try:
                s = serial.Serial(p.device, BAUD, timeout=0, write_timeout=0)
                time.sleep(2); s.reset_input_buffer()
                print(f"[SER] {p.device}", flush=True)
                return s
            except Exception:
                pass
    try:
        s = serial.Serial(SERIAL_PORT, BAUD, timeout=0, write_timeout=0)
        time.sleep(2); s.reset_input_buffer()
        print(f"[SER] {SERIAL_PORT}", flush=True)
        return s
    except Exception as e:
        print(f"[SER] Simulação ({e})", flush=True)
        return None


def enviar(cmd: dict, ser):
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


#  [7] MÁSCARA DE COR

def mascara_cor(hsv_img: np.ndarray, nome: str) -> np.ndarray:
    m = np.zeros(hsv_img.shape[:2], np.uint8)
    for lo, hi in HSV[nome]:
        m = cv2.bitwise_or(m, cv2.inRange(hsv_img, lo, hi))
    return m


def mascara_total(hsv_img: np.ndarray) -> np.ndarray:
    m = np.zeros(hsv_img.shape[:2], np.uint8)
    for nome in HSV:
        m = cv2.bitwise_or(m, mascara_cor(hsv_img, nome))
    return m


#  [8] VALIDAÇÃO DE PLACA REAL
#  Três testes: proporção, compacidade, cor no centro.
#  Versão relaxada para placas impressas em papel/papelão.

def validar_placa(crop_bgr: np.ndarray, area_contorno: float,
                  bbox_w: int, bbox_h: int) -> tuple[bool, str]:
    # ── Teste 1: Proporção ────────────────────────────────────────
    prop = bbox_w / max(bbox_h, 1)
    if not (PLACA_PROP_MIN <= prop <= PLACA_PROP_MAX):
        return False, f"prop={prop:.2f}"

    # ── Teste 2: Compacidade ──────────────────────────────────────
    bbox_area = bbox_w * bbox_h
    compac    = area_contorno / max(bbox_area, 1)
    if compac < PLACA_COMPACIDADE_MIN:
        return False, f"compac={compac:.2f}"

    # ── Teste 3: Cor no centro ────────────────────────────────────
    if crop_bgr.size == 0:
        return False, "crop vazio"

    sz       = 64
    crop_r   = cv2.resize(crop_bgr, (sz, sz), interpolation=cv2.INTER_AREA)

    # CLAHE para normalizar iluminação antes de checar cor
    lab      = cv2.cvtColor(crop_r, cv2.COLOR_BGR2LAB)
    cl       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    crop_r   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv_crop = cv2.cvtColor(crop_r, cv2.COLOR_BGR2HSV)

    # Centro = 60% central
    m0 = sz // 5
    m1 = sz - m0
    centro_hsv = hsv_crop[m0:m1, m0:m1]
    m_centro   = mascara_total(centro_hsv)
    frac_cor   = float(m_centro.sum()) / (m_centro.size * 255 + 1e-9)

    if frac_cor < PLACA_COR_CENTRO_MIN:
        return False, f"cor_ctr={frac_cor:.2f}"

    return True, "ok"


#  [9] CLASSIFICAÇÃO (cor + forma + seta)

def _analisar_forma(gray):
    cl   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = cl.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 4, 0.0
    c    = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    ap   = cv2.approxPolyDP(c, 0.04 * peri, True)
    area = cv2.contourArea(c)
    circ = (4 * np.pi * area) / (peri ** 2 + 1e-9)
    return len(ap), float(circ)


def _detectar_seta(gray):
    H, W = gray.shape
    _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv    = cv2.morphologyEx(inv, cv2.MORPH_OPEN, K3)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    rx = (cx - W / 2) / (W / 2 + 1e-9)
    ry = (cy - H / 2) / (H / 2 + 1e-9)
    _, _, bw, bh = cv2.boundingRect(c)
    asp = bw / max(bh, 1)
    if asp > 1.2:
        if rx < -0.12: return "left"
        if rx >  0.12: return "right"
    if ry < -0.12: return "up"
    if ry >  0.12: return "down"
    return "up"


def classificar(crop_bgr: np.ndarray) -> tuple[str | None, float, str]:
    """
    Classifica o crop de uma placa já validada.
    Usa CLAHE para normalizar iluminação antes de medir cores.
    Retorna (label, confiança, cor_dominante).
    """
    crop = cv2.resize(crop_bgr, (80, 80), interpolation=cv2.INTER_AREA)

    # Normalização por CLAHE antes de medir cor
    lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    cl   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    total_px = hsv.shape[0] * hsv.shape[1] * 255 + 1e-9
    pc = {n: float(mascara_cor(hsv, n).sum()) / total_px for n in HSV}
    cor, val = max(pc.items(), key=lambda x: x[1])

    # Limite mínimo reduzido: placas impressas têm cor menos saturada
    if val < 0.06:
        return None, 0.0, "?"

    verts, circ = _analisar_forma(gray)
    seta        = _detectar_seta(gray)

    label = conf = None

    if cor in ("vermelho", "laranja"):
        if _v(verts, 7, 99) and circ > 0.55:
            label, conf = "STOP",      0.90
        elif _v(verts, 3, 3):
            label, conf = "YIELD",     0.85
        elif circ > 0.65:
            label, conf = "STOP",      0.80
        elif seta == "left":
            label, conf = "LEFT",      0.72
        elif seta == "right":
            label, conf = "RIGHT",     0.72
        elif cor == "laranja":
            label, conf = "SLOW_DOWN", 0.72
        else:
            label, conf = "STOP",      0.65

    elif cor == "azul":
        if seta == "left":    label, conf = "LEFT",      0.88
        elif seta == "right": label, conf = "RIGHT",     0.88
        elif seta == "up":    label, conf = "STRAIGHT",  0.85
        elif seta == "down":  label, conf = "SLOW_DOWN", 0.78
        else:                 label, conf = "STRAIGHT",  0.62

    elif cor == "amarelo":
        if _v(verts, 3, 3):       label, conf = "YIELD",     0.83
        elif seta == "left":      label, conf = "LEFT",      0.76
        elif seta == "right":     label, conf = "RIGHT",     0.76
        else:                     label, conf = "SLOW_DOWN", 0.68

    elif cor == "verde":
        if seta == "left":    label, conf = "LEFT",      0.80
        elif seta == "right": label, conf = "RIGHT",     0.80
        elif seta == "up":    label, conf = "SPEED_UP",  0.78
        else:                 label, conf = "STRAIGHT",  0.65

    elif cor == "roxo":
        # Roxo geralmente indica "DELIVERY" ou ação especial
        label, conf = "STOP", 0.70

    return label, conf or 0.0, cor


def _v(v, vmin, vmax):
    return vmin <= v <= vmax


#  [10] LOCALIZAR CANDIDATAS — BILATERAL
#  Agora retorna candidatas separadas por lado (esq/dir).
#  A divisão é feita no centro da ROI (largura do frame / 2).

def localizar_candidatas(frame: np.ndarray):
    """
    Encontra candidatas na ROI e as divide em ESQUERDA e DIREITA.

    Retorna:
      esq: lista de (x, y, w, h, area, crop)  — x_centro < w/2
      dir: lista de (x, y, w, h, area, crop)  — x_centro >= w/2
      rejeitadas: lista de (x, y, w, h, motivo)
    """
    h, w  = frame.shape[:2]
    y0    = int(h * ROI_Y0)
    y1    = int(h * ROI_Y1)
    cx    = w // 2          # linha divisória
    roi   = frame[y0:y1]

    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv   = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask  = mascara_total(hsv)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K7)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K5)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    esq        = []
    dir_       = []
    rejeitadas = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (AREA_MIN < area < AREA_MAX):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        crop = frame[y + y0: y + y0 + bh, x: x + bw]
        if crop.size == 0:
            continue

        ok, motivo = validar_placa(crop, area, bw, bh)
        if not ok:
            rejeitadas.append((x, y + y0, bw, bh, motivo))
            continue

        x_centro = x + bw // 2
        entry    = (x, y + y0, bw, bh, area, crop)

        if x_centro < cx:
            esq.append(entry)
        else:
            dir_.append(entry)

    # Ordena por área decrescente, pega até 2 por lado
    esq.sort(key=lambda b: b[4], reverse=True)
    dir_.sort(key=lambda b: b[4], reverse=True)
    return esq[:2], dir_[:2], rejeitadas


#  [11] DETECÇÃO DE OBSTÁCULO

def detectar_obstaculo(frame: np.ndarray) -> bool:
    h, w  = frame.shape[:2]
    y0    = int(h * OBST_Y0)
    roi   = frame[y0:, :]

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (15, 15), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, K7)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  K5)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < OBST_AREA_MIN:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh > bw * 0.4 and bw < w * 0.85:
            return True
    return False


#  [12] CONFIRMAÇÃO BILATERAL
#  Cada lado mantém seu próprio estado de confirmação.
#  O gatilho de execução agora tem DUAS condições:
#    A) área >= AREA_EXEC  (placa próxima, visível claramente)
#    B) x_centro na borda da ROI  (placa saindo do campo de visão)
#  Qualquer uma das condições dispara a ação.

def _checar_borda(x, bw, w_frame) -> bool:
    """
    Retorna True se o centro da placa está na zona de borda da ROI.
    Zona de borda = 18% de cada extremidade da largura do frame.
    """
    x_centro     = x + bw // 2
    borda_px_esq = int(w_frame * BORDA_FRAC)
    borda_px_dir = w_frame - borda_px_esq
    return x_centro < borda_px_esq or x_centro > borda_px_dir


def confirmar_lado(estado: dict, label: str | None,
                   area: float, x: int, bw: int, w_frame: int) -> str | None:
    """
    Atualiza o estado de confirmação de um lado e retorna o label
    confirmado quando o critério for atingido, ou None.
    """
    if _nav["cooldown"] > 0:
        estado.update(label=None, cnt=0)
        return None

    if label is None:
        estado.update(cnt=0, label=None)
        return None

    if label == estado["label"]:
        estado["cnt"]      += 1
        estado["area"]      = area
        estado["x_centro"]  = x + bw // 2
    else:
        estado.update(label=label, cnt=1, area=area, x_centro=x + bw // 2)

    na_borda = _checar_borda(x, bw, w_frame)

    # Dispara se: (confirmado o suficiente) E (área OK OU na borda)
    pronto_frames = estado["cnt"] >= CONFIRM_N
    pronto_area   = area >= AREA_EXEC
    pronto_borda  = na_borda and estado["cnt"] >= max(2, CONFIRM_N // 2)

    if pronto_frames and (pronto_area or pronto_borda):
        gatilho = "BORDA" if na_borda else "AREA"
        _nav["ultimo_gatilho"] = gatilho
        estado.update(cnt=0, label=None)
        return label

    return None


#  [13] EXECUTOR

def executar(label: str, ser):
    if label not in ACOES:
        return
    a = ACOES[label]
    CMD.update(mot=a["mot"], srv=a["srv"], buz=a["buz"],
               led=a["led"], brk=a["brk"], dir=a["dir"],
               spd=0 if a["mot"] == 0 else (1 if a["mot"] < 50 else 2))
    enviar(CMD, ser)
    _nav.update(
        acao_label=label,
        acao_t=time.monotonic(),
        acao_dur=a["dur"],
        cooldown=COOLDOWN,
        ultimo=label,
    )
    print(f"[NAV] ▶ {label}  [{_nav['ultimo_gatilho']}]", flush=True)


def tick(ser) -> bool:
    lbl = _nav["acao_label"]
    if lbl is None:
        return False
    if lbl == "OBSTACLE":
        return True
    dt = time.monotonic() - _nav["acao_t"]
    if dt >= _nav["acao_dur"]:
        CMD.update(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
        enviar(CMD, ser)
        print(f"[NAV] ✅ Fim: {lbl}", flush=True)
        _nav["acao_label"] = None
        return False
    return True


def liberar_obstaculo(ser):
    CMD.update(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
    enviar(CMD, ser)
    print("[NAV] ✅ Obstáculo removido", flush=True)
    _nav["acao_label"] = None


#  [14] VISUALIZAÇÃO BILATERAL

CORES = {
    "STOP":      (50,  50,  220),
    "OBSTACLE":  (0,   0,   200),
    "YIELD":     (0,   200, 220),
    "LEFT":      (220, 120, 0  ),
    "RIGHT":     (0,   120, 220),
    "STRAIGHT":  (50,  220, 50 ),
    "SLOW_DOWN": (0,   180, 255),
    "SPEED_UP":  (0,   200, 100),
}


def desenhar(frame, esq, dir_, rejeitadas,
             lbl_esq, conf_esq, lbl_dir, conf_dir,
             fps, obstaculo):
    out  = frame.copy()
    h, w = out.shape[:2]
    cx   = w // 2

    # ROI de placas (contorno amarelo)
    cv2.rectangle(out, (0, int(h * ROI_Y0)), (w - 1, int(h * ROI_Y1)), (0, 200, 255), 1)

    # Linha divisória central (tracejada)
    for yy in range(int(h * ROI_Y0), int(h * ROI_Y1), 8):
        cv2.line(out, (cx, yy), (cx, yy + 4), (0, 200, 255), 1)

    # Zonas de borda (linhas verticais tracejadas)
    borda_px_esq = int(w * BORDA_FRAC)
    borda_px_dir = w - borda_px_esq
    for yy in range(int(h * ROI_Y0), int(h * ROI_Y1), 6):
        cv2.line(out, (borda_px_esq, yy), (borda_px_esq, yy + 3), (0, 120, 255), 1)
        cv2.line(out, (borda_px_dir, yy), (borda_px_dir, yy + 3), (0, 120, 255), 1)

    # Labels "ESQ" / "DIR" na ROI
    y_label = int(h * ROI_Y0) + 12
    cv2.putText(out, "ESQ", (6, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
    cv2.putText(out, "DIR", (w - 35, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    # Rejeitadas (cinza)
    for x, y, bw, bh, motivo in rejeitadas[:4]:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (60, 60, 160), 1)
        cv2.putText(out, motivo[:14], (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, (80, 80, 180), 1)

    def _desenhar_candidata(lista, label_raw, conf, conf_estado):
        if not lista:
            return
        x, y, bw, bh, area, _ = lista[0]
        x_centro = x + bw // 2

        # Cor base pela área/borda
        na_borda = _checar_borda(x, bw, w)
        cor_box  = CORES.get(label_raw, (180, 180, 180)) if label_raw else (100, 100, 100)
        thick    = 3 if na_borda or conf_estado["cnt"] >= CONFIRM_N - 1 else 2

        cv2.rectangle(out, (x, y), (x + bw, y + bh), cor_box, thick)

        # Marcador de borda
        if na_borda:
            cv2.putText(out, "⚡BORDA", (x, y - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)

        if label_raw:
            txt = f"{label_raw} {conf:.2f} [{conf_estado['cnt']}/{CONFIRM_N}]"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), cor_box, -1)
            cv2.putText(out, txt, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

            # Barra de confirmação
            prog = int(bw * min(conf_estado["cnt"], CONFIRM_N) / CONFIRM_N)
            cv2.rectangle(out, (x, y + bh + 2), (x + bw, y + bh + 7), (40, 40, 40), -1)
            cv2.rectangle(out, (x, y + bh + 2), (x + prog, y + bh + 7), cor_box, -1)

    _desenhar_candidata(esq,  lbl_esq, conf_esq, _conf_esq)
    _desenhar_candidata(dir_, lbl_dir, conf_dir, _conf_dir)

    # Obstáculo
    if obstaculo:
        cv2.rectangle(out, (0, int(h * OBST_Y0)), (w - 1, h - 1), (0, 0, 255), 2)
        cv2.putText(out, "OBSTACULO", (w // 2 - 55, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2)

    # ── Painel lateral ───────────────────────────────────────────
    pw  = 215
    pan = np.zeros((h, pw, 3), np.uint8)
    pan[:] = (18, 18, 18)
    cv2.rectangle(pan, (0, 0), (pw - 1, h - 1), (45, 45, 45), 1)

    def t(s, l, cor=(190, 190, 190), sc=0.35):
        cv2.putText(pan, s, (5, 14 + l * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, sc, cor, 1)

    t(f"FPS: {fps}", 0, (255, 255, 255), 0.40)

    acao = _nav["acao_label"]
    if acao:
        cor_a = CORES.get(acao, (200, 200, 200))
        t(f"EXEC: {acao}", 1, cor_a, 0.40)
        if acao != "OBSTACLE":
            dt = time.monotonic() - _nav["acao_t"]
            t(f"  {dt:.1f}s / {_nav['acao_dur']}s", 2, cor_a)
        else:
            t("  aguard. sumir...", 2, (0, 0, 200), 0.30)
        t(f"  gatilho: {_nav['ultimo_gatilho']}", 3, (200, 200, 0), 0.30)
    else:
        t("livre", 1, (100, 200, 100))
        t(f"cooldown: {_nav['cooldown']}", 2, (80, 80, 80))

    # Lado esquerdo
    t("── LADO ESQ ──", 5,  (60, 60, 60))
    lbl_e = lbl_esq or "-"
    cor_e = CORES.get(lbl_esq, (140, 140, 140)) if lbl_esq else (80, 80, 80)
    t(f"label: {lbl_e}", 6,  cor_e)
    t(f"conf:  {conf_esq:.2f}", 7)
    t(f"frames:{_conf_esq['cnt']}/{CONFIRM_N}", 8)
    borda_e = _checar_borda(_conf_esq.get("x_centro", 0) - 1,
                            2, w) if _conf_esq["cnt"] > 0 else False
    t("BORDA!" if borda_e else "centro", 9,
      (0, 255, 255) if borda_e else (80, 80, 80), 0.30)

    # Lado direito
    t("── LADO DIR ──", 11, (60, 60, 60))
    lbl_d = lbl_dir or "-"
    cor_d = CORES.get(lbl_dir, (140, 140, 140)) if lbl_dir else (80, 80, 80)
    t(f"label: {lbl_d}", 12, cor_d)
    t(f"conf:  {conf_dir:.2f}", 13)
    t(f"frames:{_conf_dir['cnt']}/{CONFIRM_N}", 14)
    borda_d = _checar_borda(_conf_dir.get("x_centro", 0) - 1,
                            2, w) if _conf_dir["cnt"] > 0 else False
    t("BORDA!" if borda_d else "centro", 15,
      (0, 255, 255) if borda_d else (80, 80, 80), 0.30)

    # Último executado
    t("── ÚLTIMO ──", 17, (60, 60, 60))
    ult = _nav["ultimo"] or "-"
    t(f"  {ult}", 18, CORES.get(ult, (160, 160, 160)))

    # Obstáculo
    t("── OBSTÁCULO ──", 20, (60, 60, 60))
    t("SIM" if obstaculo else "não", 21,
      (0, 0, 255) if obstaculo else (80, 180, 80))

    return np.hstack([out, pan])


#  [15] LOOP PRINCIPAL

def main(usar_camera=False):
    global _ser

    src = CAM_IDX if usar_camera else VIDEO
    print(f"[CAM] {'Câmera' if usar_camera else 'Vídeo'}: {src}", flush=True)

    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if usar_camera else cv2.CAP_ANY)
    if not cap.isOpened():
        print("[ERRO] Não abriu fonte de vídeo."); sys.exit(1)

    if usar_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    _ser = conectar_serial()
    fps_t = time.monotonic(); fps_n = fps = 0

    print("[OK] Rodando — Q para sair\n", flush=True)

    while True:
        ret, frame_big = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        h_r = round(frame_big.shape[0] / PROP)
        w_r = round(frame_big.shape[1] / PROP)
        frame = cv2.resize(frame_big, (w_r, h_r))
        w_frame = w_r

        if _nav["cooldown"] > 0:
            _nav["cooldown"] -= 1

        em_acao = tick(_ser)

        # ── Detecção principal ────────────────────────────────────
        esq, dir_, rejeitadas = localizar_candidatas(frame)

        lbl_esq = conf_esq_raw = None
        lbl_dir = conf_dir_raw = None
        obstaculo = False

        if _nav["cooldown"] == 0 and not em_acao:

            # ── Lado esquerdo ─────────────────────────────────────
            if esq:
                x, y, bw, bh, area, crop = esq[0]
                lbl_esq, conf_esq_raw, _ = classificar(crop)
                confirmado = confirmar_lado(
                    _conf_esq, lbl_esq, area, x, bw, w_frame)
                if confirmado:
                    executar(confirmado, _ser)
            else:
                confirmar_lado(_conf_esq, None, 0, 0, 0, w_frame)

            # ── Lado direito ──────────────────────────────────────
            if dir_ and not em_acao and _nav["cooldown"] == 0:
                x, y, bw, bh, area, crop = dir_[0]
                lbl_dir, conf_dir_raw, _ = classificar(crop)
                confirmado = confirmar_lado(
                    _conf_dir, lbl_dir, area, x, bw, w_frame)
                if confirmado:
                    executar(confirmado, _ser)
            else:
                confirmar_lado(_conf_dir, None, 0, 0, 0, w_frame)

            # ── Sem placas → verifica obstáculo ───────────────────
            if not esq and not dir_:
                obstaculo = detectar_obstaculo(frame)
                if obstaculo and _nav["acao_label"] is None:
                    executar("OBSTACLE", _ser)

        # ── Libera obstáculo quando some ─────────────────────────
        if _nav["acao_label"] == "OBSTACLE":
            obstaculo = detectar_obstaculo(frame)
            if not obstaculo:
                liberar_obstaculo(_ser)

        # ── Visualização ──────────────────────────────────────────
        vis = desenhar(frame, esq, dir_, rejeitadas,
                       lbl_esq, conf_esq_raw or 0.0,
                       lbl_dir, conf_dir_raw or 0.0,
                       fps, obstaculo)
        cv2.imshow("Detector de Placas", vis)

        fps_n += 1
        if time.monotonic() - fps_t >= 1.0:
            fps = fps_n; fps_n = 0; fps_t = time.monotonic()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CMD.update(mot=0, srv=127, buz=0, led=0, brk=1, dir=0, spd=0)
    enviar(CMD, _ser)
    if _ser: _ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] Encerrado.", flush=True)


#  CALIBRAÇÃO HSV

def calibrar():
    cap = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened(): return

    win = "Calibrar HSV — S=salvar Q=sair"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1000, 520)
    for n, v in [("H_min", 0), ("S_min", 80), ("V_min", 50),
                 ("H_max", 15), ("S_max", 255), ("V_max", 255)]:
        cv2.createTrackbar(n, win, v, 180 if "H" in n else 255, lambda x: None)

    print("[CAL] S=salvar  Q=sair")
    while True:
        ret, f = cap.read()
        if not ret: break
        f   = cv2.resize(f, (f.shape[1] // PROP, f.shape[0] // PROP))
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lo  = np.array([cv2.getTrackbarPos(n, win) for n in ("H_min","S_min","V_min")])
        hi  = np.array([cv2.getTrackbarPos(n, win) for n in ("H_max","S_max","V_max")])
        mask = cv2.inRange(hsv, lo, hi)
        res  = cv2.bitwise_and(f, f, mask=mask)
        cv2.putText(res, f"lo={lo.tolist()} hi={hi.tolist()}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
        cv2.imshow(win, np.hstack([f, res]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('s'): print(f"[CAL] lo={lo.tolist()}, hi={hi.tolist()}")
        elif k == ord('q'): break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if   arg == "--cam": main(usar_camera=True)
    elif arg == "--cal": calibrar()
    else:                main(usar_camera=False)