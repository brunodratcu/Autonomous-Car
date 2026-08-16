"""
================================================================
  CARRO AUTÔNOMO v5.2 — PIPELINE MONO GEOMÉTRICO
  171 Garage · FIAP/Mercedes-Benz Challenge 2026
================================================================
  webcam → mono+CLAHE → varredura única de contornos
                              │
                 ┌────────────┴────────────┐
              OCTÓGONO                  CÍRCULO
                 │                         │
            PARE (PGOM→CNN)          letra A/B/C
                 │                         │
                 └──────► MISSÃO ◄─────────┘
                              │
                       serial → Portenta H7
================================================================

    COMO CHAMAR:
python carro-autonomo.py                    # roda direto na webcam
python carro-autonomo.py --debug-abc        # calibrar AREA_MIN_CHEGADA
python carro-autonomo.py --auto             # A→B sem depender do QR
python carro-autonomo.py --cam-idx 2        # se a webcam não for o índice 1
python carro-autonomo.py --fast --yolo      # todo frame + YOLO auxiliar
python carro-autonomo.py --gerar-abc        # PNGs das placas para imprimir
"""

import cv2
import numpy as np
import serial, serial.tools.list_ports
import time, sys, os, json
from collections import deque, Counter

# ================================================================
#  [0] CONFIGURAÇÃO
# ================================================================

CAM_IDX     = 0                      # webcam externa (a integrada costuma ser 0)
CAM_FLIP    = 1
IMG_W       = 640

SERIAL_PORT = "COM3"
BAUD        = 115200

YOLO_MODEL  = "./models/sign_detector.onnx"   # custom: Stop + Semaforo (2 classes)
COCO_MODEL  = "./models/yolov8n.onnx"         # COCO pré-treinado: Pessoa (+ Carro de apoio)
CNN_MODEL   = "./models/sign_classifier.tflite"
OOD_FILE    = "./models/ood_thresholds.json"

CNN_SIZE    = 96
OOD_DEFAULT = 0.55

# YOLO (auxiliar — a geometria sempre tem a palavra final)
YOLO_CONF   = 0.25
YOLO_NMS    = 0.45
YOLO_SIZE   = 640
YOLO_EVERY  = 3                      # roda 1 frame em cada 3 (era 5 — rápido demais parado, lento em movimento)
COCO_MAP    = {11:"Stop", 0:"Pessoa", 2:"Carro", 7:"Carro", 5:"Carro", 3:"Carro", 9:"Semaforo"}
COCO_CONF_MIN = {"Stop":0.28, "Semaforo":0.40, "Carro":0.55, "Pessoa":0.62}
COCO_AREA_MIN = {"Pessoa":2000, "Carro":1500}

# Contrato da YOLO customizada (localização: Pare + Semaforo).
# Ordem = ordem do data.yaml usado no treino (yolo_localizacao/dataset.yaml).
# É lido de models/classes.txt se existir; senão usa este default.
CLASSES     = ["Stop", "Semaforo"]
NUM_CLASSES = len(CLASSES)

# Contrato da CNN de confirmação (treinar_cnn_confirmacao.py).
# Ordem = CLASSES do trainer. Fixo aqui, não deve variar por treino de YOLO
# de terceiros — evita que um classes.txt da YOLO sobrescreva o da CNN.
CNN_CLASSES = ["Semaforo", "Stop", "Fundo"]

CLASS_TO_ACTION = {"Cone":"OBSTACLE", "Carro":"OBSTACLE", "Pessoa":"OBSTACLE"}
DEBUG_FUNIL = os.environ.get("DEBUG_FUNIL", "0") == "1"   # export DEBUG_FUNIL=1 pra ativar

# Rastreio e votação temporal do ramo de trânsito
TRACK_IOU_HIGH = 0.30
TRACK_IOU_LOW  = 0.15
TRACK_MAX_AGE  = 10
CONF_HIGH_THR  = 0.45
VOTE_BUFFER    = 6          # era 10 — a 15km/h não dá tempo de acumular tanto
VOTE_MIN_DETS  = 3          # era 5 — 3 votos já é maioria qualificada com VOTE_FRAC
VOTE_FRAC      = 0.60

AREA_MIN_EXEC = 800                  # bbox mínima p/ um track valer decisão
AREA_MIN_PROX = 350                  # bbox mínima p/ avisar "aproximando" (menor que EXEC — avisa antes)
AREA_PROX_NIVEIS = [350, 900, 1800]  # níveis crescentes — cada um cruzado reenvia PROX (proxy de "cm")
AREA_CHEGADA_SINALEIRA = 3000        # limiar final p/ PARE/Semaforo — "executa agora" (calibrar na pista)
ROI_Y0, ROI_Y1 = 0.05, 0.92          # faixa vertical útil do frame

# ── Rede e painel — TP-Link TL-MR3020 em modo AP, sem internet ──
PAINEL_PORTA = 8000
CAR_ID       = "171"
WIFI_SSID    = "171garage"
WIFI_PASS    = "garagem171"
WIFI_GATEWAY = "192.168.0.1"         # gateway de fábrica do TL-MR3020

# ── Tempos da missão ───────────────────────────────────────────
ESPERA_ENTREGA_S = 7.0               # parado retirando / entregando
BUZ_PARTIDA_S    = 0.4
BUZ_ENTREGA_S    = 1.2

# ── Controle ───────────────────────────────────────────────────
MOT_CRUZEIRO = 90                    # as paradas não têm timer: quem solta é a missão

# ── Placas de ponto A/B/C ──────────────────────────────────────
LETRAS_PONTOS   = ["A", "B", "C"]
LETRA_FONTE     = cv2.FONT_HERSHEY_SIMPLEX
LETRA_ESCALA    = 4.5                # fonte do molde e do PNG de impressão
LETRA_ESPESSURA = 12

AREA_MIN_LEITURA = 400               # contorno mínimo p/ tentar LER a letra
AREA_MIN_CHEGADA = 9000              # bbox mínima p/ declarar "cheguei" (calibrar na pista)
AREA_MIN_PROX_ENTREGA = 3000         # bbox mínima p/ avisar "aproximando do ponto" (calibrar na pista)

LETRA_SCORE_MIN   = 0.50             # casamento mínimo com o molde normalizado
LETRA_SCORE_FORTE = 0.62             # acima disto a letra vence forma duvidosa
LETRA_MARGEM_MIN  = 0.06             # vantagem sobre a 2ª letra
LETRA_TINTA_MIN   = 0.10             # fração de preto no miolo do disco
LETRA_TINTA_MAX   = 0.55
LETRA_AR_MIN      = 0.45             # largura/altura da LETRA recortada
LETRA_AR_MAX      = 1.25
LETRA_PREENCH_MIN = 0.30             # tinta dentro da bbox da letra
LETRA_PREENCH_MAX = 0.85
LETRA_MAX_CIRC    = 8                # teto de leituras por frame em círculos
LETRA_MAX_DUVIDA  = 4                # teto de leituras em formas duvidosas
LETRA_VOTOS_N     = 5                # janela de votação temporal
LETRA_VOTOS_MIN   = 3                # confirmações dentro da janela

# ── Classificação de forma (8º harmônico do raio) ──────────────
FORMA_AR_MIN     = 0.55              # portão de entrada largo: a placa chega
FORMA_AR_MAX     = 1.80              # inclinada, borrada, lavada
FORMA_CIRC_MIN   = 0.68
FORMA_CONVEX_MIN = 0.85
FORMA_H8_CIRCULO = 0.022             # ≤ isto → círculo   (medido: 0.0033–0.0212)
FORMA_H8_OCTOG   = 0.025             # ≥ isto → octógono  (medido: 0.0266–0.0335)
FORMA_CIRC_OCTOG = 0.85              # octógono exige contorno bem formado

# ── Segmentação e ROI ──────────────────────────────────────────
GEO_AREA_MIN   = 400
GEO_AREA_FRAC  = 0.25                # teto: 25% do frame
GEO_MAX_CANDS  = 10
ROI_FULL_EVERY = 15                  # a cada N frames varre o frame inteiro
ROI_EXPAND     = 1.8

# ── Peneira de ROI ─────────────────────────────────────────────
ROI_ESCURA_MIN, ROI_ESCURA_MAX = 12, 248
ROI_FOCO_MIN    = 25.0
ROI_BORDA_MIN   = 0.03
ROI_TEXTURA_MAX = 0.72
ROI_AREA_FRAC_MIN, ROI_AREA_FRAC_MAX = 0.0008, 0.25

# ── PGOM (persistência geométrica) ─────────────────────────────
PGOM_PROMOVE     = 0.85              # evidência mínima p/ liberar o candidato à CNN
PGOM_PROMOVE_SEM = 0.72
PGOM_MAX_MISS    = 3
PGOM_MATCH_D     = 60                # distância máx. de associação entre frames
PGOM_HIST        = 10
PGOM_PERSIST_N   = 8
PESOS_PLACA    = dict(forma=0.18, convex=0.05, aspecto=0.08, simbolo=0.12,
                      persist=0.18, estab=0.12, fingerprint=0.05, persp=0.12, simet=0.10)
PESOS_SEMAFORO = dict(forma=0.15, convex=0.05, aspecto=0.15, simbolo=0.35,
                      persist=0.05, estab=0.15, fingerprint=0.10, persp=0.00, simet=0.00)

K3 = np.ones((3,3), np.uint8)
K5 = np.ones((5,5), np.uint8)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

COR_CLASSE = {"Stop":(50,50,220), "SemRetorno":(20,20,180), "Esquerda":(220,120,0),
              "Direita":(0,120,220), "Verde":(50,220,50), "Cone":(0,165,255),
              "Carro":(200,0,200), "Pessoa":(0,0,255), "Semaforo":(0,220,220)}
COR_ACAO = {"STOP":(50,50,220), "STRAIGHT":(50,220,50)}

# Instâncias globais preenchidas em main()
LEITOR = None
BT     = None
_ser   = None


# ================================================================
#  [1] PRÉ-PROCESSAMENTO E SEGMENTAÇÃO
# ================================================================

def preprocessar(frame_bgr):
    """BGR → cinza normalizado por CLAHE, largura fixa IMG_W."""
    h0, w0 = frame_bgr.shape[:2]
    if w0 != IMG_W:
        frame_bgr = cv2.resize(frame_bgr, (IMG_W, int(h0*IMG_W/w0)), interpolation=cv2.INTER_LINEAR)
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)


def segmentar(gray):
    """Três binarizações em paralelo — nenhuma sozinha pega a placa em toda luz."""
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    saidas = [
        cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5),
        cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,     31, 5),
        cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]
    return [cv2.morphologyEx(cv2.morphologyEx(t, cv2.MORPH_OPEN, K3), cv2.MORPH_CLOSE, K5)
            for t in saidas]


class ROIDinamico:
    """Janela de busca que segue os últimos alvos — e reabre o frame inteiro a cada N."""
    def __init__(self):
        self.hist = deque(maxlen=12); self.fn = 0

    def registrar(self, bbox):
        self.hist.append(bbox)

    def janela(self, h, w):
        self.fn += 1
        if not self.hist or self.fn % ROI_FULL_EVERY == 0:
            return (0, int(h*ROI_Y0), w, int(h*ROI_Y1))
        xs0 = min(b[0] for b in self.hist); ys0 = min(b[1] for b in self.hist)
        xs1 = max(b[2] for b in self.hist); ys1 = max(b[3] for b in self.hist)
        cx, cy = (xs0+xs1)/2, (ys0+ys1)/2
        bw = max(xs1-xs0, 80)*ROI_EXPAND; bh = max(ys1-ys0, 80)*ROI_EXPAND
        x0 = int(max(0, cx-bw)); x1 = int(min(w, cx+bw))
        y0 = int(max(h*ROI_Y0, cy-bh)); y1 = int(min(h*ROI_Y1, cy+bh))
        if x1-x0 < 64 or y1-y0 < 64:
            return (0, int(h*ROI_Y0), w, int(h*ROI_Y1))
        return (x0, y0, x1, y1)

ROI_DIN = ROIDinamico()


def analisar_farois(crop_gray):
    """Semáforo = 2–3 manchas redondas empilhadas e alinhadas na vertical."""
    if crop_gray.size == 0: return 0, False
    g = cv2.resize(crop_gray, (36, 100))
    circulos = []
    for thr in (cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
                cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]):
        cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (40 < a < 1200): continue
            x, y, w, h = cv2.boundingRect(c)
            if not (0.6 < w/max(h,1) < 1.7): continue
            if 4*np.pi*a/max(cv2.arcLength(c,True)**2, 1) > 0.55:
                circulos.append((x + w/2, y + h/2))
    if len(circulos) < 2: return len(circulos), False
    circulos.sort(key=lambda p: p[1])
    empilhados = [circulos[0]]
    for p in circulos[1:]:
        if p[1] - empilhados[-1][1] > 15: empilhados.append(p)
    if len(empilhados) < 2: return len(empilhados), False
    xs = [p[0] for p in empilhados]
    return len(empilhados), (max(xs) - min(xs)) < 12


def _tem_farois(crop_gray):
    n, alinhado = analisar_farois(crop_gray)
    return 2 <= n <= 3 and alinhado


def _tem_simbolo(crop_gray):
    """Placa real tem contraste interno; superfície lisa não."""
    if crop_gray.size == 0: return False
    miolo = cv2.resize(crop_gray, (48,48))[10:38, 10:38]
    _, t = cv2.threshold(miolo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f = float(t.mean())/255.0
    return min(f, 1.0-f) > 0.08


def peneira_roi(crop_gray, area_rel, exige_simbolo=True):
    """Descarta recorte inútil antes de gastar CPU: tamanho, brilho, foco, textura."""
    if crop_gray is None or crop_gray.size == 0: return False, "vazia"
    h, w = crop_gray.shape[:2]
    if h < 10 or w < 10:                  return False, "muito pequena (px)"
    if area_rel < ROI_AREA_FRAC_MIN:      return False, "muito pequena"
    if area_rel > ROI_AREA_FRAC_MAX:      return False, "muito grande"
    m = float(crop_gray.mean())
    if m < ROI_ESCURA_MIN:                return False, "muito escura"
    if m > ROI_ESCURA_MAX:                return False, "estourada"
    g = cv2.resize(crop_gray, (48,48))
    foco = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if foco < ROI_FOCO_MIN:               return False, f"desfocada ({foco:.0f})"
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    dens = float((mag > 60).mean())
    if dens < ROI_BORDA_MIN:              return False, "sem bordas fortes"
    if dens > ROI_TEXTURA_MAX:            return False, f"textura complexa ({dens:.2f})"
    if exige_simbolo and not _tem_simbolo(crop_gray): return False, "sem símbolo interno"
    return True, "ok"


def estado_semaforo(crop_gray):
    """Retifica o crop em 3 posições esperadas (topo/meio/base) e decide qual
    LÂMPADA ACESA existe comparando as posições ENTRE SI — não contra a média
    global do crop. Isso importa fisicamente: em escala de cinza, vermelho
    saturado converte pra luminância BAIXA (Y=0.299R+0.587G+0.114B pesa pouco
    o vermelho), então um vermelho aceso pode ficar mais escuro que o bezel
    plástico claro ao redor — comparar contra o fundo global sistematicamente
    perde o vermelho. Comparar irmã-contra-irmã (a posição acesa domina as
    outras duas, que estão apagadas/pretas) é robusto a isso.
    Área luminosa do núcleo confirma que não é ruído pontual."""
    if crop_gray is None or crop_gray.size == 0: return None
    if crop_gray.ndim == 3: crop_gray = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(crop_gray, (30, 90)).astype(np.float32)

    nucleos, areas = [], []
    for i in range(3):
        bloco = g[i*30:(i+1)*30]
        cy, cx = 15, 15
        nucleo = bloco[max(0,cy-8):cy+8, max(0,cx-8):cx+8]
        borda_mask = np.ones_like(bloco, dtype=bool)
        borda_mask[max(0,cy-8):cy+8, max(0,cx-8):cx+8] = False
        anel = bloco[borda_mask]
        b_central = float(nucleo.mean()) if nucleo.size else 0.0
        b_anel = float(anel.mean()) if anel.size else b_central
        nucleos.append(b_central)
        areas.append(float((nucleo > (b_anel + 15)).mean()) if nucleo.size else 0.0)

    i_top = int(np.argmax(nucleos))
    ordenado = sorted(nucleos, reverse=True)
    margem = ordenado[0] - ordenado[1]      # domina as OUTRAS 2 posições, não o fundo global
    if margem < 10 or areas[i_top] < 0.05:  # 3 parecidas (apagado/ambíguo) ou sem área real acesa
        return None
    return ["vermelho", "amarelo", "verde"][i_top]


# ================================================================
#  [2] CLASSIFICADOR DE FORMA — árbitro único da visão
#      approxPolyDP dá 8 lados TANTO para círculo quanto para
#      octógono; quem separa de verdade é o 8º harmônico do raio.
# ================================================================

def harmonico_8(c, N=64):
    """Força do 8º harmônico de r(θ) — exige contorno com CHAIN_APPROX_NONE."""
    if c is None or len(c) < 16: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    p = c.reshape(-1, 2)
    if len(p) > 360: p = p[::max(1, len(p)//360)]     # 360 pontos bastam
    p = p.astype(np.float64)
    th = np.arctan2(p[:,1]-cy, p[:,0]-cx)
    r  = np.hypot(p[:,0]-cx, p[:,1]-cy)
    o  = np.argsort(th); th, r = th[o], r[o]
    rr = np.interp(np.linspace(-np.pi, np.pi, N, endpoint=False), th, r, period=2*np.pi)
    med = rr.mean()
    if med <= 1e-6: return None
    return float(np.abs(np.fft.rfft(rr - med))[8] / (med * N / 2))


def classificar_forma(c, bw, bh, peri=None, area=None, circ=None):
    """-> 'circulo' | 'octogono' | '?'  — o '?' é zona morta proposital."""
    if peri is None: peri = cv2.arcLength(c, True)
    if peri <= 0: return "?"
    if area is None: area = cv2.contourArea(c)
    if not (FORMA_AR_MIN <= bw/max(bh,1) <= FORMA_AR_MAX): return "?"
    if circ is None: circ = 4*np.pi*area/(peri*peri)
    if circ < FORMA_CIRC_MIN: return "?"
    if area/max(cv2.contourArea(cv2.convexHull(c)), 1) < FORMA_CONVEX_MIN: return "?"
    h8 = harmonico_8(c)
    if h8 is None: return "?"
    if h8 >= FORMA_H8_OCTOG and circ >= FORMA_CIRC_OCTOG: return "octogono"
    if h8 <= FORMA_H8_CIRCULO: return "circulo"
    return "?"


def score_perspectiva_octogono(poly) -> float:
    """Os vértices cabem num octógono regular sob alguma homografia? Ruído não cabe."""
    if poly is None or len(poly) < 6: return 0.5
    pts = np.array(poly, dtype=np.float32)
    c = pts.mean(axis=0)
    pts_ord = pts[np.argsort(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))]
    theta = np.linspace(0, 2*np.pi, len(pts_ord), endpoint=False) + np.pi/8
    molde = (np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)*100 + 100)
    try:
        H, _ = cv2.findHomography(molde, pts_ord, method=0)
        if H is None: return 0.2
        proj = cv2.perspectiveTransform(molde.reshape(-1,1,2), H).reshape(-1,2)
    except Exception:
        return 0.2
    erro = float(np.mean(np.linalg.norm(proj-pts_ord, axis=1)))
    escala = float(np.linalg.norm(pts_ord.max(axis=0)-pts_ord.min(axis=0))) + 1e-6
    return float(np.clip(1.0 - (erro/escala)*4.0, 0, 1))


def score_simetria(crop_gray) -> float:
    """Placa real é simétrica ao espelhar; sombra e objeto aleatório não são."""
    if crop_gray is None or crop_gray.size == 0: return 0.5
    g = cv2.resize(crop_gray, (48,48)).astype(np.float32)
    diff = float(np.mean(np.abs(g - cv2.flip(g, 1))))/255.0
    return float(np.clip(1.0 - diff*2.2, 0, 1))


# ================================================================
#  [3] LEITURA DAS PLACAS A/B/C — dois juízes independentes
#      TOPOLOGIA (A=1 buraco, B=2, C=0) não depende de fonte;
#      MOLDE normalizado não depende de escala nem de centragem.
#      Só passa quando os dois apontam a mesma letra.
# ================================================================

_MASC_CIRC = None

def _mascara_circular(n, raio_frac=0.62):
    """Miolo do disco — o anel é igual nas três placas e só atrapalha."""
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (n//2, n//2), int(n*raio_frac/2), 255, -1)
    return m


def _binarizar_miolo(crop, n=96):
    """Recorte da placa → (binário do miolo, fração de tinta); corrige placa invertida."""
    global _MASC_CIRC
    if _MASC_CIRC is None or _MASC_CIRC.shape[0] != n:
        _MASC_CIRC = _mascara_circular(n)
    g = cv2.GaussianBlur(cv2.resize(crop, (n,n)), (3,3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b[_MASC_CIRC == 0] = 255
    tinta = float((b[_MASC_CIRC > 0] == 0).mean())
    if tinta > 0.55:                                   # disco escuro com letra clara
        b = cv2.bitwise_not(b); b[_MASC_CIRC == 0] = 255
        tinta = float((b[_MASC_CIRC > 0] == 0).mean())
    return b, tinta


def _recorte_tinta(b, n=48):
    """Recorta a letra na PRÓPRIA bbox e estica num quadrado — mata fonte/escala/centragem."""
    ys, xs = np.where(b == 0)
    if len(xs) < 30: return None, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x1-x0) < 8 or (y1-y0) < 8: return None, None
    rec = b[y0:y1+1, x0:x1+1]
    hh, ww = rec.shape; lado = max(hh, ww)
    quad = np.full((lado, lado), 255, np.uint8)
    quad[(lado-hh)//2:(lado-hh)//2+hh, (lado-ww)//2:(lado-ww)//2+ww] = rec
    return cv2.resize(quad, (n,n), interpolation=cv2.INTER_AREA), rec


def _contar_buracos(rec):
    """Buracos fechados na letra: A=1, B=2, C=0 — assinatura independente de fonte."""
    r = cv2.copyMakeBorder(rec, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    cnts, hier = cv2.findContours(cv2.bitwise_not(r), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not cnts: return None, -1
    hier = hier[0]
    a_letra = max(cv2.contourArea(c) for c in cnts)
    nb = sum(1 for i, c in enumerate(cnts)
             if hier[i][3] != -1 and cv2.contourArea(c) > 0.02*a_letra)
    return {0:"C", 1:"A", 2:"B"}.get(nb), nb


def _molde_letra(letra, lado=200, com_moldura=True):
    """Placa circular branca com anel preto e a letra ao centro."""
    img = np.full((lado, lado), 255, np.uint8)
    k = lado/200.0; c = lado//2
    if com_moldura:
        cv2.circle(img, (c,c), int(c*0.94), 0, max(2, int(round(5*k))))
    esc = LETRA_ESCALA*k; espl = max(1, int(round(LETRA_ESPESSURA*k)))
    (w, h), _ = cv2.getTextSize(letra, LETRA_FONTE, esc, espl)
    cv2.putText(img, letra, ((lado-w)//2, (lado+h)//2), LETRA_FONTE, esc, 0, espl)
    return img


class LeitorLetras:
    """Lê a letra dentro de um círculo já achado — não procura contornos."""

    def __init__(self, debug=False):
        self.moldes = {}
        for L in LETRAS_PONTOS:
            b, _ = _binarizar_miolo(_molde_letra(L, 200))
            self.moldes[L] = _recorte_tinta(b)[0]
        self.votos = deque(maxlen=LETRA_VOTOS_N)
        self.debug = debug; self._fn = 0

    def ler(self, sub, c, bbox):
        """-> (letra|None, score). Em debug, diz sempre por que reprovou."""
        x, y, bw, bh = bbox
        cru = sub[y:y+bh, x:x+bw]
        if cru.size == 0: return None, 0.0

        b, tinta = _binarizar_miolo(cru)
        letra = None; s1 = mg = 0.0; nb = -1

        if not (LETRA_TINTA_MIN < tinta < LETRA_TINTA_MAX):
            motivo = f"tinta {tinta:.2f}"
        else:
            norm, rec = _recorte_tinta(b)
            if norm is None:
                motivo = "tinta insuficiente"
            else:
                hh, ww = rec.shape
                ar = ww/max(hh,1); pre = float((rec == 0).mean())
                if not (LETRA_AR_MIN < ar < LETRA_AR_MAX):
                    motivo = f"proporção {ar:.2f}"
                elif not (LETRA_PREENCH_MIN < pre < LETRA_PREENCH_MAX):
                    motivo = f"preenchimento {pre:.2f}"
                else:
                    L_topo, nb = _contar_buracos(rec)
                    notas = sorted(((float(cv2.matchTemplate(norm, mo, cv2.TM_CCOEFF_NORMED).max()), L)
                                    for L, mo in self.moldes.items()), reverse=True)
                    (s1, L_mol), (s2, _) = notas[0], notas[1]
                    mg = s1 - s2
                    if   L_topo is None:            motivo = f"{nb} buracos"
                    elif L_topo != L_mol:           motivo = f"discordam ({L_topo}/{L_mol})"
                    elif s1 < LETRA_SCORE_MIN:      motivo = f"score {s1:.2f}"
                    elif mg < LETRA_MARGEM_MIN:     motivo = f"margem {mg:.2f}"
                    else:                           letra, motivo = L_topo, "ok"

        if self.debug:
            self._fn += 1
            if self._fn % 12 == 1:
                print(f"[ABC?] {bw}x{bh} tinta={tinta:.2f} buracos={nb} "
                      f"score={s1:.2f} margem={mg:.2f} -> {letra or motivo}", flush=True)
        return letra, s1

    def votar(self, entregas):
        """Confirma a letra só com LETRA_VOTOS_MIN de LETRA_VOTOS_N frames."""
        melhor = max(entregas, key=lambda d: d["area"]) if entregas else None
        self.votos.append(melhor["ponto"] if melhor else None)
        if melhor is None: return None
        if list(self.votos).count(melhor["ponto"]) < LETRA_VOTOS_MIN: return None
        return melhor


def gerar_placas_abc(pasta="./placas_abc", px=700):
    """--gerar-abc → PNGs de impressão com a mesma fonte do molde."""
    os.makedirs(pasta, exist_ok=True)
    for L in LETRAS_PONTOS:
        canvas = cv2.copyMakeBorder(_molde_letra(L, lado=px), 40, 110, 40, 40,
                                    cv2.BORDER_CONSTANT, value=255)
        cv2.putText(canvas, f"PONTO {L}", (46, canvas.shape[0]-34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        cv2.imwrite(os.path.join(pasta, f"ponto_{L}.png"), canvas)
    print(f"[ABC] PNGs em {pasta}/ — imprima ~15cm de diâmetro em papel FOSCO,\n"
          f"      sem recortar rente ao anel, fixados na altura da câmera.", flush=True)


# ================================================================
#  [4] VARREDURA ÚNICA — uma passada de contornos alimenta os
#      dois ramos (trânsito e delivery)
# ================================================================

def agrupar_farois_soltos(sub):
    """A moldura do semáforo raramente forma um contorno fechado próprio —
    o que o segmentador acha de verdade são as 3 ABERTURAS CIRCULARES soltas
    (cada lâmpada, acesa ou apagada, é um blob isolado). Aqui a gente acha
    esses círculos primeiro, agrupa os que estão empilhados na mesma coluna,
    e sintetiza o retângulo do corpo do semáforo ao redor do grupo — em vez
    de exigir que o retângulo já viesse pronto como um contorno único."""
    h, w = sub.shape[:2]
    blobs = []
    for thr in (cv2.threshold(sub,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
                cv2.threshold(sub,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]):
        cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (40 < a < 1400): continue
            x, y, bw, bh = cv2.boundingRect(c)
            if not (0.6 < bw/max(bh,1) < 1.7): continue          # é redondo?
            if 4*np.pi*a/max(cv2.arcLength(c,True)**2, 1) < 0.55: continue
            blobs.append((x, y, bw, bh))
    if len(blobs) < 2: return []

    blobs.sort(key=lambda b: b[1])          # de cima pra baixo
    usados = [False]*len(blobs)
    grupos = []
    for i, b in enumerate(blobs):
        if usados[i]: continue
        cx, diam = b[0]+b[2]/2, max(b[2], b[3])
        grupo = [b]; usados[i] = True
        for j in range(i+1, len(blobs)):
            if usados[j] or len(grupo) == 3: continue
            b2 = blobs[j]; cx2 = b2[0]+b2[2]/2
            # mesma coluna (x parecido) e espaçamento vertical plausível
            if abs(cx2-cx) < diam*0.7 and 0 < (b2[1]-grupo[-1][1]) < diam*4:
                grupo.append(b2); usados[j] = True
        if 2 <= len(grupo) <= 3:
            xs = [g[0] for g in grupo]; xe = [g[0]+g[2] for g in grupo]
            ys = [g[1] for g in grupo]; ye = [g[1]+g[3] for g in grupo]
            diam_med = float(np.mean([max(g[2],g[3]) for g in grupo]))
            mx, my = int(diam_med*0.35), int(diam_med*0.5)   # margem: molde/corpo ao redor
            x1r = max(0, min(xs)-mx); y1r = max(0, min(ys)-my)
            x2r = min(w, max(xe)+mx); y2r = min(h, max(ye)+my)
            if x2r > x1r and y2r > y1r:
                grupos.append((x1r, y1r, x2r-x1r, y2r-y1r))
    return grupos


def detectar_geometrico(gray):
    """-> candidatos {'Stop','Semaforo','Delivery'}; Delivery traz 'ponto' e 'score'."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = ROI_DIN.janela(h, w)
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0: return []

    cands = []
    n_letra = [0, 0]                    # orçamento separado: [circulo, "?"]

    # DESATIVADO — o agrupador de faróis soltos causou regressão grave em
    # teste real (inundou candidatos de "Semaforo" falso, atropelou PARE e
    # Delivery). Revertido até calibrar com frame real da câmera, não foto
    # de tela. A função agrupar_farois_soltos() continua definida abaixo,
    # só não é mais chamada aqui.

    for m in segmentar(sub):
        # CHAIN_APPROX_NONE: harmonico_8 precisa da borda inteira, não só dos cantos
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < GEO_AREA_MIN or a > GEO_AREA_FRAC*h*w: continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 20 or bh < 20: continue
            ar = bw/max(bh,1); solid = a/max(bw*bh,1)
            if solid < 0.35: continue
            peri = cv2.arcLength(c, True)
            circ = 4*np.pi*a/max(peri*peri, 1)
            crop = sub[y:y+bh, x:x+bw]

            if 0.25 < ar < 0.62 and bh > 50 and _tem_farois(crop):
                hint, forma, ponto, score = "Semaforo", "?", None, 0.0

            else:
                forma = classificar_forma(c, bw, bh, peri, a, circ)
                if forma == "octogono":
                    hint, ponto, score = "Stop", None, 0.0

                elif forma in ("circulo", "?") and a >= AREA_MIN_LEITURA:
                    # orçamento separado: o "?" é abundante em frame sujo e,
                    # com teto único, consumia as vagas antes do círculo real
                    i = 0 if forma == "circulo" else 1
                    n_letra[i] += 1
                    if n_letra[i] > (LETRA_MAX_CIRC if i == 0 else LETRA_MAX_DUVIDA): continue
                    ponto, score = LEITOR.ler(sub, c, (x, y, bw, bh))
                    if ponto is None: continue
                    if forma == "?" and score < LETRA_SCORE_FORTE: continue   # conteúdo vence forma duvidosa
                    hint, forma = "Delivery", "circulo"
                else:
                    continue

            ok, _motivo = peneira_roi(crop, (bw*bh)/float(h*w), exige_simbolo=(hint == "Stop"))
            if not ok: continue

            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            poly = approx.reshape(-1,2).tolist()
            persp = score_perspectiva_octogono(poly) if hint == "Stop" else 1.0
            simet = score_simetria(crop)             if hint == "Stop" else 1.0
            d = {"bbox": (x+x0, y+y0, x+x0+bw, y+y0+bh), "class_name": hint, "class_id": -1,
                 "conf": float(solid), "geo": True, "lados": len(approx), "circ": float(circ),
                 "ar": float(ar), "area": float(a), "simbolo": True, "poly": poly, "forma": forma,
                 "convex": float(a/max(cv2.contourArea(cv2.convexHull(c)),1)),
                 "persp": float(persp), "simet": float(simet)}
            if hint == "Delivery":
                d["ponto"] = ponto; d["score"] = float(score)
            cands.append(d)

    # dedup por IoU sobre a UNIÃO — por contenção a folha branca engoliria o disco
    cands.sort(key=lambda d: -d["circ"])
    keep = []
    for d in cands:
        ax1, ay1, ax2, ay2 = d["bbox"]; dup = False
        for k in keep:
            bx1, by1, bx2, by2 = k["bbox"]
            inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
            uniao = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            if inter > 0.70*max(uniao, 1): dup = True; break
        if not dup: keep.append(d)
    return keep[:GEO_MAX_CANDS]


# ================================================================
#  [5] PGOM — acumula evidência geométrica ao longo dos frames.
#      Só quem passa de PGOM_PROMOVE chega à CNN.
# ================================================================

class Ficha:
    """Histórico de um candidato: quanto mais coerente no tempo, maior a evidência."""
    _nid = 0

    def __init__(self, det):
        self.id = Ficha._nid; Ficha._nid += 1
        d = lambda: deque(maxlen=PGOM_HIST)
        self.centros, self.areas, self.formas = d(), d(), d()
        self.convexs, self.aspectos, self.simbolos = d(), d(), d()
        self.fingerps, self.persps, self.simets = d(), d(), d()
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
        """Assinatura grosseira — se muda a cada frame, é ruído, não placa."""
        nv = det.get("lados", 0)
        grupo = 3 if nv == 3 else 4 if nv in (4,5) else 7 if nv in (6,7) else 9
        return (grupo, int(det.get("circ",0)*3), int(det.get("ar",1.0)*3))

    def _push(self, det):
        x1, y1, x2, y2 = det["bbox"]
        self.centros.append(((x1+x2)/2, (y1+y2)/2))
        self.areas.append(max(1.0, (x2-x1)*(y2-y1)))
        self.formas.append(self._score_forma(det))
        self.convexs.append(float(det.get("convex", 0.5)))
        ideal = 0.40 if det["class_name"] == "Semaforo" else 1.00
        self.aspectos.append(float(np.clip(1.0 - abs(float(det.get("ar",1.0))-ideal)/ideal, 0, 1)))
        self.simbolos.append(1.0 if det.get("simbolo") else 0.0)
        self.fingerps.append(self._fingerprint(det))
        self.persps.append(float(det.get("persp", 0.5)))
        self.simets.append(float(det.get("simet", 0.5)))
        self.vistos += 1; self.missed = 0; self.det = det

    def perto_de(self, det):
        x1, y1, x2, y2 = det["bbox"]
        px, py = self.centros[-1]
        return (((x1+x2)/2-px)**2 + ((y1+y2)/2-py)**2) ** 0.5

    def evidencias(self):
        e_persist = float(np.clip(self.vistos/PGOM_PERSIST_N, 0, 1))
        if len(self.centros) >= 3:
            passos = [((self.centros[i+1][0]-self.centros[i][0])**2 +
                       (self.centros[i+1][1]-self.centros[i][1])**2)**0.5
                      for i in range(len(self.centros)-1)]
            razoes = [self.areas[i+1]/self.areas[i] for i in range(len(self.areas)-1)]
            e_estab = (float(np.clip(1.0-np.std(passos)/12.0, 0, 1)) +
                       float(np.clip(1.0-np.std(razoes)/0.25, 0, 1))) / 2
        else:
            e_estab = 0.4
        e_fp = Counter(self.fingerps).most_common(1)[0][1]/len(self.fingerps)
        return dict(forma=float(np.mean(self.formas)), convex=float(np.mean(self.convexs)),
                    aspecto=float(np.mean(self.aspectos)), simbolo=float(np.mean(self.simbolos)),
                    persist=e_persist, estab=float(e_estab), fingerprint=float(e_fp),
                    persp=float(np.mean(self.persps)), simet=float(np.mean(self.simets)))

    def total(self):
        pesos = PESOS_SEMAFORO if self.det["class_name"] == "Semaforo" else PESOS_PLACA
        ev = self.evidencias()
        return sum(pesos[k]*ev[k] for k in pesos)


class PGOM:
    """Associa detecções a fichas por proximidade e promove as que juntaram evidência."""
    def __init__(self):
        self.fichas = []

    def update(self, dets):
        geo = [d for d in dets if d.get("geo")]
        outros = [d for d in dets if not d.get("geo")]
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
                promovidos.append(d)
        return promovidos + outros

PGOM_M = PGOM()


def score_final(det, cnn_conf):
    """A evidência geométrica pesa 85%; a CNN só desempata."""
    return 0.85*float(det.get("evtot", det.get("conf", 0.5))) + 0.15*float(cnn_conf)


# ================================================================
#  [6] REDES AUXILIARES — YOLO (detecção), CNN (classificação fina),
#      OOD (rejeição), Verificador (contrato forma↔classe)
# ================================================================

class YOLODetector:
    """ONNX Runtime; aceita YOLOv8 COCO pré-treinado ou modelo próprio."""

    def __init__(self, path, esperar_custom=False):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(path, opts,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.in_n  = self.sess.get_inputs()[0].name
        self.out_n = self.sess.get_outputs()[0].name
        hm, wm = self.sess.get_inputs()[0].shape[2:4]
        self.input_size = hm if isinstance(hm, int) and isinstance(wm, int) else YOLO_SIZE
        o1 = self.sess.get_outputs()[0].shape[1]
        self.coco_mode = ((o1 - 4) == 80) if isinstance(o1, int) else True
        print(f"[YOLO] {path} | {self.sess.get_providers()[0]} | "
              f"{'COCO' if self.coco_mode else 'custom'} | {self.input_size}px", flush=True)
        if esperar_custom and self.coco_mode:
            raise RuntimeError(
                f"{path} é um modelo COCO (80 classes), esperava o custom Stop/Semaforo. "
                f"Você apontou o arquivo errado — confira YOLO_MODEL.")
        if not esperar_custom and not self.coco_mode:
            raise RuntimeError(
                f"{path} não é o modelo COCO esperado (achou {o1-4} classes, não 80). "
                f"Confira COCO_MODEL.")
        if not self.coco_mode and isinstance(o1, int):
            nc_modelo = o1 - 4
            if nc_modelo != len(CLASSES):
                print(f"[YOLO][ERRO FATAL] modelo tem {nc_modelo} classes na saída, "
                      f"mas CLASSES (classes.txt) tem {len(CLASSES)}: {CLASSES}. "
                      f"Os nomes vão sair TROCADOS silenciosamente — corrija o classes.txt "
                      f"antes de continuar (deve ter {nc_modelo} linha(s), nessa ordem exata).",
                      flush=True)
                raise RuntimeError(
                    f"classes.txt incompatível: modelo={nc_modelo} classes, CLASSES={len(CLASSES)}")

    @staticmethod
    def _letterbox(img, size):
        h, w = img.shape[:2]; sc = size/max(h,w); nh, nw = int(h*sc), int(w*sc)
        canvas = np.full((size,size,3), 114, np.uint8)
        py, px = (size-nh)//2, (size-nw)//2
        canvas[py:py+nh, px:px+nw] = cv2.resize(img, (nw,nh))
        return canvas, sc, px, py

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        if not len(boxes): return []
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2-x1)*(y2-y1); order = scores.argsort()[::-1]; keep = []
        while len(order):
            i = order[0]; keep.append(i)
            if len(order) == 1: break
            inter = (np.maximum(0, np.minimum(x2[i],x2[order[1:]]) - np.maximum(x1[i],x1[order[1:]])) *
                     np.maximum(0, np.minimum(y2[i],y2[order[1:]]) - np.maximum(y1[i],y1[order[1:]])))
            iou = inter/(areas[i]+areas[order[1:]]-inter+1e-9)
            order = order[1:][iou < iou_thr]
        return keep

    def detectar(self, frame):
        if frame.ndim == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h0, w0 = frame.shape[:2]
        canvas, sc, px, py = self._letterbox(frame, self.input_size)
        inp = (canvas[:,:,::-1].astype(np.float32)/255.0).transpose(2,0,1)[np.newaxis]
        preds = self.sess.run([self.out_n], {self.in_n: inp})[0][0].T
        cls_sc = preds[:,4:]; max_c = cls_sc.max(axis=1)
        mask = max_c >= YOLO_CONF
        if not mask.any(): return []
        preds, cls_sc, max_c = preds[mask], cls_sc[mask], max_c[mask]
        cx, cy, bw, bh = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
        bxs = np.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], axis=1)
        cls_ids = cls_sc.argmax(axis=1)
        results = []
        for cid in np.unique(cls_ids):
            if self.coco_mode:
                if int(cid) not in COCO_MAP: continue
                nome = COCO_MAP[int(cid)]
            else:
                nome = CLASSES[int(cid)] if int(cid) < NUM_CLASSES else "?"
            idx = np.where(cls_ids == cid)[0]
            for k in self._nms(bxs[idx], max_c[idx], YOLO_NMS):
                i = idx[k]
                x1 = int(np.clip((bxs[i,0]-px)/sc, 0, w0-1)); y1 = int(np.clip((bxs[i,1]-py)/sc, 0, h0-1))
                x2 = int(np.clip((bxs[i,2]-px)/sc, 0, w0-1)); y2 = int(np.clip((bxs[i,3]-py)/sc, 0, h0-1))
                if y1 < h0*ROI_Y0 or y2 > h0*ROI_Y1: continue
                if self.coco_mode:
                    if max_c[i] < COCO_CONF_MIN.get(nome, YOLO_CONF): continue
                    if (x2-x1)*(y2-y1) < COCO_AREA_MIN.get(nome, 0): continue
                results.append({"bbox": (x1,y1,x2,y2), "class_name": nome,
                                "class_id": int(cid), "conf": float(max_c[i]), "coco": self.coco_mode})
        return results


class CNNClassifier:
    """MobileNetV3Small em TFLite — só classifica o que o PGOM já promoveu."""

    def __init__(self, path):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path); interp.allocate_tensors()
        d = interp.get_input_details()[0]
        self._interp = interp; self._in = d["index"]
        self._out = interp.get_output_details()[0]["index"]
        self._q = d["dtype"] == np.uint8
        self.size = int(d["shape"][1])
        global CNN_SIZE; CNN_SIZE = self.size
        print(f"[CNN] {self.size}px {'INT8' if self._q else 'FP32'}", flush=True)

    def predict(self, img):
        inp = img[np.newaxis].astype(np.float32)
        if self._q: inp = (inp*255).astype(np.uint8)
        self._interp.set_tensor(self._in, inp); self._interp.invoke()
        out = self._interp.get_tensor(self._out)[0]
        return out.astype(np.float32)/255.0 if self._q else out


def prep_mono(crop):
    """Recorte → tensor 3 canais normalizado, do jeito que a CNN foi treinada."""
    if crop.ndim == 3: crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = _CLAHE.apply(cv2.resize(crop, (CNN_SIZE, CNN_SIZE)))
    return np.stack([g,g,g], axis=-1).astype(np.float32)/255.0


def carregar_classes():
    """Lê o classes.txt AO LADO do modelo YOLO (models/classes.txt) — nunca
    o da CNN. Plug-and-play: qualquer novo treino de YOLO só precisa trocar
    esse arquivo, sem tocar em código. Convenção interna: a placa de PARE
    deve se chamar 'Stop' aqui (não 'Pare'), pra bater com FORMA_CLASSE,
    COCO_MAP e o resto do pipeline que já usa esse nome."""
    p = os.path.join(os.path.dirname(YOLO_MODEL), "classes.txt")
    if os.path.exists(p):
        with open(p) as f: cls = [l.strip() for l in f if l.strip()]
        if cls:
            print(f"[YOLO] classes.txt: {cls}", flush=True)
            if len(cls) != 2 or "Semaforo" not in cls:
                print(f"[YOLO][AVISO] esperava 2 classes (Stop, Semaforo) — "
                      f"achou {len(cls)}: {cls}. Se isso veio do TRAIN_YOLO.py antigo "
                      f"(8 classes), é um classes.txt órfão — apague ou regenere.", flush=True)
            return cls
    print(f"[YOLO] classes.txt não encontrado, usando default: {CLASSES}", flush=True)
    return CLASSES


class OODRejector:
    """Softmax alta não é evidência — abaixo do limiar da classe, descarta."""
    def __init__(self, path):
        self._t = {}
        if os.path.exists(path):
            with open(path) as f: self._t = json.load(f)
    def aceitar(self, cls, score):
        return score >= self._t.get(cls, OOD_DEFAULT)


class VerificadorPlaca:
    """Contrato forma↔classe: PARE só é aceito se a geometria disser octógono."""
    FORMA_CLASSE = {"Stop": {"octogono"}}

    def __init__(self, margem=0.25, entropia_max=1.30):
        self.margem_min = margem; self.entropia_max = entropia_max

    def e_placa(self, scores, cls_nm, det):
        ordenado = np.sort(scores)[::-1]
        margem = float(ordenado[0]) - (float(ordenado[1]) if len(ordenado) > 1 else 0.0)
        p = np.clip(scores, 1e-9, 1.0)
        if margem < self.margem_min: return False, "margem baixa", margem
        if float(-(p*np.log(p)).sum()) > self.entropia_max: return False, "entropia alta", margem
        esperadas = self.FORMA_CLASSE.get(cls_nm)
        if esperadas is not None:
            fg = det.get("forma", "?")
            if fg != "?" and fg not in esperadas: return False, f"forma {fg}", margem
        return True, "ok", margem


def geometria_concorda(gray, det) -> bool:
    """YOLO diz PARE → é octógono? Não → descarta. Classes sem contrato passam direto."""
    esperadas = VerificadorPlaca.FORMA_CLASSE.get(det.get("class_name"))
    if esperadas is None: return True
    x1, y1, x2, y2 = det["bbox"]; m = 4       # folga: o box do YOLO corta a borda
    crop = gray[max(0,y1-m):min(gray.shape[0],y2+m), max(0,x1-m):min(gray.shape[1],x2+m)]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        if DEBUG_FUNIL: print(f"[geo] {det.get('class_name')} crop inválido, descartado", flush=True)
        return False
    _, b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(cv2.morphologyEx(b, cv2.MORPH_CLOSE, K5),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        if DEBUG_FUNIL: print(f"[geo] {det.get('class_name')} sem contorno, descartado", flush=True)
        return False
    c = max(cnts, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(c)
    det["forma"] = classificar_forma(c, bw, bh)
    ok = det["forma"] in esperadas
    if DEBUG_FUNIL and not ok:
        print(f"[geo] YOLO disse {det.get('class_name')} (conf={det.get('conf',0):.2f}) "
              f"mas forma={det['forma']}, descartado", flush=True)
    return ok


# ================================================================
#  [7] RASTREIO E VOTAÇÃO — ByteTrack-lite (só ramo de trânsito)
# ================================================================

class Track:
    _nid = 0
    N_CONSECUTIVO = 3       # PARE: 3 consecutivos
    COR_N_CONSECUTIVO = 4   # semáforo (vermelho/verde): 4 consecutivos — mais rigoroso

    def __init__(self, bbox, hint, conf):
        self.id = Track._nid; Track._nid += 1
        self.bbox = bbox; self.class_hint = hint; self.conf = conf
        self.buf = deque(maxlen=VOTE_BUFFER); self.age = 0; self.missed = 0
        self.state = "tentative"
        self.evt_prox_enviado = False   # legado — mantido por compat, não usado mais pra Stop/Semaforo
        self.nivel_prox = 0             # quantos níveis de AREA_PROX_NIVEIS já foram avisados
        self.evt_chegada_enviado = False
        self.evt_conf_enviado = None    # já confirmou (e qual valor) pro Portenta?
        self.cor_buf = deque(maxlen=5)  # histórico de cor do semáforo (separado do buf de classe)
        self.stop_consumido = False     # já mandou parar por causa deste track? não repete até ele sumir

    def atualizar(self, bbox, hint, conf, cnn_lbl, cnn_conf):
        self.bbox = bbox; self.class_hint = hint; self.conf = conf
        self.missed = 0; self.age += 1
        if cnn_lbl: self.buf.append((cnn_lbl, cnn_conf))
        if self.age >= 2: self.state = "confirmed"   # era 3 — em movimento, 2 já filtra ruído de 1 frame

    def consecutivo(self):
        """Estrito: as últimas N_CONSECUTIVO leituras têm que ser TODAS iguais —
        diferente de votar() (maioria no buffer inteiro). É o que o Portenta precisa
        pra tratar como confirmação de verdade, não uma tendência estatística."""
        if len(self.buf) < self.N_CONSECUTIVO: return None
        ultimos = [lb for lb, _ in list(self.buf)[-self.N_CONSECUTIVO:]]
        if len(set(ultimos)) == 1: return ultimos[0]
        return None

    def cor_consecutiva(self, cor_atual):
        """Mesmo critério, só que pra cor do semáforo, que muda frame a frame
        e não vive no buf de classe. Vale igual pras 3 cores (precisa de
        COR_N_CONSECUTIVO iguais em sequência pra reportar) — o tratamento
        especial do amarelo (não libera, não cancela vermelho) é regra da
        Missão, não deste método."""
        self.cor_buf.append(cor_atual)
        if len(self.cor_buf) < self.COR_N_CONSECUTIVO: return None
        ultimos = list(self.cor_buf)[-self.COR_N_CONSECUTIVO:]
        if None not in ultimos and len(set(ultimos)) == 1: return ultimos[0]
        return None

    def votar(self):
        """Maioria qualificada no buffer — um frame isolado nunca decide."""
        if len(self.buf) < VOTE_MIN_DETS:
            if DEBUG_FUNIL:
                print(f"[voto] track#{self.id} buffer={len(self.buf)}/{VOTE_MIN_DETS} insuficiente", flush=True)
            return None, 0.0
        top, n = Counter(lb for lb,_ in self.buf).most_common(1)[0]
        if n/len(self.buf) < VOTE_FRAC:
            if DEBUG_FUNIL:
                print(f"[voto] track#{self.id} maioria {n}/{len(self.buf)} < {VOTE_FRAC}, não confirmado", flush=True)
            return None, 0.0
        return top, float(np.mean([c for lb,c in self.buf if lb == top]))

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return max(0, (x2-x1)*(y2-y1))


class ByteTrackLite:
    """Associa em duas passadas: primeiro alta confiança, depois o resto."""

    def __init__(self):
        self.tracks = []

    @staticmethod
    def _iou(a, b):
        inter = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
        if not inter: return 0.0
        aA = (a[2]-a[0])*(a[3]-a[1]); aB = (b[2]-b[0])*(b[3]-b[1])
        return inter/(aA+aB-inter+1e-9)

    def _match(self, tracks_idx, det_idx, dets, iou_min):
        pares = sorted(((self._iou(self.tracks[ti].bbox, dets[di]["bbox"]), ti, di)
                        for ti in tracks_idx for di in det_idx), reverse=True)
        mt, md, casados = set(), set(), []
        for iou, ti, di in pares:
            if iou < iou_min or ti in mt or di in md: continue
            mt.add(ti); md.add(di); casados.append((ti, di))
        return casados, [i for i in tracks_idx if i not in mt], [i for i in det_idx if i not in md]

    def update(self, dets, frame, cnn, ood, verif=None):
        hi = [i for i,d in enumerate(dets) if d["conf"] >= CONF_HIGH_THR]
        lo = [i for i,d in enumerate(dets) if d["conf"] <  CONF_HIGH_THR]
        c1, resta_t, novos = self._match(list(range(len(self.tracks))), hi, dets, TRACK_IOU_HIGH)
        c2, resta_t2, _    = self._match(resta_t, lo, dets, TRACK_IOU_LOW)
        for ti, di in c1 + c2:
            lbl, conf = self._classificar(dets[di], frame, cnn, ood, verif)
            self.tracks[ti].atualizar(dets[di]["bbox"], dets[di]["class_name"],
                                      dets[di]["conf"], lbl, conf)
        for ti in resta_t2: self.tracks[ti].missed += 1
        for di in novos:
            d = dets[di]
            lbl, conf = self._classificar(d, frame, cnn, ood, verif)
            t = Track(d["bbox"], d["class_name"], d["conf"])
            if lbl: t.buf.append((lbl, conf))
            self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= TRACK_MAX_AGE]
        return self.tracks

    @staticmethod
    def _classificar(det, frame, cnn, ood, verif=None):
        """Portão final: CNN → verificador → OOD. Qualquer 'não' vira None."""
        cn = det.get("class_name", "?")
        if det.get("coco", False): return det["class_name"], det["conf"]
        if det.get("geo") and det["class_name"] == "Semaforo":
            return "Semaforo", score_final(det, det["conf"])
        if cnn is None:
            if DEBUG_FUNIL: print(f"[cnn] {cn}: CNN indisponível, descartado", flush=True)
            return None, 0.0
        x1, y1, x2, y2 = det["bbox"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or (x2-x1) < 8 or (y2-y1) < 8:
            if DEBUG_FUNIL: print(f"[cnn] {cn}: crop pequeno demais, descartado", flush=True)
            return None, 0.0
        scores = cnn.predict(prep_mono(crop))
        i = int(scores.argmax())
        nome = CNN_CLASSES[i] if i < len(CNN_CLASSES) else None
        if nome in (None, "Fundo"):
            if DEBUG_FUNIL:
                print(f"[cnn] {cn}: CNN disse Fundo (scores={np.round(scores,2)}), descartado", flush=True)
            return None, 0.0
        if verif is not None:
            ok, motivo, margem = verif.e_placa(scores, nome, det)
            if not ok:
                if DEBUG_FUNIL: print(f"[pgom] {nome}: verificador rejeitou ({motivo}, margem={margem:.2f})", flush=True)
                return None, 0.0
        if ood and not ood.aceitar(nome, float(scores.max())):
            if DEBUG_FUNIL:
                thr = ood._t.get(nome, OOD_DEFAULT)
                print(f"[ood] {nome}: score={scores.max():.2f} < thr={thr:.2f}, descartado", flush=True)
            return None, 0.0
        return nome, score_final(det, float(scores.max()))


# ================================================================
#  [8] MISSÃO — "o que isso significa?"
#      VISÃO diz o que existe · MISSÃO decide · CONTROLE executa.
#      Prioridade absoluta: obstáculo > vermelho > PARE > rota.
# ================================================================

class Missao:
    """FSM da entrega; guarda o estado de retorno para retomar após parada absoluta."""
    RODANDO = ("IND_RETIRADA", "IND_ENTREGA", "LIVRE")
    PARADAS = ("PARADO_PARE", "PARADO_SEM", "PARADO_OBST")

    def __init__(self):
        self.estado   = "AGUARDANDO"
        self.retirada = None
        self.entrega  = None
        self.fase     = "retirada"        # o que ainda falta cumprir
        self.retorno  = "LIVRE"           # p/ onde voltar após parada absoluta
        self.t_estado = time.monotonic()
        self.entregas = 0
        self.qr_wifi  = None              # QR 1: entra na rede
        self.qr_img   = None              # QR 2: abre o painel

    def nova_sessao(self):
        """Dois QRs fixos no canto: um entra no Wi-Fi, outro abre o painel."""
        url = BT.url if (BT is not None and getattr(BT, "url", None)) \
              else f"http://{PainelLocal._meu_ip()}:{PAINEL_PORTA}/"
        self.qr_wifi = gerar_qr_wifi(WIFI_SSID, WIFI_PASS, lado=120)
        self.qr_img  = gerar_qr(url, lado=120)
        print(f"[QR] Wi-Fi '{WIFI_SSID}' → painel {url}", flush=True)

    def set_estado(self, novo):
        if novo != self.estado:
            print(f"[MISSAO] {self.estado} → {novo}", flush=True)
            self.estado = novo; self.t_estado = time.monotonic()

    def tempo_no_estado(self):
        return time.monotonic() - self.t_estado

    def parado_por_regra(self):
        return self.estado in self.PARADAS

    def alvo(self):
        """Qual letra interessa AGORA — as outras são ignoradas."""
        if self.fase == "retirada" and self.retirada: return self.retirada
        if self.fase == "entrega"  and self.entrega:  return self.entrega
        return None

    @staticmethod
    def regra_absoluta(percep):
        if percep.get("obstaculo"):              return "PARADO_OBST"
        if percep.get("semaforo") == "vermelho": return "PARADO_SEM"
        if percep.get("pare"):                   return "PARADO_PARE"
        return None

    def status(self):
        return (f"ST;estado={self.estado};rota={self.retirada or '-'}>{self.entrega or '-'};"
                f"fase={self.fase};entregas={self.entregas}")

MISSAO = Missao()


def pare_satisfeito(percep):
    """Verdadeiro se não há PARE ativo agora, OU se já ficamos parados o
    tempo mínimo com o PARE ativo — mesmo que o motivo de ter parado tenha
    sido outro (ex.: semáforo vermelho no mesmo cruzamento). Sem isso, o
    carro podia sair andando no verde sem nunca ter 'pago' o PARE ao lado."""
    if not percep.get("pare"):
        return True
    if MISSAO.tempo_no_estado() >= 2.5:
        trk = percep.get("pare_trk")
        if trk is not None: trk.stop_consumido = True
        return True
    return False


def semaforo_libera(est, percep):
    """Amarelo NUNCA libera o carro de um PARADO_SEM, e nunca cancela um
    vermelho anterior — é transição, não permissão. Só o verde libera
    quando o motivo de ter parado foi o próprio semáforo. Nos outros
    estados de parada (PARE, obstáculo), quem não foi causado pelo
    semáforo só continua bloqueado se o semáforo estiver EM vermelho
    agora — amarelo ali não segura o carro por conta própria."""
    cor = percep.get("semaforo")
    if cor == "vermelho": return False
    if est == "PARADO_SEM": return cor == "verde"
    return True


# ================================================================
#  [9] CONTROLE E SERIAL
# ================================================================

CMD  = dict(mot=0, srv=127, buz=0, led=0, brk=0, dir=0, spd=0)
_seq = dict(n=0, pendente=None, t_envio=0.0)
_buz = dict(ate=0.0)
_ultimo = dict(acao=None, ponto=None)


def conectar_serial():
    """Acha a Portenta pela descrição da porta; sem ela, roda em simulação."""
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower()
               for k in ("arduino","ch340","cp210","uart","portenta")):
            try:
                s = serial.Serial(p.device, BAUD, timeout=0, write_timeout=0)
                time.sleep(2); s.reset_input_buffer()
                print(f"[SER] {p.device}", flush=True); return s
            except Exception: pass
    try:
        s = serial.Serial(SERIAL_PORT, BAUD, timeout=0, write_timeout=0)
        time.sleep(2); s.reset_input_buffer()
        print(f"[SER] {SERIAL_PORT}", flush=True); return s
    except Exception:
        print("[SER] sem porta — modo simulação", flush=True); return None


def enviar(cmd, ser):
    """Uma linha JSON por comando, com seq para o ack do firmware."""
    _seq["n"] += 1
    j = (f'{{"seq":{_seq["n"]},"mot":{cmd["mot"]},"srv":{cmd["srv"]},"buz":{cmd["buz"]},'
         f'"led":{cmd["led"]},"brk":{cmd["brk"]},"dir":{cmd["dir"]},'
         f'"spd":{0 if cmd["mot"]==0 else (1 if cmd["mot"]<50 else 2)}}}')
    if ser:
        try:
            ser.write((j+"\n").encode())
            _seq["pendente"] = _seq["n"]; _seq["t_envio"] = time.monotonic()
        except Exception: pass


def sinalizar_proximidade(tipo, ser):
    """LOG 1 pro Portenta: 'tem uma sinaleira/ponto chegando, se prepare.'
    Enviado UMA VEZ por aproximação — antes de qualquer confirmação de cor/PARE.
    tipo: 'PARE' | 'SEM' | 'ENTREGA'"""
    print(f"[SERIAL] proximidade -> {tipo}", flush=True)
    if ser:
        try: ser.write((f'{{"evt":"PROX","tipo":"{tipo}"}}\n').encode())
        except Exception: pass


def sinalizar_confirmacao(tipo, valor, ser):
    """LOG 2 pro Portenta: 'confirmado, é isso mesmo — decide aí o que fazer.'
    Enviado UMA VEZ por evento, depois de N_CONSECUTIVO frames concordando.
    O Portenta passa a decidir parar/seguir com essa informação — o Python
    não manda mais mot/srv calculado pra esse caso, só o fato confirmado.
    tipo: 'PARE' | 'SEM' | 'ENTREGA'   valor: 'ok' | 'vermelho'/'amarelo'/'verde' | letra ABC"""
    print(f"[SERIAL] confirmação -> {tipo}:{valor}", flush=True)
    if ser:
        try: ser.write((f'{{"evt":"CONF","tipo":"{tipo}","valor":"{valor}"}}\n').encode())
        except Exception: pass


def sinalizar_chegada(tipo, ser):
    """LOG 3 pro Portenta: 'chegou perto o suficiente — executa agora.'
    Separado da confirmação de cor/classe: é só sobre DISTÂNCIA (área do
    bbox como proxy de 'cm'), dispara uma vez ao cruzar AREA_CHEGADA_SINALEIRA."""
    print(f"[SERIAL] chegada -> {tipo}", flush=True)
    if ser:
        try: ser.write((f'{{"evt":"CHEG","tipo":"{tipo}"}}\n').encode())
        except Exception: pass


def ler_serial(ser):
    """Consome acks; retransmite uma vez se o firmware não confirmou em 200ms."""
    if not ser: return
    try:
        while ser.in_waiting:
            linha = ser.readline().decode(errors="ignore").strip()
            if not linha: continue
            try:
                m = json.loads(linha)
                if m.get("ack") == _seq["pendente"]: _seq["pendente"] = None
            except json.JSONDecodeError: pass
    except Exception: pass
    if _seq["pendente"] is not None and time.monotonic() - _seq["t_envio"] > 0.2:
        _seq["pendente"] = None; enviar(CMD, ser)


def buzinar(ser, dur_s):
    """Liga o buzzer e agenda o desligamento — sem travar o loop."""
    CMD.update(buz=1); enviar(CMD, ser)
    _buz["ate"] = time.monotonic() + dur_s


def tick_buzzer(ser):
    if _buz["ate"] and time.monotonic() >= _buz["ate"]:
        _buz["ate"] = 0.0; CMD.update(buz=0); enviar(CMD, ser)


def parar(ser, motivo=""):
    """Freia e mantém parado — quem decide voltar é a missão, nunca um timer."""
    if _ultimo["acao"] == "STOP" and CMD["mot"] == 0: return
    CMD.update(mot=0, srv=127, led=1, brk=1, dir=0); enviar(CMD, ser)
    _ultimo["acao"] = "STOP"


def seguir(ser, com_buzzer=True):
    """Retoma cruzeiro; o beep curto sinaliza a partida."""
    if com_buzzer: buzinar(ser, BUZ_PARTIDA_S)
    CMD.update(mot=MOT_CRUZEIRO, srv=127, led=0, brk=0, dir=3); enviar(CMD, ser)
    _ultimo["acao"] = "STRAIGHT"


# ================================================================
#  [10] PAINEL LOCAL — HTTP na rede do TP-Link, sem internet
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
    """Servidor HTTP em thread separada — nunca bloqueia o loop de visão."""

    def __init__(self, porta=PAINEL_PORTA):
        import threading, queue
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import urlparse, parse_qs
        self.fila = queue.Queue()
        self.ip   = self._meu_ip()
        self.url  = f"http://{self.ip}:{porta}/"
        painel = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
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
                    if   a in ("r","e") and p in ("A","B","C"): painel.fila.put((a.upper(), p))
                    elif a == "off":                            painel.fila.put(("DESLIGAR", None))
                    self._envia(json.dumps(dict(estado=MISSAO.estado, retirada=MISSAO.retirada,
                                                entrega=MISSAO.entrega, fase=MISSAO.fase,
                                                entregas=MISSAO.entregas)), "application/json")
                else:
                    self._envia(PAGINA.replace("CAR_ID_AQUI", CAR_ID))

        try:
            self.srv = ThreadingHTTPServer(("0.0.0.0", porta), H)
            threading.Thread(target=self.srv.serve_forever, daemon=True).start()
            print(f"[PAINEL] {self.url}", flush=True)
        except Exception as e:
            self.srv = None
            print(f"[PAINEL] não subiu ({e}) — use as teclas 1/2/3", flush=True)

    @staticmethod
    def _meu_ip():
        """Pergunta ao SO qual interface alcançaria o gateway — nada é enviado."""
        import socket
        for alvo in (WIFI_GATEWAY, "192.168.1.1"):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((alvo, 80)); ip = s.getsockname()[0]; s.close()
                if ip and ip != "0.0.0.0": return ip
            except Exception: continue
        return "127.0.0.1"

    def poll(self):
        """-> comandos pendentes, sem bloquear."""
        import queue
        cmds = []
        while True:
            try: cmds.append(self.fila.get_nowait())
            except queue.Empty: return cmds

    def fechar(self):
        try:
            if self.srv: self.srv.shutdown()
        except Exception: pass


def gerar_qr(url, lado=260):
    """QR do painel; sem a lib qrcode, devolve um cartão com a URL legível."""
    try:
        import qrcode
        q = cv2.resize(np.array(qrcode.make(url).convert("L")), (lado,lado),
                       interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(q, cv2.COLOR_GRAY2BGR)
    except Exception:
        card = np.full((lado,lado,3), 255, np.uint8)
        cv2.putText(card, "pip install qrcode", (12,40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
        for i in range(0, len(url), 26):
            cv2.putText(card, url[i:i+26], (10, 80+(i//26)*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1)
        return card


def gerar_qr_wifi(ssid, senha, lado=150):
    """Formato WIFI: que iOS e Android reconhecem pela câmera, sem app."""
    import re
    esc = lambda t: re.sub(r'([\\;,:"])', r'\\\1', t)   # exigência do payload WIFI:
    return gerar_qr(f"WIFI:T:WPA;S:{esc(ssid)};P:{esc(senha)};H:false;;", lado=lado)


# ================================================================
#  [11] HUD
# ================================================================

def desenhar(frame_e, tracks, fps, modo):
    out = cv2.cvtColor(frame_e, cv2.COLOR_GRAY2BGR) if frame_e.ndim == 2 else frame_e.copy()
    h, w = out.shape[:2]; PW = 225
    cv2.rectangle(out, (0,int(h*ROI_Y0)), (w-1,int(h*ROI_Y1)), (0,200,255), 1)
    for trk in tracks:
        x1, y1, x2, y2 = trk.bbox
        cor = COR_CLASSE.get(trk.class_hint, (180,180,180))
        lbl, cm = trk.votar()
        cv2.rectangle(out, (x1,y1), (x2,y2), cor, 3 if (trk.state=="confirmed" and lbl) else 1)
        hdr = f"#{trk.id} {trk.class_hint}" + (f" → {lbl} {cm*100:.0f}%" if lbl else "")
        (tw, th), _ = cv2.getTextSize(hdr, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
        cv2.rectangle(out, (x1,y1-th-6), (x1+tw+4,y1), cor, -1)
        cv2.putText(out, hdr, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255,255,255), 1)
        prog = int((x2-x1)*len(trk.buf)/VOTE_BUFFER)
        cv2.rectangle(out, (x1,y2+2), (x2,y2+6), (40,40,40), -1)
        cv2.rectangle(out, (x1,y2+2), (x1+prog,y2+6), cor, -1)

    pan = np.full((h,PW,3), (18,18,18), np.uint8)
    cv2.rectangle(pan, (0,0), (PW-1,h-1), (45,45,45), 1)
    def t(s, ln, cor=(190,190,190), sc=0.33):
        cv2.putText(pan, s, (5,14+ln*16), cv2.FONT_HERSHEY_SIMPLEX, sc, cor, 1)
    cor_missao = {"AGUARDANDO":(0,220,220), "AGUARDA_ENTREGA":(220,220,0),
                  "IND_RETIRADA":(100,220,100), "IND_ENTREGA":(100,220,100),
                  "LIVRE":(160,220,160), "PARADO_SEM":(50,50,220), "PARADO_PARE":(50,50,220),
                  "PARADO_OBST":(0,0,200), "RETIRANDO":(220,180,0),
                  "ENTREGANDO":(220,180,0), "DESLIGADO":(120,120,120)}.get(MISSAO.estado, (190,190,190))
    t(f"FPS:{fps:.0f} [{modo}]", 0, (255,255,255), 0.38)
    t(f"MISSAO: {MISSAO.estado}", 1, cor_missao, 0.36)
    t(f"Rota:{MISSAO.retirada or '-'}>{MISSAO.entrega or '-'}  fase:{MISSAO.fase[:4]}  "
      f"Entregas:{MISSAO.entregas}", 2, (200,200,120))
    t(f"Trk:{len(tracks)} Fichas:{len(PGOM_M.fichas)}", 3)
    if MISSAO.estado in ("RETIRANDO","ENTREGANDO"):
        t(f"  {MISSAO.tempo_no_estado():.1f}/{ESPERA_ENTREGA_S:.0f}s", 4, cor_missao, 0.40)
    elif MISSAO.parado_por_regra():
        t("  PARADO (regra absoluta)", 4, (50,50,220), 0.36)
    else:
        t(f"  mot={CMD['mot']}", 4, (100,200,100))
    t("── CONFIRMADOS ──", 6, (60,60,60)); ln = 7
    for trk in tracks[:5]:
        lbl, cm = trk.votar()
        if lbl:
            t(f"#{trk.id} {lbl} {cm*100:.0f}%", ln, COR_CLASSE.get(lbl,(180,180,180))); ln += 1
    t("── ÚLTIMO ──", 18, (60,60,60))
    t(f" {_ultimo['acao'] or '-'}", 19, COR_ACAO.get(_ultimo["acao"], (160,160,160)))

    # dois QRs fixos no canto: a sessão vale a missão inteira
    if MISSAO.qr_wifi is not None and MISSAO.qr_img is not None:
        q1, q2 = MISSAO.qr_wifi, MISSAO.qr_img
        qh, qw = q1.shape[:2]; gap = 14
        qx2, qy = w-qw-10, h-qh-20; qx1 = qx2-qw-gap
        if qx1 > 0 and qy > 0:
            cv2.rectangle(out, (qx1-6,qy-20), (qx2+qw+6,qy+qh+6), (255,255,255), -1)
            cv2.putText(out, "1. Wi-Fi",  (qx1,qy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)
            cv2.putText(out, "2. Painel", (qx2,qy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)
            out[qy:qy+qh, qx1:qx1+qw] = q1
            out[qy:qy+qh, qx2:qx2+qw] = q2

    vis = np.empty((h,w+PW,3), np.uint8)
    vis[:,:w] = out; vis[:,w:] = pan
    return vis


# ================================================================
#  [12] LOOP PRINCIPAL
# ================================================================

def abrir_camera(idx):
    """Abre a webcam externa via DirectShow (Windows). Falha alto: pegar a
    câmera errada em silêncio é pior do que não abrir."""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        print(f"[CAM] índice {idx} não respondeu — confira o cabo ou use --cam-idx N",
              flush=True)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[CAM] idx={idx} {cap.get(3):.0f}x{cap.get(4):.0f}@{cap.get(5):.0f}fps", flush=True)
    return cap


def main(debug_abc=False, auto=False, fast=False, usar_yolo=False, cam_idx=None):
    global _ser, CLASSES, NUM_CLASSES, CAM_FLIP, LEITOR, BT
    global VOTE_BUFFER, VOTE_MIN_DETS

    if fast:
        VOTE_BUFFER, VOTE_MIN_DETS = 4, 2     # --fast: ainda mais rápido, pra pista em movimento

    # classes.txt (se existir) só afeta a YOLO — a CNN é sempre CNN_CLASSES fixo.
    CLASSES = carregar_classes(); NUM_CLASSES = len(CLASSES)

    cnn = ood = verif = None
    if os.path.exists(CNN_MODEL):
        try:
            cnn = CNNClassifier(CNN_MODEL); ood = OODRejector(OOD_FILE)
            verif = VerificadorPlaca()
        except Exception as e:
            print(f"[CNN] indisponível: {e}", flush=True)

    yolo = yolo_coco = None; modo = "GEO"
    if usar_yolo and os.path.exists(YOLO_MODEL):
        try:
            yolo = YOLODetector(YOLO_MODEL, esperar_custom=True); modo = "GEO+YOLO"
        except Exception as e:
            print(f"[YOLO] custom indisponível: {e}", flush=True)
    if usar_yolo and os.path.exists(COCO_MODEL):
        try:
            yolo_coco = YOLODetector(COCO_MODEL, esperar_custom=False)
            modo = modo + "+COCO"
        except Exception as e:
            print(f"[YOLO] coco indisponível: {e}", flush=True)

    LEITOR = LeitorLetras(debug=debug_abc)
    BT = PainelLocal()
    tracker = ByteTrackLite()

    cap = abrir_camera(CAM_IDX if cam_idx is None else cam_idx)
    _ser = conectar_serial()

    if auto:
        MISSAO.retirada, MISSAO.entrega, MISSAO.fase = "A", "B", "retirada"
        MISSAO.set_estado("IND_RETIRADA"); seguir(_ser, com_buzzer=False)
    MISSAO.nova_sessao()
    print(f"[OK] detector={modo} · Q=sair SPACE=pausa 1/2/3=pontos "
          f"S=status D=desliga R=reset F=flip", flush=True)

    fps_t = time.monotonic(); fps_n = 0; fps = 0.0; fn = 0
    SKIP = 1 if fast else 2
    frame_e = None; tracks = []

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("[CAM] frame perdido", flush=True); continue
        if CAM_FLIP is not None:
            frame_raw = cv2.flip(frame_raw, CAM_FLIP)
        fn += 1
        tick_buzzer(_ser); ler_serial(_ser)

        if fn % SKIP == 0:
            frame_e = preprocessar(frame_raw)
            h, w = frame_e.shape[:2]
            dets = detectar_geometrico(frame_e)
            n_geo = len(dets)

            n_yolo_raw = n_yolo_ok = 0
            if yolo and fn % YOLO_EVERY == 0:
                rx0, ry0, rx1, ry1 = ROI_DIN.janela(h, w)
                sub = frame_e[ry0:ry1, rx0:rx1]
                if sub.size:
                    yolo_dets = yolo.detectar(sub)
                    n_yolo_raw = len(yolo_dets)
                    for d in yolo_dets:
                        x1, y1, x2, y2 = d["bbox"]
                        d["bbox"] = (x1+rx0, y1+ry0, x2+rx0, y2+ry0)
                        if geometria_concorda(frame_e, d):   # a geometria tem a palavra final
                            dets.append(d); n_yolo_ok += 1
                    if DEBUG_FUNIL and n_yolo_raw == 0:
                        print(f"[yolo] frame {fn}: nenhuma detecção (nada chegou até a CNN)", flush=True)
                    elif DEBUG_FUNIL and n_yolo_ok < n_yolo_raw:
                        print(f"[yolo] frame {fn}: {n_yolo_raw} bruto -> {n_yolo_ok} passou no [geo]", flush=True)

            if yolo_coco and fn % YOLO_EVERY == 0:
                rx0, ry0, rx1, ry1 = ROI_DIN.janela(h, w)
                sub = frame_e[ry0:ry1, rx0:rx1]
                if sub.size:
                    for d in yolo_coco.detectar(sub):
                        x1, y1, x2, y2 = d["bbox"]
                        d["bbox"] = (x1+rx0, y1+ry0, x2+rx0, y2+ry0)
                        dets.append(d)   # coco=True já pula geometria/CNN em _classificar
            if DEBUG_FUNIL and n_geo == 0 and n_yolo_raw == 0 and fn % YOLO_EVERY == 0:
                print(f"[funil] frame {fn}: nada detectado nem por geo nem por yolo "
                      f"(perdeu ANTES da classificação — checar ROI/exposição/distância)", flush=True)

            # delivery não passa por PGOM/CNN — geometria + topologia + votação
            entregas = [d for d in dets if d["class_name"] == "Delivery"]
            dets     = [d for d in dets if d["class_name"] != "Delivery"]
            dets = [d for d in dets
                    if (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]) >= AREA_MIN_EXEC]
            tracks = tracker.update(PGOM_M.update(dets), frame_e, cnn, ood, verif)
            for trk in tracks:
                if trk.state == "confirmed" or trk.age >= 2: ROI_DIN.registrar(trk.bbox)

            # ── VISÃO: constatação de fatos, nenhuma decisão ──
            percep = dict(pare=False, semaforo=None, obstaculo=False, ponto=None)
            for trk in tracks:
                # LOG 1 (proximidade): repete a cada nível de área cruzado —
                # é o proxy de "cada N cm de aproximação" sem sensor de distância.
                if trk.class_hint in ("Stop", "Semaforo"):
                    while (trk.nivel_prox < len(AREA_PROX_NIVEIS)
                           and trk.area >= AREA_PROX_NIVEIS[trk.nivel_prox]):
                        sinalizar_proximidade("PARE" if trk.class_hint == "Stop" else "SEM", _ser)
                        trk.nivel_prox += 1
                    # LOG 3 (chegada): separado da confirmação de cor/classe —
                    # só sobre distância. "Chegou perto, executa agora."
                    if trk.area >= AREA_CHEGADA_SINALEIRA and not trk.evt_chegada_enviado:
                        sinalizar_chegada("PARE" if trk.class_hint == "Stop" else "SEM", _ser)
                        trk.evt_chegada_enviado = True

                lbl, _ = trk.votar()
                if not lbl or trk.state != "confirmed" or trk.area < AREA_MIN_EXEC: continue
                if lbl == "Semaforo":
                    x1, y1, x2, y2 = trk.bbox
                    cor = estado_semaforo(frame_e[y1:y2, x1:x2])
                    # LOG 2 (confirmação): só depois de N_CONSECUTIVO frames com a
                    # MESMA cor seguida — não é maioria estatística, é sequência.
                    # A MISSÃO só enxerga a cor CONFIRMADA, nunca a leitura crua —
                    # é o que evita a missão reagir antes da hora.
                    cor_conf = trk.cor_consecutiva(cor)
                    if cor_conf:
                        percep["semaforo"] = cor_conf
                        if trk.evt_conf_enviado != cor_conf:
                            sinalizar_confirmacao("SEM", cor_conf, _ser)
                            trk.evt_conf_enviado = cor_conf
                elif lbl == "Stop":
                    # Mesma lógica: a MISSÃO só vê "pare" quando os 3 consecutivos
                    # bateram — e só UMA VEZ por track (stop_consumido), pra não
                    # reabrir a decisão de parar enquanto a mesma placa segue no
                    # campo de visão (ela nunca deixa de ser vista até o carro passar).
                    if trk.consecutivo() == "Stop" and not trk.stop_consumido:
                        percep["pare"] = True
                        percep["pare_trk"] = trk    # pra marcar consumido só quando a missão executar
                        if trk.evt_conf_enviado != "ok":
                            sinalizar_confirmacao("PARE", "ok", _ser)
                            trk.evt_conf_enviado = "ok"
                elif CLASS_TO_ACTION.get(lbl) == "OBSTACLE":
                    percep["obstaculo"] = True

            m0 = LEITOR.votar(entregas)
            alvo_atual = MISSAO.alvo()   # o Portenta não precisa saber que vimos B se o alvo é C
            if m0 is not None:
                bx1, by1, bx2, by2 = m0["bbox"]
                area_bbox = (bx2-bx1)*(by2-by1)
                e_alvo = (alvo_atual is not None and m0["ponto"] == alvo_atual)
                # só é "candidato à chegada" se a letra bater com o alvo da missão —
                # ver B procurando C não gera evento nenhum, nem proximidade.
                if e_alvo:
                    if area_bbox >= AREA_MIN_PROX_ENTREGA and not _ultimo.get("prox_enviada"):
                        sinalizar_proximidade("ENTREGA", _ser)
                        _ultimo["prox_enviada"] = True
                    if area_bbox >= AREA_MIN_CHEGADA:
                        percep["ponto"] = m0["ponto"]
                        if _ultimo.get("conf_enviada") != m0["ponto"]:
                            sinalizar_confirmacao("ENTREGA", m0["ponto"], _ser)
                            _ultimo["conf_enviada"] = m0["ponto"]
                if m0["ponto"] != _ultimo["ponto"]:
                    _ultimo["ponto"] = m0["ponto"]
                    _ultimo["prox_enviada"] = False; _ultimo["conf_enviada"] = None
                    print(f"[ABC] ponto '{m0['ponto']}' — área={area_bbox:.0f} "
                          f"({'perto' if area_bbox >= AREA_MIN_CHEGADA else f'longe, min={AREA_MIN_CHEGADA}'})"
                          f" alvo={MISSAO.alvo()}", flush=True)
            elif _ultimo["ponto"] is not None:
                _ultimo["ponto"] = None
                _ultimo["prox_enviada"] = False; _ultimo["conf_enviada"] = None

            # ── COMANDOS DO PAINEL ──
            for cmd, arg in BT.poll():
                if cmd == "R":
                    MISSAO.retirada = arg; MISSAO.fase = "retirada"
                    if MISSAO.estado == "DESLIGADO": MISSAO.set_estado("AGUARDANDO")
                    if MISSAO.estado in ("AGUARDANDO", "LIVRE"):
                        MISSAO.set_estado("IND_RETIRADA"); seguir(_ser)
                elif cmd == "E":
                    MISSAO.entrega = arg
                    if MISSAO.estado == "AGUARDA_ENTREGA":
                        MISSAO.fase = "entrega"
                        MISSAO.set_estado("IND_ENTREGA"); seguir(_ser)
                elif cmd == "DESLIGAR":
                    MISSAO.set_estado("DESLIGADO"); parar(_ser)

            # ── MISSÃO ──
            est = MISSAO.estado
            alvo = MISSAO.alvo()
            chegou = (alvo is not None and percep["ponto"] == alvo)

            def _chegada(ser):
                """Transição comum de chegada ao alvo da fase atual."""
                MISSAO.set_estado("RETIRANDO" if MISSAO.fase == "retirada" else "ENTREGANDO")
                parar(ser); buzinar(ser, BUZ_ENTREGA_S)

            if est == "AGUARDANDO":
                parar(_ser)

            elif est in Missao.RODANDO:
                motivo = MISSAO.regra_absoluta(percep)          # prioridade máxima
                if motivo:
                    MISSAO.retorno = est
                    MISSAO.set_estado(motivo); parar(_ser)
                elif chegou:
                    _chegada(_ser)

            elif MISSAO.parado_por_regra():
                # a chegada vale mesmo parado: o ponto costuma ficar junto de um cruzamento
                if chegou:
                    _chegada(_ser)
                elif (not percep["obstaculo"] and semaforo_libera(est, percep)
                        and pare_satisfeito(percep)):
                    MISSAO.set_estado(MISSAO.retorno); seguir(_ser)

            elif est == "RETIRANDO":
                if MISSAO.tempo_no_estado() >= ESPERA_ENTREGA_S:
                    MISSAO.fase = "entrega"
                    if not MISSAO.entrega:
                        MISSAO.set_estado("AGUARDA_ENTREGA")   # espera a escolha aqui mesmo
                    elif MISSAO.entrega == MISSAO.retirada:
                        _chegada(_ser)
                    else:
                        MISSAO.set_estado("IND_ENTREGA"); seguir(_ser)

            elif est == "ENTREGANDO":
                if MISSAO.tempo_no_estado() >= ESPERA_ENTREGA_S:
                    MISSAO.entregas += 1
                    MISSAO.retirada = MISSAO.entrega = None
                    MISSAO.fase = "retirada"; MISSAO.retorno = "LIVRE"
                    MISSAO.set_estado("LIVRE"); seguir(_ser)   # segue rodando, não para

            # AGUARDA_ENTREGA e DESLIGADO só saem por comando externo

        disp = frame_e if frame_e is not None else \
               cv2.resize(frame_raw, (IMG_W, int(frame_raw.shape[0]*IMG_W/frame_raw.shape[1])))
        cv2.imshow("Carro Autônomo v5.2", desenhar(disp, tracks, fps, modo))

        fps_n += 1
        if time.monotonic()-fps_t >= 1.0:
            fps = fps_n; fps_n = 0; fps_t = time.monotonic()

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord(' '): cv2.waitKey(0)
        elif k == ord('f'):
            ciclo = [None, 1, 0, -1]; CAM_FLIP = ciclo[(ciclo.index(CAM_FLIP)+1) % 4]
        elif k == ord('s'): print(MISSAO.status(), flush=True)
        elif k == ord('r'):
            MISSAO.retirada = MISSAO.entrega = None
            MISSAO.fase = "retirada"; MISSAO.retorno = "LIVRE"
            MISSAO.set_estado("AGUARDANDO"); parar(_ser)
        elif k == ord('d'):
            if MISSAO.estado == "DESLIGADO": MISSAO.set_estado("AGUARDANDO")
            else: MISSAO.set_estado("DESLIGADO"); parar(_ser)
        elif k in (ord('1'), ord('2'), ord('3')):
            p = {ord('1'):"A", ord('2'):"B", ord('3'):"C"}[k]
            if MISSAO.retirada is None:                        # 1ª tecla = retirada
                MISSAO.retirada = p; MISSAO.fase = "retirada"
                if MISSAO.estado in ("AGUARDANDO","LIVRE"):
                    MISSAO.set_estado("IND_RETIRADA"); seguir(_ser)
            else:                                              # 2ª tecla = entrega
                MISSAO.entrega = p
                if MISSAO.estado == "AGUARDA_ENTREGA":
                    MISSAO.fase = "entrega"
                    MISSAO.set_estado("IND_ENTREGA"); seguir(_ser)

    CMD.update(mot=0, srv=127, buz=0, led=0, brk=1, dir=0); enviar(CMD, _ser)
    BT.fechar()
    if _ser: _ser.close()
    cap.release(); cv2.destroyAllWindows()
    print("[OK] encerrado.", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--gerar-abc" in args:
        gerar_placas_abc()
    else:
        cam_idx = None
        if "--cam-idx" in args:
            try: cam_idx = int(args[args.index("--cam-idx")+1])
            except (IndexError, ValueError):
                print("[ERRO] use: --cam-idx N"); sys.exit(1)
        main(debug_abc="--debug-abc" in args, auto="--auto" in args,
             fast="--fast" in args, usar_yolo="--yolo" in args, cam_idx=cam_idx)