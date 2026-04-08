import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys

# ================================================================
#  CONFIGURAÇÃO
# ================================================================

PROP = 2          # divisor de resolução (2 = metade do tamanho)

# ── Fonte de vídeo — escolha UMA das duas linhas ────────────────
cap = cv2.VideoCapture("./pista_01.mov")   # arquivo de vídeo
# cap = cv2.VideoCapture(0)               # câmera USB conectada

# ── Serial Arduino ───────────────────────────────────────────────
SERIAL_PORT = "COM3"   # ajustar para sua porta
BAUD        = 115200

# ── PID ─────────────────────────────────────────────────────────
KP, KI, KD = 0.55, 0.005, 0.12

# ── Saídas ──────────────────────────────────────────────────────
VEL_NORMAL  = 62
VEL_DEVAGAR = 35
VEL_PARADO  = 0

# ── Threshold de pista ──────────────────────────────────────────
THRESH_VALOR = 95     # limiar de binarização (ajustar conforme pista)


# ================================================================
#  ESTADO GLOBAL
# ================================================================

CMD = {
    "mot": 0,     # velocidade do motor (0–100 ou PWM equivalente)
    "srv": 0,     # direção do servo (-40 a +40 graus)
    "buz": 0,     # buzzer (0=off, 1=on)
    "led": 0,     # LED status (0=off, 1=on)

    # NOVAS VARIÁVEIS IMPORTANTES ↓↓↓
    "mode": 0,    # modo de operação do carro
    "brk": 0,     # freio ativo (0=livre, 1=freando)
    "dir": 0,     # direção lógica (0=neutro, 1=esq, 2=dir, 3=frente)
    "spd": 0,     # perfil de velocidade (0=parado, 1=lento, 2=normal, 3=rápido)

    # OPCIONAL (para debug/telemetria)
    "err": 0      # erro lateral (escala reduzida para int)
}
_pid   = {"i": 0.0, "e_ant": 0.0, "t_ant": time.monotonic()}
_hist  = []   # histórico do centro para média móvel

# ================================================================
# ESTADO DE NAVEGAÇÃO / PLACAS / MARCADOR NO CHÃO
# ================================================================

NAV = {
    "acao_pendente": None,      # None, LEFT, RIGHT, STRAIGHT, STOP, YIELD, DELIVERY
    "placa_confirmada": False,
    "frames_placa": 0,
    "ultimo_tipo_placa": None,

    "modo": "SEGUIR_PISTA",     # SEGUIR_PISTA, EXEC_STOP, EXEC_YIELD, EXEC_LEFT, EXEC_RIGHT, EXEC_STRAIGHT, EXEC_DELIVERY
    "t_inicio_modo": time.monotonic(),

    "cooldown_placa": 0,        # evita reconhecer a mesma placa várias vezes
    "cooldown_marcador": 0,     # evita disparar 2x no mesmo marcador
}

frame_id = 0
# ================================================================
#  VERIFICAÇÃO DA FONTE DE VÍDEO
# ================================================================

if not cap.isOpened():
    print("Erro — vídeo/câmera não encontrado!")
    sys.exit()


# ================================================================
#  SERIAL (opcional — não trava se Arduino não estiver conectado)
# ================================================================

def conectar_serial():
    kws = ["arduino", "ch340", "cp210", "uart"]
    for p in serial.tools.list_ports.comports():
        if any(k in (p.description or "").lower() for k in kws):
            print(f"[SER] Arduino em {p.device}")
            break
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0, write_timeout=0)
        time.sleep(2)
        print(f"[SER] Conectado: {SERIAL_PORT}")
        return ser
    except Exception as e:
        print(f"[SER] Sem Arduino ({e}) — modo visualização")
        return None

ser = conectar_serial()


# ================================================================
#  PID
# ================================================================

def pid_calc(erro: float) -> float:
    agora = time.monotonic()
    dt    = max(agora - _pid["t_ant"], 1e-4)
    _pid["i"]     = float(np.clip(_pid["i"] + erro * dt, -1.0, 1.0))
    deriv         = (erro - _pid["e_ant"]) / dt
    _pid["e_ant"] = erro
    _pid["t_ant"] = agora
    return float(np.clip(KP*erro + KI*_pid["i"] + KD*deriv, -1.0, 1.0))


# ================================================================
#  PIPELINE DE PISTA
#  Baseado na lógica original do usuário:
#    BGR → Gray → medianBlur → threshold → detecção de linhas
# ================================================================

def processar_pista(frame):
    """
    Retorna:
      img_gray     — frame em cinza
      filtrada_vis — visualização do processamento
      erro         — desvio lateral normalizado (-1.0 a +1.0)
      pista_ok     — bool: pista detectada
      cp_suave     — centro suavizado da pista
    """
    global _hist

    h, w = frame.shape[:2]
    cx_ref = w // 2

    # 1) Escala de cinza
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Blur
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 3) Threshold automático
    _, binaria = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Se a pista estiver clara no fundo escuro, talvez precise inverter:
    # binaria = cv2.bitwise_not(binaria)

    # 4) Morfologia
    kernel = np.ones((3, 3), np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    # 5) ROI trapezoidal
    roi = np.zeros_like(binaria)
    pts = np.array([[
        (int(w * 0.10), h),
        (int(w * 0.40), int(h * 0.60)),
        (int(w * 0.60), int(h * 0.60)),
        (int(w * 0.90), h)
    ]], dtype=np.int32)
    cv2.fillPoly(roi, pts, 255)
    roi_mask = cv2.bitwise_and(binaria, roi)

    # 6) Bordas
    bordas = cv2.Canny(roi_mask, 50, 150)

    # 7) Hough
    linhas = cv2.HoughLinesP(
        bordas,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=30
    )

    esq_params = []
    dir_params = []

    if linhas is not None:
        for seg in linhas:
            x1, y1, x2, y2 = seg[0]

            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue

            comprimento = np.hypot(dx, dy)
            if comprimento < 25:
                continue

            inc = dy / dx

            # Rejeita linhas quase horizontais ou exageradas
            if abs(inc) < 0.3 or abs(inc) > 3.5:
                continue

            # Ajusta reta x = (y - b)/m  -> via polyfit y(x)
            m, b = np.polyfit([x1, x2], [y1, y2], 1)

            if inc < 0:
                esq_params.append((m, b))
            else:
                dir_params.append((m, b))

    pista_ok = False
    cp = cx_ref
    y_ref = int(h * 0.85)

    x_esq = None
    x_dir = None

    if esq_params:
        m_esq = np.mean([p[0] for p in esq_params])
        b_esq = np.mean([p[1] for p in esq_params])
        if abs(m_esq) > 1e-6:
            x_esq = int((y_ref - b_esq) / m_esq)

    if dir_params:
        m_dir = np.mean([p[0] for p in dir_params])
        b_dir = np.mean([p[1] for p in dir_params])
        if abs(m_dir) > 1e-6:
            x_dir = int((y_ref - b_dir) / m_dir)

    if x_esq is not None and x_dir is not None:
        cp = (x_esq + x_dir) // 2
        pista_ok = True
    elif x_esq is not None:
        cp = x_esq + w // 4
        pista_ok = True
    elif x_dir is not None:
        cp = x_dir - w // 4
        pista_ok = True

    cp = int(np.clip(cp, 0, w - 1))

    # 8) Suavização
    _hist.append(cp)
    if len(_hist) > 5:
        _hist.pop(0)

    cp_suave = int(np.mean(_hist))
    erro = float(np.clip((cp_suave - cx_ref) / cx_ref, -1.0, 1.0))

    # 9) Visualização
    filtrada_vis = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)

    cv2.line(filtrada_vis, (cx_ref, int(h * 0.60)), (cx_ref, h), (0, 0, 255), 1)
    cv2.line(filtrada_vis, (cp_suave, int(h * 0.60)), (cp_suave, h), (0, 255, 0), 2)
    cv2.line(filtrada_vis, (0, y_ref), (w, y_ref), (255, 0, 0), 1)

    if x_esq is not None:
        cv2.circle(filtrada_vis, (x_esq, y_ref), 4, (255, 255, 0), -1)

    if x_dir is not None:
        cv2.circle(filtrada_vis, (x_dir, y_ref), 4, (0, 255, 255), -1)

    return img_gray, filtrada_vis, erro, pista_ok, cp_suave

# ================================================================
#  DECISÃO E COMANDOPROCURA FAIXA ESCURA HORIZONTAL
# ================================================================
def detectar_marcador_chao(frame):
    """
    Detecta um marcador no chão para disparar manobra.
    Estratégia:
    - usa faixa inferior da imagem
    - converte para cinza
    - threshold invertido para pegar regiões escuras
    - procura um contorno largo e baixo (faixa transversal)

    Retorna:
        marcador_ok: bool
        vis: imagem BGR para visualização
    """
    h, w = frame.shape[:2]

    # ROI do chão: faixa inferior da imagem
    y1 = int(h * 0.72)
    y2 = int(h * 0.95)
    roi = frame[y1:y2, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold invertido: escuro vira branco
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # limpeza morfológica
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marcador_ok = False
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / float(max(hh, 1))
        cobertura = ww / float(w)

        # marcador ideal:
        # - largo
        # - relativamente baixo
        # - ocupa boa parte da largura da pista
        if aspect > 3.0 and cobertura > 0.45 and hh > 8:
            marcador_ok = True
            cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.putText(vis, "MARCADOR", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            break

    return marcador_ok, vis

def registrar_placa(tipo_placa):
    """
    tipo_placa: None, STOP, YIELD, LEFT, RIGHT, STRAIGHT, DELIVERY
    Confirma a placa em alguns frames e grava a ação pendente.
    """
    if tipo_placa is None:
        NAV["frames_placa"] = 0
        NAV["placa_confirmada"] = False
        NAV["ultimo_tipo_placa"] = None
        return

    if NAV["cooldown_placa"] > 0:
        return

    if NAV["ultimo_tipo_placa"] == tipo_placa:
        NAV["frames_placa"] += 1
    else:
        NAV["ultimo_tipo_placa"] = tipo_placa
        NAV["frames_placa"] = 1

    # confirma após 3 leituras
    if NAV["frames_placa"] >= 3:
        NAV["placa_confirmada"] = True
        NAV["acao_pendente"] = tipo_placa
        NAV["cooldown_placa"] = 20

def disparar_acao_no_marcador(marcador_ok):
    """
    Se houver ação pendente e o marcador for detectado,
    inicia a execução da manobra.
    """
    if NAV["cooldown_marcador"] > 0:
        return

    if not marcador_ok:
        return

    if NAV["acao_pendente"] is None:
        return

    acao = NAV["acao_pendente"]
    NAV["t_inicio_modo"] = time.monotonic()
    NAV["cooldown_marcador"] = 25

    if acao == "STOP":
        NAV["modo"] = "EXEC_STOP"
    elif acao == "YIELD":
        NAV["modo"] = "EXEC_YIELD"
    elif acao == "LEFT":
        NAV["modo"] = "EXEC_LEFT"
    elif acao == "RIGHT":
        NAV["modo"] = "EXEC_RIGHT"
    elif acao == "STRAIGHT":
        NAV["modo"] = "EXEC_STRAIGHT"
    elif acao == "DELIVERY":
        NAV["modo"] = "EXEC_DELIVERY"

    # limpa intenção após disparar
    NAV["acao_pendente"] = None
    NAV["placa_confirmada"] = False
    NAV["frames_placa"] = 0
    NAV["ultimo_tipo_placa"] = None

def controle_modo():
    """
    Executa manobras temporizadas.
    Retorna True se o modo especial assumiu o controle.
    """
    agora = time.monotonic()
    dt = agora - NAV["t_inicio_modo"]
    modo = NAV["modo"]

    if modo == "SEGUIR_PISTA":
        return False

    # ------------------------------------------------------------
    # STOP
    # ------------------------------------------------------------
    if modo == "EXEC_STOP":
        CMD["mot"] = 0
        CMD["srv"] = 0
        CMD["buz"] = 0
        CMD["led"] = 1

        if dt > 2.0:
            NAV["modo"] = "SEGUIR_PISTA"
            CMD["led"] = 0
        return True

    # ------------------------------------------------------------
    # YIELD = reduz por um tempo e segue
    # ------------------------------------------------------------
    if modo == "EXEC_YIELD":
        CMD["mot"] = VEL_DEVAGAR
        CMD["srv"] = 0
        CMD["buz"] = 0
        CMD["led"] = 0

        if dt > 1.0:
            NAV["modo"] = "SEGUIR_PISTA"
        return True

    # ------------------------------------------------------------
    # LEFT
    # ------------------------------------------------------------
    if modo == "EXEC_LEFT":
        # 1ª fase: pequena aproximação reta
        if dt < 0.35:
            CMD["mot"] = 40
            CMD["srv"] = 0
        # 2ª fase: curva
        elif dt < 1.15:
            CMD["mot"] = 38
            CMD["srv"] = -32
        # 3ª fase: estabilização
        elif dt < 1.45:
            CMD["mot"] = 35
            CMD["srv"] = 0
        else:
            NAV["modo"] = "SEGUIR_PISTA"
        CMD["buz"] = 0
        CMD["led"] = 0
        return True

    # ------------------------------------------------------------
    # RIGHT
    # ------------------------------------------------------------
    if modo == "EXEC_RIGHT":
        if dt < 0.35:
            CMD["mot"] = 40
            CMD["srv"] = 0
        elif dt < 1.15:
            CMD["mot"] = 38
            CMD["srv"] = 32
        elif dt < 1.45:
            CMD["mot"] = 35
            CMD["srv"] = 0
        else:
            NAV["modo"] = "SEGUIR_PISTA"
        CMD["buz"] = 0
        CMD["led"] = 0
        return True

    # ------------------------------------------------------------
    # STRAIGHT
    # ------------------------------------------------------------
    if modo == "EXEC_STRAIGHT":
        CMD["mot"] = 45
        CMD["srv"] = 0
        CMD["buz"] = 0
        CMD["led"] = 0

        if dt > 0.9:
            NAV["modo"] = "SEGUIR_PISTA"
        return True

    # ------------------------------------------------------------
    # DELIVERY
    # ------------------------------------------------------------
    if modo == "EXEC_DELIVERY":
        CMD["mot"] = 0
        CMD["srv"] = 0
        CMD["buz"] = 1
        CMD["led"] = 1

        if dt > 3.0:
            CMD["buz"] = 0
            CMD["led"] = 0
            NAV["modo"] = "SEGUIR_PISTA"
        return True

    return False

def atualizar_cooldowns():
    if NAV["cooldown_placa"] > 0:
        NAV["cooldown_placa"] -= 1

    if NAV["cooldown_marcador"] > 0:
        NAV["cooldown_marcador"] -= 1

def detectar_placa(frame):
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        return "LEFT"
    elif key == ord('d'):
        return "RIGHT"
    elif key == ord('w'):
        return "STRAIGHT"
    elif key == ord('s'):
        return "STOP"
    elif key == ord('y'):
        return "YIELD"
    elif key == ord('e'):
        return "DELIVERY"
    return None
# ================================================================
#  DECISÃO E COMANDO
# ================================================================

def decidir(pista_ok, erro):
    if pista_ok:
        angulo = int(np.clip(pid_calc(erro) * 40, -40, 40))
        CMD["mot"] = VEL_NORMAL
        CMD["srv"] = angulo
        CMD["buz"] = 0
        CMD["led"] = 0
    else:
        CMD["mot"] = VEL_PARADO
        CMD["srv"] = 0
        CMD["buz"] = 0
        CMD["led"] = 1   # LED aceso = pista perdida


def enviar(ser):
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
#  LOOP PRINCIPAL
# ================================================================

print("Pressione Q para sair\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo ou erro de leitura.")
        break

    frame_id += 1
    atualizar_cooldowns()

    # ── Redimensionar conforme proporção ─────────────────────────
    altura  = round(frame.shape[0] / PROP)
    largura = round(frame.shape[1] / PROP)
    frame   = cv2.resize(frame, (largura, altura))

    # ── Processar pista ──────────────────────────────────────────
    img_gray, filtrada_vis, erro, pista_ok, centro = processar_pista(frame)

    # ── Detectar placa a cada 3 frames ───────────────────────────
    tipo_placa = None
    if frame_id % 3 == 0:
        tipo_placa = detectar_placa(frame)
        registrar_placa(tipo_placa)

    # ── Detectar marcador no chão ────────────────────────────────
    marcador_ok, marcador_vis = detectar_marcador_chao(frame)

    # ── Se houver ação pendente, marcador dispara manobra ───────
    disparar_acao_no_marcador(marcador_ok)

    # ── Primeiro: controle de manobra especial ───────────────────
    ocupado = controle_modo()

    # ── Se não estiver executando manobra, segue pista ───────────
    if not ocupado:
        decidir(pista_ok, erro)

    # ── Envio serial ─────────────────────────────────────────────
    enviar(ser)

    # ── Visualização ─────────────────────────────────────────────
    img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # status texto
    status1 = f"mot:{CMD['mot']} srv:{CMD['srv']:+d} err:{erro:+.2f}"
    status2 = f"placa:{NAV['acao_pendente']} modo:{NAV['modo']}"
    status3 = f"marcador:{marcador_ok}"

    cor = (0, 220, 50) if pista_ok else (0, 40, 255)
    cv2.putText(frame, status1, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    cv2.putText(frame, status2, (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 2)

    cv2.putText(frame, status3, (8, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 2)

    # opcional: mostrar marcador
    marcador_vis = cv2.resize(marcador_vis, (frame.shape[1], frame.shape[0]))

    img_concat = cv2.hconcat([frame, img_gray_bgr, filtrada_vis])
    cv2.imshow("Visão do Carro", img_concat)
    cv2.imshow("Marcador no Chao", marcador_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ================================================================
#  ENCERRAMENTO
# ================================================================

CMD.update({"mot": 0, "srv": 0, "buz": 0, "led": 0})
enviar(ser)

if ser:
    ser.close()

cap.release()
cv2.destroyAllWindows()