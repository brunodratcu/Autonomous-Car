# 171 Garage — Carro Autônomo
### Challenge 2026 · Mercedes-Benz · FIAP

Veículo autônomo em miniatura com percepção por webcam, decisão de missão e atuação via serial em um Arduino Portenta H7. O carro circula em velocidade de cruzeiro, para **apenas** diante de placa PARE, semáforo vermelho ou obstáculo, e executa entregas entre os pontos **A**, **B** e **C** conforme o destino solicitado por um app externo.

---

## 1. Arquitetura em três camadas

A separação entre as camadas é estrita: cada uma responde a uma pergunta diferente e não invade a responsabilidade da outra.

```
VISÃO     → "o que existe?"           percepção pura, sem decisão
MISSÃO    → "o que isso significa?"   regras absolutas + FSM de entrega
CONTROLE  → "o que o motor faz?"      parar() / seguir() / buzinar()
```

### Pipeline de percepção

```
Webcam (CAP_DSHOW no Windows / CAP_V4L2 no Linux + MJPG)
   ↓ Grayscale                    ← a câmera da pista entrega imagem
   ↓ CLAHE                          sem cor; toda lógica é monocromática
   ↓ Gaussian Blur
   ↓ Adaptive Threshold (3 máscaras: escura, clara, Otsu)
   ↓ Morfologia (open + close)
   ↓ Contornos → ApproxPolyDP
   ↓ Filtro geométrico ──► só 2 alvos: octógono (PARE) e semáforo
   ↓ Peneira de ROI (7 testes: foco, brilho, bordas, textura, símbolo…)
   ↓ ROI Dinâmico (busca onde as placas costumam aparecer;
   │               varredura completa a cada 15 frames)
   ↓ PGOM — ficha de evidências por objeto, promoção só com ≥85%
   ↓ CNN 64×64 INT8 ──► confirmadora, não detectora
   ↓ ByteTrack-lite (associação entre frames)
   ↓ Votação temporal (consenso de 60%)
   ↓
   ├──► percep = {pare, semaforo, obstaculo, ponto}
   │
   ArUco (paralelo, sem CNN) ──► ponto A/B/C
```

### Por que dois estágios de classificação

O softmax de uma CNN **não sabe rejeitar o desconhecido** — ele força qualquer entrada para dentro de uma classe conhecida. Por isso a arquitetura é obrigatoriamente:

| Estágio | Pergunta | Quem responde |
|---|---|---|
| 1 | "isso é mesmo uma placa?" | Geometria + PGOM + `VerificadorPlaca` |
| 2 | "qual placa é?" | CNN (só recebe candidatos com ≥85% de evidência) |

Sem esse portão, a CNN gera falsos positivos persistentes.

---

## 2. Detecção do PARE — evidências geométricas

O PGOM (*Persistent Geometric Object Manager*) mantém uma **ficha de evidências** por objeto rastreado. A promoção à CNN exige score total ≥ 0.85.

| Evidência | Peso | O que mede |
|---|---|---|
| `forma` | 0.18 | nº de vértices + circularidade |
| `persist` | 0.18 | apareceu em 5–8 frames consecutivos |
| `simbolo` | 0.12 | há conteúdo interno (texto/desenho) no miolo |
| `estab` | 0.12 | centro e área variam de forma suave |
| **`persp`** | **0.12** | **os vértices cabem num octógono real sob perspectiva** |
| **`simet`** | **0.10** | **a placa é simétrica ao espelhar horizontalmente** |
| `aspecto` | 0.08 | proporção largura/altura |
| `convex` | 0.05 | área do contorno vs. do fecho convexo |
| `fingerprint` | 0.05 | assinatura geométrica estável entre frames |

**Perspectiva** ajusta uma homografia entre os vértices detectados e um octógono-molde e mede o erro de reprojeção. Um octógono real pontua 1.00 mesmo bastante inclinado; ruído aleatório de 8 vértices fica em ~0.60.

**Simetria** compara o recorte com seu espelho horizontal: 0.95 para placa real, 0.25 para sombra ou objeto assimétrico.

---

## 3. Semáforo — leitura sem cor

A câmera da pista entrega imagem monocromática (saturação ≈ 0), então **qualquer lógica em HSV falha silenciosamente**. A leitura é geométrica e estatística:

1. Localiza retângulo vertical (aspecto 0.25–0.62, altura > 50px)
2. Confirma 2–3 círculos empilhados e alinhados (variação horizontal < 12px)
3. Divide em três terços e soma a intensidade de pixels de cada um
4. O terço mais brilhante indica o farol aceso:

```
terço superior  → VERMELHO → parar
terço do meio   → AMARELO
terço inferior  → VERDE    → seguir
```

Exige contraste mínimo de 8 níveis entre o primeiro e o segundo terço — abaixo disso retorna `None` (indeterminado) em vez de chutar.

---

## 4. Pontos A/B/C — ArUco, sem CNN

Cada ponto de entrega recebe um marcador ArUco impresso (`DICT_4X4_50`):

| Marcador | ID | Ponto |
|---|---|---|
| `ponto_A_id0.png` | 0 | A |
| `ponto_B_id1.png` | 1 | B |
| `ponto_C_id2.png` | 2 | C |

Detecção determinística, robusta a rotação, escala e perspectiva, com verificação de erro embutida no próprio código do marcador. Custo computacional desprezível e zero treino.

**Gerar os marcadores:**
```bash
python carro-autonomo.py --gerar-abc
```

**Impressão e fixação:**
- Imprima em ~10×10 cm em papel fosco (papel brilhante reflete e atrapalha)
- **Não corte a margem branca** — ela faz parte do marcador
- Fixe perpendicular à pista, na altura da câmera, num trecho onde o carro passe de frente

---

## 5. Máquina de estados da missão

### Regras absolutas (prioridade máxima, sempre vencem a rota)

```
obstáculo         → PARAR      ┐
semáforo vermelho → PARAR      ├─ ordem de severidade
placa PARE        → PARAR      ┘
```

### Regras de missão (marcador visto)

```
A:  destino = A ? entregar : continuar
B:  destino = B ? entregar : continuar
C:  destino = C ? entregar : continuar
```

### Ciclo completo

```
AGUARDANDO
    ↓
ESPERA_QR ─────── 7 segundos parado
    ↓
SELECIONANDO ──── QR na tela → app externo publica o destino (MQTT)
    ↓
RODANDO ───────── cruzeiro (mot=90), respeitando as regras absolutas
    │
    ├─► PARADO_SEM  ──[verde]────────────► RODANDO
    ├─► PARADO_PARE ──[2.5s + via livre]─► RODANDO
    ├─► PARADO_OBST ──[via livre]────────► RODANDO
    │
    ↓ (marcador do destino avistado)
ENTREGANDO ────── buzzer longo (1.2s) + 7 segundos parado
    ↓
ESPERA_QR ─────── pronto para o próximo pedido
```

O **buzzer curto (0.4s) toca em toda partida**, sinalizando a saída do veículo. O acionamento é não-bloqueante: `tick_buzzer()` desliga o buzzer no tempo certo sem prender o loop principal.

---

## 6. Comunicação com o sistema externo

Toda a camada de rede é **best-effort**: se o broker cair ou o endpoint não responder, o carro continua operando normalmente.

| Função | Protocolo | Config no topo do arquivo |
|---|---|---|
| Receber destino | MQTT (subscribe) | `MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC_DEST` |
| Enviar telemetria | REST POST (JSON) | `TELEMETRY_URL` |
| Seleção do destino | QR code na tela | `QR_BASE_URL`, `CAR_ID` |

**Payload esperado no tópico MQTT** (aceita as duas formas):
```json
{"destino": "A"}
```
```
A
```

**Telemetria enviada** (POST JSON) em cada transição relevante:
```json
{"carro":"171","destino":"B","evento":"entregue","timestamp":1786227854.85}
```

Eventos emitidos: `qr_exibido`, `pedido_recebido`, `pedido_manual`, `parado_sem`, `parado_pare`, `parado_obst`, `retomou_verde`, `retomou_pare`, `retomou_obstaculo`, `chegou_no_ponto`, `entregue`.

> Os valores atuais em `MQTT_BROKER` e `TELEMETRY_URL` são **placeholders genéricos**. Substitua pelos endereços reais do dashboard antes da apresentação.

### Degradação sem rede

| Falta | Comportamento |
|---|---|
| `paho-mqtt` não instalado ou broker fora | destino pelas teclas `1`/`2`/`3` |
| `qrcode` não instalado | cartão com a URL legível no lugar do QR |
| `TELEMETRY_URL` inacessível | payload logado no terminal, missão segue |
| Arduino desconectado | modo simulação, comandos impressos no terminal |

---

## 7. Requisitos e instalação

- **Python 3.11 ou 3.12** (evite o 3.13 — quebra o LabelImg; usamos anotador próprio)
- Webcam USB (índice 1 = câmera externa; 0 costuma ser a embutida)
- Arduino Portenta H7 + HAT Carrier (opcional para teste)

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash no Windows
# ou: .venv\Scripts\activate.bat     # CMD

pip install opencv-contrib-python numpy pyserial onnxruntime tensorflow
pip install paho-mqtt qrcode
pip install matplotlib seaborn scikit-learn ultralytics   # só para treino
```

> **`opencv-contrib-python` é obrigatório** — o módulo ArUco não vem no `opencv-python` comum. Se já tiver o pacote simples instalado, desinstale antes: `pip uninstall opencv-python`.

---

## 8. Estrutura de arquivos

```
autonomous-car/
├── carro-autonomo.py           ← runtime principal (visão + missão + controle)
├── carro_portenta_curto.ino    ← firmware do Portenta H7
│
├── TRAIN_SIGN_CNN.py           ← treina a CNN classificadora + export INT8
├── EXPORT_COCO.py              ← exporta YOLOv8n COCO para ONNX (auxiliar)
├── EXTRACT_FRAMES.py           ← extrai frames do vídeo para anotação
├── ANNOTATE.py                 ← anotador de bounding boxes em OpenCV
├── PREPARE_YOLO_DATASET.py     ← converte dataset de classificação → YOLO
├── TRAIN_YOLO.py               ← treina YOLO customizado
│
├── dataset_final/              ← imagens de treino (PARE + semáforo)
├── marcadores_abc/             ← PNGs dos ArUco (gerado por --gerar-abc)
├── models/
│   ├── sign_classifier.tflite  ← CNN INT8
│   ├── ood_thresholds.json     ← limiares de rejeição por classe
│   └── sign_detector.onnx      ← YOLO opcional (flag --yolo)
│
├── videoplayback.mp4           ← vídeo de simulação
└── pre_config.json             ← preprocessing calibrado (gerado por --cal)
```

---

## 9. Passo a passo — do zero até rodar

**Passo 1 — Gerar e imprimir os marcadores A/B/C**
```bash
python carro-autonomo.py --gerar-abc
```
Imprima os três PNGs (~10 cm) e fixe nos pontos da pista.

**Passo 2 — Treinar a CNN classificadora**
```bash
python TRAIN_SIGN_CNN.py --dataset ./dataset_final --int8
```
MobileNetV3Small, 64×64, 35 épocas com EarlyStopping. Leva de 10 a 25 min em CPU. Gera `models/sign_classifier.tflite` e `models/ood_thresholds.json`.

**Passo 3 — Calibrar o preprocessing na pista real**
```bash
python carro-autonomo.py --cal --cam
```
Ajuste os sliders até as placas aparecerem limpas na imagem de threshold. `S` salva em `pre_config.json`, `Q` sai.

**Passo 4 — Ajustar os thresholds de rejeição**

Abra `models/ood_thresholds.json`. Valores autocalibrados sobem para 0.85–0.92, o que **raramente é atingido em vídeo real** — foram medidos em imagens limpas de validação. Reduza manualmente para **0.45–0.50**.

**Passo 5 — Configurar a comunicação**

No topo de `carro-autonomo.py`, substitua os placeholders por `MQTT_BROKER`, `MQTT_TOPIC_DEST`, `TELEMETRY_URL` e `QR_BASE_URL` reais.

**Passo 6 — Rodar**
```bash
python carro-autonomo.py --cam --fast          # pista, webcam externa
python carro-autonomo.py                       # vídeo de simulação
python carro-autonomo.py --cam --auto --fast   # pula a espera do QR (demo)
```

---

## 10. Flags e controles

### Flags de linha de comando

| Flag | Efeito |
|---|---|
| `--cam` | usa webcam ao vivo em vez do vídeo |
| `--cam-idx N` | força o índice da câmera (padrão: 1) |
| `--fast` | processa todo frame + votação rápida (6 frames, consenso 4/6) |
| `--auto` | inicia direto em RODANDO com destino A, sem esperar o QR |
| `--debug` | exibe a imagem de threshold abaixo do vídeo |
| `--yolo` | ativa o YOLO auxiliar (exige `models/sign_detector.onnx`) |
| `--canny` | adiciona Canny à segmentação ⚠️ gera falsos positivos nas bordas da janela |
| `--cal` | abre o calibrador de preprocessing |
| `--gerar-abc` | gera os PNGs dos marcadores e encerra |

### Teclas durante a execução

| Tecla | Ação |
|---|---|
| `Q` | encerra |
| `Espaço` | pausa/despausa |
| `1` `2` `3` | define destino A/B/C — **fallback** quando o MQTT está fora |
| `R` | reseta a missão e volta a aguardar pedido |
| `F` | cicla o flip da câmera (None → horizontal → vertical → 180°) |

### Painel lateral

Mostra em tempo real: FPS e modo de detecção, estado da missão, destino ativo, total de entregas, contagem de tracks e fichas do PGOM, cronômetro dos estados temporizados, classificações confirmadas e buffer de votação por track. Durante `SELECIONANDO`, o QR aparece sobreposto ao centro da imagem.

---

## 11. Protocolo serial (Python ↔ Portenta H7)

Formato curto de caractere único — JSON se mostrou pesado demais para controle em tempo real:

```
F62*    motor à frente, velocidade 62
L40*    esterço à esquerda, intensidade 40
R40*    esterço à direita
P0*     parar
```

O Python orquestra as sequências temporizadas; o Arduino executa e mantém um **watchdog**: se nenhum comando chegar dentro do timeout, o motor é cortado. Isso protege contra queda do script ou cabo USB solto.

**Arduino → Python:**
```json
{"ack":12}          // confirmação de recebimento
{"btn":"start"}     // botão físico de partida
{"dist":25.4}       // sensor TOF10120, em cm
```

---

## 12. Solução de problemas

**`module 'cv2' has no attribute 'aruco'`**
→ Instalou o `opencv-python` em vez do `opencv-contrib-python`. Desinstale o primeiro e instale o segundo.

**Marcador ArUco não é detectado**
→ Verifique: a margem branca foi cortada? o papel é brilhante e está refletindo? o marcador está muito pequeno ou muito inclinado no frame? Teste imprimindo maior.

**Imagem espelhada**
→ Pressione `F` durante a execução até a orientação ficar correta, e fixe o valor em `CAM_FLIP`.

**A câmera errada abriu**
→ Use `--cam-idx N`. O índice 0 costuma ser a webcam embutida do notebook; a externa geralmente é 1.

**Falsos positivos de PARE**
→ Suba `PGOM_PROMOVE` de 0.85 para 0.90, ou aumente o peso de `persp` e `simet`. Não use `--canny`: as bordas da janela do OpenCV viram contornos.

**O carro não para no PARE**
→ Confira se a placa atinge `AREA_MIN_EXEC` (800 px²) na distância de frenagem desejada, e se a CNN foi treinada. Sem `models/sign_classifier.tflite`, o estágio 2 nunca confirma.

**O carro não reconhece o ponto de entrega**
→ O limiar de "cheguei" também é `AREA_MIN_EXEC`. Meça a área do marcador na distância real de parada e ajuste. `ARUCO_MATCH_DIST` está reservado para uma alternativa por centralidade no frame, ainda não usada.

**Classificações inconsistentes**
→ Thresholds em `models/ood_thresholds.json` altos demais. Reduza para 0.45–0.50.

**`[SER] Simulação — could not open port`**
→ Normal sem hardware. O sistema roda e imprime os comandos no terminal. Para conectar, ajuste `SERIAL_PORT`.

**Semáforo lido errado**
→ A leitura é por posição, não por cor. Confirme que o semáforo está na vertical no frame (use `F` se a imagem estiver rotacionada) e que os três faróis cabem no recorte.

---

## 13. Decisões técnicas registradas

Cada uma destas veio de uma falha real na pista, não de especulação:

- **Cor é inutilizável** — a câmera entrega saturação ≈ 0. Todo HSV foi removido; o semáforo é lido por posição do farol.
- **A CNN não rejeita desconhecidos** — daí o portão geométrico obrigatório antes dela.
- **JSON no serial é lento demais** — substituído por protocolo de caractere único com watchdog no Arduino.
- **YOLO customizado falhou** — bounding boxes autogeradas com 85% de cobertura não generalizam para vídeo real, onde o objeto ocupa 5–15% do frame. O COCO pré-treinado funcionou direto, e hoje é apenas auxiliar opcional.
- **Canny gera falsos positivos** — as bordas da própria janela viram contornos.
- **Thresholds OOD autocalibrados ficam altos demais** — precisam de ajuste manual para baixo.

---

## 14. Próximos passos

- Medir na pista real a distância de frenagem e ajustar `AREA_MIN_EXEC` para PARE e para os marcadores
- Refinar os pesos do PGOM com dados de desempenho real
- Substituir os placeholders de MQTT e telemetria pelos endpoints definitivos do dashboard
- Integração do botão físico de confirmação de entrega no Portenta
