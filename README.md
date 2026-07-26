# 171 Garage — Carro Autônomo
### Challenge 2026 · Mercedes-Benz · FIAP — RM 93657

Sistema de visão computacional para veículo autônomo em miniatura. Detecta placas de trânsito, semáforos, pessoas e veículos via webcam, decide comandos de navegação e os transmite por serial em JSON para um Arduino Portenta H7, que aciona motor, servo de direção, freio, buzzer e LED.

---

## 1. Visão geral da arquitetura

```
Webcam / Vídeo
   │
   ▼
Preprocessing (resize · CLAHE · bilateral filter · sharpen)
   │
   ▼
YOLOv8n ONNX  ──── detecta objetos no frame
   │                  │
   │                  └── se vazio → fallback por contornos + threshold adaptativo
   ▼
ByteTrack-lite ──── associa detecções ao mesmo objeto entre frames (Track ID)
   │
   ▼
CNN MobileNetV3Small ──── classifica o crop (pulado em modo COCO)
   │
   ▼
OOD Rejection (MaxSoftmax) ──── descarta classificações incertas
   │
   ▼
Temporal Vote (10 frames, 60% consenso) ──── confirma antes de agir
   │
   ▼
Máquina de Estados da Missão (AGUARDANDO → RODANDO → PARADO_SEM → ENTREGANDO)
   │
   ▼
Serial JSON com ACK ──── Arduino Portenta H7 ──── motor / servo / freio / buzzer / LED
```

O sistema opera em dois modos de detecção, escolhidos automaticamente pelo modelo carregado em `models/sign_detector.onnx`:

- **Modo COCO** (recomendado para começar): usa o YOLOv8n pré-treinado, que já reconhece `stop sign`, `person`, `car`, `truck`, `bus` e `traffic light` de fábrica — **zero treino necessário**. A classe do YOLO é usada diretamente, sem passar pela CNN.
- **Modo customizado**: usa um YOLO treinado nas 8 classes do projeto (Stop, Esquerda, Direita, SemRetorno, Verde, Cone, Carro, Pessoa) e a CNN para classificação fina. Exige dataset anotado.

---

## 2. Requisitos

- Python 3.11 ou 3.12 (evite 3.13 — o LabelImg quebra nessa versão; usamos anotador próprio)
- Windows, Git Bash ou terminal equivalente
- Webcam USB ou vídeo de simulação
- Arduino Portenta H7 + HAT Carrier (opcional para teste sem hardware)

### Instalação das dependências

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash no Windows
# ou: .venv\Scripts\activate.bat     # CMD

pip install opencv-python numpy pyserial onnxruntime tensorflow ultralytics
pip install matplotlib seaborn scikit-learn        # usado só no treino da CNN
```

Se for usar GPU (opcional, acelera o treino):
```bash
pip install onnxruntime-gpu
```

---

## 3. Estrutura de arquivos

```
autonomous-car/
├── carro-autonomo.py          ← executável principal (roda o sistema)
├── carro_portenta.ino         ← firmware do Arduino Portenta H7
│
├── EXPORT_COCO.py             ← exporta YOLO pré-treinado (caminho rápido, sem treino)
├── EXTRACT_FRAMES.py          ← extrai frames do vídeo para anotação
├── PREPARE_YOLO_DATASET.py    ← converte dataset de classificação → formato YOLO
├── TRAIN_YOLO.py              ← treina YOLOv8n customizado + exporta ONNX
├── TRAIN_SIGN_CNN.py          ← treina a CNN classificadora (MobileNetV3Small)
│
├── dataset_v3/                ← imagens de treino, uma pasta por classe
├── yolo_dataset/               ← gerado automaticamente (formato YOLO)
├── frames_anotacao/            ← gerado pelo EXTRACT_FRAMES.py
├── runs/                        ← saída dos treinos YOLO (métricas, pesos)
│
├── models/
│   ├── sign_detector.onnx      ← YOLO ativo (COCO ou customizado)
│   ├── sign_classifier.tflite  ← CNN classificadora
│   └── ood_thresholds.json     ← thresholds de rejeição por classe
│
├── videoplayback.mp4            ← vídeo de simulação para testes
└── pre_config.json              ← parâmetros de preprocessing calibrados (gerado)
```

---

## 4. Instalação passo a passo — do zero até rodar

### Caminho rápido (recomendado) — sem treino, usando YOLO pré-treinado

```bash
python EXPORT_COCO.py
python carro-autonomo.py
```

Isso já detecta placas de PARE, pessoas, carros e semáforos no vídeo de simulação. Ideal para validar o pipeline completo antes de investir em treino customizado.

### Caminho completo — YOLO customizado nas 8 classes do projeto

**Passo 1 — Extrair frames do vídeo para anotação:**
```bash
python EXTRACT_FRAMES.py --video ./videoplayback.mp4 --step 30
```
Gera ~100-150 frames em `frames_anotacao/`.

**Passo 2 — Treinar o YOLO com os frames reais:**
```bash
python TRAIN_YOLO.py --annotated ./frames_anotacao --real-only --epochs 100 --batch 8
```
Gera `models/sign_detector.onnx` automaticamente.

**Passo 3 — Treinar a CNN classificadora:**
```bash
python TRAIN_SIGN_CNN.py --dataset ./dataset_v3
```
Gera `models/sign_classifier.tflite` e `models/ood_thresholds.json`.

**Passo 4 — Rodar:**
```bash
python carro-autonomo.py --cam     # webcam
python carro-autonomo.py           # vídeo de simulação
```

---

## 5. Uso do executável principal

```bash
python carro-autonomo.py           # roda com o vídeo de simulação
python carro-autonomo.py --cam     # roda com webcam ao vivo
python carro-autonomo.py --cal     # calibra parâmetros de preprocessing
python carro-autonomo.py --debug   # mostra a imagem de threshold do fallback
```

### Controles durante a execução

| Tecla | Ação |
|---|---|
| `Q` | encerra o programa |
| `Espaço` | pausa/despausa |
| `+` / `-` | acelera/desacelera reprodução (só em modo vídeo) |
| `G` | força a partida da missão (equivalente ao semáforo verde) |
| `1` / `2` / `3` | seleciona destino A / B / C (antes da partida) |

### Painel lateral

Mostra em tempo real: FPS e modo de detecção ativo, estado da missão, destino e progresso da rota, tracks ativas e confirmadas, últimas classificações confirmadas, buffer de votação por track, última ação executada.

---

## 6. Máquina de estados da missão

```
AGUARDANDO ──[semáforo verde │ botão do Arduino │ tecla G]──► RODANDO
RODANDO ──[placa/obstáculo confirmado]──► executa ação, avança rota
RODANDO ──[semáforo vermelho]──► PARADO_SEM ──[verde]──► RODANDO
RODANDO ──[rota completa]──► ENTREGANDO ──► FINALIZADO
```

A rota de cada destino (A, B, C) é definida no dicionário `ROTAS` no topo de `carro-autonomo.py` — ajuste conforme o layout físico da pista.

---

## 7. Protocolo serial (Python ↔ Arduino Portenta H7)

**Python → Arduino** (a cada comando, com número de sequência):
```json
{"seq":12,"mot":40,"srv":50,"buz":0,"led":0,"brk":0,"dir":1,"spd":1}
```

**Arduino → Python:**
```json
{"ack":12}          // confirma recebimento do comando
{"btn":"start"}     // botão físico de partida pressionado
{"btn":"D"}         // botão físico de entrega pressionado
{"dist":25.4}       // leitura do sensor TOF10120 em cm
```

Se o ACK não chega em 200ms, o comando é retransmitido automaticamente. O firmware (`carro_portenta.ino`) tem um watchdog: se nenhum comando novo chegar em 1 segundo, o motor é parado por segurança — protege contra queda do script Python ou cabo USB solto.


---

## 8. Calibração de preprocessing

```bash
python carro-autonomo.py --cal --cam
```
---

## 9. Solução de problemas

**"YOLO ONNX não encontrado"**
→ Rode `python EXPORT_COCO.py` para o caminho rápido, ou complete o treino customizado (seção 4).

**YOLO não detecta nada no vídeo, mesmo com objetos visíveis**
→ Sintoma clássico de modelo treinado em crops centrados (objeto ocupando 85% da imagem) tentando generalizar para vídeo real (objeto ocupando 5-15% do frame). Solução: use o modo COCO (`EXPORT_COCO.py`) ou anote frames reais do próprio vídeo (`EXTRACT_FRAMES.py` + `ANNOTATE.py`) antes de treinar.

**Classificações inconsistentes / comandos disparando errado**
→ Verifique `models/ood_thresholds.json`. Thresholds acima de 0.85 raramente são atingidos em vídeo real (foram calibrados em imagens limpas de validação). Reduza manualmente para 0.45–0.55 se necessário.

**Arduino não conecta ("Simulação — could not open port")**
→ Normal se não houver hardware conectado; o sistema roda em modo simulação e imprime os comandos JSON no terminal. Para conectar de fato, ajuste `SERIAL_PORT` no topo de `carro-autonomo.py` ou confirme que o driver da porta está instalado.

**Erro `FileNotFoundError` em `dataset.yaml` durante `yolo train`**
→ Problema de diretório de trabalho. Use os scripts `TRAIN_YOLO.py` (que resolve caminhos absolutos automaticamente) em vez da CLI `yolo` diretamente.

**LabelImg trava com `TypeError: setValue(...) argument 1 has unexpected type 'float'`**
→ Bug conhecido do LabelImg no Python 3.13. Use `python ANNOTATE.py` no lugar — anotador próprio, sem essa dependência.

---

## 10. Classes e mapeamento de ações

| Classe | Ação | Fonte |
|---|---|---|
| Stop | STOP | placa / COCO stop sign |
| SemRetorno | STOP | placa customizada |
| Esquerda | LEFT | placa customizada |
| Direita | RIGHT | placa customizada |
| Verde | STRAIGHT | placa customizada |
| Cone | OBSTACLE | placa customizada |
| Carro | OBSTACLE | placa customizada / COCO car, truck, bus |
| Pessoa | OBSTACLE | placa customizada / COCO person |
| Semáforo | conforme cor (verde/vermelho) | COCO traffic light |

---


## 11. Próximos passos planejados

- Sistema de delivery com seleção de destino (em planejamento — opções avaliadas: botão/RFID, radiofrequência, dashboard web + Power BI)
- Expansão do dataset customizado com mais frames reais anotados
- Refinamento da rota por trecho da pista física