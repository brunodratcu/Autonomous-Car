# 171 Garage — Carro Autônomo

Veículo autônomo em miniatura para o **FIAP/Mercedes-Benz Challenge 2026**. Detecta placas de trânsito (PARE, semáforo), desvia de obstáculos e executa um serviço de entrega entre três pontos (A/B/C) usando apenas visão computacional — sem GPS, sem LiDAR, sem depender de internet na pista.

```
webcam → mono + CLAHE → varredura única de contornos
                              │
                 ┌────────────┴────────────┐
              OCTÓGONO                  CÍRCULO
                 │                         │
            PARE (PGOM→CNN)          letra A/B/C
                 │                         │
                 └──────► MISSÃO ◄─────────┘
                              │
                       serial → Portenta H7
```

---

## Sumário

- [Visão geral](#visão-geral)
- [Hardware](#hardware)
- [Instalação](#instalação)
- [Uso](#uso)
- [Arquitetura do pipeline](#arquitetura-do-pipeline)
- [Sistema de entrega (placas A/B/C)](#sistema-de-entrega-placas-abc)
- [Máquina de estados da missão](#máquina-de-estados-da-missão)
- [Painel de controle (celular)](#painel-de-controle-celular)
- [Protocolo serial](#protocolo-serial)
- [Estrutura de arquivos](#estrutura-de-arquivos)
- [Calibração](#calibração)
- [Decisões de projeto e porquês](#decisões-de-projeto-e-porquês)
- [Limitações conhecidas](#limitações-conhecidas)
- [Roadmap](#roadmap)

---

## Visão geral

O carro roda inteiramente em **tons de cinza**: a câmera do projeto entrega imagem próxima de binária, então toda a pipeline — segmentação, classificação de forma, leitura de placa — foi construída sobre contraste, não sobre cor. Não existe HSV em nenhum ponto do código.

Duas fontes de decisão trabalham em paralelo sobre a mesma varredura de contornos:

- **Trânsito** — placas octogonais (PARE) e semáforo, com geometria decidindo forma, PGOM acumulando evidência entre frames e uma CNN entrando só para classificação fina.
- **Delivery** — placas circulares com letras A/B/C, lidas por um método 100% determinístico (sem rede neural): topologia de buracos + casamento de molde normalizado.

Uma máquina de estados (`Missao`) decide o que fazer com essas percepções e envia comandos por serial para o Arduino Portenta H7, que aciona motor, servo, buzzer e LED.

---

## Hardware

| Item | Especificação |
|---|---|
| Compute | Laptop Windows rodando o pipeline Python |
| Câmera | Webcam externa USB, índice fixo (`CAM_IDX = 1`), flip horizontal |
| Controlador | Arduino Portenta H7 — motor, servo, buzzer, LED via serial |
| Rede | TP-Link TL-MR3020 em modo AP, sem internet (gateway `192.168.0.1`) |
| Entre-eixos | 720 mm |
| Track width | 795 mm |
| Diâmetro da roda | 350 mm |
| Ângulo de esterçamento interno / externo | 31,32° / 20° |
| Raio mínimo de giro | ~1,58 m |
| Desvio Ackermann | < 0,5% (verificado em CAD) |

---

## Instalação

```bash
pip install opencv-python numpy pyserial qrcode
# opcionais — só entram em ação se os arquivos existirem em ./models/
pip install onnxruntime tensorflow
```

Modelos esperados (opcionais; o sistema roda em modo puramente geométrico sem eles):

```
models/
├── sign_detector.onnx        # YOLOv8 (COCO pré-treinado ou custom)
├── sign_classifier.tflite    # MobileNetV3Small, INT8
└── ood_thresholds.json       # limiares por classe para rejeição OOD
```

---

## Uso

```bash
# Execução padrão — webcam + serial + painel HTTP
python carro-autonomo.py

# Calibração das placas A/B/C (mostra motivo de cada rejeição)
python carro-autonomo.py --debug-abc

# Retirada A → entrega B automática, sem depender do QR/celular
python carro-autonomo.py --auto

# Webcam em índice diferente de 1
python carro-autonomo.py --cam-idx 2

# Processa todo frame (em vez de 1 em cada 2) + YOLO auxiliar
python carro-autonomo.py --fast --yolo

# Gera os PNGs das placas A/B/C para impressão
python carro-autonomo.py --gerar-abc
```

### Teclas em tempo real

| Tecla | Ação |
|---|---|
| `Q` | Sair |
| `Espaço` | Pausar/retomar |
| `1` / `2` / `3` | Define ponto de retirada (1ª vez) ou entrega (2ª vez) — A/B/C |
| `S` | Imprime status da missão no console |
| `D` | Liga/desliga motores |
| `R` | Reseta a missão |
| `F` | Alterna o flip da câmera (nenhum / horizontal / vertical) |

---

## Arquitetura do pipeline

O arquivo é organizado em 12 blocos, nesta ordem:

| Bloco | Conteúdo | Papel |
|---|---|---|
| `[0]` | Configuração | Todas as constantes calibráveis, num só lugar |
| `[1]` | Pré-processamento e segmentação | `preprocessar`, `segmentar`, `ROIDinamico`, peneira de ROI |
| `[2]` | Classificador de forma | `harmonico_8` — árbitro único entre círculo e octógono |
| `[3]` | Leitura das placas A/B/C | `LeitorLetras` — dois juízes independentes |
| `[4]` | Varredura única | `detectar_geometrico` — uma passada de contornos alimenta os dois ramos |
| `[5]` | PGOM | Acúmulo de evidência geométrica entre frames antes de liberar a CNN |
| `[6]` | Redes auxiliares | YOLO, CNN, OOD, verificador de contrato forma↔classe |
| `[7]` | Rastreio | `ByteTrackLite` — tracking-by-detection com votação temporal |
| `[8]` | Missão | FSM da entrega |
| `[9]` | Controle e serial | Protocolo JSON com ack, buzzer não bloqueante |
| `[10]` | Painel local | Servidor HTTP em thread própria |
| `[11]` | HUD | Overlay de depuração na janela do OpenCV |
| `[12]` | Loop principal | `main()` |

### Por que uma varredura única de contornos e não dois detectores separados

Segmentar a imagem duas vezes (uma para trânsito, outra para delivery) dobraria o custo por frame sem necessidade. `detectar_geometrico` percorre os contornos **uma vez** e roteia cada um pela forma: octógono vira candidato a `Stop`, círculo vira candidato a `Delivery`, o resto é descartado. Medido: **~11 ms/frame** no laptop de desenvolvimento.

### O 8º harmônico do raio — por que não `approxPolyDP`

`approxPolyDP` devolve 8 vértices tanto para um octógono quanto para um círculo mal segmentado — contar lados não separa as duas formas. A solução (`harmonico_8`) amostra o raio `r(θ)` do contorno em 64 pontos e mede a força do 8º harmônico via FFT:

- **Círculo**: harmônico fraco, `0,0033–0,0212` (medido)
- **Octógono**: harmônico forte, `0,0266–0,0335` (medido)

Existe uma zona morta entre as duas faixas (`FORMA_H8_CIRCULO = 0.022` a `FORMA_H8_OCTOG = 0.025`) que devolve `"?"` propositalmente — não chutar é melhor que errar.

### PGOM — evidência antes de rede neural

Cada candidato geométrico ganha uma `Ficha` que acumula, ao longo de até 10 frames: coerência de forma, convexidade, aspecto, presença de símbolo interno, persistência, estabilidade de trajetória e de crescimento de área, fingerprint geométrico, perspectiva e simetria. Só quem ultrapassa o limiar (`PGOM_PROMOVE = 0.85`, ou `0.72` para semáforo) é liberado para a CNN. Isso resolve o problema clássico do softmax: uma rede de classificação não sabe dizer "isto não é nada" — o PGOM filtra antes que o ruído chegue lá.

---

## Sistema de entrega (placas A/B/C)

O delivery **não** passa por PGOM nem por CNN. É geometria + dois juízes determinísticos, escolha deliberada: são três símbolos fixos, impressos pela própria equipe, sempre no mesmo formato — o caso ideal para template matching e o caso desnecessário para rede neural.

```
círculo achado → miolo binarizado → recorte da tinta na própria bbox
                                              │
                          ┌───────────────────┴───────────────────┐
                     TOPOLOGIA                                 MOLDE
                (conta buracos fechados)          (casa contra o molde normalizado)
                  A=1 · B=2 · C=0                                 │
                          └───────────────┬───────────────────────┘
                                    concordam?
                                          │
                                  votação temporal
                              (3 confirmações em 5 frames)
```

**Por que dois juízes e não um:**
- Só topologia confunde `O` com `A` (ambos têm 1 buraco).
- Só molde é refém da fonte impressa — foi exatamente o bug original do projeto: o molde interno usava uma fonte fina (`FONT_HERSHEY_SIMPLEX`) e a placa impressa usava outra (sans-serif bold), e o casamento pixel a pixel reprovava tudo.
- **Exigir que os dois concordem** resolve os dois problemas ao mesmo tempo, sem precisar de dataset nem de treino.

Medido em bateria de testes (placas reais, 4 tamanhos × 3 níveis de desfoque × 3 ângulos de perspectiva): **108/108 acertos**, com 11 de 13 distratores (letras fora do alfabeto, ruído) corretamente rejeitados.

### Gerando e imprimindo as placas

```bash
python carro-autonomo.py --gerar-abc
```

Gera `placas_abc/ponto_A.png`, `ponto_B.png`, `ponto_C.png` com a mesma fonte usada internamente pelo molde de comparação. **Imprima em papel fosco, ~15 cm de diâmetro, sem recortar rente ao anel**, e fixe na altura da câmera. Papel brilhante cria reflexo que desloca o Otsu; recorte rente altera a proporção que os filtros de forma esperam.

---

## Máquina de estados da missão

```
AGUARDANDO ─[1/2/3 ou painel: retirada X]─► IND_RETIRADA
                                                  │ chegou em X
                                                  ▼
                                              RETIRANDO (7s)
                                                  │
                        ┌─────────────────────────┴─────────────────────┐
                  entrega já definida                              sem entrega definida
                        │                                                │
                        ▼                                                ▼
                  IND_ENTREGA                                    AGUARDA_ENTREGA
                        │ chegou em Y                              │ [1/2/3 ou painel]
                        ▼                                          ▼
                  ENTREGANDO (7s) ◄─────────────────────────── IND_ENTREGA
                        │
                        ▼
                      LIVRE  (carro continua rodando, pronto para nova retirada)
```

Sobrepostos a qualquer estado "rodando", três estados de **parada absoluta**, nesta ordem de prioridade:

```
obstáculo  >  semáforo vermelho  >  PARE
```

Cada um guarda o estado de origem (`MISSAO.retorno`) para retomar exatamente de onde parou. Duas decisões deliberadas:

- **A chegada ao ponto é avaliada mesmo parado por regra absoluta** — o ponto de entrega frequentemente fica junto de um cruzamento, então ignorar a leitura da placa enquanto o carro está parado no sinal faria o carro passar direto.
- **Depois de entregar, o carro vai para `LIVRE` e continua andando** — não trava esperando comando.

### Separação estrita de responsabilidades

```python
percep = dict(pare=False, semaforo=None, obstaculo=False, ponto=None)  # VISÃO constata
# ...
if MISSAO.regra_absoluta(percep): ...                                  # MISSÃO decide
# ...
seguir(_ser) / parar(_ser)                                             # CONTROLE executa
```

Visão nunca decide. Missão nunca acessa serial diretamente. Controle nunca interpreta percepção.

---

## Painel de controle (celular)

Um `ThreadingHTTPServer` sobe em thread daemon (`PainelLocal`), servindo uma página HTML/JS de uma peça só — sem framework, sem build step. O loop de visão só faz `poll()` na fila de comandos; o servidor nunca bloqueia a detecção.

Dois QR codes ficam fixos no canto do HUD durante toda a sessão:

1. **Wi-Fi** (`WIFI:T:WPA;S:...;P:...;;`) — entra na rede do TP-Link sem digitar senha
2. **Painel** — abre a URL do painel HTTP no navegador do celular

O painel mostra três botões por etapa (retirar em A/B/C, entregar em A/B/C) e o estado atual da missão, atualizado a cada 1,5s.

---

## Protocolo serial

```json
{"seq": 42, "mot": 90, "srv": 127, "buz": 0, "led": 0, "brk": 0, "dir": 3, "spd": 2}
```

| Campo | Significado |
|---|---|
| `seq` | número sequencial — usado para confirmação (`ack`) |
| `mot` | potência do motor (0 = parado) |
| `srv` | posição do servo de direção (127 = centro) |
| `buz` | buzzer ligado/desligado |
| `led` | LED indicador |
| `brk` | freio |
| `dir` | direção lógica |
| `spd` | faixa de velocidade derivada de `mot` (0/1/2) |

`enviar()` escreve uma linha JSON por comando. `ler_serial()` consome o `ack` da Portenta e retransmite automaticamente se não confirmado em 200 ms. O buzzer é **agendado**, não bloqueante — `buzinar()` marca um timestamp de desligamento e `tick_buzzer()` verifica a cada volta do loop, sem travar a captura de frames.

> ⚠️ **Alinhar com o firmware.** Este runtime envia JSON. Se o `.ino` da Portenta espera outro formato, ajuste um dos dois lados antes do teste em hardware — é a única peça que acopla os dois arquivos.

---

## Estrutura de arquivos

```
.
├── carro-autonomo.py       # runtime único — tudo em um arquivo, de propósito
├── models/
│   ├── sign_detector.onnx      (opcional)
│   ├── sign_classifier.tflite  (opcional)
│   └── ood_thresholds.json     (opcional)
└── placas_abc/              # gerado por --gerar-abc
    ├── ponto_A.png
    ├── ponto_B.png
    └── ponto_C.png
```

**Arquitetura single-file é deliberada**, não falta de organização: qualquer pessoa da equipe — inclusive quem só mexe no firmware da Portenta — consegue abrir um arquivo e ler o fluxo inteiro sem navegar entre módulos.

---

## Calibração

A única constante que **precisa** ser recalibrada na pista é `AREA_MIN_CHEGADA` (bbox mínima da placa para o carro considerar "cheguei"), porque depende da lente da câmera usada no dia.

```bash
python carro-autonomo.py --debug-abc
```

Com a placa impressa (papel fosco) na distância exata em que o carro deve parar, leia no console:

```
[ABC] ponto 'C' — área=NNNN (perto/longe, min=9000) alvo=C
```

Ajuste `AREA_MIN_CHEGADA` em `carro-autonomo.py` para ~80% do valor observado. Repita para as três letras se a distância de leitura variar por ponto.

Demais constantes de forma e leitura já vêm calibradas com dados reais (placas impressas, múltiplos ângulos e desfoques) e não deveriam precisar de ajuste, exceto se o material de impressão mudar significativamente.

---
