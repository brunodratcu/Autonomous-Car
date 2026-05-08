# 🚦 DETECTOR DE PLACAS — `sign_detector.py`

> Visão Computacional Pura · Sem Modelo Treinado · Serial Arduino

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Instalação](#2-instalação)
3. [Como Usar](#3-como-usar)
4. [Configuração](#4-configuração)
5. [Arquitetura do Código](#5-arquitetura-do-código)
6. [Protocolo Serial](#6-protocolo-serial)
7. [Diagnóstico de Problemas](#7-diagnóstico-de-problemas)
8. [Customizando Novas Placas](#8-customizando-novas-placas)
9. [Fluxo Resumido](#9-fluxo-resumido)
10. [Glossário](#10-glossário)

---

## 1. Visão Geral

O `sign_detector.py` é um sistema de visão computacional que detecta placas de trânsito em tempo real usando **apenas OpenCV e NumPy** — sem nenhum modelo de machine learning ou treinamento prévio.

Ao confirmar uma placa, o programa envia automaticamente um **comando JSON para o Arduino via serial**, que executa a ação correspondente (parar, virar, acelerar etc.).

> **Em uma frase:** Lê frames de câmera ou vídeo → localiza regiões coloridas (placas) → classifica pela cor e forma → confirma por N frames seguidos → envia JSON ao Arduino via serial.

### 1.1 Características Principais

- ✅ Funciona **sem treinamento** — classificação por cor HSV + análise geométrica de contorno
- ✅ **ROI configurável**: só analisa a faixa vertical onde placas realmente aparecem
- ✅ **Sistema de confirmação**: exige N frames consecutivos antes de agir (evita falsos positivos)
- ✅ **Proxy de distância**: só executa quando a placa tiver área mínima no frame (placa próxima)
- ✅ **Cooldown** pós-execução: ignora a mesma placa por um período após agir
- ✅ **Modo calibração HSV**: ferramenta interativa para ajustar as cores para seu ambiente
- ✅ **Painel de debug** em tempo real: candidatas, label, confiança, barra de confirmação e comandos

### 1.2 Placas Reconhecidas

| Label | Cor Dominante | Forma / Símbolo | Ação no Arduino |
|-------|--------------|-----------------|-----------------|
| `STOP` | Vermelho | Octógono (≥7 lados) ou círculo | Motor=0, freio, LED aceso por 2,5s |
| `YIELD` | Vermelho ou Amarelo | Triângulo (3 lados) | Devagar (35%) por 1,5s |
| `LEFT` | Azul ou Vermelho | Seta apontando à esquerda | Curva esquerda por 1,4s |
| `RIGHT` | Azul ou Vermelho | Seta apontando à direita | Curva direita por 1,4s |
| `STRAIGHT` | Azul | Seta apontando para cima | Segue reto (62%) por 1,0s |
| `DELIVERY` | Verde | Qualquer forma sem seta | Para 3s, buzina + LED |
| `PARKING` | Verde | Qualquer forma sem seta | Para 4s, buzina + LED |
| `SPEED_UP` | Verde | Seta para cima | Acelera (80%) por 2,0s |
| `SLOW_DOWN` | Azul, Amarelo ou Laranja | Seta para baixo ou losango | Devagar (30%) por 2,0s |

---

## 2. Instalação

### 2.1 Dependências Python

```bash
pip install opencv-python numpy pyserial

# Python 3.10+ recomendado (usa type hints modernas)
# Testado no Windows 10/11 e Ubuntu 22.04
```

### 2.2 Hardware Necessário

- Câmera USB (qualquer webcam 640×480 ou superior) **ou** arquivo de vídeo `.mov/.mp4/.avi`
- Arduino Uno / Mega / Nano com firmware que lê JSON pela serial
- Cabo USB para conexão Arduino–Computador

### 2.3 Estrutura de Arquivos

```
sign_detector.py      ← programa principal (único arquivo necessário)
pista_01.mov          ← vídeo de teste (opcional)
```

---

## 3. Como Usar

### 3.1 Modos de Execução

| Comando | O que faz |
|---------|-----------|
| `python sign_detector.py` | Abre o arquivo de vídeo definido em `VIDEO` (padrão: `pista_01.mov`) |
| `python sign_detector.py --cam` | Abre a câmera USB (índice `CAM_IDX`, padrão: `0`) |
| `python sign_detector.py --cal` | Abre ferramenta interativa de calibração HSV |

### 3.2 Controles em Tempo Real

| Tecla | Ação |
|-------|------|
| `Q` | Encerra o programa com segurança (envia `brk=1` ao Arduino antes de fechar) |

### 3.3 Calibração HSV (`--cal`)

Se as placas do seu ambiente não estão sendo detectadas, use o modo `--cal` para encontrar os ranges HSV corretos:

1. Execute: `python sign_detector.py --cal`
2. Aponte a câmera para a placa desejada
3. Ajuste os sliders `H_min/H_max`, `S_min/S_max`, `V_min/V_max` até a placa ficar **branca** na máscara
4. Pressione **S** para imprimir os valores no terminal
5. Cole os valores no dicionário `HSV[]` no topo do arquivo `sign_detector.py`

---

## 4. Configuração

Todos os parâmetros ficam no bloco `[1] CONFIGURAÇÃO` no topo do arquivo. Você **não precisa** mexer no restante do código para adaptar ao seu setup.

| Parâmetro | Padrão | O que controla |
|-----------|--------|----------------|
| `VIDEO` | `pista_01.mov` | Caminho do arquivo de vídeo |
| `CAM_IDX` | `0` | Índice da câmera USB (0=primeira, 1=segunda...) |
| `PROP` | `2` | Divisor de resolução: 2=metade, 1=resolução original |
| `SERIAL_PORT` | `COM3` | Porta serial do Arduino (Linux: `/dev/ttyUSB0`) |
| `BAUD` | `115200` | Velocidade da comunicação serial |
| `ROI_Y0` | `0.05` | Topo da faixa de detecção (5% da altura do frame) |
| `ROI_Y1` | `0.80` | Base da faixa de detecção (80% da altura do frame) |
| `AREA_MIN` | `800` | Área mínima do contorno candidato (px²) — filtra ruído |
| `AREA_MAX` | `120000` | Área máxima do contorno candidato (px²) — evita fundo |
| `CONFIRM_N` | `5` | Frames consecutivos necessários para confirmar uma placa |
| `COOLDOWN` | `50` | Frames de espera após executar uma ação (~1,7s a 30fps) |
| `AREA_EXEC` | `2500` | Área mínima para executar — só age se a placa estiver perto |

### Dicas de Ajuste de Sensibilidade

```
Muitos falsos positivos?     → Aumente CONFIRM_N (ex: 7) e AREA_EXEC (ex: 4000)
Confirma muito devagar?      → Diminua CONFIRM_N (ex: 3)
Placa longe não detecta?     → Diminua AREA_MIN (ex: 400) e AREA_EXEC (ex: 1200)
Detecta ruído no chão?       → Diminua ROI_Y1 (ex: 0.65)
```

---

## 5. Arquitetura do Código

O programa é organizado em **13 blocos numerados**, cada um com responsabilidade única. O fluxo de dados é linear:

```
frame → ROI → candidatas → classificação → confirmação → execução → serial
```

---

### Bloco [1] — CONFIGURAÇÃO

Define todas as constantes editáveis pelo usuário: caminhos, serial, ROI, thresholds. É o **único bloco** que precisa ser editado para adaptar o sistema a um novo ambiente.

---

### Bloco [2] — MAPA DE AÇÕES

Dicionário Python que mapeia cada label de placa para os parâmetros do comando JSON:

```python
ACOES = {
    "STOP":  dict(mot=0,  srv=127, buz=0, led=1, brk=1, dir=0, dur=2.5),
    "LEFT":  dict(mot=40, srv=50,  buz=0, led=0, brk=0, dir=1, dur=1.4),
    # ...
}
```

| Campo | Descrição |
|-------|-----------|
| `mot` | Velocidade do motor (0–100% PWM) |
| `srv` | Ângulo do servo (0–254, onde 127=centro, 50=esquerda, 204=direita) |
| `buz` | Buzina (0=off, 1=on) |
| `led` | LED de status (0=off, 1=on) |
| `brk` | Freio (0=livre, 1=freado) |
| `dir` | Direção lógica: 0=neutro, 1=esq, 2=dir, 3=frente |
| `dur` | Duração da ação em segundos |

---

### Bloco [3] — RANGES HSV

Define os intervalos de cor no espaço HSV para cada cor de placa. O **vermelho usa dois ranges** porque envolve o ponto 0° do hue (wrap-around de 168°–180° e 0°–12°).

```python
HSV = {
    "vermelho": [
        (np.array([0,   90,  70]),  np.array([12,  255, 255])),  # vermelho baixo
        (np.array([168, 90,  70]),  np.array([180, 255, 255])),  # vermelho alto
    ],
    "azul":    [(np.array([95, 80, 60]),   np.array([135, 255, 255]))],
    "amarelo": [(np.array([18, 100, 100]), np.array([35,  255, 255]))],
    "verde":   [(np.array([40, 70,  60]),  np.array([85,  255, 255]))],
    "laranja": [(np.array([8,  130, 100]), np.array([20,  255, 255]))],
}
```

---

### Bloco [6] — SERIAL

Duas funções principais:

**`conectar_serial()`**
Auto-detecta Arduino por descrição da porta (`ch340`, `cp210`, `arduino`...). Se não encontrar, usa `SERIAL_PORT` fixo. Se falhar, entra em **modo simulação** (imprime no terminal, não trava).

**`enviar(cmd, ser)`**
Monta o JSON e envia. Sempre imprime no terminal mesmo sem Arduino conectado, permitindo testar sem hardware.

```python
# Exemplo de JSON enviado:
{"mot":0,"srv":127,"buz":0,"led":1,"brk":1,"dir":0,"spd":0}
```

---

### Bloco [7] — UTILITÁRIOS DE VISÃO

Quatro funções auxiliares que formam a base da classificação:

#### `mascara_cor(hsv_img, nome)`
Une todos os ranges HSV da cor em uma máscara binária única.
- **Entrada:** imagem HSV + nome da cor
- **Saída:** máscara `uint8` (0 ou 255)

#### `proporcao_cor(hsv_crop, nome)`
Retorna a fração `[0,1]` de pixels da cor no crop. Usado para determinar a **cor dominante**: a cor com maior proporção ganha.

#### `analisar_forma(gray_crop)`
Binariza o crop (Otsu), encontra o contorno externo e calcula:
- `vertices` — número de lados (3=triângulo, 4=quadrado, 8+=octógono/círculo)
- `circularidade` — quão circular é o contorno (1.0=círculo perfeito)
- `area_ratio` — razão entre área do contorno e área do convex hull (1.0=convexa)

#### `detectar_seta(gray_crop)`
Inverte a binarização para pegar o símbolo escuro no fundo claro. Calcula o **centróide** do maior contorno interno:
- Centróide à **esquerda** do centro → `"left"`
- Centróide à **direita** → `"right"`
- Centróide **acima** → `"up"` (frente)
- Centróide **abaixo** → `"down"`

---

### Bloco [8] — CLASSIFICAÇÃO

Função `classificar()` — coração do sistema. Recebe o crop BGR e aplica a hierarquia de decisão:

```
1. Calcula proporção de cada cor no crop (vermelho, azul, amarelo, verde, laranja)
2. Determina a cor dominante
3. Analisa a forma do contorno (vértices + circularidade)
4. Detecta a direção da seta interna
5. Aplica regras hierárquicas por cor:

   VERMELHO:
     → vertices >= 7  E  circ > 0.60   →  STOP   (conf=0.90)
     → vertices == 3                   →  YIELD  (conf=0.85)
     → circ > 0.72                     →  STOP   (conf=0.78)
     → seta esquerda / direita         →  LEFT / RIGHT

   AZUL:
     → seta left / right / up / down   →  LEFT / RIGHT / STRAIGHT / SLOW_DOWN

   AMARELO:
     → vertices == 3                   →  YIELD
     → vertices == 4 + seta            →  LEFT / RIGHT

   VERDE:
     → sem seta                        →  DELIVERY
     → com seta                        →  LEFT / RIGHT / SPEED_UP

   LARANJA:
     → qualquer forma                  →  SLOW_DOWN
```

---

### Bloco [9] — LOCALIZAÇÃO DE CANDIDATAS

Função `localizar_candidatas()` — encontra **onde** as placas estão no frame:

```
1. Recorta a ROI vertical (ROI_Y0 a ROI_Y1)
2. Converte para HSV + Gaussian Blur 5×5
3. Une as máscaras de todas as cores
4. Morfologia: CLOSE(K7) fecha buracos, OPEN(K5) remove ruído
5. findContours → extrai contornos externos
6. Filtra por AREA_MIN < área < AREA_MAX
7. Filtra por proporção: 0.35 < largura/altura < 2.0
8. Ordena por área decrescente (mais próxima primeiro)
9. Retorna até 4 candidatas como lista de (x, y, w, h, area)
```

---

### Bloco [10] — CONFIRMAÇÃO

A confirmação por múltiplos frames é o que distingue uma detecção legítima de ruído passageiro:

```python
atualizar_confirmacao(label_raw, area):

  Se label_raw == _conf['label']:
    _conf['cnt'] += 1        # mesmo label → incrementa contador
  Senão:
    _conf['label'] = label_raw
    _conf['cnt'] = 1         # label diferente → reinicia do zero

  Se cnt >= CONFIRM_N  E  area >= AREA_EXEC:
    → Retorna o label CONFIRMADO  (ação será executada)
  Senão:
    → Retorna None  (ainda aguardando confirmação)
```

---

### Bloco [11] — EXECUTOR DE AÇÃO

Duas funções que gerenciam o ciclo de vida de uma manobra:

**`executar_acao(label, ser)`**
Busca os parâmetros em `ACOES[label]`, atualiza `CMD`, envia o JSON **imediatamente**, registra horário de início e ativa o cooldown.

**`tick_acao(ser)`**
Chamada a cada frame. Verifica se o tempo decorrido ultrapassou a duração da ação. Quando sim, envia comando de parada (`mot=0, brk=0`) e libera para próxima detecção.

---

### Bloco [12] — VISUALIZAÇÃO

Função `desenhar()` — renderiza o frame de debug:

- 🟠 **Retângulo laranja** — limites da ROI de detecção
- ⬜ **Retângulos cinzas** — todas as candidatas encontradas (com área em px²)
- 🟩 **Retângulo colorido** — melhor candidata classificada (cor varia por label)
- 📊 **Barra de progresso** — confirmação (cinza → cor do label ao completar)
- 📋 **Painel lateral** — FPS, ação ativa com timer, label/confiança/frames, último comando, legenda

---

### Bloco [13] — LOOP PRINCIPAL

O loop principal em `main()` executa a seguinte sequência a cada frame:

```
while True:
  1. cap.read()               → lê frame da câmera/vídeo
  2. resize()                 → reduz resolução (÷ PROP)
  3. cooldown -= 1            → atualiza cooldown
  4. tick_acao()              → mantém / finaliza ação em curso
  5. localizar_candidatas()   → encontra regiões coloridas na ROI
  6. classificar(crop)        → classifica a maior candidata
  7. atualizar_confirmacao()  → acumula frames com mesmo label
  8. executar_acao()          → se confirmado: envia serial
  9. desenhar()               → debug visual
 10. waitKey(1)               → Q para sair
```

---

## 6. Protocolo Serial

### 6.1 Formato do Comando

Todo comando é um JSON de linha única terminado em `\n`:

```json
{"mot":40,"srv":50,"buz":0,"led":0,"brk":0,"dir":1,"spd":1}
```

| Campo | Tipo | Faixa | Descrição |
|-------|------|-------|-----------|
| `mot` | int | 0–100 | Velocidade do motor em % de PWM |
| `srv` | int | 0–254 | Posição do servo: 0=máx esq, 127=centro, 254=máx dir |
| `buz` | int | 0 ou 1 | Buzina: 0=silêncio, 1=ativa |
| `led` | int | 0 ou 1 | LED de status: 0=apagado, 1=aceso |
| `brk` | int | 0 ou 1 | Freio: 0=livre, 1=frear |
| `dir` | int | 0–3 | Direção lógica: 0=neutro, 1=esq, 2=dir, 3=frente |
| `spd` | int | 0–3 | Perfil de velocidade: 0=parado, 1=lento, 2=normal, 3=rápido |

### 6.2 Exemplo de Firmware Arduino

```cpp
#include <ArduinoJson.h>

void loop() {
  if (Serial.available()) {
    String linha = Serial.readStringUntil('\n');
    StaticJsonDocument<200> doc;
    if (!deserializeJson(doc, linha)) {
      int mot = doc["mot"];
      int srv = doc["srv"];
      int buz = doc["buz"];
      int led = doc["led"];
      int brk = doc["brk"];
      // aplicar aos pinos...
    }
  }
}
```

### 6.3 Configurações de Conexão

| Parâmetro | Valor |
|-----------|-------|
| Velocidade | 115200 baud |
| Terminador | `\n` (newline) |
| Timeout | `write_timeout=0` (não bloqueia) |
| Auto-detect | Busca `arduino`, `ch340`, `cp210`, `uart` na descrição da porta |
| Fallback | Usa `SERIAL_PORT = "COM3"` se auto-detect falhar |

---

## 7. Diagnóstico de Problemas

| Sintoma | Causa Provável | Solução |
|---------|---------------|---------|
| Placa não detectada | Range HSV muito restrito | Use `--cal` para recalibrar a cor |
| Placa detectada mas não executa | Área abaixo de `AREA_EXEC` | Aproxime mais a placa ou diminua `AREA_EXEC` |
| Muitos falsos positivos | Objetos coloridos no cenário | Aumente `CONFIRM_N` ou ajuste `ROI_Y0`/`ROI_Y1` |
| Label errado (ex: YIELD→STOP) | Iluminação muda forma percebida | Calibre HSV e ajuste limiares em `classificar()` |
| Serial não conecta | Porta errada ou driver ausente | Cheque Gerenciador de Dispositivos; instale driver CH340/CP210 |
| Travamento no `cv2.imshow` | Resolução muito alta | Aumente `PROP` para 3 ou 4 |
| Lento (< 15 FPS) | Câmera ou CPU lenta | Aumente `PROP`, reduza ROI, desconecte câmeras extras |

---

## 8. Customizando Novas Placas

Para adicionar suporte a uma nova placa (exemplo: `REVERSE` — azul com seta para baixo):

**Passo 1** — Adicione a ação no dicionário `ACOES`:
```python
"REVERSE": dict(mot=35, srv=127, buz=1, led=0, brk=0, dir=0, dur=1.5),
```

**Passo 2** — Adicione a regra em `classificar()`, dentro do bloco da cor azul:
```python
elif seta == "down":
    label, conf = "REVERSE", 0.82
```

**Passo 3** — Adicione a cor de visualização em `LABEL_BGR`:
```python
"REVERSE": (100, 100, 220),
```

> **Dica para novas cores:** se a placa tiver uma cor que não está em `HSV[]`, use `--cal` para encontrar os ranges e adicione em `HSV[]`. A nova cor passará a ser detectada automaticamente.

---

## 9. Fluxo Resumido

```
FRAME DA CÂMERA
     │
     ▼
RESIZE (÷ PROP)  →  reduz CPU
     │
     ▼
localizar_candidatas(frame)
  ├── Recorta ROI_Y0 a ROI_Y1
  ├── HSV + Blur
  ├── Une máscaras de TODAS as cores
  ├── Morfologia (fecha buracos, remove ruído)
  ├── findContours → filtra por área e proporção
  └── Retorna lista de (x, y, w, h, area) ordenada por área
     │
     ▼
classificar(crop_maior_candidata)
  ├── proporcao_cor()    → cor dominante
  ├── analisar_forma()   → vértices, circularidade
  ├── detectar_seta()    → left / right / up / down
  └── Regras hierárquicas → (label, confiança)
     │
     ▼
atualizar_confirmacao(label, area)
  ├── cnt++ se mesmo label, reset se mudou
  ├── Se cnt >= CONFIRM_N  E  area >= AREA_EXEC:
  │     └── retorna label CONFIRMADO
  └── Senão: retorna None
     │
     ▼ (confirmado)
executar_acao(label, ser)
  ├── CMD ← parâmetros de ACOES[label]
  ├── enviar(CMD, ser)   → JSON → Arduino via serial
  └── ativa cooldown = COOLDOWN frames
     │
     ▼
tick_acao(ser)  →  repete CMD até duração esgotar, então para
     │
     ▼
desenhar()  →  frame + ROI + bbox + barra + painel
     │
     └── volta para FRAME
```

---

## 10. Glossário

| Termo | Definição |
|-------|-----------|
| **HSV** | Espaço de cor Hue/Saturation/Value. Mais robusto que RGB para segmentação por cor sob diferentes iluminações. |
| **ROI** | Region of Interest — faixa do frame onde o algoritmo procura placas. Reduz processamento e falsos positivos. |
| **Morfologia** | Operações em imagem binária. `CLOSE` fecha buracos; `OPEN` remove ruídos pequenos. |
| **Contorno** | Curva que delimita uma região de mesmo valor em imagem binária. `findContours()` os extrai. |
| **Binarização Otsu** | Algoritmo que encontra automaticamente o melhor threshold para separar fundo de objeto. |
| **Circularidade** | `4π·área / perímetro²`. Vale 1.0 para círculo perfeito. Útil para distinguir octógono de triângulo. |
| **Convex Hull** | Menor polígono convexo que envolve um contorno. `area_ratio = contorno/hull` ≈ 1 para formas convexas. |
| **CONFIRM_N** | Número de frames consecutivos com o mesmo label para considerar a detecção válida. |
| **AREA_EXEC** | Área mínima (px²) da placa no frame para executar a ação — funciona como proxy de distância. |
| **PWM** | Pulse Width Modulation — controla velocidade do motor variando o ciclo de trabalho do sinal (0–100%). |
| **JSON** | JavaScript Object Notation — formato de texto simples usado para enviar comandos ao Arduino via serial. |

---
