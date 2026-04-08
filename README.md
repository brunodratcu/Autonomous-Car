# 🚗 Carro Autônomo com Visão Computacional  
### Detecção de pista, placas e marcador de decisão

---

## 📌 Visão Geral

Este projeto implementa um sistema de **navegação autônoma para um carro**, utilizando:

- 📷 **Visão computacional (OpenCV)**
- 🧠 **Controle PID para seguir pista**
- 🚦 **Reconhecimento de placas de trânsito**
- 📍 **Marcador no chão para execução precisa de manobras**
- 🔌 **Comunicação serial com Arduino**

O sistema é dividido em dois níveis:

Python (Visão + Decisão) → Arduino (Execução Mecânica)

---

## 🧠 Arquitetura do Sistema

Frame da câmera
↓
Processamento de imagem
↓
Detecção de pista + placas + marcador
↓
Tomada de decisão (estado do carro)
↓
Envio de comandos via serial
↓
Arduino executa (motor + servo)

---

## 🖼️ Processamento de Imagem

### Pipeline de detecção de pista

1. Conversão para escala de cinza  
2. Filtro de ruído (Median Blur)  
3. Binarização (threshold)  
4. ROI (Região inferior)  
5. Canny (bordas)  
6. HoughLinesP  
7. Separação esquerda/direita  
8. Centro da pista  
9. Erro lateral  

---

## 🎯 Controle PID

KP, KI, KD aplicados ao erro lateral para ajuste da direção.

---

## 🚦 Detecção de Placas

- Cor (HSV)
- Forma (contornos)
- Direção (seta)

---

## 📍 Marcador no Chão

Detecta faixa escura transversal para executar manobras no ponto correto.

---

## 🧠 Lógica

Placa define ação → marcador executa ação.

---

## 🔌 Comunicação com Arduino

Formato JSON:

{
  "mot": 62,
  "srv": -20,
  "mode": 3
}

---

## 🚀 Resultado

Sistema robusto de navegação com execução precisa.

