"""
Diagnóstico de câmera — roda os MESMOS testes que peneira_roi() usa
no pipeline, mas sobre o frame inteiro, para dizer exatamente por
que a detecção está falhando: ruído, foco, ou luz.

Uso:
    python diag_camera.py            # usa CAM_IDX=1, DSHOW (Windows)
    python diag_camera.py --idx 2
"""
import cv2, numpy as np, sys, time

IDX = 1
if "--idx" in sys.argv:
    IDX = int(sys.argv[sys.argv.index("--idx")+1])

cap = cv2.VideoCapture(IDX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"[ERRO] câmera índice {IDX} não abriu. Tente --idx 0, 1, 2...")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Lendo 60 frames (~2s)... aponte para uma placa PARE ou delivery se tiver à mão.")
print("Pressione Q na janela para sair mais cedo.\n")

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

focos, ruidos, brilhos, quedas = [], [], [], 0
t0 = time.time()
n = 0
prev = None

while n < 60 and time.time()-t0 < 8:
    ret, frame = cap.read()
    if not ret:
        quedas += 1
        continue
    n += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = CLAHE.apply(gray)

    # foco: mesmo cálculo do peneira_roi (variância do Laplaciano)
    foco = float(cv2.Laplacian(cv2.resize(g,(48,48)), cv2.CV_64F).var())
    focos.append(foco)

    # ruído: desvio padrão entre frames consecutivos na mesma região
    # (se a cena está parada, qualquer diferença é ruído do sensor/cabo)
    if prev is not None:
        diff = cv2.absdiff(gray, prev)
        ruidos.append(float(diff.std()))
    prev = gray

    brilhos.append(float(gray.mean()))

    cv2.imshow("Diagnostico - Q para sair", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()

if n == 0:
    print("[ERRO] nenhum frame foi lido — problema é na conexão/driver, não no sinal.")
    sys.exit(1)

foco_med = np.median(focos)
ruido_med = np.median(ruidos) if ruidos else -1
brilho_med = np.median(brilhos)
taxa_perda = quedas / max(1, n+quedas) * 100

print("="*58)
print(f"  Frames lidos:        {n}   (perdidos: {quedas}, {taxa_perda:.0f}%)")
print(f"  Foco (Laplaciano):   {foco_med:.1f}   (pipeline exige >= 25.0)")
print(f"  Ruído entre frames:  {ruido_med:.2f}   (cena parada: normal é < 2.0)")
print(f"  Brilho médio:        {brilho_med:.0f}   (pipeline aceita 12-248)")
print("="*58)

print("\nDIAGNÓSTICO:")
if taxa_perda > 5:
    print("  ⚠ PERDA DE FRAMES — típico de cabo USB com mau contato ou banda")
    print("    insuficiente. Confirma problema físico de conexão, não de threshold.")
if ruido_med > 6:
    print("  ⚠ RUÍDO ALTO ENTRE FRAMES — com a cena parada, isso não deveria")
    print("    variar. É a assinatura de neve/interferência no sinal de vídeo:")
    print("    cabo USB mal contatado, cabo tracionado, ou EMI do motor/servo")
    print("    perto do cabo da câmera. NÃO é um problema de threshold do")
    print("    pipeline — ajustar ROI_FOCO_MIN ou AREA_MIN não resolve isso.")
elif ruido_med > 3:
    print("  ~ ruído moderado — pode ainda passar no pipeline, mas vale investigar")
else:
    print("  ✓ ruído normal")
if foco_med < 25:
    print(f"  ⚠ FOCO ABAIXO DO LIMIAR ({foco_med:.1f} < 25.0) — o pipeline vai")
    print("    rejeitar a maioria dos recortes por 'desfocada'. Se a câmera tem")
    print("    foco manual, ajuste; ruído alto também derruba esta métrica.")
else:
    print("  ✓ foco OK")
if not (12 < brilho_med < 248):
    print("  ⚠ BRILHO fora da faixa aceita pelo pipeline")
else:
    print("  ✓ brilho OK")

print("\nPróximo passo se o ruído estiver alto:")
print("  1. Desconecte e reconecte o cabo USB nas duas pontas, sem tensão")
print("  2. Troque de porta USB (evite hub, prefira USB 3.0 direta)")
print("  3. Afaste o cabo da câmera dos fios de motor/servo/Portenta")
print("  4. Rode este script de novo e compare o número de 'Ruído entre frames'")