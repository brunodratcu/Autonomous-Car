"""
================================================================
  SCAN_CAMERAS.py — descobre qual índice é a sua câmera
  ─────────────────────────────────────────────────────────────
  A webcam embutida costuma ser o índice 0. A câmera externa
  entra como 1, 2, 3... Este script varre os índices, abre cada
  câmera que responder e mostra o vídeo com o número dela.

  USO:
    python SCAN_CAMERAS.py

  Olhe cada janela que abrir: quando aparecer a imagem da SUA
  câmera (a da pista), anote o número no título da janela — é
  esse valor que vai em CAM_IDX no carro-autonomo.py.

  Teclas:  Q = próxima câmera   ESC = sair
================================================================
"""

import cv2
import sys

# Backend correto por plataforma (o mesmo do carro-autonomo.py)
if sys.platform.startswith("win"):
    BACKEND, NOME = cv2.CAP_DSHOW, "CAP_DSHOW (Windows)"
elif sys.platform.startswith("linux"):
    BACKEND, NOME = cv2.CAP_V4L2, "CAP_V4L2 (Linux)"
else:
    BACKEND, NOME = cv2.CAP_ANY, "CAP_ANY"

print(f"[SCAN] Plataforma: {sys.platform} | backend: {NOME}")
print("[SCAN] Varrendo índices 0..9...\n")

encontradas = []

for idx in range(10):
    cap = cv2.VideoCapture(idx, BACKEND)
    if not cap.isOpened():
        cap.release()
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  índice {idx}: abriu mas NÃO entregou frame "
              f"(pode estar em uso por outro programa)")
        cap.release()
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fstr = "".join(chr((fourcc >> 8*i) & 0xFF) for i in range(4)).strip()

    print(f"  índice {idx}: OK  {w}x{h} @ {fps:.0f}fps  fourcc={fstr}")
    encontradas.append(idx)

    win = f"Camera indice {idx}  --  Q=proxima  ESC=sair"
    print(f"           → mostrando janela. Se for a SUA camera, "
          f"anote: CAM_IDX = {idx}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        etiqueta = frame.copy()
        cv2.putText(etiqueta, f"CAM_IDX = {idx}   ({w}x{h})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0), 2)
        cv2.putText(etiqueta, "Q = proxima camera    ESC = sair",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 1)
        cv2.imshow(win, etiqueta)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            print("\n[SCAN] Encerrado pelo usuário.")
            sys.exit(0)

    cap.release()
    cv2.destroyWindow(win)

cv2.destroyAllWindows()

print("\n" + "="*60)
if encontradas:
    print(f"[SCAN] Câmeras encontradas nos índices: {encontradas}")
    print(f"[SCAN] A embutida do laptop costuma ser a 0.")
    print(f"[SCAN] Edite CAM_IDX no carro-autonomo.py com o número")
    print(f"       da câmera da pista que você identificou.")
else:
    print("[SCAN] NENHUMA câmera respondeu.")
    print("       · A câmera está conectada e o cabo firme?")
    print("       · Algum outro programa (Zoom, OBS, navegador)")
    print("         está usando a câmera? Feche e tente de novo.")
    print("       · No Linux: o usuário está no grupo 'video'?")
print("="*60)