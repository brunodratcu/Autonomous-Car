"""
================================================================
  TESTE RÁPIDO — YOLO de Localização (Pare + Semaforo)
  171 Garage · roda o .onnx treinado em cima de imagens novas
================================================================
  Mesma lógica do testar_cnn.py: ver se generaliza pra fora do
  vídeo de treino ANTES de confiar no carro. Com mAP50=0.995 e
  só 40 imagens de treino (uma sessão só), a suspeita é de
  overfitting de cena — este script existe pra checar isso.

  USO (imagens):
      python testar_yolo.py --imgs pasta/com/imagens/novas
      python testar_yolo.py --imgs pasta/com/imagens/novas --conf 0.4

  USO (vídeo):
      python testar_yolo.py --video video.mp4
      python testar_yolo.py --video video.mp4 --show      (abre janela ao vivo)
      python testar_yolo.py --video video.mp4 --step 3     (1 a cada 3 frames, mais rápido)

  Imagens: salva com as caixas desenhadas em ./revisao_yolo/
  Vídeo: salva um .mp4 novo com as caixas desenhadas + resumo no console.
================================================================
"""
import os, sys, argparse
from pathlib import Path
import numpy as np
import cv2

CLASSES = ["Pare", "Semaforo"]     # mesma ordem do dataset.yaml usado no treino
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CORES = [(0, 0, 255), (0, 255, 0)]  # Pare=vermelho, Semaforo=verde

def carregar_onnx(model_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    return sess, inp.name, inp.shape

def preprocessar(img, imgsz=640):
    """Letterbox simples + BGR->RGB->CHW->float32/255, igual ao que o Ultralytics usa."""
    h0, w0 = img.shape[:2]
    r = min(imgsz / h0, imgsz / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return x[None, ...], r, left, top

def nms(boxes, scores, iou_thr=0.45):
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[idx[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idx[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idx[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idx[1:],3])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        inter = w*h
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_o = (boxes[idx[1:],2]-boxes[idx[1:],0])*(boxes[idx[1:],3]-boxes[idx[1:],1])
        iou = inter / (area_i + area_o - inter + 1e-9)
        idx = idx[1:][iou < iou_thr]
    return keep

def detectar(sess, in_name, img, conf_thr=0.35, imgsz=640):
    x, r, left, top = preprocessar(img, imgsz)
    out = sess.run(None, {in_name: x})[0]        # [1, 4+nc, N]
    pred = out[0].T                                # [N, 4+nc]
    boxes_cxcywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    conf = scores_all.max(axis=1)

    mask = conf >= conf_thr
    boxes_cxcywh, cls, conf = boxes_cxcywh[mask], cls[mask], conf[mask]
    if len(conf) == 0:
        return []

    cx, cy, w, h = boxes_cxcywh.T
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    # desfaz o letterbox pra coordenada da imagem original
    x1 = (x1 - left) / r; x2 = (x2 - left) / r
    y1 = (y1 - top) / r;  y2 = (y2 - top) / r
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    dets = []
    for c in np.unique(cls):
        m = cls == c
        keep = nms(boxes[m], conf[m])
        for k in keep:
            dets.append((boxes[m][k], float(conf[m][k]), int(c)))
    return dets

def rodar_video(sess, in_name, args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERRO] não consegui abrir o vídeo: {args.video}"); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Vídeo: {w}x{h} @ {fps:.0f}fps, {total} frames (step={args.step})")

    out_path = str(Path(args.out) / (Path(args.video).stem + "_detectado.mp4"))
    os.makedirs(args.out, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.step, (w, h))

    frame_idx = 0
    frames_processados = 0
    frames_sem_deteccao = 0
    contagem = {c: 0 for c in CLASSES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.step != 0:
            frame_idx += 1
            continue

        dets = detectar(sess, in_name, frame, conf_thr=args.conf, imgsz=args.imgsz)
        frames_processados += 1
        if not dets:
            frames_sem_deteccao += 1

        for box, conf, c in dets:
            nome = CLASSES[c] if c < len(CLASSES) else f"cls{c}"
            contagem[nome] = contagem.get(nome, 0) + 1
            x1, y1, x2, y2 = box.astype(int)
            cor = CORES[c] if c < len(CORES) else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, f"{nome} {conf:.2f}", (x1, max(15, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        writer.write(frame)
        if args.show:
            cv2.imshow("YOLO - Pare/Semaforo (Q pra sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frames_processados % 30 == 0:
            print(f"  ...{frames_processados} frames processados")
        frame_idx += 1

    cap.release(); writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\n  Frames processados : {frames_processados}")
    print(f"  Sem detecção nenhuma: {frames_sem_deteccao} ({100*frames_sem_deteccao/max(1,frames_processados):.1f}%)")
    print(f"  Contagem por classe : {contagem}")
    print(f"\n  [OK] vídeo salvo em {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--model", default="runs/detect/runs_yolo/pare_semaforo/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="revisao_yolo")
    ap.add_argument("--step", type=int, default=1, help="processa 1 a cada N frames (vídeo)")
    ap.add_argument("--show", action="store_true", help="abre janela ao vivo (vídeo)")
    args = ap.parse_args()

    if not args.imgs and not args.video:
        print("[ERRO] passa --imgs pasta/ ou --video arquivo.mp4"); sys.exit(1)

    if not os.path.exists(args.model):
        print(f"[ERRO] modelo não encontrado: {args.model}")
        print("       ajusta --model pro caminho real do teu best.onnx")
        sys.exit(1)

    sess, in_name, in_shape = carregar_onnx(args.model)
    print(f"  Modelo carregado: {args.model}  (input shape: {in_shape})")

    if args.video:
        rodar_video(sess, in_name, args)
        return

    imgs = sorted(p for p in Path(args.imgs).rglob("*") if p.suffix.lower() in EXTS)
    if not imgs:
        print(f"[ERRO] nenhuma imagem em {args.imgs}"); sys.exit(1)
    print(f"  Testando {len(imgs)} imagens\n")

    os.makedirs(args.out, exist_ok=True)
    n_sem_deteccao = 0
    contagem = {c: 0 for c in CLASSES}

    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [pular] não consegui ler {p.name}"); continue
        dets = detectar(sess, in_name, img, conf_thr=args.conf, imgsz=args.imgsz)

        if not dets:
            n_sem_deteccao += 1
            print(f"  {p.name:<30s} -> nada detectado")
        else:
            partes = []
            for box, conf, c in dets:
                nome = CLASSES[c] if c < len(CLASSES) else f"cls{c}"
                contagem[nome] = contagem.get(nome, 0) + 1
                partes.append(f"{nome}({conf:.2f})")
                x1, y1, x2, y2 = box.astype(int)
                cor = CORES[c] if c < len(CORES) else (255, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(img, f"{nome} {conf:.2f}", (x1, max(15, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
            print(f"  {p.name:<30s} -> {', '.join(partes)}")

        cv2.imwrite(str(Path(args.out) / p.name), img)

    print(f"\n  Sem detecção nenhuma: {n_sem_deteccao}/{len(imgs)}")
    print(f"  Contagem por classe: {contagem}")
    print(f"\n  [OK] imagens com as caixas salvas em ./{args.out}/")

if __name__ == "__main__":
    main()