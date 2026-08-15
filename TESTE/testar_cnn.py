"""
================================================================
  TESTE RÁPIDO — CNN DE CONFIRMAÇÃO
  171 Garage · roda o .tflite treinado em cima de imagens novas
================================================================
  Objetivo: ver se o modelo generaliza pra fora do vídeo de treino,
  ANTES de confiar nele no carro. Usa o mesmo prep_mono() do treino
  e do carro — se aqui já falhar, no carro também vai falhar.

  USO:
      python testar_cnn.py --imgs caminho/para/imagens/novas
      python testar_cnn.py --imgs caminho/para/imagens/novas --out revisao.jpg

  Aceita tanto uma pasta flat de imagens quanto uma pasta com
  subpastas por classe esperada (aí ele também calcula acerto/erro).
================================================================
"""
import os, sys, json, argparse
import numpy as np
import cv2

CLASSES  = ["Semaforo", "Stop", "Fundo"]
CNN_SIZE = 96
MODEL    = "./models/sign_classifier.tflite"
OOD_FILE = "./models/ood_thresholds.json"
EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def prep_mono(crop):
    """IDÊNTICO ao prep_mono() do treino e do carro. Não editar um sem os outros."""
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = _CLAHE.apply(cv2.resize(crop, (CNN_SIZE, CNN_SIZE)))
    return np.stack([g, g, g], axis=-1).astype(np.float32) / 255.0

def carregar_interpretador(model_path):
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path)

def prever(interp, in_idx, out_idx, img):
    x = prep_mono(img)[None, ...]
    interp.set_tensor(in_idx, x)
    interp.invoke()
    probs = interp.get_tensor(out_idx)[0]
    cid = int(np.argmax(probs))
    return CLASSES[cid], float(probs[cid]), probs

def listar_imagens(pasta):
    out = []
    for raiz, _, arqs in os.walk(pasta):
        for a in arqs:
            if os.path.splitext(a)[1].lower() in EXTS:
                out.append(os.path.join(raiz, a))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgs", required=True, help="pasta com imagens novas (flat ou com subpastas Semaforo/Stop/Fundo)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ood", default=OOD_FILE)
    ap.add_argument("--out", default=None, help="caminho pra salvar um contact-sheet de revisão (opcional)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERRO] modelo não encontrado: {args.model}"); sys.exit(1)

    thr = {}
    if os.path.exists(args.ood):
        thr = json.load(open(args.ood))
        print(f"  OOD carregado: {thr}")
    else:
        print("  [AVISO] ood_thresholds.json não encontrado — rodando sem corte de confiança")

    interp = carregar_interpretador(args.model)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    imgs = listar_imagens(args.imgs)
    if not imgs:
        print(f"[ERRO] nenhuma imagem em {args.imgs}"); sys.exit(1)
    print(f"\n  Testando {len(imgs)} imagens de {args.imgs}\n")

    # se a pasta tem subpastas com nome de classe, usa como rótulo esperado p/ comparar
    tem_rotulo = any(os.path.basename(os.path.dirname(p)) in CLASSES for p in imgs)

    acertos = erros = rejeitados = 0
    linhas_erro = []
    resultados_visual = []

    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            print(f"  [pular] não consegui ler {p}"); continue
        pred, conf, probs = prever(interp, in_idx, out_idx, img)

        limite = thr.get(pred, 0.0)
        rejeitado = conf < limite
        if rejeitado:
            rejeitados += 1

        esperado = os.path.basename(os.path.dirname(p))
        marca = ""
        if tem_rotulo and esperado in CLASSES:
            if esperado == pred and not rejeitado:
                acertos += 1
                marca = "OK"
            else:
                erros += 1
                marca = "ERRO"
                linhas_erro.append(f"    {os.path.basename(p):<30s} esperado={esperado:<9s} previsto={pred:<9s} conf={conf:.3f}{'  [REJEITADO p/ OOD]' if rejeitado else ''}")

        tag = "REJEITADO" if rejeitado else pred
        print(f"  {os.path.basename(p):<32s} -> {tag:<12s} conf={conf:.3f}  {marca}")
        resultados_visual.append((p, pred, conf, rejeitado, esperado if tem_rotulo else None))

    print(f"\n  Rejeitados por OOD (baixa confiança): {rejeitados}/{len(imgs)}")
    if tem_rotulo:
        total = acertos + erros
        print(f"\n  Acurácia nas imagens rotuladas: {acertos}/{total} ({100*acertos/max(1,total):.1f}%)")
        if linhas_erro:
            print("\n  Casos errados:")
            for l in linhas_erro:
                print(l)

    if args.out:
        salvar_contact_sheet(resultados_visual, args.out)
        print(f"\n  [OK] revisão visual salva em {args.out}")

def salvar_contact_sheet(resultados, out_path, cel=110, cols=8):
    n = len(resultados)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows*cel, cols*cel, 3), 40, np.uint8)
    for i, (p, pred, conf, rej, esperado) in enumerate(resultados):
        img = cv2.imread(p)
        if img is None: continue
        img = cv2.resize(img, (cel-10, cel-30))
        r, c = divmod(i, cols)
        y0, x0 = r*cel+5, c*cel+5
        sheet[y0:y0+img.shape[0], x0:x0+img.shape[1]] = img
        cor = (0,0,255) if rej else ((0,255,0) if (esperado is None or esperado==pred) else (0,140,255))
        texto = f"{pred[:4]} {conf:.2f}"
        cv2.putText(sheet, texto, (x0, y0+img.shape[0]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cor, 1)
    cv2.imwrite(out_path, sheet)

if __name__ == "__main__":
    main()