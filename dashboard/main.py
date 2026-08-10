"""
================================================================
  171 GARAGE — SELEÇÃO DE DESTINO
  ─────────────────────────────────────────────────────────────
  Um QR por trajeto. O carro gera um token de sessão quando um
  novo trajeto começa e mostra o MESMO QR em todas as suas
  paradas. Este servidor aceita UMA seleção por token: depois de
  despachado, reescanear o mesmo QR só informa o destino já
  escolhido, sem republicar.

  Rotas:
    GET  /select?car=171&s=<token>   tela de seleção
    POST /api/destino                publica o destino no MQTT
    POST /api/telemetria             recebe eventos do carro (só log)
    GET  /health                     status do app e do broker
================================================================
"""

import os, json, time
from flask import Flask, render_template_string, request, jsonify, redirect

# ── CONFIG (variáveis de ambiente no Railway) ──────────────────
MQTT_BROKER = os.environ.get("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT   = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC  = os.environ.get("MQTT_TOPIC", "171garage/carro/destino")
MQTT_USER   = os.environ.get("MQTT_USER") or None
MQTT_PASS   = os.environ.get("MQTT_PASS") or None
EQUIPE      = os.environ.get("EQUIPE", "171 Garage")

PONTOS = {"A": "Ponto A", "B": "Ponto B", "C": "Ponto C"}

# Tokens já despachados: {token: destino}. Some quando o app
# reinicia — aceitável, cada trajeto dura minutos.
USADOS = {}

PAGINA = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#141414">
<title>{{ equipe }} — Destino do pacote</title>
<style>
:root{
  --bg:#141414; --card:#1E1E1E; --linha:#333;
  --texto:#F2F2F2; --fraco:#8A8A8A;
  --verde:#22A45D; --vermelho:#C4341C;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{
  background:var(--bg); color:var(--texto);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  min-height:100svh; display:flex; align-items:center; justify-content:center;
  padding:26px 20px calc(26px + env(safe-area-inset-bottom));
}
.wrap{width:100%;max-width:380px}

.equipe{
  font-size:12px; letter-spacing:.22em; text-transform:uppercase;
  color:var(--fraco); text-align:center; padding-bottom:16px;
  border-bottom:1px solid var(--linha);
}
h1{
  font-size:26px; font-weight:600; line-height:1.25;
  text-align:center; margin:30px 0 26px;
}

.btn{
  display:flex; align-items:center; gap:16px; width:100%;
  background:var(--card); color:var(--texto);
  border:1px solid var(--linha); border-radius:12px;
  padding:20px; margin-bottom:12px;
  font:inherit; cursor:pointer; text-align:left;
  transition:background .15s, border-color .15s;
}
.btn:active{background:#262626}
.btn:focus-visible{outline:2px solid var(--verde); outline-offset:2px}
.letra{
  font-size:26px; font-weight:700; width:46px; height:46px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  border:1px solid var(--linha); border-radius:10px;
}
.nome{font-size:17px; font-weight:500}

/* selecionado: fica verde */
.btn.ok{background:var(--verde); border-color:var(--verde); color:#fff}
.btn.ok .letra{border-color:rgba(255,255,255,.5); background:rgba(255,255,255,.14)}
.btn:disabled{cursor:default}
.btn:disabled:not(.ok){opacity:.32}

.msg{
  margin-top:20px; text-align:center; font-size:14px; line-height:1.5;
  color:var(--fraco); min-height:21px;
}
.msg.erro{color:var(--vermelho)}
.msg.bom{color:var(--verde)}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<div class="wrap">

  <div class="equipe">{{ equipe }}</div>

  <h1>Para onde vai o pacote?</h1>

  {% for letra, nome in pontos.items() %}
  <button class="btn" data-destino="{{ letra }}">
    <span class="letra">{{ letra }}</span>
    <span class="nome">{{ nome }}</span>
  </button>
  {% endfor %}

  <div class="msg" id="msg"></div>
</div>

<script>
const TOKEN = "{{ token }}";
const JA    = "{{ ja_usado or '' }}";
const msg   = document.getElementById('msg');
const btns  = document.querySelectorAll('.btn');

function confirmar(destino){
  btns.forEach(b => {
    b.disabled = true;
    if (b.dataset.destino === destino) b.classList.add('ok');
  });
}

// QR já usado: mostra o destino que foi despachado
if (JA) {
  confirmar(JA);
  msg.textContent = 'Pacote já despachado para o ponto ' + JA + '.';
  msg.className = 'msg bom';
}

btns.forEach(btn => btn.addEventListener('click', async () => {
  const destino = btn.dataset.destino;
  btns.forEach(b => b.disabled = true);
  msg.textContent = 'Enviando…';
  msg.className = 'msg';

  try{
    const r = await fetch('/api/destino', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({destino, token: TOKEN})
    });
    const d = await r.json();
    if(!d.ok) throw new Error(d.erro || 'Não foi possível enviar.');

    confirmar(destino);
    msg.textContent = 'Pacote a caminho do ponto ' + destino + '.';
    msg.className = 'msg bom';

  }catch(e){
    btns.forEach(b => { b.disabled = false; b.classList.remove('ok'); });
    msg.textContent = e.message;
    msg.className = 'msg erro';
  }
}));
</script>
</body>
</html>
"""

app = Flask(__name__)

@app.errorhandler(500)
def erro_interno(e):
    """Sem isto, um 500 no Railway aparece sem causa nenhuma no log."""
    import traceback
    traceback.print_exc()
    return "Erro interno. Veja o traceback em: railway logs", 500

# ── MQTT ───────────────────────────────────────────────────────
_mqtt = {"cli": None, "ok": False}

def _iniciar_mqtt():
    try:
        import paho.mqtt.client as mqtt
        cli = mqtt.Client(client_id=f"171-select-{int(time.time())}")
        if MQTT_USER:
            cli.username_pw_set(MQTT_USER, MQTT_PASS)

        def on_connect(c, u, f, rc):
            _mqtt["ok"] = (rc == 0)
            print(f"[MQTT] conectado={_mqtt['ok']} {MQTT_BROKER}:{MQTT_PORT}", flush=True)

        def on_disconnect(c, u, rc):
            _mqtt["ok"] = False

        cli.on_connect, cli.on_disconnect = on_connect, on_disconnect
        cli.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=30)
        cli.loop_start()
        _mqtt["cli"] = cli
    except Exception as e:
        print(f"[MQTT][ERRO] {e}", flush=True)

_iniciar_mqtt()

# ── ROTAS ──────────────────────────────────────────────────────

@app.route("/")
def raiz():
    # Sem token: quem entra pela raiz cai na tela de seleção mesmo assim.
    # Sem token o servidor não consegue impedir uma segunda seleção, então
    # o caminho correto continua sendo escanear o QR do carro.
    return redirect("/select", code=302)


@app.route("/select")
def select():
    token = request.args.get("s", "")
    return render_template_string(PAGINA, equipe=EQUIPE, pontos=PONTOS,
                           token=token, ja_usado=USADOS.get(token))


@app.route("/api/destino", methods=["POST"])
def api_destino():
    dados   = request.get_json(silent=True) or {}
    destino = str(dados.get("destino", "")).strip().upper()[:1]
    token   = str(dados.get("token", "")).strip()

    if destino not in PONTOS:
        return jsonify(ok=False, erro="Destino inválido."), 400

    # Uma requisição por QR gerado.
    if token and token in USADOS:
        return jsonify(ok=False,
                       erro=f"Este QR já despachou o pacote para o ponto {USADOS[token]}."), 409

    cli = _mqtt["cli"]
    if cli is None or not _mqtt["ok"]:
        return jsonify(ok=False,
                       erro="Sem conexão com o veículo. Tente de novo."), 503

    try:
        # retain=False: com retain=True o carro receberia este mesmo
        # pedido de novo a cada reconexão ao broker.
        info = cli.publish(MQTT_TOPIC, json.dumps({"destino": destino}),
                           qos=1, retain=False)
        info.wait_for_publish(timeout=3)
    except Exception as e:
        return jsonify(ok=False, erro=f"Falha ao enviar: {e}"), 502

    if token:
        USADOS[token] = destino
    print(f"[PUB] {MQTT_TOPIC} ← {destino} (token={token or '-'})", flush=True)
    return jsonify(ok=True, destino=destino)


@app.route("/api/telemetria", methods=["POST"])
def api_telemetria():
    """O carro posta aqui a cada transição. Só registra no log."""
    print(f"[TELEM] {request.get_json(silent=True)}", flush=True)
    return jsonify(ok=True)


@app.route("/health")
def health():
    return jsonify(ok=True, mqtt=_mqtt["ok"],
                   broker=f"{MQTT_BROKER}:{MQTT_PORT}", topico=MQTT_TOPIC)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)