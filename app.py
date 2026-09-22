from flask import Flask, request, render_template_string
from backend import run_analysis_for_player
from trainer import BridgePerformanceTrainer
import os
import time
import re

app = Flask(__name__)

form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bridge Performance Analyzer</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: linear-gradient(165deg, #11111b 0%, #1e1e2e 45%, #181825 100%);
      color: #cdd6f4;
      display: flex; align-items: center; justify-content: center;
      padding: 24px;
    }
    .card {
      width: min(440px, 100%);
      background: #1e1e2e;
      border: 1px solid #313244;
      border-radius: 16px;
      padding: 36px 32px 28px;
      box-shadow: 0 16px 48px rgba(0,0,0,0.4);
      text-align: center;
    }
    .logo {
      width: 56px; height: 56px; margin: 0 auto 18px;
      border-radius: 14px;
      background: linear-gradient(135deg, #89b4fa, #cba6f7);
      display: flex; align-items: center; justify-content: center;
      font-size: 1.6em;
    }
    h1 {
      margin: 0 0 8px; font-size: 1.45em; font-weight: 700;
      color: #cdd6f4; letter-spacing: -0.02em;
    }
    .subtitle { color: #6c7086; font-size: 0.9em; margin-bottom: 28px; }
    label {
      display: block; text-align: left; color: #a6adc8;
      font-size: 0.85em; margin-bottom: 8px; font-weight: 500;
    }
    input[type="text"] {
      width: 100%; padding: 12px 14px; font-size: 1em;
      background: #313244; border: 1px solid #45475a; border-radius: 10px;
      color: #cdd6f4; outline: none; margin-bottom: 16px;
    }
    input[type="text"]:focus { border-color: #89b4fa; }
    input[type="text"]::placeholder { color: #585b70; }
    button[type="submit"] {
      width: 100%; padding: 12px 16px; font-size: 1em; font-weight: 600;
      border: none; border-radius: 10px; cursor: pointer;
      background: linear-gradient(135deg, #89b4fa, #cba6f7);
      color: #1e1e2e;
    }
    button[type="submit"]:hover { filter: brightness(1.08); }
    button[type="submit"]:disabled { opacity: 0.7; cursor: wait; }
    #status { margin-top: 18px; color: #a6adc8; font-size: 0.88em; min-height: 1.4em; }
    #status.loading { color: #f9e2af; }
    .spinner {
      display: inline-block; width: 14px; height: 14px;
      border: 2px solid #45475a; border-top-color: #89b4fa;
      border-radius: 50%; animation: spin 0.7s linear infinite;
      vertical-align: middle; margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .hint { color: #585b70; font-size: 0.8em; margin-top: 14px; line-height: 1.4; }
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🃏</div>
    <h1>Bridge Performance Analyzer</h1>
    <p class="subtitle">Player stats &amp; coaching vs double-dummy</p>
    <form action="/analyze" method="post" id="mainForm" onsubmit="onSubmit()">
      <label for="player">Player name or IBFN number</label>
      <input type="text" id="player" name="player"
             placeholder="e.g. 19137 · שפי רון · רון שפי" required autofocus>
      <button type="submit" id="btn">Run Analysis</button>
    </form>
    <div id="status"></div>
    <p class="hint">Name order flexible: last-first or first-last both work.</p>
  </div>
  <script>
    function onSubmit() {
      var btn = document.getElementById('btn');
      var st = document.getElementById('status');
      btn.disabled = true;
      btn.textContent = 'Analyzing…';
      st.className = 'loading';
      st.innerHTML = '<span class="spinner"></span>Working — this may take a few minutes. Keep this tab open.';
    }
  </script>
</body>
</html>
"""

REPORT_TOPBAR = """
<div style="
  position: sticky; top: 0; z-index: 100;
  background: rgba(30,30,46,0.95); backdrop-filter: blur(8px);
  border-bottom: 1px solid #313244;
  padding: 12px 16px; margin: -24px -16px 24px;
">
  <form action="/analyze" method="post" style="
    max-width: 960px; margin: 0 auto;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: center;
  " onsubmit="this.querySelector('button').disabled=true; this.querySelector('button').textContent='Analyzing…';">
    <a href="/" style="color:#89b4fa;text-decoration:none;font-weight:600;margin-right:8px;">← Home</a>
    <span style="color:#6c7086;font-size:0.9em;">Analyze another player:</span>
    <input type="text" name="player" required placeholder="Name or IBFN (any order)"
      style="padding:8px 12px;border-radius:8px;border:1px solid #45475a;background:#313244;color:#cdd6f4;min-width:180px;">
    <button type="submit" style="
      padding:8px 16px;border:none;border-radius:8px;font-weight:600;cursor:pointer;
      background:linear-gradient(135deg,#89b4fa,#cba6f7);color:#1e1e2e;
    ">Run Analysis</button>
  </form>
</div>
"""


@app.route("/", methods=["GET"])
def index():
    return form_html


@app.route("/analyze", methods=["POST"])
def analyze():
    player = request.form.get("player", "").strip()
    if not player:
        return render_template_string(
            form_html.replace(
                '<div id="status"></div>',
                '<div id="status" style="color:#f38ba8">Please enter a player name or number.</div>',
            )
        )

    print(f"Starting analysis for {player}...")
    out_fname = run_analysis_for_player(player)
    time.sleep(0.5)

    if os.path.exists(out_fname):
        with open(out_fname, "r", encoding="utf-8") as f:
            report_html = f.read()
        if "<body>" in report_html:
            report_html = report_html.replace("<body>", "<body>" + REPORT_TOPBAR, 1)
        else:
            report_html = re.sub(r"(<body[^>]*>)", r"\1" + REPORT_TOPBAR, report_html, count=1)
        return report_html

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Error</title>
    <style>body{{font-family:system-ui;background:#1e1e2e;color:#cdd6f4;padding:40px;text-align:center}}</style>
    </head><body>
    <h2>Analysis finished, but no report file was found for {player}.</h2>
    <p><a href="/" style="color:#89b4fa">Back to home</a></p>
    </body></html>"""


@app.route("/train", methods=["POST"])
def trigger_training():
    trainer = BridgePerformanceTrainer()
    weights = getattr(trainer, "load_model", lambda: None)()
    if weights:
        return f"<h2>Model loaded successfully!</h2><p>{weights}</p>"
    return "<h2>No trained model found. Run analysis first.</h2>"


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
