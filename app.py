from flask import Flask, request, render_template_string
from backend import run_analysis_for_player  # import your analysis function
from trainer import BridgePerformanceTrainer
import os
import time


app = Flask(__name__)

# === Simple HTML template for the input form ===
form_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Bridge Player Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; text-align: center; }
        h1 { color: #2c3e50; }
        form { margin-top: 30px; }
        input, button { padding: 8px 12px; font-size: 16px; margin-top: 10px; }
        #status { margin-top: 20px; color: #555; font-style: italic; }
    </style>
</head>
<body>
    <h1>Bridge Performance Analyzer</h1>
    <form action="/analyze" method="post">
        <label><b>Enter player name or number:</b></label><br>
        <input type="text" name="player" placeholder="e.g., נצר זיידנברג or 10359" required>
        <br><button type="submit">Run Analysis</button>
    </form>
    <p><i>Note: The analysis may take about 14 minutes. Please wait after submitting.</i></p>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    """Show the input form."""
    return form_html


@app.route("/analyze", methods=["POST"])
def analyze():
    """Run the analysis and show the resulting HTML."""
    player = request.form["player"].strip()

    # Optionally show progress in console
    print(f"Starting analysis for {player}...")

    # Run your heavy bridge analysis (the ~14 minute function)
    out_fname = run_analysis_for_player(player)

    # Wait for file creation (if your function writes asynchronously)
    time.sleep(1)

    # Load the generated HTML report and return it to the browser
    if os.path.exists(out_fname):
        with open(out_fname, "r", encoding="utf-8") as f:
            report_html = f.read()
        return report_html
    else:
        return f"<h2>Analysis complete, but no report found for {player}.</h2>"


@app.route("/train", methods=["POST"])
def trigger_training():
    """Optional endpoint to check trained model status."""
    trainer = BridgePerformanceTrainer()
    weights = trainer.load_model()
    if weights:
        return f"<h2>Model loaded successfully!</h2><p>{weights}</p>"
    return "<h2>No trained model found. Run analysis first.</h2>"


if __name__ == "__main__":
    # Launch the local web server
    app.run(debug=True, use_reloader=False)