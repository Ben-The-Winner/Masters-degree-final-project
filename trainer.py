import pandas as pd
import pickle
from datetime import datetime
from ml_trainer import BridgeDigitalTwin


class BridgePerformanceTrainer:
    def __init__(self, model_path="trained_model.pkl"):
        self.model_path = model_path
        self.weights = {}

    def prepare_features(self, df_boards):
        if df_boards is None or not isinstance(df_boards, pd.DataFrame) or df_boards.empty:
            return pd.DataFrame()
        df = df_boards.copy()
        is_defense_clean = df.get('is_defense', False).fillna(False).astype(bool)
        df['is_declarer'] = (~is_defense_clean).astype(int)
        df['contract_level'] = pd.to_numeric(df.get('contract_level', 0), errors='coerce').fillna(0)
        return df[['contract_level', 'is_declarer']]

    def train(self, df_boards):
        print("Starting Model Training...")
        features = self.prepare_features(df_boards)
        if features.empty:
            print("No features to train on.")
            return
        self.weights = {
            'avg_level': float(features['contract_level'].mean()),
            'declarer_ratio': float(features['is_declarer'].mean()),
            'trained_at': datetime.now().isoformat(),
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"Training completed successfully. Weights saved: {self.weights}")

    def generate_html_summary(self, df_boards=None):
        if df_boards is None or not isinstance(df_boards, pd.DataFrame) or df_boards.empty:
            print("Warning: df_boards is missing or empty in generate_html_summary.")
            return ""

        twin = BridgeDigitalTwin()
        twin.train(df_boards)
        data = twin.generate_ml_insights(df_boards)

        if not isinstance(data, dict) or not data.get('ok'):
            return """
            <div style="background:#1e1e2e;color:#cdd6f4;padding:24px;border-radius:12px;margin-top:24px;
                        font-family:'Segoe UI',system-ui,sans-serif;border:1px solid #45475a;">
              <p>Not enough boards with DDS data for coaching insights yet.</p>
            </div>
            """

        avg_gap = data['avg_gap']
        under_rate = data['under_rate'] * 100
        n, n_under = data['n'], data['n_under']
        avg_gap_under = data['avg_gap_under']
        tone, label = data['proximity_tone'], data['proximity_label']
        tone_color = {'good': '#a6e3a1', 'ok': '#f9e2af', 'warn': '#f38ba8'}.get(tone, '#89b4fa')
        gap_color = '#a6e3a1' if avg_gap <= 0 else '#f38ba8'

        def_rate, dec_rate = data['def_rate'], data['dec_rate']
        def_pct = f"{def_rate*100:.0f}" if def_rate is not None else "—"
        dec_pct = f"{dec_rate*100:.0f}" if dec_rate is not None else "—"
        def_bar = int(round((def_rate or 0) * 100))
        dec_bar = int(round((dec_rate or 0) * 100))
        def_avg_s = f"{data['def_avg']:+.2f}" if data['def_avg'] is not None else "—"
        dec_avg_s = f"{data['dec_avg']:+.2f}" if data['dec_avg'] is not None else "—"

        if def_rate is not None and dec_rate is not None:
            if def_rate > dec_rate + 0.03:
                role_verdict = "You lose relatively more on <strong style='color:#f38ba8'>defense</strong>."
            elif dec_rate > def_rate + 0.03:
                role_verdict = "You lose relatively more as <strong style='color:#f38ba8'>declarer</strong>."
            else:
                role_verdict = "Losses are <strong style='color:#a6e3a1'>balanced</strong> between roles."
        else:
            role_verdict = "Not enough boards in both roles to compare."

        high_rate, high_n, high_avg = data['high_rate'], data['high_n'], data['high_avg']
        if high_rate is not None:
            high_block = (
                f"<div style='font-size:1.6em;font-weight:700;color:#cba6f7'>{high_rate*100:.0f}%</div>"
                f"<div style='color:#a6adc8;font-size:0.85em'>under-performance · {high_n} boards</div>"
                f"<div style='margin-top:6px;color:#bac2de'>avg gap <strong>{high_avg:+.2f}</strong></div>"
            )
        else:
            high_block = "<div style='color:#6c7086'>Not enough game/slam boards</div>"

        case = data.get('case')
        if case:
            case_html = f"""
            <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:center;">
              <div style="background:#45475a;border-radius:10px;padding:14px 18px;min-width:120px;text-align:center;">
                <div style="color:#a6adc8;font-size:0.75em;text-transform:uppercase;letter-spacing:0.05em;">Board</div>
                <div style="font-size:1.5em;font-weight:700;color:#89b4fa;">#{case['board']}</div>
              </div>
              <div style="flex:1;min-width:200px;">
                <div style="color:#cdd6f4;font-size:1.05em;margin-bottom:6px;">
                  As <strong style="color:#f9e2af">{case['role']}</strong> in
                  <strong style="color:#89b4fa">{case['contract']}</strong>
                </div>
                <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px;">
                  <div>
                    <span style="color:#a6adc8;font-size:0.8em;">You took</span><br>
                    <span style="font-size:1.4em;font-weight:700;color:#f38ba8">{case['actual']}</span>
                  </div>
                  <div style="color:#585b70;font-size:1.4em;align-self:center;">→</div>
                  <div>
                    <span style="color:#a6adc8;font-size:0.8em;">DDS optimal</span><br>
                    <span style="font-size:1.4em;font-weight:700;color:#a6e3a1">{case['dds']}</span>
                  </div>
                  <div style="background:#f38ba820;border:1px solid #f38ba850;border-radius:8px;padding:8px 14px;">
                    <span style="color:#a6adc8;font-size:0.8em;">Loss</span><br>
                    <span style="font-size:1.3em;font-weight:700;color:#f38ba8">{case['gap']} tricks</span>
                  </div>
                </div>
              </div>
            </div>
            """
        else:
            case_html = "<div style='color:#6c7086'>No costly board found.</div>"

        tips_html = ""
        tip_colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#cba6f7']
        for i, tip in enumerate(data.get('tips') or []):
            c = tip_colors[i % len(tip_colors)]
            tips_html += f"""
            <div style="background:#313244;border-radius:10px;padding:14px 16px;border-left:4px solid {c};margin-bottom:10px;">
              <div style="color:{c};font-weight:600;font-size:0.95em;margin-bottom:4px;">{tip['title']}</div>
              <div style="color:#bac2de;font-size:0.9em;line-height:1.55;">{tip['text']}</div>
            </div>
            """

        under_bar = min(100, int(round(under_rate)))

        return f"""
<div style="
  background: linear-gradient(160deg, #1e1e2e 0%, #181825 100%);
  color: #cdd6f4; padding: 28px 24px 24px; border-radius: 16px; margin-top: 28px;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  border: 1px solid #313244; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
  text-align: left; direction: ltr;
">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
    <div style="width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,#89b4fa,#cba6f7);
                display:flex;align-items:center;justify-content:center;font-size:1.3em;">🎯</div>
    <div>
      <h2 style="margin:0;color:#cdd6f4;font-size:1.35em;font-weight:700;">Coaching Insights vs Double Dummy</h2>
      <div style="color:#6c7086;font-size:0.82em;margin-top:2px;">
        Your results compared to perfect double-dummy play · {n} boards analyzed
      </div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0 26px;">
    <div style="background:#313244;border-radius:12px;padding:16px;text-align:center;border:1px solid #45475a;">
      <div style="color:#a6adc8;font-size:0.72em;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Avg gap vs DDS</div>
      <div style="font-size:1.75em;font-weight:800;color:{gap_color};">{avg_gap:+.2f}</div>
      <div style="color:#6c7086;font-size:0.78em;margin-top:4px;">tricks (neg = better)</div>
    </div>
    <div style="background:#313244;border-radius:12px;padding:16px;text-align:center;border:1px solid #45475a;">
      <div style="color:#a6adc8;font-size:0.72em;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Under-performance</div>
      <div style="font-size:1.75em;font-weight:800;color:#f9e2af;">{under_rate:.0f}%</div>
      <div style="background:#45475a;border-radius:4px;height:6px;margin-top:8px;overflow:hidden;">
        <div style="background:#f9e2af;height:100%;width:{under_bar}%;border-radius:4px;"></div>
      </div>
      <div style="color:#6c7086;font-size:0.78em;margin-top:4px;">{n_under}/{n} boards</div>
    </div>
    <div style="background:#313244;border-radius:12px;padding:16px;text-align:center;border:1px solid #45475a;">
      <div style="color:#a6adc8;font-size:0.72em;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">When you miss</div>
      <div style="font-size:1.75em;font-weight:800;color:#f38ba8;">{avg_gap_under:.2f}</div>
      <div style="color:#6c7086;font-size:0.78em;margin-top:4px;">avg tricks lost</div>
    </div>
    <div style="background:#313244;border-radius:12px;padding:16px;text-align:center;border:1px solid #45475a;">
      <div style="color:#a6adc8;font-size:0.72em;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Overall verdict</div>
      <div style="font-size:1.15em;font-weight:700;color:{tone_color};margin-top:8px;">{label}</div>
    </div>
  </div>

  <div style="background:#313244;border-radius:12px;padding:18px 20px;margin-bottom:14px;border-left:4px solid {tone_color};">
    <div style="color:#89b4fa;font-weight:600;font-size:0.95em;margin-bottom:8px;">Q1 · Am I close to double-dummy (DDS)?</div>
    <div style="color:#cdd6f4;font-size:0.95em;line-height:1.6;">{data['proximity_text']}</div>
  </div>

  <div style="background:#313244;border-radius:12px;padding:18px 20px;margin-bottom:14px;border-left:4px solid #89b4fa;">
    <div style="color:#89b4fa;font-weight:600;font-size:0.95em;margin-bottom:10px;">Q2 · Where do I lose more — declarer or defense?</div>
    <div style="color:#cdd6f4;font-size:0.95em;margin-bottom:14px;">{role_verdict}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
      <div style="background:#1e1e2e;border-radius:10px;padding:14px;">
        <div style="color:#a6adc8;font-size:0.8em;margin-bottom:6px;">🛡️ Defense</div>
        <div style="font-size:1.4em;font-weight:700;color:#89dceb;">{def_pct}%</div>
        <div style="background:#45475a;border-radius:4px;height:6px;margin:8px 0;overflow:hidden;">
          <div style="background:#89dceb;height:100%;width:{def_bar}%;border-radius:4px;"></div>
        </div>
        <div style="color:#6c7086;font-size:0.8em;">{data['def_under_n']}/{data['def_n']} boards · avg {def_avg_s}</div>
      </div>
      <div style="background:#1e1e2e;border-radius:10px;padding:14px;">
        <div style="color:#a6adc8;font-size:0.8em;margin-bottom:6px;">🃏 Declarer</div>
        <div style="font-size:1.4em;font-weight:700;color:#f5c2e7;">{dec_pct}%</div>
        <div style="background:#45475a;border-radius:4px;height:6px;margin:8px 0;overflow:hidden;">
          <div style="background:#f5c2e7;height:100%;width:{dec_bar}%;border-radius:4px;"></div>
        </div>
        <div style="color:#6c7086;font-size:0.8em;">{data['dec_under_n']}/{data['dec_n']} boards · avg {dec_avg_s}</div>
      </div>
    </div>
  </div>

  <div style="background:#313244;border-radius:12px;padding:18px 20px;margin-bottom:14px;border-left:4px solid #cba6f7;">
    <div style="color:#89b4fa;font-weight:600;font-size:0.95em;margin-bottom:10px;">Q3 · Are high-level contracts (game/slam) a problem?</div>
    {high_block}
  </div>

  <div style="background:#313244;border-radius:12px;padding:18px 20px;margin-bottom:14px;border-left:4px solid #f38ba8;">
    <div style="color:#89b4fa;font-weight:600;font-size:0.95em;margin-bottom:12px;">Q4 · Concrete example of a costly board</div>
    {case_html}
  </div>

  <div style="background:#313244;border-radius:12px;padding:18px 20px;margin-bottom:8px;border-left:4px solid #a6e3a1;">
    <div style="color:#89b4fa;font-weight:600;font-size:0.95em;margin-bottom:12px;">Q5 · What should I focus on to improve?</div>
    {tips_html}
  </div>

  <p style="font-size:0.75em;color:#585b70;margin:16px 0 0;border-top:1px solid #313244;padding-top:12px;">
    Positive trick gap = tricks lost vs optimal double-dummy play. Analysis uses only boards with both your result and a DDS baseline.
  </p>
</div>
"""
