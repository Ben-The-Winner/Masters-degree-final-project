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

        avg_level = features['contract_level'].mean()
        declarer_ratio = features['is_declarer'].mean()

        self.weights = {
            'avg_level': float(avg_level),
            'declarer_ratio': float(declarer_ratio),
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
        ml_insights = twin.generate_ml_insights(df_boards)

        if not isinstance(ml_insights, dict):
            return ""

        qa_list = ml_insights.get('qa') or []
        if not qa_list:
            return ""

        qa_html = ""
        for i, item in enumerate(qa_list, 1):
            q = item.get('q', '')
            a = item.get('a', '')
            qa_html += f"""
            <div style="margin-bottom: 18px; padding: 14px 16px; background: #313244; border-radius: 8px; border-left: 4px solid #89b4fa;">
                <div style="color: #89b4fa; font-weight: 600; font-size: 1.02em; margin-bottom: 8px;">
                    Q{i}. {q}
                </div>
                <div style="color: #cdd6f4; font-size: 0.95em; line-height: 1.65;">
                    {a}
                </div>
            </div>
            """

        ml_html_block = f"""
        <div style="background-color: #1e1e2e; color: #cdd6f4; padding: 22px; border-radius: 10px; margin-top: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; border: 1px solid #45475a; text-align: left; direction: ltr;">
            <h3 style="color: #89b4fa; margin-top: 0; font-size: 1.3em;">🎯 Coaching Insights vs Double Dummy</h3>
            <p style="font-size: 0.95em; color: #bac2de; margin-bottom: 18px;">
                Answers based on your actual results compared to double-dummy optimal play (DDS).
                Positive gap = tricks lost relative to perfect double-dummy play.
            </p>
            {qa_html}
            <p style="font-size: 0.8em; color: #6c7086; margin-top: 12px; border-top: 1px solid #313244; padding-top: 10px;">
                Analysis uses only boards where both your result and a DDS baseline are available.
            </p>
        </div>
        """
        return ml_html_block
