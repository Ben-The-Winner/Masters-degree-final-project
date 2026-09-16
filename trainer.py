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
            'trained_at': datetime.now().isoformat()
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

        if isinstance(ml_insights, dict):
            patterns_html = ""
            for pattern in ml_insights.get('patterns', []):
                patterns_html += f"<li style='margin-bottom: 6px;'>{pattern}</li>"

            ml_html_block = f"""
            <div style="background-color: #1e1e2e; color: #cdd6f4; padding: 20px; border-radius: 10px; margin-top: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; border: 1px solid #45475a; text-align: left; direction: ltr;">
                <h3 style="color: #89b4fa; margin-top: 0; font-size: 1.25em;">🤖 Digital Twin & Behavioral ML Insights</h3>
                <p style="font-size: 1.0em; margin-bottom: 12px;"><strong>The Machine Learning model analyzed your gameplay against Double Dummy Solver (DDS) optimal baselines:</strong></p>
                
                <ul style="line-height: 1.8; font-size: 0.95em; margin-bottom: 15px;">
                    <li><strong>Model Accuracy Score:</strong> <span style="color: #a6e3a1; font-weight: bold;">{ml_insights['accuracy']}</span></li>
                    <li><strong>Primary Risk Predictor:</strong> <span style="color: #f38ba8; font-weight: bold;">{ml_insights['top_feature']}</span></li>
                </ul>

                <h4 style="color: #f9e2af; margin-bottom: 8px; font-size: 1.05em;">📊 Detected Behavioral Patterns & Outliers:</h4>
                <ul style="line-height: 1.7; color: #bac2de; font-size: 0.95em;">
                    {patterns_html if patterns_html else "<li>No critical systematic anomalies detected in the current board set.</li>"}
                </ul>

                <p style="font-size: 0.85em; color: #a6adc8; margin-top: 15px; border-top: 1px solid #313244; padding-top: 10px;">
                    <em>* The Digital Twin benchmarks actual trick results against DDS analysis to identify systematic decision-making biases.</em>
                </p>
            </div>
            """
            return ml_html_block
        
        return ""