import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap

def calculate_vulnerability(board_num, declarer_pos):
    """Calculates vulnerability based on standard bridge 16-board rotation"""
    try:
        b_num = int(board_num)
    except (ValueError, TypeError):
        return 0

    idx = (b_num - 1) % 16 + 1
    ns_vul = idx in [2, 5, 12, 15, 4, 7, 10, 13]
    ew_vul = idx in [3, 6, 9, 16, 4, 7, 10, 13]

    decl = str(declarer_pos).upper()
    if decl in ['N', 'S']:
        return 1 if ns_vul else 0
    elif decl in ['E', 'W']:
        return 1 if ew_vul else 0
    return 0


def calculate_trick_gap(df):
    """
    Calculates trick gap accurately for both Declarer and Defense roles.
    """
    dds_tricks = pd.to_numeric(df.get('dds_tricks', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
    actual_tricks = pd.to_numeric(df.get('actual_tricks', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

    if 'is_defense' in df.columns:
        is_def = df['is_defense'].fillna(False).astype(bool)
    else:
        is_def = pd.Series(False, index=df.index)

    if dds_tricks.mean() > 3:
        # Declarer expects dds_tricks, Defense expects (13 - dds_tricks)
        expected_tricks = np.where(is_def, 13 - dds_tricks, dds_tricks)
        trick_gap = expected_tricks - actual_tricks
    else:
        trick_gap = -1 * (actual_tricks - dds_tricks)

    return pd.Series(trick_gap, index=df.index).clip(lower=0)


class BridgeDigitalTwin:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = []
        self.accuracy = 0.0
        self.explainer = None

    def prepare_dataset(self, df_boards):
        """Prepare dataset: Target y = 1 for any underperformance vs DDS baseline"""
        if df_boards is None or not isinstance(df_boards, pd.DataFrame) or df_boards.empty:
            return None, None

        df = df_boards.copy()

        # 1. Defense / Declarer
        if 'is_defense' in df.columns:
            is_def_series = df['is_defense']
        else:
            is_def_series = pd.Series(False, index=df.index)

        df['is_defense_clean'] = is_def_series.fillna(False).astype(bool).astype(int)
        df['is_declarer'] = (df['is_defense_clean'] == 0).astype(int)

        declarer_pos = df.get('declarer_pos', pd.Series('N', index=df.index)).fillna('N')

        # 2. Vulnerability
        board_nums = df.get('board_num', pd.Series(1, index=df.index)).fillna(1)
        df['is_vulnerable'] = [
            calculate_vulnerability(b, pos) for b, pos in zip(board_nums, declarer_pos)
        ]

        # 3. HCP & Doubled
        df['hcp'] = pd.to_numeric(df.get('hcp', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        if 'is_doubled' in df.columns:
            df['is_doubled_clean'] = df['is_doubled'].fillna(False).astype(bool).astype(int)
        elif 'bidding' in df.columns:
            df['is_doubled_clean'] = df['bidding'].astype(str).str.contains('X|D', case=False, regex=True).astype(int)
        else:
            df['is_doubled_clean'] = 0

        # 4. Contract Level
        df['contract_level'] = pd.to_numeric(df.get('contract_level', pd.Series(0, index=df.index)), errors='coerce').fillna(0)

        # 5. Trump Suit
        if 'trump' in df.columns:
            trump_series = df['trump'].fillna('NT').astype(str)
        else:
            trump_series = pd.Series('NT', index=df.index)

        for suit in ['S', 'H', 'D', 'C', 'NT']:
            df[f'trump_{suit}'] = (trump_series == suit).astype(int)

        # 6. Target y = 1 if player fell short of DDS
        df['trick_gap'] = calculate_trick_gap(df)
        y = (df['trick_gap'] >= 1).astype(int)

        feature_cols = [
            'contract_level', 
            'is_declarer', 
            'is_defense_clean',
            'is_vulnerable',
            'hcp',
            'is_doubled_clean',
            'trump_S', 'trump_H', 'trump_D', 'trump_C', 'trump_NT'
        ]

        existing_cols = [c for c in feature_cols if c in df.columns]
        self.feature_names = existing_cols

        X = df[existing_cols]
        return X, y

    def train(self, df_boards):
        """Train the model and initialize TreeExplainer"""
        X, y = self.prepare_dataset(df_boards)
        if X is None or len(X) < 5:
            print("Insufficient data for ML training.")
            return False

        if len(X) < 15:
            self.model.fit(X, y)
            self.is_trained = True
            self.accuracy = self.model.score(X, y)
            self.explainer = shap.TreeExplainer(self.model)
            return True

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        preds = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, preds)

        self.explainer = shap.TreeExplainer(self.model)
        return True

    def _generate_shap_recommendations(self, df_boards):
        """Analyze root causes of errors using SHAP and return actionable advice"""
        X, y = self.prepare_dataset(df_boards)
        if X is None or self.explainer is None:
            return []

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

        mean_shap = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Impact': mean_shap
        }).sort_values(by='SHAP_Impact', ascending=False)

        top_causes = shap_df.head(2)['Feature'].tolist()
        recommendations = []

        feature_to_advice = {
            'contract_level': "<strong>Focus Area - Contract Level:</strong> High contracts (Level 4+) drive most of your trick losses. Review pushy bidding decisions and practice safety plays in game contracts.",
            'is_defense_clean': "<strong>Focus Area - Defensive Play:</strong> Defense is your largest source of deviation. Practice signaling, tracking declarer's line, and avoiding passive lead mistakes.",
            'is_declarer': "<strong>Focus Area - Declarer Execution:</strong> As declarer, plan your entries and trump-draw timing before playing to Trick 1.",
            'is_vulnerable': "<strong>Focus Area - Vulnerability Risk:</strong> Vulnerable conditions trigger high-risk play. Consider playing more conservatively when red.",
            'hcp': "<strong>Focus Area - Hand Evaluation:</strong> Underperformance is closely linked to HCP misevaluations. Ensure high HCP hands aren't overvalued without shape context.",
            'trump_NT': "<strong>Focus Area - No Trump Play:</strong> High error rate in NT contracts. Work on ducking schemes, entry management, and holding up stoppers."
        }

        for cause in top_causes:
            if cause in feature_to_advice:
                recommendations.append(feature_to_advice[cause])

        return recommendations

    def _extract_case_study(self, df_boards):
        """Find a representative example board with realistic tactical trick loss"""
        if df_boards is None or df_boards.empty:
            return None

        df = df_boards.copy()
        
        # זיהוי עמודת הלקיחות בפועל
        actual_col = next((c for c in ['actual_tricks', 'tricks', 'tricks_won', 'actual', 'result_tricks'] if c in df.columns), None)
        actual = pd.to_numeric(df[actual_col], errors='coerce').fillna(0) if actual_col else pd.Series(0, index=df.index)

        # זיהוי עמודת ה-DDS
        dds_col = next((c for c in ['dds_tricks', 'dds', 'optimal_tricks', 'dds_res'] if c in df.columns), None)
        dds = pd.to_numeric(df[dds_col], errors='coerce').fillna(0) if dds_col else pd.Series(0, index=df.index)

        is_def = df['is_defense'].fillna(False).astype(bool) if 'is_defense' in df.columns else pd.Series(False, index=df.index)

        # נרמול DDS מול לקיחות בפועל
        dds_clean = dds.clip(lower=0, upper=13)
        expected = np.where(is_def, 13 - dds_clean, dds_clean)
        gap = expected - actual

        df['actual_clean'] = actual
        df['expected_tricks'] = expected
        df['trick_gap'] = gap

        # סינון מדורג למציאת לוח מייצג
        valid_boards = df[(df['trick_gap'] >= 1) & (df['trick_gap'] <= 5) & (df['actual_clean'] >= 0)].sort_values(by='trick_gap', ascending=False)
        
        if valid_boards.empty:
            valid_boards = df[df['trick_gap'] > 0].sort_values(by='trick_gap', ascending=False)

        if valid_boards.empty:
            return None

        worst = valid_boards.iloc[0]

        board_num = worst.get('board_num', worst.get('board', '1'))
        if pd.isna(board_num) or str(board_num).strip().lower() in ['nan', 'n/a', 'none', '']:
            board_num = '1'

        contract = worst.get('contract', worst.get('contract_level', 'N/A'))
        trump = worst.get('trump', 'NT')
        role = "Defense" if bool(worst.get('is_defense', False)) else "Declarer"
        
        actual_val = int(float(worst.get('actual_clean', 0)))
        expected_val = int(float(worst.get('expected_tricks', 0)))
        gap_val = int(float(worst.get('trick_gap', 0)))

        return (
            f"<strong>Representative Case Study (Board #{board_num}):</strong> "
            f"Playing as <strong>{role}</strong> in <strong>{contract} {trump}</strong>, "
            f"actual tricks were <strong>{actual_val + expected_val}</strong> vs. optimal DDS baseline <strong>{expected_val + 13 - expected_val}</strong> "
            f"(Loss of <strong>{13 - gap_val} tricks</strong>). This board highlights a key tactical breakdown under pressure."
        )

    def _extract_behavioral_patterns(self, df_boards):
        """Extract insights matching actual dataset behavior"""
        patterns = []
        df = df_boards.copy()

        is_def = pd.Series(df.get('is_defense', False), index=df.index).fillna(False).astype(bool)
        contract_level = pd.to_numeric(df.get('contract_level', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
        trump = pd.Series(df.get('trump', 'NT'), index=df.index).fillna('NT').astype(str)

        declarer_pos = df.get('declarer_pos', pd.Series('N', index=df.index)).fillna('N')
        board_nums = df.get('board_num', pd.Series(1, index=df.index)).fillna(1)
        
        df['is_vulnerable'] = [
            calculate_vulnerability(b, pos) for b, pos in zip(board_nums, declarer_pos)
        ]

        df['trick_gap'] = calculate_trick_gap(df)
        df['underperform'] = df['trick_gap'] >= 1

        # 1. Defense vs Declarer
        def_boards = df[is_def]
        dec_boards = df[~is_def]

        if len(def_boards) > 0:
            def_rate = (def_boards['underperform'].sum() / len(def_boards)) * 100
            patterns.append(
                f"Defense Vulnerability: In <strong>{def_rate:.1f}%</strong> of defense hands, actual tricks fell short of the optimal DDS baseline."
            )

        if len(dec_boards) > 0:
            dec_rate = (dec_boards['underperform'].sum() / len(dec_boards)) * 100
            patterns.append(
                f"Declarer Play Gap: As Declarer, underperformance vs DDS occurred in <strong>{dec_rate:.1f}%</strong> of hands."
            )

        # 2. High-Contract Pressure
        high_contracts = df[contract_level >= 4]
        if len(high_contracts) >= 3:
            high_rate = (high_contracts['underperform'].sum() / len(high_contracts)) * 100
            patterns.append(
                f"High-Contract Pressure: In Game/Slam contracts (Level 4+), the deviation rate from DDS optimal play is <strong>{high_rate:.1f}%</strong> across {len(high_contracts)} boards."
            )

        # 3. NT Performance Split (Declarer vs Defense)
        nt_dec = df[(trump == 'NT') & (~is_def)]
        nt_def = df[(trump == 'NT') & (is_def)]

        if len(nt_dec) >= 3:
            nt_dec_rate = (nt_dec['underperform'].sum() / len(nt_dec)) * 100
            patterns.append(
                f"No Trump (NT) Declarer Performance: As Declarer in NT contracts, underperformance rate is <strong>{nt_dec_rate:.1f}%</strong>."
            )

        if len(nt_def) >= 3:
            nt_def_rate = (nt_def['underperform'].sum() / len(nt_def)) * 100
            patterns.append(
                f"No Trump (NT) Defense Performance: On Defense against NT contracts, deviation rate is <strong>{nt_def_rate:.1f}%</strong>."
            )

        # 4. SHAP Actionable Recommendations
        recs = self._generate_shap_recommendations(df_boards)
        patterns.extend(recs)

        # 5. Case Study Example
        case_study = self._extract_case_study(df_boards)
        if case_study:
            patterns.append(case_study)

        return patterns

    def generate_ml_insights(self, df_boards=None):
        """Generate structured ML insights for HTML injection"""
        if not self.is_trained:
            return "ML Model not trained yet."

        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        top_feature = df_imp.iloc[0]['Feature'] if not df_imp.empty else "N/A"
        
        feature_dict = {
            'contract_level': 'Contract Level',
            'is_declarer': 'Declarer Role',
            'is_defense_clean': 'Defense Role',
            'is_vulnerable': 'Vulnerability State',
            'hcp': 'High Card Points (HCP)',
            'is_doubled_clean': 'Doubled Contract',
            'trump_S': 'Spades Suit',
            'trump_H': 'Hearts Suit',
            'trump_D': 'Diamonds Suit',
            'trump_C': 'Clubs Suit',
            'trump_NT': 'No Trump (NT)'
        }
        
        top_feature_desc = feature_dict.get(top_feature, top_feature)

        behavioral_patterns = []
        if df_boards is not None and isinstance(df_boards, pd.DataFrame):
            behavioral_patterns = self._extract_behavioral_patterns(df_boards)

        insights = {
            'accuracy': f"{self.accuracy * 100:.1f}%",
            'top_feature': top_feature_desc,
            'patterns': behavioral_patterns
        }
        return insights