import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calculate_vulnerability(board_num, declarer_pos):
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
    if decl in ['E', 'W']:
        return 1 if ew_vul else 0
    return 0


def _hand_hcp(hand_str):
    if not hand_str or not isinstance(hand_str, str):
        return 0
    values = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum(values.get(c, 0) for c in hand_str.upper())


def _parse_pbn_hands(pbn):
    if not pbn or not isinstance(pbn, str):
        return {'N': '', 'E': '', 'S': '', 'W': ''}
    pbn = pbn.strip()
    if len(pbn) > 2 and pbn[1] == ':':
        rest = pbn[2:].strip()
    else:
        rest = pbn
    parts = rest.split()
    if len(parts) < 4:
        return {'N': '', 'E': '', 'S': '', 'W': ''}
    return {'N': parts[0], 'E': parts[1], 'S': parts[2], 'W': parts[3]}


def partnership_hcp(pbn, side):
    hands = _parse_pbn_hands(pbn)
    if str(side).upper() == 'NS':
        return _hand_hcp(hands['N']) + _hand_hcp(hands['S'])
    return _hand_hcp(hands['E']) + _hand_hcp(hands['W'])


def _get_actual_tricks(df):
    if 'tricks_made' in df.columns:
        return pd.to_numeric(df['tricks_made'], errors='coerce')
    if 'actual_tricks' in df.columns:
        return pd.to_numeric(df['actual_tricks'], errors='coerce')
    return pd.Series(np.nan, index=df.index)


def calculate_trick_gap(df):
    """
    Positive = under-performed vs DDS in the player's role.
    dds_tricks / tricks_made are always DECLARER counts.
    """
    dds = (
        pd.to_numeric(df['dds_tricks'], errors='coerce')
        if 'dds_tricks' in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    actual = _get_actual_tricks(df)
    if 'is_defense' in df.columns:
        is_def = df['is_defense'].astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        is_def = pd.Series(False, index=df.index)
    gap = np.where(is_def, actual - dds, dds - actual)
    return pd.Series(gap, index=df.index)


def _is_on_opening_lead(row):
    declarer = str(row.get('declarer', '') or '').upper()[:1]
    side = str(row.get('side', '') or '').upper()
    if declarer not in 'NESW' or side not in ('NS', 'EW'):
        return 0
    lead_map = {'N': 'W', 'E': 'N', 'S': 'E', 'W': 'S'}
    leader = lead_map.get(declarer)
    seats = ('N', 'S') if side == 'NS' else ('E', 'W')
    return 1 if leader in seats else 0


def _prepare_analysis_df(df_boards):
    """Filter to boards with real DDS + result and attach derived columns."""
    if df_boards is None or not isinstance(df_boards, pd.DataFrame) or df_boards.empty:
        return None

    df = df_boards.copy()

    if 'is_defense' in df.columns:
        df['is_defense_clean'] = df['is_defense'].astype(str).str.lower().isin(
            ['true', '1', 'yes']
        )
    else:
        df['is_defense_clean'] = False

    df['contract_level'] = pd.to_numeric(
        df.get('contract_level', 0), errors='coerce'
    ).fillna(0)

    if 'strain' in df.columns:
        df['strain_u'] = df['strain'].astype(str).str.upper().str[0]
    else:
        df['strain_u'] = 'N'

    actual = _get_actual_tricks(df)
    has_dds = 'dds_tricks' in df.columns and df['dds_tricks'].notna()
    has_result = actual.notna()
    df = df.loc[has_dds & has_result].copy()
    if df.empty:
        return None

    df['trick_gap'] = calculate_trick_gap(df)
    df = df[df['trick_gap'].notna()].copy()
    if df.empty:
        return None

    df['underperform'] = df['trick_gap'] >= 1
    df['is_on_lead'] = df.apply(_is_on_opening_lead, axis=1)

    if 'deal_pbn' in df.columns and 'side' in df.columns:
        df['hcp'] = [
            partnership_hcp(p, s) for p, s in zip(df['deal_pbn'], df['side'])
        ]
    else:
        df['hcp'] = 0

    if 'optimal_level' in df.columns:
        opt = pd.to_numeric(df['optimal_level'], errors='coerce')
        df['level_vs_optimal'] = df['contract_level'] - opt.fillna(df['contract_level'])
        df['overbid'] = df['level_vs_optimal'] > 0
    else:
        df['level_vs_optimal'] = 0
        df['overbid'] = False

    return df


def _verdict_proximity(avg_gap, under_rate):
    """Human-readable verdict for closeness to DDS."""
    if avg_gap <= -0.15:
        return (
            "Yes — on average you are <strong>slightly better than double-dummy</strong> "
            f"(avg gap <strong>{avg_gap:+.2f}</strong>). That usually means opponents err "
            "more often than you do."
        )
    if abs(avg_gap) < 0.15 and under_rate < 0.25:
        return (
            "Yes — you are <strong>very close to DDS</strong>. "
            f"Average gap is <strong>{avg_gap:+.2f}</strong> tricks, and you under-perform "
            f"on only <strong>{under_rate*100:.0f}%</strong> of boards."
        )
    if abs(avg_gap) < 0.35 and under_rate < 0.40:
        return (
            "Mostly — you are <strong>reasonably close to DDS</strong>. "
            f"Average gap <strong>{avg_gap:+.2f}</strong>, under-performance on "
            f"<strong>{under_rate*100:.0f}%</strong> of boards. There is still room to tighten play."
        )
    return (
        "Not consistently — there is a meaningful gap vs DDS. "
        f"Average gap <strong>{avg_gap:+.2f}</strong>, under-performance on "
        f"<strong>{under_rate*100:.0f}%</strong> of boards. Focus areas below will help."
    )


def _extract_case_study(df):
    valid = df[df['underperform']].sort_values('trick_gap', ascending=False)
    if valid.empty:
        return None

    worst = valid.iloc[0]
    board_num = worst.get('board_id', worst.get('board_num', '?'))

    raw_contract = worst.get('contract_raw') or worst.get('contract') or ''
    if not raw_contract or str(raw_contract).strip() in ('', 'nan'):
        level = worst.get('contract_level', '?')
        strain = worst.get('strain', 'N')
        decl = worst.get('declarer', '')
        try:
            level = int(float(level))
        except (TypeError, ValueError):
            pass
        raw_contract = f"{level}{strain}{decl}"

    is_def = bool(worst.get('is_defense_clean', False))
    role = "Defense" if is_def else "Declarer"

    try:
        raw_actual = int(float(worst.get('tricks_made', worst.get('actual_tricks', 0))))
    except (TypeError, ValueError):
        raw_actual = 0
    try:
        raw_dds = int(float(worst.get('dds_tricks', 0)))
    except (TypeError, ValueError):
        raw_dds = 0

    if is_def:
        actual_val, dds_val = 13 - raw_actual, 13 - raw_dds
    else:
        actual_val, dds_val = raw_actual, raw_dds

    gap_val = int(round(float(worst.get('trick_gap', 0))))

    return {
        'board': board_num,
        'role': role,
        'contract': raw_contract,
        'actual': actual_val,
        'dds': dds_val,
        'gap': gap_val,
    }


def _build_focus_areas(df):
    """Derive 2–3 concrete coaching tips from the data."""
    tips = []

    def_boards = df[df['is_defense_clean']]
    dec_boards = df[~df['is_defense_clean']]

    def_rate = def_boards['underperform'].mean() if len(def_boards) else 0
    dec_rate = dec_boards['underperform'].mean() if len(dec_boards) else 0

    if len(def_boards) >= 5 and def_rate > dec_rate + 0.05:
        tips.append(
            "<strong>Defense:</strong> Your under-performance rate is higher on defense "
            f"({def_rate*100:.0f}% vs {dec_rate*100:.0f}% as declarer). "
            "Prioritize signaling, counting, and discarding."
        )
    elif len(dec_boards) >= 5 and dec_rate > def_rate + 0.05:
        tips.append(
            "<strong>Declarer play:</strong> More of your trick losses come as declarer "
            f"({dec_rate*100:.0f}% vs {def_rate*100:.0f}% on defense). "
            "Plan the whole hand before Trick 1 — entries, trump timing, communications."
        )

    high = df[df['contract_level'] >= 4]
    if len(high) >= 5:
        high_rate = high['underperform'].mean()
        low = df[df['contract_level'] < 4]
        low_rate = low['underperform'].mean() if len(low) else 0
        if high_rate > low_rate + 0.05:
            tips.append(
                "<strong>Game/Slam contracts:</strong> Deviation from DDS is higher at level 4+ "
                f"({high_rate*100:.0f}% vs {low_rate*100:.0f}% in part-scores). "
                "Review thin games and slam decision-making."
            )

    if 'overbid' in df.columns and df['overbid'].any():
        over = df[df['overbid']]
        fair = df[~df['overbid']]
        if len(over) >= 3 and len(fair) >= 3:
            over_avg = over['trick_gap'].mean()
            fair_avg = fair['trick_gap'].mean()
            if over_avg > fair_avg + 0.15:
                tips.append(
                    "<strong>Overbidding:</strong> When the contract is above the DDS-optimal level, "
                    f"your average gap is <strong>{over_avg:+.2f}</strong> vs "
                    f"<strong>{fair_avg:+.2f}</strong> on non-overbids. "
                    "Tighten judgment on borderline bids."
                )

    on_lead = df[df['is_on_lead'] == 1]
    not_lead = df[df['is_on_lead'] == 0]
    if len(on_lead) >= 5 and len(not_lead) >= 5:
        lead_rate = on_lead['underperform'].mean()
        other_rate = not_lead['underperform'].mean()
        if lead_rate > other_rate + 0.08:
            tips.append(
                "<strong>Opening leads:</strong> Under-performance is more frequent when you are "
                f"on lead ({lead_rate*100:.0f}% vs {other_rate*100:.0f}%). "
                "Review lead agreements and passive vs active choices."
            )

    if not tips:
        tips.append(
            "<strong>General:</strong> No single glaring bias stood out. "
            "Keep reviewing the representative case study and boards where gap ≥ 2."
        )

    return tips[:3]


# ---------------------------------------------------------------------------
# Main class – coaching Q&A (no reliance on weak predictive metrics)
# ---------------------------------------------------------------------------

class BridgeDigitalTwin:
    """
    Analyzes player boards vs DDS and answers coaching questions.
    Predictive ML metrics are intentionally not exposed.
    """

    def __init__(self):
        self.is_trained = False
        self._qa = []

    def train(self, df_boards, n_splits=5):
        """Build Q&A from the board set. n_splits kept for API compatibility."""
        df = _prepare_analysis_df(df_boards)
        if df is None or len(df) < 3:
            self.is_trained = False
            self._qa = [{
                'q': "Is there enough data?",
                'a': "Not yet — fewer than 3 boards have both a real result and DDS data.",
            }]
            return False

        n = len(df)
        under = df['underperform']
        n_under = int(under.sum())
        under_rate = n_under / n
        avg_gap = float(df['trick_gap'].mean())
        avg_gap_under = float(df.loc[under, 'trick_gap'].mean()) if n_under else 0.0

        def_boards = df[df['is_defense_clean']]
        dec_boards = df[~df['is_defense_clean']]

        def_rate = def_boards['underperform'].mean() if len(def_boards) else None
        def_avg = float(def_boards['trick_gap'].mean()) if len(def_boards) else None
        dec_rate = dec_boards['underperform'].mean() if len(dec_boards) else None
        dec_avg = float(dec_boards['trick_gap'].mean()) if len(dec_boards) else None

        high = df[df['contract_level'] >= 4]
        high_rate = high['underperform'].mean() if len(high) >= 3 else None
        high_avg = float(high['trick_gap'].mean()) if len(high) >= 3 else None
        high_n = len(high) if len(high) >= 3 else 0

        case = _extract_case_study(df)
        tips = _build_focus_areas(df)

        # ----- Q1: Close to DDS? -----
        q1 = {
            'q': "Am I close to double-dummy (DDS)?",
            'a': (
                _verdict_proximity(avg_gap, under_rate)
                + f"<br><span style='color:#a6adc8;font-size:0.9em;'>"
                f"Based on <strong>{n}</strong> boards with DDS data. "
                f"Under-performed on <strong>{n_under}/{n}</strong> "
                f"({under_rate*100:.0f}%). "
                f"When you did, average loss was <strong>{avg_gap_under:.2f}</strong> tricks."
                f"</span>"
            ),
        }

        # ----- Q2: Declarer vs Defense -----
        if def_rate is not None and dec_rate is not None:
            if def_rate > dec_rate + 0.03:
                where = "You lose relatively more on <strong>defense</strong>."
            elif dec_rate > def_rate + 0.03:
                where = "You lose relatively more as <strong>declarer</strong>."
            else:
                where = "Losses are <strong>balanced</strong> between declarer and defense."

            q2_a = (
                f"{where}<br>"
                f"• Defense: under-performance <strong>{def_rate*100:.0f}%</strong> "
                f"({int(def_boards['underperform'].sum())}/{len(def_boards)} boards), "
                f"avg gap <strong>{def_avg:+.2f}</strong><br>"
                f"• Declarer: under-performance <strong>{dec_rate*100:.0f}%</strong> "
                f"({int(dec_boards['underperform'].sum())}/{len(dec_boards)} boards), "
                f"avg gap <strong>{dec_avg:+.2f}</strong>"
            )
        else:
            q2_a = "Not enough boards in both roles to compare declarer vs defense."

        q2 = {
            'q': "Where do I lose more tricks — as declarer or on defense?",
            'a': q2_a,
        }

        # ----- Q3: High contracts -----
        if high_rate is not None:
            low = df[df['contract_level'] < 4]
            low_rate = low['underperform'].mean() if len(low) else None
            if low_rate is not None and high_rate > low_rate + 0.05:
                verdict = (
                    f"Yes — game/slam contracts show a higher deviation rate "
                    f"(<strong>{high_rate*100:.0f}%</strong> vs "
                    f"<strong>{low_rate*100:.0f}%</strong> in part-scores)."
                )
            elif high_rate < 0.20:
                verdict = (
                    f"No major red flag — at level 4+ you under-perform on only "
                    f"<strong>{high_rate*100:.0f}%</strong> of boards."
                )
            else:
                verdict = (
                    f"Moderate pressure at level 4+: under-performance "
                    f"<strong>{high_rate*100:.0f}%</strong> across {high_n} boards "
                    f"(avg gap <strong>{high_avg:+.2f}</strong>)."
                )
            q3_a = verdict
        else:
            q3_a = "Not enough game/slam boards (level 4+) to judge."

        q3 = {
            'q': "Are high-level contracts (game/slam) a problem for me?",
            'a': q3_a,
        }

        # ----- Q4: Concrete example -----
        if case:
            q4_a = (
                f"Board <strong>#{case['board']}</strong> — as <strong>{case['role']}</strong> "
                f"in <strong>{case['contract']}</strong>:<br>"
                f"You took <strong>{case['actual']}</strong> tricks vs DDS optimal "
                f"<strong>{case['dds']}</strong> "
                f"(loss of <strong>{case['gap']}</strong> "
                f"trick{'s' if case['gap'] != 1 else ''})."
            )
        else:
            q4_a = "No board with a clear under-performance vs DDS was found."

        q4 = {
            'q': "Can you show a concrete example of a costly board?",
            'a': q4_a,
        }

        # ----- Q5: What to focus on -----
        tips_html = "<br>".join(f"• {t}" for t in tips)
        q5 = {
            'q': "What should I focus on to improve?",
            'a': tips_html,
        }

        self._qa = [q1, q2, q3, q4, q5]
        self.is_trained = True
        return True

    def generate_ml_insights(self, df_boards=None):
        """
        Returns a dict consumed by trainer.generate_html_summary.
        'qa' is the main coaching content; weak predictive metrics are omitted.
        """
        if not self.is_trained and df_boards is not None:
            self.train(df_boards)

        if not self.is_trained:
            return "Not enough data for coaching insights yet."

        return {
            'qa': self._qa,
            # Kept empty / N/A so old HTML keys do not break if referenced
            'accuracy': 'N/A',
            'precision': 'N/A',
            'recall': 'N/A',
            'f1_score': 'N/A',
            'top_feature': 'N/A',
            'patterns': [],
        }
