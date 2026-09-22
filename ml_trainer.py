import numpy as np
import pandas as pd


def _hand_hcp(hand_str):
    if not hand_str or not isinstance(hand_str, str):
        return 0
    values = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}
    return sum(values.get(c, 0) for c in hand_str.upper())


def _parse_pbn_hands(pbn):
    if not pbn or not isinstance(pbn, str):
        return {'N': '', 'E': '', 'S': '', 'W': ''}
    pbn = pbn.strip()
    rest = pbn[2:].strip() if len(pbn) > 2 and pbn[1] == ':' else pbn
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
    return pd.Series(np.where(is_def, actual - dds, dds - actual), index=df.index)


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
    if df_boards is None or not isinstance(df_boards, pd.DataFrame) or df_boards.empty:
        return None
    df = df_boards.copy()
    if 'is_defense' in df.columns:
        df['is_defense_clean'] = df['is_defense'].astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        df['is_defense_clean'] = False
    df['contract_level'] = pd.to_numeric(df.get('contract_level', 0), errors='coerce').fillna(0)
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
        df['hcp'] = [partnership_hcp(p, s) for p, s in zip(df['deal_pbn'], df['side'])]
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
    if avg_gap <= -0.15:
        return "Above DDS", "good", (
            f"On average you are <strong>slightly better than double-dummy</strong> "
            f"(avg gap <strong>{avg_gap:+.2f}</strong>). Opponents likely err more than you."
        )
    if abs(avg_gap) < 0.15 and under_rate < 0.25:
        return "Very close", "good", (
            f"You are <strong>very close to DDS</strong>. Average gap "
            f"<strong>{avg_gap:+.2f}</strong>, under-performance on only "
            f"<strong>{under_rate*100:.0f}%</strong> of boards."
        )
    if abs(avg_gap) < 0.35 and under_rate < 0.40:
        return "Reasonably close", "ok", (
            f"You are <strong>reasonably close to DDS</strong>. Average gap "
            f"<strong>{avg_gap:+.2f}</strong>, under-performance on "
            f"<strong>{under_rate*100:.0f}%</strong> of boards. Still room to tighten play."
        )
    return "Room to improve", "warn", (
        f"There is a meaningful gap vs DDS. Average gap "
        f"<strong>{avg_gap:+.2f}</strong>, under-performance on "
        f"<strong>{under_rate*100:.0f}%</strong> of boards."
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
        'board': board_num, 'role': role, 'contract': raw_contract,
        'actual': actual_val, 'dds': dds_val, 'gap': gap_val,
    }


def _build_focus_areas(df):
    tips = []
    def_boards = df[df['is_defense_clean']]
    dec_boards = df[~df['is_defense_clean']]
    def_rate = def_boards['underperform'].mean() if len(def_boards) else 0
    dec_rate = dec_boards['underperform'].mean() if len(dec_boards) else 0

    if len(def_boards) >= 5 and def_rate > dec_rate + 0.05:
        tips.append({
            'title': 'Defense',
            'text': (
                f"Higher under-performance on defense ({def_rate*100:.0f}% vs "
                f"{dec_rate*100:.0f}% as declarer). Prioritize signaling, counting, discarding."
            ),
        })
    elif len(dec_boards) >= 5 and dec_rate > def_rate + 0.05:
        tips.append({
            'title': 'Declarer play',
            'text': (
                f"More trick losses as declarer ({dec_rate*100:.0f}% vs "
                f"{def_rate*100:.0f}% on defense). Plan entries and trump timing before Trick 1."
            ),
        })

    high = df[df['contract_level'] >= 4]
    if len(high) >= 5:
        high_rate = high['underperform'].mean()
        low = df[df['contract_level'] < 4]
        low_rate = low['underperform'].mean() if len(low) else 0
        if high_rate > low_rate + 0.05:
            tips.append({
                'title': 'Game / Slam',
                'text': (
                    f"Deviation higher at level 4+ ({high_rate*100:.0f}% vs "
                    f"{low_rate*100:.0f}% in part-scores). Review thin games and slam decisions."
                ),
            })

    if 'overbid' in df.columns and df['overbid'].any():
        over = df[df['overbid']]
        fair = df[~df['overbid']]
        if len(over) >= 3 and len(fair) >= 3:
            over_avg = over['trick_gap'].mean()
            fair_avg = fair['trick_gap'].mean()
            if over_avg > fair_avg + 0.15:
                tips.append({
                    'title': 'Overbidding',
                    'text': (
                        f"Above DDS-optimal level: avg gap {over_avg:+.2f} vs "
                        f"{fair_avg:+.2f} otherwise. Tighten borderline bid judgment."
                    ),
                })

    on_lead = df[df['is_on_lead'] == 1]
    not_lead = df[df['is_on_lead'] == 0]
    if len(on_lead) >= 5 and len(not_lead) >= 5:
        lead_rate = on_lead['underperform'].mean()
        other_rate = not_lead['underperform'].mean()
        if lead_rate > other_rate + 0.08:
            tips.append({
                'title': 'Opening leads',
                'text': (
                    f"More under-performance on lead ({lead_rate*100:.0f}% vs "
                    f"{other_rate*100:.0f}%). Review lead agreements and passive vs active choices."
                ),
            })

    if not tips:
        tips.append({
            'title': 'General',
            'text': 'No single glaring bias stood out. Review the case study and boards with gap ≥ 2.',
        })
    return tips[:3]


class BridgeDigitalTwin:
    def __init__(self):
        self.is_trained = False
        self._payload = {}

    def train(self, df_boards, n_splits=5):
        df = _prepare_analysis_df(df_boards)
        if df is None or len(df) < 3:
            self.is_trained = False
            self._payload = {'ok': False}
            return False

        n = len(df)
        under = df['underperform']
        n_under = int(under.sum())
        under_rate = n_under / n
        avg_gap = float(df['trick_gap'].mean())
        avg_gap_under = float(df.loc[under, 'trick_gap'].mean()) if n_under else 0.0

        def_boards = df[df['is_defense_clean']]
        dec_boards = df[~df['is_defense_clean']]
        def_rate = float(def_boards['underperform'].mean()) if len(def_boards) else None
        def_avg = float(def_boards['trick_gap'].mean()) if len(def_boards) else None
        def_n = len(def_boards)
        def_under_n = int(def_boards['underperform'].sum()) if def_n else 0
        dec_rate = float(dec_boards['underperform'].mean()) if len(dec_boards) else None
        dec_avg = float(dec_boards['trick_gap'].mean()) if len(dec_boards) else None
        dec_n = len(dec_boards)
        dec_under_n = int(dec_boards['underperform'].sum()) if dec_n else 0

        high = df[df['contract_level'] >= 4]
        high_rate = float(high['underperform'].mean()) if len(high) >= 3 else None
        high_avg = float(high['trick_gap'].mean()) if len(high) >= 3 else None
        high_n = len(high) if len(high) >= 3 else 0

        label, tone, prox_text = _verdict_proximity(avg_gap, under_rate)
        case = _extract_case_study(df)
        tips = _build_focus_areas(df)

        self._payload = {
            'ok': True, 'n': n, 'n_under': n_under, 'under_rate': under_rate,
            'avg_gap': avg_gap, 'avg_gap_under': avg_gap_under,
            'proximity_label': label, 'proximity_tone': tone, 'proximity_text': prox_text,
            'def_rate': def_rate, 'def_avg': def_avg, 'def_n': def_n, 'def_under_n': def_under_n,
            'dec_rate': dec_rate, 'dec_avg': dec_avg, 'dec_n': dec_n, 'dec_under_n': dec_under_n,
            'high_rate': high_rate, 'high_avg': high_avg, 'high_n': high_n,
            'case': case, 'tips': tips,
        }
        self.is_trained = True
        return True

    def generate_ml_insights(self, df_boards=None):
        if not self.is_trained and df_boards is not None:
            self.train(df_boards)
        if not self.is_trained:
            return {'ok': False}
        return self._payload
