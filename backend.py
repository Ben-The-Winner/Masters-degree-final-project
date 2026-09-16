import matplotlib
matplotlib.use('Agg')  # מונע שימוש ב-GUI/Tkinter ומאפשר שמירת תמונות מתוך Thread משני
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import re
from ddstable.ddstable import get_ddstable
import bisect
from pymongo import MongoClient
from datetime import datetime
import pickle
from trainer import BridgePerformanceTrainer


# === MONGODB CONNECTION ===
client = MongoClient("mongodb://localhost:27017/")
db = client["bridge_stats"]
players_collection = db["players"]


# === GLOBAL DDS CACHE ===
GLOBAL_DDS_CACHE = {}
DDS_CACHE_FILE = "/home/ben/Desktop/Final_Project/dds_cache.pkl"


def load_global_dds_cache():
    """Load the global DDS cache from disk if it exists"""
    global GLOBAL_DDS_CACHE
    if os.path.exists(DDS_CACHE_FILE):
        try:
            with open(DDS_CACHE_FILE, 'rb') as f:
                GLOBAL_DDS_CACHE = pickle.load(f)
            print(f"Loaded {len(GLOBAL_DDS_CACHE)} DDS entries from cache")
        except Exception as e:
            print(f"Error loading DDS cache: {e}")
            GLOBAL_DDS_CACHE = {}
    else:
        GLOBAL_DDS_CACHE = {}


def save_global_dds_cache():
    """Save the global DDS cache to disk"""
    try:
        with open(DDS_CACHE_FILE, 'wb') as f:
            pickle.dump(GLOBAL_DDS_CACHE, f)
        print(f"Saved {len(GLOBAL_DDS_CACHE)} DDS entries to cache")
    except Exception as e:
        print(f"Error saving DDS cache: {e}")


def precompute_all_dds(folder_path):
    """
    Pre-calculate DDS for all unique boards across all XML files.
    This should be called once at startup.
    """
    global GLOBAL_DDS_CACHE
    
    print("Pre-computing DDS for all boards...")
    unique_boards = set()
    
    # First pass: collect all unique board PBNs
    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue
        filepath = os.path.join(folder_path, filename)
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue
        
        for board in root.iter('board'):
            deal_pbn = board_to_pbn(board)
            if deal_pbn:
                unique_boards.add(deal_pbn)
    
    print(f"Found {len(unique_boards)} unique boards across all files")
    
    # Second pass: calculate DDS for boards not in cache
    new_calculations = 0
    for i, board_pbn in enumerate(unique_boards, 1):
        # Check if already in cache
        full_cache_key = (board_pbn, "FULL")
        if full_cache_key in GLOBAL_DDS_CACHE:
            continue
        
        # Calculate DDS
        try:
            pbn = rotate_pbn_to_north(board_pbn)
            if 'N:' in pbn and len(re.findall(r'\.', pbn)) == 12:
                dds_result = get_ddstable(pbn.encode("utf-8"))
                GLOBAL_DDS_CACHE[full_cache_key] = dds_result
                new_calculations += 1
                
                if new_calculations % 100 == 0:
                    print(f"Calculated DDS for {new_calculations} new boards ({i}/{len(unique_boards)})")
            else:
                GLOBAL_DDS_CACHE[full_cache_key] = None
        except Exception as e:
            GLOBAL_DDS_CACHE[full_cache_key] = None
    
    print(f"Pre-computation complete. {new_calculations} new DDS calculations, {len(unique_boards) - new_calculations} from cache")
    
    # Save cache to disk
    save_global_dds_cache()


def board_to_pbn(board_elem):
    """
    Convert a <board> XML element into a PBN string that DDS can read.
    """
    dealer_map = {"North": "N", "East": "E", "South": "S", "West": "W"}
    dealer = dealer_map.get(board_elem.findtext("dealer"), "N")

    def fix_hand(hand):
        return hand if hand else "."

    north = fix_hand(board_elem.findtext("north"))
    east  = fix_hand(board_elem.findtext("east"))
    south = fix_hand(board_elem.findtext("south"))
    west  = fix_hand(board_elem.findtext("west"))

    return f"{dealer}:{north} {east} {south} {west}"


def rotate_pbn_to_north(pbn_text):
    """Ensure PBN always starts with North"""
    if not pbn_text:
        return pbn_text
    
    if pbn_text.startswith("N:"):
        return pbn_text
    
    parts = pbn_text.split()
    if len(parts) != 5:
        return pbn_text
    
    seat_order = parts[0][0]
    hands = parts[1:]
    
    seat_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    current_idx = seat_map.get(seat_order, 0)
    
    rotated_hands = hands[-current_idx:] + hands[:-current_idx]
    return f"N:{rotated_hands[0]} E:{rotated_hands[1]} S:{rotated_hands[2]} W:{rotated_hands[3]}"


def get_dds_tricks(deal_pbn, declarer, strain):
    """Get double dummy tricks from global cache"""
    if not deal_pbn or not declarer or not strain:
        return None
    
    # Check global cache
    full_cache_key = (deal_pbn, "FULL")
    if full_cache_key not in GLOBAL_DDS_CACHE:
        return None
    
    dds_result = GLOBAL_DDS_CACHE[full_cache_key]
    if not dds_result:
        return None
    
    strain_key = "NT" if strain == "N" else strain.upper()
    
    if declarer in dds_result and strain_key in dds_result[declarer]:
        return dds_result[declarer][strain_key]
    
    return None


def get_dds_all_tricks(deal_pbn):
    """Get full DDS table for all declarers and strains from global cache"""
    if not deal_pbn:
        return None
    
    full_cache_key = (deal_pbn, "FULL")
    return GLOBAL_DDS_CACHE.get(full_cache_key)


def get_optimal_level(deal_pbn, side):
    """Get optimal contract level for the side"""
    if not deal_pbn or not side:
        return None
    
    try:
        dds_result = get_dds_all_tricks(deal_pbn)
        if not dds_result:
            return None
        
        declarers = ['N', 'S'] if side == "NS" else ['E', 'W']
        strains = ['NT', 'S', 'H', 'D', 'C']
        
        max_tricks = 0
        for d in declarers:
            for s in strains:
                if d in dds_result and s in dds_result[d]:
                    max_tricks = max(max_tricks, dds_result[d][s])
        
        optimal_level = max(0, max_tricks - 6)
        return optimal_level
    except Exception as e:
        return None


# === CONFIGURATION ===
folder_path = "/home/ben/Desktop/Final_Project/bridge_data"
plot_dir = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(plot_dir, exist_ok=True)

# === INITIALIZE DDS CACHE ON MODULE LOAD ===
def initialize_dds_cache():
    """Initialize the DDS cache when the module is loaded"""
    load_global_dds_cache()
    precompute_all_dds(folder_path)

# Call initialization when module is imported
initialize_dds_cache()


# === NORMALIZE HEBREW ===
def normalize_hebrew(s):
    if not s:
        return ""
    s = re.sub(r"[^א-ת\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# === PLAYER ANALYSIS WRAPPER ===
def run_analysis_for_player(player_identifier):
    """
    Run the full bridge performance analysis for a given player name or number.
    Returns the HTML report file path.
    """

    # === CHECK IF PLAYER ALREADY CACHED ===
    cached = players_collection.find_one(
        {"$or": [{"ibfn": player_identifier}, {"name": player_identifier}]}
    )


    if cached:
        html_path = cached.get("html_report")
        # FIXED: Check if the file actually exists
        if html_path and os.path.exists(html_path):
            print(f"Using cached data for {player_identifier}")
            return cached["html_report"]
        else:
            # File was deleted, recalculate
            print(f"Cached file not found for {player_identifier}, regenerating...")
            # Delete the stale cache entry
            players_collection.delete_one({"_id": cached["_id"]})


    


    # --- Determine player name or IBFN ---
    if isinstance(player_identifier, int) or (isinstance(player_identifier, str) and player_identifier.isdigit()):
        target_ibfn = str(player_identifier)
        target_name = None
    else:
        target_ibfn = None
        target_name = player_identifier

    target_name_norm = normalize_hebrew(target_name) if target_name else None


# === MAIN ANALYSIS LOGIC ===
  
    # === UTILITIES ===
    def parse_contract(contract_str):
        if not contract_str:
            return None
        cs = contract_str.strip()
        m = re.match(r"^(\d)([SHDCN])(X{0,2})([NESW])([=+-]\d+|=)?$", cs)
        if not m:
            return None
        level = int(m.group(1))
        strain = m.group(2)
        doubled = m.group(3)  # X or XX or empty
        declarer = m.group(4)
        result_str = m.group(5) or "="
        tricks_bid = level + 6
        tricks_made = tricks_bid
        if result_str.startswith("+"):
            tricks_made += int(result_str[1:])
        elif result_str.startswith("-"):
            tricks_made -= int(result_str[1:])
        return {
            "contract_level": level,
            "strain": strain,
            "doubled": doubled,
            "declarer": declarer,
            "tricks_bid": tricks_bid,
            "tricks_made": tricks_made,
            "over_under": tricks_made - tricks_bid,
            "success": tricks_made >= tricks_bid,
            "raw": contract_str
        }

    def calculate_score(contract_level, strain, doubled, declarer_vulnerable, tricks_made, tricks_bid):
        """Calculate bridge score for a contract"""
        if tricks_made < tricks_bid:
            # Contract failed - calculate penalty
            undertricks = tricks_bid - tricks_made
            if not doubled:
                if declarer_vulnerable:
                    penalty = undertricks * 100
                else:
                    penalty = undertricks * 50
            elif doubled == "X":
                if declarer_vulnerable:
                    penalty = 200 + (undertricks - 1) * 300
                else:
                    penalty = 100 + (undertricks - 1) * 200
            else:  # XX
                penalty = penalty * 2 if doubled == "X" else 0
            return -penalty
        
        # Contract made
        overtricks = tricks_made - tricks_bid
        
        # Base score
        if strain in ['C', 'D']:
            base_per_trick = 20
        elif strain in ['H', 'S']:
            base_per_trick = 30
        else:  # NT
            base_per_trick = 30
            base_score = 40 + (contract_level - 1) * 30  # First trick is 40
        
        if strain != 'N':
            base_score = contract_level * base_per_trick
        
        # Apply doubling
        if doubled == "X":
            base_score *= 2
        elif doubled == "XX":
            base_score *= 4
        
        # Game/part-game bonus
        if base_score >= 100:
            game_bonus = 500 if declarer_vulnerable else 300
        else:
            game_bonus = 50
        
        # Slam bonus
        slam_bonus = 0
        if contract_level == 6:
            slam_bonus = 750 if declarer_vulnerable else 500
        elif contract_level == 7:
            slam_bonus = 1500 if declarer_vulnerable else 1000
        
        # Overtricks
        if not doubled:
            if strain in ['C', 'D']:
                overtrick_value = overtricks * 20
            else:
                overtrick_value = overtricks * 30
        elif doubled == "X":
            overtrick_value = overtricks * (200 if declarer_vulnerable else 100)
        else:  # XX
            overtrick_value = overtricks * (400 if declarer_vulnerable else 200)
        
        # Double/redouble bonus
        double_bonus = 50 if doubled == "X" else 100 if doubled == "XX" else 0
        
        total = base_score + game_bonus + slam_bonus + overtrick_value + double_bonus
        return total

    def get_declarer_side(declarer):
        if not declarer:
            return None
        return "NS" if declarer in ("N", "S") else "EW"

    # === DATA EXTRACTION ===
    all_results = []
    board_records = []
    board_records_field = []
    pbn_count = 0
    dds_success_count = 0

    target_name_norm = normalize_hebrew(target_name)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".xml"):
            continue
        filepath = os.path.join(folder_path, filename)
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue

        found_player = False
        target_pair_id = None
        target_score = np.nan

        # Loop through all pairs and find matching player by name or IBFN
        for pair in root.iter("pair"):
            names = pair.findtext("names") or ""
            ibfn1 = pair.findtext("ibfn1") or ""
            ibfn2 = pair.findtext("ibfn2") or ""
            rank = pair.findtext("rank")
            score = pair.findtext("restot")

            if target_name_norm and target_name_norm in normalize_hebrew(names):
                found_player = True
            elif target_ibfn and (target_ibfn in ibfn1 or target_ibfn in ibfn2):
                found_player = True

            if found_player:
                target_pair_id = pair.attrib.get("id")
                target_score = float(score) if score else np.nan
                all_results.append({
                    "file": filename,
                    "pair_id": target_pair_id,
                    "score": target_score,
                    "ibfn1": ibfn1,
                    "ibfn2": ibfn2,
                    "names": names
                })
                print(f"Found player in {filename}: {names} (pair_id={target_pair_id}, score={target_score})")
                break
        
        # Collect board-level field data (for ALL files)
        b_id = 0
        for board in root.iter('board'):
            b_id += 1
            deal_pbn = board_to_pbn(board)
            
            if deal_pbn:
                pbn_count += 1
            
            for data in board.findall('data'):
                contract_str_all = data.attrib.get("C") or ""
                parsed_all = parse_contract(contract_str_all)
                declarer = parsed_all["declarer"] if parsed_all else None

                # Determine side for field pair
                if data.attrib.get('N') or data.attrib.get('Nss'):
                    side = "NS"
                elif data.attrib.get('E') or data.attrib.get('Ews'):
                    side = "EW"
                else:
                    side = None

                if not side:
                    continue

                declarer_side = get_declarer_side(declarer)
                is_defense = (side != declarer_side) if declarer_side else None

                # Matchpoint percentage for field result
                field_pct = float(data.attrib.get("Nss" if side == "NS" else "Ews", 0))

                field_record = {
                    "file": filename,
                    "board_id": str(b_id),
                    "side": side,
                    "player_pct": field_pct,
                    "contract_raw": contract_str_all,
                    "declarer_side": declarer_side,
                    "is_defense": is_defense,
                    "deal_pbn": deal_pbn,
                    "ns_vulnerable": data.attrib.get("Nv", "0") == "1",
                    "ew_vulnerable": data.attrib.get("Ev", "0") == "1",
                }

                if parsed_all:
                    field_record.update(parsed_all)

                # Use global cache for DDS lookups
                if parsed_all and declarer and deal_pbn:
                    dds_tricks = get_dds_tricks(deal_pbn, declarer, parsed_all["strain"])
                    if dds_tricks is not None:
                        field_record["dds_tricks"] = dds_tricks
                        dds_success_count += 1
                
                board_records_field.append(field_record)

        # boards of this pair
        if target_pair_id:
            b_id = 0
            for board in root.iter('board'):
                b_id += 1
                deal_pbn = board_to_pbn(board)
                
                for data in board.findall('data'):
                    if data.attrib.get('N') == target_pair_id or data.attrib.get('E') == target_pair_id:
                        side = "NS" if data.attrib.get('N') == target_pair_id else "EW"
                        player_pct = float(data.attrib.get("Nss" if side == "NS" else "Ews", 0))
                        contract_str = data.attrib.get("C") or ""
                        parsed = parse_contract(contract_str)
                        declarer = parsed["declarer"] if parsed else None
                        declarer_side = get_declarer_side(declarer)
                        is_defense = side != declarer_side if declarer_side else None

                        player_record = {
                            "file": filename,
                            "board_id": str(b_id),
                            "pair_id": target_pair_id,
                            "side": side,
                            "player_pct": player_pct,
                            "contract_raw": contract_str,
                            "declarer_side": declarer_side,
                            "is_defense": is_defense,
                            "deal_pbn": deal_pbn,
                            "ns_vulnerable": data.attrib.get("Nv", "0") == "1",
                            "ew_vulnerable": data.attrib.get("Ev", "0") == "1",
                        }
                        
                        if parsed:
                            player_record.update(parsed)
                        
                        # Use global cache for DDS lookups
                        if parsed and declarer and deal_pbn:
                            dds_tricks = get_dds_tricks(deal_pbn, declarer, parsed["strain"])
                            if dds_tricks is not None:
                                player_record["dds_tricks"] = dds_tricks
                            
                            if not is_defense:
                                optimal_level = get_optimal_level(deal_pbn, side)
                                if optimal_level is not None:
                                    player_record["optimal_level"] = optimal_level
                        
                        board_records.append(player_record)

    print(f"Total PBNs parsed: {pbn_count}")
    print(f"Total DDS successes: {dds_success_count}")

    # === DATAFRAMES ===
    df_all = pd.DataFrame(all_results)
    df_boards = pd.DataFrame(board_records)
    df_boards_field = pd.DataFrame(board_records_field)

    print(f"Found {len(df_all)} total results for player")
    print(f"Found {len(df_boards)} board records for player")
    print(f"Found {len(df_boards_field)} field board records")

    # Debug DDS data
    dds_count_player = 0
    dds_count_field = 0
    if 'dds_tricks' in df_boards.columns:
        dds_count_player = df_boards['dds_tricks'].notna().sum()
    if 'dds_tricks' in df_boards_field.columns:
        dds_count_field = df_boards_field['dds_tricks'].notna().sum()

    print(f"Player boards with DDS data: {dds_count_player}/{len(df_boards)}")
    print(f"Field boards with DDS data: {dds_count_field}/{len(df_boards_field)}")
    if 'file' not in df_boards.columns:
        print("Warning: 'file' column missing in df_boards, restoring from board_records")
        df_boards['file'] = [r.get('file', 'Unknown.xml') for r in board_records]

    df_boards['file'] = df_boards['file'].fillna("Unknown.xml").astype(str)
    df_boards['year'] = df_boards['file'].str.extract(r'(\d{4})').astype(int)


# === DATAFRAMES ===
    df_all = pd.DataFrame(all_results)
    df_boards = pd.DataFrame(board_records)
    df_boards_field = pd.DataFrame(board_records_field)

    # --- הוספה: הפעלת ה-Trainer ---
    if not df_boards.empty:
        try:
            BridgePerformanceTrainer(df_boards)
        except Exception as e:
            print(f"Trainer error: {e}")
    # -------------------------------


    # === REPORT ===
    report_lines = []
    def print_and_record(s=""):
        print(s)
        report_lines.append(s)

    def plot_comparison(categories, player_vals, field_vals, dds_vals, title, filename, ylabel="", show_dds=True):
        x = np.arange(len(categories))
        width = 0.25
        plt.figure(figsize=(10, 6))
        
        # Replace NaN with 0 for display
        player_vals = [v if not np.isnan(v) else 0 for v in player_vals]
        field_vals = [v if not np.isnan(v) else 0 for v in field_vals]
        
        if show_dds and len(dds_vals) == len(categories):
            dds_vals = [v if not np.isnan(v) else 0 for v in dds_vals]
        else:
            dds_vals = []  # prevent plotting DDS when not needed

        # Plot bars
        bars1 = plt.bar(x - width, player_vals, width, label="Player", color='#2E86AB', alpha=0.8)
        bars2 = plt.bar(x, field_vals, width, label="Field", color='#A23B72', alpha=0.8)
        
        bars3 = []
        if show_dds and dds_vals:
            bars3 = plt.bar(x + width, dds_vals, width, label="DDS Optimal", color='#F18F01', alpha=0.8)
        
        plt.xticks(x, categories, rotation=45)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Annotate values on bars
        for bars, vals in [(bars1, player_vals), (bars2, field_vals), (bars3, dds_vals)]:
            for bar, val in zip(bars, vals):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        path = os.path.join(plot_dir, filename)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        
        # FIXED: Convert to relative web path for HTML
        relative_path = os.path.relpath(path, os.path.join(os.path.dirname(__file__), "static"))
        return f'/static/{relative_path}'

    def plot_yearly_trend(df, year_col, value_col, title, filename, ylabel):
        """Creates a line graph that shows improvement over years"""
        plt.figure(figsize=(10, 6))

        # Ensure year column is numeric
        df = df.dropna(subset=[year_col, value_col]).copy()
        if df.empty:
            print(f"No yearly data available for {title}")
            plt.close()
            return None
            
        df[year_col] = df[year_col].astype(int)

        yearly_avg = df.groupby(year_col)[value_col].mean().reset_index()

        if yearly_avg.empty:
            print(f"No yearly data available for {title}")
            plt.close()
            return None

        plt.plot(yearly_avg[year_col], yearly_avg[value_col], marker='o', linewidth=2, 
                color='#2E86AB', markersize=8, label='Player Performance')
        
        # Add value labels on points
        for _, row in yearly_avg.iterrows():
            plt.text(row[year_col], row[value_col], f'{row[value_col]:.2f}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.title(f"{title} Over the Years", fontsize=14, fontweight='bold')
        plt.xlabel("Year", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(yearly_avg[year_col])
        plt.grid(True, alpha=0.3)
        plt.legend()

        path = os.path.join(plot_dir, filename)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close()
        
        relative_path = os.path.relpath(path, os.path.join(os.path.dirname(__file__), "static"))
        return f'/static/{relative_path}'

    if df_all.empty:
        print(f"Couldn't find {target_name}")
    else:
        player_declarer = df_boards[df_boards["is_defense"] == False].copy()
        player_defender = df_boards[df_boards["is_defense"] == True].copy()
        field_declarer = df_boards_field[df_boards_field["is_defense"] == False].copy()
        field_defender = df_boards_field[df_boards_field["is_defense"] == True].copy()
        
        print(f"Player declarer records: {len(player_declarer)}")
        print(f"Player defender records: {len(player_defender)}")
        print(f"Field declarer records: {len(field_declarer)}")
        print(f"Field defender records: {len(field_defender)}")

        # === AGGRESSION ===
        print_and_record("\n--- Aggression ---")
        avg_level_player = player_declarer["contract_level"].mean() if not player_declarer.empty and 'contract_level' in player_declarer.columns else np.nan
        avg_level_field = field_declarer["contract_level"].mean() if not field_declarer.empty and 'contract_level' in field_declarer.columns else np.nan
        avg_level_dds = player_declarer["optimal_level"].mean() if not player_declarer.empty and 'optimal_level' in player_declarer.columns else np.nan
        print_and_record(f"Average contract level: Player={avg_level_player:.2f}, Field={avg_level_field:.2f}, DDS={avg_level_dds:.2f}")
        
        path = plot_comparison(["Contract Level"], [avg_level_player], [avg_level_field], [avg_level_dds], 
                             "Aggression", "aggression.png", "Avg Level")
        report_lines.append(f'<img src="{path}" width="600">')
        
        # Yearly Aggression Trend
        if 'year' in player_declarer.columns and 'contract_level' in player_declarer.columns:
            path_year = plot_yearly_trend(
                player_declarer, 'year', 'contract_level',
                "Aggression (Average Bidding Level)",
                "aggression_trend.png",
                "Avg Contract Level"
            )
            if path_year:
                report_lines.append(f'<img src="{path_year}" width="600">')

        # === DECLARER DOUBLE DUMMY DIFFERENCE ===
        print_and_record("\n--- Declarer Double Dummy Difference ---")
        player_declarer_dd = player_declarer.dropna(subset=['dds_tricks', 'tricks_made']).copy()
        field_declarer_dd = field_declarer.dropna(subset=['dds_tricks', 'tricks_made']).copy()
        
        if not player_declarer_dd.empty:
            player_declarer_dd['dd_diff'] = player_declarer_dd['tricks_made'] - player_declarer_dd['dds_tricks']
            avg_dd_diff_player = player_declarer_dd['dd_diff'].mean()
            print_and_record(f"Player declarer DD difference (actual - optimal): {avg_dd_diff_player:.2f} tricks")
            print_and_record(f"Based on {len(player_declarer_dd)} boards with DDS data")
        else:
            avg_dd_diff_player = np.nan
            print_and_record("No player declarer DDS data available")
        
        if not field_declarer_dd.empty:
            field_declarer_dd['dd_diff'] = field_declarer_dd['tricks_made'] - field_declarer_dd['dds_tricks']
            avg_dd_diff_field = field_declarer_dd['dd_diff'].mean()
            print_and_record(f"Field declarer DD difference (actual - optimal): {avg_dd_diff_field:.2f} tricks")
        else:
            avg_dd_diff_field = np.nan
            print_and_record("No field declarer DDS data available")
        
        avg_dd_diff_dds = 0.0
        
        path = plot_comparison(["Declarer DD Diff"], [avg_dd_diff_player], [avg_dd_diff_field], [avg_dd_diff_dds], 
                            "Declarer Double Dummy Difference", "declarer_dd_diff.png", "Avg Tricks (Actual - Optimal)")
        report_lines.append(f'<img src="{path}" width="600">')
        
        # Yearly Declarer DD Diff Trend
        if not player_declarer_dd.empty and 'year' in player_declarer_dd.columns:
            path_year = plot_yearly_trend(
                player_declarer_dd, 'year', 'dd_diff',
                "Declarer Play Quality",
                "declarer_dd_diff_trend.png",
                "Avg Tricks (Actual - Optimal)"
            )
            if path_year:
                report_lines.append(f'<img src="{path_year}" width="600">')

        # === DEFENDER DOUBLE DUMMY DIFFERENCE ===
        print_and_record("\n--- Defender Double Dummy Difference ---")
        player_defender_dd = player_defender.dropna(subset=['dds_tricks', 'tricks_made']).copy()
        field_defender_dd = field_defender.dropna(subset=['dds_tricks', 'tricks_made']).copy()
        
        if not player_defender_dd.empty:
            player_defender_dd['dd_diff'] = player_defender_dd['tricks_made'] - player_defender_dd['dds_tricks']
            avg_dd_diff_def_player = player_defender_dd['dd_diff'].mean()
            print_and_record(f"Player defender DD difference (tricks given away): {avg_dd_diff_def_player:.2f} tricks")
            print_and_record(f"Based on {len(player_defender_dd)} boards with DDS data")
        else:
            avg_dd_diff_def_player = np.nan
            print_and_record("No player defender DDS data available")
        
        if not field_defender_dd.empty:
            field_defender_dd['dd_diff'] = field_defender_dd['tricks_made'] - field_defender_dd['dds_tricks']
            avg_dd_diff_def_field = (field_defender_dd['dd_diff'].mean())*(-0.5)
            print_and_record(f"Field defender DD difference (tricks given away): {avg_dd_diff_def_field:.2f} tricks")
        else:
            avg_dd_diff_def_field = np.nan
            print_and_record("No field defender DDS data available")
        
        avg_dd_diff_def_dds = 0.0
        
        path = plot_comparison(["Defender DD Diff"], [avg_dd_diff_def_player], [avg_dd_diff_def_field], [avg_dd_diff_def_dds], 
                            "Defender Double Dummy Difference", "defender_dd_diff.png", "Avg Tricks Given Away")
        report_lines.append(f'<img src="{path}" width="600">')
        
        # Yearly Defender DD Diff Trend
        if not player_defender_dd.empty and 'year' in player_defender_dd.columns:
            path_year = plot_yearly_trend(
                player_defender_dd, 'year','dd_diff',
                "Defensive Play Quality",
                "defender_dd_diff_trend.png",
                "Avg Tricks Given Away"
            )
            if path_year:
                report_lines.append(f'<img src="{path_year}" width="600">')

        # === TRICK LOSING PLAYS - OPENING LEAD ===
        print_and_record("\n--- Trick-Losing Opening Leads (Strict Definition) ---")

        lead_map = {'N': 'W', 'E': 'N', 'S': 'E', 'W': 'S'}  # Correct: player to declarer's LEFT left

        # Player on opening lead AND gave away a beatable contract
        player_lead_errors = 0
        player_lead_opportunities = 0
        player_lead_records = []

        for _, row in player_defender.dropna(subset=['declarer', 'dds_tricks', 'tricks_bid', 'tricks_made']).iterrows():
            if pd.isna(row['declarer']):
                continue
            leader = lead_map.get(row['declarer'][0])  # first char
            if not leader:
                continue

            player_seats = ['N','S'] if row['side'] == 'NS' else ['E','W']
            on_lead = leader in player_seats

            if on_lead:
                player_lead_opportunities += 1
                # Was contract beatable DD but made anyway? → opening lead likely fatal
                if row['dds_tricks'] < row['tricks_bid'] and row['tricks_made'] >= row['tricks_bid']:
                    player_lead_errors += 1
                    player_lead_records.append({'year': row.get('year'), 'tlp': 1})
                else:
                    player_lead_records.append({'year': row.get('year'), 'tlp': 0})

        player_lead_rate = player_lead_errors / player_lead_opportunities if player_lead_opportunities > 0 else np.nan

        # Field average
        field_lead_errors = 0
        field_lead_opportunities = 0

        for _, row in field_defender_dd.iterrows():
            if pd.isna(row.get('declarer')):
                continue
            leader = lead_map.get(row['declarer'][0])
            if not leader:
                continue
            side_seats = ['N','S'] if row['side'] == 'NS' else ['E','W']
            if leader in side_seats:
                field_lead_opportunities += 1
                if row['dds_tricks'] < row['tricks_bid'] and row['tricks_made'] >= row['tricks_bid']:
                    field_lead_errors += 1

        field_lead_rate = field_lead_errors / field_lead_opportunities if field_lead_opportunities > 0 else np.nan

        print_and_record(f"Player fatal opening leads: {player_lead_errors}/{player_lead_opportunities} = {player_lead_rate:.2%}")
        print_and_record(f"Field average: {field_lead_errors}/{field_lead_opportunities} = {field_lead_rate:.2%}")

        path = plot_comparison(
            ["Fatal Opening Leads"], 
            [player_lead_rate], [field_lead_rate], [0.0],
            "Trick-Losing Opening Leads (Gave away beatable contract)",
            "tlp_opening_lead_strict.png", "Error Rate"
        )
        report_lines.append(f'<img src="{path}" width="600">')

        # === YEARLY TREND FOR FATAL OPENING LEADS ===
        if player_lead_records:
            df_lead_trend = pd.DataFrame(player_lead_records)
            if 'year' in df_lead_trend.columns and not df_lead_trend['year'].isna().all():
                path_year = plot_yearly_trend(
                    df_lead_trend, 
                    'year', 
                    'tlp',
                    "Opening Lead Quality Over Time",
                    "tlp_opening_lead_trend.png",
                    "Fatal Opening Lead Rate"
                )
                if path_year:
                    report_lines.append(f'<img src="{path_year}" width="600">')

        # === TRICK LOSING PLAYS - AS DEFENDER (excluding opening lead) ===
        print_and_record("\n--- Trick Losing Plays - As Defender (Excluding Opening Lead) ---")
        
        if not player_defender_dd.empty:
            player_defender_dd['tlp'] = (player_defender_dd['tricks_made'] > player_defender_dd['dds_tricks']).astype(int)
            tlp_def_count_player = player_defender_dd['tlp'].sum()
            tlp_def_total_player = len(player_defender_dd)
            tlp_def_rate_player = tlp_def_count_player / tlp_def_total_player if tlp_def_total_player > 0 else np.nan
            print_and_record(f"Player TLP as defender: {tlp_def_rate_player:.2%} ({tlp_def_count_player}/{tlp_def_total_player} boards)")
        else:
            tlp_def_rate_player = np.nan
            print_and_record("No player defender TLP data available")
        
        if not field_defender_dd.empty:
            tlp_def_count_field = (field_defender_dd['tricks_made'] > field_defender_dd['dds_tricks']).sum()
            tlp_def_total_field = len(field_defender_dd)
            tlp_def_rate_field = tlp_def_count_field / tlp_def_total_field if tlp_def_total_field > 0 else np.nan
            print_and_record(f"Field TLP as defender: {tlp_def_rate_field:.2%} ({tlp_def_count_field}/{tlp_def_total_field} boards)")
        else:
            tlp_def_rate_field = np.nan
            print_and_record("No field defender TLP data available")
        
        tlp_def_rate_dds = 0.0
        
        path = plot_comparison(["Defender TLP"], [tlp_def_rate_player], [tlp_def_rate_field], [tlp_def_rate_dds], 
                            "Trick Losing Plays - As Defender", "tlp_defender.png", "Frequency")
        report_lines.append(f'<img src="{path}" width="600">')
        
        # Yearly Defender TLP Trend
        if not player_defender_dd.empty and 'year' in player_defender_dd.columns and 'tlp' in player_defender_dd.columns:
            path_year = plot_yearly_trend(
                player_defender_dd, 'year', 'tlp',
                "Defensive TLP Rate",
                "tlp_defender_trend.png",
                "TLP Frequency"
            )
            if path_year:
                report_lines.append(f'<img src="{path_year}" width="600">')

        # === RESULTS WITHOUT PLAY (CONTRACT QUALITY) ===
        print_and_record("\n--- Results Without Play (Contract Quality) ---")

        mp_records = []
        mp_values = []

        # For each board the player played
        player_boards_with_mp = df_boards.dropna(subset=["board_id", "contract_level", "strain", "side"]).copy()

        for idx, row in player_boards_with_mp.iterrows():
            board_id = row["board_id"]
            file_name = row["file"]
            player_side = row["side"]
            
            # Get all field results for this board on the SAME side
            board_field_same_side = df_boards_field[
                (df_boards_field["board_id"] == board_id) &
                (df_boards_field["file"] == file_name) &
                (df_boards_field["side"] == player_side) &
                (df_boards_field["contract_level"].notna())
            ].copy()

            if len(board_field_same_side) < 2:  # Need at least 2 results to compare
                continue

            # Calculate matchpoints for the player's CONTRACT (not result)
            player_contract = row["contract_level"]
            player_strain = row["strain"]
            
            # Compare player's contract against all field contracts
            mp_score = 0
            comparisons = 0
            
            for _, field_row in board_field_same_side.iterrows():
                field_contract = field_row["contract_level"]
                field_strain = field_row["strain"]
                
                # Skip comparing against self (if player is in field data)
                if field_contract == player_contract and field_strain == player_strain:
                    continue
                    
                comparisons += 1
                
                # Higher level contracts are more aggressive
                if player_contract > field_contract:
                    mp_score += 1
                elif player_contract == field_contract:
                    # If same level, compare strain value (NT=5, S=4, H=3, D=2, C=1)
                    strain_values = {'C': 1, 'D': 2, 'H': 3, 'S': 4, 'N': 5}
                    player_strain_val = strain_values.get(player_strain, 0)
                    field_strain_val = strain_values.get(field_strain, 0)
                    
                    if player_strain_val > field_strain_val:
                        mp_score += 1
                    elif player_strain_val == field_strain_val:
                        mp_score += 0.5
            
            if comparisons > 0:
                # Convert to percentage (0-100%)
                mp_pct = (mp_score / comparisons) * 100
                field_avg = 50.0  # By definition, field average is 50%
                
                # Relative performance
                rel_mp = mp_pct - field_avg
                
                mp_values.append(rel_mp)
                mp_records.append({
                    "year": row.get("year"),
                    "mp_relative": rel_mp
                })

        # === Aggregate and Output ===
        if mp_values:
            avg_mp_rel = np.mean(mp_values)

            print_and_record(f"Average Contract Quality (vs Field): {avg_mp_rel:+.2f}%")
            print_and_record(f"Boards with comparison data: {len(mp_values)}")

            # Plot overall comparison
            path = plot_comparison(
                ["Contract Quality"],
                [avg_mp_rel],
                [0], [],
                "Results Without Play (Contract Quality)",
                "results_without_play.png",
                "Relative MP (%) vs Field",
                show_dds=False
            )
            report_lines.append(f'<img src="{path}" width="600">')

            # Yearly trend
            df_mp_year = pd.DataFrame(mp_records)
            if not df_mp_year.empty and "year" in df_mp_year.columns:
                path_year = plot_yearly_trend(
                    df_mp_year, "year", "mp_relative",
                    "Contract Quality Over Time",
                    "results_without_play_trend.png",
                    "Relative MP (%) vs Field"
                )
                if path_year:
                    report_lines.append(f'<img src="{path_year}" width="600">')

        else:
            print_and_record("Insufficient data for Results Without Play (Contract Quality)")

        # === MAKING POST-LEAD MAKEABLE CONTRACTS ===
        print_and_record("\n--- Making Post-Lead Makeable Contracts ---")
        
        player_declarer_postlead = player_declarer.dropna(subset=['dds_tricks', 'tricks_made', 'tricks_bid']).copy()
        field_declarer_postlead = field_declarer.dropna(subset=['dds_tricks', 'tricks_made', 'tricks_bid']).copy()
        
        player_makeable = player_declarer_postlead[player_declarer_postlead['dds_tricks'] >= player_declarer_postlead['tricks_bid']].copy()
        field_makeable = field_declarer_postlead[field_declarer_postlead['dds_tricks'] >= field_declarer_postlead['tricks_bid']].copy()
        
        if not player_makeable.empty:
            player_makeable['made'] = (player_makeable['tricks_made'] >= player_makeable['tricks_bid']).astype(int)
            player_made_makeable = player_makeable['made'].sum()
            player_total_makeable = len(player_makeable)
            player_rate = player_made_makeable / player_total_makeable if player_total_makeable > 0 else np.nan
            
            print_and_record(f"Player making post-lead makeable contracts: {player_rate:.2%} ({player_made_makeable}/{player_total_makeable})")
        else:
            player_rate = np.nan
            print_and_record("No player data for post-lead makeable contracts")
        
        if not field_makeable.empty:
            field_made_makeable = (field_makeable['tricks_made'] >= field_makeable['tricks_bid']).sum()
            field_total_makeable = len(field_makeable)
            field_rate = field_made_makeable / field_total_makeable if field_total_makeable > 0 else np.nan
            
            print_and_record(f"Field making post-lead makeable contracts: {field_rate:.2%} ({field_made_makeable}/{field_total_makeable})")
        else:
            field_rate = np.nan
            print_and_record("No field data for post-lead makeable contracts")
        
        dds_rate = 1.0
        
        if not np.isnan(player_rate):
            path = plot_comparison(["Post-Lead Makeable"], [player_rate], [field_rate], [dds_rate], 
                                "Making Post-Lead Makeable Contracts", "postlead_makeable.png", "Success Rate")
            report_lines.append(f'<img src="{path}" width="600">')
            
            # Yearly Post-Lead Makeable Trend
            if not player_makeable.empty and 'year' in player_makeable.columns and 'made' in player_makeable.columns:
                path_year = plot_yearly_trend(
                    player_makeable, 'year', 'made',
                    "Making Makeable Contracts",
                    "postlead_makeable_trend.png",
                    "Success Rate"
                )
                if path_year:
                    report_lines.append(f'<img src="{path_year}" width="600">')

    


    # === OUTPUT HTML ===
    safe_name = re.sub(r"[^א-תA-Za-z0-9_\- ]", "", target_name or str(target_ibfn))
    out_fname = f"/home/ben/Desktop/Final_Project/player_stats_{safe_name.replace(' ', '_')}.html"
    html_content = f"""<!DOCTYPE html>
    <html><head><meta charset='utf-8'><title>Stats for Player {target_name}</title>
    <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
    h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 30px; }}
    p {{ line-height: 1.6; margin: 10px 0; }}
    img {{ display: block; margin: 20px auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .debug {{ background-color: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 0.9em; }}
    form {{ text-align: center; margin-bottom: 25px; }}
    input, button {{ padding: 6px 10px; font-size: 1em; }}
    #status {{ color: #555; font-style: italic; margin-top: 10px; }}
    </style>
    </head><body>

    <!-- ===== Player Input Form ===== -->
    <form id="playerForm">
    <label for="playerInput"><b>Enter player name or number:</b></label><br>
    <input type="text" id="playerInput" name="playerInput" placeholder="" style="width:250px;">
    <button type="button" onclick="startAnalysis()">Run Analysis</button>
    </form>

    <div id="status"><i>Waiting for input...</i></div>

    <script>
    function startAnalysis() {{
    var player = document.getElementById("playerInput").value;
    document.getElementById("status").innerHTML =
        "<b>Running analysis for " + player + "...</b><br><i>This is gonna take a while. Please wait.</i>";
    }}
    </script>

    <!-- ===== Report Header ===== -->
    <h1>Bridge Performance Analysis for {target_name or f'Player #{target_ibfn}'}</h1>
    """


    html_content += f"""
    <div class="debug">
        <strong>Debug Info:</strong> {len(df_boards)} player boards, {dds_count_player} with DDS data<br>
        Field: {len(df_boards_field)} boards, {dds_count_field} with DDS data<br>
        PBNs parsed: {pbn_count}, DDS successes: {dds_success_count}
    </div>
    """

    for line in report_lines:
        if line.strip().startswith("---"):
            html_content += f"<h2>{line.strip('- ').strip()}</h2>"
        elif line.strip().startswith("<img"):
            html_content += line
        else:
            html_content += f"<p>{line}</p>"

    html_content += "</body></html>"



    with open(out_fname, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Report written to {out_fname}")

    # === NOW SAVE RESULTS TO MONGODB ===
    result_data = {
        "ibfn": target_ibfn,
        "name": target_name,
        "timestamp": datetime.now().isoformat(),
        "html_report": out_fname
    }

    players_collection.update_one(
        {"ibfn": target_ibfn},
        {"$set": result_data},
        upsert=True
    )


   # === RUN TRAINER & INJECT HTML ===
    if not df_boards.empty:
        try:
            trainer = BridgePerformanceTrainer()
            trainer.train(df_boards)
            
            if os.path.exists(out_fname):
                with open(out_fname, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                trainer_html = trainer.generate_html_summary(df_boards=df_boards)
                updated_content = content.replace("</body>", f"{trainer_html}\n</body>")
                
                with open(out_fname, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
        except Exception as e:
            print(f"Trainer error: {e}")

    return out_fname

plt.close('all')  # או plt.close(fig)


if __name__ == "__main__":
    # DDS cache is already initialized at module load
    # Just run player analysis
    player_identifier = input("Enter player name or number: ").strip()
    run_analysis_for_player(player_identifier)

