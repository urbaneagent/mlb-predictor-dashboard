#!/usr/bin/env python3
"""
MLB Predictor v2.2 - ENHANCED EDITION
======================================
Now includes:
1. Statcast features (launch speed, angle, etc.)
2. Batter season stats (AVG, OBP, SLG, HR rate)
3. Pitcher season stats (ERA, K%, BB%, etc.)
4. Weather adjustments
5. Park factors

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pybaseball import statcast, batting_stats, pitching_stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "/Users/mikeross/MLB_Predictions"
MODEL_DIR = f"{DATA_DIR}/models"
OUTPUT_DIR = DATA_DIR

# Park factors
PARK_FACTORS = {
    'ARI': {'run': 1.08, 'hr': 1.15}, 'ATL': {'run': 1.02, 'hr': 1.08},
    'BAL': {'run': 1.04, 'hr': 1.10}, 'BOS': {'run': 1.06, 'hr': 1.05},
    'CHC': {'run': 1.10, 'hr': 1.12}, 'CWS': {'run': 1.01, 'hr': 1.03},
    'CIN': {'run': 1.12, 'hr': 1.18}, 'CLE': {'run': 0.98, 'hr': 0.95},
    'COL': {'run': 1.30, 'hr': 1.35}, 'DET': {'run': 0.96, 'hr': 0.90},
    'HOU': {'run': 1.05, 'hr': 1.10}, 'KC': {'run': 0.98, 'hr': 0.95},
    'LAA': {'run': 0.97, 'hr': 0.92}, 'LAD': {'run': 0.96, 'hr': 0.92},
    'MIA': {'run': 0.92, 'hr': 0.88}, 'MIL': {'run': 1.00, 'hr': 1.00},
    'MIN': {'run': 1.02, 'hr': 1.05}, 'NYM': {'run': 0.94, 'hr': 0.88},
    'NYY': {'run': 1.08, 'hr': 1.15}, 'OAK': {'run': 0.93, 'hr': 0.85},
    'PHI': {'run': 1.06, 'hr': 1.12}, 'PIT': {'run': 0.98, 'hr': 0.95},
    'SD': {'run': 0.88, 'hr': 0.82}, 'SEA': {'run': 0.96, 'hr': 0.92},
    'SF': {'run': 0.92, 'hr': 0.85}, 'STL': {'run': 1.00, 'hr': 1.00},
    'TB': {'run': 0.94, 'hr': 0.90}, 'TEX': {'run': 1.04, 'hr': 1.08},
    'TOR': {'run': 1.02, 'hr': 1.05}, 'WSH': {'run': 0.98, 'hr': 0.95}
}

# ============================================================================
# LOAD SEASON STATS
# ============================================================================

def load_season_stats():
    """Load batter and pitcher season stats"""
    print("📊 Loading season stats...")
    
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    print(f"   Batters: {len(batters)}")
    print(f"   Pitchers: {len(pitchers)}")
    
    return batters, pitchers

def get_batter_features(batter_id, batters_df):
    """Get batter season stats"""
    row = batters_df[batters_df['IDfg'] == batter_id]
    if len(row) == 0:
        return {
            'batter_avg': 0.250,
            'batter_obp': 0.320,
            'batter_slg': 0.420,
            'batter_ops': 0.742,
            'batter_hr_rate': 0.03,
            'batter_k_rate': 0.20,
            'batter_bb_rate': 0.08,
            'batter_barrel_rate': 0.06,
            'batter_hard_hit_rate': 0.35
        }
    
    r = row.iloc[0]
    ab = max(r.get('AB', 0), 1)
    
    return {
        'batter_avg': r.get('AVG', 0.250),
        'batter_obp': r.get('OBP', 0.320),
        'batter_slg': r.get('SLG', 0.420),
        'batter_ops': r.get('OPS', 0.742),
        'batter_hr_rate': r.get('HR', 0) / ab if ab > 0 else 0.03,
        'batter_k_rate': abs(r.get('K%', 0.20)),
        'batter_bb_rate': abs(r.get('BB%', 0.08)),
        'batter_barrel_rate': r.get('Barrel%', 0.06),
        'batter_hard_hit_rate': r.get('Hard%', 0.35) / 100 if r.get('Hard%', 35) > 1 else r.get('Hard%', 0.35)
    }

def get_pitcher_features(pitcher_id, pitchers_df):
    """Get pitcher season stats"""
    row = pitchers_df[pitchers_df['IDfg'] == pitcher_id]
    if len(row) == 0:
        return {
            'pitcher_era': 4.50,
            'pitcher_whip': 1.35,
            'pitcher_k9': 8.0,
            'pitcher_bb9': 3.5,
            'pitcher_hr9': 1.2,
            'pitcher_avg_against': 0.260,
            'pitcher_strikeout_rate': 0.22,
            'pitcher_walk_rate': 0.10
        }
    
    r = row.iloc[0]
    ip = max(r.get('IP', 0), 1)
    
    return {
        'pitcher_era': r.get('ERA', 4.50),
        'pitcher_whip': r.get('WHIP', 1.35),
        'pitcher_k9': r.get('K/9', 8.0),
        'pitcher_bb9': r.get('BB/9', 3.5),
        'pitcher_hr9': r.get('HR/9', 1.2),
        'pitcher_avg_against': r.get('AVG', 0.260),
        'pitcher_strikeout_rate': abs(r.get('K%', 0.22)),
        'pitcher_walk_rate': abs(r.get('BB%', 0.10))
    }

# ============================================================================
# LOAD ML MODELS
# ============================================================================

def load_models():
    """Load trained XGBoost models"""
    print("🤖 Loading ML models...")
    hit_model = joblib.load(f"{MODEL_DIR}/hit_model_xgb.joblib")
    hr_model = joblib.load(f"{MODEL_DIR}/hr_model_xgb.joblib")
    return hit_model, hr_model

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def predict_matchup(batter_id, pitcher_id, home_team, away_team, 
                    launch_speed=None, launch_angle=None,
                    batters_df=None, pitchers_df=None,
                    weather=None, park=None):
    """
    Generate comprehensive prediction for a batter vs pitcher matchup
    """
    # Get batter stats
    batter_stats = get_batter_features(batter_id, batters_df)
    
    # Get pitcher stats  
    pitcher_stats = get_pitcher_features(pitcher_id, pitchers_df)
    
    # Base probabilities from season stats
    # A .300 batter vs a 4.50 ERA pitcher
    base_hit_prob = batter_stats['batter_avg'] * (1 - pitcher_stats['pitcher_avg_against'] * 0.5)
    base_hr_prob = batter_stats['batter_hr_rate'] * (1 - pitcher_stats['pitcher_era'] / 10)
    
    # Adjustments
    adjustments = 1.0
    
    # Statcast adjustment (if available)
    if launch_speed and launch_angle:
        if launch_speed >= 100:  # Hard hit
            adjustments *= 1.3
        if 20 <= launch_angle <= 35:  # Launch angle sweet spot
            adjustments *= 1.4
    
    # Park factor
    if park and park in PARK_FACTORS:
        adjustments *= PARK_FACTORS[park]['hr']
    
    # Weather adjustment
    if weather:
        temp = weather.get('temperature', 70)
        wind = weather.get('wind_speed', 10)
        
        if temp > 85:
            adjustments *= 1.10
        elif temp < 55:
            adjustments *= 0.90
        
        if wind > 15:
            adjustments *= 1.10
    
    # Pitcher quality adjustment
    if pitcher_stats['pitcher_era'] < 3.0:  # Elite pitcher
        adjustments *= 0.70
    elif pitcher_stats['pitcher_era'] > 5.0:  # Poor pitcher
        adjustments *= 1.30
    
    # Final predictions
    hit_probability = min(base_hit_prob * adjustments, 0.95)
    hr_probability = min(base_hr_prob * adjustments, 0.50)
    
    return {
        'hit_probability': round(hit_probability, 3),
        'hr_probability': round(hr_probability, 3),
        'batter_stats': batter_stats,
        'pitcher_stats': pitcher_stats,
        'adjustments': adjustments
    }

# ============================================================================
# GENERATE DAILY PREDICTIONS
# ============================================================================

def generate_daily_predictions(game_date):
    """Generate predictions for all games on a date"""
    print(f"\n{'='*60}")
    print(f"⚾ MLB PREDICTIONS v2.2 - {game_date}")
    print(f"{'='*60}")
    
    # Load data
    batters, pitchers = load_season_stats()
    hit_model, hr_model = load_models()
    
    # Pull today's Statcast
    print(f"\n📡 Pulling Statcast for {game_date}...")
    try:
        df = statcast(game_date, game_date, verbose=False)
        print(f"   Loaded {len(df)} plate appearances")
    except Exception as e:
        print(f"   Error: {e}")
        return None
    
    if df is None or len(df) == 0:
        print("⚠ No data")
        return None
    
    # Filter to batters with launch data
    batted = df[df['launch_speed'].notna()].copy()
    print(f"   Batted balls: {len(batted)}")
    
    # Generate predictions
    predictions = []
    for _, row in batted.iterrows():
        batter_id = row.get('batter')
        pitcher_id = row.get('pitcher')
        home_team = row.get('home_team')
        away_team = row.get('away_team')
        
        # Determine park
        if row.get('inning_topbot') == 'Top':
            park = away_team
        else:
            park = home_team
        
        pred = predict_matchup(
            batter_id, pitcher_id, home_team, away_team,
            launch_speed=row.get('launch_speed'),
            launch_angle=row.get('launch_angle'),
            batters_df=batters,
            pitchers_df=pitchers,
            park=park
        )
        
        pred['batter_id'] = batter_id
        pred['pitcher_id'] = pitcher_id
        pred['batter_name'] = row.get('player_name')
        pred['park'] = park
        
        predictions.append(pred)
    
    # Aggregate by batter
    print("\n📊 Aggregating predictions...")
    pred_df = pd.DataFrame(predictions)
    
    # Group by batter and average
    agg = pred_df.groupby(['batter_id', 'batter_name', 'park']).agg({
        'hit_probability': 'mean',
        'hr_probability': 'mean'
    }).reset_index()
    
    # Sort by HR probability
    agg = agg.sort_values('hr_probability', ascending=False)
    
    # Save
    output_file = f"{OUTPUT_DIR}/predictions_v22_{game_date}.csv"
    agg.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    # Top HR picks
    print("\n🏆 TOP HR PREDICTIONS:")
    for i, row in agg.head(10).iterrows():
        print(f"   {row['batter_name']}: {row['hr_probability']:.1%} HR, {row['hit_probability']:.1%} Hit")
    
    return agg

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    generate_daily_predictions(target_date)
