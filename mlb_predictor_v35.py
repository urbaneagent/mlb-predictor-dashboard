#!/usr/bin/env python3
"""
MLB Predictor v3.5 - DAILY PREDICTIONS
=====================================
Uses retrained models with season stats:
- Batter: AVG, OPS, HR rate
- Pitcher: ERA, WHIP, K/9
- Game situation: count, runners, inning
- Park factors

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "/Users/mikeross/MLB_Predictions"
MODEL_DIR = f"{DATA_DIR}/models"

# Park factors
PARK_FACTORS = {
    'ARI': 1.15, 'ATL': 1.08, 'BAL': 1.10, 'BOS': 1.05,
    'CHC': 1.12, 'CWS': 1.03, 'CIN': 1.18, 'CLE': 0.95,
    'COL': 1.35, 'DET': 0.90, 'HOU': 1.10, 'KC': 0.95,
    'LAA': 0.92, 'LAD': 0.92, 'MIA': 0.88, 'MIL': 1.00,
    'MIN': 1.05, 'NYM': 0.88, 'NYY': 1.15, 'OAK': 0.85,
    'PHI': 1.12, 'PIT': 0.95, 'SD': 0.82, 'SEA': 0.92,
    'SF': 0.85, 'STL': 1.00, 'TB': 0.90, 'TEX': 1.08,
    'TOR': 1.05, 'WSH': 0.95
}

# ============================================================================
# LOAD DATA
# ============================================================================

def load_season_stats():
    """Load latest season stats"""
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    return batters, pitchers

def load_models():
    """Load v3.5 models"""
    hit_model = joblib.load(f"{MODEL_DIR}/hit_model_v35.joblib")
    hr_model = joblib.load(f"{MODEL_DIR}/hr_model_v35.joblib")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_columns_v35.joblib")
    return hit_model, hr_model, feature_cols

def load_batter_names():
    """Load batter ID to name mapping"""
    return pd.read_csv(f"{DATA_DIR}/batter_names.csv")

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def get_batter_features(batter_id, batters_df):
    """Get batter season stats"""
    row = batters_df[batters_df['IDfg'] == batter_id]
    if len(row) == 0:
        return {'batter_avg': 0.250, 'batter_ops': 0.750, 'batter_hr_rate': 0.03}
    r = row.iloc[0]
    return {
        'batter_avg': r.get('AVG', 0.250),
        'batter_ops': r.get('OPS', 0.750),
        'batter_hr_rate': r.get('HR', 0) / 500
    }

def get_pitcher_features(pitcher_id, pitchers_df):
    """Get pitcher season stats"""
    row = pitchers_df[pitchers_df['IDfg'] == pitcher_id]
    if len(row) == 0:
        return {'pitcher_era': 4.50, 'pitcher_whip': 1.35, 'pitcher_k9': 8.0}
    r = row.iloc[0]
    return {
        'pitcher_era': r.get('ERA', 4.50),
        'pitcher_whip': r.get('WHIP', 1.35),
        'pitcher_k9': r.get('K/9', 8.0)
    }

def predict_lineup(lineup, home_team, batters_df, pitchers_df, batter_names_df, hit_model, hr_model, feature_cols):
    """Predict for entire lineup"""
    predictions = []
    
    for batter_id in lineup:
        # Get features
        bs = get_batter_features(batter_id, batters_df)
        
        # Use average pitcher for now (in production, get actual matchup)
        ps = {'pitcher_era': 4.0, 'pitcher_whip': 1.25, 'pitcher_k9': 8.5}
        
        # Build feature vector
        features = {
            'balls': 0, 'strikes': 0, 'outs_when_up': 0, 'inning': 1,
            'on_1b': 0, 'on_2b': 0, 'on_3b': 0,
            'inning_topbot_code': 0,
            'p_throws_code': 0,
            'batter_avg': bs['batter_avg'],
            'batter_ops': bs['batter_ops'],
            'batter_hr_rate': bs['batter_hr_rate'],
            'pitcher_era': ps['pitcher_era'],
            'pitcher_whip': ps['pitcher_whip'],
            'pitcher_k9': ps['pitcher_k9'],
            'park_factor': PARK_FACTORS.get(home_team, 1.0),
            'count_advantage': 0,
            'late_game': 0,
            'risp': 0
        }
        
        X = pd.DataFrame([features])[feature_cols]
        
        # Predict
        hit_prob = hit_model.predict_proba(X)[0, 1]
        hr_prob = hr_model.predict_proba(X)[0, 1]
        
        # Get name
        name = 'Unknown'
        if batter_names_df is not None:
            row = batter_names_df[batter_names_df['batter_id'] == batter_id]
            if len(row) > 0:
                name = row.iloc[0]['batter_name']
        
        predictions.append({
            'batter_id': batter_id,
            'batter_name': name,
            'batter_avg': bs['batter_avg'],
            'batter_ops': bs['batter_ops'],
            'pitcher_era': ps['pitcher_era'],
            'hit_prob': round(hit_prob, 3),
            'hr_prob': round(hr_prob, 3),
            'park_factor': features['park_factor']
        })
    
    return predictions

# ============================================================================
# EXAMPLE: TOP HR PREDICTIONS
# ============================================================================

def generate_top_predictions():
    """Generate top HR predictions for the day"""
    print("\n" + "="*60)
    print("⚾ MLB PREDICTOR v3.5 - TOP HR PREDICTIONS")
    print("="*60)
    
    # Load data
    batters_df, pitchers_df = load_season_stats()
    hit_model, hr_model, feature_cols = load_models()
    batter_names_df = load_batter_names()
    
    print(f"\n📊 Models loaded:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Feature list: {feature_cols}")
    
    # Get all batters with stats
    all_batters = batters_df[['IDfg', 'Name', 'AVG', 'OPS', 'HR']].copy()
    all_batters = all_batters.sort_values('HR', ascending=False).head(50)
    
    predictions = []
    
    for _, row in all_batters.iterrows():
        batter_id = row['IDfg']
        
        # Get batter features
        bs = {
            'batter_avg': row['AVG'],
            'batter_ops': row['OPS'],
            'batter_hr_rate': row['HR'] / 500
        }
        
        # Average pitcher (league average)
        ps = {'pitcher_era': 4.0, 'pitcher_whip': 1.25, 'pitcher_k9': 8.5}
        
        # Build features
        features = {
            'balls': 0, 'strikes': 0, 'outs_when_up': 0, 'inning': 5,
            'on_1b': 0, 'on_2b': 0, 'on_3b': 0,
            'inning_topbot_code': 0,
            'p_throws_code': 0,
            'batter_avg': bs['batter_avg'],
            'batter_ops': bs['batter_ops'],
            'batter_hr_rate': bs['batter_hr_rate'],
            'pitcher_era': ps['pitcher_era'],
            'pitcher_whip': ps['pitcher_whip'],
            'pitcher_k9': ps['pitcher_k9'],
            'park_factor': 1.0,
            'count_advantage': 0,
            'late_game': 0,
            'risp': 0
        }
        
        X = pd.DataFrame([features])[feature_cols]
        
        hit_prob = hit_model.predict_proba(X)[0, 1]
        hr_prob = hr_model.predict_proba(X)[0, 1]
        
        predictions.append({
            'batter_name': row['Name'],
            'batter_avg': row['AVG'],
            'batter_ops': row['OPS'],
            'batter_hr': row['HR'],
            'hit_prob': round(hit_prob, 3),
            'hr_prob': round(hr_prob, 3)
        })
    
    # Sort by HR probability
    df = pd.DataFrame(predictions)
    df = df.sort_values('hr_prob', ascending=False)
    
    print("\n🏆 TOP 20 HR PREDICTIONS:")
    print("-" * 70)
    for i, row in df.head(20).iterrows():
        print(f"  {row['batter_name']:25} | AVG: {row['batter_avg']:.3f} | "
              f"OPS: {row['batter_ops']:.3f} | HR: {row['batter_hr']:2.0f} | "
              f"HR%: {row['hr_prob']:.1%} | Hit%: {row['hit_prob']:.1%}")
    
    # Save
    output_file = f"{DATA_DIR}/v35_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file}")
    
    return df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    generate_top_predictions()
