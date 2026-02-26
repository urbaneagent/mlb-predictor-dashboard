#!/usr/bin/env python3
"""
MLB Predictor v4.0 - FORMULA-BASED (More Reliable)
==================================================
Uses proven formula:
- HR% = Batter HR Rate × Park Factor × (1 - Pitcher ERA Adjustment)
- Hit% = Batter AVG × (1 - Pitcher Quality Factor)

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "/Users/mikeross/MLB_Predictions"

# Park factors (HR boost)
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
# PREDICTION ENGINE
# ============================================================================

def predict_hr(batter_avg, batter_hr, batter_ops, pitcher_era, park_factor=1.0):
    """Predict HR probability"""
    # Base HR rate from season
    hr_rate = batter_hr / 500  # Approximate AB per HR
    
    # Pitcher adjustment (elite pitchers suppress HRs)
    pitcher_factor = 1 - (pitcher_era / 20)  # Lower ERA = lower HR rate
    
    # Park adjustment
    park_adj = park_factor
    
    # Final HR probability
    hr_prob = hr_rate * pitcher_factor * park_adj
    
    return min(hr_prob, 0.5)  # Cap at 50%

def predict_hit(batter_avg, pitcher_era, pitcher_whip, park_factor=1.0):
    """Predict hit probability"""
    # Base: batter's average
    base = batter_avg
    
    # Pitcher suppresses hitting
    pitcher_factor = 1 - (pitcher_era / 15)  # Higher ERA = more hits allowed
    whip_factor = 1 - (pitcher_whip - 1) / 2  # WHIP > 1 suppresses contact
    
    # Park factor
    park_adj = park_factor
    
    hit_prob = base * pitcher_factor * whip_factor * park_adj
    
    return min(hit_prob, 0.6)  # Cap at 60%

# ============================================================================
# MAIN PREDICTION
# ============================================================================

def generate_predictions():
    """Generate top HR predictions"""
    print("\n" + "="*70)
    print("⚾ MLB PREDICTOR v4.0 - HR PREDICTIONS")
    print("="*70)
    
    # Load data
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    print(f"\n📊 Loaded: {len(batters)} batters, {len(pitchers)} pitchers")
    
    # Get top HR hitters
    top_hr = batters.nlargest(100, 'HR')[['IDfg', 'Name', 'Team', 'AVG', 'OPS', 'HR', 'AB']].copy()
    top_hr['hr_rate'] = top_hr['HR'] / top_hr['AB'].clip(lower=1)
    
    # Get pitchers by quality
    elite_pitchers = pitchers.nsmallest(20, 'ERA')[['IDfg', 'Name', 'Team', 'ERA', 'WHIP', 'K/9']].copy()
    bad_pitchers = pitchers.nlargest(20, 'ERA')[['IDfg', 'Name', 'Team', 'ERA', 'WHIP', 'K/9']].copy()
    
    predictions = []
    
    print("\n" + "="*70)
    print("🏆 BEST HR PLAYS (vs BAD PITCHERS + HR PARKS)")
    print("="*70)
    
    for _, pitcher in bad_pitchers.iterrows():
        for _, batter in top_hr.iterrows():
            park = PARK_FACTORS.get(batter['Team'], 1.0)
            
            hr_prob = predict_hr(batter['AVG'], batter['HR'], batter['OPS'], pitcher['ERA'], park)
            hit_prob = predict_hit(batter['AVG'], pitcher['ERA'], pitcher['WHIP'], park)
            
            predictions.append({
                'batter': batter['Name'],
                'batter_team': batter['Team'],
                'batter_avg': batter['AVG'],
                'batter_hr': batter['HR'],
                'pitcher': pitcher['Name'],
                'pitcher_team': pitcher['Team'],
                'pitcher_era': pitcher['ERA'],
                'park': batter['Team'],
                'park_factor': park,
                'hr_prob': hr_prob,
                'hit_prob': hit_prob,
                'value_score': hr_prob * 100 + (1/pitcher['ERA']) * 10
            })
    
    # Sort by value
    df = pd.DataFrame(predictions)
    df = df.sort_values('value_score', ascending=False)
    
    # Top 30
    print("\n🎯 TOP 30 HR PLAYS:")
    print("-"*70)
    for i, row in df.head(30).iterrows():
        print(f"  {row['batter']:20} vs {row['pitcher']:18} "
              f"| {row['pitcher_team']:3} ERA: {row['pitcher_era']:.2f} "
              f"| HR: {row['hr_prob']:.1%} | Hit: {row['hit_prob']:.1%}")
    
    # Save
    output = f"{DATA_DIR}/v40_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output, index=False)
    print(f"\n✅ Saved: {output}")
    
    return df

if __name__ == "__main__":
    generate_predictions()
