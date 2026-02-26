#!/usr/bin/env python3
"""
MLB Predictor v5.0 - ADVANCED MATCHUP ANALYTICS
==============================================
Features:
- Batter hot zones (plate_x, plate_z)
- Pitcher tendencies by zone
- Pitch type effectiveness vs batter
- Velocity preferences
- Historical matchup data

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = "/Users/mikeross/MLB_Predictions"
PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

def load_data():
    """Load statcast data"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read([
        'batter', 'pitcher', 'events', 'game_date',
        'pitch_type', 'zone', 'plate_x', 'plate_z', 
        'release_speed', 'effective_speed', 'stand', 'p_throws'
    ]).to_pandas()
    return df

def get_batter_profile(batter_id, df):
    """Get comprehensive batter profile"""
    batter_pas = df[df['batter'] == batter_id]
    
    if len(batter_pas) == 0:
        return None
    
    # Hits
    hits = batter_pas[batter_pas['events'].isin(['single', 'double', 'triple', 'home_run'])]
    total_pa = len(batter_pas[batter_pas['events'].notna()])
    
    if len(hits) == 0:
        return None
    
    # Zone analysis
    hits['zone_horiz'] = pd.cut(hits['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
    hits['zone_vert'] = pd.cut(hits['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
    
    # Hot zone
    zone_counts = hits.groupby(['zone_horiz', 'zone_vert']).size()
    hot_zone = zone_counts.idxmax() if len(zone_counts) > 0 else ('heart', 'heart')
    
    # Pitch type effectiveness
    pitch_rates = hits.groupby('pitch_type').size() / len(hits)
    best_pitch = pitch_rates.idxmax() if len(pitch_rates) > 0 else 'FF'
    
    # Velocity
    avg_velo = hits['release_speed'].mean()
    
    return {
        'batter_id': batter_id,
        'total_pa': total_pa,
        'hits': len(hits),
        'avg': round(len(hits) / total_pa, 3) if total_pa > 0 else 0,
        'hot_zone': f"{hot_zone[0]}-{hot_zone[1]}",
        'best_pitch': best_pitch,
        'avg_velocity': round(avg_velo, 1),
        'pitch_rates': pitch_rates.head(5).to_dict()
    }

def get_pitcher_profile(pitcher_id, df):
    """Get comprehensive pitcher profile"""
    pitcher_pas = df[df['pitcher'] == pitcher_id]
    
    if len(pitcher_pas) == 0:
        return None
    
    total_batters = len(pitcher_pas[pitcher_pas['events'].notna()])
    
    # Zone tendencies
    pitcher_pas['zone_horiz'] = pd.cut(pitcher_pas['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
    pitcher_pas['zone_vert'] = pd.cut(pitcher_pas['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
    
    zone_dist = pitcher_pas.groupby(['zone_horiz', 'zone_vert']).size()
    pref_zone = zone_dist.idxmax() if len(zone_dist) > 0 else ('heart', 'heart')
    
    # Pitch mix
    pitch_mix = pitcher_pas['pitch_type'].value_counts(normalize=True).head(4).to_dict()
    
    # Velocity
    avg_velo = pitcher_pas['release_speed'].mean()
    
    # Out rate
    outs = pitcher_pas[pitcher_pas['events'].isin(['strikeout', 'field_out', 'force_out', 'double_play'])]
    out_rate = len(outs) / total_batters if total_batters > 0 else 0
    
    return {
        'pitcher_id': pitcher_id,
        'batters_faced': total_batters,
        'out_rate': round(out_rate, 3),
        'preferred_zone': f"{pref_zone[0]}-{pref_zone[1]}",
        'pitch_mix': pitch_mix,
        'avg_velocity': round(avg_velo, 1)
    }

def analyze_matchup(batter_id, pitcher_id, df):
    """Full matchup analysis"""
    batter = get_batter_profile(batter_id, df)
    pitcher = get_pitcher_profile(pitcher_id, df)
    
    if not batter or not pitcher:
        return None
    
    # Calculate advantages
    advantages = []
    
    # Zone exploitation
    bz = batter['hot_zone'].split('-')
    pz = pitcher['preferred_zone'].split('-')
    
    zone_advantage = 0
    if bz[0] != pz[0] or bz[1] != pz[1]:
        zone_advantage = 0.1
        advantages.append(f"Batter hits {batter['hot_zone']}, pitcher uses {pitcher['preferred_zone']}")
    
    # Pitch type advantage
    if pitcher['pitch_mix']:
        pitcher_top_pitch = list(pitcher['pitch_mix'].keys())[0]
        if pitcher_top_pitch in batter['pitch_rates']:
            # Batter hits this pitch type well
            advantages.append(f"Batter hits {pitcher_top_pitch} well ({batter['pitch_rates'].get(pitcher_top_pitch, 0)*100:.0f}% of his hits)")
    
    # Velocity
    velo_diff = batter['avg_velocity'] - pitcher['avg_velocity']
    if velo_diff < -5:
        advantages.append(f"Pitcher throws {abs(velo_diff):.0f} mph harder - power pitcher vs contact hitter")
    elif velo_diff > 5:
        advantages.append(f"Batter handles {batter['avg_velocity']:.0f} mph well")
    
    return {
        'batter': batter,
        'pitcher': pitcher,
        'zone_advantage': zone_advantage,
        'advantages': advantages
    }

# ============================================================================
# PREDICTION WITH MATCHUPS
# ============================================================================

def predict_with_matchups():
    """Generate predictions with matchup analytics"""
    print("\n" + "="*70)
    print("⚾ MLB PREDICTOR v5.0 - ADVANCED MATCHUP ANALYTICS")
    print("="*70)
    
    df = load_data()
    
    # Load season stats
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    print(f"\n📊 Loaded {len(df)} pitches, {len(batters)} batters, {len(pitchers)} pitchers")
    
    # Top HR hitters
    top_hr = batters.nlargest(20, 'HR')[['IDfg', 'Name', 'Team', 'AVG', 'OPS', 'HR']].copy()
    
    # Sample pitchers
    sample_pitchers = pitchers.nlargest(10, 'IP')[['IDfg', 'Name', 'Team', 'ERA', 'IP']].copy()
    
    predictions = []
    
    print("\n" + "="*70)
    print("🏆 TOP HR PLAYS WITH MATCHUP ANALYSIS")
    print("="*70)
    
    for _, batter_row in top_hr.iterrows():
        batter_id = batter_row['IDfg']
        batter_name = batter_row['Name']
        
        batter_profile = get_batter_profile(batter_id, df)
        
        if not batter_profile:
            continue
        
        for _, pitcher_row in sample_pitchers.iterrows():
            pitcher_id = pitcher_row['IDfg']
            pitcher_name = pitcher_row['Name']
            
            pitcher_profile = get_pitcher_profile(pitcher_id, df)
            
            if not pitcher_profile:
                continue
            
            # Matchup analysis
            matchup = analyze_matchup(batter_id, pitcher_id, df)
            
            if matchup:
                # Base HR probability
                hr_prob = (batter_row['HR'] / 500) * (1 - pitcher_row['ERA'] / 20)
                
                # Adjust for zone matchup
                if matchup['zone_advantage'] > 0:
                    hr_prob *= (1 + matchup['zone_advantage'])
                
                predictions.append({
                    'batter': batter_name,
                    'batter_team': batter_row['Team'],
                    'batter_avg': batter_row['AVG'],
                    'batter_hr': batter_row['HR'],
                    'batter_hot_zone': batter_profile['hot_zone'],
                    'pitcher': pitcher_name,
                    'pitcher_team': pitcher_row['Team'],
                    'pitcher_era': pitcher_row['ERA'],
                    'pitcher_zone': pitcher_profile['preferred_zone'],
                    'pitcher_pitch_mix': list(pitcher_profile['pitch_mix'].keys())[0] if pitcher_profile['pitch_mix'] else 'N/A',
                    'matchup_advantage': len(matchup['advantages']),
                    'hr_prob': round(min(hr_prob, 0.5), 3),
                    'notes': matchup['advantages'][:2] if matchup['advantages'] else ['Neutral matchup']
                })
    
    # Sort by HR probability
    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values('hr_prob', ascending=False)
    
    # Display top 20
    print("\n🎯 TOP 20 HR PLAYS:")
    print("-"*70)
    for i, row in df_pred.head(20).iterrows():
        print(f"  {row['batter']:20} vs {row['pitcher']:18} | "
              f"HR: {row['hr_prob']:.1%} | "
              f"Zone: {row['batter_hot_zone'][:5]} vs {row['pitcher_zone'][:5]} | "
              f"Matchup: {row['matchup_advantage']}")
        if row['notes'] and row['notes'][0] != 'Neutral matchup':
            print(f"    → {row['notes'][0]}")
    
    # Save
    output = f"{DATA_DIR}/v50_predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    df_pred.to_csv(output, index=False)
    print(f"\n✅ Saved: {output}")
    
    return df_pred

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    predict_with_matchups()
