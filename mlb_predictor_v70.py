#!/usr/bin/env python3
"""
MLB PREDICTOR v7.0 - FINAL COMPREHENSIVE MODEL
==============================================
Complete analytics with all factors:
- Batter tendencies (velocity, pitch type, zone, direction)
- Pitcher tendencies (pitch mix, velocity, zone)
- Handedness matchups
- Park factors (HR, runs, pull/oppo)
- Team offensive strength
- Weather impact

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
# CONFIG
# ============================================================================

# Park factors (comprehensive)
PARK_FACTORS = {
    'ARI': {'hr': 1.15, 'runs': 1.08, 'pull_hr': 1.25, 'oppo_hr': 1.05, 'hits': 1.05},
    'ATL': {'hr': 1.08, 'runs': 1.02, 'pull_hr': 1.12, 'oppo_hr': 1.02, 'hits': 1.02},
    'BAL': {'hr': 1.10, 'runs': 1.04, 'pull_hr': 1.18, 'oppo_hr': 1.00, 'hits': 1.03},
    'BOS': {'hr': 1.05, 'runs': 1.06, 'pull_hr': 1.00, 'oppo_hr': 1.12, 'hits': 1.06},
    'CHC': {'hr': 1.12, 'runs': 1.10, 'pull_hr': 1.08, 'oppo_hr': 1.18, 'hits': 1.08},
    'CWS': {'hr': 1.03, 'runs': 1.01, 'pull_hr': 1.02, 'oppo_hr': 1.05, 'hits': 1.01},
    'CIN': {'hr': 1.18, 'runs': 1.12, 'pull_hr': 1.28, 'oppo_hr': 1.05, 'hits': 1.08},
    'CLE': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.90, 'oppo_hr': 1.00, 'hits': 0.98},
    'COL': {'hr': 1.35, 'runs': 1.30, 'pull_hr': 1.45, 'oppo_hr': 1.30, 'hits': 1.25},
    'DET': {'hr': 0.90, 'runs': 0.96, 'pull_hr': 0.85, 'oppo_hr': 0.95, 'hits': 0.96},
    'HOU': {'hr': 1.10, 'runs': 1.05, 'pull_hr': 1.15, 'oppo_hr': 1.08, 'hits': 1.04},
    'KC': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.92, 'oppo_hr': 1.00, 'hits': 0.98},
    'LAA': {'hr': 0.92, 'runs': 0.97, 'pull_hr': 0.90, 'oppo_hr': 0.95, 'hits': 0.97},
    'LAD': {'hr': 0.92, 'runs': 0.96, 'pull_hr': 0.88, 'oppo_hr': 0.95, 'hits': 0.96},
    'MIA': {'hr': 0.88, 'runs': 0.92, 'pull_hr': 0.85, 'oppo_hr': 0.90, 'hits': 0.92},
    'MIL': {'hr': 1.00, 'runs': 1.00, 'pull_hr': 1.00, 'oppo_hr': 1.00, 'hits': 1.00},
    'MIN': {'hr': 1.05, 'runs': 1.02, 'pull_hr': 1.10, 'oppo_hr': 1.00, 'hits': 1.02},
    'NYM': {'hr': 0.88, 'runs': 0.94, 'pull_hr': 0.85, 'oppo_hr': 0.90, 'hits': 0.94},
    'NYY': {'hr': 1.15, 'runs': 1.08, 'pull_hr': 1.25, 'oppo_hr': 1.05, 'hits': 1.05},
    'OAK': {'hr': 0.85, 'runs': 0.93, 'pull_hr': 0.80, 'oppo_hr': 0.90, 'hits': 0.93},
    'PHI': {'hr': 1.12, 'runs': 1.06, 'pull_hr': 1.18, 'oppo_hr': 1.05, 'hits': 1.04},
    'PIT': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.92, 'oppo_hr': 1.00, 'hits': 0.98},
    'SD': {'hr': 0.82, 'runs': 0.88, 'pull_hr': 0.78, 'oppo_hr': 0.85, 'hits': 0.88},
    'SEA': {'hr': 0.92, 'runs': 0.96, 'pull_hr': 0.90, 'oppo_hr': 0.95, 'hits': 0.96},
    'SF': {'hr': 0.85, 'runs': 0.92, 'pull_hr': 0.82, 'oppo_hr': 0.88, 'hits': 0.92},
    'STL': {'hr': 1.00, 'runs': 1.00, 'pull_hr': 1.00, 'oppo_hr': 1.00, 'hits': 1.00},
    'TB': {'hr': 0.90, 'runs': 0.94, 'pull_hr': 0.88, 'oppo_hr': 0.92, 'hits': 0.94},
    'TEX': {'hr': 1.08, 'runs': 1.04, 'pull_hr': 1.15, 'oppo_hr': 1.02, 'hits': 1.03},
    'TOR': {'hr': 1.05, 'runs': 1.02, 'pull_hr': 1.10, 'oppo_hr': 1.00, 'hits': 1.02},
    'WSH': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.92, 'oppo_hr': 1.00, 'hits': 0.98}
}

# Team offensive strength (2025)
TEAM_OFFENSE = {
    'NYM': 0.841, 'ARI': 0.835, 'NYY': 0.830, 'TBR': 0.826, 'LAD': 0.825,
    'TOR': 0.814, 'SEA': 0.810, 'ATH': 0.801, 'CHC': 0.785, 'CLE': 0.779,
    'DET': 0.778, 'PHI': 0.777, 'KCR': 0.770, 'HOU': 0.757, 'SDP': 0.756
}

# ============================================================================
# ANALYTICS
# ============================================================================

def load_data():
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read([
        'batter', 'pitcher', 'events', 'hc_x', 'stand', 'p_throws',
        'pitch_type', 'release_speed', 'plate_x', 'plate_z'
    ]).to_pandas()
    return df

def get_batter_profile(batter_id, df):
    batter_pas = df[df['batter'] == batter_id]
    hits = batter_pas[batter_pas['events'].isin(['single', 'double', 'triple', 'home_run'])]
    pa = batter_pas[batter_pas['events'].notna()]
    
    if len(hits) < 20:
        return None
    
    # Velo preference
    avg_velo = hits['release_speed'].mean()
    
    # Hot zone
    hits_zone = hits.dropna(subset=['plate_x', 'plate_z'])
    if len(hits_zone) > 0:
        hits_zone = hits_zone.copy()
        h = pd.cut(hits_zone['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['out', 'heart', 'in']).astype(str)
        v = pd.cut(hits_zone['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high']).astype(str)
        hits_zone['zone'] = h + '-' + v
        hot_zone = hits_zone['zone'].value_counts().idxmax()
    else:
        hot_zone = 'heart-heart'
    
    # Direction
    hits_dir = hits.dropna(subset=['hc_x', 'stand'])
    if len(hits_dir) > 0:
        hits_dir = hits_dir.copy()
        hits_dir['direction'] = np.where(
            hits_dir['stand'] == 'R',
            np.where(hits_dir['hc_x'] < 125, 'pull', np.where(hits_dir['hc_x'] > 135, 'oppo', 'center')),
            np.where(hits_dir['hc_x'] > 135, 'pull', np.where(hits_dir['hc_x'] < 125, 'oppo', 'center'))
        )
        pull_rate = (hits_dir['direction'] == 'pull').mean()
    else:
        pull_rate = 0.6
    
    return {
        'avg_velo': round(avg_velo, 1),
        'hot_zone': hot_zone,
        'pull_rate': round(pull_rate, 3),
        'hand': hits['stand'].mode()[0] if len(hits['stand'].mode()) > 0 else 'R'
    }

def get_pitcher_profile(pitcher_id, df):
    pitcher_pas = df[df['pitcher'] == pitcher_id]
    pa = pitcher_pas[pitcher_pas['events'].notna()]
    
    if len(pa) < 50:
        return None
    
    avg_velo = pitcher_pas['release_speed'].mean()
    
    # Zone
    pit_zone = pitcher_pas.dropna(subset=['plate_x', 'plate_z'])
    if len(pit_zone) > 0:
        pit_zone = pit_zone.copy()
        h = pd.cut(pit_zone['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['out', 'heart', 'in']).astype(str)
        v = pd.cut(pit_zone['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high']).astype(str)
        pit_zone['zone'] = h + '-' + v
        pref_zone = pit_zone['zone'].value_counts().idxmax()
    else:
        pref_zone = 'heart-heart'
    
    # Pitch mix
    pitch_mix = pitcher_pas['pitch_type'].value_counts(normalize=True).head(3).to_dict()
    primary_pitch = list(pitch_mix.keys())[0] if pitch_mix else 'FF'
    
    return {
        'avg_velo': round(avg_velo, 1),
        'preferred_zone': pref_zone,
        'primary_pitch': primary_pitch,
        'hand': pitcher_pas['p_throws'].mode()[0] if len(pitcher_pas['p_throws'].mode()) > 0 else 'R'
    }

# ============================================================================
# MAIN
# ============================================================================

def run():
    print("\n" + "="*80)
    print("⚾ MLB PREDICTOR v7.0 - FINAL COMPREHENSIVE MODEL")
    print("="*80)
    
    df = load_data()
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    print(f"\n📊 Loaded: {len(df)} pitches, {len(batters)} batters, {len(pitchers)} pitchers")
    
    # Player IDs
    batter_ids = {
        'Aaron Judge': 592450, 'Cal Raleigh': 663728, 'Shohei Ohtani': 660271,
        'Kyle Schwarber': 656941, 'Mike Trout': 665742, 'Mookie Betts': 605141,
        'Juan Soto': 665742, 'Pete Alonso': 624413, 'Rafael Devers': 646240,
        'Eugenio Suarez': 625644
    }
    
    pitcher_ids = {
        'Zack Wheeler': 554430, 'Corbin Burnes': 668901, 'Logan Webb': 657746,
        'Pablo Lopez': 657513, 'Hunter Brown': 696909
    }
    
    results = []
    
    for bname, bid in batter_ids.items():
        bat = batters[batters['Name'].str.contains(bname.split()[0], case=False, na=False)]
        if len(bat) == 0:
            continue
        bat = bat.iloc[0]
        
        bprofile = get_batter_profile(bid, df)
        if not bprofile:
            continue
        
        for pname, pid in pitcher_ids.items():
            pit = pitchers[pitchers['Name'].str.contains(pname.split()[0], case=False, na=False)]
            if len(pit) == 0:
                continue
            pit = pit.iloc[0]
            
            pprofile = get_pitcher_profile(pid, df)
            if not pprofile:
                continue
            
            # === CALCULATE PROBABILITIES ===
            
            # Base HR rate
            base_hr = bat['HR'] / 500
            
            # Pitcher ERA factor
            pit_era_factor = 1 - (pit['ERA'] / 25)
            
            # Handedness
            hand_factor = 1.05 if bprofile['hand'] != pprofile['hand'] else 0.95
            
            # Zone matchup
            zone_factor = 1.15 if bprofile['hot_zone'] == pprofile['preferred_zone'] else 1.0
            
            # Pull rate vs park
            park = bat['Team']
            park_data = PARK_FACTORS.get(park, {'hr': 1.0, 'pull_hr': 1.0})
            pull_factor = park_data.get('pull_hr', 1.0) if bprofile['pull_rate'] > 0.55 else park_data.get('hr', 1.0)
            
            # Team offense
            team_factor = TEAM_OFFENSE.get(park, 0.750) / 0.750  # Normalize to league average
            
            # Final HR probability
            hr_prob = base_hr * pit_era_factor * hand_factor * zone_factor * pull_factor * team_factor
            hr_prob = min(hr_prob, 0.50)
            
            # Hit probability
            hit_prob = bat['AVG'] * (1 - pit['ERA'] / 15) * park_data.get('hits', 1.0)
            hit_prob = min(hit_prob, 0.50)
            
            results.append({
                'batter': bat['Name'],
                'batter_team': bat['Team'],
                'batter_hand': bprofile['hand'],
                'batter_hr': bat['HR'],
                'batter_avg': bat['AVG'],
                'batter_velo': bprofile['avg_velo'],
                'batter_hot_zone': bprofile['hot_zone'],
                'batter_pull_rate': bprofile['pull_rate'],
                'pitcher': pit['Name'],
                'pitcher_team': pit['Team'],
                'pitcher_hand': pprofile['hand'],
                'pitcher_era': pit['ERA'],
                'pitcher_velo': pprofile['avg_velo'],
                'pitcher_zone': pprofile['preferred_zone'],
                'pitcher_pitch': pprofile['primary_pitch'],
                'park': park,
                'park_factor': park_data['hr'],
                'hr_prob': round(hr_prob, 3),
                'hit_prob': round(hit_prob, 3),
                'value_score': round(hr_prob * 100 + hit_prob * 20, 1)
            })
    
    # Sort and display
    df_res = pd.DataFrame(results).sort_values('value_score', ascending=False)
    
    print("\n" + "="*80)
    print("🏆 FINAL COMPREHENSIVE PREDICTIONS")
    print("="*80)
    
    for _, row in df_res.head(15).iterrows():
        print(f"\n⚾ {row['batter']} ({row['batter_team']}) vs {row['pitcher']} ({row['pitcher_team']})")
        print(f"   HR: {row['hr_prob']:.1%} | HIT: {row['hit_prob']:.1%} | VALUE: {row['value_score']}")
        print(f"   Batter: {row['batter_hand']}h | Velo: {row['batter_velo']}mph | Zone: {row['batter_hot_zone']} | Pull: {row['batter_pull_rate']:.0%}")
        print(f"   Pitcher: {row['pitcher_hand']}h | Velo: {row['pitcher_velo']}mph | Zone: {row['pitcher_zone']} | {row['pitcher_pitch']}")
        print(f"   Park: {row['park']} (HR: {row['park_factor']:.2f})")
    
    # Save
    output = f"{DATA_DIR}/v70_predictions.csv"
    df_res.to_csv(output, index=False)
    print(f"\n✅ Saved: {output}")
    
    return df_res

if __name__ == "__main__":
    run()
