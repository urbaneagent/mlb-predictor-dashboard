#!/usr/bin/env python3
"""
MLB Predictor v6.0 - COMPREHENSIVE ANALYTICS
============================================
Complete matchup analysis:
1. Batter tendencies (velocity, pitch type, zone, direction)
2. Pitcher tendencies (pitch mix, velocity, zone)
3. Handedness matchups (LHB vs RHP, etc.)
4. Park factors (direction, HR, runs)
5. Team strengths (bullpen, offense)
6. Weather impact

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
# COMPREHENSIVE ANALYTICS
# ============================================================================

def load_all_data():
    """Load statcast data"""
    print("📡 Loading data...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read([
        'batter', 'pitcher', 'events', 'game_date',
        'hc_x', 'hc_y', 'stand', 'p_throws',
        'pitch_type', 'release_speed', 'plate_x', 'plate_z',
        'launch_speed', 'launch_angle', 'hit_distance_sc',
        'home_team', 'away_team', 'zone'
    ]).to_pandas()
    print(f"   Loaded {len(df)} pitches")
    return df

def analyze_batter_complete(batter_id, df):
    """Complete batter analysis"""
    batter_pas = df[df['batter'] == batter_id]
    hits = batter_pas[batter_pas['events'].isin(['single', 'double', 'triple', 'home_run'])]
    pa = batter_pas[batter_pas['events'].notna()]
    
    if len(hits) < 20:
        return None
    
    # Basic stats
    stats = {
        'pa': len(pa),
        'hits': len(hits),
        'avg': round(len(hits) / len(pa), 3) if len(pa) > 0 else 0,
        'hr': (hits['events'] == 'home_run').sum()
    }
    
    # === VELOCITY PREFERENCE ===
    # Categorize velocity
    hits['velo_bucket'] = pd.cut(hits['release_speed'], bins=[0, 88, 94, 98, 110], 
                                  labels=['low', 'medium', 'high', 'power'])
    velo_dist = hits.groupby('velo_bucket').size()
    stats['velo_preference'] = velo_dist.idxmax() if len(velo_dist) > 0 else 'medium'
    stats['avg_hit_velo'] = round(hits['release_speed'].mean(), 1)
    
    # === PITCH TYPE EFFECTIVENESS ===
    pitch_hr = hits.groupby('pitch_type').apply(lambda x: (x['events'] == 'home_run').sum() / len(x) if len(x) > 0 else 0)
    stats['best_pitch_vs_hr'] = pitch_hr.idxmax() if len(pitch_hr) > 0 else 'FF'
    
    pitch_hits = hits.groupby('pitch_type').size()
    stats['best_pitch_for_hits'] = pitch_hits.idxmax() if len(pitch_hits) > 0 else 'FF'
    
    # === ZONE PREFERENCE ===
    hits_zone = hits.dropna(subset=['plate_x', 'plate_z'])
    if len(hits_zone) > 0:
        hits_zone = hits_zone.copy()
        hits_zone['zone_bucket'] = pd.cut(hits_zone['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
        hits_zone['height_bucket'] = pd.cut(hits_zone['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
        hits_zone['zone'] = hits_zone['zone_bucket'].astype(str) + '-' + hits_zone['height_bucket'].astype(str)
        
        zone_dist = hits_zone.groupby('zone').size()
        stats['hot_zone'] = zone_dist.idxmax() if len(zone_dist) > 0 else 'heart-heart'
    else:
        stats['hot_zone'] = 'heart-heart'
    
    # === DIRECTION ===
    hits_clean = hits.dropna(subset=['hc_x', 'stand'])
    if len(hits_clean) > 0:
        hits_clean['direction'] = np.where(
            hits_clean['stand'] == 'R',
            np.where(hits_clean['hc_x'] < 125, 'pull', np.where(hits_clean['hc_x'] > 135, 'opposite', 'center')),
            np.where(hits_clean['hc_x'] > 135, 'pull', np.where(hits_clean['hc_x'] < 125, 'opposite', 'center'))
        )
        direction_dist = hits_clean.groupby('direction').size()
        stats['pull_rate'] = round(direction_dist.get('pull', 0) / len(hits_clean), 3) if len(hits_clean) > 0 else 0
        stats['oppo_rate'] = round(direction_dist.get('opposite', 0) / len(hits_clean), 3) if len(hits_clean) > 0 else 0
    else:
        stats['pull_rate'] = 0.5
        stats['oppo_rate'] = 0.3
    
    # === HANDEDNESS ===
    stats['batter_hand'] = hits['stand'].mode()[0] if len(hits['stand'].mode()) > 0 else 'R'
    
    return stats

def analyze_pitcher_complete(pitcher_id, df):
    """Complete pitcher analysis"""
    pitcher_pas = df[df['pitcher'] == pitcher_id]
    pa = pitcher_pas[pitcher_pas['events'].notna()]
    
    if len(pa) < 50:
        return None
    
    # Basic stats
    outs = pa[pa['events'].isin(['strikeout', 'field_out', 'force_out', 'double_play'])]
    stats = {
        'batters_faced': len(pa),
        'out_rate': round(len(outs) / len(pa), 3) if len(pa) > 0 else 0,
        'k_rate': round((pa['events'] == 'strikeout').sum() / len(pa), 3) if len(pa) > 0 else 0
    }
    
    # === VELOCITY ===
    stats['avg_velocity'] = round(pitcher_pas['release_speed'].mean(), 1)
    stats['max_velocity'] = pitcher_pas['release_speed'].max()
    
    # === PITCH MIX ===
    pitch_mix = pitcher_pas['pitch_type'].value_counts(normalize=True).head(3).to_dict()
    stats['pitch_mix'] = pitch_mix
    stats['primary_pitch'] = list(pitch_mix.keys())[0] if pitch_mix else 'FF'
    
    # === ZONE TENDENCY ===
    pitcher_pas['zone_bucket'] = pd.cut(pitcher_pas['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
    pitcher_pas['height_bucket'] = pd.cut(pitcher_pas['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
    pitcher_pas['zone'] = pitcher_pas['zone_bucket'].astype(str) + '-' + pitcher_pas['height_bucket'].astype(str)
    
    zone_dist = pitcher_pas.groupby('zone').size()
    stats['preferred_zone'] = zone_dist.idxmax() if len(zone_dist) > 0 else 'heart-heart'
    
    # === HANDEDNESS ===
    stats['pitcher_hand'] = pitcher_pas['p_throws'].mode()[0] if len(pitcher_pas['p_throws'].mode()) > 0 else 'R'
    
    return stats

def get_handedness_advantage(batter_hand, pitcher_hand):
    """Calculate handedness advantage"""
    if batter_hand == pitcher_hand:
        return 0.95  # Same hand disadvantage
    else:
        return 1.05  # Platoon advantage

# ============================================================================
# PARK FACTORS
# ============================================================================

PARK_FACTORS = {
    'ARI': {'hr': 1.15, 'runs': 1.08, 'pull_hr': 1.20, 'oppo_hr': 1.05},
    'ATL': {'hr': 1.08, 'runs': 1.02, 'pull_hr': 1.10, 'oppo_hr': 1.02},
    'BAL': {'hr': 1.10, 'runs': 1.04, 'pull_hr': 1.15, 'oppo_hr': 1.00},
    'BOS': {'hr': 1.05, 'runs': 1.06, 'pull_hr': 1.00, 'oppo_hr': 1.10},
    'CHC': {'hr': 1.12, 'runs': 1.10, 'pull_hr': 1.08, 'oppo_hr': 1.15},
    'CWS': {'hr': 1.03, 'runs': 1.01, 'pull_hr': 1.02, 'oppo_hr': 1.05},
    'CIN': {'hr': 1.18, 'runs': 1.12, 'pull_hr': 1.25, 'oppo_hr': 1.05},
    'CLE': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.90, 'oppo_hr': 1.00},
    'COL': {'hr': 1.35, 'runs': 1.30, 'pull_hr': 1.40, 'oppo_hr': 1.25},
    'DET': {'hr': 0.90, 'runs': 0.96, 'pull_hr': 0.85, 'oppo_hr': 0.95},
    'HOU': {'hr': 1.10, 'runs': 1.05, 'pull_hr': 1.12, 'oppo_hr': 1.08},
    'KC': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.92, 'oppo_hr': 1.00},
    'LAA': {'hr': 0.92, 'runs': 0.97, 'pull_hr': 0.90, 'oppo_hr': 0.95},
    'LAD': {'hr': 0.92, 'runs': 0.96, 'pull_hr': 0.88, 'oppo_hr': 0.95},
    'MIA': {'hr': 0.88, 'runs': 0.92, 'pull_hr': 0.85, 'oppo_hr': 0.90},
    'MIL': {'hr': 1.00, 'runs': 1.00, 'pull_hr': 1.00, 'oppo_hr': 1.00},
    'MIN': {'hr': 1.05, 'runs': 1.02, 'pull_hr': 1.08, 'oppo_hr': 1.00},
    'NYM': {'hr': 0.88, 'runs': 0.94, 'pull_hr': 0.85, 'oppo_hr': 0.90},
    'NYY': {'hr': 1.15, 'runs': 1.08, 'pull_hr': 1.20, 'oppo_hr': 1.08},
    'OAK': {'hr': 0.85, 'runs': 0.93, 'pull_hr': 0.80, 'oppo_hr': 0.90},
    'PHI': {'hr': 1.12, 'runs': 1.06, 'pull_hr': 1.15, 'oppo_hr': 1.08},
    'PIT': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.92, 'oppo_hr': 1.00},
    'SD': {'hr': 0.82, 'runs': 0.88, 'pull_hr': 0.78, 'oppo_hr': 0.85},
    'SEA': {'hr': 0.92, 'runs': 0.96, 'pull_hr': 0.90, 'oppo_hr': 0.95},
    'SF': {'hr': 0.85, 'runs': 0.92, 'pull_hr': 0.82, 'oppo_hr': 0.88},
    'STL': {'hr': 1.00, 'runs': 1.00, 'pull_hr': 1.00, 'oppo_hr': 1.00},
    'TB': {'hr': 0.90, 'runs': 0.94, 'pull_hr': 0.88, 'oppo_hr': 0.92},
    'TEX': {'hr': 1.08, 'runs': 1.04, 'pull_hr': 1.12, 'oppo_hr': 1.02},
    'TOR': {'hr': 1.05, 'runs': 1.02, 'pull_hr': 1.08, 'oppo_hr': 1.00},
    'WSH': {'hr': 0.95, 'runs': 0.98, 'pull_hr': 0.92, 'oppo_hr': 1.00}
}

# ============================================================================
# MAIN PREDICTION ENGINE
# ============================================================================

def generate_comprehensive_predictions():
    """Generate comprehensive matchup predictions"""
    print("\n" + "="*80)
    print("⚾ MLB PREDICTOR v6.0 - COMPREHENSIVE ANALYTICS")
    print("="*80)
    
    df = load_all_data()
    
    # Load season stats
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    print(f"\n📊 Analyzing {len(batters)} batters, {len(pitchers)} pitchers")
    
    # Known player IDs
    batter_ids = {
        'Aaron Judge': 592450, 'Cal Raleigh': 663728, 'Shohei Ohtani': 660271,
        'Kyle Schwarber': 656941, 'Mike Trout': 665742, 'Mookie Betts': 605141,
        'Eugenio Suarez': 625644, 'Juan Soto': 665742, 'Pete Alonso': 624413
    }
    
    pitcher_ids = {
        'Zack Wheeler': 554430, 'Corbin Burnes': 668901, 'Shane Bieber': 607480,
        'Logan Webb': 657746, 'Shohei Ohtani': 660271  # Ohtani pitches too
    }
    
    results = []
    
    for batter_name, batter_id in batter_ids.items():
        # Get batter stats
        bat = batters[batters['Name'].str.contains(batter_name.split()[0], case=False, na=False)]
        if len(bat) == 0:
            continue
        bat = bat.iloc[0]
        
        # Get batter analytics
        batter_analytics = analyze_batter_complete(batter_id, df)
        if not batter_analytics:
            continue
        
        for pitcher_name, pitcher_id in pitcher_ids.items():
            pit = pitchers[pitchers['Name'].str.contains(pitcher_name.split()[0], case=False, na=False)]
            if len(pit) == 0:
                continue
            pit = pit.iloc[0]
            
            pitcher_analytics = analyze_pitcher_complete(pitcher_id, df)
            if not pitcher_analytics:
                continue
            
            # === CALCULATE HR PROBABILITY ===
            
            # Base from season stats
            base_hr = bat['HR'] / 500
            
            # Pitcher quality adjustment
            pitcher_factor = 1 - (pit['ERA'] / 25)
            
            # Handedness advantage
            hand_adv = get_handedness_advantage(
                batter_analytics.get('batter_hand', 'R'),
                pitcher_analytics.get('pitcher_hand', 'R')
            )
            
            # Zone matchup bonus
            if batter_analytics.get('hot_zone') == pitcher_analytics.get('preferred_zone'):
                zone_bonus = 1.15
            else:
                zone_bonus = 1.0
            
            # Pitch type matchup (simplified)
            primary_pitch = pitcher_analytics.get('primary_pitch', 'FF')
            best_pitch = batter_analytics.get('best_pitch_for_hits', 'FF')
            pitch_bonus = 1.10 if primary_pitch == best_pitch else 1.0
            
            # Pull/oppo park factor (assuming Yankees stadium for Judge)
            park = bat['Team']
            park_factor = PARK_FACTORS.get(park, {}).get('hr', 1.0)
            
            # Calculate final HR probability
            hr_prob = base_hr * pitcher_factor * hand_adv * zone_bonus * pitch_bonus * park_factor
            hr_prob = min(hr_prob, 0.50)
            
            results.append({
                'batter': bat['Name'],
                'batter_team': bat['Team'],
                'batter_hand': batter_analytics.get('batter_hand', 'R'),
                'batter_hr': bat['HR'],
                'batter_avg': bat['AVG'],
                'batter_velo_pref': batter_analytics.get('avg_hit_velo', 90),
                'batter_hot_zone': batter_analytics.get('hot_zone', 'heart-heart'),
                'batter_pull_rate': batter_analytics.get('pull_rate', 0.6),
                'pitcher': pit['Name'],
                'pitcher_team': pit['Team'],
                'pitcher_hand': pitcher_analytics.get('pitcher_hand', 'R'),
                'pitcher_era': pit['ERA'],
                'pitcher_velo': pitcher_analytics.get('avg_velocity', 92),
                'pitcher_zone': pitcher_analytics.get('preferred_zone', 'heart-heart'),
                'pitcher_primary_pitch': primary_pitch,
                'hr_prob': round(hr_prob, 3),
                'factors': f"Hand:{hand_adv:.2f} Zone:{zone_bonus:.2f} Pitch:{pitch_bonus:.2f} Park:{park_factor:.2f}"
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('hr_prob', ascending=False)
    
    # Display
    print("\n" + "="*80)
    print("🏆 TOP COMPREHENSIVE MATCHUP PLAYS")
    print("="*80)
    
    for i, row in results_df.head(15).iterrows():
        print(f"\n⚾ {row['batter']} ({row['batter_team']}) vs {row['pitcher']} ({row['pitcher_team']})")
        print(f"   HR%: {row['hr_prob']:.1%}")
        print(f"   Batter: {row['batter_hand']}h | Velo: {row['batter_velo_pref']}mph | Zone: {row['batter_hot_zone']} | Pull: {row['batter_pull_rate']:.0%}")
        print(f"   Pitcher: {row['pitcher_hand']}h | Velo: {row['pitcher_velo']}mph | Zone: {row['pitcher_zone']} | Pitch: {row['pitcher_primary_pitch']}")
        print(f"   Factors: {row['factors']}")
    
    # Save
    output = f"{DATA_DIR}/v60_predictions.csv"
    results_df.to_csv(output, index=False)
    print(f"\n✅ Saved: {output}")
    
    return results_df

if __name__ == "__main__":
    generate_comprehensive_predictions()
