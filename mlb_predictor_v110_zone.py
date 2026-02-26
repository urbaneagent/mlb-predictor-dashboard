#!/usr/bin/env python3
"""
MLB PREDICTOR v11.0 - ZONE MATCHUP ANALYSIS
============================================
Advanced zone analysis:
- Pitcher zone tendency (where they throw most)
- Batter zone preference (where they hit best)
- Mismatch exploitation
- Inside/outside, High/low splits
- Pitch type in different zones

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pybaseball import playerid_reverse_lookup

DATA_DIR = "/Users/mikeross/MLB_Predictions"
PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

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

def load_zone_data():
    """Load statcast with zone data"""
    print("📡 Loading zone data...")
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read([
        'batter', 'pitcher', 'events', 
        'plate_x', 'plate_z', 'release_speed',
        'pitch_type', 'stand', 'p_throws'
    ]).to_pandas()
    
    # Create zone buckets
    df['h_zone'] = pd.cut(df['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
    df['v_zone'] = pd.cut(df['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
    df['zone'] = df['h_zone'].astype(str) + '-' + df['v_zone'].astype(str)
    
    print(f"   Loaded {len(df)} pitches")
    return df

def get_batter_zone_profile(batter_id, df):
    """Get batter zone preferences"""
    batter_pas = df[df['batter'] == batter_id]
    hits = batter_pas[batter_pas['events'].isin(['single', 'double', 'triple', 'home_run'])]
    
    if len(hits) < 50:
        return None
    
    # Zone distribution
    h_dist = hits['h_zone'].value_counts(normalize=True).to_dict()
    v_dist = hits['v_zone'].value_counts(normalize=True).to_dict()
    zone_dist = hits['zone'].value_counts(normalize=True).to_dict()
    
    # Best zone for hits
    best_zone = max(zone_dist, key=zone_dist.get)
    
    # Pitch type in each zone
    pitch_by_zone = hits.groupby(['zone', 'pitch_type']).size().unstack(fill_value=0)
    
    return {
        'total_hits': len(hits),
        'h_preference': h_dist,
        'v_preference': v_dist,
        'best_zone': best_zone,
        'zone_dist': zone_dist
    }

def get_pitcher_zone_profile(pitcher_id, df):
    """Get pitcher zone tendencies"""
    pitcher_pas = df[df['pitcher'] == pitcher_id]
    
    if len(pitcher_pas) < 100:
        return None
    
    # Zone distribution
    zone_dist = pitcher_pas['zone'].value_counts(normalize=True).to_dict()
    h_dist = pitcher_pas['h_zone'].value_counts(normalize=True).to_dict()
    v_dist = pitcher_pas['v_zone'].value_counts(normalize=True).to_dict()
    
    # Preferred zone
    pref_zone = max(zone_dist, key=zone_dist.get)
    
    return {
        'total_pitches': len(pitcher_pas),
        'h_tendency': h_dist,
        'v_tendency': v_dist,
        'preferred_zone': pref_zone,
        'zone_dist': zone_dist
    }

def find_zone_mismatches():
    """Find exploitable zone mismatches"""
    print("\n🔍 Analyzing zone mismatches...")
    
    df = load_zone_data()
    
    # Get top pitchers by HRs allowed
    hrs = df[df['events'] == 'home_run']
    top_hr_pitchers = hrs.groupby('pitcher').size().nlargest(20).index.tolist()
    
    # Get batter IDs
    batter_names = pd.read_csv(f"{DATA_DIR}/batter_names.csv")
    name_lookup = dict(zip(batter_names['batter_id'], batter_names['batter_name']))
    
    # Known MLB IDs
    known_batters = {
        'Aaron Judge': 592450, 'Shohei Ohtani': 660271, 'Kyle Schwarber': 656941,
        'Cal Raleigh': 663728, 'Mike Trout': 665742, 'Mookie Betts': 605141
    }
    
    results = []
    
    for pitcher_id in top_hr_pitchers[:10]:
        pit_profile = get_pitcher_zone_profile(pitcher_id, df)
        if not pit_profile:
            continue
        
        pit_zone = pit_profile['preferred_zone']
        
        for batter_name, batter_id in known_batters.items():
            bat_profile = get_batter_zone_profile(batter_id, df)
            if not bat_profile:
                continue
            
            bat_zone = bat_profile['best_zone']
            
            # Calculate mismatch bonus
            if pit_zone == bat_zone:
                zone_bonus = 1.20  # 20% boost
                matchup = "MATCH"
            else:
                zone_bonus = 0.95
                matchup = f"MISMATCH: Batter={bat_zone}, Pitcher={pit_zone}"
            
            results.append({
                'batter': batter_name,
                'batter_id': batter_id,
                'pitcher_id': pitcher_id,
                'batter_best_zone': bat_zone,
                'pitcher_preferred_zone': pit_zone,
                'zone_match': matchup,
                'zone_bonus': zone_bonus
            })
    
    return pd.DataFrame(results)

def run():
    print("\n" + "="*80)
    print("⚾ MLB PREDICTOR v11.0 - ZONE MATCHUP ANALYSIS")
    print("="*80)
    
    mismatches = find_zone_mismatches()
    
    print("\n" + "="*80)
    print("🎯 ZONE MISMATCH EXPLOITS")
    print("="*80)
    
    # Show results
    for _, row in mismatches.head(20).iterrows():
        bonus = "✅" if row['zone_bonus'] > 1.0 else "⚠️"
        print(f"\n{bonus} {row['batter']} vs Pitcher {row['pitcher_id']}")
        print(f"   Batter best zone: {row['batter_best_zone']}")
        print(f"   Pitcher throws to: {row['pitcher_preferred_zone']}")
        print(f"   Zone bonus: +{int((row['zone_bonus']-1)*100)}%")
    
    # Save
    mismatches.to_csv(f"{DATA_DIR}/zone_mismatches.csv", index=False)
    print(f"\n✅ Saved: {DATA_DIR}/zone_mismatches.csv")
    
    return mismatches

if __name__ == "__main__":
    run()
