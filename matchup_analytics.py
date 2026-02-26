#!/usr/bin/env python3
"""
MLB Matchup Analyzer v1.0
=========================
Advanced analytics:
- Batter hot zones (where they hit)
- Pitcher tendencies (where they pitch)
- Pitch type effectiveness
- Velocity preferences
- Zone exploitation

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

DATA_DIR = "/Users/mikeross/MLB_Predictions"
PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

# ============================================================================
# ANALYZER
# ============================================================================

def load_statcast():
    """Load statcast data"""
    print("📡 Loading Statcast data...")
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read([
        'batter', 'pitcher', 'events', 'game_date',
        'pitch_type', 'zone', 'plate_x', 'plate_z', 
        'release_speed', 'effective_speed', 'stand', 'p_throws'
    ]).to_pandas()
    print(f"   Loaded {len(df)} pitches")
    return df

def analyze_batter(batter_id, df):
    """Analyze batter tendencies"""
    batter_pas = df[df['batter'] == batter_id]
    
    if len(batter_pas) == 0:
        return None
    
    # Get hits
    hits = batter_pas[batter_pas['events'].isin(['single', 'double', 'triple', 'home_run'])]
    
    # Zone analysis
    hits['zone_bucket'] = pd.cut(hits['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
    hits['height_bucket'] = pd.cut(hits['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
    
    # Pitch type analysis
    pitch_stats = hits.groupby('pitch_type').agg({
        'events': 'count',
    }).rename(columns={'events': 'hits'})
    
    # Total PA for this batter
    total_pa = len(batter_pas[batter_pas['events'].notna()])
    pitch_stats['hit_rate'] = pitch_stats['hits'] / total_pa
    
    # Velocity preference
    velo_preference = hits['release_speed'].mean()
    
    # Hot zone
    zone_stats = hits.groupby(['zone_bucket', 'height_bucket']).size()
    if len(zone_stats) > 0:
        hot_zone = zone_stats.idxmax()
    else:
        hot_zone = ('heart', 'heart')
    
    return {
        'total_pa': total_pa,
        'total_hits': len(hits),
        'avg_velocity': round(velo_preference, 1),
        'hot_zone': f"{hot_zone[0]}-{hot_zone[1]}",
        'pitch_preferences': pitch_stats.sort_values('hit_rate', ascending=False).head(5)['hit_rate'].to_dict()
    }

def analyze_pitcher(pitcher_id, df):
    """Analyze pitcher tendencies"""
    pitcher_pas = df[df['pitcher'] == pitcher_id]
    
    if len(pitcher_pas) == 0:
        return None
    
    # Where do they pitch?
    pitcher_pas['zone_bucket'] = pd.cut(pitcher_pas['plate_x'], bins=[-3, -0.7, 0.7, 3], labels=['outside', 'heart', 'inside'])
    pitcher_pas['height_bucket'] = pd.cut(pitcher_pas['plate_z'], bins=[-3, 1.5, 3, 5], labels=['low', 'heart', 'high'])
    
    # Zone distribution
    zone_dist = pitcher_pas.groupby(['zone_bucket', 'height_bucket']).size()
    preferred_zone = zone_dist.idxmax() if len(zone_dist) > 0 else ('heart', 'heart')
    
    # Pitch type distribution
    pitch_mix = pitcher_pas['pitch_type'].value_counts(normalize=True).head(5).to_dict()
    
    # Average velocity
    avg_velo = pitcher_pas['release_speed'].mean()
    
    # Get results
    outs = pitcher_pas[pitcher_pas['events'].isin(['strikeout', 'field_out', 'force_out', 'double_play'])]
    out_rate = len(outs) / len(pitcher_pas[pitcher_pas['events'].notna()]) if len(pitcher_pas[pitcher_pas['events'].notna()]) > 0 else 0
    
    return {
        'total_batters_faced': len(pitcher_pas[pitcher_pas['events'].notna()]),
        'out_rate': round(out_rate, 3),
        'avg_velocity': round(avg_velo, 1),
        'preferred_zone': f"{preferred_zone[0]}-{preferred_zone[1]}",
        'pitch_mix': pitch_mix
    }

def find_matchup_advantages(batter_id, pitcher_id, df):
    """Find exploitable matchup advantages"""
    batter = analyze_batter(batter_id, df)
    pitcher = analyze_pitcher(pitcher_id, df)
    
    if not batter or not pitcher:
        return None
    
    advantages = []
    
    # Zone exploitation
    batter_zone = batter['hot_zone'].split('-')
    pitcher_zone = pitcher['preferred_zone'].split('-')
    
    if batter_zone != pitcher_zone:
        advantages.append({
            'type': 'zone_mismatch',
            'description': f"Batter hits {batter['hot_zone']}, pitcher pitches {pitcher['preferred_zone']}",
            'impact': 'medium'
        })
    
    # Velocity exploitation
    velo_diff = batter['avg_velocity'] - pitcher['avg_velocity']
    if abs(velo_diff) > 3:
        advantages.append({
            'type': 'velocity_mismatch',
            'description': f"Batter prefers {batter['avg_velocity']}mph, pitcher throws {pitcher['avg_velocity']}mph",
            'impact': 'low'
        })
    
    return {
        'batter': batter,
        'pitcher': pitcher,
        'advantages': advantages
    }

# ============================================================================
# SAVE ANALYTICS
# ============================================================================

def build_analytics():
    """Build comprehensive matchup analytics"""
    print("\n" + "="*60)
    print("⚾ MLB MATCHUP ANALYTICS v1.0")
    print("="*60)
    
    df = load_statcast()
    
    # Get unique batters and pitchers with enough data
    batter_counts = df.groupby('batter').apply(lambda x: len(x[x['events'].notna()]))
    qualified_batters = batter_counts[batter_counts >= 100].index.tolist()
    
    pitcher_counts = df.groupby('pitcher').apply(lambda x: len(x[x['events'].notna()]))
    qualified_pitchers = pitcher_counts[pitcher_counts >= 100].index.tolist()
    
    print(f"\n📊 Analyzing {len(qualified_batters)} batters, {len(qualified_pitchers)} pitchers")
    
    # Save analytics
    analytics = {
        'generated': datetime.now().isoformat(),
        'total_batters': len(qualified_batters),
        'total_pitchers': len(qualified_pitchers)
    }
    
    # Save to JSON
    with open(f"{DATA_DIR}/matchup_analytics.json", 'w') as f:
        json.dump(analytics, f, indent=2)
    
    print(f"\n✅ Saved to {DATA_DIR}/matchup_analytics.json")
    
    # Show sample analysis for top players
    print("\n" + "="*60)
    print("📋 SAMPLE: AARON JUDGE (batter) vs PAUL SKENES (pitcher)")
    print("="*60)
    
    # Find IDs (these would need proper lookup in production)
    # For now show structure
    
    return analytics

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    build_analytics()
