#!/usr/bin/env python3
"""
MLB PREDICTOR v9.0 - VELOCITY TREND ANALYSIS
=============================================
Identifies WHY pitchers are getting worse:
- Velocity loss ( mph)
- Pitch mix degradation
- HR rate trend
- Zone control

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

def analyze_pitcher_decline():
    """Deep analysis of why pitchers decline"""
    print("\n📡 Loading data for decline analysis...")
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read(['pitcher', 'game_year', 'release_speed', 'pitch_type', 'events', 'plate_x', 'plate_z']).to_pandas()
    
    # Group by pitcher and year
    results = []
    
    # Get all pitchers with 3 years of data
    pitcher_years = df.groupby(['pitcher', 'game_year']).size().unstack()
    qualified = pitcher_years[pitcher_years.notna().all(axis=1)].index.tolist()
    
    for pitcher_id in qualified:
        p = df[df['pitcher'] == pitcher_id]
        
        yearly_data = {}
        for year in [2023, 2024, 2025]:
            yr = p[p['game_year'] == year]
            bf = yr[yr['events'].notna()]
            hits = bf[bf['events'].isin(['single', 'double', 'triple', 'home_run'])]
            hrs = bf[bf['events'] == 'home_run']
            
            yearly_data[year] = {
                'bf': len(bf),
                'hits': len(hits),
                'hrs': len(hrs),
                'hr_rate': len(hrs) / len(bf) * 100 if len(bf) > 0 else 0,
                'velo': yr['release_speed'].mean()
            }
        
        # Calculate trends
        velo_trend = yearly_data[2025]['velo'] - yearly_data[2023]['velo']
        hr_trend = yearly_data[2025]['hr_rate'] - yearly_data[2023]['hr_rate']
        
        results.append({
            'pitcher_id': pitcher_id,
            'bf_2025': yearly_data[2025]['bf'],
            'hrs_2023': yearly_data[2023]['hrs'],
            'hrs_2024': yearly_data[2024]['hrs'],
            'hrs_2025': yearly_data[2025]['hrs'],
            'hr_rate_2023': yearly_data[2023]['hr_rate'],
            'hr_rate_2025': yearly_data[2025]['hr_rate'],
            'hr_trend': hr_trend,
            'velo_2023': yearly_data[2023]['velo'],
            'velo_2025': yearly_data[2025]['velo'],
            'velo_trend': velo_trend
        })
    
    trends_df = pd.DataFrame(results)
    
    # Filter to significant
    trends_df = trends_df[trends_df['bf_2025'] >= 400]  # At least 400 batters faced
    
    # Add names
    pitcher_ids = trends_df['pitcher_id'].tolist()
    names = playerid_reverse_lookup(pitcher_ids, key_type='mlbam')
    
    trends_df = trends_df.merge(
        names[['key_mlbam', 'name_first', 'name_last']],
        left_on='pitcher_id', right_on='key_mlbam', how='left'
    )
    trends_df['name'] = trends_df['name_first'] + ' ' + trends_df['name_last']
    
    # Sort by HR trend
    worst = trends_df.nlargest(20, 'hr_trend')
    
    print("\n" + "="*80)
    print("🚨 PITCHERS GETTING WORSE - ROOT CAUSE ANALYSIS")
    print("="*80)
    
    for _, row in worst.iterrows():
        velo_icon = "📉" if row['velo_trend'] < 0 else "📈"
        print(f"\n{row['name']}")
        print(f"   HRs: {row['hrs_2023']} (2023) → {row['hrs_2025']} (2025) | Trend: +{row['hr_trend']:.1f}%")
        print(f"   Velo: {row['velo_2023']:.1f} → {row['velo_2025']:.1f} mph {velo_icon} ({row['velo_trend']:+.1f})")
        
        # Calculate correlation
        if row['velo_trend'] < -1 and row['hr_trend'] > 1:
            print(f"   ⚠️ VELOCITY LOSS CORRELATED WITH HR INCREASE!")
    
    # Save
    trends_df.to_csv(f"{DATA_DIR}/pitcher_decline_analysis.csv", index=False)
    print(f"\n✅ Saved: {DATA_DIR}/pitcher_decline_analysis.csv")
    
    return trends_df

def generate_v9_predictions():
    """Generate predictions with velocity + HR trend"""
    print("\n" + "="*80)
    print("⚾ MLB PREDICTOR v9.0 - VELOCITY TREND MODEL")
    print("="*80)
    
    # Load data
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    decline = pd.read_csv(f"{DATA_DIR}/pitcher_decline_analysis.csv")
    
    # Merge
    pitchers['Name_lower'] = pitchers['Name'].str.lower()
    decline['name_lower'] = decline['name'].str.lower()
    pitchers = pitchers.merge(
        decline[['name_lower', 'hr_trend', 'velo_trend', 'hr_rate_2025']], 
        left_on='Name_lower', right_on='name_lower', how='left'
    )
    pitchers['hr_trend'] = pitchers['hr_trend'].fillna(0)
    pitchers['velo_trend'] = pitchers['velo_trend'].fillna(0)
    pitchers['hr_rate_2025'] = pitchers['hr_rate_2025'].fillna(3.0)
    
    # Find declining pitchers
    declining = pitchers[pitchers['hr_trend'] > 1].nlargest(15, 'hr_trend')
    
    print("\n" + "="*80)
    print("🏆 BEST HR PLAYS vs DECLINING PITCHERS")
    print("="*80)
    
    results = []
    
    for _, pit in declining.iterrows():
        top_hr = batters.nlargest(15, 'HR')
        
        for _, bat in top_hr.iterrows():
            park = PARK_FACTORS.get(bat['Team'], 1.0)
            
            # Base HR
            hr = bat['HR'] / 500
            
            # ERA factor
            era_factor = 1 - (pit['ERA'] / 20)
            
            # HR trend factor (pitcher giving up more HRs)
            trend_factor = 1 + (pit['hr_trend'] / 20)
            
            # Velocity decline factor
            velo_factor = 1 + (abs(pit['velo_trend']) / 50) if pit['velo_trend'] < 0 else 1
            
            # Final
            final_hr = hr * era_factor * trend_factor * velo_factor * park
            final_hr = min(final_hr, 0.50)
            
            hit = bat['AVG'] * (1 - pit['ERA'] / 15)
            hit = min(hit, 0.40)
            
            results.append({
                'batter': bat['Name'],
                'batter_team': bat['Team'],
                'batter_hr': bat['HR'],
                'pitcher': pit['Name'],
                'pitcher_team': pit['Team'],
                'pitcher_era': pit['ERA'],
                'pitcher_hr_trend': pit['hr_trend'],
                'pitcher_velo_trend': pit['velo_trend'],
                'hr_prob': round(final_hr, 3),
                'hit_prob': round(hit, 3),
                'value': round(final_hr * 100 + hit * 15, 1)
            })
    
    df = pd.DataFrame(results).sort_values('value', ascending=False)
    
    for _, row in df.head(15).iterrows():
        velo_icon = "📉" if row['pitcher_velo_trend'] < 0 else ""
        print(f"\n⚾ {row['batter']} vs {row['pitcher']}")
        print(f"   HR: {row['hr_prob']:.1%} | Hit: {row['hit_prob']:.1%} | Value: {row['value']}")
        print(f"   Pitcher: ERA {row['pitcher_era']} | HR Trend: +{row['pitcher_hr_trend']:.1f}% | Velo: {row['pitcher_velo_trend']:+.1f} {velo_icon}")
    
    df.to_csv(f"{DATA_DIR}/v90_predictions.csv", index=False)
    print(f"\n✅ Saved: {DATA_DIR}/v90_predictions.csv")

if __name__ == "__main__":
    analyze_pitcher_decline()
    generate_v9_predictions()
