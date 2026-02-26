#!/usr/bin/env python3
"""
MLB PREDICTOR v8.0 - MULTI-YEAR TREND ANALYSIS
==============================================
Using 2023-2025 data to identify:
- Pitchers trending UP in HRs allowed (getting worse)
- Pitchers trending DOWN (getting better)
- Multi-year averages
- Career vs recent performance

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pybaseball import playerid_reverse_lookup
from datetime import datetime

DATA_DIR = "/Users/mikeross/MLB_Predictions"
PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

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

def load_data():
    """Load statcast with year info"""
    print("📡 Loading data...")
    pf = pq.ParquetFile(PARQUET_FILE)
    df = pf.read([
        'pitcher', 'batter', 'events', 'game_year',
        'hc_x', 'stand', 'p_throws',
        'pitch_type', 'release_speed', 'plate_x', 'plate_z'
    ]).to_pandas()
    print(f"   Loaded {len(df)} pitches")
    return df

def analyze_pitcher_trends(df):
    """Analyze pitcher HR trends across years"""
    print("\n📊 Analyzing pitcher trends...")
    
    hrs = df[df['events'] == 'home_run']
    hits = df[df['events'].isin(['single', 'double', 'triple', 'home_run'])]
    pa = df[df['events'].notna()]
    
    # HR by year (convert year to string)
    hr_by_year = hrs.groupby(['pitcher', 'game_year']).size().unstack(fill_value=0)
    bf_by_year = pa.groupby(['pitcher', 'game_year']).size().unstack(fill_value=0)
    
    # Convert columns to string
    hr_by_year.columns = hr_by_year.columns.astype(str)
    bf_by_year.columns = bf_by_year.columns.astype(str)
    
    # HR rate by year
    hr_rate = hr_by_year / bf_by_year * 100
    
    # Multi-year stats
    hr_by_year['hr_total'] = hr_by_year.sum(axis=1)
    hr_by_year['hr_avg'] = hr_by_year[['2023', '2024', '2025']].mean(axis=1)
    hr_by_year['hr_2025'] = hr_by_year.get('2025', 0)
    hr_by_year['hr_trend'] = hr_by_year.get('2025', 0) - hr_by_year.get('2023', 0)
    
    # Filter to pitchers with 3 years of data
    year_cols = ['2023', '2024', '2025']
    qualified = hr_by_year[hr_by_year[year_cols].notna().all(axis=1)]
    
    # Get pitcher names
    pitcher_ids = qualified.nlargest(30, 'hr_total').index.tolist()
    names = playerid_reverse_lookup(pitcher_ids, key_type='mlbam')
    
    results = []
    for pid in qualified.nlargest(50, 'hr_total').index:
        row = names[names['key_mlbam'] == pid]
        if len(row) > 0:
            name = row.iloc[0]['name_first'] + ' ' + row.iloc[0]['name_last']
            yr = qualified.loc[pid]
            results.append({
                'pitcher_id': pid,
                'name': name,
                'hr_2023': int(yr.get('2023', 0)),
                'hr_2024': int(yr.get('2024', 0)),
                'hr_2025': int(yr.get('2025', 0)),
                'hr_avg': round(yr['hr_avg'], 1),
                'hr_trend': int(yr['hr_trend'])
            })
    
    return pd.DataFrame(results)

def find_hr_prone_pitchers(trends_df):
    """Find pitchers who give up a lot of HRs"""
    
    # Worst HR rates (getting worse or consistently bad)
    worst = trends_df[trends_df['hr_avg'] >= 25].sort_values('hr_trend', ascending=False)
    
    print("\n" + "="*70)
    print("🚨 PITCHERS GIVING UP MOST HRs (2023-2025)")
    print("="*70)
    
    for _, row in worst.head(15).iterrows():
        trend_icon = "📈" if row['hr_trend'] > 0 else "📉"
        print(f"{row['name']:20} | Avg: {row['hr_avg']:.0f}/yr | "
              f"2023:{row['hr_2023']} 2024:{row['hr_2024']} 2025:{row['hr_2025']} {trend_icon}")
    
    return worst

def generate_predictions_with_trends():
    """Generate predictions using multi-year trends"""
    print("\n" + "="*70)
    print("⚾ MLB PREDICTOR v8.0 - MULTI-YEAR TRENDS")
    print("="*70)
    
    df = load_data()
    
    # Analyze trends
    trends = analyze_pitcher_trends(df)
    worst_pitchers = find_hr_prone_pitchers(trends)
    
    # Save
    trends.to_csv(f"{DATA_DIR}/pitcher_trends_2023_2025.csv", index=False)
    print(f"\n✅ Saved: {DATA_DIR}/pitcher_trends_2023_2025.csv")
    
    # Now generate predictions using trend-adjusted ERA
    print("\n" + "="*70)
    print("🏆 TOP HR PLAYS vs HR-PRONE PITCHERS")
    print("="*70)
    
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    # Get worst pitcher names
    worst_names = worst_pitchers['name'].head(10).tolist()
    
    # Find these in season stats
    for wname in worst_names[:5]:
        first_name = wname.split()[0].lower()
        pit = pitchers[pitchers['Name'].str.lower().str.contains(first_name, na=False)]
        if len(pit) > 0:
            pit = pit.iloc[0]
            print(f"\n⚠️ vs {wname} ({pit['Team']}) - ERA: {pit['ERA']}")
            
            # Top HR hitters vs this pitcher
            top_hr = batters.nlargest(10, 'HR')
            for _, bat in top_hr.iterrows():
                park = PARK_FACTORS.get(bat['Team'], 1.0)
                hr_prob = (bat['HR'] / 500) * (1 - pit['ERA'] / 20) * park
                if hr_prob > 0.08:
                    print(f"   → {bat['Name']}: {hr_prob:.1%}")

if __name__ == "__main__":
    generate_predictions_with_trends()
