#!/usr/bin/env python3
"""
MLB PREDICTOR v12 - HIT-FOCUSED MODEL
=====================================
Focus on:
- Hits per stadium (which parks give up most hits)
- Pitchers giving up most hits
- Home vs Road performance factors
- Win probability

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np

DATA_DIR = "/Users/mikeross/MLB_Predictions"

# HIT-friendly parks (more hits allowed)
HIT_FACTORS = {
    'COL': 1.25,  # Coors Field - most hitter friendly
    'BOS': 1.12,  # Fenway
    'ARI': 1.10,
    'PHI': 1.08,
    'WSH': 1.08,
    'MIA': 1.07,
    'ATL': 1.07,
    'STL': 1.06,
    'BAL': 1.05,
    'MIN': 1.05,
    'KC': 1.05,
    'TOR': 1.05,
    'LAD': 1.04,
    'CLE': 1.03,
    'TEX': 1.03,
    'HOU': 1.03,
    'NYY': 1.03,
    'CHC': 1.02,
    'CIN': 1.02,
    'DET': 1.01,
    'MIL': 1.00,
    'SF': 0.98,
    'TB': 0.97,
    'SEA': 0.97,
    'OAK': 0.96,
    'LAA': 0.96,
    'SD': 0.95,
    'PIT': 0.95,
    'NYM': 0.94
}

# Park run factors for wins
RUN_FACTORS = {
    'COL': 1.30, 'BOS': 1.08, 'ARI': 1.08, 'CIN': 1.12,
    'CHC': 1.10, 'NYY': 1.08, 'HOU': 1.05, 'PHI': 1.06,
    'BAL': 1.04, 'TEX': 1.04, 'MIN': 1.02, 'ATL': 1.02,
    'TOR': 1.02, 'STL': 1.00, 'LAD': 0.96, 'CLE': 0.98,
    'KC': 0.98, 'DET': 0.96, 'MIA': 0.92, 'NYM': 0.94,
    'SEA': 0.96, 'TB': 0.94, 'SF': 0.92, 'OAK': 0.93,
    'SD': 0.88, 'PIT': 0.98, 'LAA': 0.97, 'MIL': 1.00,
    'WSH': 0.98
}

def run():
    print("\n" + "="*90)
    print("⚾ MLB PREDICTOR v12 - HIT-FOCUSED MODEL")
    print("="*90)
    
    # Load data
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    decline = pd.read_csv(f"{DATA_DIR}/pitcher_decline_analysis.csv")
    
    # Merge
    pitchers['Name_lower'] = pitchers['Name'].str.lower()
    decline['name_lower'] = decline['name'].str.lower()
    pitchers = pitchers.merge(
        decline[['name_lower', 'hr_trend', 'velo_trend']], 
        left_on='Name_lower', right_on='name_lower', how='left'
    )
    pitchers['hr_trend'] = pitchers['hr_trend'].fillna(0)
    pitchers['velo_trend'] = pitchers['velo_trend'].fillna(0)
    
    # Team offense
    team_offense = batters.groupby('Team').agg({
        'AVG': 'mean', 'HR': 'sum', 'OPS': 'mean', 'AB': 'sum'
    })
    team_offense['runs_per_game'] = (team_offense['HR'] * 1.5 + team_offense['AVG'] * team_offense['AB'] * 0.3) / team_offense['AB'] * 9
    
    # Top hit-prone pitchers
    worst_hr_pitchers = pitchers.nlargest(20, 'hr_trend')
    
    print("\n=== PITCHERS GIVING UP MOST HITS ===")
    # These are typically high-volume starters
    for _, p in worst_hr_pitchers.head(10).iterrows():
        print(f"  {p['Name']}: ERA {p['ERA']}, Trend: {p['hr_trend']:+.1f}%")
    
    # Generate predictions
    results = []
    
    for _, pit in worst_hr_pitchers.iterrows():
        pit_team = pit['Team']
        
        for _, bat in batters.nlargest(30, 'HR').iterrows():
            bat_team = bat['Team']
            
            # HIT PROBABILITY
            park_hit = HIT_FACTORS.get(bat_team, 1.0)
            era_factor = 1 - (pit['ERA'] / 18)  # More hits with higher ERA
            trend_factor = 1 + (pit['hr_trend'] / 20)
            
            hit_prob = bat['AVG'] * era_factor * park_hit * trend_factor
            hit_prob = min(hit_prob, 0.50)
            
            # HR from hit probability
            hr_prob = hit_prob * (bat['HR'] / max(bat['AB'], 1)) / 10
            hr_prob = min(hr_prob, 0.25)
            
            # WIN PROBABILITY
            bat_runs = team_offense.loc[bat_team, 'runs_per_game'] if bat_team in team_offense.index else 4.5
            pit_runs = team_offense.loc[pit_team, 'runs_per_game'] if pit_team in team_offense.index else 4.5
            
            # Adjust for park
            pit_runs_adj = pit_runs * RUN_FACTORS.get(pit_team, 1.0)
            bat_runs_adj = bat_runs * RUN_FACTORS.get(bat_team, 1.0)
            
            run_diff = bat_runs_adj - pit_runs_adj
            win_prob = max(0.20, min(0.80, 0.5 + run_diff / 12))
            
            # VALUE
            value = hit_prob * 100 + win_prob * 20
            
            results.append({
                'batter': bat['Name'],
                'batter_team': bat_team,
                'batter_avg': bat['AVG'],
                'batter_hr': bat['HR'],
                'pitcher': pit['Name'],
                'pitcher_team': pit_team,
                'pitcher_era': pit['ERA'],
                'pitcher_trend': round(pit['hr_trend'], 1),
                'park_factor': park_hit,
                'hit_prob': round(hit_prob, 3),
                'hr_prob': round(hr_prob, 3),
                'win_prob': round(win_prob, 3),
                'value': round(value, 1)
            })
    
    df = pd.DataFrame(results).sort_values('value', ascending=False)
    
    print("\n" + "="*90)
    print("🏆 TOP HIT + WIN PLAYS")
    print("="*90)
    
    for _, row in df.head(20).iterrows():
        print(f"\n⚾ {row['batter']} ({row['batter_team']}) vs {row['pitcher']} ({row['pitcher_team']})")
        print(f"   HIT: {row['hit_prob']:.1%} | HR: {row['hr_prob']:.1%} | WIN: {row['win_prob']:.1%}")
        print(f"   Park: {row['park_factor']:.2f}x hits | Trend: {row['pitcher_trend']:+.1f}%")
    
    # Save
    df.to_csv(f"{DATA_DIR}/v12_hit_focused.csv", index=False)
    print(f"\n✅ Saved: {DATA_DIR}/v12_hit_focused.csv")
    
    return df

if __name__ == "__main__":
    run()
