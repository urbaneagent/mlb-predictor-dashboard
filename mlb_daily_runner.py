#!/usr/bin/env python3
"""
MLB Daily Predictions Runner
Generates daily predictions using the latest model
"""

import os
import sys
import pandas as pd
from datetime import datetime

VENV_PYTHON = '/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/venv/bin/python'
DATA_DIR = '/Users/mikeross/MLB_Predictions'

# Import our factors
HIT_PARK = {'COL': 1.25, 'BOS': 1.12, 'ARI': 1.10, 'PHI': 1.08, 'WSH': 1.08,
    'MIA': 1.07, 'ATL': 1.07, 'STL': 1.06, 'BAL': 1.05, 'MIN': 1.05,
    'KC': 1.05, 'TOR': 1.05, 'LAD': 1.04, 'CLE': 1.03, 'TEX': 1.03,
    'HOU': 1.03, 'NYY': 1.03, 'CHC': 1.02, 'CIN': 1.02, 'DET': 1.01,
    'MIL': 1.00, 'SF': 0.98, 'TB': 0.97, 'SEA': 0.97, 'OAK': 0.96,
    'LAA': 0.96, 'SD': 0.95, 'PIT': 0.95, 'NYM': 0.94}

BULLPEN = {'NYY': 3.45, 'LAD': 3.52, 'ATL': 3.65, 'TB': 3.75, 'HOU': 3.78,
    'PHI': 3.82, 'SF': 3.85, 'STL': 3.88, 'MIN': 3.92, 'SD': 3.92,
    'CLE': 3.95, 'CHC': 3.95, 'MIL': 4.05, 'TEX': 4.05, 'LAA': 4.15,
    'CIN': 4.15, 'MIA': 4.18, 'BOS': 4.12, 'TOR': 3.98, 'PIT': 4.22,
    'WSH': 4.25, 'CWS': 4.28, 'KC': 4.35, 'SEA': 4.38, 'DET': 4.45,
    'OAK': 4.55, 'ARI': 4.02, 'COL': 5.15, 'BAL': 4.08, 'NYM': 4.12}

def run_predictions():
    """Generate daily predictions"""
    print(f"MLB Daily Predictions - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)
    
    # Load data
    batters = pd.read_csv(f'{DATA_DIR}/batters_2025.csv')
    pitchers = pd.read_csv(f'{DATA_DIR}/pitchers_2025.csv')
    
    # Build predictions using v29 factors
    pitchers['hr_per_ip'] = pitchers['HR'] / pitchers['IP']
    pitchers['hits_per_ip'] = pitchers['H'] / pitchers['IP']
    pitchers['bullpen_era'] = pitchers['Team'].map(BULLPEN).fillna(4.00)
    
    batters['pa_per_game'] = batters['AB'] / 162
    batters['opportunity'] = batters['pa_per_game'] * batters['AVG']
    batters['barrel_factor'] = 1 + (batters['Barrel%'].fillna(0.05) - 0.05)
    
    results = []
    for _, pit in pitchers.nlargest(30, 'hr_per_ip').iterrows():
        pit_bullpen = BULLPEN.get(pit['Team'], 4.00)
        
        for _, bat in batters.nlargest(40, 'opportunity').iterrows():
            bat_park = HIT_PARK.get(bat['Team'], 1.0)
            barrel = bat.get('barrel_factor', 1.0)
            
            hit_prob = (bat['AVG'] + pit['hits_per_ip']/10) / 2
            hit_prob *= bat_park
            
            hr_prob = (bat['HR']/500) * (pit['hr_per_ip'] * 10) * barrel
            
            win_prob = (1 - pit['ERA']/20 + 1 - pit_bullpen/20) / 2
            win_prob = max(0.30, min(0.70, win_prob))
            
            opp = bat['pa_per_game'] / 3.5
            dfs = hr_prob * 10 + hit_prob * 3 * opp
            
            results.append({
                'batter': bat['Name'],
                'pitcher': pit['Name'],
                'hit_prob': round(hit_prob, 3),
                'hr_prob': round(hr_prob, 3),
                'win_prob': round(win_prob, 3),
                'dfs': round(dfs, 2)
            })
    
    df = pd.DataFrame(results).sort_values('dfs', ascending=False)
    
    # Save
    output = f"{DATA_DIR}/daily_predictions.csv"
    df.to_csv(output, index=False)
    
    print(f"Generated {len(df)} predictions")
    print(f"Saved to: {output}")
    print()
    print("TOP 10:")
    for i, row in df.head(10).iterrows():
        print(f"  {row['batter']} vs {row['pitcher']}: HIT {row['hit_prob']:.1%} | HR {row['hr_prob']:.1%}")
    
    return df

if __name__ == '__main__':
    run_predictions()
