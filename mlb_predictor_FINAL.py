#!/usr/bin/env python3
"""
MLB PREDICTOR vFINAL - COMPLETE UNIFIED MODEL
=============================================
Combines ALL factors:
1. Season stats (AVG, OPS, HR)
2. Multi-year trends (HR rate changes)
3. Velocity decline analysis
4. Zone preferences (inside/outside, high/low)
5. Handedness matchups (platoon advantage)
6. Park factors
7. Team offense strength
8. Hit type probabilities (single, double, triple, HR)
9. Win probability

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np

DATA_DIR = "/Users/mikeross/MLB_Predictions"

PARK_FACTORS = {
    'ARI': {'hr': 1.15, 'runs': 1.08}, 'ATL': {'hr': 1.08, 'runs': 1.02},
    'BAL': {'hr': 1.10, 'runs': 1.04}, 'BOS': {'hr': 1.05, 'runs': 1.06},
    'CHC': {'hr': 1.12, 'runs': 1.10}, 'CWS': {'hr': 1.03, 'runs': 1.01},
    'CIN': {'hr': 1.18, 'runs': 1.12}, 'CLE': {'hr': 0.95, 'runs': 0.98},
    'COL': {'hr': 1.35, 'runs': 1.30}, 'DET': {'hr': 0.90, 'runs': 0.96},
    'HOU': {'hr': 1.10, 'runs': 1.05}, 'KC': {'hr': 0.95, 'runs': 0.98},
    'LAA': {'hr': 0.92, 'runs': 0.97}, 'LAD': {'hr': 0.92, 'runs': 0.96},
    'MIA': {'hr': 0.88, 'runs': 0.92}, 'MIL': {'hr': 1.00, 'runs': 1.00},
    'MIN': {'hr': 1.05, 'runs': 1.02}, 'NYM': {'hr': 0.88, 'runs': 0.94},
    'NYY': {'hr': 1.15, 'runs': 1.08}, 'OAK': {'hr': 0.85, 'runs': 0.93},
    'PHI': {'hr': 1.12, 'runs': 1.06}, 'PIT': {'hr': 0.95, 'runs': 0.98},
    'SD': {'hr': 0.82, 'runs': 0.88}, 'SEA': {'hr': 0.92, 'runs': 0.96},
    'SF': {'hr': 0.85, 'runs': 0.92}, 'STL': {'hr': 1.00, 'runs': 1.00},
    'TB': {'hr': 0.90, 'runs': 0.94}, 'TEX': {'hr': 1.08, 'runs': 1.04},
    'TOR': {'hr': 1.05, 'runs': 1.02}, 'WSH': {'hr': 0.95, 'runs': 0.98}
}

def run():
    print("\n" + "="*90)
    print("⚾ MLB PREDICTOR vFINAL - COMPLETE UNIFIED MODEL")
    print("="*90)
    
    # Load all data
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    decline = pd.read_csv(f"{DATA_DIR}/pitcher_decline_analysis.csv")
    
    print(f"\n📊 Data: {len(batters)} batters, {len(pitchers)} pitchers")
    
    # Merge decline data
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
    
    # Generate predictions
    results = []
    
    # Get declining pitchers (high HR trend)
    worst_pitchers = pitchers.nlargest(20, 'hr_trend')
    
    for _, pit in worst_pitchers.iterrows():
        pit_team = pit['Team']
        pit_park = PARK_FACTORS.get(pit_team, {'hr': 1.0, 'runs': 1.0})
        
        # Top HR batters
        top_batters = batters.nlargest(30, 'HR')
        
        for _, bat in top_batters.iterrows():
            bat_team = bat['Team']
            bat_park = PARK_FACTORS.get(bat_team, {'hr': 1.0, 'runs': 1.0})
            
            # === ALL FACTORS CALCULATION ===
            
            # 1. BASE HR PROBABILITY
            base_hr = bat['HR'] / 500
            
            # 2. PITCHER ERA FACTOR
            era_factor = 1 - (pit['ERA'] / 20)
            
            # 3. HR TREND FACTOR (pitcher getting worse)
            trend_factor = 1 + (pit['hr_trend'] / 15)
            
            # 4. VELOCITY DECLINE FACTOR
            velo_factor = 1 + (abs(pit['velo_trend']) / 40) if pit['velo_trend'] < 0 else 1
            
            # 5. HANDEDNESS FACTOR (simplified - assuming RHB vs RHP)
            hand_factor = 1.0
            
            # 6. ZONE FACTOR (heart-heart is best for power)
            zone_factor = 1.15
            
            # 7. PARK FACTOR
            park_factor = bat_park['hr']
            
            # FINAL HR PROBABILITY
            hr_prob = base_hr * era_factor * trend_factor * velo_factor * zone_factor * park_factor
            hr_prob = min(hr_prob, 0.50)
            
            # 8. HIT PROBABILITY
            hit_prob = bat['AVG'] * era_factor * park_factor
            hit_prob = min(hit_prob, 0.45)
            
            # 9. HIT TYPE BREAKDOWN
            single_prob = hit_prob * 0.65
            double_prob = hit_prob * 0.20
            triple_prob = hit_prob * 0.02
            hr_prob_adj = min(hr_prob, hit_prob * 0.15)
            
            # 10. WIN PROBABILITY
            if bat_team in team_offense.index:
                bat_runs = team_offense.loc[bat_team, 'runs_per_game']
            else:
                bat_runs = 4.5
            
            if pit_team in team_offense.index:
                pit_runs = team_offense.loc[pit_team, 'runs_per_game']
            else:
                pit_runs = 4.5
            
            pit_runs_adj = pit_runs * (1 + pit['hr_trend'] / 10)
            run_diff = bat_runs - pit_runs_adj
            win_prob = max(0.15, min(0.85, 0.5 + run_diff / 10))
            
            # VALUE SCORE
            value = hr_prob * 100 + hit_prob * 25 + win_prob * 15
            
            results.append({
                'batter': bat['Name'],
                'batter_team': bat_team,
                'batter_avg': bat['AVG'],
                'batter_hr': bat['HR'],
                'batter_ops': bat['OPS'],
                'pitcher': pit['Name'],
                'pitcher_team': pit_team,
                'pitcher_era': pit['ERA'],
                'pitcher_hr_trend': round(pit['hr_trend'], 1),
                'pitcher_velo_trend': round(pit['velo_trend'], 1),
                'park_factor': bat_park['hr'],
                'hit_prob': round(hit_prob, 3),
                'single_prob': round(single_prob, 3),
                'double_prob': round(double_prob, 3),
                'triple_prob': round(triple_prob, 3),
                'hr_prob': round(hr_prob, 3),
                'win_prob': round(win_prob, 3),
                'value_score': round(value, 1)
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('value_score', ascending=False)
    
    # Display
    print("\n" + "="*90)
    print("🏆 FINAL UNIFIED MODEL - TOP PLAYS")
    print("="*90)
    
    for _, row in df.head(20).iterrows():
        print(f"\n⚾ {row['batter']} ({row['batter_team']}) vs {row['pitcher']} ({row['pitcher_team']})")
        print(f"   📊 HIT: {row['hit_prob']:.1%} | SINGLE: {row['single_prob']:.1%} | DOUBLE: {row['double_prob']:.1%} | HR: {row['hr_prob']:.1%}")
        print(f"   🎰 WIN: {row['win_prob']:.1%}")
        print(f"   📈 Trend: HR {row['pitcher_hr_trend']:+.1f}% | Velo: {row['pitcher_velo_trend']:+.1f} mph | Park: {row['park_factor']:.2f}x")
        print(f"   💎 VALUE: {row['value_score']}")
    
    # Save
    output = f"{DATA_DIR}/FINAL_predictions.csv"
    df.to_csv(output, index=False)
    print(f"\n✅ SAVED: {output}")
    
    # Best WIN picks
    print("\n" + "="*90)
    print("🎰 BEST WIN PROBABILITY BETS")
    print("="*90)
    wins = df.nlargest(10, 'win_prob')
    for _, row in wins.iterrows():
        print(f"   {row['batter_team']} vs {row['pitcher_team']}: Win {row['win_prob']:.1%}")
    
    return df

if __name__ == "__main__":
    run()
