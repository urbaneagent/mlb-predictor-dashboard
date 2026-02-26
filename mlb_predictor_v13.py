#!/usr/bin/env python3
"""
MLB PREDICTOR v13 - NORMALIZED HIT PROBABILITY
===============================================
Uses normalized metrics (per inning) instead of raw totals
- Hits per inning pitched
- HR per inning pitched
- Combined "hit-prone score"

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np

DATA_DIR = "/Users/mikeross/MLB_Predictions"

def run():
    print("\n" + "="*90)
    print("⚾ MLB PREDICTOR v13 - NORMALIZED METRICS")
    print("="*90)
    
    # Load data
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    
    # Calculate normalized metrics
    pitchers['hits_per_ip'] = pitchers['H'] / pitchers['IP']
    pitchers['hr_per_ip'] = pitchers['HR'] / pitchers['IP']
    pitchers['hit_prone_score'] = pitchers['hits_per_ip'] + pitchers['hr_per_ip']
    
    print("\n=== TOP HIT-PRONE PITCHERS (Normalized) ===")
    worst = pitchers.nlargest(15, 'hit_prone_score')
    for _, p in worst.iterrows():
        print(f"  {p['Name']}: {p['hit_prone_score']:.2f} (H: {p['hits_per_ip']:.2f}/IP, HR: {p['hr_per_ip']:.2f}/IP)")
    
    # Generate predictions using normalized data
    results = []
    
    # Top hit-prone pitchers
    for _, pit in pitchers.nlargest(25, 'hit_prone_score').iterrows():
        pit_team = pit['Team']
        
        for _, bat in batters.nlargest(30, 'HR').iterrows():
            bat_team = bat['Team']
            
            # HIT PROBABILITY using normalized pitcher data
            hit_rate = bat['AVG']
            pitcher_hit_rate = pit['hits_per_ip'] / 9  # Approximate BAA
            
            # Combine: batter skill vs pitcher vulnerability
            hit_prob = (hit_rate + pitcher_hit_rate) / 2
            hit_prob = min(hit_prob, 0.45)
            
            # HR PROBABILITY using normalized HR/IP
            hr_prob = (bat['HR'] / 500) * (pit['hr_per_ip'] * 10)
            hr_prob = min(hr_prob, 0.20)
            
            # WIN PROBABILITY (simplified)
            win_prob = 0.48 + (bat['AVG'] - pit['hits_per_ip'] / 10)
            win_prob = max(0.25, min(0.75, win_prob))
            
            # VALUE SCORE
            value = hr_prob * 100 + hit_prob * 30 + win_prob * 10
            
            results.append({
                'batter': bat['Name'],
                'batter_team': bat_team,
                'batter_avg': bat['AVG'],
                'batter_hr': bat['HR'],
                'pitcher': pit['Name'],
                'pitcher_team': pit_team,
                'pitcher_ip': pit['IP'],
                'pitcher_h_per_ip': round(pit['hits_per_ip'], 2),
                'pitcher_hr_per_ip': round(pit['hr_per_ip'], 2),
                'hit_prone_score': round(pit['hit_prone_score'], 2),
                'hit_prob': round(hit_prob, 3),
                'hr_prob': round(hr_prob, 3),
                'win_prob': round(win_prob, 3),
                'value': round(value, 1)
            })
    
    df = pd.DataFrame(results).sort_values('value', ascending=False)
    
    print("\n" + "="*90)
    print("🏆 TOP PLAYS (v13 - NORMALIZED)")
    print("="*90)
    
    for _, row in df.head(15).iterrows():
        print(f"\n⚾ {row['batter']} vs {row['pitcher']}")
        print(f"   HIT: {row['hit_prob']:.1%} | HR: {row['hr_prob']:.1%} | WIN: {row['win_prob']:.1%}")
        print(f"   Pitcher: {row['pitcher_h_per_ip']} H/IP, {row['pitcher_hr_per_ip']} HR/IP")
    
    # Save
    df.to_csv(f"{DATA_DIR}/v13_normalized.csv", index=False)
    print(f"\n✅ Saved: {DATA_DIR}/v13_normalized.csv")
    
    return df

if __name__ == "__main__":
    run()
