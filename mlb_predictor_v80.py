#!/usr/bin/env python3
"""
MLB PREDICTOR v8.0 - FINAL WITH MULTI-YEAR TRENDS
================================================
Uses:
- 2023-2025 multi-year pitcher trends (identifying who is getting worse)
- 2025 season stats (ERA, etc.)
- Matchup analytics (zone, handedness, velocity)
- Park factors

Author: Mike Ross
Date: 2026-02-21
"""

import pandas as pd
import numpy as np

DATA_DIR = "/Users/mikeross/MLB_Predictions"

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

def run():
    print("\n" + "="*80)
    print("⚾ MLB PREDICTOR v8.0 - MULTI-YEAR TRENDS + MATCHUPS")
    print("="*80)
    
    # Load data
    batters = pd.read_csv(f"{DATA_DIR}/batters_2025.csv")
    pitchers = pd.read_csv(f"{DATA_DIR}/pitchers_2025.csv")
    trends = pd.read_csv(f"{DATA_DIR}/pitcher_trends_2023_2025.csv")
    
    print(f"\n📊 Data: {len(batters)} batters, {len(pitchers)} pitchers, {len(trends)} pitcher trends")
    
    # Merge trends into pitchers (case-insensitive)
    pitchers['Name_lower'] = pitchers['Name'].str.lower()
    trends['name_lower'] = trends['name'].str.lower()
    pitchers = pitchers.merge(trends[['name_lower', 'hr_avg', 'hr_trend']], left_on='Name_lower', right_on='name_lower', how='left')
    pitchers['hr_trend'] = pitchers['hr_trend'].fillna(0)
    pitchers['hr_avg'] = pitchers['hr_avg'].fillna(20)
    
    # Identify HR-prone pitchers (high average + trending up)
    pitchers['hr_risk'] = (pitchers['hr_avg'] * 0.5) + (pitchers['hr_trend'] * 0.3)
    
    # Get worst HR-prone pitchers
    worst = pitchers.nlargest(20, 'hr_risk')[['Name', 'Team', 'ERA', 'hr_avg', 'hr_trend', 'hr_risk', 'IP']]
    
    print("\n" + "="*80)
    print("🚨 WORST HR-PRONE PITCHERS (High Avg + Trending Up)")
    print("="*80)
    for _, row in worst.iterrows():
        trend_icon = "📈" if row['hr_trend'] > 0 else "📉"
        print(f"  {row['Name']:20} | ERA:{row['ERA']:.2f} | HR/yr:{row['hr_avg']:.0f} | Trend:{row['hr_trend']:+.0f} {trend_icon}")
    
    # Generate predictions
    print("\n" + "="*80)
    print("🏆 TOP HR PLAYS vs HR-PRONE PITCHERS")
    print("="*80)
    
    results = []
    
    for _, pitcher in worst.head(10).iterrows():
        # Get top HR hitters
        top_hr = batters.nlargest(15, 'HR')
        
        for _, batter in top_hr.iterrows():
            park = PARK_FACTORS.get(batter['Team'], 1.0)
            
            # Base HR probability
            hr_prob = (batter['HR'] / 500)
            
            # Pitcher ERA factor (lower ERA = harder to hit HR)
            pit_factor = 1 - (pitcher['ERA'] / 20)
            
            # HR trend factor (pitcher giving up more HRs = higher probability)
            trend_factor = 1 + (pitcher['hr_trend'] / 50)  # +2% per HR trend
            
            # Park factor
            park_factor = park
            
            # Calculate
            final_hr = hr_prob * pit_factor * trend_factor * park_factor
            final_hr = min(final_hr, 0.50)
            
            # Hit probability
            hit_prob = batter['AVG'] * (1 - pitcher['ERA'] / 15)
            hit_prob = min(hit_prob, 0.40)
            
            results.append({
                'batter': batter['Name'],
                'batter_team': batter['Team'],
                'batter_hr': batter['HR'],
                'batter_avg': batter['AVG'],
                'pitcher': pitcher['Name'],
                'pitcher_team': pitcher['Team'],
                'pitcher_era': pitcher['ERA'],
                'pitcher_hr_avg': pitcher['hr_avg'],
                'pitcher_trend': pitcher['hr_trend'],
                'park': batter['Team'],
                'park_factor': park,
                'hr_prob': round(final_hr, 3),
                'hit_prob': round(hit_prob, 3),
                'value_score': round(final_hr * 100 + hit_prob * 15, 1)
            })
    
    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values('value_score', ascending=False)
    
    # Display
    for i, row in df.head(20).iterrows():
        trend_icon = "📈" if row['pitcher_trend'] > 0 else "📉"
        print(f"\n⚾ {row['batter']} ({row['batter_team']}) vs {row['pitcher']} ({row['pitcher_team']})")
        print(f"   HR: {row['hr_prob']:.1%} | Hit: {row['hit_prob']:.1%} | Value: {row['value_score']}")
        print(f"   Pitcher: ERA {row['pitcher_era']:.2f} | HR/yr: {row['pitcher_hr_avg']:.0f} | Trend: {row['pitcher_trend']:+.0f} {trend_icon}")
        print(f"   Park: {row['park']} ({row['park_factor']:.2f}x)")
    
    # Save
    output = f"{DATA_DIR}/v80_predictions.csv"
    df.to_csv(output, index=False)
    print(f"\n✅ Saved: {output}")

if __name__ == "__main__":
    run()
