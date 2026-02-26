#!/usr/bin/env python3
"""
MLB Batter vs Pitcher Matchup Analysis
=======================================
Real historical matchup data from 27,000+ PA
Analyzes actual performance in head-to-head situations

Features:
- Direct H2H performance (AVG, OPS, HR rate)
- Platoon advantage (L vs R, R vs L)
- Recent form (last 50 PA)
- Career stats with sample size weighting
- Confidence scoring based on PA count

Author: Mike Ross (The Architect)
Date: 2026-02-22
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

class MatchupAnalyzer:
    """Analyzes batter vs pitcher matchups from Statcast data"""
    
    def __init__(self, statcast_file=PARQUET_FILE):
        self.statcast_file = statcast_file
        self.df = None
        
    def load_data(self):
        """Load Statcast parquet with matchup-relevant columns"""
        print("📡 Loading Statcast matchup data...")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.statcast_file)
        
        # Only load columns we need for matchup analysis
        cols = [
            'batter', 'pitcher', 'events', 'game_date',
            'stand', 'p_throws', 'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle', 'launch_speed',
            'launch_angle', 'hit_distance_sc'
        ]
        
        self.df = pf.read(cols).to_pandas()
        print(f"   Loaded {len(self.df):,} pitches from Statcast")
        
        # Convert game_date to datetime
        self.df['game_date'] = pd.to_datetime(self.df['game_date'])
        
        return self.df
    
    def get_h2h_matchup(self, batter_id, pitcher_id):
        """
        Get head-to-head performance between specific batter and pitcher
        
        Args:
            batter_id: MLB batter ID
            pitcher_id: MLB pitcher ID
            
        Returns:
            dict with H2H stats or None if insufficient data
        """
        if self.df is None:
            self.load_data()
        
        # Filter to this specific matchup
        matchup = self.df[
            (self.df['batter'] == batter_id) & 
            (self.df['pitcher'] == pitcher_id)
        ]
        
        # Filter to plate appearances (events is not null)
        pa = matchup[matchup['events'].notna()]
        
        if len(pa) < 5:
            return None  # Insufficient sample size
        
        # Calculate outcomes
        hits = pa[pa['events'].isin(['single', 'double', 'triple', 'home_run'])]
        hr = pa[pa['events'] == 'home_run']
        walks = pa[pa['events'].isin(['walk'])]
        strikeouts = pa[pa['events'] == 'strikeout']
        
        # Calculate stats
        avg = len(hits) / len(pa) if len(pa) > 0 else 0
        obp = (len(hits) + len(walks)) / len(pa) if len(pa) > 0 else 0
        slg = self._calculate_slg(pa)
        ops = obp + slg
        hr_rate = len(hr) / len(pa) if len(pa) > 0 else 0
        k_rate = len(strikeouts) / len(pa) if len(pa) > 0 else 0
        
        # Get platoon split
        platoon = pa['stand'].iloc[0] if len(pa) > 0 else 'Unknown'
        pitcher_hand = pa['p_throws'].iloc[0] if len(pa) > 0 else 'Unknown'
        
        # Recent performance (last 50 PA in this matchup)
        recent_pa = pa.nlargest(min(50, len(pa)), 'game_date')
        recent_hits = recent_pa[recent_pa['events'].isin(['single', 'double', 'triple', 'home_run'])]
        recent_avg = len(recent_hits) / len(recent_pa) if len(recent_pa) > 0 else 0
        
        return {
            'plate_appearances': len(pa),
            'avg': round(avg, 3),
            'obp': round(obp, 3),
            'slg': round(slg, 3),
            'ops': round(ops, 3),
            'hr_rate': round(hr_rate, 3),
            'k_rate': round(k_rate, 3),
            'recent_avg': round(recent_avg, 3),
            'platoon': f"{platoon} vs {pitcher_hand}",
            'confidence': self._get_confidence(len(pa)),
            'last_matchup': pa['game_date'].max().strftime('%Y-%m-%d') if len(pa) > 0 else 'Unknown'
        }
    
    def get_platoon_stats(self, batter_id, pitcher_handedness):
        """
        Get batter's performance vs specific pitcher handedness (L or R)
        
        Args:
            batter_id: MLB batter ID
            pitcher_handedness: 'L' or 'R'
            
        Returns:
            dict with platoon split stats
        """
        if self.df is None:
            self.load_data()
        
        # Filter to batter vs handedness
        split = self.df[
            (self.df['batter'] == batter_id) & 
            (self.df['p_throws'] == pitcher_handedness)
        ]
        
        pa = split[split['events'].notna()]
        
        if len(pa) < 20:
            return None
        
        hits = pa[pa['events'].isin(['single', 'double', 'triple', 'home_run'])]
        hr = pa[pa['events'] == 'home_run']
        
        avg = len(hits) / len(pa) if len(pa) > 0 else 0
        slg = self._calculate_slg(pa)
        hr_rate = len(hr) / len(pa) if len(pa) > 0 else 0
        
        return {
            'handedness': pitcher_handedness,
            'pa': len(pa),
            'avg': round(avg, 3),
            'slg': round(slg, 3),
            'hr_rate': round(hr_rate, 3),
            'confidence': self._get_confidence(len(pa))
        }
    
    def get_matchup_score(self, batter_id, pitcher_id):
        """
        Generate a 0-100 matchup favorability score
        
        Higher = better for batter
        Lower = better for pitcher
        
        Args:
            batter_id: MLB batter ID
            pitcher_id: MLB pitcher ID
            
        Returns:
            int (0-100) or None if insufficient data
        """
        h2h = self.get_h2h_matchup(batter_id, pitcher_id)
        
        if h2h is None:
            return None
        
        # Weight factors
        ops_weight = 0.40
        recent_weight = 0.30
        hr_weight = 0.20
        confidence_weight = 0.10
        
        # Normalize OPS (league avg ~.750, elite ~.900+)
        ops_score = min(100, (h2h['ops'] / 0.900) * 100)
        
        # Recent performance (compare to career avg in matchup)
        if h2h['avg'] > 0:
            recent_score = min(100, (h2h['recent_avg'] / h2h['avg']) * 50)
        else:
            recent_score = 50
        
        # HR threat (league avg ~3%, elite ~8%+)
        hr_score = min(100, (h2h['hr_rate'] / 0.08) * 100)
        
        # Confidence based on sample size
        confidence_score = h2h['confidence']
        
        # Weighted score
        final_score = (
            ops_score * ops_weight +
            recent_score * recent_weight +
            hr_score * hr_weight +
            confidence_score * confidence_weight
        )
        
        return int(round(final_score))
    
    def _calculate_slg(self, pa_df):
        """Calculate slugging percentage from plate appearances"""
        if len(pa_df) == 0:
            return 0
        
        singles = len(pa_df[pa_df['events'] == 'single'])
        doubles = len(pa_df[pa_df['events'] == 'double'])
        triples = len(pa_df[pa_df['events'] == 'triple'])
        hrs = len(pa_df[pa_df['events'] == 'home_run'])
        
        total_bases = singles + (doubles * 2) + (triples * 3) + (hrs * 4)
        
        return total_bases / len(pa_df)
    
    def _get_confidence(self, pa_count):
        """
        Return confidence score (0-100) based on sample size
        
        5-20 PA = low confidence (30-50)
        20-50 PA = medium confidence (50-70)
        50-100 PA = high confidence (70-85)
        100+ PA = very high confidence (85-100)
        """
        if pa_count < 20:
            return int(30 + (pa_count / 20) * 20)
        elif pa_count < 50:
            return int(50 + ((pa_count - 20) / 30) * 20)
        elif pa_count < 100:
            return int(70 + ((pa_count - 50) / 50) * 15)
        else:
            return min(100, int(85 + ((pa_count - 100) / 100) * 15))


def demo_usage():
    """Demonstrate matchup analysis with real player IDs"""
    analyzer = MatchupAnalyzer()
    analyzer.load_data()
    
    print("\n" + "="*70)
    print("⚾ BATTER VS PITCHER MATCHUP ANALYSIS - REAL DATA")
    print("="*70)
    
    # Find actual batter and pitcher IDs from data
    batters = analyzer.df.groupby('batter')['events'].count().nlargest(10)
    pitchers = analyzer.df.groupby('pitcher')['events'].count().nlargest(10)
    
    print(f"\n📊 Database contains:")
    print(f"   {analyzer.df['batter'].nunique():,} unique batters")
    print(f"   {analyzer.df['pitcher'].nunique():,} unique pitchers")
    print(f"   {len(analyzer.df[analyzer.df['events'].notna()]):,} total plate appearances")
    
    # Show top batter ID
    top_batter_id = batters.index[0]
    print(f"\n🏏 Most frequent batter ID: {top_batter_id} ({batters.iloc[0]:,} PA)")
    
    # Show platoon splits
    platoon = analyzer.get_platoon_stats(top_batter_id, 'R')
    if platoon:
        print(f"   vs RHP: {platoon['avg']} AVG, {platoon['slg']} SLG ({platoon['pa']} PA)")
    
    # Find a matchup
    matchup_data = analyzer.df[
        (analyzer.df['batter'] == top_batter_id) & 
        (analyzer.df['events'].notna())
    ]
    
    if len(matchup_data) > 0:
        top_pitcher_id = matchup_data['pitcher'].value_counts().index[0]
        h2h = analyzer.get_h2h_matchup(top_batter_id, top_pitcher_id)
        
        if h2h and h2h['plate_appearances'] >= 5:
            print(f"\n⚔️  H2H Matchup: Batter {top_batter_id} vs Pitcher {top_pitcher_id}")
            print(f"   PA: {h2h['plate_appearances']}")
            print(f"   AVG: {h2h['avg']} | OPS: {h2h['ops']}")
            print(f"   HR Rate: {h2h['hr_rate']*100:.1f}% | K Rate: {h2h['k_rate']*100:.1f}%")
            print(f"   Recent: {h2h['recent_avg']} AVG")
            print(f"   Platoon: {h2h['platoon']}")
            print(f"   Confidence: {h2h['confidence']}/100")
            
            score = analyzer.get_matchup_score(top_batter_id, top_pitcher_id)
            print(f"\n🎯 Matchup Score: {score}/100 {'(Batter favored)' if score > 60 else '(Pitcher favored)' if score < 40 else '(Neutral)'}")
    
    print("\n✅ Matchup analysis complete - ready for integration")


if __name__ == "__main__":
    demo_usage()
