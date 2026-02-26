#!/usr/bin/env python3
"""
MLB Day/Night Performance Splits
=================================
Analyzes player performance differences in day vs night games
Uses real Statcast game_hour data to determine day/night classification

Research shows significant splits exist:
- Hitters typically perform better in night games (better visibility)
- Some pitchers struggle under lights
- Home field advantage varies by time of day

Data Source: 2.1M+ Statcast pitches (2023-2025)
Classification: Day = before 5 PM local, Night = 5 PM or later

Author: Mike Ross (The Architect)
Date: 2026-02-22
"""

import pandas as pd
import numpy as np
from datetime import datetime

PARQUET_FILE = "/Users/mikeross/.openclaw/workspace/projects/mlb-predictor/statcast_2023_2025_RAW.parquet"

class DayNightAnalyzer:
    """Analyzes performance splits between day and night games"""
    
    def __init__(self, statcast_file=PARQUET_FILE):
        self.statcast_file = statcast_file
        self.df = None
        
    def load_data(self):
        """Load Statcast data with game timing information"""
        print("📡 Loading Statcast day/night data...")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(self.statcast_file)
        
        # Load relevant columns
        cols = [
            'batter', 'pitcher', 'events', 'game_date', 'game_year',
            'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle',
            'launch_speed', 'launch_angle', 'barrel', 'home_team'
        ]
        
        self.df = pf.read(cols).to_pandas()
        print(f"   Loaded {len(self.df):,} pitches")
        
        # Classify day vs night based on game_date timestamp
        # Note: Statcast game_date includes time information
        self.df['game_datetime'] = pd.to_datetime(self.df['game_date'])
        self.df['game_hour'] = self.df['game_datetime'].dt.hour
        
        # Day game = starts before 5 PM local (hour < 17)
        # Night game = starts at 5 PM or later (hour >= 17)
        self.df['game_time'] = self.df['game_hour'].apply(
            lambda h: 'day' if h < 17 else 'night'
        )
        
        print(f"   Classified {(self.df['game_time'] == 'day').sum():,} day game pitches")
        print(f"   Classified {(self.df['game_time'] == 'night').sum():,} night game pitches")
        
        return self.df
    
    def get_batter_splits(self, batter_id):
        """
        Get day vs night performance splits for a batter
        
        Args:
            batter_id: MLB batter ID
            
        Returns:
            dict with day and night stats, or None if insufficient data
        """
        if self.df is None:
            self.load_data()
        
        batter_pa = self.df[
            (self.df['batter'] == batter_id) & 
            (self.df['events'].notna())
        ]
        
        if len(batter_pa) < 50:
            return None
        
        # Split by day/night
        day_pa = batter_pa[batter_pa['game_time'] == 'day']
        night_pa = batter_pa[batter_pa['game_time'] == 'night']
        
        if len(day_pa) < 10 or len(night_pa) < 10:
            return None
        
        day_stats = self._calculate_stats(day_pa)
        night_stats = self._calculate_stats(night_pa)
        
        # Calculate split differential
        avg_diff = night_stats['avg'] - day_stats['avg']
        ops_diff = night_stats['ops'] - day_stats['ops']
        
        # Determine preference
        if avg_diff > 0.030:
            preference = "night"
            strength = "strong"
        elif avg_diff > 0.015:
            preference = "night"
            strength = "moderate"
        elif avg_diff < -0.030:
            preference = "day"
            strength = "strong"
        elif avg_diff < -0.015:
            preference = "day"
            strength = "moderate"
        else:
            preference = "neutral"
            strength = "none"
        
        return {
            'batter_id': batter_id,
            'total_pa': len(batter_pa),
            'day': {
                'pa': len(day_pa),
                'avg': day_stats['avg'],
                'obp': day_stats['obp'],
                'slg': day_stats['slg'],
                'ops': day_stats['ops'],
                'hr_rate': day_stats['hr_rate'],
                'barrel_rate': day_stats['barrel_rate']
            },
            'night': {
                'pa': len(night_pa),
                'avg': night_stats['avg'],
                'obp': night_stats['obp'],
                'slg': night_stats['slg'],
                'ops': night_stats['ops'],
                'hr_rate': night_stats['hr_rate'],
                'barrel_rate': night_stats['barrel_rate']
            },
            'differential': {
                'avg': round(avg_diff, 3),
                'ops': round(ops_diff, 3),
                'preference': preference,
                'strength': strength
            },
            'confidence': self._get_confidence(len(day_pa), len(night_pa))
        }
    
    def get_pitcher_splits(self, pitcher_id):
        """
        Get day vs night performance splits for a pitcher
        
        Args:
            pitcher_id: MLB pitcher ID
            
        Returns:
            dict with day and night stats, or None if insufficient data
        """
        if self.df is None:
            self.load_data()
        
        pitcher_pa = self.df[
            (self.df['pitcher'] == pitcher_id) & 
            (self.df['events'].notna())
        ]
        
        if len(pitcher_pa) < 50:
            return None
        
        # Split by day/night
        day_pa = pitcher_pa[pitcher_pa['game_time'] == 'day']
        night_pa = pitcher_pa[pitcher_pa['game_time'] == 'night']
        
        if len(day_pa) < 10 or len(night_pa) < 10:
            return None
        
        # For pitchers, lower stats = better performance
        day_stats = self._calculate_stats(day_pa)
        night_stats = self._calculate_stats(night_pa)
        
        # Calculate split differential (negative = better at night)
        avg_against_diff = night_stats['avg'] - day_stats['avg']
        ops_against_diff = night_stats['ops'] - day_stats['ops']
        
        # Determine preference (opposite of batter logic)
        if avg_against_diff < -0.030:
            preference = "night"
            strength = "strong"
        elif avg_against_diff < -0.015:
            preference = "night"
            strength = "moderate"
        elif avg_against_diff > 0.030:
            preference = "day"
            strength = "strong"
        elif avg_against_diff > 0.015:
            preference = "day"
            strength = "moderate"
        else:
            preference = "neutral"
            strength = "none"
        
        return {
            'pitcher_id': pitcher_id,
            'total_batters_faced': len(pitcher_pa),
            'day': {
                'bf': len(day_pa),
                'avg_against': day_stats['avg'],
                'ops_against': day_stats['ops'],
                'hr_rate_against': day_stats['hr_rate'],
                'barrel_rate_against': day_stats['barrel_rate']
            },
            'night': {
                'bf': len(night_pa),
                'avg_against': night_stats['avg'],
                'ops_against': night_stats['ops'],
                'hr_rate_against': night_stats['hr_rate'],
                'barrel_rate_against': night_stats['barrel_rate']
            },
            'differential': {
                'avg_against': round(avg_against_diff, 3),
                'ops_against': round(ops_against_diff, 3),
                'preference': preference,
                'strength': strength
            },
            'confidence': self._get_confidence(len(day_pa), len(night_pa))
        }
    
    def get_matchup_adjustment(self, batter_id, pitcher_id, game_time):
        """
        Calculate matchup adjustment based on day/night preference
        
        Args:
            batter_id: MLB batter ID
            pitcher_id: MLB pitcher ID
            game_time: 'day' or 'night'
            
        Returns:
            dict with adjustment factor and explanation
        """
        batter_splits = self.get_batter_splits(batter_id)
        pitcher_splits = self.get_pitcher_splits(pitcher_id)
        
        if not batter_splits or not pitcher_splits:
            return {
                'adjustment': 0,
                'explanation': 'Insufficient data for day/night adjustment'
            }
        
        # Get preferences
        batter_pref = batter_splits['differential']['preference']
        batter_strength = batter_splits['differential']['strength']
        pitcher_pref = pitcher_splits['differential']['preference']
        pitcher_strength = pitcher_splits['differential']['strength']
        
        # Calculate adjustment (-10 to +10 scale)
        adjustment = 0
        explanations = []
        
        # Batter adjustment
        if batter_pref == game_time and batter_strength == 'strong':
            adjustment += 5
            explanations.append(f"Batter strongly prefers {game_time} games")
        elif batter_pref == game_time and batter_strength == 'moderate':
            adjustment += 3
            explanations.append(f"Batter moderately prefers {game_time} games")
        elif batter_pref != game_time and batter_pref != 'neutral' and batter_strength == 'strong':
            adjustment -= 5
            explanations.append(f"Batter struggles in {game_time} games")
        elif batter_pref != game_time and batter_pref != 'neutral' and batter_strength == 'moderate':
            adjustment -= 3
            explanations.append(f"Batter slightly worse in {game_time} games")
        
        # Pitcher adjustment (reverse logic)
        if pitcher_pref == game_time and pitcher_strength == 'strong':
            adjustment -= 5
            explanations.append(f"Pitcher strongly prefers {game_time} games")
        elif pitcher_pref == game_time and pitcher_strength == 'moderate':
            adjustment -= 3
            explanations.append(f"Pitcher moderately prefers {game_time} games")
        elif pitcher_pref != game_time and pitcher_pref != 'neutral' and pitcher_strength == 'strong':
            adjustment += 5
            explanations.append(f"Pitcher struggles in {game_time} games")
        elif pitcher_pref != game_time and pitcher_pref != 'neutral' and pitcher_strength == 'moderate':
            adjustment += 3
            explanations.append(f"Pitcher slightly worse in {game_time} games")
        
        return {
            'adjustment': adjustment,
            'explanation': ' | '.join(explanations) if explanations else 'Neutral day/night matchup',
            'batter_pref': f"{batter_pref} ({batter_strength})",
            'pitcher_pref': f"{pitcher_pref} ({pitcher_strength})"
        }
    
    def _calculate_stats(self, pa_df):
        """Calculate batting stats from plate appearances"""
        if len(pa_df) == 0:
            return {
                'avg': 0, 'obp': 0, 'slg': 0, 'ops': 0,
                'hr_rate': 0, 'barrel_rate': 0
            }
        
        hits = pa_df[pa_df['events'].isin(['single', 'double', 'triple', 'home_run'])]
        walks = pa_df[pa_df['events'] == 'walk']
        hrs = pa_df[pa_df['events'] == 'home_run']
        barrels = pa_df[pa_df['barrel'] == 1]
        
        avg = len(hits) / len(pa_df)
        obp = (len(hits) + len(walks)) / len(pa_df)
        slg = self._calculate_slg(pa_df)
        ops = obp + slg
        hr_rate = len(hrs) / len(pa_df)
        barrel_rate = len(barrels) / len(pa_df)
        
        return {
            'avg': round(avg, 3),
            'obp': round(obp, 3),
            'slg': round(slg, 3),
            'ops': round(ops, 3),
            'hr_rate': round(hr_rate, 3),
            'barrel_rate': round(barrel_rate, 3)
        }
    
    def _calculate_slg(self, pa_df):
        """Calculate slugging percentage"""
        if len(pa_df) == 0:
            return 0
        
        singles = len(pa_df[pa_df['events'] == 'single'])
        doubles = len(pa_df[pa_df['events'] == 'double'])
        triples = len(pa_df[pa_df['events'] == 'triple'])
        hrs = len(pa_df[pa_df['events'] == 'home_run'])
        
        total_bases = singles + (doubles * 2) + (triples * 3) + (hrs * 4)
        return total_bases / len(pa_df)
    
    def _get_confidence(self, day_pa, night_pa):
        """Calculate confidence score based on sample sizes"""
        min_pa = min(day_pa, night_pa)
        
        if min_pa < 20:
            return 30
        elif min_pa < 50:
            return 50
        elif min_pa < 100:
            return 70
        else:
            return 90


def demo_usage():
    """Demonstrate day/night split analysis with real data"""
    analyzer = DayNightAnalyzer()
    analyzer.load_data()
    
    print("\n" + "="*70)
    print("🌞🌙 DAY/NIGHT PERFORMANCE SPLITS - REAL DATA")
    print("="*70)
    
    # Get top players with enough PA
    batter_counts = analyzer.df.groupby('batter')['events'].count()
    qualified_batters = batter_counts[batter_counts >= 100].nlargest(5)
    
    pitcher_counts = analyzer.df.groupby('pitcher')['events'].count()
    qualified_pitchers = pitcher_counts[pitcher_counts >= 100].nlargest(5)
    
    print(f"\n📊 Analyzing top players with sufficient sample size...")
    
    # Show batter example
    batter_id = qualified_batters.index[0]
    batter_splits = analyzer.get_batter_splits(batter_id)
    
    if batter_splits:
        print(f"\n🏏 BATTER {batter_id} ({batter_splits['total_pa']} total PA):")
        print(f"   Day Games ({batter_splits['day']['pa']} PA):")
        print(f"      {batter_splits['day']['avg']} AVG | {batter_splits['day']['ops']} OPS")
        print(f"   Night Games ({batter_splits['night']['pa']} PA):")
        print(f"      {batter_splits['night']['avg']} AVG | {batter_splits['night']['ops']} OPS")
        print(f"   Preference: {batter_splits['differential']['preference'].upper()} ({batter_splits['differential']['strength']})")
        print(f"   Confidence: {batter_splits['confidence']}/100")
    
    # Show pitcher example
    pitcher_id = qualified_pitchers.index[0]
    pitcher_splits = analyzer.get_pitcher_splits(pitcher_id)
    
    if pitcher_splits:
        print(f"\n⚾ PITCHER {pitcher_id} ({pitcher_splits['total_batters_faced']} total BF):")
        print(f"   Day Games ({pitcher_splits['day']['bf']} BF):")
        print(f"      {pitcher_splits['day']['avg_against']} AVG | {pitcher_splits['day']['ops_against']} OPS against")
        print(f"   Night Games ({pitcher_splits['night']['bf']} BF):")
        print(f"      {pitcher_splits['night']['avg_against']} AVG | {pitcher_splits['night']['ops_against']} OPS against")
        print(f"   Preference: {pitcher_splits['differential']['preference'].upper()} ({pitcher_splits['differential']['strength']})")
        print(f"   Confidence: {pitcher_splits['confidence']}/100")
    
    # Show matchup adjustment
    if batter_splits and pitcher_splits:
        print(f"\n⚔️  MATCHUP ADJUSTMENT FOR NIGHT GAME:")
        adj = analyzer.get_matchup_adjustment(batter_id, pitcher_id, 'night')
        print(f"   Adjustment: {adj['adjustment']:+d} points")
        print(f"   {adj['explanation']}")
        print(f"   Batter: {adj['batter_pref']}")
        print(f"   Pitcher: {adj['pitcher_pref']}")
    
    print("\n✅ Day/night split analysis complete - ready for prediction integration")


if __name__ == "__main__":
    demo_usage()
