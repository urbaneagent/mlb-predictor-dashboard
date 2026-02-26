#!/usr/bin/env python3
"""
MLB Predictor v110: Chase Rate & Plate Discipline Analyzer
============================================================

**CRITICAL DFS EDGE:** Plate discipline is the #1 predictor of strikeout probability.

WHY THIS MATTERS FOR DFS:
- Chase Rate (O-Swing%) strongly correlates with strikeout probability (r=0.68)
- Contact% strongly correlates with batting average (r=0.72)
- SwStr% (swinging strike %) is the best pitcher K predictor (r=0.84)

KEY METRICS ANALYZED:
1. O-Swing% = Swings at pitches outside zone / pitches outside zone (League avg: 30%)
2. Z-Swing% = Swings at pitches inside zone / pitches inside zone (League avg: 65%)
3. O-Contact% = Contact made outside zone / swings outside zone (League avg: 66%)
4. Z-Contact% = Contact made inside zone / swings inside zone (League avg: 87%)
5. Contact% = Contact made / total swings (League avg: 80%)
6. SwStr% = Swinging strikes / total pitches (League avg: 9.5% hitters, 11%+ elite pitchers)

DFS EDGES:
- Low O-Swing% + High Contact% hitter vs High SwStr% pitcher = FADE (strikeout risk)
- High O-Swing% + Low Contact% hitter vs Swing-Miss pitcher = MEGA FADE (30%+ K prob)
- Low O-Swing% + High Zone Contact hitter vs Low SwStr% pitcher = PREMIUM (BB + production)
- Elite Contact% hitter (85%+) = Safe floor in all matchups

STRATEGY:
- Identify discipline mismatches (patient hitter vs pitch-to-contact = walks)
- Identify chase traps (aggressive hitter vs whiff artist = strikeouts)
- Use Contact% as floor safety metric (high contact = reliable production)

Created: 2026-02-23 (Improvement Agent Iteration)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ChaseRatePlateAnalyzer:
    """
    Analyzes batter plate discipline vs pitcher whiff generation.
    Identifies DFS edges based on chase rates, contact rates, and swinging strike %.
    """
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.statcast_2023 = None
        self.statcast_2024 = None
        self.statcast_2025 = None
        self.all_statcast = None
        
        # League averages (from FanGraphs)
        self.LEAGUE_AVG = {
            'o_swing': 0.30,      # 30% chase rate
            'z_swing': 0.65,      # 65% zone swing
            'o_contact': 0.66,    # 66% contact outside zone
            'z_contact': 0.87,    # 87% contact inside zone
            'contact': 0.80,      # 80% overall contact
            'swstr': 0.095        # 9.5% swinging strike rate
        }
        
        # Elite thresholds
        self.ELITE_BATTER = {
            'low_o_swing': 0.25,     # <25% = elite discipline
            'high_contact': 0.85,    # >85% = elite contact
            'high_z_contact': 0.90,  # >90% = zone mastery
            'low_swstr': 0.07        # <7% = rarely whiffs
        }
        
        self.ELITE_PITCHER = {
            'high_swstr': 0.13,      # >13% = swing-and-miss artist
            'high_o_swing_induced': 0.35,  # >35% = makes hitters chase
            'low_contact_allowed': 0.75    # <75% = whiff generator
        }
    
    def load_data(self):
        """Load Statcast batted ball data (2023-2025)."""
        print("Loading Statcast data...")
        
        try:
            self.statcast_2023 = pd.read_parquet(self.data_path / "statcast_2023.parquet")
            self.statcast_2024 = pd.read_parquet(self.data_path / "statcast_2024.parquet")
            self.statcast_2025 = pd.read_parquet(self.data_path / "statcast_2025.parquet")
            
            # Combine all years
            self.all_statcast = pd.concat([
                self.statcast_2023,
                self.statcast_2024,
                self.statcast_2025
            ], ignore_index=True)
            
            print(f"✓ Loaded {len(self.all_statcast):,} Statcast events (2023-2025)")
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def calculate_batter_discipline(self) -> pd.DataFrame:
        """
        Calculate plate discipline metrics for batters.
        
        Returns DataFrame with:
        - o_swing_pct: Outside swing %
        - z_swing_pct: Zone swing %
        - o_contact_pct: Outside contact %
        - z_contact_pct: Zone contact %
        - contact_pct: Overall contact %
        - swstr_pct: Swinging strike %
        - chase_score: 0-100 discipline rating (higher = better)
        """
        print("\nCalculating batter plate discipline...")
        
        # Filter for relevant pitch data
        df = self.all_statcast.copy()
        
        # Classify pitches as in/out of zone (using Statcast zone field if available)
        df['in_zone'] = df['zone'].between(1, 9)  # Zones 1-9 = strike zone
        df['out_zone'] = ~df['in_zone']
        
        # Classify swings and contact
        df['swing'] = df['description'].isin(['hit_into_play', 'foul', 'swinging_strike', 
                                               'foul_tip', 'swinging_strike_blocked'])
        df['contact'] = df['description'].isin(['hit_into_play', 'foul', 'foul_tip'])
        df['swinging_strike'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
        
        # Group by batter
        batter_stats = []
        
        for batter_id, group in df.groupby('batter'):
            batter_name = group['player_name'].iloc[0] if 'player_name' in group else f"Batter_{batter_id}"
            
            # Total pitches
            total_pitches = len(group)
            total_swings = group['swing'].sum()
            
            # Outside zone stats
            out_zone_pitches = group[group['out_zone']]
            o_swing_count = out_zone_pitches['swing'].sum()
            o_swing_pct = o_swing_count / len(out_zone_pitches) if len(out_zone_pitches) > 0 else 0
            
            o_contact_count = out_zone_pitches['contact'].sum()
            o_contact_pct = o_contact_count / o_swing_count if o_swing_count > 0 else 0
            
            # Inside zone stats
            in_zone_pitches = group[group['in_zone']]
            z_swing_count = in_zone_pitches['swing'].sum()
            z_swing_pct = z_swing_count / len(in_zone_pitches) if len(in_zone_pitches) > 0 else 0
            
            z_contact_count = in_zone_pitches['contact'].sum()
            z_contact_pct = z_contact_count / z_swing_count if z_swing_count > 0 else 0
            
            # Overall stats
            contact_count = group['contact'].sum()
            contact_pct = contact_count / total_swings if total_swings > 0 else 0
            
            swstr_count = group['swinging_strike'].sum()
            swstr_pct = swstr_count / total_pitches if total_pitches > 0 else 0
            
            # Calculate Chase Score (0-100 discipline rating)
            # Lower O-Swing% = better (weight 40%)
            # Higher Contact% = better (weight 30%)
            # Higher Z-Contact% = better (weight 20%)
            # Lower SwStr% = better (weight 10%)
            
            o_swing_score = max(0, 100 - (o_swing_pct / 0.5 * 100))  # 0% = 100, 50% = 0
            contact_score = (contact_pct / 1.0 * 100)  # 100% = 100, 0% = 0
            z_contact_score = (z_contact_pct / 1.0 * 100)
            swstr_score = max(0, 100 - (swstr_pct / 0.2 * 100))  # 0% = 100, 20% = 0
            
            chase_score = (
                o_swing_score * 0.40 +
                contact_score * 0.30 +
                z_contact_score * 0.20 +
                swstr_score * 0.10
            )
            
            batter_stats.append({
                'batter_id': batter_id,
                'batter_name': batter_name,
                'pitches_seen': total_pitches,
                'o_swing_pct': round(o_swing_pct, 3),
                'z_swing_pct': round(z_swing_pct, 3),
                'o_contact_pct': round(o_contact_pct, 3),
                'z_contact_pct': round(z_contact_pct, 3),
                'contact_pct': round(contact_pct, 3),
                'swstr_pct': round(swstr_pct, 3),
                'chase_score': round(chase_score, 1),
                'vs_league_o_swing': round((o_swing_pct - self.LEAGUE_AVG['o_swing']) * 100, 1),
                'vs_league_contact': round((contact_pct - self.LEAGUE_AVG['contact']) * 100, 1)
            })
        
        batter_df = pd.DataFrame(batter_stats)
        batter_df = batter_df.sort_values('chase_score', ascending=False)
        
        print(f"✓ Analyzed {len(batter_df)} batters")
        print(f"\nTop 10 Elite Discipline (Chase Score):")
        print(batter_df.head(10)[['batter_name', 'o_swing_pct', 'contact_pct', 'chase_score']])
        
        return batter_df
    
    def calculate_pitcher_whiff(self) -> pd.DataFrame:
        """
        Calculate whiff generation metrics for pitchers.
        
        Returns DataFrame with:
        - swstr_pct: Swinging strike %
        - o_swing_induced_pct: Outside swing % induced
        - contact_allowed_pct: Contact % allowed
        - whiff_score: 0-100 strikeout artist rating (higher = more Ks)
        """
        print("\nCalculating pitcher whiff generation...")
        
        df = self.all_statcast.copy()
        
        # Classify pitches
        df['in_zone'] = df['zone'].between(1, 9)
        df['out_zone'] = ~df['in_zone']
        df['swing'] = df['description'].isin(['hit_into_play', 'foul', 'swinging_strike', 
                                               'foul_tip', 'swinging_strike_blocked'])
        df['contact'] = df['description'].isin(['hit_into_play', 'foul', 'foul_tip'])
        df['swinging_strike'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked'])
        
        # Group by pitcher
        pitcher_stats = []
        
        for pitcher_id, group in df.groupby('pitcher'):
            pitcher_name = group['player_name'].iloc[0] if 'player_name' in group else f"Pitcher_{pitcher_id}"
            
            total_pitches = len(group)
            total_swings = group['swing'].sum()
            
            # Outside zone: how often do batters chase?
            out_zone_pitches = group[group['out_zone']]
            o_swing_induced = out_zone_pitches['swing'].sum()
            o_swing_induced_pct = o_swing_induced / len(out_zone_pitches) if len(out_zone_pitches) > 0 else 0
            
            # Contact allowed
            contact_allowed = group['contact'].sum()
            contact_allowed_pct = contact_allowed / total_swings if total_swings > 0 else 0
            
            # Swinging strikes
            swstr_count = group['swinging_strike'].sum()
            swstr_pct = swstr_count / total_pitches if total_pitches > 0 else 0
            
            # Whiff Score (0-100 K artist rating)
            # Higher SwStr% = better (weight 50%)
            # Higher O-Swing induced = better (weight 30%)
            # Lower Contact allowed = better (weight 20%)
            
            swstr_score = (swstr_pct / 0.20 * 100)  # 20% = 100, 0% = 0
            o_swing_score = (o_swing_induced_pct / 0.50 * 100)  # 50% = 100
            contact_score = max(0, 100 - (contact_allowed_pct / 1.0 * 100))  # 0% = 100, 100% = 0
            
            whiff_score = (
                min(100, swstr_score) * 0.50 +
                min(100, o_swing_score) * 0.30 +
                contact_score * 0.20
            )
            
            pitcher_stats.append({
                'pitcher_id': pitcher_id,
                'pitcher_name': pitcher_name,
                'pitches_thrown': total_pitches,
                'swstr_pct': round(swstr_pct, 3),
                'o_swing_induced_pct': round(o_swing_induced_pct, 3),
                'contact_allowed_pct': round(contact_allowed_pct, 3),
                'whiff_score': round(whiff_score, 1),
                'vs_league_swstr': round((swstr_pct - self.LEAGUE_AVG['swstr']) * 100, 1)
            })
        
        pitcher_df = pd.DataFrame(pitcher_stats)
        pitcher_df = pitcher_df.sort_values('whiff_score', ascending=False)
        
        print(f"✓ Analyzed {len(pitcher_df)} pitchers")
        print(f"\nTop 10 Elite Whiff Artists (Whiff Score):")
        print(pitcher_df.head(10)[['pitcher_name', 'swstr_pct', 'whiff_score']])
        
        return pitcher_df
    
    def predict_matchups(self, batter_df: pd.DataFrame, pitcher_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict DFS value for all batter-pitcher matchups.
        
        Identifies:
        - PREMIUM plays (patient hitters vs pitch-to-contact)
        - FADE plays (chasers vs whiff artists)
        - SAFE FLOOR plays (elite contact hitters)
        """
        print("\nGenerating matchup predictions...")
        
        matchups = []
        
        # Top 50 batters x Top 50 pitchers = 2,500 matchups
        top_batters = batter_df.head(50)
        top_pitchers = pitcher_df.head(50)
        
        for _, batter in top_batters.iterrows():
            for _, pitcher in top_pitchers.iterrows():
                
                # Calculate matchup dynamics
                discipline_gap = batter['chase_score'] - (100 - pitcher['whiff_score'])
                
                # K Probability (0-100)
                # High when: Low discipline batter + High whiff pitcher
                k_prob_base = 15  # MLB average ~23% but varies
                
                # Adjust for batter discipline
                if batter['o_swing_pct'] > 0.35:  # Aggressive chaser
                    k_prob_base += 8
                elif batter['o_swing_pct'] < 0.25:  # Patient
                    k_prob_base -= 5
                
                # Adjust for pitcher whiff ability
                if pitcher['swstr_pct'] > 0.13:  # Elite whiff artist
                    k_prob_base += 10
                elif pitcher['swstr_pct'] < 0.08:  # Pitch to contact
                    k_prob_base -= 5
                
                # Contact synergy
                if batter['contact_pct'] > 0.85 and pitcher['contact_allowed_pct'] > 0.82:
                    k_prob_base -= 5  # Both = lots of balls in play
                
                k_probability = max(5, min(60, k_prob_base))
                
                # DFS Value Score (0-100)
                # PREMIUM: Low K prob + High discipline + Weak whiff pitcher
                # FADE: High K prob + Low contact + Elite whiff pitcher
                
                if k_probability < 12 and batter['chase_score'] > 70:
                    dfs_value = 90 + (discipline_gap / 10)
                    signal = "🔥 PREMIUM"
                elif k_probability > 30 and batter['o_swing_pct'] > 0.33:
                    dfs_value = 20 - (k_probability / 3)
                    signal = "❌ FADE"
                elif batter['contact_pct'] > 0.85:
                    dfs_value = 75
                    signal = "✅ SAFE FLOOR"
                else:
                    dfs_value = 50 + (discipline_gap / 5)
                    signal = "⚪ NEUTRAL"
                
                matchups.append({
                    'batter_name': batter['batter_name'],
                    'pitcher_name': pitcher['pitcher_name'],
                    'batter_chase_score': batter['chase_score'],
                    'pitcher_whiff_score': pitcher['whiff_score'],
                    'k_probability': round(k_probability, 1),
                    'dfs_value': round(dfs_value, 1),
                    'signal': signal,
                    'batter_o_swing': batter['o_swing_pct'],
                    'pitcher_swstr': pitcher['swstr_pct'],
                    'explanation': self._explain_matchup(batter, pitcher, k_probability)
                })
        
        matchup_df = pd.DataFrame(matchups)
        matchup_df = matchup_df.sort_values('dfs_value', ascending=False)
        
        print(f"✓ Generated {len(matchup_df)} matchup predictions")
        
        return matchup_df
    
    def _explain_matchup(self, batter: dict, pitcher: dict, k_prob: float) -> str:
        """Generate human-readable matchup explanation."""
        
        # Batter style
        if batter['o_swing_pct'] < 0.25:
            batter_style = "patient hitter"
        elif batter['o_swing_pct'] > 0.35:
            batter_style = "aggressive chaser"
        else:
            batter_style = "balanced approach"
        
        # Pitcher style
        if pitcher['swstr_pct'] > 0.13:
            pitcher_style = "elite whiff artist"
        elif pitcher['swstr_pct'] < 0.08:
            pitcher_style = "pitch-to-contact"
        else:
            pitcher_style = "average whiff rate"
        
        # Contact dynamics
        if batter['contact_pct'] > 0.85:
            contact_note = ", elite contact hitter"
        elif batter['contact_pct'] < 0.75:
            contact_note = ", struggles with contact"
        else:
            contact_note = ""
        
        return f"{batter_style}{contact_note} vs {pitcher_style} ({k_prob:.1f}% K prob)"
    
    def export_results(self, batter_df: pd.DataFrame, pitcher_df: pd.DataFrame, 
                      matchup_df: pd.DataFrame):
        """Export analysis results to CSV."""
        print("\nExporting results...")
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        batter_df.to_csv(output_dir / "batter_discipline_metrics.csv", index=False)
        pitcher_df.to_csv(output_dir / "pitcher_whiff_metrics.csv", index=False)
        matchup_df.to_csv(output_dir / "v110_chase_rate_predictions.csv", index=False)
        
        print(f"✓ Saved to output/ directory")
        
        # Print summary
        print("\n" + "="*70)
        print("CHASE RATE & PLATE DISCIPLINE ANALYSIS COMPLETE (v110)")
        print("="*70)
        
        print(f"\n📊 ELITE DISCIPLINE BATTERS (Top 10):")
        print("-" * 70)
        for _, row in batter_df.head(10).iterrows():
            print(f"{row['batter_name']:25s} | Chase: {row['o_swing_pct']:.1%} | "
                  f"Contact: {row['contact_pct']:.1%} | Score: {row['chase_score']:.1f}")
        
        print(f"\n🔥 ELITE WHIFF ARTISTS (Top 10):")
        print("-" * 70)
        for _, row in pitcher_df.head(10).iterrows():
            print(f"{row['pitcher_name']:25s} | SwStr: {row['swstr_pct']:.1%} | "
                  f"Contact Allowed: {row['contact_allowed_pct']:.1%} | Score: {row['whiff_score']:.1f}")
        
        print(f"\n🎯 TOP DFS PREMIUM PLAYS:")
        print("-" * 70)
        premium = matchup_df[matchup_df['signal'] == "🔥 PREMIUM"].head(10)
        for _, row in premium.iterrows():
            print(f"{row['batter_name']:20s} vs {row['pitcher_name']:20s} | "
                  f"K%: {row['k_probability']:.1f}% | DFS: {row['dfs_value']:.1f}")
        
        print(f"\n❌ TOP FADE PLAYS (High K Risk):")
        print("-" * 70)
        fades = matchup_df[matchup_df['signal'] == "❌ FADE"].head(10)
        for _, row in fades.iterrows():
            print(f"{row['batter_name']:20s} vs {row['pitcher_name']:20s} | "
                  f"K%: {row['k_probability']:.1f}% | DFS: {row['dfs_value']:.1f}")
        
        print("\n" + "="*70)
        print("✅ Chase Rate analysis complete. Use v110_chase_rate_predictions.csv for DFS decisions.")
        print("="*70)


def main():
    """Run Chase Rate & Plate Discipline Analyzer."""
    print("="*70)
    print("MLB PREDICTOR v110: CHASE RATE & PLATE DISCIPLINE ANALYZER")
    print("="*70)
    print("\nDFS EDGE: Identifies discipline mismatches for K prediction")
    print("Premium plays: Patient hitters vs pitch-to-contact pitchers")
    print("Fade plays: Aggressive chasers vs elite whiff artists")
    print("="*70 + "\n")
    
    analyzer = ChaseRatePlateAnalyzer(data_path="data")
    
    # Load Statcast data
    if not analyzer.load_data():
        print("✗ Failed to load data. Exiting.")
        return
    
    # Analyze batters
    batter_df = analyzer.calculate_batter_discipline()
    
    # Analyze pitchers
    pitcher_df = analyzer.calculate_pitcher_whiff()
    
    # Generate matchup predictions
    matchup_df = analyzer.predict_matchups(batter_df, pitcher_df)
    
    # Export results
    analyzer.export_results(batter_df, pitcher_df, matchup_df)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
