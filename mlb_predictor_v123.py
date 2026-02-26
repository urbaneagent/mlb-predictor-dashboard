#!/usr/bin/env python3
"""
MLB Predictor v123 - Platoon Splits (L/R Matchup Advantage) Analyzer

RESEARCH FOUNDATION:
- FanGraphs (2022): "Platoon advantage causes 80+ point OPS swing"
- Baseball Prospectus (2006): "24-56 point advantage in OBP/SLG/OPS"
- RotoGrinders (2024): "Platoon splits are foundational DFS edge"
- FanGraphs Sabermetrics: "K% and BB% splits most predictive year-to-year"

THE SCIENCE:
- Opposite-handed batters see the ball better (breaking pitches move TOWARD them)
- Same-handed matchups = breaking pitches move AWAY = harder to track
- RHB vs RHP: .730 OPS league average
- RHB vs LHP: .810 OPS league average (+80 points!)
- LHB vs LHP: .720 OPS league average
- LHB vs RHP: .790 OPS league average (+70 points!)

KEY METRICS:
- wOBA Split: Difference between vs RHP and vs LHP
- K% Split: Strikeout rate difference (most predictive)
- BB% Split: Walk rate difference (second most predictive)
- ISO Split: Isolated power difference (third most predictive)
- Reverse-Split Players: Players who perform BETTER vs same-handed pitchers (rare but exploitable)

DFS APPLICATION:
1. Extreme platoon batters in favorable matchups = premium plays
2. Reverse-split pitchers (dominate same-handed batters) = fade opposing hitters
3. Switch hitters always have platoon advantage = slight premium
4. Pinch-hit situations: Managers platoon aggressively = DFS edge

INTEGRATION:
- Apply after barrel rate (v100), chase rate (v110), park factors (v120)
- Combine with weather (v121) and bullpen fatigue (v122)
- Example edge stack:
  - Elite barrel rate (+15%)
  - Low chase rate (+10%)
  - Short porch park (+20%)
  - Hot weather (+5%)
  - Platoon advantage (+25%)
  - Depleted bullpen (+15%)
  - TOTAL: +90% HR probability boost = GPP WINNER

VERSION: 123
AUTHOR: Mike Ross (The Architect)
DATE: 2026-02-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class PlatoonAnalyzer:
    """
    Analyzes left/right-handed matchup advantages for batters and pitchers.
    
    Calculates platoon splits from Statcast data and translates into
    DFS-relevant probability adjustments.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.statcast_data = None
        self.batter_splits = {}
        self.pitcher_splits = {}
        
        # League average platoon advantages (based on FanGraphs research)
        self.LEAGUE_AVG_SPLITS = {
            'RHB_vs_LHP': 0.080,  # +80 point OPS advantage
            'LHB_vs_RHP': 0.070,  # +70 point OPS advantage
            'RHB_vs_RHP': 0.000,  # baseline
            'LHB_vs_LHP': 0.000   # baseline
        }
        
        # Minimum PA required for reliable splits (FanGraphs guideline)
        self.MIN_PA_VS_L = 50
        self.MIN_PA_VS_R = 150
        
    def load_statcast_data(self) -> bool:
        """Load Statcast data from yearly parquet files."""
        try:
            dfs = []
            for year in [2023, 2024, 2025]:
                file_path = self.data_dir / f"statcast_{year}.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                    print(f"✓ Loaded {len(df):,} events from {year}")
            
            if not dfs:
                print("❌ No Statcast data found")
                return False
            
            self.statcast_data = pd.concat(dfs, ignore_index=True)
            print(f"✓ Total events: {len(self.statcast_data):,}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_batter_splits(self) -> pd.DataFrame:
        """
        Calculate platoon splits for all batters.
        
        Returns DataFrame with columns:
        - batter_name, stand (L/R)
        - vs_RHP: PA, wOBA, K%, BB%, ISO, SLG
        - vs_LHP: PA, wOBA, K%, BB%, ISO, SLG
        - split_wOBA: wOBA difference (vs favorable - vs unfavorable)
        - split_K: K% difference (lower is better)
        - split_ISO: ISO difference (power split)
        - platoon_score: 0-100 (how much batter benefits from platoon advantage)
        - reverse_split: Boolean (true if performs BETTER vs same-handed)
        """
        
        if self.statcast_data is None:
            print("❌ No data loaded")
            return pd.DataFrame()
        
        # Filter to batted ball events + strikeouts + walks
        df = self.statcast_data[
            self.statcast_data['events'].notna()
        ].copy()
        
        results = []
        
        # Group by batter
        for (batter, stand), group in df.groupby(['batter', 'stand']):
            batter_name = group['player_name'].iloc[0] if 'player_name' in group.columns else str(batter)
            
            # Split by pitcher handedness
            vs_rhp = group[group['p_throws'] == 'R']
            vs_lhp = group[group['p_throws'] == 'L']
            
            # Calculate metrics vs RHP
            if len(vs_rhp) >= self.MIN_PA_VS_R:
                rhp_woba = self._calculate_woba(vs_rhp)
                rhp_k_rate = (vs_rhp['events'] == 'strikeout').sum() / len(vs_rhp)
                rhp_bb_rate = (vs_rhp['events'] == 'walk').sum() / len(vs_rhp)
                rhp_iso = self._calculate_iso(vs_rhp)
                rhp_slg = self._calculate_slg(vs_rhp)
            else:
                rhp_woba = rhp_k_rate = rhp_bb_rate = rhp_iso = rhp_slg = None
            
            # Calculate metrics vs LHP
            if len(vs_lhp) >= self.MIN_PA_VS_L:
                lhp_woba = self._calculate_woba(vs_lhp)
                lhp_k_rate = (vs_lhp['events'] == 'strikeout').sum() / len(vs_lhp)
                lhp_bb_rate = (vs_lhp['events'] == 'walk').sum() / len(vs_lhp)
                lhp_iso = self._calculate_iso(vs_lhp)
                lhp_slg = self._calculate_slg(vs_lhp)
            else:
                lhp_woba = lhp_k_rate = lhp_bb_rate = lhp_iso = lhp_slg = None
            
            # Calculate splits (only if both handedness have sufficient data)
            if rhp_woba is not None and lhp_woba is not None:
                # Determine favorable matchup based on batter handedness
                if stand == 'R':  # RHB favors LHP
                    favorable_woba = lhp_woba
                    unfavorable_woba = rhp_woba
                    favorable_k = lhp_k_rate
                    unfavorable_k = rhp_k_rate
                    favorable_iso = lhp_iso
                    unfavorable_iso = rhp_iso
                else:  # LHB favors RHP
                    favorable_woba = rhp_woba
                    unfavorable_woba = lhp_woba
                    favorable_k = rhp_k_rate
                    unfavorable_k = lhp_k_rate
                    favorable_iso = rhp_iso
                    unfavorable_iso = lhp_iso
                
                split_woba = favorable_woba - unfavorable_woba
                split_k = unfavorable_k - favorable_k  # Lower K is better, so reverse
                split_iso = favorable_iso - unfavorable_iso
                
                # Detect reverse splits (performs better vs same-handed)
                reverse_split = split_woba < -0.020  # 20 point wOBA penalty
                
                # Calculate platoon score (0-100)
                # Weights: wOBA split (40%), K split (35%), ISO split (25%)
                woba_score = min(100, max(0, (split_woba + 0.050) / 0.100 * 100))
                k_score = min(100, max(0, (split_k + 0.100) / 0.150 * 100))
                iso_score = min(100, max(0, (split_iso + 0.050) / 0.100 * 100))
                
                platoon_score = (woba_score * 0.40 + k_score * 0.35 + iso_score * 0.25)
                
                results.append({
                    'batter_name': batter_name,
                    'batter_id': batter,
                    'stand': stand,
                    'pa_vs_rhp': len(vs_rhp),
                    'pa_vs_lhp': len(vs_lhp),
                    'woba_vs_rhp': rhp_woba,
                    'woba_vs_lhp': lhp_woba,
                    'k_rate_vs_rhp': rhp_k_rate,
                    'k_rate_vs_lhp': lhp_k_rate,
                    'bb_rate_vs_rhp': rhp_bb_rate,
                    'bb_rate_vs_lhp': lhp_bb_rate,
                    'iso_vs_rhp': rhp_iso,
                    'iso_vs_lhp': lhp_iso,
                    'slg_vs_rhp': rhp_slg,
                    'slg_vs_lhp': lhp_slg,
                    'split_woba': split_woba,
                    'split_k_rate': split_k,
                    'split_iso': split_iso,
                    'platoon_score': round(platoon_score, 1),
                    'reverse_split': reverse_split
                })
        
        df_splits = pd.DataFrame(results)
        
        # Sort by platoon score (most extreme splits first)
        df_splits = df_splits.sort_values('platoon_score', ascending=False)
        
        print(f"\n✓ Calculated splits for {len(df_splits)} batters")
        return df_splits
    
    def calculate_pitcher_splits(self) -> pd.DataFrame:
        """
        Calculate platoon splits for all pitchers.
        
        Returns DataFrame with columns:
        - pitcher_name, p_throws (L/R)
        - vs_RHB: PA, wOBA_allowed, K%, BB%, ISO_allowed
        - vs_LHB: PA, wOBA_allowed, K%, BB%, ISO_allowed
        - split_wOBA_allowed: wOBA difference (how much better vs same-handed)
        - split_K_rate: K% difference
        - reverse_split: Boolean (dominates opposite-handed batters)
        """
        
        if self.statcast_data is None:
            print("❌ No data loaded")
            return pd.DataFrame()
        
        df = self.statcast_data[
            self.statcast_data['events'].notna()
        ].copy()
        
        results = []
        
        # Group by pitcher
        for (pitcher, p_throws), group in df.groupby(['pitcher', 'p_throws']):
            pitcher_name = group['pitcher_name'].iloc[0] if 'pitcher_name' in group.columns else str(pitcher)
            
            # Split by batter handedness
            vs_rhb = group[group['stand'] == 'R']
            vs_lhb = group[group['stand'] == 'L']
            
            # Calculate metrics vs RHB
            if len(vs_rhb) >= self.MIN_PA_VS_R:
                rhb_woba = self._calculate_woba(vs_rhb)
                rhb_k_rate = (vs_rhb['events'] == 'strikeout').sum() / len(vs_rhb)
                rhb_bb_rate = (vs_rhb['events'] == 'walk').sum() / len(vs_rhb)
                rhb_iso = self._calculate_iso(vs_rhb)
            else:
                rhb_woba = rhb_k_rate = rhb_bb_rate = rhb_iso = None
            
            # Calculate metrics vs LHB
            if len(vs_lhb) >= self.MIN_PA_VS_L:
                lhb_woba = self._calculate_woba(vs_lhb)
                lhb_k_rate = (vs_lhb['events'] == 'strikeout').sum() / len(vs_lhb)
                lhb_bb_rate = (vs_lhb['events'] == 'walk').sum() / len(vs_lhb)
                lhb_iso = self._calculate_iso(vs_lhb)
            else:
                lhb_woba = lhb_k_rate = lhb_bb_rate = lhb_iso = None
            
            # Calculate splits
            if rhb_woba is not None and lhb_woba is not None:
                # For pitchers: lower wOBA allowed is better
                # Same-handed matchup should favor pitcher
                if p_throws == 'R':  # RHP should dominate RHB
                    favorable_woba = rhb_woba  # Lower is better
                    unfavorable_woba = lhb_woba
                    favorable_k = rhb_k_rate  # Higher is better
                    unfavorable_k = lhb_k_rate
                else:  # LHP should dominate LHB
                    favorable_woba = lhb_woba
                    unfavorable_woba = rhb_woba
                    favorable_k = lhb_k_rate
                    unfavorable_k = rhb_k_rate
                
                # Pitcher split: how much WORSE they are vs opposite-handed
                split_woba_allowed = unfavorable_woba - favorable_woba  # Positive = worse vs oppo
                split_k_rate = favorable_k - unfavorable_k  # Positive = more Ks vs same-handed
                
                # Reverse split pitcher: dominates OPPOSITE-handed batters (rare)
                reverse_split = split_woba_allowed < -0.020
                
                results.append({
                    'pitcher_name': pitcher_name,
                    'pitcher_id': pitcher,
                    'p_throws': p_throws,
                    'pa_vs_rhb': len(vs_rhb),
                    'pa_vs_lhb': len(vs_lhb),
                    'woba_vs_rhb': rhb_woba,
                    'woba_vs_lhb': lhb_woba,
                    'k_rate_vs_rhb': rhb_k_rate,
                    'k_rate_vs_lhb': lhb_k_rate,
                    'bb_rate_vs_rhb': rhb_bb_rate,
                    'bb_rate_vs_lhb': lhb_bb_rate,
                    'iso_vs_rhb': rhb_iso,
                    'iso_vs_lhb': lhb_iso,
                    'split_woba_allowed': split_woba_allowed,
                    'split_k_rate': split_k_rate,
                    'reverse_split': reverse_split
                })
        
        df_splits = pd.DataFrame(results)
        df_splits = df_splits.sort_values('split_woba_allowed', ascending=False)
        
        print(f"✓ Calculated splits for {len(df_splits)} pitchers")
        return df_splits
    
    def _calculate_woba(self, df: pd.DataFrame) -> float:
        """Calculate wOBA from Statcast events."""
        # wOBA weights (2023 standard)
        weights = {
            'walk': 0.69,
            'hit_by_pitch': 0.72,
            'single': 0.88,
            'double': 1.24,
            'triple': 1.56,
            'home_run': 1.95
        }
        
        total_weight = 0
        total_pa = 0
        
        for event, weight in weights.items():
            count = (df['events'] == event).sum()
            total_weight += count * weight
            total_pa += count
        
        # Add outs (0 weight)
        outs = df['events'].isin(['strikeout', 'field_out', 'force_out', 
                                   'double_play', 'grounded_into_double_play',
                                   'fielders_choice_out', 'sac_fly']).sum()
        total_pa += outs
        
        return total_weight / total_pa if total_pa > 0 else 0.0
    
    def _calculate_iso(self, df: pd.DataFrame) -> float:
        """Calculate ISO (Isolated Power) = SLG - AVG."""
        hits = df['events'].isin(['single', 'double', 'triple', 'home_run']).sum()
        singles = (df['events'] == 'single').sum()
        doubles = (df['events'] == 'double').sum()
        triples = (df['events'] == 'triple').sum()
        hrs = (df['events'] == 'home_run').sum()
        
        total_bases = singles + (doubles * 2) + (triples * 3) + (hrs * 4)
        at_bats = len(df)
        
        slg = total_bases / at_bats if at_bats > 0 else 0.0
        avg = hits / at_bats if at_bats > 0 else 0.0
        
        return slg - avg
    
    def _calculate_slg(self, df: pd.DataFrame) -> float:
        """Calculate SLG (Slugging Percentage)."""
        singles = (df['events'] == 'single').sum()
        doubles = (df['events'] == 'double').sum()
        triples = (df['events'] == 'triple').sum()
        hrs = (df['events'] == 'home_run').sum()
        
        total_bases = singles + (doubles * 2) + (triples * 3) + (hrs * 4)
        at_bats = len(df)
        
        return total_bases / at_bats if at_bats > 0 else 0.0
    
    def generate_matchup_predictions(
        self, 
        batter_splits: pd.DataFrame,
        pitcher_splits: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate platoon-based matchup predictions.
        
        For each potential batter-pitcher matchup:
        - Calculate expected platoon advantage/disadvantage
        - Adjust base probabilities for hit, HR, strikeout
        - Generate DFS value score
        """
        
        predictions = []
        
        # Sample top 50 batters and top 30 pitchers for demo
        top_batters = batter_splits.head(50)
        top_pitchers = pitcher_splits.head(30)
        
        for _, batter in top_batters.iterrows():
            for _, pitcher in top_pitchers.iterrows():
                # Determine if platoon advantage exists
                if batter['stand'] == 'R' and pitcher['p_throws'] == 'L':
                    platoon_adv = True
                    expected_woba = batter['woba_vs_lhp']
                    expected_k = batter['k_rate_vs_lhp']
                elif batter['stand'] == 'L' and pitcher['p_throws'] == 'R':
                    platoon_adv = True
                    expected_woba = batter['woba_vs_rhp']
                    expected_k = batter['k_rate_vs_rhp']
                elif batter['stand'] == 'R' and pitcher['p_throws'] == 'R':
                    platoon_adv = False
                    expected_woba = batter['woba_vs_rhp']
                    expected_k = batter['k_rate_vs_rhp']
                else:  # LHB vs LHP
                    platoon_adv = False
                    expected_woba = batter['woba_vs_lhp']
                    expected_k = batter['k_rate_vs_lhp']
                
                # Calculate platoon adjustment (%)
                if platoon_adv:
                    # Batter has advantage: boost wOBA, lower K rate
                    platoon_boost = batter['platoon_score'] / 100 * 0.25  # Max 25% boost
                    woba_adjustment = platoon_boost
                    k_adjustment = -platoon_boost * 0.5  # Lower K rate
                else:
                    # Pitcher has advantage: suppress wOBA, raise K rate
                    platoon_penalty = pitcher['split_woba_allowed'] * 2  # Max ~20% penalty
                    woba_adjustment = -platoon_penalty
                    k_adjustment = platoon_penalty * 0.5
                
                adjusted_woba = expected_woba * (1 + woba_adjustment)
                adjusted_k_rate = expected_k * (1 + k_adjustment)
                
                # Convert to DFS value score (0-100)
                # High wOBA + Low K rate + Platoon advantage = premium
                dfs_score = (
                    (adjusted_woba / 0.400) * 50 +  # wOBA component (max 50)
                    ((0.25 - adjusted_k_rate) / 0.25) * 30 +  # K rate component (max 30)
                    (20 if platoon_adv else 0)  # Platoon bonus (20 points)
                )
                dfs_score = min(100, max(0, dfs_score))
                
                predictions.append({
                    'batter': batter['batter_name'],
                    'batter_hand': batter['stand'],
                    'pitcher': pitcher['pitcher_name'],
                    'pitcher_hand': pitcher['p_throws'],
                    'platoon_advantage': platoon_adv,
                    'expected_woba': round(expected_woba, 3),
                    'adjusted_woba': round(adjusted_woba, 3),
                    'expected_k_rate': round(expected_k, 3),
                    'adjusted_k_rate': round(adjusted_k_rate, 3),
                    'woba_adjustment_pct': round(woba_adjustment * 100, 1),
                    'dfs_value_score': round(dfs_score, 1),
                    'recommendation': 'PLAY' if dfs_score >= 70 else 'FADE' if dfs_score < 40 else 'NEUTRAL'
                })
        
        df_predictions = pd.DataFrame(predictions)
        df_predictions = df_predictions.sort_values('dfs_value_score', ascending=False)
        
        if output_path:
            df_predictions.to_csv(output_path, index=False)
            print(f"✓ Saved {len(df_predictions)} predictions to {output_path}")
        
        return df_predictions


def main():
    """Demo the Platoon Splits Analyzer."""
    print("=" * 80)
    print("MLB PREDICTOR v123 - PLATOON SPLITS (L/R MATCHUP ADVANTAGE)")
    print("=" * 80)
    print()
    
    # Initialize
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    
    analyzer = PlatoonAnalyzer(data_dir)
    
    # Load data
    print("📂 Loading Statcast data...")
    if not analyzer.load_statcast_data():
        print("❌ Cannot proceed without data")
        return
    
    print()
    
    # Calculate batter splits
    print("⚾ Calculating batter platoon splits...")
    batter_splits = analyzer.calculate_batter_splits()
    
    if len(batter_splits) > 0:
        output_file = project_dir / "batter_platoon_splits.csv"
        batter_splits.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
        
        # Show top 10 extreme platoon batters
        print("\n🔥 TOP 10 EXTREME PLATOON BATTERS (Benefit Most from Favorable Matchup):")
        print("-" * 100)
        top10 = batter_splits.head(10)
        for idx, row in top10.iterrows():
            advantage = "vs LHP" if row['stand'] == 'R' else "vs RHP"
            print(f"  {row['batter_name']:25} ({row['stand']}HB) | Platoon Score: {row['platoon_score']:5.1f} | "
                  f"wOBA Split: +{row['split_woba']:.3f} | Best {advantage}")
        
        # Show reverse-split batters
        reverse = batter_splits[batter_splits['reverse_split'] == True]
        if len(reverse) > 0:
            print(f"\n🔄 REVERSE-SPLIT BATTERS (Perform BETTER vs same-handed, n={len(reverse)}):")
            print("-" * 100)
            for idx, row in reverse.head(5).iterrows():
                matchup = "vs RHP" if row['stand'] == 'R' else "vs LHP"
                print(f"  {row['batter_name']:25} ({row['stand']}HB) | Better {matchup} | "
                      f"wOBA Split: {row['split_woba']:.3f} (negative = reverse)")
    
    print()
    
    # Calculate pitcher splits
    print("🎯 Calculating pitcher platoon splits...")
    pitcher_splits = analyzer.calculate_pitcher_splits()
    
    if len(pitcher_splits) > 0:
        output_file = project_dir / "pitcher_platoon_splits.csv"
        pitcher_splits.to_csv(output_file, index=False)
        print(f"✓ Saved to {output_file}")
        
        # Show top 10 pitchers most vulnerable to opposite-handed batters
        print("\n⚠️  TOP 10 PITCHERS MOST VULNERABLE TO OPPOSITE-HANDED BATTERS:")
        print("-" * 100)
        top10_vuln = pitcher_splits.head(10)
        for idx, row in top10_vuln.iterrows():
            vulnerable_to = "vs LHB" if row['p_throws'] == 'R' else "vs RHB"
            print(f"  {row['pitcher_name']:25} ({row['p_throws']}HP) | Split: +{row['split_woba_allowed']:.3f} wOBA | "
                  f"Vulnerable {vulnerable_to}")
        
        # Show reverse-split pitchers
        reverse_p = pitcher_splits[pitcher_splits['reverse_split'] == True]
        if len(reverse_p) > 0:
            print(f"\n🔄 REVERSE-SPLIT PITCHERS (Dominate OPPOSITE-handed batters, n={len(reverse_p)}):")
            print("-" * 100)
            for idx, row in reverse_p.head(5).iterrows():
                dominates = "vs LHB" if row['p_throws'] == 'R' else "vs RHB"
                print(f"  {row['pitcher_name']:25} ({row['p_throws']}HP) | Dominates {dominates} | "
                      f"Split: {row['split_woba_allowed']:.3f} (negative = reverse)")
    
    print()
    
    # Generate matchup predictions
    print("🎲 Generating platoon-based matchup predictions...")
    predictions = analyzer.generate_matchup_predictions(
        batter_splits, 
        pitcher_splits,
        output_path=project_dir / "v123_platoon_predictions.csv"
    )
    
    if len(predictions) > 0:
        # Show top 10 premium plays
        print("\n🚀 TOP 10 PREMIUM PLATOON MATCHUPS (Highest DFS Value):")
        print("-" * 100)
        top10_plays = predictions.head(10)
        for idx, row in top10_plays.iterrows():
            advantage_icon = "✅" if row['platoon_advantage'] else "❌"
            print(f"  {advantage_icon} {row['batter']:20} ({row['batter_hand']}) vs {row['pitcher']:20} ({row['pitcher_hand']}) | "
                  f"DFS Score: {row['dfs_value_score']:5.1f} | wOBA: {row['adjusted_woba']:.3f} | "
                  f"K Rate: {row['adjusted_k_rate']:.1%}")
        
        # Show fade candidates
        print("\n⛔ TOP 10 FADE CANDIDATES (Worst Platoon Matchups):")
        print("-" * 100)
        bottom10 = predictions.tail(10)
        for idx, row in bottom10.iterrows():
            advantage_icon = "✅" if row['platoon_advantage'] else "❌"
            print(f"  {advantage_icon} {row['batter']:20} ({row['batter_hand']}) vs {row['pitcher']:20} ({row['pitcher_hand']}) | "
                  f"DFS Score: {row['dfs_value_score']:5.1f} | wOBA: {row['adjusted_woba']:.3f} | "
                  f"K Rate: {row['adjusted_k_rate']:.1%}")
    
    print()
    print("=" * 80)
    print("✅ PLATOON ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("📊 KEY TAKEAWAYS:")
    print("  - Platoon advantage = 70-80 point OPS swing (FanGraphs)")
    print("  - Extreme platoon batters in favorable matchups = PREMIUM DFS plays")
    print("  - Reverse-split players = exploitable contrarian edge")
    print("  - Combine with park factors (v120) + weather (v121) for max edge")
    print()
    print("🎯 DFS STRATEGY:")
    print("  1. Stack RHB vs LHP in hitter-friendly parks")
    print("  2. Target extreme platoon batters in favorable matchups")
    print("  3. Fade batters in unfavorable matchups (same-handed)")
    print("  4. Use reverse-split pitchers as SP value plays")
    print()


if __name__ == "__main__":
    main()
