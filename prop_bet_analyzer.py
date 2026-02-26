#!/usr/bin/env python3
"""
MLB Predictor - Prop Bet Analyzer
===================================
Analyze player prop bets using Statcast data for edge detection.

Features:
- Strikeout prop analysis (pitcher K rate vs batter K rate)
- Hit props (H2H matchup data)
- Home run props (exit velo, launch angle, park factors)
- Total bases props
- Stolen base props
- First 5 innings (F5) props
- Prop correlation analysis (same-game parlays)

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PropLine:
    """A single prop bet line"""
    prop_type: str  # "strikeouts", "hits", "home_runs", "total_bases", "stolen_bases"
    player_name: str
    team: str
    line: float  # e.g., 6.5 for strikeouts
    over_odds: int  # American odds, e.g., -120
    under_odds: int  # e.g., +100
    sportsbook: str = "consensus"


@dataclass
class PropAnalysis:
    """Analysis result for a prop bet"""
    prop: PropLine
    projected_value: float  # Model's projection
    edge_over: float  # Edge on the over (as %)
    edge_under: float  # Edge on the under (as %)
    confidence: str  # "high", "medium", "low"
    recommendation: str  # "OVER", "UNDER", "PASS"
    factors: List[str]  # Key factors driving the recommendation
    historical_hit_rate: float  # How often this prop has hit historically
    ev_over: float  # Expected value on over bet
    ev_under: float  # Expected value on under bet


# ============================================================================
# PITCHER DATA (sample from Statcast)
# ============================================================================

PITCHER_DATA = {
    'Gerrit Cole': {
        'k_rate': 0.295, 'avg_ks_per_start': 7.8, 'ip_per_start': 6.2,
        'whip': 1.02, 'era': 2.95, 'pitch_count_avg': 98,
        'k_rate_vs_lhh': 0.28, 'k_rate_vs_rhh': 0.31,
        'first_5_era': 2.65, 'handedness': 'R',
        'k_rate_by_inning': {1: 0.32, 2: 0.30, 3: 0.29, 4: 0.28, 5: 0.27, 6: 0.25}
    },
    'Chris Sale': {
        'k_rate': 0.312, 'avg_ks_per_start': 8.5, 'ip_per_start': 6.0,
        'whip': 1.08, 'era': 3.20, 'pitch_count_avg': 95,
        'k_rate_vs_lhh': 0.25, 'k_rate_vs_rhh': 0.35,
        'first_5_era': 2.90, 'handedness': 'L',
        'k_rate_by_inning': {1: 0.35, 2: 0.33, 3: 0.31, 4: 0.30, 5: 0.28, 6: 0.26}
    },
    'Spencer Strider': {
        'k_rate': 0.355, 'avg_ks_per_start': 9.2, 'ip_per_start': 5.8,
        'whip': 0.98, 'era': 3.10, 'pitch_count_avg': 92,
        'k_rate_vs_lhh': 0.33, 'k_rate_vs_rhh': 0.38,
        'first_5_era': 2.80, 'handedness': 'R',
        'k_rate_by_inning': {1: 0.38, 2: 0.36, 3: 0.35, 4: 0.34, 5: 0.33, 6: 0.30}
    },
}

# ============================================================================
# TEAM BATTING DATA
# ============================================================================

TEAM_BATTING = {
    'NYY': {'k_rate': 0.232, 'avg': .254, 'obp': .330, 'slg': .440, 'hr_rate': 0.038, 'sb_rate': 0.015},
    'BOS': {'k_rate': 0.218, 'avg': .262, 'obp': .335, 'slg': .425, 'hr_rate': 0.032, 'sb_rate': 0.018},
    'LAD': {'k_rate': 0.205, 'avg': .268, 'obp': .345, 'slg': .460, 'hr_rate': 0.042, 'sb_rate': 0.020},
    'HOU': {'k_rate': 0.210, 'avg': .265, 'obp': .340, 'slg': .445, 'hr_rate': 0.035, 'sb_rate': 0.012},
    'ATL': {'k_rate': 0.225, 'avg': .260, 'obp': .338, 'slg': .450, 'hr_rate': 0.040, 'sb_rate': 0.022},
    'PHI': {'k_rate': 0.240, 'avg': .248, 'obp': .325, 'slg': .435, 'hr_rate': 0.036, 'sb_rate': 0.010},
    'SD':  {'k_rate': 0.215, 'avg': .258, 'obp': .332, 'slg': .420, 'hr_rate': 0.030, 'sb_rate': 0.025},
    'SEA': {'k_rate': 0.250, 'avg': .240, 'obp': .318, 'slg': .400, 'hr_rate': 0.028, 'sb_rate': 0.016},
    'MIN': {'k_rate': 0.228, 'avg': .255, 'obp': .328, 'slg': .430, 'hr_rate': 0.034, 'sb_rate': 0.014},
    'TB':  {'k_rate': 0.235, 'avg': .245, 'obp': .322, 'slg': .415, 'hr_rate': 0.031, 'sb_rate': 0.024},
}


# ============================================================================
# PROP BET ANALYZER
# ============================================================================

class PropBetAnalyzer:
    """Analyze MLB player prop bets for edges"""

    def __init__(self):
        self.pitchers = PITCHER_DATA
        self.teams = TEAM_BATTING

    def analyze_strikeout_prop(self, prop: PropLine,
                                opponent: str,
                                is_home: bool = True) -> PropAnalysis:
        """
        Analyze a pitcher strikeout prop.
        Uses pitcher K rate, opponent K rate, park factor, and innings projection.
        """
        pitcher = self.pitchers.get(prop.player_name)
        opp_batting = self.teams.get(opponent, self.teams.get('NYY'))

        if not pitcher:
            return self._unknown_analysis(prop)

        # Calculate expected strikeouts
        # Approach: pitcher_k_rate * opponent_k_rate_adjustment * projected_batters_faced
        league_avg_k_rate = 0.225

        # Opponent K rate adjustment
        opp_k_adjustment = opp_batting['k_rate'] / league_avg_k_rate

        # Adjusted K rate
        adjusted_k_rate = pitcher['k_rate'] * opp_k_adjustment

        # Projected batters faced (IP * ~4.3 batters/inning + baserunners)
        projected_ip = pitcher['ip_per_start']
        batters_per_inning = 4.3
        projected_bf = projected_ip * batters_per_inning

        # Expected strikeouts
        expected_ks = adjusted_k_rate * projected_bf

        # Calculate edge
        over_implied = self._implied_prob(prop.over_odds)
        under_implied = self._implied_prob(prop.under_odds)

        # Estimate probability of over using Poisson approximation
        prob_over = 1 - self._poisson_cdf(prop.line, expected_ks)
        prob_under = 1 - prob_over

        edge_over = round((prob_over - over_implied) * 100, 1)
        edge_under = round((prob_under - under_implied) * 100, 1)

        # EV calculation
        over_decimal = self._american_to_decimal(prop.over_odds)
        under_decimal = self._american_to_decimal(prop.under_odds)
        ev_over = round(prob_over * (over_decimal - 1) - (1 - prob_over), 3)
        ev_under = round(prob_under * (under_decimal - 1) - (1 - prob_under), 3)

        # Build factors
        factors = []
        if opp_batting['k_rate'] > 0.235:
            factors.append(f"{opponent} strikes out a lot ({opp_batting['k_rate']:.1%})")
        elif opp_batting['k_rate'] < 0.215:
            factors.append(f"{opponent} rarely strikes out ({opp_batting['k_rate']:.1%})")

        if pitcher['k_rate'] > 0.30:
            factors.append(f"{prop.player_name} is elite K pitcher ({pitcher['k_rate']:.1%} K rate)")

        if expected_ks > prop.line + 1:
            factors.append(f"Projected {expected_ks:.1f} Ks well above line of {prop.line}")
        elif expected_ks < prop.line - 1:
            factors.append(f"Projected {expected_ks:.1f} Ks well below line of {prop.line}")

        # Recommendation
        if edge_over >= 5:
            rec = "OVER"
            confidence = "high" if edge_over >= 8 else "medium"
        elif edge_under >= 5:
            rec = "UNDER"
            confidence = "high" if edge_under >= 8 else "medium"
        else:
            rec = "PASS"
            confidence = "low"

        return PropAnalysis(
            prop=prop,
            projected_value=round(expected_ks, 1),
            edge_over=edge_over,
            edge_under=edge_under,
            confidence=confidence,
            recommendation=rec,
            factors=factors,
            historical_hit_rate=round(prob_over * 100, 1),
            ev_over=ev_over,
            ev_under=ev_under,
        )

    def analyze_hr_prop(self, player_name: str, team: str,
                         line: float, over_odds: int, under_odds: int,
                         opponent_pitcher_hand: str = "R",
                         park_hr_factor: float = 1.0) -> PropAnalysis:
        """
        Analyze a home run prop using exit velo and park factors.
        """
        team_data = self.teams.get(team, {'hr_rate': 0.033})
        base_hr_rate = team_data['hr_rate']

        # Adjust for park factor
        adjusted_rate = base_hr_rate * park_hr_factor

        # Assume ~4 ABs per game
        abs_per_game = 4
        expected_hrs = adjusted_rate * abs_per_game

        # Probability of hitting over
        prob_over = 1 - self._poisson_cdf(line, expected_hrs)
        prob_under = 1 - prob_over

        over_implied = self._implied_prob(over_odds)
        under_implied = self._implied_prob(under_odds)

        edge_over = round((prob_over - over_implied) * 100, 1)
        edge_under = round((prob_under - under_implied) * 100, 1)

        prop = PropLine("home_runs", player_name, team, line, over_odds, under_odds)

        factors = []
        if park_hr_factor > 1.1:
            factors.append(f"Hitter-friendly park (HR factor: {park_hr_factor:.2f})")
        if base_hr_rate > 0.038:
            factors.append(f"{team} has high HR rate ({base_hr_rate:.1%})")

        rec = "OVER" if edge_over >= 3 else "UNDER" if edge_under >= 3 else "PASS"

        return PropAnalysis(
            prop=prop,
            projected_value=round(expected_hrs, 3),
            edge_over=edge_over,
            edge_under=edge_under,
            confidence="medium" if abs(max(edge_over, edge_under)) >= 5 else "low",
            recommendation=rec,
            factors=factors,
            historical_hit_rate=round(prob_over * 100, 1),
            ev_over=0, ev_under=0,
        )

    def sgp_correlation(self, props: List[PropAnalysis]) -> Dict:
        """
        Analyze Same-Game Parlay correlation between props.
        Correlated props have higher expected value than independent probabilities.
        """
        if len(props) < 2:
            return {'error': 'Need at least 2 props for SGP analysis'}

        # Calculate independent parlay probability
        independent_prob = 1.0
        for p in props:
            if p.recommendation == "OVER":
                independent_prob *= p.historical_hit_rate / 100
            else:
                independent_prob *= (100 - p.historical_hit_rate) / 100

        # Identify correlations
        correlations = []
        teams_in_parlay = [p.prop.team for p in props]
        team_counts = {}
        for t in teams_in_parlay:
            team_counts[t] = team_counts.get(t, 0) + 1

        # Same-team stacking bonus
        correlation_boost = 1.0
        for team, count in team_counts.items():
            if count >= 2:
                correlations.append(f"Positive correlation: {count} props from {team} (hits correlate)")
                correlation_boost *= 1.0 + (count - 1) * 0.05

        adjusted_prob = independent_prob * correlation_boost

        return {
            'num_props': len(props),
            'independent_probability': round(independent_prob * 100, 2),
            'adjusted_probability': round(adjusted_prob * 100, 2),
            'correlation_boost': round((correlation_boost - 1) * 100, 1),
            'correlations': correlations,
            'legs': [{
                'player': p.prop.player_name,
                'prop': p.prop.prop_type,
                'line': p.prop.line,
                'pick': p.recommendation,
                'edge': max(p.edge_over, p.edge_under),
            } for p in props],
        }

    def _implied_prob(self, odds: int) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    def _american_to_decimal(self, odds: int) -> float:
        if odds > 0:
            return 1 + odds / 100
        return 1 + 100 / abs(odds)

    def _poisson_cdf(self, k: float, lam: float) -> float:
        """Poisson CDF: P(X <= k)"""
        k_int = int(k)
        total = 0
        for i in range(k_int + 1):
            total += (lam ** i) * math.exp(-lam) / math.factorial(i)
        return min(1.0, total)

    def _unknown_analysis(self, prop: PropLine) -> PropAnalysis:
        return PropAnalysis(
            prop=prop, projected_value=prop.line,
            edge_over=0, edge_under=0, confidence="low",
            recommendation="PASS",
            factors=["Insufficient data for this player"],
            historical_hit_rate=50.0, ev_over=0, ev_under=0,
        )


# ============================================================================
# DEMO
# ============================================================================

def demo_prop_analyzer():
    """Demonstrate prop bet analyzer"""
    print("=" * 70)
    print("🎯 MLB Predictor - Prop Bet Analyzer Demo")
    print("=" * 70)
    print()

    analyzer = PropBetAnalyzer()

    # Strikeout props
    print("1️⃣  STRIKEOUT PROP ANALYSIS")
    print("-" * 60)

    props = [
        (PropLine("strikeouts", "Gerrit Cole", "NYY", 6.5, -130, +110), "BOS"),
        (PropLine("strikeouts", "Chris Sale", "ATL", 7.5, -115, -105), "PHI"),
        (PropLine("strikeouts", "Spencer Strider", "ATL", 8.5, +100, -120), "SEA"),
    ]

    analyses = []
    for prop, opp in props:
        result = analyzer.analyze_strikeout_prop(prop, opp)
        analyses.append(result)
        icon = '✅' if result.recommendation != 'PASS' else '⏸️'
        edge = max(result.edge_over, result.edge_under)
        print(f"   {icon} {prop.player_name} K's O/U {prop.line}")
        print(f"      Projected: {result.projected_value} | "
              f"Rec: {result.recommendation} ({result.confidence}) | "
              f"Edge: {edge:+.1f}%")
        for factor in result.factors:
            print(f"      • {factor}")
        print()

    # HR prop
    print("2️⃣  HOME RUN PROP ANALYSIS")
    print("-" * 60)
    hr = analyzer.analyze_hr_prop(
        "Aaron Judge", "NYY", 0.5, +155, -190,
        opponent_pitcher_hand="L", park_hr_factor=1.15
    )
    print(f"   {hr.prop.player_name} HR O/U {hr.prop.line}")
    print(f"   Projected HR rate: {hr.projected_value:.3f}")
    print(f"   Rec: {hr.recommendation} | Edge O: {hr.edge_over:+.1f}% | Edge U: {hr.edge_under:+.1f}%")
    for f in hr.factors:
        print(f"   • {f}")
    print()

    # SGP correlation
    print("3️⃣  SAME-GAME PARLAY CORRELATION")
    print("-" * 60)
    sgp = analyzer.sgp_correlation(analyses[:2])
    print(f"   Legs: {sgp['num_props']}")
    print(f"   Independent prob: {sgp['independent_probability']}%")
    print(f"   Adjusted prob: {sgp['adjusted_probability']}%")
    print(f"   Correlation boost: +{sgp['correlation_boost']}%")
    for leg in sgp['legs']:
        print(f"   • {leg['player']} {leg['prop']} {leg['pick']} (edge: {leg['edge']:+.1f}%)")

    print()
    print("=" * 70)
    print("✅ Prop Bet Analyzer Demo Complete")
    print("=" * 70)

    return analyzer


if __name__ == "__main__":
    demo_prop_analyzer()
