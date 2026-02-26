#!/usr/bin/env python3
"""
Platoon & Splits Analysis Engine
=================================
Comprehensive platoon and situational splits analysis for MLB predictions.

Features:
- L/R pitcher vs L/R batter matchup matrices
- Home/away splits with statistical significance testing
- Day/night performance differentials
- Monthly performance trends (hot/cold streaks)
- First half vs second half splits
- Situational hitting (RISP, 2 outs, close & late)
- Platoon advantage quantification
- Lineup construction optimizer based on platoon advantages
- Historical split data with regression to mean adjustments
- JSON-ready predictions with confidence intervals

Author: MLB Predictor System
Version: 1.0.0
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# Enums & Constants
# ---------------------------------------------------------------------------

class HandSide(Enum):
    LEFT = "L"
    RIGHT = "R"
    SWITCH = "S"

class GameTime(Enum):
    DAY = "day"
    NIGHT = "night"

class HalfSeason(Enum):
    FIRST = "first_half"
    SECOND = "second_half"

class Situation(Enum):
    RISP = "risp"
    TWO_OUTS = "two_outs"
    CLOSE_AND_LATE = "close_and_late"
    BASES_LOADED = "bases_loaded"
    RUNNER_ON_THIRD_LESS_THAN_TWO_OUTS = "runner_3rd_lt2"
    LEADING = "leading"
    TRAILING = "trailing"
    TIED = "tied"

# Regression constants (league-average priors)
LEAGUE_AVG_BA = 0.248
LEAGUE_AVG_OBP = 0.317
LEAGUE_AVG_SLG = 0.407
LEAGUE_AVG_WOBA = 0.315
LEAGUE_AVG_ISO = 0.159

# Plate appearances needed before trusting split data
PA_REGRESSION_THRESHOLD = {
    "batting_avg": 910,
    "obp": 460,
    "slg": 320,
    "woba": 340,
    "iso": 160,
    "k_rate": 150,
    "bb_rate": 120,
}

# Platoon advantage baseline (historical L/R splits)
PLATOON_ADVANTAGE_WOBA = {
    ("L", "R"): 0.012,   # LHB vs RHP slight advantage
    ("L", "L"): -0.025,  # LHB vs LHP disadvantage
    ("R", "L"): 0.018,   # RHB vs LHP advantage (platoon)
    ("R", "R"): -0.008,  # RHB vs RHP slight disadvantage
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    """Statistical line for a particular split."""
    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    rbi: int = 0
    walks: int = 0
    strikeouts: int = 0
    hit_by_pitch: int = 0
    sac_flies: int = 0
    stolen_bases: int = 0
    caught_stealing: int = 0
    gidp: int = 0
    batting_avg: float = 0.0
    obp: float = 0.0
    slg: float = 0.0
    ops: float = 0.0
    woba: float = 0.0
    iso: float = 0.0
    babip: float = 0.0
    k_rate: float = 0.0
    bb_rate: float = 0.0
    sample_size_grade: str = "F"  # A/B/C/D/F

    def compute_rates(self) -> None:
        """Calculate rate stats from counting stats."""
        if self.at_bats > 0:
            self.batting_avg = self.hits / self.at_bats
            tb = self.hits + self.doubles + 2 * self.triples + 3 * self.home_runs
            self.slg = tb / self.at_bats
            self.iso = self.slg - self.batting_avg
            # BABIP
            denominator = self.at_bats - self.strikeouts - self.home_runs + self.sac_flies
            if denominator > 0:
                self.babip = (self.hits - self.home_runs) / denominator

        if self.plate_appearances > 0:
            obp_num = self.hits + self.walks + self.hit_by_pitch
            obp_den = self.at_bats + self.walks + self.hit_by_pitch + self.sac_flies
            self.obp = obp_num / obp_den if obp_den > 0 else 0.0
            self.ops = self.obp + self.slg
            self.k_rate = self.strikeouts / self.plate_appearances
            self.bb_rate = self.walks / self.plate_appearances

            # wOBA (simplified weights — 2023 FanGraphs linear weights)
            w_bb = 0.690
            w_hbp = 0.722
            w_1b = 0.878
            w_2b = 1.242
            w_3b = 1.568
            w_hr = 2.007
            singles = self.hits - self.doubles - self.triples - self.home_runs
            woba_num = (w_bb * self.walks + w_hbp * self.hit_by_pitch +
                        w_1b * singles + w_2b * self.doubles +
                        w_3b * self.triples + w_hr * self.home_runs)
            woba_den = self.at_bats + self.walks + self.sac_flies + self.hit_by_pitch
            self.woba = woba_num / woba_den if woba_den > 0 else 0.0

        # Sample size grading
        pa = self.plate_appearances
        if pa >= 500:
            self.sample_size_grade = "A"
        elif pa >= 250:
            self.sample_size_grade = "B"
        elif pa >= 100:
            self.sample_size_grade = "C"
        elif pa >= 50:
            self.sample_size_grade = "D"
        else:
            self.sample_size_grade = "F"


@dataclass
class PlatoonMatchup:
    """Stores matchup data for a specific batter-hand vs pitcher-hand combo."""
    batter_hand: str
    pitcher_hand: str
    stats: SplitStats = field(default_factory=SplitStats)
    platoon_advantage_woba: float = 0.0
    regressed_woba: float = 0.0
    confidence: float = 0.0


@dataclass
class PlayerSplits:
    """Complete split data for a single player."""
    player_id: str
    player_name: str
    team: str
    bats: str  # L, R, S
    season: int = 2025

    # Platoon splits
    vs_lhp: SplitStats = field(default_factory=SplitStats)
    vs_rhp: SplitStats = field(default_factory=SplitStats)

    # Venue splits
    home: SplitStats = field(default_factory=SplitStats)
    away: SplitStats = field(default_factory=SplitStats)

    # Time of day
    day: SplitStats = field(default_factory=SplitStats)
    night: SplitStats = field(default_factory=SplitStats)

    # Monthly splits
    monthly: Dict[int, SplitStats] = field(default_factory=dict)

    # Half-season
    first_half: SplitStats = field(default_factory=SplitStats)
    second_half: SplitStats = field(default_factory=SplitStats)

    # Situational
    risp: SplitStats = field(default_factory=SplitStats)
    two_outs: SplitStats = field(default_factory=SplitStats)
    close_and_late: SplitStats = field(default_factory=SplitStats)
    bases_loaded: SplitStats = field(default_factory=SplitStats)


@dataclass
class ConfidenceInterval:
    """A prediction with confidence bounds."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float = 0.95
    sample_size: int = 0
    reliability_score: float = 0.0


@dataclass
class PlatoonPrediction:
    """Full prediction output for a matchup."""
    batter_id: str
    batter_name: str
    pitcher_id: str
    pitcher_name: str
    pitcher_hand: str
    predicted_woba: ConfidenceInterval = None
    predicted_ba: ConfidenceInterval = None
    predicted_ops: ConfidenceInterval = None
    platoon_edge: float = 0.0
    situational_adjustments: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------

class StatisticalTests:
    """Statistical significance tests for split comparisons."""

    @staticmethod
    def wilson_confidence_interval(successes: int, trials: int,
                                    z: float = 1.96) -> Tuple[float, float]:
        """Wilson score interval for proportions (better for small samples)."""
        if trials == 0:
            return (0.0, 0.0)
        p_hat = successes / trials
        denominator = 1 + z ** 2 / trials
        center = (p_hat + z ** 2 / (2 * trials)) / denominator
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * trials)) / trials) / denominator
        return (max(0.0, center - spread), min(1.0, center + spread))

    @staticmethod
    def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> Tuple[float, float]:
        """
        Two-proportion z-test.
        Returns (z_statistic, p_value).
        Tests H0: p1 == p2.
        """
        if n1 == 0 or n2 == 0:
            return (0.0, 1.0)
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
        if p_pool == 0 or p_pool == 1:
            return (0.0, 1.0)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            return (0.0, 1.0)
        z = (p1 - p2) / se
        # Approximate two-tailed p-value using normal CDF
        p_value = 2 * (1 - StatisticalTests._normal_cdf(abs(z)))
        return (z, p_value)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximation of the standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def bayesian_regression_to_mean(observed: float, sample_pa: int,
                                     league_avg: float,
                                     regression_pa: int) -> float:
        """
        Regress observed stat toward league average.
        Uses a simple weighted average: more PA → more weight on observed.
        """
        weight = sample_pa / (sample_pa + regression_pa)
        return weight * observed + (1 - weight) * league_avg

    @staticmethod
    def effect_size_cohens_d(mean1: float, std1: float, n1: int,
                              mean2: float, std2: float, n2: int) -> float:
        """Cohen's d effect size for two independent samples."""
        pooled_std = math.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (mean1 - mean2) / pooled_std


# ---------------------------------------------------------------------------
# Platoon Splits Engine
# ---------------------------------------------------------------------------

class PlatoonSplitsEngine:
    """
    Main engine for platoon and situational splits analysis.

    Processes raw batting data, computes regressed splits,
    quantifies platoon advantages, and outputs JSON-ready predictions.
    """

    def __init__(self):
        self.players: Dict[str, PlayerSplits] = {}
        self.matchup_cache: Dict[str, PlatoonMatchup] = {}
        self.stats_tests = StatisticalTests()
        self.league_averages: Dict[str, float] = {
            "ba": LEAGUE_AVG_BA,
            "obp": LEAGUE_AVG_OBP,
            "slg": LEAGUE_AVG_SLG,
            "woba": LEAGUE_AVG_WOBA,
            "iso": LEAGUE_AVG_ISO,
        }

    # ---- Data Ingestion ----

    def load_player(self, player_data: Dict[str, Any]) -> PlayerSplits:
        """Load a player's split data from a dictionary."""
        player_id = player_data["player_id"]
        player = PlayerSplits(
            player_id=player_id,
            player_name=player_data.get("name", "Unknown"),
            team=player_data.get("team", "UNK"),
            bats=player_data.get("bats", "R"),
            season=player_data.get("season", 2025),
        )

        # Load each split category
        for split_key in ["vs_lhp", "vs_rhp", "home", "away", "day", "night",
                          "first_half", "second_half", "risp", "two_outs",
                          "close_and_late", "bases_loaded"]:
            if split_key in player_data:
                split_stats = self._parse_split_stats(player_data[split_key])
                setattr(player, split_key, split_stats)

        # Monthly data
        if "monthly" in player_data:
            for month_str, month_data in player_data["monthly"].items():
                player.monthly[int(month_str)] = self._parse_split_stats(month_data)

        self.players[player_id] = player
        return player

    def _parse_split_stats(self, raw: Dict[str, Any]) -> SplitStats:
        """Parse raw stat dictionary into SplitStats object."""
        stats = SplitStats(
            plate_appearances=raw.get("pa", 0),
            at_bats=raw.get("ab", 0),
            hits=raw.get("h", 0),
            doubles=raw.get("2b", 0),
            triples=raw.get("3b", 0),
            home_runs=raw.get("hr", 0),
            rbi=raw.get("rbi", 0),
            walks=raw.get("bb", 0),
            strikeouts=raw.get("so", 0),
            hit_by_pitch=raw.get("hbp", 0),
            sac_flies=raw.get("sf", 0),
            stolen_bases=raw.get("sb", 0),
            caught_stealing=raw.get("cs", 0),
            gidp=raw.get("gidp", 0),
        )
        stats.compute_rates()
        return stats

    # ---- Platoon Analysis ----

    def compute_platoon_matchup(self, batter_id: str,
                                 pitcher_hand: str) -> PlatoonMatchup:
        """
        Compute regressed platoon matchup for batter vs pitcher handedness.
        Returns matchup with regressed wOBA and confidence interval.
        """
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        # Determine the relevant split
        if pitcher_hand == "L":
            raw_stats = player.vs_lhp
        else:
            raw_stats = player.vs_rhp

        # Determine effective batter hand (switch hitters)
        effective_bat_hand = player.bats
        if player.bats == "S":
            effective_bat_hand = "R" if pitcher_hand == "L" else "L"

        # Get platoon advantage
        key = (effective_bat_hand, pitcher_hand)
        platoon_adv = PLATOON_ADVANTAGE_WOBA.get(key, 0.0)

        # Regress wOBA to mean
        regressed_woba = self.stats_tests.bayesian_regression_to_mean(
            observed=raw_stats.woba,
            sample_pa=raw_stats.plate_appearances,
            league_avg=LEAGUE_AVG_WOBA,
            regression_pa=PA_REGRESSION_THRESHOLD["woba"],
        )

        # Confidence based on sample size
        pa = raw_stats.plate_appearances
        confidence = min(1.0, pa / PA_REGRESSION_THRESHOLD["woba"])

        matchup = PlatoonMatchup(
            batter_hand=effective_bat_hand,
            pitcher_hand=pitcher_hand,
            stats=raw_stats,
            platoon_advantage_woba=platoon_adv,
            regressed_woba=regressed_woba,
            confidence=confidence,
        )

        cache_key = f"{batter_id}_{pitcher_hand}"
        self.matchup_cache[cache_key] = matchup
        return matchup

    def build_platoon_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Build a full L/R matchup matrix across all loaded batters.
        Returns nested dict: {batter_id: {pitcher_hand: regressed_woba}}.
        """
        matrix = {}
        for pid, player in self.players.items():
            matrix[pid] = {}
            for p_hand in ["L", "R"]:
                matchup = self.compute_platoon_matchup(pid, p_hand)
                matrix[pid][p_hand] = {
                    "regressed_woba": round(matchup.regressed_woba, 3),
                    "raw_woba": round(matchup.stats.woba, 3),
                    "platoon_edge": round(matchup.platoon_advantage_woba, 3),
                    "confidence": round(matchup.confidence, 3),
                    "pa": matchup.stats.plate_appearances,
                    "grade": matchup.stats.sample_size_grade,
                }
        return matrix

    # ---- Home/Away Splits with Significance ----

    def analyze_home_away(self, batter_id: str) -> Dict[str, Any]:
        """
        Compare home vs away splits with statistical significance testing.
        """
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        home = player.home
        away = player.away

        # Two-proportion z-test on batting average
        z_stat, p_value = self.stats_tests.two_proportion_z_test(
            home.batting_avg, home.at_bats,
            away.batting_avg, away.at_bats,
        )

        # Confidence intervals for wOBA
        home_woba_ci = self.stats_tests.wilson_confidence_interval(
            int(home.woba * home.plate_appearances),
            home.plate_appearances,
        )
        away_woba_ci = self.stats_tests.wilson_confidence_interval(
            int(away.woba * away.plate_appearances),
            away.plate_appearances,
        )

        significant = p_value < 0.05

        return {
            "player_id": batter_id,
            "player_name": player.player_name,
            "home": {
                "ba": round(home.batting_avg, 3),
                "obp": round(home.obp, 3),
                "slg": round(home.slg, 3),
                "ops": round(home.ops, 3),
                "woba": round(home.woba, 3),
                "pa": home.plate_appearances,
                "woba_ci": [round(x, 3) for x in home_woba_ci],
            },
            "away": {
                "ba": round(away.batting_avg, 3),
                "obp": round(away.obp, 3),
                "slg": round(away.slg, 3),
                "ops": round(away.ops, 3),
                "woba": round(away.woba, 3),
                "pa": away.plate_appearances,
                "woba_ci": [round(x, 3) for x in away_woba_ci],
            },
            "difference": {
                "ba_diff": round(home.batting_avg - away.batting_avg, 3),
                "ops_diff": round(home.ops - away.ops, 3),
                "woba_diff": round(home.woba - away.woba, 3),
            },
            "significance": {
                "z_statistic": round(z_stat, 3),
                "p_value": round(p_value, 4),
                "significant_at_05": significant,
                "interpretation": (
                    "Statistically significant home/away split"
                    if significant else
                    "No significant home/away difference"
                ),
            },
        }

    # ---- Day/Night Differentials ----

    def analyze_day_night(self, batter_id: str) -> Dict[str, Any]:
        """Analyze day vs night game performance differentials."""
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        day_s = player.day
        night_s = player.night

        z_stat, p_val = self.stats_tests.two_proportion_z_test(
            day_s.batting_avg, day_s.at_bats,
            night_s.batting_avg, night_s.at_bats,
        )

        return {
            "player_id": batter_id,
            "player_name": player.player_name,
            "day": {
                "ba": round(day_s.batting_avg, 3),
                "ops": round(day_s.ops, 3),
                "woba": round(day_s.woba, 3),
                "hr": day_s.home_runs,
                "pa": day_s.plate_appearances,
                "k_rate": round(day_s.k_rate, 3),
            },
            "night": {
                "ba": round(night_s.batting_avg, 3),
                "ops": round(night_s.ops, 3),
                "woba": round(night_s.woba, 3),
                "hr": night_s.home_runs,
                "pa": night_s.plate_appearances,
                "k_rate": round(night_s.k_rate, 3),
            },
            "differential": {
                "ba_diff": round(day_s.batting_avg - night_s.batting_avg, 3),
                "ops_diff": round(day_s.ops - night_s.ops, 3),
                "significant": p_val < 0.05,
                "p_value": round(p_val, 4),
            },
            "recommendation": (
                "Day game advantage" if day_s.woba > night_s.woba + 0.015
                else "Night game advantage" if night_s.woba > day_s.woba + 0.015
                else "No meaningful day/night split"
            ),
        }

    # ---- Monthly Performance Trends ----

    def analyze_monthly_trends(self, batter_id: str) -> Dict[str, Any]:
        """Detect hot/cold streaks by month with trend analysis."""
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        month_names = {3: "March", 4: "April", 5: "May", 6: "June",
                       7: "July", 8: "August", 9: "September", 10: "October"}

        monthly_data = {}
        woba_values = []

        for month_num in sorted(player.monthly.keys()):
            ms = player.monthly[month_num]
            regressed = self.stats_tests.bayesian_regression_to_mean(
                ms.woba, ms.plate_appearances, LEAGUE_AVG_WOBA,
                PA_REGRESSION_THRESHOLD["woba"]
            )
            woba_values.append(ms.woba)
            monthly_data[month_names.get(month_num, f"Month_{month_num}")] = {
                "raw_woba": round(ms.woba, 3),
                "regressed_woba": round(regressed, 3),
                "ba": round(ms.batting_avg, 3),
                "ops": round(ms.ops, 3),
                "hr": ms.home_runs,
                "pa": ms.plate_appearances,
                "grade": ms.sample_size_grade,
            }

        # Detect hot/cold months
        if woba_values:
            avg_woba = statistics.mean(woba_values)
            std_woba = statistics.stdev(woba_values) if len(woba_values) > 1 else 0.0
            hot_threshold = avg_woba + std_woba
            cold_threshold = avg_woba - std_woba
        else:
            avg_woba = LEAGUE_AVG_WOBA
            hot_threshold = cold_threshold = avg_woba

        streaks = []
        for month_num in sorted(player.monthly.keys()):
            ms = player.monthly[month_num]
            if ms.woba >= hot_threshold and ms.plate_appearances >= 50:
                streaks.append({
                    "month": month_names.get(month_num, str(month_num)),
                    "type": "HOT",
                    "woba": round(ms.woba, 3),
                })
            elif ms.woba <= cold_threshold and ms.plate_appearances >= 50:
                streaks.append({
                    "month": month_names.get(month_num, str(month_num)),
                    "type": "COLD",
                    "woba": round(ms.woba, 3),
                })

        return {
            "player_id": batter_id,
            "player_name": player.player_name,
            "monthly_splits": monthly_data,
            "average_woba": round(avg_woba, 3),
            "volatility": round(std_woba, 3) if woba_values else 0.0,
            "streaks": streaks,
            "trend": self._compute_trend(woba_values),
        }

    @staticmethod
    def _compute_trend(values: List[float]) -> str:
        """Simple linear trend detection."""
        if len(values) < 3:
            return "insufficient_data"
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0
        if slope > 0.005:
            return "improving"
        elif slope < -0.005:
            return "declining"
        return "stable"

    # ---- First Half vs Second Half ----

    def analyze_half_season(self, batter_id: str) -> Dict[str, Any]:
        """Compare first-half vs second-half performance."""
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        h1 = player.first_half
        h2 = player.second_half

        z_stat, p_val = self.stats_tests.two_proportion_z_test(
            h1.batting_avg, h1.at_bats,
            h2.batting_avg, h2.at_bats,
        )

        return {
            "player_id": batter_id,
            "player_name": player.player_name,
            "first_half": {
                "ba": round(h1.batting_avg, 3),
                "ops": round(h1.ops, 3),
                "woba": round(h1.woba, 3),
                "hr": h1.home_runs,
                "pa": h1.plate_appearances,
            },
            "second_half": {
                "ba": round(h2.batting_avg, 3),
                "ops": round(h2.ops, 3),
                "woba": round(h2.woba, 3),
                "hr": h2.home_runs,
                "pa": h2.plate_appearances,
            },
            "fatigue_indicator": round(h2.woba - h1.woba, 3),
            "significant": p_val < 0.05,
            "p_value": round(p_val, 4),
            "pattern": (
                "second_half_surge" if h2.woba > h1.woba + 0.020
                else "second_half_fade" if h1.woba > h2.woba + 0.020
                else "consistent"
            ),
        }

    # ---- Situational Hitting ----

    def analyze_situational(self, batter_id: str) -> Dict[str, Any]:
        """Comprehensive situational hitting analysis."""
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        overall_woba = (player.vs_lhp.woba * player.vs_lhp.plate_appearances +
                        player.vs_rhp.woba * player.vs_rhp.plate_appearances)
        total_pa = player.vs_lhp.plate_appearances + player.vs_rhp.plate_appearances
        overall_woba = overall_woba / total_pa if total_pa > 0 else LEAGUE_AVG_WOBA

        situations = {}
        for sit_name, split in [("risp", player.risp), ("two_outs", player.two_outs),
                                 ("close_and_late", player.close_and_late),
                                 ("bases_loaded", player.bases_loaded)]:
            regressed = self.stats_tests.bayesian_regression_to_mean(
                split.woba, split.plate_appearances, overall_woba,
                PA_REGRESSION_THRESHOLD["woba"]
            )
            ci = self.stats_tests.wilson_confidence_interval(
                int(split.woba * split.plate_appearances),
                split.plate_appearances
            )
            situations[sit_name] = {
                "raw_woba": round(split.woba, 3),
                "regressed_woba": round(regressed, 3),
                "ba": round(split.batting_avg, 3),
                "ops": round(split.ops, 3),
                "k_rate": round(split.k_rate, 3),
                "pa": split.plate_appearances,
                "grade": split.sample_size_grade,
                "ci_95": [round(x, 3) for x in ci],
                "clutch_delta": round(split.woba - overall_woba, 3),
            }

        # Classify as clutch/choke
        risp_delta = situations["risp"]["clutch_delta"]
        late_delta = situations["close_and_late"]["clutch_delta"]
        avg_clutch = (risp_delta + late_delta) / 2

        return {
            "player_id": batter_id,
            "player_name": player.player_name,
            "overall_woba": round(overall_woba, 3),
            "situations": situations,
            "clutch_score": round(avg_clutch, 3),
            "clutch_label": (
                "elite_clutch" if avg_clutch > 0.030
                else "clutch" if avg_clutch > 0.010
                else "neutral" if avg_clutch > -0.010
                else "anti_clutch" if avg_clutch > -0.030
                else "choke_artist"
            ),
        }

    # ---- Lineup Optimization Based on Platoon ----

    def optimize_lineup_platoon(self, roster: List[str],
                                 pitcher_hand: str,
                                 lineup_size: int = 9) -> List[Dict[str, Any]]:
        """
        Optimize lineup order based on platoon advantages against a
        pitcher of the given handedness.

        Uses a simplified model:
        - Top of order: high OBP + wOBA
        - 3-5: power (high ISO + wOBA)
        - 6-9: best remaining
        """
        candidates = []
        for pid in roster:
            if pid not in self.players:
                continue
            matchup = self.compute_platoon_matchup(pid, pitcher_hand)
            player = self.players[pid]
            split = player.vs_lhp if pitcher_hand == "L" else player.vs_rhp

            regressed_obp = self.stats_tests.bayesian_regression_to_mean(
                split.obp, split.plate_appearances, LEAGUE_AVG_OBP,
                PA_REGRESSION_THRESHOLD["obp"]
            )
            regressed_iso = self.stats_tests.bayesian_regression_to_mean(
                split.iso, split.plate_appearances, LEAGUE_AVG_ISO,
                PA_REGRESSION_THRESHOLD["iso"]
            )

            candidates.append({
                "player_id": pid,
                "player_name": player.player_name,
                "bats": player.bats,
                "woba": matchup.regressed_woba,
                "obp": regressed_obp,
                "iso": regressed_iso,
                "platoon_edge": matchup.platoon_advantage_woba,
                "confidence": matchup.confidence,
            })

        # Sort by composite score for lineup
        # Weight: 40% wOBA, 30% OBP, 30% ISO
        for c in candidates:
            c["composite"] = 0.4 * c["woba"] + 0.3 * c["obp"] + 0.3 * c["iso"]

        candidates.sort(key=lambda x: x["composite"], reverse=True)
        candidates = candidates[:lineup_size]

        # Assign batting order positions
        # 1-2: highest OBP, 3: highest wOBA, 4: highest ISO, rest fill
        obp_sorted = sorted(candidates, key=lambda x: x["obp"], reverse=True)
        woba_sorted = sorted(candidates, key=lambda x: x["woba"], reverse=True)
        iso_sorted = sorted(candidates, key=lambda x: x["iso"], reverse=True)

        lineup = [None] * min(lineup_size, len(candidates))
        used = set()

        # Leadoff: best OBP
        for c in obp_sorted:
            if c["player_id"] not in used:
                lineup[0] = c
                used.add(c["player_id"])
                break

        # 2-hole: second-best OBP with some power
        for c in obp_sorted:
            if c["player_id"] not in used:
                lineup[1] = c
                used.add(c["player_id"])
                break

        # 3-hole: best overall wOBA
        for c in woba_sorted:
            if c["player_id"] not in used:
                lineup[2] = c
                used.add(c["player_id"])
                break

        # Cleanup: best power
        for c in iso_sorted:
            if c["player_id"] not in used:
                lineup[3] = c
                used.add(c["player_id"])
                break

        # Fill rest by composite
        remaining = [c for c in candidates if c["player_id"] not in used]
        remaining.sort(key=lambda x: x["composite"], reverse=True)
        idx = 4
        for c in remaining:
            if idx >= len(lineup):
                break
            lineup[idx] = c
            used.add(c["player_id"])
            idx += 1

        # Format output
        result = []
        for i, batter in enumerate(lineup):
            if batter is None:
                continue
            result.append({
                "batting_order": i + 1,
                "player_id": batter["player_id"],
                "player_name": batter["player_name"],
                "bats": batter["bats"],
                "vs_pitcher_hand": pitcher_hand,
                "regressed_woba": round(batter["woba"], 3),
                "regressed_obp": round(batter["obp"], 3),
                "regressed_iso": round(batter["iso"], 3),
                "platoon_edge": round(batter["platoon_edge"], 3),
                "composite_score": round(batter["composite"], 3),
            })

        return result

    # ---- Full Prediction with Confidence Intervals ----

    def predict_matchup(self, batter_id: str, pitcher_id: str,
                         pitcher_hand: str, is_home: bool = True,
                         game_time: str = "night",
                         situation: str = "normal") -> Dict[str, Any]:
        """
        Generate a full prediction for a batter-pitcher matchup.
        Combines platoon, venue, time, and situational adjustments.
        """
        player = self.players.get(batter_id)
        if not player:
            raise ValueError(f"Player {batter_id} not found")

        # Base matchup
        matchup = self.compute_platoon_matchup(batter_id, pitcher_hand)
        base_woba = matchup.regressed_woba

        # Venue adjustment
        home_away = self.analyze_home_away(batter_id) if (
            player.home.plate_appearances > 0 and player.away.plate_appearances > 0
        ) else None
        venue_adj = 0.0
        if home_away:
            diff = home_away["difference"]["woba_diff"]
            if is_home and diff > 0:
                venue_adj = diff * 0.3  # Partial credit for home advantage
            elif not is_home and diff < 0:
                venue_adj = diff * 0.3

        # Time adjustment
        time_adj = 0.0
        if player.day.plate_appearances > 30 and player.night.plate_appearances > 30:
            if game_time == "day":
                time_adj = (player.day.woba - player.night.woba) * 0.2
            else:
                time_adj = (player.night.woba - player.day.woba) * 0.2

        # Situational adjustment
        sit_adj = 0.0
        if situation == "risp" and player.risp.plate_appearances > 30:
            overall_w = base_woba
            sit_adj = (player.risp.woba - overall_w) * 0.25
        elif situation == "close_and_late" and player.close_and_late.plate_appearances > 30:
            overall_w = base_woba
            sit_adj = (player.close_and_late.woba - overall_w) * 0.25

        # Final prediction
        final_woba = base_woba + venue_adj + time_adj + sit_adj
        final_woba = max(0.150, min(0.500, final_woba))  # Clamp

        # Confidence interval
        pa = matchup.stats.plate_appearances
        se = math.sqrt(final_woba * (1 - final_woba) / max(pa, 1)) if pa > 0 else 0.1
        ci_lower = max(0.0, final_woba - 1.96 * se)
        ci_upper = min(1.0, final_woba + 1.96 * se)

        reliability = min(1.0, pa / 300) * matchup.confidence

        return {
            "batter_id": batter_id,
            "batter_name": player.player_name,
            "pitcher_id": pitcher_id,
            "pitcher_hand": pitcher_hand,
            "is_home": is_home,
            "game_time": game_time,
            "situation": situation,
            "prediction": {
                "woba": round(final_woba, 3),
                "ci_lower": round(ci_lower, 3),
                "ci_upper": round(ci_upper, 3),
                "confidence_level": 0.95,
            },
            "adjustments": {
                "base_woba": round(base_woba, 3),
                "venue_adjustment": round(venue_adj, 4),
                "time_adjustment": round(time_adj, 4),
                "situational_adjustment": round(sit_adj, 4),
                "platoon_edge": round(matchup.platoon_advantage_woba, 3),
            },
            "reliability": {
                "score": round(reliability, 3),
                "grade": (
                    "A" if reliability > 0.8 else
                    "B" if reliability > 0.6 else
                    "C" if reliability > 0.4 else
                    "D" if reliability > 0.2 else "F"
                ),
                "sample_size": pa,
            },
        }

    # ---- Batch / Export ----

    def export_all_splits(self) -> str:
        """Export all player splits as JSON."""
        output = {}
        for pid, player in self.players.items():
            output[pid] = {
                "name": player.player_name,
                "team": player.team,
                "bats": player.bats,
                "platoon_matrix": self.build_platoon_matrix().get(pid, {}),
            }
            if player.home.plate_appearances > 0:
                output[pid]["home_away"] = self.analyze_home_away(pid)
            if player.day.plate_appearances > 0:
                output[pid]["day_night"] = self.analyze_day_night(pid)
            if player.monthly:
                output[pid]["monthly"] = self.analyze_monthly_trends(pid)
            if player.risp.plate_appearances > 0:
                output[pid]["situational"] = self.analyze_situational(pid)
        return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------

def _generate_demo_split(pa: int, ba_base: float = 0.260,
                          power_factor: float = 1.0) -> Dict[str, Any]:
    """Generate realistic-looking split stats for demo purposes."""
    ab = int(pa * 0.88)
    hits = int(ab * ba_base * random.uniform(0.9, 1.1))
    doubles = int(hits * 0.20 * random.uniform(0.8, 1.2))
    triples = int(hits * 0.02 * random.uniform(0.5, 2.0))
    hr = int(hits * 0.12 * power_factor * random.uniform(0.7, 1.3))
    return {
        "pa": pa, "ab": ab, "h": hits,
        "2b": doubles, "3b": triples, "hr": hr,
        "rbi": int(hr * 3.2 + hits * 0.3),
        "bb": int(pa * 0.08 * random.uniform(0.8, 1.2)),
        "so": int(pa * 0.22 * random.uniform(0.8, 1.2)),
        "hbp": int(pa * 0.01), "sf": int(pa * 0.01),
        "sb": random.randint(0, 15), "cs": random.randint(0, 5),
        "gidp": random.randint(0, 12),
    }


def demo():
    """Run a comprehensive demo of the Platoon Splits Engine."""
    print("=" * 70)
    print("  PLATOON & SPLITS ANALYSIS ENGINE — DEMO")
    print("=" * 70)

    random.seed(42)
    engine = PlatoonSplitsEngine()

    # Create demo players
    demo_players = [
        {"player_id": "660271", "name": "Shohei Ohtani", "team": "LAD", "bats": "L"},
        {"player_id": "665742", "name": "Juan Soto", "team": "NYM", "bats": "L"},
        {"player_id": "592450", "name": "Aaron Judge", "team": "NYY", "bats": "R"},
        {"player_id": "666969", "name": "Elly De La Cruz", "team": "CIN", "bats": "S"},
        {"player_id": "665487", "name": "Mookie Betts", "team": "LAD", "bats": "R"},
        {"player_id": "671096", "name": "Corbin Carroll", "team": "ARI", "bats": "L"},
        {"player_id": "668939", "name": "Bobby Witt Jr", "team": "KC", "bats": "R"},
        {"player_id": "660670", "name": "Ronald Acuna Jr", "team": "ATL", "bats": "R"},
        {"player_id": "608369", "name": "Freddie Freeman", "team": "LAD", "bats": "L"},
    ]

    for p in demo_players:
        # Platoon splits — lefties do worse vs LHP
        lhp_adj = 0.92 if p["bats"] == "L" else 1.08
        rhp_adj = 1.08 if p["bats"] == "L" else 0.96

        data = {
            **p,
            "season": 2025,
            "vs_lhp": _generate_demo_split(180, 0.255 * lhp_adj, 1.0),
            "vs_rhp": _generate_demo_split(420, 0.270 * rhp_adj, 1.1),
            "home": _generate_demo_split(310, 0.275, 1.05),
            "away": _generate_demo_split(290, 0.250, 0.95),
            "day": _generate_demo_split(160, 0.265, 1.0),
            "night": _generate_demo_split(440, 0.260, 1.0),
            "first_half": _generate_demo_split(320, 0.265, 1.0),
            "second_half": _generate_demo_split(280, 0.255, 0.95),
            "risp": _generate_demo_split(150, 0.280, 1.15),
            "two_outs": _generate_demo_split(180, 0.240, 0.90),
            "close_and_late": _generate_demo_split(120, 0.255, 1.0),
            "bases_loaded": _generate_demo_split(35, 0.290, 1.2),
            "monthly": {
                str(m): _generate_demo_split(
                    random.randint(80, 120),
                    0.260 * random.uniform(0.85, 1.15),
                    random.uniform(0.8, 1.2),
                ) for m in range(4, 10)
            },
        }
        engine.load_player(data)

    # 1) Platoon Matrix
    print("\n" + "=" * 50)
    print("  1. PLATOON MATCHUP MATRIX")
    print("=" * 50)
    matrix = engine.build_platoon_matrix()
    for pid, hands in list(matrix.items())[:4]:
        player = engine.players[pid]
        print(f"\n  {player.player_name} ({player.bats})")
        for hand, data in hands.items():
            print(f"    vs {hand}HP: wOBA={data['regressed_woba']:.3f} "
                  f"(raw={data['raw_woba']:.3f}), edge={data['platoon_edge']:+.3f}, "
                  f"conf={data['confidence']:.2f}, PA={data['pa']}")

    # 2) Home/Away Analysis
    print("\n" + "=" * 50)
    print("  2. HOME/AWAY SPLIT ANALYSIS")
    print("=" * 50)
    ha = engine.analyze_home_away("592450")
    print(f"\n  {ha['player_name']}:")
    print(f"    Home:  BA={ha['home']['ba']}, OPS={ha['home']['ops']}, wOBA={ha['home']['woba']}")
    print(f"    Away:  BA={ha['away']['ba']}, OPS={ha['away']['ops']}, wOBA={ha['away']['woba']}")
    print(f"    Diff:  wOBA {ha['difference']['woba_diff']:+.3f}")
    print(f"    Sig:   p={ha['significance']['p_value']}, "
          f"significant={ha['significance']['significant_at_05']}")

    # 3) Day/Night
    print("\n" + "=" * 50)
    print("  3. DAY/NIGHT DIFFERENTIAL")
    print("=" * 50)
    dn = engine.analyze_day_night("660271")
    print(f"\n  {dn['player_name']}:")
    print(f"    Day:   wOBA={dn['day']['woba']}, K%={dn['day']['k_rate']}")
    print(f"    Night: wOBA={dn['night']['woba']}, K%={dn['night']['k_rate']}")
    print(f"    → {dn['recommendation']}")

    # 4) Monthly Trends
    print("\n" + "=" * 50)
    print("  4. MONTHLY PERFORMANCE TRENDS")
    print("=" * 50)
    mt = engine.analyze_monthly_trends("665742")
    print(f"\n  {mt['player_name']} — Trend: {mt['trend']}, Volatility: {mt['volatility']}")
    for month, data in mt["monthly_splits"].items():
        print(f"    {month:>10}: wOBA={data['raw_woba']:.3f} → "
              f"regressed={data['regressed_woba']:.3f} (PA={data['pa']})")
    if mt["streaks"]:
        print(f"    Streaks: {mt['streaks']}")

    # 5) Situational
    print("\n" + "=" * 50)
    print("  5. SITUATIONAL HITTING ANALYSIS")
    print("=" * 50)
    sit = engine.analyze_situational("660271")
    print(f"\n  {sit['player_name']} — Clutch Score: {sit['clutch_score']:+.3f} "
          f"({sit['clutch_label']})")
    for sname, sdata in sit["situations"].items():
        print(f"    {sname:>16}: wOBA={sdata['raw_woba']:.3f} "
              f"(regressed={sdata['regressed_woba']:.3f}), "
              f"delta={sdata['clutch_delta']:+.3f}")

    # 6) Lineup Optimization
    print("\n" + "=" * 50)
    print("  6. PLATOON-OPTIMIZED LINEUP vs RHP")
    print("=" * 50)
    roster = list(engine.players.keys())
    lineup = engine.optimize_lineup_platoon(roster, "R", lineup_size=9)
    for spot in lineup:
        print(f"    #{spot['batting_order']}: {spot['player_name']:20s} "
              f"({spot['bats']}) wOBA={spot['regressed_woba']:.3f} "
              f"OBP={spot['regressed_obp']:.3f} ISO={spot['regressed_iso']:.3f}")

    # 7) Full Prediction
    print("\n" + "=" * 50)
    print("  7. FULL MATCHUP PREDICTION")
    print("=" * 50)
    pred = engine.predict_matchup(
        batter_id="660271",
        pitcher_id="477132",
        pitcher_hand="R",
        is_home=True,
        game_time="night",
        situation="risp",
    )
    print(f"\n  {pred['batter_name']} vs RHP (home, night, RISP)")
    print(f"    Predicted wOBA: {pred['prediction']['woba']:.3f} "
          f"[{pred['prediction']['ci_lower']:.3f}, {pred['prediction']['ci_upper']:.3f}]")
    print(f"    Base wOBA:      {pred['adjustments']['base_woba']:.3f}")
    print(f"    Venue adj:      {pred['adjustments']['venue_adjustment']:+.4f}")
    print(f"    Time adj:       {pred['adjustments']['time_adjustment']:+.4f}")
    print(f"    Situation adj:  {pred['adjustments']['situational_adjustment']:+.4f}")
    print(f"    Platoon edge:   {pred['adjustments']['platoon_edge']:+.3f}")
    print(f"    Reliability:    {pred['reliability']['grade']} "
          f"({pred['reliability']['score']:.2f})")

    # 8) Half-Season
    print("\n" + "=" * 50)
    print("  8. FIRST HALF vs SECOND HALF")
    print("=" * 50)
    hs = engine.analyze_half_season("668939")
    print(f"\n  {hs['player_name']}:")
    print(f"    1st Half: wOBA={hs['first_half']['woba']}, OPS={hs['first_half']['ops']}")
    print(f"    2nd Half: wOBA={hs['second_half']['woba']}, OPS={hs['second_half']['ops']}")
    print(f"    Pattern:  {hs['pattern']}, p={hs['p_value']}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — All systems operational")
    print("=" * 70)


if __name__ == "__main__":
    demo()
