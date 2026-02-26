#!/usr/bin/env python3
"""
Advanced Defensive Analytics Engine
=====================================
Comprehensive defensive metrics analysis for MLB game predictions.

Features:
- Outs Above Average (OAA) integration
- Defensive Runs Saved (DRS) by position
- Shift impact analysis (pre/post shift ban)
- Catcher framing metrics (strike zone analysis)
- Infield range metrics (groundball conversion rates)
- Outfield arm strength and accuracy ratings
- Team defensive efficiency ratings (DER)
- Position-by-position defensive WAR contributions
- Error probability by field condition (turf vs grass, weather)
- Impact on pitcher ERA (defense-adjusted ERA)
- Output: defensive adjustment factors for game predictions

Author: MLB Predictor System
Version: 1.0.0
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class Position(Enum):
    CATCHER = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SHORTSTOP = "SS"
    LEFT_FIELD = "LF"
    CENTER_FIELD = "CF"
    RIGHT_FIELD = "RF"
    PITCHER = "P"
    DESIGNATED_HITTER = "DH"

class FieldSurface(Enum):
    NATURAL_GRASS = "grass"
    ARTIFICIAL_TURF = "turf"

class WeatherCondition(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    WIND_STRONG = "wind_strong"
    COLD = "cold"
    HOT = "hot"
    DOME = "dome"

# Position weights for team defensive value
POSITION_DEFENSIVE_WEIGHT = {
    Position.CATCHER: 0.20,
    Position.SHORTSTOP: 0.16,
    Position.CENTER_FIELD: 0.14,
    Position.SECOND_BASE: 0.12,
    Position.THIRD_BASE: 0.11,
    Position.RIGHT_FIELD: 0.10,
    Position.LEFT_FIELD: 0.09,
    Position.FIRST_BASE: 0.06,
    Position.PITCHER: 0.02,
}

# Average DRS by position (league average = 0)
LEAGUE_AVG_DRS = 0.0
LEAGUE_AVG_OAA = 0.0
LEAGUE_AVG_DER = 0.700  # ~70% of balls in play converted to outs
LEAGUE_AVG_ERA = 4.20

# Framing runs per 1000 framing opportunities (league avg = 0)
LEAGUE_AVG_FRAMING_RUNS_PER_1000 = 0.0

# Surface error rate multipliers
SURFACE_ERROR_MULTIPLIER = {
    FieldSurface.NATURAL_GRASS: 1.00,
    FieldSurface.ARTIFICIAL_TURF: 0.92,  # Fewer bad hops on turf
}

# Weather error rate multipliers
WEATHER_ERROR_MULTIPLIER = {
    WeatherCondition.CLEAR: 1.00,
    WeatherCondition.CLOUDY: 1.02,
    WeatherCondition.RAIN: 1.25,
    WeatherCondition.DRIZZLE: 1.12,
    WeatherCondition.WIND_STRONG: 1.15,
    WeatherCondition.COLD: 1.08,
    WeatherCondition.HOT: 1.03,
    WeatherCondition.DOME: 0.95,
}

# Positional adjustment for WAR (runs per 162 games, relative to average)
POSITIONAL_ADJUSTMENT_RUNS = {
    Position.CATCHER: 12.5,
    Position.SHORTSTOP: 7.5,
    Position.CENTER_FIELD: 2.5,
    Position.SECOND_BASE: 3.0,
    Position.THIRD_BASE: 2.5,
    Position.RIGHT_FIELD: -7.5,
    Position.LEFT_FIELD: -7.5,
    Position.FIRST_BASE: -12.5,
    Position.DESIGNATED_HITTER: -17.5,
    Position.PITCHER: 0.0,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class FielderMetrics:
    """Comprehensive defensive metrics for a single fielder."""
    player_id: str
    player_name: str
    team: str
    position: Position
    games_played: int = 0
    innings_played: float = 0.0

    # Core metrics
    outs_above_average: int = 0       # OAA (Statcast)
    defensive_runs_saved: int = 0     # DRS (Inside Edge)
    ultimate_zone_rating: float = 0.0 # UZR
    range_runs: float = 0.0          # Range component of UZR
    error_runs: float = 0.0          # Error component of UZR

    # Counting stats
    total_chances: int = 0
    putouts: int = 0
    assists: int = 0
    errors: int = 0
    double_plays: int = 0
    fielding_pct: float = 0.0

    # Advanced
    reaction_time_ms: float = 0.0     # Sprint reaction (Statcast)
    sprint_speed_ft_s: float = 0.0    # Feet per second
    arm_strength_mph: float = 0.0     # Max throwing velocity
    exchange_time_ms: float = 0.0     # Glove-to-throw time

    # Groundball/flyball specific
    groundball_conversion_rate: float = 0.0
    flyball_catch_probability: float = 0.0
    line_drive_catch_rate: float = 0.0

    # Shift metrics
    shift_pa_faced: int = 0
    shift_hits_saved: int = 0
    shift_runs_saved: float = 0.0

    # Defensive WAR component
    defensive_war: float = 0.0

    def compute_fielding_pct(self) -> float:
        """Calculate fielding percentage."""
        if self.total_chances > 0:
            self.fielding_pct = (self.total_chances - self.errors) / self.total_chances
        return self.fielding_pct

    def compute_defensive_war(self) -> float:
        """
        Estimate defensive WAR from DRS + positional adjustment.
        dWAR ≈ (DRS + positional_adjustment) / runs_per_win
        """
        runs_per_win = 10.0  # Approximately 10 runs = 1 WAR
        pos_adj = POSITIONAL_ADJUSTMENT_RUNS.get(self.position, 0.0)
        # Pro-rate positional adjustment to games played
        games_factor = self.games_played / 162.0 if self.games_played > 0 else 0.0
        adj_runs = pos_adj * games_factor
        self.defensive_war = (self.defensive_runs_saved + adj_runs) / runs_per_win
        return self.defensive_war


@dataclass
class CatcherFramingMetrics:
    """Specialized catcher framing analysis."""
    player_id: str
    player_name: str
    team: str

    # Framing
    total_called_pitches: int = 0
    strikes_gained: int = 0        # Extra strikes above average
    strikes_lost: int = 0          # Lost strikes below average
    framing_runs: float = 0.0      # Runs saved/lost from framing
    framing_runs_per_1000: float = 0.0
    strike_rate: float = 0.0       # Called strike % on borderline pitches

    # Zone analysis (9 zones: 4 corners, 4 edges, 1 heart)
    zone_rates: Dict[str, float] = field(default_factory=dict)

    # Blocking
    wild_pitches: int = 0
    passed_balls: int = 0
    block_rate: float = 0.0        # Pitches in dirt blocked %

    # Throwing
    caught_stealing_pct: float = 0.0
    pop_time_seconds: float = 0.0  # Average 2nd base pop time
    stolen_base_attempts: int = 0
    caught_stealing: int = 0

    # Game calling
    catcher_era: float = 0.0
    pitcher_strikeout_rate_with: float = 0.0
    pitcher_walk_rate_with: float = 0.0

    def compute_framing_runs_per_1000(self) -> float:
        if self.total_called_pitches > 0:
            self.framing_runs_per_1000 = (self.framing_runs / self.total_called_pitches) * 1000
        return self.framing_runs_per_1000

    def compute_cs_pct(self) -> float:
        if self.stolen_base_attempts > 0:
            self.caught_stealing_pct = self.caught_stealing / self.stolen_base_attempts
        return self.caught_stealing_pct


@dataclass
class OutfieldArmMetrics:
    """Outfielder arm strength and accuracy metrics."""
    player_id: str
    player_name: str
    position: Position
    team: str

    max_throw_speed_mph: float = 0.0
    avg_throw_speed_mph: float = 0.0
    throw_accuracy_pct: float = 0.0   # % of throws on target
    outfield_assists: int = 0
    outfield_assist_opportunities: int = 0
    arm_runs_saved: float = 0.0
    runners_held_pct: float = 0.0     # Runners who don't advance

    def compute_assist_rate(self) -> float:
        if self.outfield_assist_opportunities > 0:
            return self.outfield_assists / self.outfield_assist_opportunities
        return 0.0


@dataclass
class TeamDefense:
    """Aggregate team defensive profile."""
    team: str
    season: int = 2025

    # Team totals
    team_drs: int = 0
    team_oaa: int = 0
    team_uzr: float = 0.0
    team_der: float = 0.0  # Defensive Efficiency Ratio
    team_errors: int = 0
    team_fielding_pct: float = 0.0
    team_defensive_war: float = 0.0

    # By position
    position_metrics: Dict[str, FielderMetrics] = field(default_factory=dict)
    catcher_framing: Optional[CatcherFramingMetrics] = None
    outfield_arms: List[OutfieldArmMetrics] = field(default_factory=list)

    # Shift analysis
    total_shift_pa: int = 0
    shift_runs_saved: float = 0.0
    pre_ban_shift_rate: float = 0.0
    post_ban_adjustment: float = 0.0

    # ERA impact
    defense_adjusted_era_factor: float = 0.0  # +/- runs per 9 innings

    # Conditions
    home_surface: FieldSurface = FieldSurface.NATURAL_GRASS
    error_rate_surface_adjusted: float = 0.0


@dataclass
class DefensiveGameAdjustment:
    """Defensive adjustment factors for a specific game prediction."""
    game_id: str
    home_team: str
    away_team: str
    home_adjustment: float = 0.0   # Runs adjustment (+ = better defense)
    away_adjustment: float = 0.0
    surface: FieldSurface = FieldSurface.NATURAL_GRASS
    weather: WeatherCondition = WeatherCondition.CLEAR
    surface_factor: float = 1.0
    weather_factor: float = 1.0
    framing_edge: float = 0.0      # Home catcher framing advantage
    arm_edge: float = 0.0          # Outfield arm advantage
    range_edge: float = 0.0        # Infield range advantage
    net_defensive_edge: float = 0.0  # Total defensive edge (home perspective)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Defensive Metrics Engine
# ---------------------------------------------------------------------------

class DefensiveMetricsEngine:
    """
    Main engine for computing advanced defensive analytics and
    their impact on game predictions.
    """

    def __init__(self):
        self.fielders: Dict[str, FielderMetrics] = {}
        self.catchers: Dict[str, CatcherFramingMetrics] = {}
        self.outfield_arms: Dict[str, OutfieldArmMetrics] = {}
        self.team_defenses: Dict[str, TeamDefense] = {}
        self.stadium_surfaces: Dict[str, FieldSurface] = self._init_stadium_surfaces()

    # ---- Stadium Data ----

    @staticmethod
    def _init_stadium_surfaces() -> Dict[str, FieldSurface]:
        """Map team codes to their home field surface type."""
        turf_teams = {"TBR", "TOR", "ARI", "MIL", "TEX", "MIA", "HOU"}
        surfaces = {}
        all_teams = [
            "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
            "COL", "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL",
            "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SD", "SF",
            "SEA", "STL", "TBR", "TEX", "TOR", "WSH",
        ]
        for team in all_teams:
            surfaces[team] = FieldSurface.ARTIFICIAL_TURF if team in turf_teams else FieldSurface.NATURAL_GRASS
        return surfaces

    # ---- Data Loading ----

    def load_fielder(self, data: Dict[str, Any]) -> FielderMetrics:
        """Load fielder metrics from a dictionary."""
        pos = Position(data.get("position", "SS"))
        fielder = FielderMetrics(
            player_id=data["player_id"],
            player_name=data.get("name", "Unknown"),
            team=data.get("team", "UNK"),
            position=pos,
            games_played=data.get("games", 0),
            innings_played=data.get("innings", 0.0),
            outs_above_average=data.get("oaa", 0),
            defensive_runs_saved=data.get("drs", 0),
            ultimate_zone_rating=data.get("uzr", 0.0),
            range_runs=data.get("range_runs", 0.0),
            error_runs=data.get("error_runs", 0.0),
            total_chances=data.get("tc", 0),
            putouts=data.get("po", 0),
            assists=data.get("a", 0),
            errors=data.get("e", 0),
            double_plays=data.get("dp", 0),
            reaction_time_ms=data.get("reaction_ms", 0.0),
            sprint_speed_ft_s=data.get("sprint_speed", 0.0),
            arm_strength_mph=data.get("arm_strength", 0.0),
            exchange_time_ms=data.get("exchange_time", 0.0),
            groundball_conversion_rate=data.get("gb_conversion", 0.0),
            flyball_catch_probability=data.get("fb_catch_prob", 0.0),
            line_drive_catch_rate=data.get("ld_catch_rate", 0.0),
            shift_pa_faced=data.get("shift_pa", 0),
            shift_hits_saved=data.get("shift_hits_saved", 0),
            shift_runs_saved=data.get("shift_runs_saved", 0.0),
        )
        fielder.compute_fielding_pct()
        fielder.compute_defensive_war()
        self.fielders[data["player_id"]] = fielder
        return fielder

    def load_catcher_framing(self, data: Dict[str, Any]) -> CatcherFramingMetrics:
        """Load catcher-specific framing metrics."""
        catcher = CatcherFramingMetrics(
            player_id=data["player_id"],
            player_name=data.get("name", "Unknown"),
            team=data.get("team", "UNK"),
            total_called_pitches=data.get("called_pitches", 0),
            strikes_gained=data.get("strikes_gained", 0),
            strikes_lost=data.get("strikes_lost", 0),
            framing_runs=data.get("framing_runs", 0.0),
            strike_rate=data.get("strike_rate", 0.0),
            zone_rates=data.get("zone_rates", {}),
            wild_pitches=data.get("wp", 0),
            passed_balls=data.get("pb", 0),
            block_rate=data.get("block_rate", 0.0),
            pop_time_seconds=data.get("pop_time", 0.0),
            stolen_base_attempts=data.get("sb_attempts", 0),
            caught_stealing=data.get("cs", 0),
            catcher_era=data.get("cera", 0.0),
            pitcher_strikeout_rate_with=data.get("k_rate_with", 0.0),
            pitcher_walk_rate_with=data.get("bb_rate_with", 0.0),
        )
        catcher.compute_framing_runs_per_1000()
        catcher.compute_cs_pct()
        self.catchers[data["player_id"]] = catcher
        return catcher

    def load_outfield_arm(self, data: Dict[str, Any]) -> OutfieldArmMetrics:
        """Load outfielder arm metrics."""
        arm = OutfieldArmMetrics(
            player_id=data["player_id"],
            player_name=data.get("name", "Unknown"),
            position=Position(data.get("position", "RF")),
            team=data.get("team", "UNK"),
            max_throw_speed_mph=data.get("max_throw", 0.0),
            avg_throw_speed_mph=data.get("avg_throw", 0.0),
            throw_accuracy_pct=data.get("accuracy", 0.0),
            outfield_assists=data.get("of_assists", 0),
            outfield_assist_opportunities=data.get("of_assist_opps", 0),
            arm_runs_saved=data.get("arm_runs", 0.0),
            runners_held_pct=data.get("runners_held", 0.0),
        )
        self.outfield_arms[data["player_id"]] = arm
        return arm

    # ---- Team Defense Aggregation ----

    def build_team_defense(self, team: str, fielder_ids: List[str],
                           catcher_id: str = None) -> TeamDefense:
        """
        Aggregate individual fielder metrics into team defensive profile.
        """
        td = TeamDefense(team=team)
        td.home_surface = self.stadium_surfaces.get(team, FieldSurface.NATURAL_GRASS)

        total_drs = 0
        total_oaa = 0
        total_uzr = 0.0
        total_errors = 0
        total_chances = 0
        total_dwar = 0.0

        for pid in fielder_ids:
            f = self.fielders.get(pid)
            if f is None:
                continue
            total_drs += f.defensive_runs_saved
            total_oaa += f.outs_above_average
            total_uzr += f.ultimate_zone_rating
            total_errors += f.errors
            total_chances += f.total_chances
            total_dwar += f.defensive_war
            td.position_metrics[f.position.value] = f

        td.team_drs = total_drs
        td.team_oaa = total_oaa
        td.team_uzr = total_uzr
        td.team_errors = total_errors
        td.team_fielding_pct = (
            (total_chances - total_errors) / total_chances
            if total_chances > 0 else 0.970
        )
        td.team_defensive_war = total_dwar

        # DER approximation based on team DRS
        # Better defense → higher DER
        td.team_der = LEAGUE_AVG_DER + (total_drs / 162) * 0.005

        # Catcher framing
        if catcher_id and catcher_id in self.catchers:
            td.catcher_framing = self.catchers[catcher_id]

        # Outfield arms
        for pid in fielder_ids:
            if pid in self.outfield_arms:
                td.outfield_arms.append(self.outfield_arms[pid])

        # Compute ERA impact
        td.defense_adjusted_era_factor = self._compute_era_impact(td)

        self.team_defenses[team] = td
        return td

    def _compute_era_impact(self, td: TeamDefense) -> float:
        """
        Calculate how team defense impacts pitcher ERA.
        Better defense = lower ERA; expressed as runs/9 innings adjustment.
        """
        # DRS per game → ERA impact
        # ~10 DRS over a season ≈ 0.15 ERA impact
        drs_per_game = td.team_drs / 162.0 if td.team_drs != 0 else 0.0
        drs_era_impact = drs_per_game * 9  # Scale to per-9-innings

        # Framing impact
        framing_era = 0.0
        if td.catcher_framing:
            # ~15 framing runs ≈ 0.2 ERA impact
            framing_era = td.catcher_framing.framing_runs / 162.0 * 9 * 0.5

        # Error impact
        # Each error ≈ 0.4 unearned runs
        error_rate = td.team_errors / 162.0 if td.team_errors > 0 else 0.5
        league_avg_errors = 85 / 162.0  # ~85 errors per season league avg
        error_impact = (league_avg_errors - error_rate) * 0.4 * 9 / 27

        total_impact = -(drs_era_impact * 0.15 + framing_era + error_impact)
        return round(total_impact, 3)

    # ---- Shift Impact Analysis ----

    def analyze_shift_impact(self, team: str) -> Dict[str, Any]:
        """
        Analyze the impact of defensive shifts (and the 2023 shift ban).
        """
        td = self.team_defenses.get(team)
        if td is None:
            return {"error": f"Team {team} not loaded"}

        total_shift_pa = 0
        total_shift_hits_saved = 0
        total_shift_runs = 0.0

        for pos, fielder in td.position_metrics.items():
            total_shift_pa += fielder.shift_pa_faced
            total_shift_hits_saved += fielder.shift_hits_saved
            total_shift_runs += fielder.shift_runs_saved

        # Estimate post-ban adjustment
        # With shift ban (2023+), infielders must stay on their side
        # Average team lost ~5-8 DRS from shift ban
        post_ban_drs_loss = -6.0  # Average league impact
        shift_rate = total_shift_pa / 5000 if total_shift_pa > 0 else 0.15
        post_ban_adjustment = post_ban_drs_loss * shift_rate

        return {
            "team": team,
            "total_shift_pa": total_shift_pa,
            "shift_hits_saved": total_shift_hits_saved,
            "shift_runs_saved": round(total_shift_runs, 1),
            "pre_ban_shift_rate": round(shift_rate, 3),
            "post_ban_drs_impact": round(post_ban_adjustment, 1),
            "pre_vs_post": {
                "pre_ban_team_drs": td.team_drs,
                "estimated_post_ban_drs": td.team_drs + int(post_ban_adjustment),
                "era_impact_change": round(post_ban_adjustment * 0.15 / 162 * 9, 3),
            },
            "most_affected_positions": ["2B", "SS", "3B"],
        }

    # ---- Infield Range Metrics ----

    def analyze_infield_range(self, team: str) -> Dict[str, Any]:
        """Analyze infield range and groundball conversion rates."""
        td = self.team_defenses.get(team)
        if td is None:
            return {"error": f"Team {team} not loaded"}

        infield_positions = [Position.FIRST_BASE, Position.SECOND_BASE,
                            Position.THIRD_BASE, Position.SHORTSTOP]
        infield_data = {}
        total_gb_rate = 0.0
        count = 0

        for pos in infield_positions:
            f = td.position_metrics.get(pos.value)
            if f is None:
                continue
            infield_data[pos.value] = {
                "player": f.player_name,
                "oaa": f.outs_above_average,
                "drs": f.defensive_runs_saved,
                "range_runs": round(f.range_runs, 1),
                "gb_conversion": round(f.groundball_conversion_rate, 3),
                "reaction_ms": round(f.reaction_time_ms, 1),
                "sprint_speed": round(f.sprint_speed_ft_s, 1),
                "errors": f.errors,
                "double_plays": f.double_plays,
            }
            total_gb_rate += f.groundball_conversion_rate
            count += 1

        avg_gb_rate = total_gb_rate / count if count > 0 else 0.72

        return {
            "team": team,
            "infield_range": infield_data,
            "avg_gb_conversion": round(avg_gb_rate, 3),
            "league_avg_gb_conversion": 0.720,
            "advantage": round(avg_gb_rate - 0.720, 3),
            "runs_impact": round((avg_gb_rate - 0.720) * 162 * 4, 1),  # Approx
        }

    # ---- Outfield Arm Analysis ----

    def analyze_outfield_arms(self, team: str) -> Dict[str, Any]:
        """Analyze outfield arm strength and accuracy."""
        td = self.team_defenses.get(team)
        if td is None:
            return {"error": f"Team {team} not loaded"}

        arm_data = {}
        total_arm_runs = 0.0

        for arm in td.outfield_arms:
            assist_rate = arm.compute_assist_rate()
            arm_data[arm.position.value] = {
                "player": arm.player_name,
                "max_throw_mph": round(arm.max_throw_speed_mph, 1),
                "avg_throw_mph": round(arm.avg_throw_speed_mph, 1),
                "accuracy_pct": round(arm.throw_accuracy_pct * 100, 1),
                "assists": arm.outfield_assists,
                "assist_rate": round(assist_rate, 3),
                "arm_runs_saved": round(arm.arm_runs_saved, 1),
                "runners_held_pct": round(arm.runners_held_pct * 100, 1),
            }
            total_arm_runs += arm.arm_runs_saved

        return {
            "team": team,
            "outfield_arms": arm_data,
            "total_arm_runs_saved": round(total_arm_runs, 1),
            "arm_rating": (
                "elite" if total_arm_runs > 5
                else "above_average" if total_arm_runs > 2
                else "average" if total_arm_runs > -2
                else "below_average" if total_arm_runs > -5
                else "poor"
            ),
        }

    # ---- Catcher Framing Analysis ----

    def analyze_catcher_framing(self, catcher_id: str) -> Dict[str, Any]:
        """Detailed catcher framing analysis with zone breakdown."""
        catcher = self.catchers.get(catcher_id)
        if catcher is None:
            return {"error": f"Catcher {catcher_id} not found"}

        # ERA impact from framing
        # ~12 framing runs ≈ 0.15 ERA reduction for pitchers
        framing_era_impact = catcher.framing_runs * 0.0125

        return {
            "player_id": catcher_id,
            "player_name": catcher.player_name,
            "team": catcher.team,
            "framing": {
                "total_called_pitches": catcher.total_called_pitches,
                "strikes_gained": catcher.strikes_gained,
                "strikes_lost": catcher.strikes_lost,
                "net_strikes": catcher.strikes_gained - catcher.strikes_lost,
                "framing_runs": round(catcher.framing_runs, 1),
                "framing_runs_per_1000": round(catcher.framing_runs_per_1000, 2),
                "strike_rate": round(catcher.strike_rate * 100, 1),
                "era_impact": round(framing_era_impact, 3),
            },
            "zone_breakdown": catcher.zone_rates,
            "blocking": {
                "wild_pitches": catcher.wild_pitches,
                "passed_balls": catcher.passed_balls,
                "block_rate": round(catcher.block_rate * 100, 1),
            },
            "throwing": {
                "cs_pct": round(catcher.caught_stealing_pct * 100, 1),
                "pop_time": round(catcher.pop_time_seconds, 3),
                "sb_attempts": catcher.stolen_base_attempts,
                "caught_stealing": catcher.caught_stealing,
            },
            "game_calling": {
                "catcher_era": round(catcher.catcher_era, 2),
                "k_rate_with": round(catcher.pitcher_strikeout_rate_with * 100, 1),
                "bb_rate_with": round(catcher.pitcher_walk_rate_with * 100, 1),
            },
            "overall_rating": self._rate_catcher(catcher),
        }

    @staticmethod
    def _rate_catcher(c: CatcherFramingMetrics) -> Dict[str, Any]:
        """Generate an overall catcher defensive rating."""
        # Weighted score: 40% framing, 25% blocking, 20% throwing, 15% game calling
        framing_score = min(100, max(0, 50 + c.framing_runs * 3))
        blocking_score = min(100, max(0, c.block_rate * 100))
        throwing_score = min(100, max(0, c.caught_stealing_pct * 100 * 2.5))
        calling_score = min(100, max(0, 50 + (LEAGUE_AVG_ERA - c.catcher_era) * 15))

        overall = (framing_score * 0.40 + blocking_score * 0.25 +
                   throwing_score * 0.20 + calling_score * 0.15)

        return {
            "overall_score": round(overall, 1),
            "framing_score": round(framing_score, 1),
            "blocking_score": round(blocking_score, 1),
            "throwing_score": round(throwing_score, 1),
            "calling_score": round(calling_score, 1),
            "tier": (
                "elite" if overall > 75
                else "above_average" if overall > 60
                else "average" if overall > 45
                else "below_average" if overall > 30
                else "poor"
            ),
        }

    # ---- Error Probability by Conditions ----

    def compute_error_probability(self, team: str,
                                   surface: FieldSurface = None,
                                   weather: WeatherCondition = WeatherCondition.CLEAR
                                   ) -> Dict[str, Any]:
        """
        Calculate adjusted error probability based on field conditions.
        """
        td = self.team_defenses.get(team)
        if td is None:
            # Use league average
            base_error_rate = 85 / (162 * 27)  # Errors per opportunity
        else:
            games = max(1, sum(f.games_played for f in td.position_metrics.values()) / 9)
            base_error_rate = td.team_errors / (games * 27)

        if surface is None:
            surface = self.stadium_surfaces.get(team, FieldSurface.NATURAL_GRASS)

        surf_mult = SURFACE_ERROR_MULTIPLIER.get(surface, 1.0)
        weather_mult = WEATHER_ERROR_MULTIPLIER.get(weather, 1.0)

        adjusted_rate = base_error_rate * surf_mult * weather_mult

        return {
            "team": team,
            "base_error_rate": round(base_error_rate, 5),
            "surface": surface.value,
            "surface_multiplier": surf_mult,
            "weather": weather.value,
            "weather_multiplier": weather_mult,
            "adjusted_error_rate": round(adjusted_rate, 5),
            "expected_errors_per_game": round(adjusted_rate * 27, 2),
            "impact_on_runs": round((adjusted_rate - base_error_rate) * 27 * 0.4, 3),
        }

    # ---- Defense-Adjusted ERA ----

    def compute_defense_adjusted_era(self, pitcher_era: float,
                                      team: str) -> Dict[str, Any]:
        """
        Adjust a pitcher's ERA for team defensive quality.
        dERA = ERA + defensive_factor
        """
        td = self.team_defenses.get(team)
        if td is None:
            return {
                "raw_era": round(pitcher_era, 2),
                "adjusted_era": round(pitcher_era, 2),
                "adjustment": 0.0,
                "note": "No team defense data available",
            }

        adj_factor = td.defense_adjusted_era_factor

        # Framing adjustment
        framing_adj = 0.0
        if td.catcher_framing:
            framing_adj = td.catcher_framing.framing_runs * 0.0125

        total_adj = adj_factor + framing_adj
        adjusted_era = pitcher_era + total_adj

        return {
            "raw_era": round(pitcher_era, 2),
            "adjusted_era": round(max(0, adjusted_era), 2),
            "total_adjustment": round(total_adj, 3),
            "defense_component": round(adj_factor, 3),
            "framing_component": round(framing_adj, 3),
            "team_drs": td.team_drs,
            "team_der": round(td.team_der, 3),
            "interpretation": (
                f"Defense {'helps' if total_adj < 0 else 'hurts'} this pitcher by "
                f"{abs(total_adj):.2f} ERA points"
            ),
        }

    # ---- Team DER (Defensive Efficiency) ----

    def compute_team_der(self, team: str) -> Dict[str, Any]:
        """
        Compute and analyze team Defensive Efficiency Rating.
        DER = 1 - (H - HR) / (PA - BB - SO - HBP - HR)
        """
        td = self.team_defenses.get(team)
        if td is None:
            return {"error": f"Team {team} not loaded"}

        return {
            "team": team,
            "der": round(td.team_der, 3),
            "league_avg": LEAGUE_AVG_DER,
            "difference": round(td.team_der - LEAGUE_AVG_DER, 3),
            "rank_estimate": (
                "top_5" if td.team_der > 0.715
                else "above_average" if td.team_der > 0.705
                else "average" if td.team_der > 0.695
                else "below_average" if td.team_der > 0.685
                else "bottom_5"
            ),
            "runs_impact_per_game": round((td.team_der - LEAGUE_AVG_DER) * 4.5 * 9, 2),
        }

    # ---- Position-by-Position WAR ----

    def position_war_breakdown(self, team: str) -> Dict[str, Any]:
        """Break down defensive WAR by position."""
        td = self.team_defenses.get(team)
        if td is None:
            return {"error": f"Team {team} not loaded"}

        breakdown = {}
        for pos, fielder in td.position_metrics.items():
            weight = POSITION_DEFENSIVE_WEIGHT.get(Position(pos), 0.05)
            breakdown[pos] = {
                "player": fielder.player_name,
                "drs": fielder.defensive_runs_saved,
                "oaa": fielder.outs_above_average,
                "d_war": round(fielder.defensive_war, 2),
                "position_weight": round(weight, 2),
                "weighted_contribution": round(fielder.defensive_war * weight * 10, 2),
            }

        total_dwar = sum(d["d_war"] for d in breakdown.values())

        return {
            "team": team,
            "total_defensive_war": round(total_dwar, 2),
            "positions": breakdown,
            "strongest_position": max(breakdown.items(), key=lambda x: x[1]["d_war"])[0] if breakdown else None,
            "weakest_position": min(breakdown.items(), key=lambda x: x[1]["d_war"])[0] if breakdown else None,
        }

    # ---- Game Prediction Adjustment ----

    def compute_game_adjustment(self, home_team: str, away_team: str,
                                 surface: FieldSurface = None,
                                 weather: WeatherCondition = WeatherCondition.CLEAR
                                 ) -> DefensiveGameAdjustment:
        """
        Compute the net defensive adjustment for a game prediction.
        Positive home_adjustment = home team defense is better.
        """
        if surface is None:
            surface = self.stadium_surfaces.get(home_team, FieldSurface.NATURAL_GRASS)

        home_td = self.team_defenses.get(home_team)
        away_td = self.team_defenses.get(away_team)

        home_adj = 0.0
        away_adj = 0.0
        framing_edge = 0.0
        arm_edge = 0.0
        range_edge = 0.0

        if home_td:
            home_adj = -home_td.defense_adjusted_era_factor  # Negative ERA factor = good
        if away_td:
            away_adj = -away_td.defense_adjusted_era_factor

        # Framing edge
        if home_td and home_td.catcher_framing and away_td and away_td.catcher_framing:
            framing_edge = (home_td.catcher_framing.framing_runs -
                          away_td.catcher_framing.framing_runs) * 0.01

        # Arm edge
        if home_td and away_td:
            home_arm = sum(a.arm_runs_saved for a in home_td.outfield_arms)
            away_arm = sum(a.arm_runs_saved for a in away_td.outfield_arms)
            arm_edge = (home_arm - away_arm) * 0.1

        # Range edge (DRS-based)
        if home_td and away_td:
            range_edge = (home_td.team_drs - away_td.team_drs) * 0.01

        net_edge = (home_adj - away_adj + framing_edge + arm_edge + range_edge)

        # Confidence based on data quality
        conf_factors = []
        if home_td:
            conf_factors.append(min(1.0, sum(
                f.games_played for f in home_td.position_metrics.values()
            ) / (8 * 100)))
        if away_td:
            conf_factors.append(min(1.0, sum(
                f.games_played for f in away_td.position_metrics.values()
            ) / (8 * 100)))
        confidence = statistics.mean(conf_factors) if conf_factors else 0.3

        # Surface and weather factors
        surf_factor = SURFACE_ERROR_MULTIPLIER.get(surface, 1.0)
        weather_factor = WEATHER_ERROR_MULTIPLIER.get(weather, 1.0)

        return DefensiveGameAdjustment(
            game_id=f"{home_team}_vs_{away_team}",
            home_team=home_team,
            away_team=away_team,
            home_adjustment=round(home_adj, 3),
            away_adjustment=round(away_adj, 3),
            surface=surface,
            weather=weather,
            surface_factor=surf_factor,
            weather_factor=weather_factor,
            framing_edge=round(framing_edge, 3),
            arm_edge=round(arm_edge, 3),
            range_edge=round(range_edge, 3),
            net_defensive_edge=round(net_edge, 3),
            confidence=round(confidence, 3),
        )

    def adjustment_to_dict(self, adj: DefensiveGameAdjustment) -> Dict[str, Any]:
        """Convert DefensiveGameAdjustment to JSON-ready dict."""
        return {
            "game_id": adj.game_id,
            "home_team": adj.home_team,
            "away_team": adj.away_team,
            "defensive_edge": {
                "net_edge_runs": adj.net_defensive_edge,
                "home_adjustment": adj.home_adjustment,
                "away_adjustment": adj.away_adjustment,
                "framing_edge": adj.framing_edge,
                "arm_edge": adj.arm_edge,
                "range_edge": adj.range_edge,
            },
            "conditions": {
                "surface": adj.surface.value,
                "weather": adj.weather.value,
                "surface_factor": adj.surface_factor,
                "weather_factor": adj.weather_factor,
            },
            "confidence": adj.confidence,
            "recommendation": (
                f"Home team ({adj.home_team}) has defensive edge of {adj.net_defensive_edge:+.3f} runs"
                if adj.net_defensive_edge > 0
                else f"Away team ({adj.away_team}) has defensive edge of {-adj.net_defensive_edge:+.3f} runs"
                if adj.net_defensive_edge < 0
                else "Defensive matchup is even"
            ),
        }


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------

def demo():
    """Run a comprehensive demo of the Defensive Metrics Engine."""
    print("=" * 70)
    print("  ADVANCED DEFENSIVE ANALYTICS ENGINE — DEMO")
    print("=" * 70)

    random.seed(42)
    engine = DefensiveMetricsEngine()

    # ---- Build two full team defenses ----
    teams_data = {
        "LAD": {
            "fielders": [
                {"player_id": "c001", "name": "Will Smith", "team": "LAD", "position": "C",
                 "games": 130, "innings": 1080, "oaa": 3, "drs": 5, "uzr": 4.2,
                 "range_runs": 2.1, "error_runs": 0.5, "tc": 820, "po": 750, "a": 55,
                 "e": 5, "dp": 3, "gb_conversion": 0.0, "sprint_speed": 26.5,
                 "arm_strength": 82.0},
                {"player_id": "f001", "name": "Freddie Freeman", "team": "LAD", "position": "1B",
                 "games": 155, "innings": 1350, "oaa": 1, "drs": 2, "uzr": 1.5,
                 "range_runs": 0.8, "error_runs": 0.2, "tc": 1350, "po": 1280, "a": 65,
                 "e": 5, "dp": 120, "gb_conversion": 0.78, "sprint_speed": 26.0},
                {"player_id": "f002", "name": "Gavin Lux", "team": "LAD", "position": "2B",
                 "games": 120, "innings": 980, "oaa": 5, "drs": 7, "uzr": 5.8,
                 "range_runs": 4.2, "error_runs": 0.3, "tc": 450, "po": 200, "a": 240,
                 "e": 8, "dp": 65, "gb_conversion": 0.75, "sprint_speed": 27.8,
                 "reaction_ms": 15.2},
                {"player_id": "f003", "name": "Max Muncy", "team": "LAD", "position": "3B",
                 "games": 140, "innings": 1150, "oaa": -2, "drs": -3, "uzr": -2.1,
                 "range_runs": -1.8, "error_runs": -0.8, "tc": 320, "po": 85, "a": 220,
                 "e": 12, "dp": 22, "gb_conversion": 0.68, "sprint_speed": 25.5,
                 "reaction_ms": 18.5},
                {"player_id": "f004", "name": "Miguel Rojas", "team": "LAD", "position": "SS",
                 "games": 135, "innings": 1100, "oaa": 8, "drs": 10, "uzr": 8.5,
                 "range_runs": 6.5, "error_runs": 0.8, "tc": 520, "po": 180, "a": 330,
                 "e": 9, "dp": 78, "gb_conversion": 0.76, "sprint_speed": 28.2,
                 "reaction_ms": 14.0},
                {"player_id": "f005", "name": "Chris Taylor", "team": "LAD", "position": "LF",
                 "games": 110, "innings": 850, "oaa": 2, "drs": 3, "uzr": 2.5,
                 "range_runs": 1.8, "error_runs": 0.2, "tc": 180, "po": 170, "a": 5,
                 "e": 3, "dp": 0, "fb_catch_prob": 0.94, "sprint_speed": 27.0},
                {"player_id": "f006", "name": "James Outman", "team": "LAD", "position": "CF",
                 "games": 140, "innings": 1200, "oaa": 12, "drs": 14, "uzr": 11.2,
                 "range_runs": 9.5, "error_runs": 0.5, "tc": 350, "po": 340, "a": 6,
                 "e": 2, "dp": 0, "fb_catch_prob": 0.97, "sprint_speed": 29.5},
                {"player_id": "f007", "name": "Mookie Betts", "team": "LAD", "position": "RF",
                 "games": 150, "innings": 1300, "oaa": 10, "drs": 12, "uzr": 9.8,
                 "range_runs": 7.2, "error_runs": 1.0, "tc": 300, "po": 280, "a": 12,
                 "e": 3, "dp": 0, "fb_catch_prob": 0.96, "sprint_speed": 28.0},
            ],
            "catcher": {
                "player_id": "c001", "name": "Will Smith", "team": "LAD",
                "called_pitches": 8500, "strikes_gained": 45, "strikes_lost": 12,
                "framing_runs": 12.5, "strike_rate": 0.485,
                "zone_rates": {"upper_left": 0.42, "upper_right": 0.45, "lower_left": 0.52,
                               "lower_right": 0.55, "up_edge": 0.38, "down_edge": 0.48,
                               "left_edge": 0.44, "right_edge": 0.46, "heart": 0.92},
                "wp": 5, "pb": 3, "block_rate": 0.975,
                "pop_time": 1.92, "sb_attempts": 85, "cs": 28,
                "cera": 3.65, "k_rate_with": 0.245, "bb_rate_with": 0.072,
            },
            "outfield_arms": [
                {"player_id": "f005", "name": "Chris Taylor", "team": "LAD", "position": "LF",
                 "max_throw": 88.5, "avg_throw": 85.2, "accuracy": 0.78, "of_assists": 4,
                 "of_assist_opps": 25, "arm_runs": 1.2, "runners_held": 0.35},
                {"player_id": "f006", "name": "James Outman", "team": "LAD", "position": "CF",
                 "max_throw": 92.0, "avg_throw": 88.5, "accuracy": 0.82, "of_assists": 6,
                 "of_assist_opps": 30, "arm_runs": 2.8, "runners_held": 0.42},
                {"player_id": "f007", "name": "Mookie Betts", "team": "LAD", "position": "RF",
                 "max_throw": 94.5, "avg_throw": 91.0, "accuracy": 0.88, "of_assists": 10,
                 "of_assist_opps": 35, "arm_runs": 5.5, "runners_held": 0.52},
            ],
        },
        "NYY": {
            "fielders": [
                {"player_id": "c010", "name": "Jose Trevino", "team": "NYY", "position": "C",
                 "games": 110, "innings": 900, "oaa": 1, "drs": 2, "uzr": 1.5,
                 "tc": 650, "po": 600, "a": 40, "e": 8, "dp": 2},
                {"player_id": "f010", "name": "Anthony Rizzo", "team": "NYY", "position": "1B",
                 "games": 120, "innings": 1000, "oaa": -1, "drs": -2, "uzr": -1.5,
                 "tc": 1100, "po": 1050, "a": 45, "e": 6, "dp": 95, "gb_conversion": 0.74},
                {"player_id": "f011", "name": "Gleyber Torres", "team": "NYY", "position": "2B",
                 "games": 145, "innings": 1200, "oaa": -3, "drs": -5, "uzr": -4.2,
                 "range_runs": -3.5, "tc": 500, "po": 220, "a": 260, "e": 14,
                 "dp": 70, "gb_conversion": 0.69, "sprint_speed": 26.5},
                {"player_id": "f012", "name": "Josh Donaldson", "team": "NYY", "position": "3B",
                 "games": 100, "innings": 800, "oaa": -4, "drs": -6, "uzr": -5.0,
                 "range_runs": -4.0, "tc": 250, "po": 60, "a": 175, "e": 12,
                 "gb_conversion": 0.65, "sprint_speed": 24.8},
                {"player_id": "f013", "name": "Anthony Volpe", "team": "NYY", "position": "SS",
                 "games": 155, "innings": 1350, "oaa": 6, "drs": 8, "uzr": 6.5,
                 "range_runs": 5.0, "tc": 580, "po": 200, "a": 365, "e": 15,
                 "dp": 85, "gb_conversion": 0.73, "sprint_speed": 28.5, "reaction_ms": 14.5},
                {"player_id": "f014", "name": "Harrison Bader", "team": "NYY", "position": "LF",
                 "games": 100, "innings": 780, "oaa": 3, "drs": 4, "uzr": 3.0,
                 "tc": 160, "po": 150, "a": 4, "e": 2, "fb_catch_prob": 0.95,
                 "sprint_speed": 28.8},
                {"player_id": "f015", "name": "Aaron Judge", "team": "NYY", "position": "CF",
                 "games": 150, "innings": 1300, "oaa": -2, "drs": -3, "uzr": -2.5,
                 "tc": 320, "po": 310, "a": 4, "e": 3, "fb_catch_prob": 0.92,
                 "sprint_speed": 27.2},
                {"player_id": "f016", "name": "Juan Soto", "team": "NYY", "position": "RF",
                 "games": 155, "innings": 1320, "oaa": -5, "drs": -7, "uzr": -5.5,
                 "tc": 280, "po": 260, "a": 8, "e": 5, "fb_catch_prob": 0.90,
                 "sprint_speed": 26.0},
            ],
            "catcher": {
                "player_id": "c010", "name": "Jose Trevino", "team": "NYY",
                "called_pitches": 7200, "strikes_gained": 35, "strikes_lost": 8,
                "framing_runs": 10.2, "strike_rate": 0.478,
                "zone_rates": {"upper_left": 0.40, "upper_right": 0.43, "lower_left": 0.50,
                               "lower_right": 0.53, "up_edge": 0.36, "down_edge": 0.46,
                               "left_edge": 0.42, "right_edge": 0.44, "heart": 0.91},
                "wp": 8, "pb": 6, "block_rate": 0.960,
                "pop_time": 1.95, "sb_attempts": 95, "cs": 25,
                "cera": 3.85, "k_rate_with": 0.235, "bb_rate_with": 0.078,
            },
            "outfield_arms": [
                {"player_id": "f014", "name": "Harrison Bader", "team": "NYY", "position": "LF",
                 "max_throw": 87.0, "avg_throw": 83.5, "accuracy": 0.75, "of_assists": 3,
                 "of_assist_opps": 20, "arm_runs": 0.8, "runners_held": 0.30},
                {"player_id": "f015", "name": "Aaron Judge", "team": "NYY", "position": "CF",
                 "max_throw": 93.0, "avg_throw": 89.0, "accuracy": 0.72, "of_assists": 3,
                 "of_assist_opps": 28, "arm_runs": -0.5, "runners_held": 0.28},
                {"player_id": "f016", "name": "Juan Soto", "team": "NYY", "position": "RF",
                 "max_throw": 86.0, "avg_throw": 82.0, "accuracy": 0.70, "of_assists": 2,
                 "of_assist_opps": 32, "arm_runs": -2.5, "runners_held": 0.22},
            ],
        },
    }

    # Load all data
    for team_code, team_data in teams_data.items():
        fielder_ids = []
        for fd in team_data["fielders"]:
            engine.load_fielder(fd)
            fielder_ids.append(fd["player_id"])
        engine.load_catcher_framing(team_data["catcher"])
        for arm in team_data["outfield_arms"]:
            engine.load_outfield_arm(arm)
        engine.build_team_defense(team_code, fielder_ids, team_data["catcher"]["player_id"])

    # ---- 1. Position WAR Breakdown ----
    print("\n" + "=" * 50)
    print("  1. POSITION-BY-POSITION DEFENSIVE WAR")
    print("=" * 50)
    for team in ["LAD", "NYY"]:
        war = engine.position_war_breakdown(team)
        print(f"\n  {team} — Total dWAR: {war['total_defensive_war']}")
        print(f"    Strongest: {war['strongest_position']}, Weakest: {war['weakest_position']}")
        for pos, data in war["positions"].items():
            print(f"    {pos:>3s}: {data['player']:20s} DRS={data['drs']:+3d}  "
                  f"OAA={data['oaa']:+3d}  dWAR={data['d_war']:+5.2f}")

    # ---- 2. Team DER ----
    print("\n" + "=" * 50)
    print("  2. TEAM DEFENSIVE EFFICIENCY (DER)")
    print("=" * 50)
    for team in ["LAD", "NYY"]:
        der = engine.compute_team_der(team)
        print(f"  {team}: DER={der['der']:.3f} (Lg Avg={der['league_avg']:.3f}), "
              f"Diff={der['difference']:+.3f}, Rank: {der['rank_estimate']}")

    # ---- 3. Catcher Framing ----
    print("\n" + "=" * 50)
    print("  3. CATCHER FRAMING ANALYSIS")
    print("=" * 50)
    for cid, name in [("c001", "Will Smith"), ("c010", "Jose Trevino")]:
        cf = engine.analyze_catcher_framing(cid)
        print(f"\n  {cf['player_name']} ({cf['team']}):")
        fr = cf["framing"]
        print(f"    Framing Runs: {fr['framing_runs']:+.1f} "
              f"({fr['framing_runs_per_1000']:+.2f}/1000)")
        print(f"    Strikes Gained/Lost: +{fr['strikes_gained']}/-{fr['strikes_lost']} "
              f"(Net: {fr['net_strikes']:+d})")
        print(f"    ERA Impact: {fr['era_impact']:+.3f}")
        bl = cf["blocking"]
        print(f"    Blocking: {bl['block_rate']:.1f}% "
              f"(WP={bl['wild_pitches']}, PB={bl['passed_balls']})")
        th = cf["throwing"]
        print(f"    CS%: {th['cs_pct']:.1f}%, Pop Time: {th['pop_time']:.3f}s")
        rat = cf["overall_rating"]
        print(f"    Overall: {rat['overall_score']:.1f} ({rat['tier']})")

    # ---- 4. Outfield Arms ----
    print("\n" + "=" * 50)
    print("  4. OUTFIELD ARM ANALYSIS")
    print("=" * 50)
    for team in ["LAD", "NYY"]:
        arms = engine.analyze_outfield_arms(team)
        print(f"\n  {team} — Arm Rating: {arms['arm_rating']} "
              f"(Total ARM Runs: {arms['total_arm_runs_saved']:+.1f})")
        for pos, data in arms["outfield_arms"].items():
            print(f"    {pos}: {data['player']:18s} "
                  f"Throw={data['max_throw_mph']:.0f}mph, "
                  f"Acc={data['accuracy_pct']:.0f}%, "
                  f"Assists={data['assists']}, "
                  f"ARM={data['arm_runs_saved']:+.1f}")

    # ---- 5. Infield Range ----
    print("\n" + "=" * 50)
    print("  5. INFIELD RANGE METRICS")
    print("=" * 50)
    for team in ["LAD", "NYY"]:
        rng = engine.analyze_infield_range(team)
        print(f"\n  {team} — Avg GB Conversion: {rng['avg_gb_conversion']:.3f} "
              f"(Lg: {rng['league_avg_gb_conversion']:.3f}), "
              f"Adv: {rng['advantage']:+.3f}")
        for pos, data in rng["infield_range"].items():
            print(f"    {pos}: {data['player']:18s} "
                  f"GB%={data['gb_conversion']:.3f}, "
                  f"OAA={data['oaa']:+d}, "
                  f"Range={data['range_runs']:+.1f}")

    # ---- 6. Shift Impact ----
    print("\n" + "=" * 50)
    print("  6. SHIFT IMPACT ANALYSIS")
    print("=" * 50)
    for team in ["LAD", "NYY"]:
        shift = engine.analyze_shift_impact(team)
        print(f"  {team}: Shift PAs={shift['total_shift_pa']}, "
              f"Runs Saved={shift['shift_runs_saved']:.1f}, "
              f"Post-Ban Impact={shift['post_ban_drs_impact']:+.1f} DRS")

    # ---- 7. Error Probability by Conditions ----
    print("\n" + "=" * 50)
    print("  7. ERROR PROBABILITY BY CONDITIONS")
    print("=" * 50)
    conditions = [
        (FieldSurface.NATURAL_GRASS, WeatherCondition.CLEAR),
        (FieldSurface.NATURAL_GRASS, WeatherCondition.RAIN),
        (FieldSurface.ARTIFICIAL_TURF, WeatherCondition.DOME),
        (FieldSurface.NATURAL_GRASS, WeatherCondition.WIND_STRONG),
    ]
    for surface, weather in conditions:
        ep = engine.compute_error_probability("LAD", surface, weather)
        print(f"  LAD on {surface.value}/{weather.value}: "
              f"Errors/Game={ep['expected_errors_per_game']:.2f}, "
              f"Run Impact={ep['impact_on_runs']:+.3f}")

    # ---- 8. Defense-Adjusted ERA ----
    print("\n" + "=" * 50)
    print("  8. DEFENSE-ADJUSTED ERA")
    print("=" * 50)
    pitchers = [
        ("Clayton Kershaw", 3.45, "LAD"),
        ("Gerrit Cole", 3.20, "NYY"),
        ("Generic Pitcher", 4.50, "LAD"),
        ("Generic Pitcher", 4.50, "NYY"),
    ]
    for name, era, team in pitchers:
        adj = engine.compute_defense_adjusted_era(era, team)
        print(f"  {name:20s} ({team}): ERA={adj['raw_era']:.2f} → "
              f"dERA={adj['adjusted_era']:.2f} ({adj['total_adjustment']:+.3f})")
        print(f"    {adj['interpretation']}")

    # ---- 9. Game Prediction Adjustment ----
    print("\n" + "=" * 50)
    print("  9. GAME DEFENSIVE ADJUSTMENT")
    print("=" * 50)
    adj = engine.compute_game_adjustment(
        "LAD", "NYY",
        surface=FieldSurface.NATURAL_GRASS,
        weather=WeatherCondition.CLEAR,
    )
    result = engine.adjustment_to_dict(adj)
    print(f"\n  {result['home_team']} vs {result['away_team']}:")
    print(f"    Net Defensive Edge: {result['defensive_edge']['net_edge_runs']:+.3f} runs")
    print(f"    Home Adj: {result['defensive_edge']['home_adjustment']:+.3f}")
    print(f"    Away Adj: {result['defensive_edge']['away_adjustment']:+.3f}")
    print(f"    Framing Edge: {result['defensive_edge']['framing_edge']:+.3f}")
    print(f"    Arm Edge: {result['defensive_edge']['arm_edge']:+.3f}")
    print(f"    Range Edge: {result['defensive_edge']['range_edge']:+.3f}")
    print(f"    Confidence: {result['confidence']:.3f}")
    print(f"    → {result['recommendation']}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — All systems operational")
    print("=" * 70)


if __name__ == "__main__":
    demo()
