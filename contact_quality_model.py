"""
MLB Predictor - Plate Discipline & Contact Quality Model
Advanced batter metrics that predict run scoring better than traditional stats.

Features:
1. Chase rate analysis (O-Swing%, Z-Contact%)
2. Hard hit rate and barrel rate projections
3. Launch angle optimization scoring
4. Expected batting average (xBA) integration
5. Exit velocity percentile rankings
6. Swing decision scoring (chase vs zone rates)
7. Quality of contact aggregation per lineup
8. Platoon split adjustments (L/R pitcher matchup)
"""
import json
import math
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class BatterContactProfile:
    """Contact quality profile for a batter."""
    batter_name: str
    batter_id: str = ""
    team: str = ""
    bats: str = "R"  # R, L, S (switch)

    # Plate discipline metrics
    o_swing_pct: float = 0.30    # % swinging at pitches outside zone (lower = better)
    z_swing_pct: float = 0.70    # % swinging at pitches inside zone (higher = better)
    o_contact_pct: float = 0.60  # Contact rate on pitches outside zone
    z_contact_pct: float = 0.85  # Contact rate on pitches in zone
    swing_pct: float = 0.46      # Overall swing rate
    zone_pct: float = 0.45       # % of pitches in strike zone seen
    first_pitch_swing_pct: float = 0.28  # First pitch swing tendency
    two_strike_approach: float = 0.30    # Chase rate with 2 strikes

    # Contact quality metrics (Statcast)
    avg_exit_velocity: float = 88.0     # mph
    max_exit_velocity: float = 108.0
    hard_hit_pct: float = 0.35          # % of batted balls >= 95mph
    barrel_pct: float = 0.08            # Barrel rate (ideal EV + LA combination)
    sweet_spot_pct: float = 0.33        # Launch angle 8-32 degrees
    avg_launch_angle: float = 12.0      # degrees

    # Expected metrics (Statcast xStats)
    xba: float = 0.260    # Expected batting average
    xslg: float = 0.420   # Expected slugging
    xwoba: float = 0.320  # Expected wOBA
    xiso: float = 0.160   # Expected isolated power

    # Actual stats
    batting_avg: float = 0.260
    obp: float = 0.330
    slg: float = 0.420
    wrc_plus: float = 105  # Weighted Runs Created+
    ops: float = 0.750

    # Platoon splits
    vs_rhp_wrc_plus: float = 105
    vs_lhp_wrc_plus: float = 100

    # Run value
    run_value_per_100: float = 0.5  # Runs above average per 100 PA
    sprint_speed: float = 27.0       # ft/sec


@dataclass
class LineupContactAnalysis:
    """Complete contact quality analysis for a lineup."""
    team: str
    date: str
    lineup: List[BatterContactProfile] = field(default_factory=list)

    # Aggregate metrics
    avg_exit_velocity: float = 0.0
    avg_hard_hit_pct: float = 0.0
    avg_barrel_pct: float = 0.0
    avg_xwoba: float = 0.0
    avg_wrc_plus: float = 0.0
    avg_chase_rate: float = 0.0
    avg_zone_contact: float = 0.0
    lineup_quality_score: float = 0.0  # 0-100

    # Projected output
    projected_runs: float = 0.0
    projected_hits: float = 0.0
    strikeout_vulnerability: float = 0.0  # Higher = more K prone

    # Strengths and weaknesses
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)


class ContactQualityModel:
    """
    Analyzes lineup contact quality and plate discipline
    to improve run scoring projections.
    """

    # League average reference (2024 data)
    LEAGUE_AVG = {
        "exit_velocity": 88.5,
        "hard_hit_pct": 0.37,
        "barrel_pct": 0.075,
        "sweet_spot_pct": 0.34,
        "o_swing_pct": 0.31,
        "z_contact_pct": 0.82,
        "xwoba": 0.312,
        "wrc_plus": 100,
        "chase_rate": 0.285,
    }

    # Contact quality tiers
    QUALITY_TIERS = {
        "elite": {"min_ev": 92, "min_hh": 0.45, "min_barrel": 0.12},
        "above_avg": {"min_ev": 89.5, "min_hh": 0.40, "min_barrel": 0.09},
        "average": {"min_ev": 87, "min_hh": 0.35, "min_barrel": 0.065},
        "below_avg": {"min_ev": 84, "min_hh": 0.28, "min_barrel": 0.04},
    }

    def analyze_lineup(self, batters: List[BatterContactProfile],
                        vs_pitcher_hand: str = "R",
                        date: str = None) -> LineupContactAnalysis:
        """
        Analyze complete lineup contact quality.

        Args:
            batters: List of batter profiles (lineup order)
            vs_pitcher_hand: Pitcher throwing hand (R/L)
            date: Game date
        """
        if not date:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        analysis = LineupContactAnalysis(
            team=batters[0].team if batters else "",
            date=date,
            lineup=batters
        )

        if not batters:
            return analysis

        # Calculate aggregate metrics
        n = len(batters)
        analysis.avg_exit_velocity = sum(b.avg_exit_velocity for b in batters) / n
        analysis.avg_hard_hit_pct = sum(b.hard_hit_pct for b in batters) / n
        analysis.avg_barrel_pct = sum(b.barrel_pct for b in batters) / n
        analysis.avg_xwoba = sum(b.xwoba for b in batters) / n
        analysis.avg_chase_rate = sum(b.o_swing_pct for b in batters) / n
        analysis.avg_zone_contact = sum(b.z_contact_pct for b in batters) / n

        # Platoon-adjusted wRC+
        platoon_wrc = []
        for b in batters:
            if vs_pitcher_hand == "R":
                wrc = b.vs_rhp_wrc_plus
            elif vs_pitcher_hand == "L":
                wrc = b.vs_lhp_wrc_plus
            else:
                wrc = b.wrc_plus
            platoon_wrc.append(wrc)
        analysis.avg_wrc_plus = sum(platoon_wrc) / n

        # Lineup quality score (0-100)
        ev_score = self._normalize(analysis.avg_exit_velocity, 84, 93) * 20
        hh_score = self._normalize(analysis.avg_hard_hit_pct, 0.25, 0.50) * 20
        barrel_score = self._normalize(analysis.avg_barrel_pct, 0.03, 0.15) * 15
        xwoba_score = self._normalize(analysis.avg_xwoba, 0.270, 0.380) * 25
        discipline_score = self._normalize(1 - analysis.avg_chase_rate, 0.60, 0.80) * 10
        wrc_score = self._normalize(analysis.avg_wrc_plus, 80, 130) * 10

        analysis.lineup_quality_score = round(min(100, max(0,
            ev_score + hh_score + barrel_score + xwoba_score + discipline_score + wrc_score
        )), 1)

        # Run projection (based on wRC+ and park factors)
        base_runs = 4.3  # League average
        wrc_factor = analysis.avg_wrc_plus / 100
        contact_factor = 1 + (analysis.avg_hard_hit_pct - self.LEAGUE_AVG["hard_hit_pct"]) * 2
        analysis.projected_runs = round(base_runs * wrc_factor * contact_factor, 1)

        # Strikeout vulnerability
        avg_contact = sum(b.z_contact_pct * b.z_swing_pct for b in batters) / n
        analysis.strikeout_vulnerability = round(1 - avg_contact, 3)

        # Projected hits
        analysis.projected_hits = round(sum(b.xba for b in batters) * 4, 1)  # ~4 AB per batter

        # Strengths and weaknesses
        analysis.strengths = self._identify_strengths(analysis, batters)
        analysis.weaknesses = self._identify_weaknesses(analysis, batters)

        return analysis

    def score_batter(self, batter: BatterContactProfile,
                      vs_pitcher_hand: str = "R") -> dict:
        """Score an individual batter's contact quality."""
        # Contact quality tier
        tier = "below_avg"
        for tier_name, thresholds in sorted(
            self.QUALITY_TIERS.items(),
            key=lambda x: x[1]["min_ev"], reverse=True
        ):
            if (batter.avg_exit_velocity >= thresholds["min_ev"] and
                batter.hard_hit_pct >= thresholds["min_hh"] and
                batter.barrel_pct >= thresholds["min_barrel"]):
                tier = tier_name
                break

        # Discipline grade
        chase_diff = self.LEAGUE_AVG["chase_rate"] - batter.o_swing_pct
        if chase_diff > 0.05:
            discipline_grade = "A"
        elif chase_diff > 0.02:
            discipline_grade = "B"
        elif chase_diff > -0.02:
            discipline_grade = "C"
        else:
            discipline_grade = "D"

        # Platoon adjustment
        if vs_pitcher_hand == "R":
            platoon_wrc = batter.vs_rhp_wrc_plus
        elif vs_pitcher_hand == "L":
            platoon_wrc = batter.vs_lhp_wrc_plus
        else:
            platoon_wrc = batter.wrc_plus

        platoon_advantage = platoon_wrc > batter.wrc_plus

        # Power score (combo of barrel rate, hard hit, and max EV)
        power_score = (
            batter.barrel_pct * 200 +
            batter.hard_hit_pct * 50 +
            (batter.max_exit_velocity - 100) * 2 +
            batter.xiso * 100
        )
        power_score = min(100, max(0, power_score))

        # Speed score
        speed_score = min(100, max(0, (batter.sprint_speed - 24) * 10))

        return {
            "batter_name": batter.batter_name,
            "contact_tier": tier,
            "discipline_grade": discipline_grade,
            "power_score": round(power_score, 1),
            "speed_score": round(speed_score, 1),
            "platoon_wrc_plus": platoon_wrc,
            "platoon_advantage": platoon_advantage,
            "xwoba": batter.xwoba,
            "barrel_pct": batter.barrel_pct,
            "hard_hit_pct": batter.hard_hit_pct,
            "chase_rate": batter.o_swing_pct,
            "run_value_per_100": batter.run_value_per_100,
            "overall_rating": round(
                (power_score * 0.35 + speed_score * 0.10 +
                 (platoon_wrc - 80) * 0.4 +
                 (1 - batter.o_swing_pct) * 50 * 0.15), 1
            ),
        }

    def matchup_projection(self, lineup: List[BatterContactProfile],
                            pitcher_hand: str, pitcher_k_rate: float,
                            pitcher_whiff_rate: float,
                            pitcher_era: float) -> dict:
        """
        Project lineup performance against a specific pitcher type.
        """
        analysis = self.analyze_lineup(lineup, pitcher_hand)

        # Adjust for pitcher quality
        era_factor = 4.00 / max(2.0, pitcher_era)  # Lower ERA = fewer runs
        k_factor = 1 + (pitcher_k_rate - 0.22) * 2  # High K pitcher reduces scoring
        whiff_impact = pitcher_whiff_rate / 0.25  # Relative to league avg whiff

        # Lineup discipline vs pitcher stuff
        lineup_chase = analysis.avg_chase_rate
        if pitcher_whiff_rate > 0.30 and lineup_chase > 0.32:
            # Free-swinging lineup vs high-whiff pitcher = bad matchup
            matchup_factor = 0.85
        elif pitcher_whiff_rate < 0.22 and lineup_chase < 0.28:
            # Disciplined lineup vs low-whiff pitcher = good matchup
            matchup_factor = 1.10
        else:
            matchup_factor = 1.0

        adjusted_runs = analysis.projected_runs * era_factor * matchup_factor / k_factor

        return {
            "team": analysis.team,
            "vs_pitcher_hand": pitcher_hand,
            "lineup_quality_score": analysis.lineup_quality_score,
            "raw_run_projection": analysis.projected_runs,
            "adjusted_run_projection": round(adjusted_runs, 1),
            "era_factor": round(era_factor, 3),
            "k_factor": round(k_factor, 3),
            "matchup_factor": round(matchup_factor, 3),
            "matchup_quality": (
                "favorable" if matchup_factor > 1.05 else
                "unfavorable" if matchup_factor < 0.90 else
                "neutral"
            ),
            "strikeout_risk": round(analysis.strikeout_vulnerability * whiff_impact, 3),
            "power_potential": round(analysis.avg_barrel_pct * 100, 1),
            "strengths": analysis.strengths,
            "weaknesses": analysis.weaknesses,
        }

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))

    def _identify_strengths(self, analysis: LineupContactAnalysis,
                             batters: List[BatterContactProfile]) -> List[str]:
        """Identify lineup strengths."""
        strengths = []

        if analysis.avg_exit_velocity > 89.5:
            strengths.append(f"💪 Elite exit velocity ({analysis.avg_exit_velocity:.1f} mph)")
        if analysis.avg_hard_hit_pct > 0.40:
            strengths.append(f"🔥 High hard-hit rate ({analysis.avg_hard_hit_pct:.0%})")
        if analysis.avg_barrel_pct > 0.10:
            strengths.append(f"💣 Elite barrel rate ({analysis.avg_barrel_pct:.1%})")
        if analysis.avg_chase_rate < 0.28:
            strengths.append(f"👁️ Great plate discipline (chase {analysis.avg_chase_rate:.0%})")
        if analysis.avg_wrc_plus > 110:
            strengths.append(f"⚡ Potent lineup (wRC+ {analysis.avg_wrc_plus:.0f})")

        elite_batters = [b for b in batters if b.xwoba > 0.360]
        if len(elite_batters) >= 3:
            strengths.append(f"🌟 {len(elite_batters)} elite bats in lineup")

        fast_runners = [b for b in batters if b.sprint_speed > 28.5]
        if len(fast_runners) >= 3:
            strengths.append(f"💨 {len(fast_runners)} above-average speed threats")

        return strengths[:4]

    def _identify_weaknesses(self, analysis: LineupContactAnalysis,
                              batters: List[BatterContactProfile]) -> List[str]:
        """Identify lineup weaknesses."""
        weaknesses = []

        if analysis.avg_exit_velocity < 87:
            weaknesses.append(f"Weak contact ({analysis.avg_exit_velocity:.1f} mph avg EV)")
        if analysis.avg_chase_rate > 0.33:
            weaknesses.append(f"Free-swinging (chase {analysis.avg_chase_rate:.0%})")
        if analysis.avg_wrc_plus < 95:
            weaknesses.append(f"Below-avg offense (wRC+ {analysis.avg_wrc_plus:.0f})")
        if analysis.strikeout_vulnerability > 0.25:
            weaknesses.append(f"K-vulnerable ({analysis.strikeout_vulnerability:.0%})")

        weak_spots = [b for b in batters if b.xwoba < 0.280]
        if len(weak_spots) >= 3:
            weaknesses.append(f"⚠️ {len(weak_spots)} weak bats in lineup")

        return weaknesses[:3]


# Flask API routes
def register_contact_routes(app, model: ContactQualityModel = None):
    from flask import request, jsonify

    if model is None:
        model = ContactQualityModel()

    @app.route("/api/contact/lineup-analysis", methods=["POST"])
    def analyze_lineup():
        data = request.json
        batters = [BatterContactProfile(**b) for b in data.get("lineup", [])]
        analysis = model.analyze_lineup(batters, data.get("vs_pitcher_hand", "R"))
        return jsonify(asdict(analysis))

    @app.route("/api/contact/batter-score", methods=["POST"])
    def score_batter():
        data = request.json
        batter = BatterContactProfile(**data.get("batter", {}))
        result = model.score_batter(batter, data.get("vs_pitcher_hand", "R"))
        return jsonify(result)

    @app.route("/api/contact/matchup", methods=["POST"])
    def matchup():
        data = request.json
        batters = [BatterContactProfile(**b) for b in data.get("lineup", [])]
        result = model.matchup_projection(
            batters,
            data.get("pitcher_hand", "R"),
            data.get("pitcher_k_rate", 0.22),
            data.get("pitcher_whiff_rate", 0.25),
            data.get("pitcher_era", 4.00)
        )
        return jsonify(result)
