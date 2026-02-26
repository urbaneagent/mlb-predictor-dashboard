"""
MLB Predictor - Clutch Performance & High Leverage Model
Quantifies team/player performance in high-pressure situations.

Features:
1. Clutch scoring (WPA in high-leverage situations)
2. Late-inning performance splits (7th+ innings)
3. Close game performance (1-run games, tied games)
4. Extra innings performance prediction
5. Bullpen lock-down scoring (save/hold conversion)
6. Team comeback probability (based on win expectancy)
7. Run differential in close vs blowout games
"""
import json
import math
import logging
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ClutchProfile:
    """Clutch performance profile for a team."""
    team: str
    season: int = 2025

    # Close game performance
    record_1run_games: str = ""  # e.g., "25-18"
    win_pct_1run: float = 0.500
    record_tied_7th: str = ""
    win_pct_tied_7th: float = 0.500
    record_extra_innings: str = ""
    win_pct_extra_innings: float = 0.500

    # Late inning splits
    era_7th_plus: float = 4.00  # Bullpen ERA in 7th+
    batting_avg_7th_plus: float = 0.250  # Team BA in 7th+
    ops_7th_plus: float = 0.720
    scoring_pct_risp: float = 0.250  # % of RISP runners that score

    # Closer/save metrics
    save_pct: float = 0.85
    blown_saves: int = 0
    holds: int = 0
    hold_pct: float = 0.80
    closer_era: float = 3.00

    # Comeback metrics
    come_from_behind_wins: int = 0
    come_from_behind_pct: float = 0.300
    largest_deficit_overcome: int = 0
    avg_runs_scored_trailing: float = 0.0

    # Clutch composite
    clutch_score: float = 0.0  # -5 to +5 scale
    clutch_grade: str = ""     # A through F


@dataclass
class ClutchAdjustment:
    """Win probability adjustment based on clutch factors."""
    team: str
    base_adjustment: float = 0.0
    close_game_factor: float = 0.0
    late_inning_factor: float = 0.0
    bullpen_lockdown_factor: float = 0.0
    comeback_factor: float = 0.0
    total_adjustment: float = 0.0
    applies_when: str = ""  # "close_game", "all_games", "trailing"
    confidence: float = 0.6
    details: List[str] = field(default_factory=list)


class ClutchPerformanceModel:
    """
    Models clutch performance and high-leverage situation outcomes.
    """

    # League baselines
    LEAGUE_AVG_1RUN_WIN_PCT = 0.500
    LEAGUE_AVG_EXTRA_WIN_PCT = 0.500
    LEAGUE_AVG_SAVE_PCT = 0.88
    LEAGUE_AVG_COMEBACK_PCT = 0.28
    LEAGUE_AVG_RISP_SCORING = 0.260

    def calculate_clutch_score(self, profile: ClutchProfile) -> float:
        """
        Calculate composite clutch score (-5 to +5).
        Positive = clutch, Negative = chokes in pressure.
        """
        components = []

        # Close game performance (weight: 30%)
        close_diff = profile.win_pct_1run - self.LEAGUE_AVG_1RUN_WIN_PCT
        components.append(("1run_games", close_diff * 10, 0.30))

        # Extra innings (weight: 10%)
        extra_diff = profile.win_pct_extra_innings - self.LEAGUE_AVG_EXTRA_WIN_PCT
        components.append(("extra_innings", extra_diff * 10, 0.10))

        # Save conversion (weight: 20%)
        save_diff = profile.save_pct - self.LEAGUE_AVG_SAVE_PCT
        components.append(("saves", save_diff * 15, 0.20))

        # Late-inning offense (weight: 20%)
        late_ops_diff = profile.ops_7th_plus - 0.720
        components.append(("late_offense", late_ops_diff * 10, 0.20))

        # Comeback ability (weight: 20%)
        comeback_diff = profile.come_from_behind_pct - self.LEAGUE_AVG_COMEBACK_PCT
        components.append(("comebacks", comeback_diff * 10, 0.20))

        score = sum(val * weight for _, val, weight in components)
        return round(max(-5, min(5, score)), 2)

    def grade_clutch(self, score: float) -> str:
        """Convert clutch score to letter grade."""
        if score >= 2.5:
            return "A+"
        elif score >= 1.5:
            return "A"
        elif score >= 0.8:
            return "B+"
        elif score >= 0.3:
            return "B"
        elif score >= -0.3:
            return "C"
        elif score >= -1.0:
            return "D"
        return "F"

    def calculate_adjustment(self, home_profile: ClutchProfile,
                              away_profile: ClutchProfile,
                              game_context: dict = None) -> dict:
        """
        Calculate clutch-based adjustments for a game prediction.

        Args:
            home_profile: Home team's clutch profile
            away_profile: Away team's clutch profile
            game_context: Context like expected game closeness
        """
        context = game_context or {}
        expected_close = context.get("expected_run_diff", 2.0) <= 2.0
        expected_trailing = context.get("underdog", False)

        # Calculate scores
        home_score = self.calculate_clutch_score(home_profile)
        away_score = self.calculate_clutch_score(away_profile)
        home_profile.clutch_score = home_score
        home_profile.clutch_grade = self.grade_clutch(home_score)
        away_profile.clutch_score = away_score
        away_profile.clutch_grade = self.grade_clutch(away_score)

        home_adj = ClutchAdjustment(team=home_profile.team)
        away_adj = ClutchAdjustment(team=away_profile.team)

        # Only apply clutch factors when game is expected to be close
        if expected_close:
            # Close game factor
            home_adj.close_game_factor = (home_profile.win_pct_1run - 0.500) * 0.04
            away_adj.close_game_factor = (away_profile.win_pct_1run - 0.500) * 0.04

            # Late inning/bullpen factor
            home_bp = (self.LEAGUE_AVG_SAVE_PCT - (1 - home_profile.save_pct)) * 0.03
            away_bp = (self.LEAGUE_AVG_SAVE_PCT - (1 - away_profile.save_pct)) * 0.03
            home_adj.bullpen_lockdown_factor = home_bp
            away_adj.bullpen_lockdown_factor = away_bp

            # Late inning offense
            home_adj.late_inning_factor = (home_profile.ops_7th_plus - 0.720) * 0.02
            away_adj.late_inning_factor = (away_profile.ops_7th_plus - 0.720) * 0.02

            home_adj.applies_when = "close_game"
            away_adj.applies_when = "close_game"
        else:
            home_adj.applies_when = "blowout_expected"
            away_adj.applies_when = "blowout_expected"

        # Comeback factor for underdogs
        if expected_trailing:
            away_adj.comeback_factor = (away_profile.come_from_behind_pct - 0.28) * 0.03

        # Totals
        home_adj.total_adjustment = round(
            (home_adj.close_game_factor + home_adj.bullpen_lockdown_factor +
             home_adj.late_inning_factor + home_adj.comeback_factor) * 100, 2
        )
        away_adj.total_adjustment = round(
            (away_adj.close_game_factor + away_adj.bullpen_lockdown_factor +
             away_adj.late_inning_factor + away_adj.comeback_factor) * 100, 2
        )

        # Details
        if home_profile.win_pct_1run > 0.55:
            home_adj.details.append(f"Strong in 1-run games ({home_profile.win_pct_1run:.3f})")
        if home_profile.save_pct > 0.90:
            home_adj.details.append(f"Reliable closer ({home_profile.save_pct:.0%} save rate)")
        if away_profile.come_from_behind_pct > 0.35:
            away_adj.details.append(f"Good comeback team ({away_profile.come_from_behind_pct:.0%})")

        net = home_adj.total_adjustment - away_adj.total_adjustment

        return {
            "home_adjustment": asdict(home_adj),
            "away_adjustment": asdict(away_adj),
            "home_clutch_score": home_score,
            "away_clutch_score": away_score,
            "home_clutch_grade": home_profile.clutch_grade,
            "away_clutch_grade": away_profile.clutch_grade,
            "net_clutch_advantage": round(net, 2),
            "advantage_team": home_profile.team if net > 0 else away_profile.team,
            "expected_close_game": expected_close,
            "impact_description": (
                f"Clutch factor: {home_profile.team if net > 0 else away_profile.team} "
                f"has {abs(net):.1f}% edge in high-leverage situations"
            ) if abs(net) > 0.5 else "Clutch factors roughly equal",
        }

    def calculate_comeback_probability(self, team_profile: ClutchProfile,
                                        deficit: int, inning: int) -> dict:
        """Calculate probability of a team coming back from a deficit."""
        # Base comeback rates by deficit (league averages)
        base_rates = {
            1: {1: 0.85, 3: 0.70, 5: 0.55, 7: 0.35, 9: 0.15},
            2: {1: 0.72, 3: 0.55, 5: 0.38, 7: 0.20, 9: 0.08},
            3: {1: 0.58, 3: 0.40, 5: 0.25, 7: 0.12, 9: 0.04},
            4: {1: 0.42, 3: 0.28, 5: 0.15, 7: 0.06, 9: 0.02},
            5: {1: 0.30, 3: 0.18, 5: 0.08, 7: 0.03, 9: 0.01},
        }

        deficit_clamped = min(5, max(1, deficit))
        inning_clamped = min(9, max(1, inning))

        # Interpolate base rate
        rates = base_rates.get(deficit_clamped, base_rates[5])
        if inning_clamped in rates:
            base_rate = rates[inning_clamped]
        else:
            # Interpolate between available innings
            innings = sorted(rates.keys())
            for i in range(len(innings) - 1):
                if innings[i] <= inning_clamped <= innings[i + 1]:
                    t = (inning_clamped - innings[i]) / (innings[i + 1] - innings[i])
                    base_rate = rates[innings[i]] * (1 - t) + rates[innings[i + 1]] * t
                    break
            else:
                base_rate = rates[innings[-1]]

        # Adjust for team clutch ability
        clutch_multiplier = 1 + (team_profile.clutch_score * 0.05)
        comeback_multiplier = team_profile.come_from_behind_pct / max(0.01, self.LEAGUE_AVG_COMEBACK_PCT)

        adjusted_rate = base_rate * clutch_multiplier * (0.7 + 0.3 * comeback_multiplier)
        adjusted_rate = max(0.001, min(0.95, adjusted_rate))

        return {
            "team": team_profile.team,
            "deficit": deficit,
            "inning": inning,
            "base_comeback_probability": round(base_rate, 3),
            "adjusted_probability": round(adjusted_rate, 3),
            "clutch_factor": round(clutch_multiplier, 3),
            "team_clutch_grade": team_profile.clutch_grade,
            "verdict": (
                "Still alive" if adjusted_rate > 0.20
                else "Unlikely but possible" if adjusted_rate > 0.05
                else "Extreme longshot"
            ),
        }


# Flask API routes
def register_clutch_routes(app, model: ClutchPerformanceModel = None):
    from flask import request, jsonify

    if model is None:
        model = ClutchPerformanceModel()

    @app.route("/api/clutch/score", methods=["POST"])
    def clutch_score():
        data = request.json
        profile = ClutchProfile(**data)
        score = model.calculate_clutch_score(profile)
        grade = model.grade_clutch(score)
        return jsonify({"team": profile.team, "clutch_score": score, "grade": grade})

    @app.route("/api/clutch/matchup", methods=["POST"])
    def clutch_matchup():
        data = request.json
        home = ClutchProfile(**data["home"])
        away = ClutchProfile(**data["away"])
        result = model.calculate_adjustment(home, away, data.get("context"))
        return jsonify(result)

    @app.route("/api/clutch/comeback", methods=["POST"])
    def comeback_prob():
        data = request.json
        profile = ClutchProfile(**data["team"])
        result = model.calculate_comeback_probability(
            profile, data["deficit"], data["inning"]
        )
        return jsonify(result)
