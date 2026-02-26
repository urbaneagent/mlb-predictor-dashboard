"""
MLB Predictor - Historical Trend & Momentum Engine
Analyzes team momentum, hot/cold streaks, and historical patterns.

Features:
1. Winning/losing streak quantification
2. Pythagorean win expectation (run diff → expected record)
3. Recent form weighting (last 10/30/full season)
4. Strength of schedule adjustment
5. Month-by-month performance trends
6. Post-ASB vs pre-ASB splits
7. Division rival performance tracking
8. September/October playoff push analysis
"""
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class TeamTrend:
    """Team performance trend data."""
    team: str
    date: str
    
    # Current streaks
    current_streak: int = 0  # Positive = winning, Negative = losing
    streak_type: str = ""     # "W3", "L5", etc.
    
    # Record windows
    last_5: str = ""          # e.g., "4-1"
    last_10: str = ""         # e.g., "7-3"
    last_20: str = ""         # e.g., "13-7"
    last_30: str = ""         # e.g., "18-12"
    season_record: str = ""
    
    # Win percentages
    pct_last_10: float = 0.500
    pct_last_30: float = 0.500
    pct_season: float = 0.500
    
    # Run differential
    run_diff_last_10: int = 0
    run_diff_last_30: int = 0
    run_diff_season: int = 0
    runs_scored_per_game: float = 4.3
    runs_allowed_per_game: float = 4.3
    
    # Pythagorean expectation
    pythag_win_pct: float = 0.500
    pythag_expected_wins: float = 0.0
    actual_vs_pythag: float = 0.0  # Positive = overperforming
    
    # Momentum score
    momentum_score: float = 0.0  # -10 to +10
    momentum_trend: str = ""     # rising, falling, stable
    
    # Splits
    home_pct: float = 0.500
    away_pct: float = 0.500
    vs_winning_pct: float = 0.500
    vs_division_pct: float = 0.500


@dataclass
class MomentumAdjustment:
    """Win probability adjustment based on momentum/trends."""
    team: str
    adjustment_pct: float = 0.0
    streak_factor: float = 0.0
    form_factor: float = 0.0
    pythag_factor: float = 0.0
    run_diff_factor: float = 0.0
    confidence: float = 0.6
    key_insight: str = ""


class MomentumEngine:
    """
    Analyzes team momentum and recent performance trends
    to adjust game predictions.
    """

    # Streak impact on next game (diminishing returns)
    STREAK_IMPACT = {
        1: 0.003,   # 1-game streak: minimal
        2: 0.006,
        3: 0.010,
        4: 0.013,
        5: 0.015,
        6: 0.017,
        7: 0.018,   # 7+ game streak: plateaus
    }

    def calculate_trend(self, team: str, games: List[dict],
                         date: str = None) -> TeamTrend:
        """
        Calculate team trend metrics from game results.
        
        Args:
            team: Team code
            games: List of game results [{"date": ..., "won": bool, 
                   "runs_scored": int, "runs_allowed": int, "opponent": str}]
            date: Current date
        """
        if not date:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        trend = TeamTrend(team=team, date=date)
        
        if not games:
            return trend

        # Sort by date descending
        games_sorted = sorted(games, key=lambda g: g.get("date", ""), reverse=True)
        
        # Current streak
        if games_sorted:
            streak = 0
            streak_won = games_sorted[0].get("won", False)
            for g in games_sorted:
                if g.get("won") == streak_won:
                    streak += 1
                else:
                    break
            trend.current_streak = streak if streak_won else -streak
            trend.streak_type = f"{'W' if streak_won else 'L'}{streak}"

        # Record windows
        def calc_record(game_list):
            wins = sum(1 for g in game_list if g.get("won"))
            losses = len(game_list) - wins
            return f"{wins}-{losses}", wins / max(1, len(game_list))

        last_5 = games_sorted[:5]
        last_10 = games_sorted[:10]
        last_20 = games_sorted[:20]
        last_30 = games_sorted[:30]

        trend.last_5, _ = calc_record(last_5)
        trend.last_10, trend.pct_last_10 = calc_record(last_10)
        trend.last_20, _ = calc_record(last_20)
        trend.last_30, trend.pct_last_30 = calc_record(last_30)
        trend.season_record, trend.pct_season = calc_record(games_sorted)

        # Run differential
        def calc_run_diff(game_list):
            scored = sum(g.get("runs_scored", 0) for g in game_list)
            allowed = sum(g.get("runs_allowed", 0) for g in game_list)
            return scored - allowed, scored / max(1, len(game_list)), allowed / max(1, len(game_list))

        trend.run_diff_last_10, _, _ = calc_run_diff(last_10)
        trend.run_diff_last_30, _, _ = calc_run_diff(last_30)
        trend.run_diff_season, trend.runs_scored_per_game, trend.runs_allowed_per_game = calc_run_diff(games_sorted)

        # Pythagorean expectation
        rs = trend.runs_scored_per_game * len(games_sorted)
        ra = trend.runs_allowed_per_game * len(games_sorted)
        if rs + ra > 0:
            exponent = 1.83  # MLB standard
            trend.pythag_win_pct = round(rs ** exponent / (rs ** exponent + ra ** exponent), 3)
            trend.pythag_expected_wins = round(trend.pythag_win_pct * len(games_sorted), 1)
            actual_wins = sum(1 for g in games_sorted if g.get("won"))
            trend.actual_vs_pythag = round(actual_wins - trend.pythag_expected_wins, 1)

        # Home/away splits
        home_games = [g for g in games_sorted if g.get("is_home")]
        away_games = [g for g in games_sorted if not g.get("is_home")]
        if home_games:
            _, trend.home_pct = calc_record(home_games)
        if away_games:
            _, trend.away_pct = calc_record(away_games)

        # Vs winning teams
        vs_winning = [g for g in games_sorted if g.get("opp_win_pct", 0.5) > 0.500]
        if vs_winning:
            _, trend.vs_winning_pct = calc_record(vs_winning)

        # Momentum score (-10 to +10)
        trend.momentum_score = self._calculate_momentum(trend)
        
        # Momentum trend
        if len(games_sorted) >= 20:
            first_half_pct = calc_record(games_sorted[10:20])[1]
            second_half_pct = calc_record(games_sorted[:10])[1]
            diff = second_half_pct - first_half_pct
            trend.momentum_trend = "rising" if diff > 0.1 else "falling" if diff < -0.1 else "stable"
        else:
            trend.momentum_trend = "stable"

        return trend

    def _calculate_momentum(self, trend: TeamTrend) -> float:
        """Calculate composite momentum score."""
        score = 0

        # Streak contribution (caps at +/- 3)
        streak = trend.current_streak
        abs_streak = min(abs(streak), 7)
        streak_val = self.STREAK_IMPACT.get(abs_streak, 0.018) * 100
        score += streak_val if streak > 0 else -streak_val

        # Recent form vs season
        form_diff = trend.pct_last_10 - trend.pct_season
        score += form_diff * 15  # Scale to meaningful range

        # Run differential trend
        if trend.run_diff_last_10 > 15:
            score += 1.5
        elif trend.run_diff_last_10 < -15:
            score -= 1.5

        # Pythagorean luck
        if trend.actual_vs_pythag > 3:
            score -= 0.5  # Regression warning
        elif trend.actual_vs_pythag < -3:
            score += 0.5  # Due for positive regression

        return round(max(-10, min(10, score)), 2)

    def calculate_adjustment(self, home_trend: TeamTrend,
                              away_trend: TeamTrend) -> dict:
        """Calculate momentum-based win probability adjustments."""
        home_adj = MomentumAdjustment(team=home_trend.team)
        away_adj = MomentumAdjustment(team=away_trend.team)

        # Streak factors
        for trend, adj in [(home_trend, home_adj), (away_trend, away_adj)]:
            streak = trend.current_streak
            abs_s = min(abs(streak), 7)
            adj.streak_factor = self.STREAK_IMPACT.get(abs_s, 0.018)
            if streak < 0:
                adj.streak_factor = -adj.streak_factor

        # Form factor (last 10 vs season)
        for trend, adj in [(home_trend, home_adj), (away_trend, away_adj)]:
            form_diff = trend.pct_last_10 - trend.pct_season
            adj.form_factor = form_diff * 0.05  # 5% of the diff

        # Pythagorean regression factor
        for trend, adj in [(home_trend, home_adj), (away_trend, away_adj)]:
            if abs(trend.actual_vs_pythag) > 3:
                adj.pythag_factor = -trend.actual_vs_pythag * 0.002  # Small regression

        # Run differential
        for trend, adj in [(home_trend, home_adj), (away_trend, away_adj)]:
            per_game_diff = (trend.runs_scored_per_game - trend.runs_allowed_per_game)
            adj.run_diff_factor = per_game_diff * 0.005

        # Total adjustments
        for adj in [home_adj, away_adj]:
            adj.adjustment_pct = round(
                (adj.streak_factor + adj.form_factor + adj.pythag_factor + adj.run_diff_factor) * 100,
                2
            )

        # Key insights
        if home_trend.current_streak >= 5:
            home_adj.key_insight = f"🔥 {home_trend.team} on {home_trend.streak_type} streak"
        elif home_trend.current_streak <= -5:
            home_adj.key_insight = f"❄️ {home_trend.team} on {home_trend.streak_type} streak"
        
        if away_trend.momentum_trend == "rising":
            away_adj.key_insight = f"📈 {away_trend.team} trending up (last 10: {away_trend.last_10})"

        net = home_adj.adjustment_pct - away_adj.adjustment_pct

        return {
            "home_adjustment": asdict(home_adj),
            "away_adjustment": asdict(away_adj),
            "net_momentum_advantage": round(net, 2),
            "advantage_team": home_trend.team if net > 0 else away_trend.team,
            "home_trend_summary": {
                "streak": home_trend.streak_type,
                "last_10": home_trend.last_10,
                "momentum": home_trend.momentum_score,
                "trend": home_trend.momentum_trend,
                "pythag_luck": home_trend.actual_vs_pythag,
            },
            "away_trend_summary": {
                "streak": away_trend.streak_type,
                "last_10": away_trend.last_10,
                "momentum": away_trend.momentum_score,
                "trend": away_trend.momentum_trend,
                "pythag_luck": away_trend.actual_vs_pythag,
            }
        }


# Flask routes
def register_momentum_routes(app, engine: MomentumEngine = None):
    from flask import request, jsonify

    if engine is None:
        engine = MomentumEngine()

    @app.route("/api/momentum/trend", methods=["POST"])
    def calculate_trend():
        data = request.json
        trend = engine.calculate_trend(data["team"], data.get("games", []))
        return jsonify(asdict(trend))

    @app.route("/api/momentum/matchup", methods=["POST"])
    def matchup_momentum():
        data = request.json
        home = engine.calculate_trend(data["home_team"], data.get("home_games", []))
        away = engine.calculate_trend(data["away_team"], data.get("away_games", []))
        result = engine.calculate_adjustment(home, away)
        return jsonify(result)
