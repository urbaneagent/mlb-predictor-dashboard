"""
MLB Predictor - Umpire Bias Engine
Analyzes umpire tendencies for balls/strikes, ejections, and
their impact on game outcomes. Integrates with prediction models
to adjust win probabilities based on home plate umpire assignment.
"""
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UmpireProfile:
    """Comprehensive umpire tendency profile."""
    name: str
    umpire_id: str
    games_behind_plate: int = 0
    # Strike zone tendencies
    avg_called_strikes_per_game: float = 0.0
    avg_called_balls_per_game: float = 0.0
    strike_rate: float = 0.0  # % of borderline pitches called strikes
    # Zone bias
    high_strike_rate: float = 0.0    # % above zone called strikes
    low_strike_rate: float = 0.0     # % below zone called strikes
    inside_strike_rate: float = 0.0  # % inside called strikes
    outside_strike_rate: float = 0.0 # % outside called strikes
    # Game impact
    avg_runs_per_game: float = 0.0
    over_total_rate: float = 0.0  # % games going over total
    under_total_rate: float = 0.0
    home_win_rate: float = 0.0
    away_win_rate: float = 0.0
    # Consistency
    consistency_score: float = 0.0  # 0-100, higher = more consistent
    controversy_score: float = 0.0  # 0-100, higher = more controversial
    # Pitcher/hitter impact
    pitcher_era_impact: float = 0.0   # +/- ERA when this ump is behind plate
    hitter_avg_impact: float = 0.0    # +/- batting avg impact
    strikeout_rate_impact: float = 0.0
    walk_rate_impact: float = 0.0
    ejection_rate: float = 0.0  # ejections per 100 games


# ─── UMPIRE DATABASE (2024-2025 based on historical trends) ──

UMPIRE_DATABASE = {
    "angel_hernandez": UmpireProfile(
        name="Angel Hernandez", umpire_id="AH01",
        games_behind_plate=320,
        avg_called_strikes_per_game=32.1,
        avg_called_balls_per_game=41.3,
        strike_rate=0.28,
        high_strike_rate=0.15, low_strike_rate=0.22,
        inside_strike_rate=0.31, outside_strike_rate=0.19,
        avg_runs_per_game=9.2,
        over_total_rate=0.54, under_total_rate=0.46,
        home_win_rate=0.51, away_win_rate=0.49,
        consistency_score=42, controversy_score=85,
        pitcher_era_impact=0.35, hitter_avg_impact=0.008,
        strikeout_rate_impact=-0.012, walk_rate_impact=0.015,
        ejection_rate=3.2,
    ),
    "pat_hoberg": UmpireProfile(
        name="Pat Hoberg", umpire_id="PH01",
        games_behind_plate=280,
        avg_called_strikes_per_game=34.5,
        avg_called_balls_per_game=38.2,
        strike_rate=0.36,
        high_strike_rate=0.20, low_strike_rate=0.28,
        inside_strike_rate=0.34, outside_strike_rate=0.25,
        avg_runs_per_game=8.1,
        over_total_rate=0.47, under_total_rate=0.53,
        home_win_rate=0.53, away_win_rate=0.47,
        consistency_score=88, controversy_score=15,
        pitcher_era_impact=-0.15, hitter_avg_impact=-0.005,
        strikeout_rate_impact=0.010, walk_rate_impact=-0.008,
        ejection_rate=0.8,
    ),
    "cb_bucknor": UmpireProfile(
        name="CB Bucknor", umpire_id="CB01",
        games_behind_plate=350,
        avg_called_strikes_per_game=31.8,
        avg_called_balls_per_game=42.1,
        strike_rate=0.26,
        high_strike_rate=0.12, low_strike_rate=0.20,
        inside_strike_rate=0.28, outside_strike_rate=0.17,
        avg_runs_per_game=9.5,
        over_total_rate=0.56, under_total_rate=0.44,
        home_win_rate=0.50, away_win_rate=0.50,
        consistency_score=38, controversy_score=82,
        pitcher_era_impact=0.42, hitter_avg_impact=0.010,
        strikeout_rate_impact=-0.015, walk_rate_impact=0.018,
        ejection_rate=2.8,
    ),
    "lance_barksdale": UmpireProfile(
        name="Lance Barksdale", umpire_id="LB01",
        games_behind_plate=300,
        avg_called_strikes_per_game=33.2,
        avg_called_balls_per_game=39.8,
        strike_rate=0.32,
        high_strike_rate=0.18, low_strike_rate=0.25,
        inside_strike_rate=0.32, outside_strike_rate=0.22,
        avg_runs_per_game=8.6,
        over_total_rate=0.50, under_total_rate=0.50,
        home_win_rate=0.52, away_win_rate=0.48,
        consistency_score=65, controversy_score=45,
        pitcher_era_impact=0.10, hitter_avg_impact=0.003,
        strikeout_rate_impact=-0.005, walk_rate_impact=0.008,
        ejection_rate=1.5,
    ),
    "dan_bellino": UmpireProfile(
        name="Dan Bellino", umpire_id="DB01",
        games_behind_plate=260,
        avg_called_strikes_per_game=35.1,
        avg_called_balls_per_game=37.5,
        strike_rate=0.38,
        high_strike_rate=0.22, low_strike_rate=0.30,
        inside_strike_rate=0.36, outside_strike_rate=0.27,
        avg_runs_per_game=7.8,
        over_total_rate=0.44, under_total_rate=0.56,
        home_win_rate=0.54, away_win_rate=0.46,
        consistency_score=82, controversy_score=22,
        pitcher_era_impact=-0.22, hitter_avg_impact=-0.007,
        strikeout_rate_impact=0.014, walk_rate_impact=-0.010,
        ejection_rate=1.0,
    ),
    "laz_diaz": UmpireProfile(
        name="Laz Diaz", umpire_id="LD01",
        games_behind_plate=340,
        avg_called_strikes_per_game=32.5,
        avg_called_balls_per_game=40.8,
        strike_rate=0.30,
        high_strike_rate=0.16, low_strike_rate=0.23,
        inside_strike_rate=0.30, outside_strike_rate=0.20,
        avg_runs_per_game=9.0,
        over_total_rate=0.53, under_total_rate=0.47,
        home_win_rate=0.51, away_win_rate=0.49,
        consistency_score=52, controversy_score=65,
        pitcher_era_impact=0.25, hitter_avg_impact=0.006,
        strikeout_rate_impact=-0.008, walk_rate_impact=0.012,
        ejection_rate=2.2,
    ),
    "jordan_baker": UmpireProfile(
        name="Jordan Baker", umpire_id="JB01",
        games_behind_plate=240,
        avg_called_strikes_per_game=34.8,
        avg_called_balls_per_game=37.9,
        strike_rate=0.37,
        high_strike_rate=0.21, low_strike_rate=0.29,
        inside_strike_rate=0.35, outside_strike_rate=0.26,
        avg_runs_per_game=7.9,
        over_total_rate=0.45, under_total_rate=0.55,
        home_win_rate=0.53, away_win_rate=0.47,
        consistency_score=80, controversy_score=25,
        pitcher_era_impact=-0.18, hitter_avg_impact=-0.006,
        strikeout_rate_impact=0.012, walk_rate_impact=-0.009,
        ejection_rate=0.9,
    ),
    "jeff_nelson": UmpireProfile(
        name="Jeff Nelson", umpire_id="JN01",
        games_behind_plate=310,
        avg_called_strikes_per_game=33.0,
        avg_called_balls_per_game=40.2,
        strike_rate=0.31,
        high_strike_rate=0.17, low_strike_rate=0.24,
        inside_strike_rate=0.31, outside_strike_rate=0.21,
        avg_runs_per_game=8.8,
        over_total_rate=0.51, under_total_rate=0.49,
        home_win_rate=0.52, away_win_rate=0.48,
        consistency_score=60, controversy_score=50,
        pitcher_era_impact=0.15, hitter_avg_impact=0.004,
        strikeout_rate_impact=-0.006, walk_rate_impact=0.010,
        ejection_rate=1.8,
    ),
}


class UmpireBiasEngine:
    """
    Analyzes umpire tendencies and their impact on game predictions.
    """

    def __init__(self):
        self.umpires = UMPIRE_DATABASE

    def get_umpire(self, name_key: str) -> Optional[UmpireProfile]:
        """Get umpire profile by key or partial name match."""
        # Direct lookup
        if name_key in self.umpires:
            return self.umpires[name_key]

        # Partial name search
        name_lower = name_key.lower().replace(" ", "_")
        for key, ump in self.umpires.items():
            if name_lower in key or name_lower in ump.name.lower():
                return ump
        return None

    def calculate_game_adjustments(self, umpire_key: str,
                                    home_pitcher_type: str = "mixed",
                                    away_pitcher_type: str = "mixed") -> dict:
        """
        Calculate adjustments to apply to game predictions
        based on umpire assignment.

        pitcher_type: "strikeout", "groundball", "flyball", "mixed"
        """
        ump = self.get_umpire(umpire_key)
        if not ump:
            return {"adjustments": None, "error": "Umpire not found"}

        # Base adjustments
        run_adjustment = (ump.avg_runs_per_game - 8.5) / 8.5
        home_adjustment = (ump.home_win_rate - 0.54) * 100  # vs. league avg

        # Pitcher-specific adjustments
        pitcher_adj = 0
        if home_pitcher_type == "strikeout":
            # Strikeout pitchers benefit from generous strike zones
            pitcher_adj = ump.strikeout_rate_impact * 10
        elif home_pitcher_type == "groundball":
            # Groundball pitchers less affected
            pitcher_adj = ump.strikeout_rate_impact * 3

        return {
            "umpire": ump.name,
            "consistency_score": ump.consistency_score,
            "run_total_adjustment": round(run_adjustment * 100, 1),  # % change
            "over_lean": ump.over_total_rate > 0.52,
            "under_lean": ump.under_total_rate > 0.52,
            "home_win_adjustment": round(home_adjustment, 2),
            "pitcher_era_impact": ump.pitcher_era_impact,
            "walk_rate_impact": ump.walk_rate_impact,
            "strikeout_rate_impact": ump.strikeout_rate_impact,
            "pitcher_advantage_adjustment": round(pitcher_adj, 3),
            "ejection_risk": "HIGH" if ump.ejection_rate > 2.0 else (
                "MEDIUM" if ump.ejection_rate > 1.0 else "LOW"
            ),
            "zone_classification": self._classify_zone(ump),
            "betting_edges": self._find_betting_edges(ump),
        }

    def _classify_zone(self, ump: UmpireProfile) -> str:
        """Classify the umpire's strike zone tendency."""
        if ump.strike_rate >= 0.35:
            return "LARGE (pitcher-friendly)"
        elif ump.strike_rate >= 0.30:
            return "AVERAGE"
        else:
            return "SMALL (hitter-friendly)"

    def _find_betting_edges(self, ump: UmpireProfile) -> list:
        """Find potential betting edges based on umpire tendencies."""
        edges = []

        if ump.over_total_rate >= 0.54:
            edges.append({
                "type": "OVER",
                "strength": "STRONG" if ump.over_total_rate >= 0.56 else "MODERATE",
                "historical_rate": f"{ump.over_total_rate * 100:.1f}%",
                "explanation": f"{ump.name} has a small strike zone, leading to "
                               f"more walks and higher-scoring games.",
            })

        if ump.under_total_rate >= 0.54:
            edges.append({
                "type": "UNDER",
                "strength": "STRONG" if ump.under_total_rate >= 0.56 else "MODERATE",
                "historical_rate": f"{ump.under_total_rate * 100:.1f}%",
                "explanation": f"{ump.name} has a generous strike zone, leading to "
                               f"more strikeouts and lower-scoring games.",
            })

        if ump.home_win_rate >= 0.55:
            edges.append({
                "type": "HOME_FAVORITE",
                "strength": "MODERATE",
                "historical_rate": f"{ump.home_win_rate * 100:.1f}%",
                "explanation": f"{ump.name} shows notable home team advantage.",
            })

        if ump.consistency_score < 50:
            edges.append({
                "type": "VOLATILITY",
                "strength": "MODERATE",
                "historical_rate": f"Consistency: {ump.consistency_score}/100",
                "explanation": f"Inconsistent zone creates unpredictable outcomes. "
                               f"Avoid heavy bets.",
            })

        return edges

    def rank_umpires_for_overs(self) -> list:
        """Rank umpires by over tendency (useful for total bets)."""
        ranked = sorted(
            self.umpires.values(),
            key=lambda u: u.over_total_rate,
            reverse=True
        )
        return [
            {
                "name": u.name,
                "over_rate": f"{u.over_total_rate * 100:.1f}%",
                "avg_runs": u.avg_runs_per_game,
                "consistency": u.consistency_score,
            }
            for u in ranked
        ]

    def rank_umpires_for_unders(self) -> list:
        """Rank umpires by under tendency."""
        ranked = sorted(
            self.umpires.values(),
            key=lambda u: u.under_total_rate,
            reverse=True
        )
        return [
            {
                "name": u.name,
                "under_rate": f"{u.under_total_rate * 100:.1f}%",
                "avg_runs": u.avg_runs_per_game,
                "consistency": u.consistency_score,
            }
            for u in ranked
        ]

    def get_umpire_report(self, umpire_key: str) -> str:
        """Generate a detailed umpire scouting report."""
        ump = self.get_umpire(umpire_key)
        if not ump:
            return f"Umpire '{umpire_key}' not found."

        zone = self._classify_zone(ump)
        edges = self._find_betting_edges(ump)

        report = f"""
╔══════════════════════════════════════════════════╗
║        UMPIRE SCOUTING REPORT                    ║
╚══════════════════════════════════════════════════╝

  👤 {ump.name} ({ump.umpire_id})
  📊 {ump.games_behind_plate} games behind plate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                STRIKE ZONE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Zone Size:         {zone}
  Strike Rate:       {ump.strike_rate * 100:.1f}%
  Strikes/Game:      {ump.avg_called_strikes_per_game:.1f}
  Balls/Game:        {ump.avg_called_balls_per_game:.1f}

  Zone Edges:
    High:   {ump.high_strike_rate * 100:.1f}% called strikes
    Low:    {ump.low_strike_rate * 100:.1f}% called strikes
    Inside: {ump.inside_strike_rate * 100:.1f}% called strikes
    Outside:{ump.outside_strike_rate * 100:.1f}% called strikes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              GAME IMPACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Avg Runs/Game:     {ump.avg_runs_per_game:.1f}
  Over Rate:         {ump.over_total_rate * 100:.1f}%
  Under Rate:        {ump.under_total_rate * 100:.1f}%
  Home Win Rate:     {ump.home_win_rate * 100:.1f}%

  Pitcher Impact:
    ERA Adjustment:  {'+' if ump.pitcher_era_impact > 0 else ''}{ump.pitcher_era_impact:.2f}
    K Rate Impact:   {'+' if ump.strikeout_rate_impact > 0 else ''}{ump.strikeout_rate_impact:.3f}
    BB Rate Impact:  {'+' if ump.walk_rate_impact > 0 else ''}{ump.walk_rate_impact:.3f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             CONSISTENCY & RISK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Consistency:       {ump.consistency_score}/100
  Controversy:       {ump.controversy_score}/100
  Ejection Rate:     {ump.ejection_rate:.1f} per 100 games

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
             BETTING EDGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        if edges:
            for edge in edges:
                report += f"  🎯 {edge['type']} ({edge['strength']}): "
                report += f"{edge['historical_rate']}\n"
                report += f"     {edge['explanation']}\n\n"
        else:
            report += "  No significant edges detected.\n"

        return report


if __name__ == "__main__":
    engine = UmpireBiasEngine()

    # Umpire report
    print(engine.get_umpire_report("angel_hernandez"))
    print(engine.get_umpire_report("pat_hoberg"))

    # Game adjustments
    adj = engine.calculate_game_adjustments(
        "cb_bucknor",
        home_pitcher_type="strikeout"
    )
    print(f"\n🎯 Game Adjustments (CB Bucknor):")
    print(f"  Run Total Adj: {adj['run_total_adjustment']}%")
    print(f"  Over Lean: {adj['over_lean']}")
    print(f"  Zone: {adj['zone_classification']}")
    print(f"  Ejection Risk: {adj['ejection_risk']}")

    # Rankings
    print("\n📊 Top Umpires for OVERS:")
    for u in engine.rank_umpires_for_overs()[:5]:
        print(f"  {u['name']}: {u['over_rate']} (avg {u['avg_runs']} runs)")
