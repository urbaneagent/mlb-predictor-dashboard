"""
MLB Predictor - Umpire Tendencies Tracker
Models individual umpire strike zones and their impact on scoring.
Umpire tendencies are one of the most underappreciated edges in MLB betting.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class UmpireProfile:
    """Detailed umpire profile with statistical tendencies."""
    name: str
    umpire_id: str
    experience_years: int
    # Strike zone metrics
    zone_consistency: float  # 0-1, higher = more consistent
    zone_size: str  # wide, average, tight
    zone_width_inches: float  # Relative to rulebook zone
    zone_height_bias: str  # high, neutral, low (expands up or down)
    # Game impact metrics
    avg_total_runs: float  # Avg runs/game when this ump is behind plate
    league_avg_runs: float  # League average for comparison
    runs_over_avg: float  # +/- from league avg
    k_per_9_above_avg: float  # Strikeouts relative to average
    bb_per_9_above_avg: float  # Walks relative to average
    # Scoring tendencies
    over_rate: float  # How often games go OVER the total
    under_rate: float
    home_team_win_rate: float  # Does ump favor home team?
    # Betting impact
    scoring_environment: str  # pitcher, neutral, hitter
    total_lean: str  # over, neutral, under
    recommended_adjustment: float  # +/- runs to add to total


# Curated umpire database with 2024-2025 tendencies
UMPIRE_DATABASE = {
    "angel_hernandez": UmpireProfile(
        "Angel Hernandez", "AH01", 33,
        zone_consistency=0.82, zone_size="wide", zone_width_inches=1.8,
        zone_height_bias="low",
        avg_total_runs=8.1, league_avg_runs=8.8, runs_over_avg=-0.7,
        k_per_9_above_avg=0.8, bb_per_9_above_avg=-0.3,
        over_rate=44.2, under_rate=55.8, home_team_win_rate=53.1,
        scoring_environment="pitcher", total_lean="under",
        recommended_adjustment=-0.5
    ),
    "cb_bucknor": UmpireProfile(
        "CB Bucknor", "CB01", 27,
        zone_consistency=0.78, zone_size="wide", zone_width_inches=2.1,
        zone_height_bias="neutral",
        avg_total_runs=7.9, league_avg_runs=8.8, runs_over_avg=-0.9,
        k_per_9_above_avg=1.2, bb_per_9_above_avg=-0.5,
        over_rate=42.5, under_rate=57.5, home_team_win_rate=51.8,
        scoring_environment="pitcher", total_lean="under",
        recommended_adjustment=-0.7
    ),
    "dan_bellino": UmpireProfile(
        "Dan Bellino", "DB01", 14,
        zone_consistency=0.91, zone_size="average", zone_width_inches=0.3,
        zone_height_bias="neutral",
        avg_total_runs=8.9, league_avg_runs=8.8, runs_over_avg=0.1,
        k_per_9_above_avg=0.0, bb_per_9_above_avg=0.1,
        over_rate=50.5, under_rate=49.5, home_team_win_rate=51.2,
        scoring_environment="neutral", total_lean="neutral",
        recommended_adjustment=0.0
    ),
    "pat_hoberg": UmpireProfile(
        "Pat Hoberg", "PH01", 8,
        zone_consistency=0.96, zone_size="tight", zone_width_inches=-0.5,
        zone_height_bias="neutral",
        avg_total_runs=9.3, league_avg_runs=8.8, runs_over_avg=0.5,
        k_per_9_above_avg=-0.6, bb_per_9_above_avg=0.4,
        over_rate=55.8, under_rate=44.2, home_team_win_rate=50.5,
        scoring_environment="hitter", total_lean="over",
        recommended_adjustment=0.4
    ),
    "marvin_hudson": UmpireProfile(
        "Marvin Hudson", "MH01", 24,
        zone_consistency=0.85, zone_size="wide", zone_width_inches=1.5,
        zone_height_bias="low",
        avg_total_runs=8.2, league_avg_runs=8.8, runs_over_avg=-0.6,
        k_per_9_above_avg=0.5, bb_per_9_above_avg=-0.2,
        over_rate=45.0, under_rate=55.0, home_team_win_rate=52.5,
        scoring_environment="pitcher", total_lean="under",
        recommended_adjustment=-0.4
    ),
    "ron_kulpa": UmpireProfile(
        "Ron Kulpa", "RK01", 23,
        zone_consistency=0.83, zone_size="average", zone_width_inches=0.8,
        zone_height_bias="high",
        avg_total_runs=9.1, league_avg_runs=8.8, runs_over_avg=0.3,
        k_per_9_above_avg=-0.2, bb_per_9_above_avg=0.3,
        over_rate=53.5, under_rate=46.5, home_team_win_rate=50.8,
        scoring_environment="hitter", total_lean="over",
        recommended_adjustment=0.3
    ),
    "lance_barksdale": UmpireProfile(
        "Lance Barksdale", "LB01", 25,
        zone_consistency=0.80, zone_size="wide", zone_width_inches=2.0,
        zone_height_bias="low",
        avg_total_runs=7.8, league_avg_runs=8.8, runs_over_avg=-1.0,
        k_per_9_above_avg=1.0, bb_per_9_above_avg=-0.6,
        over_rate=41.0, under_rate=59.0, home_team_win_rate=52.0,
        scoring_environment="pitcher", total_lean="under",
        recommended_adjustment=-0.8
    ),
    "laz_diaz": UmpireProfile(
        "Laz Diaz", "LD01", 24,
        zone_consistency=0.79, zone_size="tight", zone_width_inches=-0.8,
        zone_height_bias="high",
        avg_total_runs=9.5, league_avg_runs=8.8, runs_over_avg=0.7,
        k_per_9_above_avg=-0.8, bb_per_9_above_avg=0.6,
        over_rate=57.2, under_rate=42.8, home_team_win_rate=49.5,
        scoring_environment="hitter", total_lean="over",
        recommended_adjustment=0.6
    ),
    "jim_wolf": UmpireProfile(
        "Jim Wolf", "JW01", 22,
        zone_consistency=0.92, zone_size="average", zone_width_inches=0.2,
        zone_height_bias="neutral",
        avg_total_runs=8.7, league_avg_runs=8.8, runs_over_avg=-0.1,
        k_per_9_above_avg=0.1, bb_per_9_above_avg=0.0,
        over_rate=49.0, under_rate=51.0, home_team_win_rate=51.0,
        scoring_environment="neutral", total_lean="neutral",
        recommended_adjustment=0.0
    ),
    "chad_fairchild": UmpireProfile(
        "Chad Fairchild", "CF01", 12,
        zone_consistency=0.88, zone_size="average", zone_width_inches=0.5,
        zone_height_bias="neutral",
        avg_total_runs=8.6, league_avg_runs=8.8, runs_over_avg=-0.2,
        k_per_9_above_avg=0.3, bb_per_9_above_avg=-0.1,
        over_rate=48.0, under_rate=52.0, home_team_win_rate=51.5,
        scoring_environment="neutral", total_lean="neutral",
        recommended_adjustment=-0.1
    ),
}


class UmpireTendenciesTracker:
    """Tracks and exploits umpire tendencies for betting edges."""

    def __init__(self):
        self.umpires = UMPIRE_DATABASE

    def get_umpire_analysis(self, umpire_name: str) -> dict:
        """Get full umpire analysis with betting recommendations."""
        key = umpire_name.lower().replace(' ', '_').replace("'", "")
        ump = self.umpires.get(key)

        if not ump:
            # Fuzzy match
            for k, v in self.umpires.items():
                if umpire_name.lower() in v.name.lower():
                    ump = v
                    break

        if not ump:
            return {"error": f"Umpire '{umpire_name}' not found", "available": list(self.umpires.keys())}

        profile = asdict(ump)

        # Generate betting recommendations
        recs = []
        if ump.runs_over_avg > 0.3:
            recs.append(f"🔴 OVER lean: {ump.name} games average {ump.runs_over_avg:+.1f} runs vs league avg. Look at OVER.")
        elif ump.runs_over_avg < -0.3:
            recs.append(f"🔵 UNDER lean: {ump.name} games average {ump.runs_over_avg:+.1f} runs vs league avg. Look at UNDER.")

        if ump.zone_size == "wide":
            recs.append(f"📏 Wide zone: Expanded zone benefits pitchers. K's up, walks down. Favors aces.")
        elif ump.zone_size == "tight":
            recs.append(f"📏 Tight zone: Compressed zone benefits hitters. More walks, fewer K's. Favors patient lineups.")

        if ump.zone_consistency < 0.85:
            recs.append(f"⚠️ Inconsistent zone ({ump.zone_consistency:.0%}). Wild card — anything can happen.")
        elif ump.zone_consistency >= 0.93:
            recs.append(f"✅ Highly consistent zone ({ump.zone_consistency:.0%}). Predictable game flow.")

        if abs(ump.home_team_win_rate - 50) > 2:
            bias = "home" if ump.home_team_win_rate > 52 else "away"
            recs.append(f"🏠 Home bias: {ump.name} has {ump.home_team_win_rate:.1f}% home team win rate. Slight {bias} lean.")

        profile["betting_recommendations"] = recs
        profile["total_adjustment"] = ump.recommended_adjustment

        return profile

    def get_game_adjustment(self, umpire_name: str, game_total: float) -> dict:
        """Get umpire-adjusted game total for a specific game."""
        key = umpire_name.lower().replace(' ', '_').replace("'", "")
        ump = self.umpires.get(key)

        if not ump:
            for k, v in self.umpires.items():
                if umpire_name.lower() in v.name.lower():
                    ump = v
                    break

        if not ump:
            return {"adjusted_total": game_total, "adjustment": 0, "note": "Umpire not found"}

        adjusted = game_total + ump.recommended_adjustment

        return {
            "umpire": ump.name,
            "posted_total": game_total,
            "umpire_adjustment": ump.recommended_adjustment,
            "adjusted_total": round(adjusted, 1),
            "scoring_environment": ump.scoring_environment,
            "lean": ump.total_lean,
            "over_rate": ump.over_rate,
            "under_rate": ump.under_rate,
            "edge": "OVER" if adjusted > game_total + 0.3 else "UNDER" if adjusted < game_total - 0.3 else "NEUTRAL"
        }

    def get_leaderboard(self, sort_by: str = "runs_over_avg") -> list:
        """Get umpire leaderboard sorted by various metrics."""
        umps = list(self.umpires.values())

        if sort_by == "runs_over_avg":
            umps.sort(key=lambda u: u.runs_over_avg, reverse=True)
        elif sort_by == "consistency":
            umps.sort(key=lambda u: u.zone_consistency, reverse=True)
        elif sort_by == "over_rate":
            umps.sort(key=lambda u: u.over_rate, reverse=True)

        return [{
            "name": u.name,
            "zone_size": u.zone_size,
            "consistency": f"{u.zone_consistency:.0%}",
            "runs_over_avg": f"{u.runs_over_avg:+.1f}",
            "over_rate": f"{u.over_rate:.1f}%",
            "scoring_environment": u.scoring_environment,
            "adjustment": f"{u.recommended_adjustment:+.1f}"
        } for u in umps]


def create_umpire_routes(app, tracker: UmpireTendenciesTracker):
    """Create FastAPI routes."""

    @app.get("/api/v1/umpires/{name}")
    async def get_umpire(name: str):
        return tracker.get_umpire_analysis(name)

    @app.get("/api/v1/umpires/{name}/adjust")
    async def adjust_total(name: str, total: float):
        return tracker.get_game_adjustment(name, total)

    @app.get("/api/v1/umpires/leaderboard")
    async def leaderboard(sort_by: str = "runs_over_avg"):
        return tracker.get_leaderboard(sort_by)

    return tracker
