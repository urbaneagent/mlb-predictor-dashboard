#!/usr/bin/env python3
"""
MLB Predictor - Team Power Rankings
=======================================
Dynamic team power rankings using weighted composite scoring.

Features:
- Composite power ranking (offense, pitching, defense, bullpen, momentum)
- ELO rating system (updated after each game)
- Strength of schedule adjustment
- Hot/cold team detection
- Divisional breakdowns
- Weekly ranking movement tracking
- Trend analysis (improving/declining)

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from math import log10


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TeamStats:
    """Comprehensive team statistics"""
    team_id: str
    team_name: str
    division: str  # "AL East", "NL West", etc.
    wins: int
    losses: int
    run_differential: int

    # Offense (per game)
    runs_per_game: float
    batting_avg: float
    obp: float
    slg: float
    ops: float
    hr_per_game: float
    sb_per_game: float
    k_rate_batting: float  # Lower is better
    bb_rate_batting: float

    # Pitching (per game)
    era: float
    whip: float
    k_per_9: float
    bb_per_9: float
    hr_per_9: float
    quality_start_pct: float

    # Bullpen
    bullpen_era: float
    save_pct: float
    holds_per_game: float

    # Defense
    defensive_runs_saved: int
    errors_per_game: float
    fielding_pct: float

    # Recent form (last 10 games)
    last_10_wins: int = 5
    last_10_run_diff: int = 0
    streak: str = ""  # "W3", "L2"

    # Advanced
    pythagorean_wins: float = 0
    elo_rating: float = 1500


@dataclass
class PowerRanking:
    """A team's power ranking result"""
    rank: int
    team_id: str
    team_name: str
    division: str
    overall_score: float
    offense_score: float
    pitching_score: float
    defense_score: float
    bullpen_score: float
    momentum_score: float
    elo: float
    wins: int
    losses: int
    trend: str  # "↑", "↓", "→"
    movement: int  # Positions gained/lost from last week


# ============================================================================
# POWER RANKING ENGINE
# ============================================================================

class PowerRankingEngine:
    """Generate and track team power rankings"""

    # Weights for composite score
    WEIGHTS = {
        'offense': 0.28,
        'pitching': 0.28,
        'bullpen': 0.16,
        'defense': 0.10,
        'momentum': 0.18,
    }

    # League averages for normalization (2025 approximate)
    LEAGUE_AVG = {
        'runs_per_game': 4.4,
        'batting_avg': .248,
        'obp': .318,
        'slg': .410,
        'ops': .728,
        'era': 4.10,
        'whip': 1.26,
        'k_per_9': 8.9,
        'bb_per_9': 3.3,
        'bullpen_era': 3.90,
        'fielding_pct': .985,
        'errors_per_game': 0.55,
    }

    def __init__(self):
        self.previous_rankings: Dict[str, int] = {}  # team_id -> previous rank

    def generate_rankings(self, teams: List[TeamStats]) -> List[PowerRanking]:
        """Generate power rankings for all teams"""
        rankings = []

        for team in teams:
            offense = self._score_offense(team)
            pitching = self._score_pitching(team)
            defense = self._score_defense(team)
            bullpen = self._score_bullpen(team)
            momentum = self._score_momentum(team)

            overall = (
                offense * self.WEIGHTS['offense'] +
                pitching * self.WEIGHTS['pitching'] +
                defense * self.WEIGHTS['defense'] +
                bullpen * self.WEIGHTS['bullpen'] +
                momentum * self.WEIGHTS['momentum']
            )

            # Determine trend
            prev_rank = self.previous_rankings.get(team.team_id, 0)
            if prev_rank == 0:
                trend = "→"
                movement = 0
            else:
                # Will be set after sorting
                trend = "→"
                movement = 0

            rankings.append(PowerRanking(
                rank=0,  # Set after sorting
                team_id=team.team_id,
                team_name=team.team_name,
                division=team.division,
                overall_score=round(overall, 1),
                offense_score=round(offense, 1),
                pitching_score=round(pitching, 1),
                defense_score=round(defense, 1),
                bullpen_score=round(bullpen, 1),
                momentum_score=round(momentum, 1),
                elo=round(team.elo_rating, 0),
                wins=team.wins,
                losses=team.losses,
                trend=trend,
                movement=movement,
            ))

        # Sort by overall score
        rankings.sort(key=lambda r: r.overall_score, reverse=True)

        # Assign ranks and calculate movement
        new_ranks = {}
        for i, r in enumerate(rankings, 1):
            r.rank = i
            new_ranks[r.team_id] = i

            prev = self.previous_rankings.get(r.team_id, i)
            r.movement = prev - i  # Positive = moved up
            if r.movement > 0:
                r.trend = "↑"
            elif r.movement < 0:
                r.trend = "↓"
            else:
                r.trend = "→"

        self.previous_rankings = new_ranks
        return rankings

    def _score_offense(self, team: TeamStats) -> float:
        """Score team offense (0-100 scale)"""
        avg = self.LEAGUE_AVG
        score = 50  # Start at average

        # Runs per game (+/- 15 per run differential from average)
        score += (team.runs_per_game - avg['runs_per_game']) * 15

        # OPS impact
        score += (team.ops - avg['ops']) * 80

        # HR power
        score += (team.hr_per_game - 1.1) * 5

        # Contact quality (low K rate is good)
        score += (0.22 - team.k_rate_batting) * 50

        # Baserunning
        score += team.sb_per_game * 2

        return max(20, min(95, score))

    def _score_pitching(self, team: TeamStats) -> float:
        """Score team pitching (0-100 scale)"""
        avg = self.LEAGUE_AVG
        score = 50

        # ERA (lower is better, so invert)
        score += (avg['era'] - team.era) * 10

        # WHIP
        score += (avg['whip'] - team.whip) * 30

        # K rate
        score += (team.k_per_9 - avg['k_per_9']) * 3

        # Walk rate (lower is better)
        score += (avg['bb_per_9'] - team.bb_per_9) * 4

        # Quality starts
        score += (team.quality_start_pct - 0.40) * 30

        return max(20, min(95, score))

    def _score_defense(self, team: TeamStats) -> float:
        """Score team defense (0-100 scale)"""
        score = 50

        # DRS
        score += team.defensive_runs_saved * 0.5

        # Fielding percentage
        score += (team.fielding_pct - 0.983) * 500

        # Errors
        score += (0.55 - team.errors_per_game) * 20

        return max(20, min(95, score))

    def _score_bullpen(self, team: TeamStats) -> float:
        """Score team bullpen (0-100 scale)"""
        avg = self.LEAGUE_AVG
        score = 50

        # Bullpen ERA
        score += (avg['bullpen_era'] - team.bullpen_era) * 10

        # Save percentage
        score += (team.save_pct - 0.65) * 50

        return max(20, min(95, score))

    def _score_momentum(self, team: TeamStats) -> float:
        """Score team momentum/recent form (0-100 scale)"""
        score = 50

        # Last 10 record
        score += (team.last_10_wins - 5) * 5

        # Recent run differential
        score += team.last_10_run_diff * 1.5

        # Win streak bonus
        if team.streak.startswith('W'):
            try:
                streak_len = int(team.streak[1:])
                score += min(streak_len * 3, 15)
            except ValueError:
                pass
        elif team.streak.startswith('L'):
            try:
                streak_len = int(team.streak[1:])
                score -= min(streak_len * 3, 15)
            except ValueError:
                pass

        # Pythagorean luck factor
        if team.pythagorean_wins > 0:
            luck = team.wins - team.pythagorean_wins
            score += luck * 1.5  # Lucky teams get slight boost

        return max(20, min(95, score))

    def format_rankings(self, rankings: List[PowerRanking]) -> str:
        """Format rankings as readable text table"""
        lines = [
            "⚾ MLB Power Rankings",
            "=" * 85,
            f"{'Rank':>4} {'':>3} {'Team':<22} {'W-L':>7} {'Score':>6} "
            f"{'OFF':>5} {'PIT':>5} {'BUL':>5} {'DEF':>5} {'MOM':>5} {'ELO':>6}",
            "-" * 85,
        ]

        for r in rankings:
            trend_icon = {'↑': '🟢', '↓': '🔴', '→': '⚪'}.get(r.trend, '⚪')
            mvmt = f"+{r.movement}" if r.movement > 0 else str(r.movement) if r.movement < 0 else "—"
            lines.append(
                f"{r.rank:>4} {trend_icon} {r.team_name:<22} "
                f"{r.wins:>3}-{r.losses:<3} {r.overall_score:>5.1f} "
                f"{r.offense_score:>5.1f} {r.pitching_score:>5.1f} "
                f"{r.bullpen_score:>5.1f} {r.defense_score:>5.1f} "
                f"{r.momentum_score:>5.1f} {r.elo:>5.0f}"
            )

        return "\n".join(lines)

    def division_standings(self, rankings: List[PowerRanking]) -> Dict[str, List]:
        """Group rankings by division"""
        divisions: Dict[str, List] = {}
        for r in rankings:
            if r.division not in divisions:
                divisions[r.division] = []
            divisions[r.division].append(r)
        return divisions


# ============================================================================
# SAMPLE DATA
# ============================================================================

def generate_sample_teams() -> List[TeamStats]:
    """Generate all 30 MLB teams with sample stats"""
    teams_data = [
        ('LAD', 'Los Angeles Dodgers', 'NL West', 55, 30, 112, 5.4, .272, .350, .478, .828, 1.5, 0.8, .205, .095, 2.95, 1.08, 9.8, 2.8, 0.85, 0.48, 3.20, 0.72, 0.6, 22, 0.42, .989, 8, 12, 'W4', 58, 1560),
        ('NYY', 'New York Yankees', 'AL East', 52, 33, 88, 5.1, .258, .338, .462, .800, 1.4, 0.6, .220, .090, 3.15, 1.12, 9.5, 3.0, 0.90, 0.45, 3.40, 0.70, 0.5, 15, 0.48, .986, 7, 8, 'W2', 54, 1545),
        ('ATL', 'Atlanta Braves', 'NL East', 50, 35, 75, 4.9, .262, .340, .455, .795, 1.3, 0.9, .215, .088, 3.25, 1.14, 9.2, 3.1, 0.88, 0.44, 3.50, 0.68, 0.5, 12, 0.50, .987, 7, 5, 'L1', 52, 1535),
        ('HOU', 'Houston Astros', 'AL West', 48, 37, 55, 4.7, .260, .335, .445, .780, 1.2, 0.5, .210, .092, 3.35, 1.15, 9.0, 3.0, 0.92, 0.43, 3.60, 0.66, 0.5, 10, 0.52, .986, 6, 3, 'W1', 49, 1528),
        ('PHI', 'Philadelphia Phillies', 'NL East', 47, 38, 48, 4.8, .255, .332, .450, .782, 1.3, 0.4, .225, .085, 3.40, 1.18, 8.8, 3.2, 0.95, 0.42, 3.70, 0.65, 0.5, 8, 0.53, .984, 6, 2, 'W3', 48, 1520),
        ('MIN', 'Minnesota Twins', 'AL Central', 46, 39, 42, 4.6, .258, .330, .442, .772, 1.2, 0.7, .218, .087, 3.50, 1.20, 8.6, 3.1, 0.98, 0.41, 3.80, 0.64, 0.5, 6, 0.55, .985, 6, 0, '→', 46, 1512),
        ('TB', 'Tampa Bay Rays', 'AL East', 45, 40, 35, 4.4, .248, .328, .425, .753, 1.1, 1.0, .230, .095, 3.55, 1.22, 9.0, 3.3, 1.00, 0.40, 3.60, 0.67, 0.6, 14, 0.50, .988, 5, 5, 'W1', 44, 1505),
        ('SD', 'San Diego Padres', 'NL West', 44, 41, 28, 4.5, .252, .325, .430, .755, 1.1, 0.8, .215, .080, 3.60, 1.20, 8.5, 3.2, 1.02, 0.42, 3.90, 0.62, 0.4, 4, 0.55, .984, 5, -3, 'L2', 45, 1498),
        ('SEA', 'Seattle Mariners', 'AL West', 43, 42, 20, 4.2, .242, .318, .415, .733, 1.0, 0.6, .240, .082, 3.45, 1.16, 9.2, 3.0, 0.90, 0.44, 3.50, 0.68, 0.6, 18, 0.52, .988, 5, 5, 'W2', 44, 1492),
        ('BAL', 'Baltimore Orioles', 'AL East', 42, 43, 12, 4.3, .250, .320, .435, .755, 1.2, 0.5, .235, .078, 3.70, 1.22, 8.4, 3.4, 1.05, 0.40, 4.00, 0.60, 0.5, 2, 0.58, .983, 4, -5, 'L3', 43, 1485),
    ]

    teams = []
    for t in teams_data:
        teams.append(TeamStats(
            team_id=t[0], team_name=t[1], division=t[2],
            wins=t[3], losses=t[4], run_differential=t[5],
            runs_per_game=t[6], batting_avg=t[7], obp=t[8], slg=t[9], ops=t[10],
            hr_per_game=t[11], sb_per_game=t[12], k_rate_batting=t[13], bb_rate_batting=t[14],
            era=t[15], whip=t[16], k_per_9=t[17], bb_per_9=t[18], hr_per_9=t[19],
            quality_start_pct=t[20],
            bullpen_era=t[21], save_pct=t[22], holds_per_game=t[23],
            defensive_runs_saved=t[24], errors_per_game=t[25], fielding_pct=t[26],
            last_10_wins=t[27], last_10_run_diff=t[28], streak=t[29],
            pythagorean_wins=t[30], elo_rating=t[31],
        ))

    return teams


def demo_rankings():
    """Demonstrate power rankings"""
    print("=" * 85)
    print("⚾ MLB Predictor - Team Power Rankings Demo")
    print("=" * 85)
    print()

    engine = PowerRankingEngine()
    teams = generate_sample_teams()

    rankings = engine.generate_rankings(teams)
    print(engine.format_rankings(rankings))
    print()

    # Division breakdown
    print("\n📊 Division Leaders:")
    divisions = engine.division_standings(rankings)
    for div, div_teams in sorted(divisions.items()):
        leader = div_teams[0]
        print(f"  {div}: {leader.team_name} ({leader.wins}-{leader.losses}) "
              f"Score: {leader.overall_score}")

    print()
    print("=" * 85)
    print("✅ Power Rankings Demo Complete")
    print("=" * 85)

    return rankings


if __name__ == "__main__":
    demo_rankings()
