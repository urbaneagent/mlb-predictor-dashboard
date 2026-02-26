"""
MLB Predictor - Lineup Impact Model
Quantifies the impact of lineup changes on game predictions.
Rest days, platoon splits, and batting order position matter.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class PlayerImpact:
    """Individual player's impact on team performance."""
    name: str
    position: str
    batting_order: int
    ops: float  # On-base plus slugging
    wrc_plus: int  # Weighted runs created+ (100 = league avg)
    war_pace: float  # WAR projected over 162 games
    vs_rhp: float  # OPS vs right-handed pitchers
    vs_lhp: float  # OPS vs left-handed pitchers
    rest_days: int = 0
    hot_cold: str = ""  # hot (last 7: >.300), cold (<.200), neutral
    lineup_impact_runs: float = 0  # Expected runs above replacement


@dataclass
class LineupAnalysis:
    """Full lineup analysis for a game."""
    team: str
    opponent: str
    lineup: list = field(default_factory=list)
    total_wrc_plus: int = 0
    lineup_strength: str = ""  # elite, strong, average, weak, depleted
    vs_pitcher_advantage: str = ""
    key_absences: list = field(default_factory=list)
    platoon_advantage: bool = False
    expected_runs_adjustment: float = 0
    betting_angle: str = ""


# Lineup position run production weights (per 650 PA season)
BATTING_ORDER_WEIGHTS = {
    1: 1.15,  # Leadoff: high OBP, sets table
    2: 1.20,  # Best hitter often bats 2nd
    3: 1.18,  # Classic cleanup approach, high RBI opportunity
    4: 1.12,  # Run producer
    5: 1.05,
    6: 0.98,
    7: 0.92,
    8: 0.85,
    9: 0.75,  # Weakest spot (or pitcher in NL pre-DH)
}


class LineupImpactModel:
    """Models the impact of daily lineups on game outcomes."""

    def __init__(self):
        self.team_base_wrc = {}  # team -> season wRC+

    def analyze_lineup(self, team: str, lineup: list, opponent_pitcher_hand: str = "R") -> LineupAnalysis:
        """
        Analyze a team's daily lineup.

        lineup: list of dicts with player info
        opponent_pitcher_hand: R or L
        """
        players = []
        total_wrc = 0
        absences = []

        for i, player in enumerate(lineup):
            order = i + 1
            wrc = player.get('wrc_plus', 100)
            total_wrc += wrc

            # Platoon split
            if opponent_pitcher_hand == "R":
                split_ops = player.get('vs_rhp', player.get('ops', .750))
            else:
                split_ops = player.get('vs_lhp', player.get('ops', .750))

            # Hot/cold detection
            last7_avg = player.get('last_7_avg', .250)
            hot_cold = "hot" if last7_avg > .300 else "cold" if last7_avg < .200 else "neutral"

            # Run impact based on batting order position
            position_weight = BATTING_ORDER_WEIGHTS.get(order, 0.85)
            runs_above_replacement = (wrc - 80) / 100 * position_weight * 0.5

            p = PlayerImpact(
                name=player.get('name', ''),
                position=player.get('position', ''),
                batting_order=order,
                ops=player.get('ops', .750),
                wrc_plus=wrc,
                war_pace=player.get('war_pace', 2.0),
                vs_rhp=player.get('vs_rhp', .750),
                vs_lhp=player.get('vs_lhp', .750),
                rest_days=player.get('rest_days', 0),
                hot_cold=hot_cold,
                lineup_impact_runs=round(runs_above_replacement, 2)
            )
            players.append(p)

        avg_wrc = total_wrc / len(lineup) if lineup else 100

        # Lineup strength classification
        if avg_wrc >= 115:
            strength = "elite"
        elif avg_wrc >= 105:
            strength = "strong"
        elif avg_wrc >= 95:
            strength = "average"
        elif avg_wrc >= 85:
            strength = "weak"
        else:
            strength = "depleted"

        # Platoon advantage
        platoon_adv = False
        if opponent_pitcher_hand == "L":
            # Check if lineup stacked with RHB
            rhb_count = sum(1 for p in lineup if p.get('bats', 'R') == 'R')
            platoon_adv = rhb_count >= 6

        elif opponent_pitcher_hand == "R":
            lhb_count = sum(1 for p in lineup if p.get('bats', 'L') == 'L')
            platoon_adv = lhb_count >= 5

        # Expected runs adjustment
        runs_adj = (avg_wrc - 100) / 100 * 1.5  # Scale to runs

        # Hot hitters bonus
        hot_count = len([p for p in players if p.hot_cold == "hot"])
        cold_count = len([p for p in players if p.hot_cold == "cold"])
        runs_adj += hot_count * 0.1 - cold_count * 0.08

        # Betting angle
        angle = ""
        if strength in ["elite", "strong"] and platoon_adv:
            angle = f"🔥 STRONG lineup with platoon advantage vs {opponent_pitcher_hand}HP. Lean team total OVER."
        elif strength == "depleted":
            angle = "⚠️ Depleted lineup. Multiple key absences. Lean UNDER or fade this team."
        elif hot_count >= 3:
            angle = f"🔥 {hot_count} hitters are hot (>.300 last 7 days). Momentum play — lean OVER."
        elif cold_count >= 3:
            angle = f"❄️ {cold_count} hitters are cold (<.200 last 7 days). Lean UNDER."

        return LineupAnalysis(
            team=team,
            opponent="",
            lineup=[asdict(p) for p in players],
            total_wrc_plus=round(avg_wrc),
            lineup_strength=strength,
            platoon_advantage=platoon_adv,
            key_absences=absences,
            expected_runs_adjustment=round(runs_adj, 2),
            betting_angle=angle
        )

    def compare_lineups(self, home_lineup: LineupAnalysis, away_lineup: LineupAnalysis) -> dict:
        """Compare two lineups for a matchup edge."""
        home_wrc = home_lineup.total_wrc_plus
        away_wrc = away_lineup.total_wrc_plus
        diff = home_wrc - away_wrc

        if diff > 15:
            edge = f"🏠 Home lineup significantly stronger (+{diff} wRC+). Home team advantage."
        elif diff < -15:
            edge = f"✈️ Away lineup significantly stronger (+{abs(diff)} wRC+). Away team advantage."
        elif diff > 5:
            edge = f"🏠 Slight home lineup edge (+{diff} wRC+)."
        elif diff < -5:
            edge = f"✈️ Slight away lineup edge (+{abs(diff)} wRC+)."
        else:
            edge = "📊 Lineups roughly equivalent. Look at pitching matchup instead."

        return {
            "home_wrc_plus": home_wrc,
            "away_wrc_plus": away_wrc,
            "wrc_differential": diff,
            "home_strength": home_lineup.lineup_strength,
            "away_strength": away_lineup.lineup_strength,
            "home_runs_adj": home_lineup.expected_runs_adjustment,
            "away_runs_adj": away_lineup.expected_runs_adjustment,
            "edge": edge
        }


def create_lineup_routes(app, model: LineupImpactModel):
    """Create FastAPI routes."""

    @app.post("/api/v1/lineups/analyze")
    async def analyze(team: str, lineup: list, pitcher_hand: str = "R"):
        return asdict(model.analyze_lineup(team, lineup, pitcher_hand))

    @app.post("/api/v1/lineups/compare")
    async def compare(home: dict, away: dict):
        h = model.analyze_lineup(home['team'], home['lineup'], home.get('pitcher_hand', 'R'))
        a = model.analyze_lineup(away['team'], away['lineup'], away.get('pitcher_hand', 'R'))
        return model.compare_lineups(h, a)

    return model
