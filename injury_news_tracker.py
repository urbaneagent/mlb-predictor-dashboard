"""
MLB Predictor - Injury & News Intelligence Tracker
Scrapes MLB injury reports, roster moves, and breaking news.
Feeds into prediction model to adjust probabilities.
"""
import json
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

MLB_STATSAPI_BASE = "https://statsapi.mlb.com/api"


@dataclass
class InjuryReport:
    """A single injury entry."""
    player_id: int
    player_name: str
    team: str
    team_id: int
    position: str
    injury_type: str  # IL-10, IL-15, IL-60, DTD (day-to-day)
    injury_description: str
    injury_date: str
    expected_return: str = ""
    status: str = ""  # active, IL, DTD, out
    impact_rating: float = 0.0  # 0-10 impact on team's prediction
    war_2025: float = 0.0  # WAR value of injured player
    replacement_player: str = ""


@dataclass
class RosterMove:
    """A roster transaction."""
    transaction_id: str
    date: str
    team: str
    team_id: int
    player_name: str
    player_id: int
    move_type: str  # callup, optioned, DFA, trade, signed, released
    description: str
    impact_rating: float = 0.0


@dataclass
class NewsItem:
    """A news item relevant to predictions."""
    headline: str
    summary: str
    source: str
    url: str
    published_at: str
    teams_mentioned: list = field(default_factory=list)
    players_mentioned: list = field(default_factory=list)
    sentiment: str = "neutral"  # positive, negative, neutral
    prediction_impact: str = ""  # How this might affect predictions
    category: str = ""  # injury, trade, lineup, weather, suspension


@dataclass
class TeamHealthReport:
    """Aggregated injury impact for a team."""
    team: str
    team_id: int
    healthy_war: float = 0.0  # Total WAR if fully healthy
    current_war: float = 0.0  # WAR with injuries
    war_lost_to_injury: float = 0.0
    players_on_il: int = 0
    key_players_out: list = field(default_factory=list)
    day_to_day: list = field(default_factory=list)
    recent_returns: list = field(default_factory=list)
    health_score: float = 100.0  # 0-100, 100 = fully healthy
    prediction_adjustment: float = 0.0  # Win prob adjustment


class InjuryNewsTracker:
    """
    Tracks MLB injuries, roster moves, and news.
    Uses MLB StatsAPI for official data + web scraping for news.
    """

    # Approximate WAR values for star players (2025 projections)
    STAR_PLAYERS_WAR = {
        "Shohei Ohtani": 8.5, "Aaron Judge": 7.2, "Mookie Betts": 7.0,
        "Ronald Acuna Jr.": 7.5, "Corey Seager": 5.5, "Freddie Freeman": 5.8,
        "Juan Soto": 6.5, "Trea Turner": 5.0, "Marcus Semien": 4.8,
        "Matt Olson": 4.5, "Julio Rodriguez": 5.2, "Corbin Carroll": 5.0,
        "Bobby Witt Jr.": 6.0, "Gunnar Henderson": 5.5, "Elly De La Cruz": 5.0,
        "Gerrit Cole": 5.5, "Spencer Strider": 5.0, "Zack Wheeler": 4.8,
        "Shane McClanahan": 4.5, "Corbin Burnes": 5.2, "Logan Webb": 4.3,
        "Yoshinobu Yamamoto": 4.0, "Tyler Glasnow": 4.2,
    }

    # Position importance weights for prediction adjustment
    POSITION_WEIGHTS = {
        "P": 1.5, "SP": 1.5, "RP": 0.5, "CL": 0.8,
        "C": 0.8, "1B": 0.7, "2B": 0.8, "3B": 0.8,
        "SS": 0.9, "LF": 0.7, "CF": 0.9, "RF": 0.7,
        "DH": 0.6, "OF": 0.7, "IF": 0.8
    }

    def __init__(self):
        self.injuries = {}  # team_id -> [InjuryReport]
        self.roster_moves = []  # Recent roster moves
        self.news_items = []  # Recent news
        self.last_update = None

    async def fetch_injuries(self, team_id: int = None) -> list:
        """
        Fetch current injuries from MLB StatsAPI.
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                params = {"sportId": 1}
                if team_id:
                    params["teamId"] = team_id

                # Fetch 40-man roster with status
                if team_id:
                    response = await client.get(
                        f"{MLB_STATSAPI_BASE}/v1/teams/{team_id}/roster",
                        params={"rosterType": "depthChart"},
                        timeout=15
                    )
                else:
                    # Fetch all teams' injuries
                    response = await client.get(
                        f"{MLB_STATSAPI_BASE}/v1/injuries",
                        params=params,
                        timeout=15
                    )

                response.raise_for_status()
                data = response.json()

                injuries = []
                for entry in data.get('roster', data.get('injuries', [])):
                    player = entry.get('person', entry.get('player', {}))
                    status_data = entry.get('status', {})

                    if status_data.get('code') in ['IL10', 'IL15', 'IL60', 'D7', 'D10', 'D60']:
                        player_name = player.get('fullName', '')
                        war = self.STAR_PLAYERS_WAR.get(player_name, 1.5)

                        injury = InjuryReport(
                            player_id=player.get('id', 0),
                            player_name=player_name,
                            team=entry.get('team', {}).get('name', ''),
                            team_id=entry.get('team', {}).get('id', team_id or 0),
                            position=entry.get('position', {}).get('abbreviation', ''),
                            injury_type=status_data.get('code', ''),
                            injury_description=status_data.get('description', ''),
                            injury_date=entry.get('date', ''),
                            status=status_data.get('code', ''),
                            war_2025=war,
                            impact_rating=self._calculate_impact(war, entry.get('position', {}).get('abbreviation', ''))
                        )
                        injuries.append(injury)

                        # Cache by team
                        tid = injury.team_id
                        if tid not in self.injuries:
                            self.injuries[tid] = []
                        self.injuries[tid].append(injury)

                self.last_update = datetime.now(timezone.utc).isoformat()
                return injuries

        except Exception as e:
            logger.error(f"Injury fetch error: {e}")
            # Return cached
            if team_id and team_id in self.injuries:
                return self.injuries[team_id]
            return []

    async def fetch_roster_moves(self, days: int = 7) -> list:
        """Fetch recent roster transactions."""
        import httpx

        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MLB_STATSAPI_BASE}/v1/transactions",
                    params={"startDate": start_date, "endDate": end_date},
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()

                moves = []
                for txn in data.get('transactions', []):
                    player = txn.get('person', {})
                    team = txn.get('team', txn.get('toTeam', {}))
                    description = txn.get('description', '')

                    # Determine move type
                    move_type = self._classify_move(txn.get('typeCode', ''), description)

                    move = RosterMove(
                        transaction_id=str(txn.get('id', '')),
                        date=txn.get('date', ''),
                        team=team.get('name', ''),
                        team_id=team.get('id', 0),
                        player_name=player.get('fullName', ''),
                        player_id=player.get('id', 0),
                        move_type=move_type,
                        description=description,
                        impact_rating=self._estimate_move_impact(player.get('fullName', ''), move_type)
                    )
                    moves.append(move)

                self.roster_moves = moves
                return moves

        except Exception as e:
            logger.error(f"Roster moves fetch error: {e}")
            return self.roster_moves

    def get_team_health_report(self, team_id: int) -> TeamHealthReport:
        """Generate comprehensive health report for a team."""
        injuries = self.injuries.get(team_id, [])

        total_war_lost = sum(inj.war_2025 for inj in injuries)
        key_players = [inj for inj in injuries if inj.war_2025 >= 3.0]
        dtd_players = [inj for inj in injuries if inj.injury_type in ['DTD', 'D7']]
        il_players = [inj for inj in injuries if inj.injury_type in ['IL10', 'IL15', 'IL60', 'D10', 'D60']]

        # Health score: 100 minus weighted WAR impact
        health_score = max(0, 100 - (total_war_lost * 5))

        # Prediction adjustment (negative = worse than expected)
        # Each WAR point lost reduces team win probability by ~0.5%
        prediction_adj = -total_war_lost * 0.005

        return TeamHealthReport(
            team="",  # Fill from external data
            team_id=team_id,
            war_lost_to_injury=round(total_war_lost, 1),
            players_on_il=len(il_players),
            key_players_out=[asdict(p) for p in key_players],
            day_to_day=[asdict(p) for p in dtd_players],
            health_score=round(health_score, 1),
            prediction_adjustment=round(prediction_adj, 4)
        )

    def get_prediction_adjustment(self, home_team_id: int, away_team_id: int) -> dict:
        """
        Get prediction adjustment based on injuries for a specific matchup.
        Returns adjustment to add to base win probability.
        """
        home_report = self.get_team_health_report(home_team_id)
        away_report = self.get_team_health_report(away_team_id)

        # Net adjustment: if home team is healthier, positive adjustment
        net_adjustment = home_report.prediction_adjustment - away_report.prediction_adjustment

        return {
            "home_adjustment": round(home_report.prediction_adjustment, 4),
            "away_adjustment": round(away_report.prediction_adjustment, 4),
            "net_home_advantage": round(net_adjustment, 4),
            "home_health_score": home_report.health_score,
            "away_health_score": away_report.health_score,
            "home_war_lost": home_report.war_lost_to_injury,
            "away_war_lost": away_report.war_lost_to_injury,
            "home_key_out": len(home_report.key_players_out),
            "away_key_out": len(away_report.key_players_out),
            "recommendation": self._injury_recommendation(home_report, away_report)
        }

    def _calculate_impact(self, war: float, position: str) -> float:
        """Calculate impact rating (0-10) for an injured player."""
        pos_weight = self.POSITION_WEIGHTS.get(position, 0.7)
        return min(10, war * pos_weight * 1.2)

    def _classify_move(self, type_code: str, description: str) -> str:
        """Classify a roster transaction."""
        desc_lower = description.lower()
        if 'recall' in desc_lower or 'selected' in desc_lower:
            return 'callup'
        elif 'optioned' in desc_lower:
            return 'optioned'
        elif 'designated' in desc_lower or 'dfa' in desc_lower:
            return 'DFA'
        elif 'traded' in desc_lower:
            return 'trade'
        elif 'signed' in desc_lower:
            return 'signed'
        elif 'released' in desc_lower:
            return 'released'
        elif 'injured' in desc_lower or 'disabled' in desc_lower:
            return 'IL'
        elif 'activated' in desc_lower or 'reinstated' in desc_lower:
            return 'activated'
        return type_code

    def _estimate_move_impact(self, player_name: str, move_type: str) -> float:
        """Estimate impact of a roster move on predictions."""
        war = self.STAR_PLAYERS_WAR.get(player_name, 1.0)

        impact_multipliers = {
            'callup': 0.5, 'optioned': -0.3, 'DFA': -0.2,
            'trade': 0.8, 'signed': 0.3, 'released': -0.1,
            'IL': -1.0, 'activated': 0.8
        }

        multiplier = impact_multipliers.get(move_type, 0.1)
        return round(war * multiplier, 1)

    def _injury_recommendation(self, home_report, away_report) -> str:
        """Generate recommendation based on injury comparison."""
        diff = home_report.health_score - away_report.health_score

        if diff > 20:
            return f"STRONG HOME EDGE: Home team significantly healthier ({home_report.health_score:.0f} vs {away_report.health_score:.0f})"
        elif diff > 10:
            return f"SLIGHT HOME EDGE: Home team healthier ({home_report.health_score:.0f} vs {away_report.health_score:.0f})"
        elif diff < -20:
            return f"STRONG AWAY EDGE: Away team significantly healthier ({away_report.health_score:.0f} vs {home_report.health_score:.0f})"
        elif diff < -10:
            return f"SLIGHT AWAY EDGE: Away team healthier ({away_report.health_score:.0f} vs {home_report.health_score:.0f})"
        else:
            return f"NEUTRAL: Both teams similarly healthy ({home_report.health_score:.0f} vs {away_report.health_score:.0f})"


def create_injury_routes(app, tracker: InjuryNewsTracker):
    """Create FastAPI routes for injury and news data."""

    @app.get("/api/v1/injuries")
    async def get_injuries(team_id: int = None):
        """Get current injuries."""
        injuries = await tracker.fetch_injuries(team_id)
        return {"count": len(injuries), "injuries": [asdict(i) for i in injuries]}

    @app.get("/api/v1/roster-moves")
    async def get_roster_moves(days: int = 7):
        """Get recent roster transactions."""
        moves = await tracker.fetch_roster_moves(days)
        return {"count": len(moves), "moves": [asdict(m) for m in moves]}

    @app.get("/api/v1/team-health/{team_id}")
    async def get_team_health(team_id: int):
        """Get team health report."""
        # Ensure we have data
        await tracker.fetch_injuries(team_id)
        report = tracker.get_team_health_report(team_id)
        return asdict(report)

    @app.get("/api/v1/matchup-health/{home_id}/{away_id}")
    async def get_matchup_health(home_id: int, away_id: int):
        """Get injury-adjusted prediction for a matchup."""
        await tracker.fetch_injuries(home_id)
        await tracker.fetch_injuries(away_id)
        return tracker.get_prediction_adjustment(home_id, away_id)

    return tracker
