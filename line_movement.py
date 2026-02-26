"""
MLB Predictor - Line Movement Tracker
Monitors opening lines vs current lines to detect sharp money.
Reverse line movement (RLM) is one of the strongest betting signals.
"""
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict


@dataclass
class LineSnapshot:
    """Point-in-time odds snapshot."""
    timestamp: str
    home_ml: int
    away_ml: int
    total: float
    home_spread: float
    home_spread_odds: int
    away_spread_odds: int
    source: str = ""


@dataclass
class LineMovementAnalysis:
    """Analysis of line movement for a game."""
    game_id: str
    home_team: str
    away_team: str
    opening_line: dict = field(default_factory=dict)
    current_line: dict = field(default_factory=dict)
    movement: dict = field(default_factory=dict)
    public_betting: dict = field(default_factory=dict)
    sharp_indicators: list = field(default_factory=list)
    reverse_line_movement: bool = False
    steam_move: bool = False
    recommendation: str = ""


class LineMovementTracker:
    """
    Tracks and analyzes line movements to identify sharp action.

    Key concepts:
    - Reverse Line Movement (RLM): Line moves AGAINST public betting
    - Steam Moves: Sudden coordinated moves across sportsbooks
    - CLV (Closing Line Value): Getting a better line than closing
    """

    def __init__(self):
        self.games = {}  # game_id -> [LineSnapshot]

    def record_line(self, game_id: str, snapshot: dict):
        """Record a line snapshot."""
        if game_id not in self.games:
            self.games[game_id] = []

        snap = LineSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            home_ml=snapshot.get('home_ml', 0),
            away_ml=snapshot.get('away_ml', 0),
            total=snapshot.get('total', 0),
            home_spread=snapshot.get('home_spread', 0),
            home_spread_odds=snapshot.get('home_spread_odds', -110),
            away_spread_odds=snapshot.get('away_spread_odds', -110),
            source=snapshot.get('source', '')
        )

        self.games[game_id].append(snap)

    def analyze_movement(self, game_id: str, home_team: str, away_team: str,
                         public_home_pct: float = 50, public_over_pct: float = 50) -> LineMovementAnalysis:
        """Analyze line movement for a game."""
        snapshots = self.games.get(game_id, [])

        if len(snapshots) < 2:
            return LineMovementAnalysis(
                game_id=game_id, home_team=home_team, away_team=away_team,
                recommendation="Not enough data — need opening + current line."
            )

        opening = snapshots[0]
        current = snapshots[-1]

        # Movement calculation
        ml_move = current.home_ml - opening.home_ml
        total_move = current.total - opening.total
        spread_move = current.home_spread - opening.home_spread

        movement = {
            "home_ml_change": ml_move,
            "total_change": total_move,
            "spread_change": spread_move,
            "direction": "home" if ml_move < 0 else "away" if ml_move > 0 else "stable",
            "total_direction": "up" if total_move > 0 else "down" if total_move < 0 else "stable"
        }

        opening_dict = asdict(opening)
        current_dict = asdict(current)

        public = {
            "home_pct": public_home_pct,
            "away_pct": 100 - public_home_pct,
            "over_pct": public_over_pct,
            "under_pct": 100 - public_over_pct
        }

        # Sharp indicators
        sharp = []
        rlm = False
        steam = False

        # RLM: Public on one side but line moves the other way
        if public_home_pct > 60 and ml_move > 0:
            rlm = True
            sharp.append({
                "signal": "REVERSE LINE MOVEMENT",
                "description": f"Public is {public_home_pct:.0f}% on {home_team} but line moving toward {away_team}. Sharp money on {away_team}.",
                "strength": "strong",
                "side": away_team
            })
        elif public_home_pct < 40 and ml_move < 0:
            rlm = True
            sharp.append({
                "signal": "REVERSE LINE MOVEMENT",
                "description": f"Public is {100-public_home_pct:.0f}% on {away_team} but line moving toward {home_team}. Sharp money on {home_team}.",
                "strength": "strong",
                "side": home_team
            })

        # Total RLM
        if public_over_pct > 65 and total_move < -0.5:
            sharp.append({
                "signal": "TOTAL RLM",
                "description": f"Public is {public_over_pct:.0f}% on OVER but total dropped {total_move}. Sharp money on UNDER.",
                "strength": "strong",
                "side": "UNDER"
            })
        elif public_over_pct < 35 and total_move > 0.5:
            sharp.append({
                "signal": "TOTAL RLM",
                "description": f"Public is {100-public_over_pct:.0f}% on UNDER but total rose {total_move}. Sharp money on OVER.",
                "strength": "strong",
                "side": "OVER"
            })

        # Steam move detection (big move in short time)
        if len(snapshots) >= 3:
            recent = snapshots[-3:]
            recent_ml_change = recent[-1].home_ml - recent[0].home_ml
            if abs(recent_ml_change) >= 15:
                steam = True
                direction = home_team if recent_ml_change < 0 else away_team
                sharp.append({
                    "signal": "STEAM MOVE",
                    "description": f"Rapid {abs(recent_ml_change)} cent move toward {direction}. Coordinated sharp action.",
                    "strength": "very_strong",
                    "side": direction
                })

        # Large movement
        if abs(ml_move) >= 20:
            direction = home_team if ml_move < 0 else away_team
            sharp.append({
                "signal": "SIGNIFICANT MOVEMENT",
                "description": f"Line moved {abs(ml_move)} cents toward {direction}. Major money on this side.",
                "strength": "medium",
                "side": direction
            })

        # Recommendation
        rec = ""
        if sharp:
            strongest = max(sharp, key=lambda s: {"very_strong": 3, "strong": 2, "medium": 1}.get(s['strength'], 0))
            rec = f"🎯 Sharp side: {strongest['side']} ({strongest['signal']}). Follow the money, not the public."
        else:
            rec = "📊 No significant sharp action detected. Line movement is consistent with public betting."

        return LineMovementAnalysis(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            opening_line=opening_dict,
            current_line=current_dict,
            movement=movement,
            public_betting=public,
            sharp_indicators=[s for s in sharp],
            reverse_line_movement=rlm,
            steam_move=steam,
            recommendation=rec
        )

    def get_biggest_movers(self, min_move: int = 15) -> list:
        """Get games with the biggest line movements."""
        movers = []
        for game_id, snaps in self.games.items():
            if len(snaps) < 2:
                continue
            opening = snaps[0]
            current = snaps[-1]
            ml_move = abs(current.home_ml - opening.home_ml)
            total_move = abs(current.total - opening.total)

            if ml_move >= min_move or total_move >= 0.5:
                movers.append({
                    "game_id": game_id,
                    "ml_move": ml_move,
                    "total_move": total_move,
                    "opening_ml": f"{opening.home_ml}/{opening.away_ml}",
                    "current_ml": f"{current.home_ml}/{current.away_ml}",
                    "opening_total": opening.total,
                    "current_total": current.total
                })

        return sorted(movers, key=lambda m: m['ml_move'], reverse=True)


def create_line_routes(app, tracker: LineMovementTracker):
    """Create FastAPI routes."""

    @app.post("/api/v1/lines/{game_id}/record")
    async def record_line(game_id: str, snapshot: dict):
        tracker.record_line(game_id, snapshot)
        return {"recorded": True}

    @app.get("/api/v1/lines/{game_id}/analysis")
    async def analyze(game_id: str, home_team: str, away_team: str,
                      public_home_pct: float = 50, public_over_pct: float = 50):
        result = tracker.analyze_movement(game_id, home_team, away_team,
                                          public_home_pct, public_over_pct)
        return asdict(result)

    @app.get("/api/v1/lines/movers")
    async def biggest_movers(min_move: int = 15):
        return tracker.get_biggest_movers(min_move)

    return tracker
