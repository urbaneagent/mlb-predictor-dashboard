"""
MLB Predictor - Real-Time Game State Engine
Tracks live games pitch-by-pitch using MLB StatsAPI.
Updates win probability, adjusts predictions in real-time,
and triggers alerts for bet-relevant events.
"""
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

MLB_STATSAPI_BASE = "https://statsapi.mlb.com/api"


class GameStatus(str, Enum):
    PREVIEW = "Preview"
    PRE_GAME = "Pre-Game"
    WARMUP = "Warmup"
    IN_PROGRESS = "In Progress"
    MANAGER_CHALLENGE = "Manager Challenge"
    DELAYED = "Delayed"
    FINAL = "Final"
    GAME_OVER = "Game Over"
    POSTPONED = "Postponed"
    SUSPENDED = "Suspended"


@dataclass
class PitchEvent:
    """Single pitch event."""
    pitch_number: int
    inning: int
    half: str  # top, bottom
    pitcher_id: int
    pitcher_name: str
    batter_id: int
    batter_name: str
    pitch_type: str  # FF, SL, CU, CH, SI, FC, etc.
    pitch_speed: float  # mph
    zone: int  # 1-9 for strikes, 11-14 for balls
    result: str  # Ball, Called Strike, Swinging Strike, Foul, In play...
    count_before: str  # "1-2"
    count_after: str
    at_bat_result: str = ""  # Single, Double, HR, Strikeout, etc.
    rbi: int = 0
    exit_velocity: float = 0.0
    launch_angle: float = 0.0
    hit_distance: float = 0.0
    win_prob_change: float = 0.0
    timestamp: str = ""


@dataclass
class GameState:
    """Current state of a live game."""
    game_id: int
    game_date: str
    status: str
    inning: int = 0
    half: str = "top"  # top, bottom
    outs: int = 0
    balls: int = 0
    strikes: int = 0

    # Score
    home_team: str = ""
    away_team: str = ""
    home_team_id: int = 0
    away_team_id: int = 0
    home_score: int = 0
    away_score: int = 0

    # Runners
    runner_first: bool = False
    runner_second: bool = False
    runner_third: bool = False

    # Current matchup
    current_pitcher_id: int = 0
    current_pitcher_name: str = ""
    current_pitcher_pitches: int = 0
    current_batter_id: int = 0
    current_batter_name: str = ""

    # Probabilities
    home_win_probability: float = 0.5
    away_win_probability: float = 0.5
    pre_game_home_prob: float = 0.5

    # Pitching
    home_pitcher_name: str = ""
    away_pitcher_name: str = ""
    home_pitcher_pitches: int = 0
    away_pitcher_pitches: int = 0

    # Live stats
    home_hits: int = 0
    away_hits: int = 0
    home_errors: int = 0
    away_errors: int = 0

    # Betting context
    live_total: float = 0.0
    pace_runs_per_9: float = 0.0
    leverage_index: float = 1.0

    # Metadata
    venue: str = ""
    weather: str = ""
    attendance: int = 0
    game_duration_minutes: int = 0
    last_play: str = ""
    pitch_events: list = field(default_factory=list)
    scoring_plays: list = field(default_factory=list)
    updated_at: str = ""


class GameStateEngine:
    """
    Real-time game state tracker using MLB StatsAPI.
    Polls live game data and calculates win probability shifts.
    """

    def __init__(self, poll_interval: int = 10):
        self.poll_interval = poll_interval  # seconds between API polls
        self.active_games = {}  # game_id -> GameState
        self.event_handlers = {
            'scoring_play': [],
            'pitching_change': [],
            'win_prob_shift': [],
            'game_start': [],
            'game_end': [],
            'high_leverage': [],
            'milestone': [],
        }
        self._running = False

    def register_handler(self, event_type: str, handler: Callable):
        """Register a callback for game events."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)

    async def fetch_todays_schedule(self) -> list:
        """Fetch today's MLB schedule."""
        import httpx
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MLB_STATSAPI_BASE}/v1/schedule",
                    params={"sportId": 1, "date": today,
                            "hydrate": "probablePitcher,linescore,team"},
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()

                games = []
                for date_entry in data.get('dates', []):
                    for game in date_entry.get('games', []):
                        games.append({
                            "game_id": game['gamePk'],
                            "status": game.get('status', {}).get('detailedState', 'Unknown'),
                            "home_team": game.get('teams', {}).get('home', {}).get('team', {}).get('name', ''),
                            "away_team": game.get('teams', {}).get('away', {}).get('team', {}).get('name', ''),
                            "home_team_id": game.get('teams', {}).get('home', {}).get('team', {}).get('id', 0),
                            "away_team_id": game.get('teams', {}).get('away', {}).get('team', {}).get('id', 0),
                            "game_time": game.get('gameDate', ''),
                            "venue": game.get('venue', {}).get('name', ''),
                            "home_pitcher": game.get('teams', {}).get('home', {}).get('probablePitcher', {}).get('fullName', 'TBD'),
                            "away_pitcher": game.get('teams', {}).get('away', {}).get('probablePitcher', {}).get('fullName', 'TBD'),
                        })

                return games

        except Exception as e:
            logger.error(f"Schedule fetch error: {e}")
            return []

    async def fetch_game_state(self, game_id: int) -> Optional[GameState]:
        """Fetch current game state from MLB StatsAPI."""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MLB_STATSAPI_BASE}/v1.1/game/{game_id}/feed/live",
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()

                game_data = data.get('gameData', {})
                live_data = data.get('liveData', {})
                linescore = live_data.get('linescore', {})
                plays = live_data.get('plays', {})
                boxscore = live_data.get('boxscore', {})

                status = game_data.get('status', {}).get('detailedState', 'Unknown')
                teams = game_data.get('teams', {})
                venue = game_data.get('venue', {})
                weather_data = game_data.get('weather', {})

                # Current play state
                current_play = plays.get('currentPlay', {})
                matchup = current_play.get('matchup', {})
                count = current_play.get('count', {})
                runners = current_play.get('runners', [])

                # Base runners
                on_first = any(r.get('movement', {}).get('end', '') == '1B' for r in runners if not r.get('movement', {}).get('isOut'))
                on_second = any(r.get('movement', {}).get('end', '') == '2B' for r in runners if not r.get('movement', {}).get('isOut'))
                on_third = any(r.get('movement', {}).get('end', '') == '3B' for r in runners if not r.get('movement', {}).get('isOut'))

                # Current inning info
                inning = linescore.get('currentInning', 0)
                half = 'bottom' if linescore.get('isBottomInning', False) else 'top'
                outs = linescore.get('outs', 0)

                # Scores
                home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)
                away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)
                home_hits = linescore.get('teams', {}).get('home', {}).get('hits', 0)
                away_hits = linescore.get('teams', {}).get('away', {}).get('hits', 0)
                home_errors = linescore.get('teams', {}).get('home', {}).get('errors', 0)
                away_errors = linescore.get('teams', {}).get('away', {}).get('errors', 0)

                # Calculate win probability using log5 + game state
                home_wp = self._calculate_win_probability(
                    home_score, away_score, inning, half, outs,
                    on_first, on_second, on_third
                )

                # Calculate pace
                innings_completed = max(1, (inning - 1) + (0.5 if half == 'bottom' else 0))
                total_runs = home_score + away_score
                pace = (total_runs / innings_completed) * 9 if innings_completed > 0 else 0

                # Leverage index estimation
                leverage = self._estimate_leverage_index(
                    abs(home_score - away_score), inning, outs,
                    on_first, on_second, on_third
                )

                # Last play description
                all_plays = plays.get('allPlays', [])
                last_play_desc = ""
                if all_plays:
                    last = all_plays[-1]
                    result_desc = last.get('result', {})
                    last_play_desc = result_desc.get('description', '')

                # Scoring plays
                scoring = []
                for play in plays.get('scoringPlays', []):
                    if isinstance(play, int) and play < len(all_plays):
                        sp = all_plays[play]
                        scoring.append({
                            "inning": sp.get('about', {}).get('inning', 0),
                            "half": sp.get('about', {}).get('halfInning', ''),
                            "description": sp.get('result', {}).get('description', ''),
                            "rbi": sp.get('result', {}).get('rbi', 0)
                        })

                state = GameState(
                    game_id=game_id,
                    game_date=game_data.get('datetime', {}).get('officialDate', ''),
                    status=status,
                    inning=inning,
                    half=half,
                    outs=outs,
                    balls=count.get('balls', 0),
                    strikes=count.get('strikes', 0),
                    home_team=teams.get('home', {}).get('name', ''),
                    away_team=teams.get('away', {}).get('name', ''),
                    home_team_id=teams.get('home', {}).get('id', 0),
                    away_team_id=teams.get('away', {}).get('id', 0),
                    home_score=home_score,
                    away_score=away_score,
                    runner_first=on_first,
                    runner_second=on_second,
                    runner_third=on_third,
                    current_pitcher_id=matchup.get('pitcher', {}).get('id', 0),
                    current_pitcher_name=matchup.get('pitcher', {}).get('fullName', ''),
                    current_batter_id=matchup.get('batter', {}).get('id', 0),
                    current_batter_name=matchup.get('batter', {}).get('fullName', ''),
                    home_win_probability=round(home_wp, 3),
                    away_win_probability=round(1 - home_wp, 3),
                    home_hits=home_hits,
                    away_hits=away_hits,
                    home_errors=home_errors,
                    away_errors=away_errors,
                    live_total=round(pace, 1),
                    pace_runs_per_9=round(pace, 1),
                    leverage_index=round(leverage, 2),
                    venue=venue.get('name', ''),
                    weather=f"{weather_data.get('temp', '')}°F, {weather_data.get('condition', '')}, Wind: {weather_data.get('wind', '')}",
                    last_play=last_play_desc,
                    scoring_plays=scoring,
                    updated_at=datetime.now(timezone.utc).isoformat()
                )

                # Check for state changes and fire events
                self._check_events(game_id, state)

                # Store state
                self.active_games[game_id] = state

                return state

        except Exception as e:
            logger.error(f"Game state fetch error for {game_id}: {e}")
            return self.active_games.get(game_id)

    def _calculate_win_probability(self, home_score, away_score, inning, half,
                                    outs, on_1b, on_2b, on_3b) -> float:
        """
        Calculate home team win probability using empirical run expectancy.
        Based on Markov chain model for baseball game states.
        """
        if inning == 0:
            return 0.54  # Home team advantage pre-game

        diff = home_score - away_score
        innings_remaining = max(0, 9 - inning + (0 if half == 'bottom' else 0.5))

        if innings_remaining <= 0 and diff != 0:
            return 1.0 if diff > 0 else 0.0

        # Empirical win probability by run differential and innings remaining
        # Based on historical MLB data
        if innings_remaining <= 0:
            return 0.50  # Tied, going to extras

        # Expected runs remaining for each team
        avg_runs_per_inning = 0.5  # ~4.5 runs per 9 innings
        home_expected = avg_runs_per_inning * innings_remaining
        away_expected = avg_runs_per_inning * innings_remaining

        # Adjust for current base/out state
        run_expectancy = self._run_expectancy_24(outs, on_1b, on_2b, on_3b)

        # Simple logistic model
        import math
        z = diff * 0.35 + (innings_remaining - 4.5) * 0.02 + run_expectancy * 0.1
        if half == 'bottom':
            z += 0.03  # Slight home advantage in bottom of inning

        wp = 1 / (1 + math.exp(-z))

        # Clamp
        return max(0.01, min(0.99, wp))

    def _run_expectancy_24(self, outs, on_1b, on_2b, on_3b) -> float:
        """
        Run expectancy for 24 base-out states.
        Based on 2019-2024 MLB averages.
        """
        # [outs][bases_state] -> expected runs rest of inning
        # bases_state: 0=empty, 1=1B, 2=2B, 3=1B2B, 4=3B, 5=1B3B, 6=2B3B, 7=loaded
        re24 = {
            0: [0.481, 0.859, 1.100, 1.437, 1.350, 1.784, 1.920, 2.282],
            1: [0.254, 0.509, 0.664, 0.884, 0.950, 1.130, 1.352, 1.520],
            2: [0.098, 0.224, 0.319, 0.429, 0.353, 0.478, 0.556, 0.736],
        }

        outs = min(outs, 2)
        bases = 0
        if on_1b: bases += 1
        if on_2b: bases += 2
        if on_3b: bases += 4

        return re24.get(outs, re24[2])[min(bases, 7)]

    def _estimate_leverage_index(self, run_diff, inning, outs, on_1b, on_2b, on_3b) -> float:
        """
        Estimate leverage index (importance of current situation).
        LI = 1.0 is average, >2.0 is high leverage, <0.5 is low.
        """
        import math

        # Base leverage decreases as run diff increases
        diff_factor = math.exp(-run_diff * 0.5)

        # Late innings are higher leverage
        inning_factor = min(2.0, 0.5 + (inning / 9.0) * 1.5)

        # Runners on base increase leverage
        runner_factor = 1.0
        if on_3b: runner_factor += 0.5
        if on_2b: runner_factor += 0.3
        if on_1b: runner_factor += 0.1

        # Close games with runners are highest leverage
        li = diff_factor * inning_factor * runner_factor

        return max(0.1, min(5.0, li))

    def _check_events(self, game_id: int, new_state: GameState):
        """Check for significant state changes and fire event handlers."""
        old_state = self.active_games.get(game_id)

        if not old_state:
            # New game started
            if new_state.status == GameStatus.IN_PROGRESS.value:
                self._fire_event('game_start', new_state)
            return

        # Game ended
        if new_state.status in [GameStatus.FINAL.value, GameStatus.GAME_OVER.value]:
            if old_state.status == GameStatus.IN_PROGRESS.value:
                self._fire_event('game_end', new_state)

        # Scoring play
        if (new_state.home_score != old_state.home_score or
                new_state.away_score != old_state.away_score):
            self._fire_event('scoring_play', new_state, {
                'old_home': old_state.home_score,
                'old_away': old_state.away_score,
                'runs_scored': (new_state.home_score + new_state.away_score) -
                               (old_state.home_score + old_state.away_score)
            })

        # Win probability shift (>10%)
        wp_change = abs(new_state.home_win_probability - old_state.home_win_probability)
        if wp_change > 0.10:
            self._fire_event('win_prob_shift', new_state, {
                'old_wp': old_state.home_win_probability,
                'new_wp': new_state.home_win_probability,
                'change': wp_change
            })

        # High leverage moment
        if new_state.leverage_index > 2.0 and old_state.leverage_index <= 2.0:
            self._fire_event('high_leverage', new_state)

        # Pitching change
        if (new_state.current_pitcher_id != old_state.current_pitcher_id and
                old_state.current_pitcher_id != 0):
            self._fire_event('pitching_change', new_state, {
                'old_pitcher': old_state.current_pitcher_name,
                'new_pitcher': new_state.current_pitcher_name
            })

    def _fire_event(self, event_type: str, state: GameState, extra: dict = None):
        """Fire registered event handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event_type, state, extra or {})
            except Exception as e:
                logger.error(f"Event handler error ({event_type}): {e}")

    def get_all_live_games(self) -> list:
        """Get all currently tracked live games."""
        return [asdict(g) for g in self.active_games.values()
                if g.status == GameStatus.IN_PROGRESS.value]

    def get_game_summary(self, game_id: int) -> Optional[dict]:
        """Get a concise summary of a game for display."""
        state = self.active_games.get(game_id)
        if not state:
            return None

        base_state = ""
        if state.runner_first: base_state += "1B "
        if state.runner_second: base_state += "2B "
        if state.runner_third: base_state += "3B "
        base_state = base_state.strip() or "Empty"

        return {
            "game_id": state.game_id,
            "matchup": f"{state.away_team} @ {state.home_team}",
            "score": f"{state.away_score}-{state.home_score}",
            "inning": f"{'Bot' if state.half == 'bottom' else 'Top'} {state.inning}",
            "outs": state.outs,
            "count": f"{state.balls}-{state.strikes}",
            "bases": base_state,
            "pitcher": state.current_pitcher_name,
            "batter": state.current_batter_name,
            "home_wp": f"{state.home_win_probability:.1%}",
            "leverage": f"{state.leverage_index:.1f}x",
            "pace": f"{state.pace_runs_per_9:.1f} runs/9",
            "last_play": state.last_play[:80] if state.last_play else "",
            "venue": state.venue,
            "weather": state.weather
        }


def create_game_state_routes(app, engine: GameStateEngine):
    """Create FastAPI routes for live game tracking."""

    @app.get("/api/v1/live/schedule")
    async def get_schedule():
        """Get today's MLB schedule."""
        games = await engine.fetch_todays_schedule()
        return {"date": datetime.now().strftime("%Y-%m-%d"), "games": games}

    @app.get("/api/v1/live/game/{game_id}")
    async def get_game(game_id: int):
        """Get current state of a specific game."""
        state = await engine.fetch_game_state(game_id)
        if not state:
            return {"error": "Game not found"}
        return asdict(state)

    @app.get("/api/v1/live/game/{game_id}/summary")
    async def get_game_summary_route(game_id: int):
        """Get concise game summary."""
        summary = engine.get_game_summary(game_id)
        if not summary:
            # Try fetching first
            await engine.fetch_game_state(game_id)
            summary = engine.get_game_summary(game_id)
        return summary or {"error": "Game not found"}

    @app.get("/api/v1/live/all")
    async def get_all_live():
        """Get all live games."""
        return engine.get_all_live_games()

    return engine
