"""
Live Predictions Engine
========================
Orchestrates the live data pipeline, hit model, and win model
to produce today's predictions. Handles both live and demo modes.

Caching: Predictions refresh every 2 hours during game days.
Off-season: Shows demo data marked as "preseason projections."
"""

import logging
import time
from datetime import datetime, date
from typing import Dict, List, Any, Optional

from .mlb_live import (
    mlb_live, get_demo_schedule, get_demo_lineups,
    get_demo_pitcher_stats, get_demo_batter_stats, PARK_FACTORS,
)
from .hit_model import generate_top_hitters
from .win_model import generate_win_predictions, DEMO_TEAM_OPS, DEMO_BULLPEN_ERA

logger = logging.getLogger(__name__)

CACHE_TTL = 7200  # 2 hours
MLB_SEASON_START = date(2026, 3, 27)  # Opening Day 2026


class LivePredictions:
    """Main prediction engine combining live data + models."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}

    def _is_cached(self, key: str) -> bool:
        if key in self._cache_time:
            return (time.time() - self._cache_time[key]) < CACHE_TTL
        return False

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = data
        self._cache_time[key] = time.time()

    def _get_cache(self, key: str) -> Optional[Any]:
        if self._is_cached(key):
            return self._cache[key]
        return None

    @property
    def is_regular_season(self) -> bool:
        """Check if we're in the regular season."""
        today = date.today()
        return today >= MLB_SEASON_START

    @property
    def is_offseason(self) -> bool:
        return not self.is_regular_season

    def get_todays_hits(self, top_n: int = 30) -> Dict[str, Any]:
        """
        Get today's top hitters with hit probabilities.
        Uses live data during season, demo data during off-season.
        """
        cache_key = f"hits_{date.today().isoformat()}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        if self.is_regular_season:
            result = self._live_hits(top_n)
        else:
            result = self._demo_hits(top_n)

        self._set_cache(cache_key, result)
        return result

    def get_todays_wins(self) -> Dict[str, Any]:
        """
        Get today's game win predictions.
        Uses live data during season, demo data during off-season.
        """
        cache_key = f"wins_{date.today().isoformat()}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        if self.is_regular_season:
            result = self._live_wins()
        else:
            result = self._demo_wins()

        self._set_cache(cache_key, result)
        return result

    def get_combined(self) -> Dict[str, Any]:
        """Get both hits and wins combined."""
        hits = self.get_todays_hits()
        wins = self.get_todays_wins()

        return {
            "date": date.today().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "is_live": self.is_regular_season,
            "is_demo": self.is_offseason,
            "mode": "live" if self.is_regular_season else "preseason_projections",
            "season_starts": MLB_SEASON_START.isoformat(),
            "top_hitters": hits,
            "game_picks": wins,
        }

    # ── Live Mode ─────────────────────────────────────────────

    def _live_hits(self, top_n: int = 30) -> Dict[str, Any]:
        """Generate hit predictions from live MLB data."""
        try:
            full_data = mlb_live.get_todays_full_data()
        except Exception as e:
            logger.error(f"Failed to fetch live data: {e}")
            return self._demo_hits(top_n)

        games = full_data.get("games", [])
        if not games:
            return self._demo_hits(top_n)

        # Build pitcher stats dict from live data
        pitcher_stats = {}
        batter_stats = {}

        for game in games:
            for side in ("away", "home"):
                pitcher_key = f"{side}_pitcher_stats"
                ps = game.get(pitcher_key)
                if ps:
                    name = game[side]["probable_pitcher"]["name"]
                    pitcher_stats[name] = ps

                # Fetch batter stats for lineup players
                lineup = game.get(f"{side}_lineup", [])
                for batter in lineup:
                    bid = batter.get("id")
                    if bid and batter["name"] not in batter_stats:
                        try:
                            stats = mlb_live.get_player_hitting_stats(bid)
                            if stats:
                                batter_stats[batter["name"]] = {
                                    "avg": stats["avg"],
                                    "ops": stats["ops"],
                                    "obp": stats["obp"],
                                    "slg": stats["slg"],
                                    "hr": stats["home_runs"],
                                    "bat_side": batter.get("bat_side", "R"),
                                }
                        except Exception:
                            pass

        # Generate predictions
        hitters = generate_top_hitters(
            games=games,
            batter_stats=batter_stats,
            pitcher_stats=pitcher_stats,
        )

        is_spring = full_data.get("is_spring_training", False)
        mode = "spring_training" if is_spring else "regular_season"

        return {
            "date": date.today().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "mode": mode,
            "total_games": len(games),
            "total_hitters": len(hitters),
            "top_hitters": hitters[:top_n],
        }

    def _live_wins(self) -> Dict[str, Any]:
        """Generate win predictions from live MLB data."""
        try:
            full_data = mlb_live.get_todays_full_data()
        except Exception as e:
            logger.error(f"Failed to fetch live data: {e}")
            return self._demo_wins()

        games = full_data.get("games", [])
        if not games:
            return self._demo_wins()

        pitcher_stats = {}
        for game in games:
            for side in ("away", "home"):
                ps = game.get(f"{side}_pitcher_stats")
                if ps:
                    name = game[side]["probable_pitcher"]["name"]
                    pitcher_stats[name] = ps

        predictions = generate_win_predictions(
            games=games,
            pitcher_stats=pitcher_stats,
        )

        is_spring = full_data.get("is_spring_training", False)
        mode = "spring_training" if is_spring else "regular_season"

        return {
            "date": date.today().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "mode": mode,
            "total_games": len(games),
            "predictions": predictions,
        }

    # ── Demo Mode ─────────────────────────────────────────────

    def _demo_hits(self, top_n: int = 30) -> Dict[str, Any]:
        """Generate demo hit predictions from pre-set data."""
        games = get_demo_schedule()
        demo_lineups = get_demo_lineups()
        batter_stats = get_demo_batter_stats()
        pitcher_stats = get_demo_pitcher_stats()

        hitters = generate_top_hitters(
            games=games,
            batter_stats=batter_stats,
            pitcher_stats=pitcher_stats,
            demo_lineups=demo_lineups,
        )

        return {
            "date": date.today().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "mode": "preseason_projections",
            "disclaimer": "⚠️ PRESEASON PROJECTIONS based on 2025 stats. Season starts March 27, 2026.",
            "total_games": len(games),
            "total_hitters": len(hitters),
            "top_hitters": hitters[:top_n],
        }

    def _demo_wins(self) -> Dict[str, Any]:
        """Generate demo win predictions from pre-set data."""
        games = get_demo_schedule()
        pitcher_stats = get_demo_pitcher_stats()

        predictions = generate_win_predictions(
            games=games,
            pitcher_stats=pitcher_stats,
            team_ops=DEMO_TEAM_OPS,
            bullpen_era=DEMO_BULLPEN_ERA,
        )

        return {
            "date": date.today().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "mode": "preseason_projections",
            "disclaimer": "⚠️ PRESEASON PROJECTIONS based on 2025 stats. Season starts March 27, 2026.",
            "total_games": len(games),
            "predictions": predictions,
        }


# Singleton
live_predictions = LivePredictions()
