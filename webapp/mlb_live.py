"""
MLB Live Data Pipeline
========================
Pulls today's schedule, lineups, and player stats from the MLB Stats API.
Caches data to avoid spamming the API (2-hour refresh during game days).

MLB Stats API: https://statsapi.mlb.com/api/v1/
No API key needed — free and public.
"""

import json
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import httpx

logger = logging.getLogger(__name__)

MLB_BASE = "https://statsapi.mlb.com/api/v1"
CACHE_TTL_SECONDS = 7200  # 2 hours
OFFSEASON_CACHE_TTL = 86400  # 24 hours for off-season demo data

# MLB Team ID → abbreviation mapping
TEAM_ID_TO_ABBR = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

ABBR_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_ABBR.items()}

# Park factors (runs per game relative to league average, 1.0 = neutral)
PARK_FACTORS = {
    "COL": 1.25, "CIN": 1.08, "TEX": 1.06, "AZ": 1.05, "BOS": 1.04,
    "CHC": 1.03, "ATL": 1.02, "PHI": 1.02, "MIL": 1.01, "LAA": 1.01,
    "NYY": 1.00, "CLE": 1.00, "MIN": 1.00, "DET": 0.99, "BAL": 0.99,
    "HOU": 0.98, "LAD": 0.98, "KC": 0.98, "STL": 0.97, "WSH": 0.97,
    "TOR": 0.97, "PIT": 0.96, "SD": 0.96, "SEA": 0.95, "SF": 0.95,
    "NYM": 0.95, "TB": 0.94, "CWS": 0.94, "MIA": 0.93, "ATH": 0.93,
}


class MLBLiveData:
    """Fetches and caches live MLB data from the Stats API."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("/tmp/mlb_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._memory_cache_time: Dict[str, float] = {}
        self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=15.0,
                headers={"User-Agent": "MLBPredictor/2.0"},
                follow_redirects=True,
            )
        return self._client

    def _cache_key(self, endpoint: str) -> str:
        return endpoint.replace("/", "_").replace("?", "_").replace("&", "_")

    def _get_cached(self, key: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[Any]:
        """Check memory cache, then disk cache."""
        now = time.time()
        if key in self._memory_cache:
            age = now - self._memory_cache_time.get(key, 0)
            if age < ttl:
                return self._memory_cache[key]

        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            age = now - cache_file.stat().st_mtime
            if age < ttl:
                try:
                    data = json.loads(cache_file.read_text())
                    self._memory_cache[key] = data
                    self._memory_cache_time[key] = cache_file.stat().st_mtime
                    return data
                except Exception:
                    pass
        return None

    def _set_cache(self, key: str, data: Any):
        self._memory_cache[key] = data
        self._memory_cache_time[key] = time.time()
        try:
            cache_file = self.cache_dir / f"{key}.json"
            cache_file.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache write failed for {key}: {e}")

    def _fetch(self, endpoint: str, ttl: int = CACHE_TTL_SECONDS) -> Optional[Dict]:
        """Fetch from MLB API with caching."""
        key = self._cache_key(endpoint)
        cached = self._get_cached(key, ttl)
        if cached is not None:
            logger.debug(f"Cache hit: {endpoint}")
            return cached

        url = f"{MLB_BASE}{endpoint}"
        try:
            logger.info(f"Fetching: {url}")
            resp = self._get_client().get(url)
            resp.raise_for_status()
            data = resp.json()
            self._set_cache(key, data)
            return data
        except Exception as e:
            logger.error(f"MLB API error for {endpoint}: {e}")
            return None

    # ── Schedule ──────────────────────────────────────────────

    def get_schedule(self, game_date: Optional[date] = None,
                     game_type: str = "R") -> List[Dict]:
        """
        Get today's MLB schedule.
        game_type: R=regular, S=spring training, P=postseason
        Returns list of game dicts with teams, pitchers, times.
        """
        if game_date is None:
            game_date = date.today()

        date_str = game_date.strftime("%Y-%m-%d")
        endpoint = f"/schedule?date={date_str}&sportId=1&hydrate=probablePitcher,team"
        data = self._fetch(endpoint)

        if not data or not data.get("dates"):
            return []

        games = []
        for game_data in data["dates"][0].get("games", []):
            gt = game_data.get("gameType", "R")
            # Accept requested game type, or all if none specified
            if game_type and gt != game_type:
                continue

            game_pk = game_data["gamePk"]
            away_team_data = game_data["teams"]["away"]
            home_team_data = game_data["teams"]["home"]

            away_id = away_team_data["team"]["id"]
            home_id = home_team_data["team"]["id"]

            away_abbr = TEAM_ID_TO_ABBR.get(away_id, "???")
            home_abbr = TEAM_ID_TO_ABBR.get(home_id, "???")

            # Probable pitchers
            away_pitcher = away_team_data.get("probablePitcher", {})
            home_pitcher = home_team_data.get("probablePitcher", {})

            game = {
                "game_pk": game_pk,
                "game_date": date_str,
                "game_time": game_data.get("gameDate", ""),
                "game_type": gt,
                "status": game_data.get("status", {}).get("detailedState", "Scheduled"),
                "away": {
                    "team_id": away_id,
                    "abbr": away_abbr,
                    "name": away_team_data["team"]["name"],
                    "record": away_team_data.get("leagueRecord", {}),
                    "probable_pitcher": {
                        "id": away_pitcher.get("id"),
                        "name": away_pitcher.get("fullName", "TBD"),
                    } if away_pitcher else {"id": None, "name": "TBD"},
                },
                "home": {
                    "team_id": home_id,
                    "abbr": home_abbr,
                    "name": home_team_data["team"]["name"],
                    "record": home_team_data.get("leagueRecord", {}),
                    "probable_pitcher": {
                        "id": home_pitcher.get("id"),
                        "name": home_pitcher.get("fullName", "TBD"),
                    } if home_pitcher else {"id": None, "name": "TBD"},
                },
                "venue": game_data.get("venue", {}).get("name", ""),
                "park_factor": PARK_FACTORS.get(home_abbr, 1.0),
            }
            games.append(game)

        return games

    def get_todays_schedule(self) -> List[Dict]:
        """Get today's schedule — tries regular season first, falls back to spring training."""
        games = self.get_schedule(game_type="R")
        if not games:
            games = self.get_schedule(game_type="S")
        return games

    # ── Lineups / Boxscore ────────────────────────────────────

    def get_lineup(self, game_pk: int) -> Dict[str, List[Dict]]:
        """
        Get lineups for a specific game.
        Returns {"away": [...players], "home": [...players]}
        Falls back to roster if lineups not yet posted.
        """
        endpoint = f"/game/{game_pk}/boxscore"
        data = self._fetch(endpoint, ttl=1800)  # 30 min cache for lineups

        if not data:
            return {"away": [], "home": []}

        result = {}
        for side in ("away", "home"):
            team_data = data.get("teams", {}).get(side, {})
            batters = team_data.get("batters", [])
            players = team_data.get("players", {})
            batting_order = team_data.get("battingOrder", batters)

            lineup = []
            for pid in batting_order:
                player_key = f"ID{pid}"
                player_data = players.get(player_key, {})
                person = player_data.get("person", {})
                position = player_data.get("position", {})

                if position.get("abbreviation") == "P":
                    continue  # Skip pitchers in batting lineup

                lineup.append({
                    "id": pid,
                    "name": person.get("fullName", f"Player {pid}"),
                    "position": position.get("abbreviation", ""),
                    "bat_side": player_data.get("batSide", {}).get("code", "R"),
                })

            result[side] = lineup[:9]  # Max 9 batters

        return result

    # ── Player Stats ──────────────────────────────────────────

    def get_player_hitting_stats(self, player_id: int,
                                  season: int = 2025) -> Optional[Dict]:
        """Get season hitting stats for a player."""
        endpoint = f"/people/{player_id}/stats?stats=season&season={season}&group=hitting"
        data = self._fetch(endpoint, ttl=OFFSEASON_CACHE_TTL)

        if not data:
            return None

        stats_list = data.get("stats", [])
        if not stats_list:
            return None

        splits = stats_list[0].get("splits", [])
        if not splits:
            return None

        stat = splits[0]["stat"]
        return {
            "player_id": player_id,
            "season": season,
            "games": stat.get("gamesPlayed", 0),
            "at_bats": stat.get("atBats", 0),
            "hits": stat.get("hits", 0),
            "avg": float(stat.get("avg", ".000").replace(".", "0.", 1) if stat.get("avg", ".000").startswith(".") else stat.get("avg", "0")),
            "obp": float(stat.get("obp", "0") or "0"),
            "slg": float(stat.get("slg", "0") or "0"),
            "ops": float(stat.get("ops", "0") or "0"),
            "home_runs": stat.get("homeRuns", 0),
            "doubles": stat.get("doubles", 0),
            "triples": stat.get("triples", 0),
            "stolen_bases": stat.get("stolenBases", 0),
            "strikeouts": stat.get("strikeOuts", 0),
            "walks": stat.get("baseOnBalls", 0),
            "plate_appearances": stat.get("plateAppearances", 0),
            "rbi": stat.get("rbi", 0),
            "runs": stat.get("runs", 0),
        }

    def get_player_pitching_stats(self, player_id: int,
                                   season: int = 2025) -> Optional[Dict]:
        """Get season pitching stats for a player."""
        endpoint = f"/people/{player_id}/stats?stats=season&season={season}&group=pitching"
        data = self._fetch(endpoint, ttl=OFFSEASON_CACHE_TTL)

        if not data:
            return None

        stats_list = data.get("stats", [])
        if not stats_list:
            return None

        splits = stats_list[0].get("splits", [])
        if not splits:
            return None

        stat = splits[0]["stat"]
        return {
            "player_id": player_id,
            "season": season,
            "games": stat.get("gamesPlayed", 0),
            "games_started": stat.get("gamesStarted", 0),
            "wins": stat.get("wins", 0),
            "losses": stat.get("losses", 0),
            "era": float(stat.get("era", "0") or "0"),
            "whip": float(stat.get("whip", "0") or "0"),
            "innings_pitched": float(stat.get("inningsPitched", "0") or "0"),
            "strikeouts": stat.get("strikeOuts", 0),
            "walks": stat.get("baseOnBalls", 0),
            "hits_allowed": stat.get("hits", 0),
            "home_runs_allowed": stat.get("homeRuns", 0),
            "k_per_9": float(stat.get("strikeoutsPer9Inn", "0") or "0"),
            "bb_per_9": float(stat.get("walksPer9Inn", "0") or "0"),
            "hr_per_9": float(stat.get("homeRunsPer9Inn", "0") or "0"),
            "avg_against": float(stat.get("avg", "0") or "0"),
            "pitch_hand": None,  # Will be populated from person endpoint
        }

    def get_player_splits(self, player_id: int, season: int = 2025,
                           group: str = "hitting") -> Dict[str, Any]:
        """
        Get platoon splits for a player (vs LHP / vs RHP).
        """
        endpoint = (f"/people/{player_id}/stats"
                     f"?stats=vsTeam5Y,statSplits&season={season}"
                     f"&group={group}&sitCodes=vl,vr")
        data = self._fetch(endpoint, ttl=OFFSEASON_CACHE_TTL)

        splits_result = {"vs_left": None, "vs_right": None}
        if not data:
            return splits_result

        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                split_name = split.get("split", {}).get("code", "")
                stat = split.get("stat", {})
                if split_name == "vl":
                    splits_result["vs_left"] = {
                        "avg": float(stat.get("avg", "0") or "0"),
                        "ops": float(stat.get("ops", "0") or "0"),
                        "at_bats": stat.get("atBats", 0),
                    }
                elif split_name == "vr":
                    splits_result["vs_right"] = {
                        "avg": float(stat.get("avg", "0") or "0"),
                        "ops": float(stat.get("ops", "0") or "0"),
                        "at_bats": stat.get("atBats", 0),
                    }

        return splits_result

    def get_player_info(self, player_id: int) -> Optional[Dict]:
        """Get player biographical info (bat side, pitch hand, etc.)."""
        endpoint = f"/people/{player_id}"
        data = self._fetch(endpoint, ttl=OFFSEASON_CACHE_TTL)

        if not data or not data.get("people"):
            return None

        person = data["people"][0]
        return {
            "id": player_id,
            "name": person.get("fullName", ""),
            "bat_side": person.get("batSide", {}).get("code", "R"),
            "pitch_hand": person.get("pitchHand", {}).get("code", "R"),
            "position": person.get("primaryPosition", {}).get("abbreviation", ""),
            "team_id": person.get("currentTeam", {}).get("id"),
            "team_abbr": TEAM_ID_TO_ABBR.get(
                person.get("currentTeam", {}).get("id", 0), "???"
            ),
            "age": person.get("currentAge", 0),
        }

    # ── Team Roster ───────────────────────────────────────────

    def get_team_roster(self, team_id: int, season: int = 2025) -> List[Dict]:
        """Get team's 40-man roster with player IDs."""
        endpoint = f"/teams/{team_id}/roster?rosterType=active&season={season}"
        data = self._fetch(endpoint, ttl=OFFSEASON_CACHE_TTL)

        if not data:
            return []

        roster = []
        for entry in data.get("roster", []):
            person = entry.get("person", {})
            position = entry.get("position", {})
            roster.append({
                "id": person.get("id"),
                "name": person.get("fullName", ""),
                "jersey": entry.get("jerseyNumber", ""),
                "position": position.get("abbreviation", ""),
                "position_type": position.get("type", ""),
                "status": entry.get("status", {}).get("code", "A"),
            })

        return roster

    # ── Composite: Today's Full Data ──────────────────────────

    def get_todays_full_data(self) -> Dict[str, Any]:
        """
        Master function: fetches today's schedule, lineups, and pitcher stats.
        Returns all data needed for hit/win models.
        """
        cache_key = f"full_data_{date.today().isoformat()}"
        cached = self._get_cached(cache_key, CACHE_TTL_SECONDS)
        if cached:
            return cached

        schedule = self.get_todays_schedule()
        is_regular_season = any(g["game_type"] == "R" for g in schedule)
        is_spring_training = any(g["game_type"] == "S" for g in schedule)

        result = {
            "date": date.today().isoformat(),
            "fetched_at": datetime.now().isoformat(),
            "is_regular_season": is_regular_season,
            "is_spring_training": is_spring_training,
            "total_games": len(schedule),
            "games": [],
        }

        for game in schedule[:16]:  # Cap at 16 games to limit API calls
            game_info = {
                **game,
                "away_lineup": [],
                "home_lineup": [],
                "away_pitcher_stats": None,
                "home_pitcher_stats": None,
            }

            # Get pitcher stats if IDs available
            away_pid = game["away"]["probable_pitcher"].get("id")
            home_pid = game["home"]["probable_pitcher"].get("id")

            if away_pid:
                pitcher_stats = self.get_player_pitching_stats(away_pid)
                pitcher_info = self.get_player_info(away_pid)
                if pitcher_stats and pitcher_info:
                    pitcher_stats["pitch_hand"] = pitcher_info.get("pitch_hand", "R")
                game_info["away_pitcher_stats"] = pitcher_stats

            if home_pid:
                pitcher_stats = self.get_player_pitching_stats(home_pid)
                pitcher_info = self.get_player_info(home_pid)
                if pitcher_stats and pitcher_info:
                    pitcher_stats["pitch_hand"] = pitcher_info.get("pitch_hand", "R")
                game_info["home_pitcher_stats"] = pitcher_stats

            # Try to get lineups (may not be available pre-game)
            try:
                lineups = self.get_lineup(game["game_pk"])
                game_info["away_lineup"] = lineups.get("away", [])
                game_info["home_lineup"] = lineups.get("home", [])
            except Exception as e:
                logger.warning(f"Could not fetch lineup for game {game['game_pk']}: {e}")

            result["games"].append(game_info)

        self._set_cache(cache_key, result)
        return result

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


# ── Demo Data for Off-Season ──────────────────────────────────

def get_demo_schedule() -> List[Dict]:
    """
    Generate demo schedule with realistic 2025 season data.
    Used during off-season to show how the system works.
    """
    today = date.today().isoformat()
    return [
        {
            "game_pk": 900001,
            "game_date": today,
            "game_time": f"{today}T23:10:00Z",
            "game_type": "DEMO",
            "status": "Preseason Projection",
            "away": {
                "team_id": 111, "abbr": "BOS", "name": "Boston Red Sox",
                "record": {"wins": 89, "losses": 73},
                "probable_pitcher": {"id": 678394, "name": "Garrett Crochet"},
            },
            "home": {
                "team_id": 147, "abbr": "NYY", "name": "New York Yankees",
                "record": {"wins": 95, "losses": 67},
                "probable_pitcher": {"id": 543037, "name": "Gerrit Cole"},
            },
            "venue": "Yankee Stadium",
            "park_factor": 1.00,
        },
        {
            "game_pk": 900002,
            "game_date": today,
            "game_time": f"{today}T23:40:00Z",
            "game_type": "DEMO",
            "status": "Preseason Projection",
            "away": {
                "team_id": 137, "abbr": "SF", "name": "San Francisco Giants",
                "record": {"wins": 80, "losses": 82},
                "probable_pitcher": {"id": 657277, "name": "Logan Webb"},
            },
            "home": {
                "team_id": 119, "abbr": "LAD", "name": "Los Angeles Dodgers",
                "record": {"wins": 98, "losses": 64},
                "probable_pitcher": {"id": 808967, "name": "Yoshinobu Yamamoto"},
            },
            "venue": "Dodger Stadium",
            "park_factor": 0.98,
        },
        {
            "game_pk": 900003,
            "game_date": today,
            "game_time": f"{today}T00:10:00Z",
            "game_type": "DEMO",
            "status": "Preseason Projection",
            "away": {
                "team_id": 140, "abbr": "TEX", "name": "Texas Rangers",
                "record": {"wins": 78, "losses": 84},
                "probable_pitcher": {"id": 621244, "name": "Nathan Eovaldi"},
            },
            "home": {
                "team_id": 117, "abbr": "HOU", "name": "Houston Astros",
                "record": {"wins": 88, "losses": 74},
                "probable_pitcher": {"id": 434378, "name": "Justin Verlander"},
            },
            "venue": "Minute Maid Park",
            "park_factor": 0.98,
        },
        {
            "game_pk": 900004,
            "game_date": today,
            "game_time": f"{today}T23:05:00Z",
            "game_type": "DEMO",
            "status": "Preseason Projection",
            "away": {
                "team_id": 143, "abbr": "PHI", "name": "Philadelphia Phillies",
                "record": {"wins": 95, "losses": 67},
                "probable_pitcher": {"id": 624133, "name": "Zack Wheeler"},
            },
            "home": {
                "team_id": 144, "abbr": "ATL", "name": "Atlanta Braves",
                "record": {"wins": 89, "losses": 73},
                "probable_pitcher": {"id": 612898, "name": "Chris Sale"},
            },
            "venue": "Truist Park",
            "park_factor": 1.02,
        },
        {
            "game_pk": 900005,
            "game_date": today,
            "game_time": f"{today}T01:10:00Z",
            "game_type": "DEMO",
            "status": "Preseason Projection",
            "away": {
                "team_id": 142, "abbr": "MIN", "name": "Minnesota Twins",
                "record": {"wins": 82, "losses": 80},
                "probable_pitcher": {"id": 656302, "name": "Pablo López"},
            },
            "home": {
                "team_id": 116, "abbr": "DET", "name": "Detroit Tigers",
                "record": {"wins": 87, "losses": 75},
                "probable_pitcher": {"id": 669373, "name": "Tarik Skubal"},
            },
            "venue": "Comerica Park",
            "park_factor": 0.99,
        },
    ]


def get_demo_lineups() -> Dict[int, Dict[str, List[Dict]]]:
    """Demo lineups for off-season display."""
    return {
        900001: {  # BOS @ NYY
            "away": [
                {"id": 646240, "name": "Jarren Duran", "position": "CF", "bat_side": "L"},
                {"id": 678394, "name": "Ceddanne Rafaela", "position": "SS", "bat_side": "R"},
                {"id": 680776, "name": "Rafael Devers", "position": "3B", "bat_side": "L"},
                {"id": 596019, "name": "Masataka Yoshida", "position": "LF", "bat_side": "L"},
                {"id": 605141, "name": "Triston Casas", "position": "1B", "bat_side": "L"},
                {"id": 608070, "name": "Tyler O'Neill", "position": "RF", "bat_side": "R"},
                {"id": 680916, "name": "Wilyer Abreu", "position": "DH", "bat_side": "L"},
                {"id": 666182, "name": "Reese McGuire", "position": "C", "bat_side": "L"},
                {"id": 672695, "name": "Enmanuel Valdez", "position": "2B", "bat_side": "L"},
            ],
            "home": [
                {"id": 660271, "name": "Juan Soto", "position": "RF", "bat_side": "L"},
                {"id": 592450, "name": "Aaron Judge", "position": "CF", "bat_side": "R"},
                {"id": 596115, "name": "Cody Bellinger", "position": "1B", "bat_side": "L"},
                {"id": 665489, "name": "Jazz Chisholm Jr.", "position": "3B", "bat_side": "L"},
                {"id": 673490, "name": "Austin Wells", "position": "C", "bat_side": "L"},
                {"id": 676391, "name": "Anthony Volpe", "position": "SS", "bat_side": "R"},
                {"id": 682998, "name": "Jasson Domínguez", "position": "LF", "bat_side": "S"},
                {"id": 608324, "name": "Giancarlo Stanton", "position": "DH", "bat_side": "R"},
                {"id": 666152, "name": "Gleyber Torres", "position": "2B", "bat_side": "R"},
            ],
        },
        900002: {  # SF @ LAD
            "away": [
                {"id": 571431, "name": "Jung Hoo Lee", "position": "CF", "bat_side": "L"},
                {"id": 605204, "name": "Matt Chapman", "position": "3B", "bat_side": "R"},
                {"id": 663993, "name": "Heliot Ramos", "position": "RF", "bat_side": "R"},
                {"id": 656024, "name": "LaMonte Wade Jr.", "position": "1B", "bat_side": "L"},
                {"id": 640461, "name": "Mike Yastrzemski", "position": "LF", "bat_side": "L"},
                {"id": 680811, "name": "Tyler Fitzgerald", "position": "SS", "bat_side": "R"},
                {"id": 608070, "name": "Patrick Bailey", "position": "C", "bat_side": "S"},
                {"id": 571657, "name": "Wilmer Flores", "position": "DH", "bat_side": "R"},
                {"id": 682928, "name": "Brett Wisely", "position": "2B", "bat_side": "R"},
            ],
            "home": [
                {"id": 660271, "name": "Shohei Ohtani", "position": "DH", "bat_side": "L"},
                {"id": 605141, "name": "Mookie Betts", "position": "SS", "bat_side": "R"},
                {"id": 518692, "name": "Freddie Freeman", "position": "1B", "bat_side": "L"},
                {"id": 665742, "name": "Teoscar Hernández", "position": "RF", "bat_side": "R"},
                {"id": 680776, "name": "Max Muncy", "position": "3B", "bat_side": "L"},
                {"id": 682928, "name": "Tommy Edman", "position": "CF", "bat_side": "S"},
                {"id": 660670, "name": "Will Smith", "position": "C", "bat_side": "R"},
                {"id": 673962, "name": "Andy Pages", "position": "LF", "bat_side": "R"},
                {"id": 666158, "name": "Gavin Lux", "position": "2B", "bat_side": "L"},
            ],
        },
        900003: {  # TEX @ HOU
            "away": [
                {"id": 665487, "name": "Marcus Semien", "position": "2B", "bat_side": "R"},
                {"id": 608369, "name": "Corey Seager", "position": "SS", "bat_side": "L"},
                {"id": 608070, "name": "Wyatt Langford", "position": "RF", "bat_side": "R"},
                {"id": 572041, "name": "Nathaniel Lowe", "position": "1B", "bat_side": "L"},
                {"id": 668804, "name": "Josh Jung", "position": "3B", "bat_side": "R"},
                {"id": 660271, "name": "Adolis García", "position": "CF", "bat_side": "R"},
                {"id": 606192, "name": "Jonah Heim", "position": "C", "bat_side": "S"},
                {"id": 642086, "name": "Leody Taveras", "position": "LF", "bat_side": "S"},
                {"id": 677951, "name": "Josh Smith", "position": "DH", "bat_side": "L"},
            ],
            "home": [
                {"id": 514888, "name": "Jose Altuve", "position": "2B", "bat_side": "R"},
                {"id": 670541, "name": "Yordan Alvarez", "position": "LF", "bat_side": "L"},
                {"id": 608070, "name": "Alex Bregman", "position": "3B", "bat_side": "R"},
                {"id": 670032, "name": "Kyle Tucker", "position": "RF", "bat_side": "L"},
                {"id": 677594, "name": "Yainer Diaz", "position": "C", "bat_side": "R"},
                {"id": 677588, "name": "Jeremy Peña", "position": "SS", "bat_side": "R"},
                {"id": 683734, "name": "Jake Meyers", "position": "CF", "bat_side": "R"},
                {"id": 596146, "name": "Jon Singleton", "position": "1B", "bat_side": "L"},
                {"id": 682136, "name": "Chas McCormick", "position": "DH", "bat_side": "R"},
            ],
        },
        900004: {  # PHI @ ATL
            "away": [
                {"id": 656305, "name": "Kyle Schwarber", "position": "LF", "bat_side": "L"},
                {"id": 656941, "name": "Trea Turner", "position": "SS", "bat_side": "R"},
                {"id": 592178, "name": "Bryce Harper", "position": "1B", "bat_side": "L"},
                {"id": 677951, "name": "Nick Castellanos", "position": "RF", "bat_side": "R"},
                {"id": 608070, "name": "Alec Bohm", "position": "3B", "bat_side": "R"},
                {"id": 656234, "name": "J.T. Realmuto", "position": "C", "bat_side": "R"},
                {"id": 680776, "name": "Brandon Marsh", "position": "CF", "bat_side": "L"},
                {"id": 673962, "name": "Bryson Stott", "position": "2B", "bat_side": "L"},
                {"id": 671096, "name": "Johan Rojas", "position": "DH", "bat_side": "R"},
            ],
            "home": [
                {"id": 660670, "name": "Ronald Acuña Jr.", "position": "RF", "bat_side": "R"},
                {"id": 656305, "name": "Ozzie Albies", "position": "2B", "bat_side": "S"},
                {"id": 608336, "name": "Matt Olson", "position": "1B", "bat_side": "L"},
                {"id": 518735, "name": "Marcell Ozuna", "position": "DH", "bat_side": "R"},
                {"id": 608070, "name": "Austin Riley", "position": "3B", "bat_side": "R"},
                {"id": 680776, "name": "Michael Harris II", "position": "CF", "bat_side": "L"},
                {"id": 680811, "name": "Orlando Arcia", "position": "SS", "bat_side": "R"},
                {"id": 608070, "name": "Sean Murphy", "position": "C", "bat_side": "R"},
                {"id": 670541, "name": "Jarred Kelenic", "position": "LF", "bat_side": "L"},
            ],
        },
        900005: {  # MIN @ DET
            "away": [
                {"id": 608070, "name": "Carlos Correa", "position": "SS", "bat_side": "R"},
                {"id": 680776, "name": "Byron Buxton", "position": "CF", "bat_side": "R"},
                {"id": 670541, "name": "Royce Lewis", "position": "3B", "bat_side": "R"},
                {"id": 682928, "name": "Willi Castro", "position": "LF", "bat_side": "S"},
                {"id": 608369, "name": "Ryan Jeffers", "position": "C", "bat_side": "R"},
                {"id": 677594, "name": "Matt Wallner", "position": "RF", "bat_side": "L"},
                {"id": 656305, "name": "Alex Kirilloff", "position": "1B", "bat_side": "L"},
                {"id": 680811, "name": "Edouard Julien", "position": "2B", "bat_side": "L"},
                {"id": 671096, "name": "Trevor Larnach", "position": "DH", "bat_side": "L"},
            ],
            "home": [
                {"id": 681481, "name": "Riley Greene", "position": "CF", "bat_side": "L"},
                {"id": 682998, "name": "Colt Keith", "position": "2B", "bat_side": "L"},
                {"id": 680776, "name": "Spencer Torkelson", "position": "1B", "bat_side": "R"},
                {"id": 671096, "name": "Kerry Carpenter", "position": "LF", "bat_side": "L"},
                {"id": 670541, "name": "Matt Vierling", "position": "RF", "bat_side": "R"},
                {"id": 682928, "name": "Javier Báez", "position": "SS", "bat_side": "R"},
                {"id": 608070, "name": "Dillon Dingler", "position": "C", "bat_side": "R"},
                {"id": 677594, "name": "Trey Sweeney", "position": "3B", "bat_side": "L"},
                {"id": 656305, "name": "Parker Meadows", "position": "DH", "bat_side": "L"},
            ],
        },
    }


def get_demo_pitcher_stats() -> Dict[str, Dict]:
    """Demo pitcher stats for off-season projections (based on 2025 data)."""
    return {
        "Gerrit Cole": {
            "era": 3.41, "whip": 1.08, "k_per_9": 10.8, "bb_per_9": 2.1,
            "hr_per_9": 1.1, "wins": 14, "losses": 6, "innings_pitched": 172.1,
            "pitch_hand": "R", "avg_against": 0.215,
        },
        "Garrett Crochet": {
            "era": 3.58, "whip": 1.10, "k_per_9": 12.3, "bb_per_9": 2.8,
            "hr_per_9": 0.9, "wins": 11, "losses": 8, "innings_pitched": 148.0,
            "pitch_hand": "L", "avg_against": 0.209,
        },
        "Logan Webb": {
            "era": 3.24, "whip": 1.15, "k_per_9": 7.8, "bb_per_9": 2.0,
            "hr_per_9": 0.8, "wins": 13, "losses": 9, "innings_pitched": 210.0,
            "pitch_hand": "R", "avg_against": 0.240,
        },
        "Yoshinobu Yamamoto": {
            "era": 2.78, "whip": 1.02, "k_per_9": 9.6, "bb_per_9": 2.2,
            "hr_per_9": 1.0, "wins": 10, "losses": 4, "innings_pitched": 132.0,
            "pitch_hand": "R", "avg_against": 0.201,
        },
        "Nathan Eovaldi": {
            "era": 3.82, "whip": 1.18, "k_per_9": 8.2, "bb_per_9": 2.3,
            "hr_per_9": 1.2, "wins": 12, "losses": 10, "innings_pitched": 175.0,
            "pitch_hand": "R", "avg_against": 0.244,
        },
        "Justin Verlander": {
            "era": 3.96, "whip": 1.22, "k_per_9": 8.0, "bb_per_9": 2.5,
            "hr_per_9": 1.3, "wins": 9, "losses": 9, "innings_pitched": 155.0,
            "pitch_hand": "R", "avg_against": 0.248,
        },
        "Zack Wheeler": {
            "era": 2.57, "whip": 0.96, "k_per_9": 10.5, "bb_per_9": 1.8,
            "hr_per_9": 0.7, "wins": 16, "losses": 7, "innings_pitched": 200.0,
            "pitch_hand": "R", "avg_against": 0.198,
        },
        "Chris Sale": {
            "era": 2.38, "whip": 0.91, "k_per_9": 11.2, "bb_per_9": 2.0,
            "hr_per_9": 0.9, "wins": 18, "losses": 3, "innings_pitched": 177.2,
            "pitch_hand": "L", "avg_against": 0.192,
        },
        "Pablo López": {
            "era": 4.08, "whip": 1.25, "k_per_9": 8.5, "bb_per_9": 2.8,
            "hr_per_9": 1.1, "wins": 10, "losses": 12, "innings_pitched": 168.0,
            "pitch_hand": "R", "avg_against": 0.251,
        },
        "Tarik Skubal": {
            "era": 2.21, "whip": 0.89, "k_per_9": 11.1, "bb_per_9": 1.9,
            "hr_per_9": 0.7, "wins": 18, "losses": 4, "innings_pitched": 192.0,
            "pitch_hand": "L", "avg_against": 0.188,
        },
    }


def get_demo_batter_stats() -> Dict[str, Dict]:
    """Demo batter stats for off-season projections (based on 2025 data)."""
    return {
        "Aaron Judge": {"avg": 0.282, "obp": 0.392, "slg": 0.622, "ops": 1.014, "hr": 55, "bat_side": "R"},
        "Juan Soto": {"avg": 0.288, "obp": 0.419, "slg": 0.569, "ops": 0.988, "hr": 41, "bat_side": "L"},
        "Shohei Ohtani": {"avg": 0.310, "obp": 0.390, "slg": 0.646, "ops": 1.036, "hr": 54, "bat_side": "L"},
        "Mookie Betts": {"avg": 0.291, "obp": 0.378, "slg": 0.543, "ops": 0.921, "hr": 28, "bat_side": "R"},
        "Freddie Freeman": {"avg": 0.282, "obp": 0.378, "slg": 0.516, "ops": 0.894, "hr": 25, "bat_side": "L"},
        "Bryce Harper": {"avg": 0.290, "obp": 0.399, "slg": 0.558, "ops": 0.957, "hr": 30, "bat_side": "L"},
        "Corey Seager": {"avg": 0.277, "obp": 0.355, "slg": 0.509, "ops": 0.864, "hr": 30, "bat_side": "L"},
        "Trea Turner": {"avg": 0.280, "obp": 0.340, "slg": 0.475, "ops": 0.815, "hr": 21, "bat_side": "R"},
        "Yordan Alvarez": {"avg": 0.293, "obp": 0.389, "slg": 0.583, "ops": 0.972, "hr": 35, "bat_side": "L"},
        "Kyle Tucker": {"avg": 0.289, "obp": 0.392, "slg": 0.575, "ops": 0.967, "hr": 30, "bat_side": "L"},
        "Rafael Devers": {"avg": 0.272, "obp": 0.345, "slg": 0.515, "ops": 0.860, "hr": 28, "bat_side": "L"},
        "Jose Altuve": {"avg": 0.295, "obp": 0.365, "slg": 0.475, "ops": 0.840, "hr": 18, "bat_side": "R"},
        "Matt Olson": {"avg": 0.248, "obp": 0.345, "slg": 0.480, "ops": 0.825, "hr": 27, "bat_side": "L"},
        "Ronald Acuña Jr.": {"avg": 0.285, "obp": 0.370, "slg": 0.535, "ops": 0.905, "hr": 25, "bat_side": "R"},
        "Marcell Ozuna": {"avg": 0.302, "obp": 0.378, "slg": 0.580, "ops": 0.958, "hr": 39, "bat_side": "R"},
        "Marcus Semien": {"avg": 0.268, "obp": 0.345, "slg": 0.465, "ops": 0.810, "hr": 24, "bat_side": "R"},
        "Riley Greene": {"avg": 0.262, "obp": 0.348, "slg": 0.458, "ops": 0.806, "hr": 22, "bat_side": "L"},
        "Jazz Chisholm Jr.": {"avg": 0.256, "obp": 0.321, "slg": 0.475, "ops": 0.796, "hr": 24, "bat_side": "L"},
        "Jarren Duran": {"avg": 0.285, "obp": 0.341, "slg": 0.492, "ops": 0.833, "hr": 21, "bat_side": "L"},
        "Bobby Witt Jr.": {"avg": 0.332, "obp": 0.389, "slg": 0.588, "ops": 0.977, "hr": 32, "bat_side": "R"},
        "Gunnar Henderson": {"avg": 0.265, "obp": 0.352, "slg": 0.510, "ops": 0.862, "hr": 37, "bat_side": "L"},
        "Kyle Schwarber": {"avg": 0.250, "obp": 0.375, "slg": 0.525, "ops": 0.900, "hr": 38, "bat_side": "L"},
        "Giancarlo Stanton": {"avg": 0.233, "obp": 0.315, "slg": 0.478, "ops": 0.793, "hr": 27, "bat_side": "R"},
        "Matt Chapman": {"avg": 0.262, "obp": 0.340, "slg": 0.475, "ops": 0.815, "hr": 27, "bat_side": "R"},
        "Austin Wells": {"avg": 0.259, "obp": 0.342, "slg": 0.452, "ops": 0.794, "hr": 17, "bat_side": "L"},
        "Anthony Volpe": {"avg": 0.253, "obp": 0.315, "slg": 0.418, "ops": 0.733, "hr": 15, "bat_side": "R"},
    }


# Singleton
mlb_live = MLBLiveData()
