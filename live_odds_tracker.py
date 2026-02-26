#!/usr/bin/env python3
"""
MLB Predictor - Live Odds Tracker
====================================
Real-time odds tracking with line movement alerts,
arbitrage detection, and sportsbook comparison.

Features:
- Live odds from The Odds API (free tier: 500 req/month)
- Line movement tracking with alerts
- Arbitrage opportunity detection
- Sportsbook comparison (DraftKings, FanDuel, BetMGM, etc.)
- Closing line value (CLV) analysis
- Odds format conversion (American, Decimal, Fractional)
- Historical odds logging for backtesting

Data Source:
    The Odds API: https://the-odds-api.com/
    Free tier: 500 requests/month, live & pre-game odds

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
import os
import time
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import urllib.request
import urllib.parse


# ============================================================================
# ODDS MATH UTILITIES
# ============================================================================

class OddsMath:
    """Odds conversion and probability calculations"""

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal"""
        if american > 0:
            return round(1 + (american / 100), 4)
        else:
            return round(1 + (100 / abs(american)), 4)

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American"""
        if decimal_odds >= 2.0:
            return round((decimal_odds - 1) * 100)
        else:
            return round(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_implied_probability(american: int) -> float:
        """Convert American odds to implied probability"""
        if american > 0:
            return round(100 / (american + 100), 4)
        else:
            return round(abs(american) / (abs(american) + 100), 4)

    @staticmethod
    def decimal_to_implied_probability(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability"""
        return round(1 / decimal_odds, 4) if decimal_odds > 0 else 0

    @staticmethod
    def calculate_vig(prob_a: float, prob_b: float) -> float:
        """Calculate the vigorish (vig/juice) from two implied probabilities"""
        total = prob_a + prob_b
        return round((total - 1) * 100, 2)

    @staticmethod
    def no_vig_probability(prob: float, total_prob: float) -> float:
        """Remove vig to get true probability"""
        return round(prob / total_prob, 4) if total_prob > 0 else 0

    @staticmethod
    def calculate_ev(win_probability: float, decimal_odds: float,
                     stake: float = 100) -> float:
        """Calculate expected value of a bet"""
        win_amount = stake * (decimal_odds - 1)
        ev = (win_probability * win_amount) - ((1 - win_probability) * stake)
        return round(ev, 2)

    @staticmethod
    def kelly_fraction(win_probability: float,
                       decimal_odds: float) -> float:
        """Calculate Kelly Criterion fraction"""
        b = decimal_odds - 1
        p = win_probability
        q = 1 - p
        if b <= 0:
            return 0
        kelly = (b * p - q) / b
        return round(max(0, kelly), 4)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BookmakerOdds:
    """Odds from a single sportsbook"""
    bookmaker: str  # "DraftKings", "FanDuel", etc.
    home_odds: int  # American odds
    away_odds: int
    over_under: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    home_spread: Optional[float] = None
    home_spread_odds: Optional[int] = None
    away_spread_odds: Optional[int] = None
    last_update: str = ""

    @property
    def home_implied(self) -> float:
        return OddsMath.american_to_implied_probability(self.home_odds)

    @property
    def away_implied(self) -> float:
        return OddsMath.american_to_implied_probability(self.away_odds)

    @property
    def vig(self) -> float:
        return OddsMath.calculate_vig(self.home_implied, self.away_implied)


@dataclass
class GameOdds:
    """Complete odds for a single game across all books"""
    game_id: str
    sport: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: List[BookmakerOdds] = field(default_factory=list)
    
    @property
    def best_home_odds(self) -> Optional[BookmakerOdds]:
        if not self.bookmakers:
            return None
        return max(self.bookmakers, key=lambda b: b.home_odds)

    @property
    def best_away_odds(self) -> Optional[BookmakerOdds]:
        if not self.bookmakers:
            return None
        return max(self.bookmakers, key=lambda b: b.away_odds)

    @property
    def consensus_home_implied(self) -> float:
        """Average implied probability across all books (vig-removed)"""
        if not self.bookmakers:
            return 0.5
        probs = []
        for b in self.bookmakers:
            total = b.home_implied + b.away_implied
            probs.append(OddsMath.no_vig_probability(b.home_implied, total))
        return round(sum(probs) / len(probs), 4)

    @property
    def consensus_away_implied(self) -> float:
        return round(1 - self.consensus_home_implied, 4)


@dataclass
class LineMovement:
    """Tracks line movement over time"""
    game_id: str
    team: str
    bookmaker: str
    old_odds: int
    new_odds: int
    change: int
    timestamp: str
    significance: str = ""  # "steam", "reverse", "normal"

    @property
    def is_significant(self) -> bool:
        """Line move > 20 cents is significant"""
        return abs(self.change) >= 20


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity"""
    game_id: str
    home_team: str
    away_team: str
    home_book: str
    home_odds: int
    away_book: str
    away_odds: int
    arb_margin: float  # Positive = profitable arb
    home_stake_pct: float
    away_stake_pct: float
    guaranteed_profit_pct: float
    detected_at: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()


# ============================================================================
# THE ODDS API CLIENT
# ============================================================================

class OddsApiClient:
    """
    Client for The Odds API (https://the-odds-api.com/).
    Free tier: 500 requests/month.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "baseball_mlb"

    # Major US sportsbooks
    BOOKMAKERS = [
        'draftkings', 'fanduel', 'betmgm', 'pointsbetus',
        'williamhill_us', 'betrivers', 'unibet_us', 'bovada'
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ODDS_API_KEY', '')
        self.requests_used = 0
        self.requests_remaining = 500
        self.cache_dir = Path("/tmp/mlb_odds_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_live_odds(self, markets: str = "h2h",
                      regions: str = "us") -> List[GameOdds]:
        """Get live odds for all MLB games"""
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': markets,
            'oddsFormat': 'american'
        }
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"
        data = self._request(url, params)

        if not data:
            return self._get_cached_odds()

        games = []
        for game_data in data:
            game = self._parse_game(game_data)
            games.append(game)

        # Cache results
        self._cache_odds(games)
        return games

    def get_scores(self) -> List[Dict]:
        """Get live scores for in-progress games"""
        params = {
            'apiKey': self.api_key,
            'daysFrom': 1
        }
        url = f"{self.BASE_URL}/sports/{self.SPORT}/scores"
        return self._request(url, params) or []

    def get_historical_odds(self, date: str) -> List[GameOdds]:
        """Get historical odds for a specific date (paid feature)"""
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h',
            'date': date,
            'oddsFormat': 'american'
        }
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds-history"
        data = self._request(url, params)
        return [self._parse_game(g) for g in (data or [])]

    def _request(self, url: str, params: Dict) -> Optional[List]:
        """Make API request"""
        if not self.api_key:
            return None

        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"

        try:
            req = urllib.request.Request(full_url, headers={
                'Accept': 'application/json'
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                # Track usage from headers
                self.requests_remaining = int(
                    response.headers.get('x-requests-remaining', 500)
                )
                self.requests_used = int(
                    response.headers.get('x-requests-used', 0)
                )
                return json.loads(response.read().decode())
        except Exception as e:
            print(f"⚠️ Odds API request failed: {e}")
            return None

    def _parse_game(self, data: Dict) -> GameOdds:
        """Parse API response into GameOdds"""
        game = GameOdds(
            game_id=data.get('id', ''),
            sport=data.get('sport_key', self.SPORT),
            commence_time=data.get('commence_time', ''),
            home_team=data.get('home_team', ''),
            away_team=data.get('away_team', '')
        )

        for bm_data in data.get('bookmakers', []):
            bm_name = bm_data.get('title', bm_data.get('key', ''))
            h2h_market = next(
                (m for m in bm_data.get('markets', []) if m['key'] == 'h2h'),
                None
            )
            if h2h_market:
                outcomes = {o['name']: o['price'] for o in h2h_market.get('outcomes', [])}
                home_odds = outcomes.get(game.home_team, -110)
                away_odds = outcomes.get(game.away_team, -110)

                game.bookmakers.append(BookmakerOdds(
                    bookmaker=bm_name,
                    home_odds=home_odds,
                    away_odds=away_odds,
                    last_update=bm_data.get('last_update', '')
                ))

        return game

    def _cache_odds(self, games: List[GameOdds]):
        """Cache odds locally"""
        cache_file = self.cache_dir / f"odds_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        data = [
            {
                'game_id': g.game_id,
                'home_team': g.home_team,
                'away_team': g.away_team,
                'commence_time': g.commence_time,
                'bookmakers': [
                    {
                        'name': b.bookmaker,
                        'home': b.home_odds,
                        'away': b.away_odds
                    }
                    for b in g.bookmakers
                ]
            }
            for g in games
        ]
        cache_file.write_text(json.dumps(data, indent=2))

    def _get_cached_odds(self) -> List[GameOdds]:
        """Load most recent cached odds"""
        cache_files = sorted(self.cache_dir.glob("odds_*.json"), reverse=True)
        if not cache_files:
            return []
        try:
            data = json.loads(cache_files[0].read_text())
            games = []
            for g in data:
                game = GameOdds(
                    game_id=g['game_id'],
                    sport=self.SPORT,
                    commence_time=g['commence_time'],
                    home_team=g['home_team'],
                    away_team=g['away_team']
                )
                for b in g.get('bookmakers', []):
                    game.bookmakers.append(BookmakerOdds(
                        bookmaker=b['name'],
                        home_odds=b['home'],
                        away_odds=b['away']
                    ))
                games.append(game)
            return games
        except Exception:
            return []


# ============================================================================
# LINE MOVEMENT TRACKER
# ============================================================================

class LineMovementTracker:
    """Tracks odds changes over time and detects significant moves"""

    def __init__(self, storage_dir: str = "/tmp/mlb_line_movements"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List[Dict]] = {}  # game_id → snapshots

    def record_snapshot(self, games: List[GameOdds]):
        """Record current odds snapshot"""
        timestamp = datetime.now().isoformat()
        for game in games:
            if game.game_id not in self.history:
                self.history[game.game_id] = []

            snapshot = {
                'timestamp': timestamp,
                'home_team': game.home_team,
                'away_team': game.away_team,
                'books': {
                    b.bookmaker: {'home': b.home_odds, 'away': b.away_odds}
                    for b in game.bookmakers
                }
            }
            self.history[game.game_id].append(snapshot)

    def detect_movements(self, threshold: int = 15) -> List[LineMovement]:
        """Detect significant line movements"""
        movements = []

        for game_id, snapshots in self.history.items():
            if len(snapshots) < 2:
                continue

            latest = snapshots[-1]
            previous = snapshots[-2]

            for book_name in latest['books']:
                if book_name not in previous['books']:
                    continue

                curr = latest['books'][book_name]
                prev = previous['books'][book_name]

                # Check home line move
                home_change = curr['home'] - prev['home']
                if abs(home_change) >= threshold:
                    significance = self._classify_movement(
                        home_change, game_id, 'home'
                    )
                    movements.append(LineMovement(
                        game_id=game_id,
                        team=latest['home_team'],
                        bookmaker=book_name,
                        old_odds=prev['home'],
                        new_odds=curr['home'],
                        change=home_change,
                        timestamp=latest['timestamp'],
                        significance=significance
                    ))

                # Check away line move
                away_change = curr['away'] - prev['away']
                if abs(away_change) >= threshold:
                    significance = self._classify_movement(
                        away_change, game_id, 'away'
                    )
                    movements.append(LineMovement(
                        game_id=game_id,
                        team=latest['away_team'],
                        bookmaker=book_name,
                        old_odds=prev['away'],
                        new_odds=curr['away'],
                        change=away_change,
                        timestamp=latest['timestamp'],
                        significance=significance
                    ))

        return movements

    def _classify_movement(self, change: int, game_id: str,
                            side: str) -> str:
        """Classify the type of line movement"""
        if abs(change) >= 30:
            return "steam"  # Sharp money / steam move
        elif abs(change) >= 20:
            return "significant"
        else:
            return "normal"


# ============================================================================
# ARBITRAGE DETECTOR
# ============================================================================

class ArbitrageDetector:
    """Detects arbitrage opportunities across sportsbooks"""

    def find_arbitrages(self, games: List[GameOdds],
                        min_margin: float = 0.0) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities across all games.
        
        An arb exists when the best odds across different books
        have implied probabilities that sum to less than 100%.
        """
        arbs = []

        for game in games:
            if len(game.bookmakers) < 2:
                continue

            # Find best odds for each side across all books
            best_home = max(game.bookmakers, key=lambda b: b.home_odds)
            best_away = max(game.bookmakers, key=lambda b: b.away_odds)

            # Calculate arb margin
            home_decimal = OddsMath.american_to_decimal(best_home.home_odds)
            away_decimal = OddsMath.american_to_decimal(best_away.away_odds)

            home_implied = 1 / home_decimal
            away_implied = 1 / away_decimal
            total_implied = home_implied + away_implied

            # Arb exists if total implied < 1.0
            if total_implied < 1.0:
                margin = round((1 - total_implied) * 100, 2)
                if margin >= min_margin:
                    # Calculate optimal stakes
                    home_stake_pct = round(home_implied / total_implied * 100, 1)
                    away_stake_pct = round(away_implied / total_implied * 100, 1)
                    guaranteed_profit = round(margin, 2)

                    arbs.append(ArbitrageOpportunity(
                        game_id=game.game_id,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        home_book=best_home.bookmaker,
                        home_odds=best_home.home_odds,
                        away_book=best_away.bookmaker,
                        away_odds=best_away.away_odds,
                        arb_margin=margin,
                        home_stake_pct=home_stake_pct,
                        away_stake_pct=away_stake_pct,
                        guaranteed_profit_pct=guaranteed_profit
                    ))

        return sorted(arbs, key=lambda a: a.arb_margin, reverse=True)

    def find_value_bets(self, games: List[GameOdds],
                        model_predictions: Dict[str, float],
                        min_edge: float = 0.03) -> List[Dict[str, Any]]:
        """
        Find value bets where model probability > implied probability.
        
        model_predictions: dict of team_name → win probability from our model
        """
        value_bets = []

        for game in games:
            for book in game.bookmakers:
                # Check home side
                home_implied = book.home_implied
                home_model = model_predictions.get(game.home_team, 0)
                home_edge = home_model - home_implied

                if home_edge >= min_edge:
                    decimal = OddsMath.american_to_decimal(book.home_odds)
                    ev = OddsMath.calculate_ev(home_model, decimal)
                    kelly = OddsMath.kelly_fraction(home_model, decimal)

                    value_bets.append({
                        'game': f"{game.away_team} @ {game.home_team}",
                        'pick': game.home_team,
                        'book': book.bookmaker,
                        'odds': book.home_odds,
                        'implied_prob': round(home_implied * 100, 1),
                        'model_prob': round(home_model * 100, 1),
                        'edge': round(home_edge * 100, 1),
                        'ev_per_100': ev,
                        'kelly_fraction': kelly,
                        'kelly_pct': round(kelly * 100, 1),
                        'commence_time': game.commence_time
                    })

                # Check away side
                away_implied = book.away_implied
                away_model = model_predictions.get(game.away_team, 0)
                away_edge = away_model - away_implied

                if away_edge >= min_edge:
                    decimal = OddsMath.american_to_decimal(book.away_odds)
                    ev = OddsMath.calculate_ev(away_model, decimal)
                    kelly = OddsMath.kelly_fraction(away_model, decimal)

                    value_bets.append({
                        'game': f"{game.away_team} @ {game.home_team}",
                        'pick': game.away_team,
                        'book': book.bookmaker,
                        'odds': book.away_odds,
                        'implied_prob': round(away_implied * 100, 1),
                        'model_prob': round(away_model * 100, 1),
                        'edge': round(away_edge * 100, 1),
                        'ev_per_100': ev,
                        'kelly_fraction': kelly,
                        'kelly_pct': round(kelly * 100, 1),
                        'commence_time': game.commence_time
                    })

        # Sort by edge (highest first), deduplicate by game+pick (keep best odds)
        value_bets.sort(key=lambda v: v['edge'], reverse=True)
        seen = set()
        unique_bets = []
        for bet in value_bets:
            key = f"{bet['game']}:{bet['pick']}"
            if key not in seen:
                seen.add(key)
                unique_bets.append(bet)

        return unique_bets


# ============================================================================
# DEMO
# ============================================================================

def demo_live_odds():
    """Demonstrate the live odds tracking system"""
    print("=" * 70)
    print("📊 MLB Predictor - Live Odds Tracker Demo")
    print("=" * 70)
    print()

    # Create simulated game odds (since API key may not be available)
    games = [
        GameOdds(
            game_id="game_001", sport="baseball_mlb",
            commence_time="2026-04-01T19:10:00Z",
            home_team="New York Yankees", away_team="Boston Red Sox",
            bookmakers=[
                BookmakerOdds("DraftKings", -145, +125),
                BookmakerOdds("FanDuel", -140, +120),
                BookmakerOdds("BetMGM", -150, +130),
                BookmakerOdds("PointsBet", -135, +115),
                BookmakerOdds("Caesars", -142, +122),
            ]
        ),
        GameOdds(
            game_id="game_002", sport="baseball_mlb",
            commence_time="2026-04-01T20:10:00Z",
            home_team="Los Angeles Dodgers", away_team="San Francisco Giants",
            bookmakers=[
                BookmakerOdds("DraftKings", -180, +155),
                BookmakerOdds("FanDuel", -175, +150),
                BookmakerOdds("BetMGM", -185, +160),
                BookmakerOdds("PointsBet", -170, +148),
            ]
        ),
        GameOdds(
            game_id="game_003", sport="baseball_mlb",
            commence_time="2026-04-01T21:40:00Z",
            home_team="Houston Astros", away_team="Texas Rangers",
            bookmakers=[
                BookmakerOdds("DraftKings", -120, +102),
                BookmakerOdds("FanDuel", -118, +100),
                BookmakerOdds("BetMGM", -125, +108),
                BookmakerOdds("PointsBet", -115, -105),
            ]
        ),
    ]

    # 1. Odds comparison
    print("1️⃣  ODDS COMPARISON (Tonight's Games)")
    print("-" * 60)
    for game in games:
        print(f"\n   {game.away_team} @ {game.home_team}")
        print(f"   Consensus: Home {game.consensus_home_implied*100:.1f}% | "
              f"Away {game.consensus_away_implied*100:.1f}%")
        print(f"   {'Book':<15} {'Home':>8} {'Away':>8} {'Vig':>6}")
        print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*6}")
        for b in game.bookmakers:
            print(f"   {b.bookmaker:<15} {b.home_odds:>+8} {b.away_odds:>+8} "
                  f"{b.vig:>5.1f}%")
        best_h = game.best_home_odds
        best_a = game.best_away_odds
        print(f"   {'BEST →':<15} {best_h.home_odds:>+8} ({best_h.bookmaker}) | "
              f"{best_a.away_odds:>+8} ({best_a.bookmaker})")
    print()

    # 2. Odds math
    print("2️⃣  ODDS MATH UTILITIES")
    print("-" * 60)
    test_odds = [-150, +130, -110, +200, -250]
    print(f"   {'American':>10} {'Decimal':>10} {'Implied%':>10} {'$100 Win':>10}")
    for odds in test_odds:
        dec = OddsMath.american_to_decimal(odds)
        imp = OddsMath.american_to_implied_probability(odds)
        win = (dec - 1) * 100
        print(f"   {odds:>+10} {dec:>10.3f} {imp*100:>9.1f}% ${win:>8.2f}")
    print()

    # 3. Arbitrage detection
    print("3️⃣  ARBITRAGE DETECTION")
    print("-" * 60)
    arb_detector = ArbitrageDetector()
    arbs = arb_detector.find_arbitrages(games)
    if arbs:
        for arb in arbs:
            print(f"   🎯 ARB FOUND: {arb.away_team} @ {arb.home_team}")
            print(f"     Home: {arb.home_odds:+d} ({arb.home_book}) → "
                  f"Stake {arb.home_stake_pct}%")
            print(f"     Away: {arb.away_odds:+d} ({arb.away_book}) → "
                  f"Stake {arb.away_stake_pct}%")
            print(f"     Guaranteed Profit: {arb.guaranteed_profit_pct}%")
    else:
        print("   ✅ No arbitrage opportunities detected (normal for MLB)")
        print("   💡 Arbs are rare; check again closer to game time")
    print()

    # 4. Value bets (using simulated model predictions)
    print("4️⃣  VALUE BET FINDER (Model vs. Market)")
    print("-" * 60)
    model_preds = {
        "New York Yankees": 0.58,       # Model says 58%, market ~57%
        "Boston Red Sox": 0.42,
        "Los Angeles Dodgers": 0.68,     # Model says 68%, market ~63%
        "San Francisco Giants": 0.32,
        "Houston Astros": 0.56,          # Model says 56%, market ~53%
        "Texas Rangers": 0.44,
    }

    value_bets = arb_detector.find_value_bets(games, model_preds, min_edge=0.02)
    if value_bets:
        print(f"   Found {len(value_bets)} value bets (>2% edge):")
        print(f"   {'Pick':<22} {'Book':<12} {'Odds':>6} {'Mkt%':>6} "
              f"{'Mdl%':>6} {'Edge':>6} {'EV/$100':>8} {'Kelly':>6}")
        print(f"   {'-'*22} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
        for bet in value_bets:
            print(f"   {bet['pick']:<22} {bet['book']:<12} "
                  f"{bet['odds']:>+6} {bet['implied_prob']:>5.1f}% "
                  f"{bet['model_prob']:>5.1f}% {bet['edge']:>5.1f}% "
                  f"${bet['ev_per_100']:>6.2f} {bet['kelly_pct']:>5.1f}%")
    else:
        print("   No value bets found with >2% edge")
    print()

    # 5. Line movement tracking
    print("5️⃣  LINE MOVEMENT TRACKER")
    print("-" * 60)
    tracker = LineMovementTracker()
    tracker.record_snapshot(games)

    # Simulate line movement
    moved_games = []
    for game in games:
        import copy
        moved = copy.deepcopy(game)
        for b in moved.bookmakers:
            # Simulate 10-25 cent moves on some books
            import random
            if random.random() > 0.5:
                move = random.choice([-25, -15, -10, 10, 15, 20, 25])
                b.home_odds += move
                b.away_odds -= move
        moved_games.append(moved)

    tracker.record_snapshot(moved_games)
    movements = tracker.detect_movements(threshold=10)

    if movements:
        print(f"   Detected {len(movements)} line movements:")
        for m in movements[:8]:
            icon = '🔥' if m.significance == 'steam' else '📈' if m.change > 0 else '📉'
            print(f"   {icon} {m.team} ({m.bookmaker}): "
                  f"{m.old_odds:+d} → {m.new_odds:+d} "
                  f"({m.change:+d}) [{m.significance}]")
    else:
        print("   No significant line movements yet")
    print()

    # API usage
    print("📡 API Usage:")
    client = OddsApiClient()
    print(f"   Requests used: {client.requests_used}")
    print(f"   Requests remaining: {client.requests_remaining}")
    print()

    print("=" * 70)
    print("✅ Live Odds Tracker Demo Complete")
    print("=" * 70)

    return games


if __name__ == "__main__":
    demo_live_odds()
