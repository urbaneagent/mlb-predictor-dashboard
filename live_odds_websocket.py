"""
MLB Predictor - Live Odds WebSocket Feed
Real-time odds streaming with line movement detection, sharp money alerts,
and multi-book comparison. Supports DraftKings, FanDuel, BetMGM, Caesars, PointsBet.
"""

import asyncio
import json
import time
import uuid
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import deque


# ──────────────────────────────────────────────
# Sportsbook & Market Models
# ──────────────────────────────────────────────

class Sportsbook(Enum):
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"
    BETMGM = "betmgm"
    CAESARS = "caesars"
    POINTSBET = "pointsbet"
    BETRIVERS = "betrivers"
    WYNN = "wynn"
    PINNACLE = "pinnacle"  # Sharp book


class MarketType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    FIRST_5 = "first_5"
    FIRST_INNING = "first_inning"
    PLAYER_PROP = "player_prop"


class OddsFormat(Enum):
    AMERICAN = "american"
    DECIMAL = "decimal"
    IMPLIED = "implied"


@dataclass
class OddsLine:
    """Single odds line from a sportsbook"""
    sportsbook: Sportsbook
    market: MarketType
    selection: str  # "home", "away", "over", "under"
    american_odds: int
    spread: float = 0.0  # For spread bets
    total: float = 0.0   # For over/under
    timestamp: float = field(default_factory=time.time)

    @property
    def decimal_odds(self) -> float:
        if self.american_odds > 0:
            return 1 + self.american_odds / 100
        return 1 + 100 / abs(self.american_odds)

    @property
    def implied_probability(self) -> float:
        if self.american_odds > 0:
            return 100 / (self.american_odds + 100)
        return abs(self.american_odds) / (abs(self.american_odds) + 100)

    def to_dict(self):
        return {
            "sportsbook": self.sportsbook.value,
            "market": self.market.value,
            "selection": self.selection,
            "american_odds": self.american_odds,
            "decimal_odds": round(self.decimal_odds, 3),
            "implied_probability": round(self.implied_probability * 100, 1),
            "spread": self.spread,
            "total": self.total,
            "timestamp": self.timestamp,
        }


@dataclass
class LineMovement:
    """Tracks a single line movement event"""
    movement_id: str
    game_id: str
    sportsbook: Sportsbook
    market: MarketType
    selection: str
    old_odds: int
    new_odds: int
    old_spread: float = 0.0
    new_spread: float = 0.0
    old_total: float = 0.0
    new_total: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def movement_size(self) -> int:
        """Size of movement in cents"""
        return abs(self.new_odds - self.old_odds)

    @property
    def direction(self) -> str:
        if self.new_odds > self.old_odds:
            return "up"
        elif self.new_odds < self.old_odds:
            return "down"
        return "flat"

    @property
    def is_significant(self) -> bool:
        """Movement > 15 cents is considered significant"""
        return self.movement_size >= 15 or abs(self.new_spread - self.old_spread) >= 0.5

    @property
    def is_sharp(self) -> bool:
        """Movement > 25 cents or half-point shift indicates sharp action"""
        return self.movement_size >= 25 or abs(self.new_spread - self.old_spread) >= 1.0

    def to_dict(self):
        return {
            "movement_id": self.movement_id,
            "game_id": self.game_id,
            "sportsbook": self.sportsbook.value,
            "market": self.market.value,
            "selection": self.selection,
            "old_odds": self.old_odds,
            "new_odds": self.new_odds,
            "movement_size": self.movement_size,
            "direction": self.direction,
            "is_significant": self.is_significant,
            "is_sharp": self.is_sharp,
            "spread_change": self.new_spread - self.old_spread,
            "total_change": self.new_total - self.old_total,
            "timestamp": self.timestamp,
        }


@dataclass
class GameOdds:
    """All odds for a single game across all sportsbooks"""
    game_id: str
    home_team: str
    away_team: str
    game_time: str
    status: str = "pre"  # pre, live, final
    inning: int = 0
    home_score: int = 0
    away_score: int = 0
    
    # Current odds by book
    moneyline: Dict[str, Dict[str, OddsLine]] = field(default_factory=dict)  # book -> {home: line, away: line}
    spread: Dict[str, Dict[str, OddsLine]] = field(default_factory=dict)
    total: Dict[str, Dict[str, OddsLine]] = field(default_factory=dict)
    
    # Consensus
    consensus_home_ml: int = 0
    consensus_away_ml: int = 0
    consensus_spread: float = 0.0
    consensus_total: float = 0.0
    
    # Movement history
    movements: List[LineMovement] = field(default_factory=list)
    
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def get_best_odds(self, selection: str, market: MarketType = MarketType.MONEYLINE) -> Optional[OddsLine]:
        """Find the best available odds for a selection across all books"""
        market_data = {
            MarketType.MONEYLINE: self.moneyline,
            MarketType.SPREAD: self.spread,
            MarketType.TOTAL: self.total,
        }.get(market, {})

        best = None
        for book_lines in market_data.values():
            line = book_lines.get(selection)
            if line and (best is None or line.american_odds > best.american_odds):
                best = line
        return best

    def get_worst_odds(self, selection: str, market: MarketType = MarketType.MONEYLINE) -> Optional[OddsLine]:
        market_data = {
            MarketType.MONEYLINE: self.moneyline,
            MarketType.SPREAD: self.spread,
            MarketType.TOTAL: self.total,
        }.get(market, {})

        worst = None
        for book_lines in market_data.values():
            line = book_lines.get(selection)
            if line and (worst is None or line.american_odds < worst.american_odds):
                worst = line
        return worst

    def calculate_consensus(self):
        """Calculate consensus (market average) odds"""
        home_mls = []
        away_mls = []
        spreads = []
        totals = []

        for book_lines in self.moneyline.values():
            if "home" in book_lines:
                home_mls.append(book_lines["home"].american_odds)
            if "away" in book_lines:
                away_mls.append(book_lines["away"].american_odds)

        for book_lines in self.spread.values():
            if "home" in book_lines:
                spreads.append(book_lines["home"].spread)

        for book_lines in self.total.values():
            if "over" in book_lines:
                totals.append(book_lines["over"].total)

        if home_mls:
            self.consensus_home_ml = int(sum(home_mls) / len(home_mls))
        if away_mls:
            self.consensus_away_ml = int(sum(away_mls) / len(away_mls))
        if spreads:
            self.consensus_spread = round(sum(spreads) / len(spreads), 1)
        if totals:
            self.consensus_total = round(sum(totals) / len(totals), 1)

    def to_dict(self) -> Dict:
        self.calculate_consensus()
        
        ml_comparison = {}
        for book, lines in self.moneyline.items():
            ml_comparison[book] = {k: v.to_dict() for k, v in lines.items()}

        spread_comparison = {}
        for book, lines in self.spread.items():
            spread_comparison[book] = {k: v.to_dict() for k, v in lines.items()}

        total_comparison = {}
        for book, lines in self.total.items():
            total_comparison[book] = {k: v.to_dict() for k, v in lines.items()}

        best_home = self.get_best_odds("home")
        best_away = self.get_best_odds("away")

        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "game_time": self.game_time,
            "status": self.status,
            "score": f"{self.away_score}-{self.home_score}" if self.status == "live" else None,
            "inning": self.inning if self.status == "live" else None,
            "consensus": {
                "home_ml": self.consensus_home_ml,
                "away_ml": self.consensus_away_ml,
                "spread": self.consensus_spread,
                "total": self.consensus_total,
            },
            "best_odds": {
                "home": best_home.to_dict() if best_home else None,
                "away": best_away.to_dict() if best_away else None,
            },
            "moneyline": ml_comparison,
            "spread": spread_comparison,
            "total": total_comparison,
            "recent_movements": [m.to_dict() for m in self.movements[-10:]],
            "movement_count": len(self.movements),
            "significant_movements": sum(1 for m in self.movements if m.is_significant),
            "sharp_movements": sum(1 for m in self.movements if m.is_sharp),
            "updated_at": self.updated_at,
        }


# ──────────────────────────────────────────────
# Sharp Money Detector
# ──────────────────────────────────────────────

@dataclass
class SharpAlert:
    alert_id: str
    game_id: str
    alert_type: str  # "reverse_line_movement", "steam_move", "sharp_book_divergence"
    description: str
    confidence: float  # 0.0 - 1.0
    side: str  # "home", "away", "over", "under"
    team: str
    movements: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


class SharpMoneyDetector:
    """Detect sharp (professional) betting action through line movements"""

    def __init__(self):
        self.alerts: List[SharpAlert] = []
        self.public_percentages: Dict[str, Dict] = {}  # game_id -> {home_pct, away_pct}

    def set_public_percentages(self, game_id: str, home_pct: float, away_pct: float):
        self.public_percentages[game_id] = {"home": home_pct, "away": away_pct}

    def analyze_movements(self, game: GameOdds) -> List[SharpAlert]:
        """Analyze line movements for sharp action indicators"""
        new_alerts = []

        # 1. Reverse Line Movement (RLM)
        # If public is on one side but line moves the other way = sharp money
        public = self.public_percentages.get(game.game_id, {})
        if public:
            recent = [m for m in game.movements if m.timestamp > time.time() - 3600]
            for m in recent:
                if m.market == MarketType.MONEYLINE and m.is_significant:
                    pub_pct = public.get(m.selection, 50)
                    if pub_pct > 60 and m.direction == "down":  # Public likes it but line drops
                        alert = SharpAlert(
                            alert_id=f"sa-{str(uuid.uuid4())[:8]}",
                            game_id=game.game_id,
                            alert_type="reverse_line_movement",
                            description=f"RLM: {pub_pct:.0f}% public on {m.selection} but line moved "
                                        f"{m.old_odds} → {m.new_odds}. Sharp money on opposite side.",
                            confidence=min(0.95, (pub_pct - 50) / 50 * 0.8 + m.movement_size / 100),
                            side="away" if m.selection == "home" else "home",
                            team=game.away_team if m.selection == "home" else game.home_team,
                            movements=[m.to_dict()],
                        )
                        new_alerts.append(alert)

        # 2. Steam Move (rapid movement across multiple books)
        if len(game.movements) >= 3:
            window = 300  # 5 minutes
            recent = [m for m in game.movements if m.timestamp > time.time() - window]
            books_moved = set(m.sportsbook.value for m in recent)
            if len(books_moved) >= 3:  # 3+ books moved in 5 min = steam
                direction = recent[0].direction if recent else "flat"
                alert = SharpAlert(
                    alert_id=f"sa-{str(uuid.uuid4())[:8]}",
                    game_id=game.game_id,
                    alert_type="steam_move",
                    description=f"STEAM: {len(books_moved)} books moved in {window // 60} min. "
                                f"Direction: {direction}. Books: {', '.join(books_moved)}",
                    confidence=min(0.9, len(books_moved) / 8 + 0.3),
                    side=recent[0].selection if recent else "home",
                    team=game.home_team,
                    movements=[m.to_dict() for m in recent],
                )
                new_alerts.append(alert)

        # 3. Sharp Book Divergence (Pinnacle vs others)
        pinnacle_ml = game.moneyline.get(Sportsbook.PINNACLE.value, {})
        if pinnacle_ml:
            for selection in ["home", "away"]:
                pin_line = pinnacle_ml.get(selection)
                best = game.get_best_odds(selection)
                if pin_line and best and best.sportsbook != Sportsbook.PINNACLE:
                    diff = abs(pin_line.american_odds - best.american_odds)
                    if diff >= 20:
                        alert = SharpAlert(
                            alert_id=f"sa-{str(uuid.uuid4())[:8]}",
                            game_id=game.game_id,
                            alert_type="sharp_book_divergence",
                            description=f"Pinnacle ({pin_line.american_odds}) vs {best.sportsbook.value} "
                                        f"({best.american_odds}): {diff} cent gap on {selection}.",
                            confidence=min(0.85, diff / 50),
                            side=selection,
                            team=game.home_team if selection == "home" else game.away_team,
                        )
                        new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_alerts(self, game_id: str = "", limit: int = 20) -> List[Dict]:
        alerts = self.alerts
        if game_id:
            alerts = [a for a in alerts if a.game_id == game_id]
        alerts.sort(key=lambda a: -a.confidence)
        return [a.to_dict() for a in alerts[:limit]]


# ──────────────────────────────────────────────
# Live Odds Feed (WebSocket Server)
# ──────────────────────────────────────────────

class LiveOddsFeed:
    """
    WebSocket-based live odds feed with line movement tracking,
    sharp money detection, and multi-book comparison.
    """

    def __init__(self):
        self.games: Dict[str, GameOdds] = {}
        self.sharp_detector = SharpMoneyDetector()
        self.subscribers: Dict[str, Set] = {}  # game_id -> set of ws connections
        self.global_subscribers: Set = set()
        self._movement_log: deque = deque(maxlen=10000)
        self._alert_callbacks: List = []

    def register_game(self, game_id: str, home: str, away: str, game_time: str) -> GameOdds:
        game = GameOdds(
            game_id=game_id,
            home_team=home,
            away_team=away,
            game_time=game_time,
        )
        self.games[game_id] = game
        return game

    def update_odds(self, game_id: str, sportsbook: str, market: str,
                    selection: str, odds: int, spread: float = 0.0,
                    total: float = 0.0) -> Optional[LineMovement]:
        """Update odds and detect line movements"""
        game = self.games.get(game_id)
        if not game:
            return None

        book = Sportsbook(sportsbook)
        mkt = MarketType(market)

        new_line = OddsLine(
            sportsbook=book,
            market=mkt,
            selection=selection,
            american_odds=odds,
            spread=spread,
            total=total,
        )

        # Get current line for movement detection
        market_data = {
            MarketType.MONEYLINE: game.moneyline,
            MarketType.SPREAD: game.spread,
            MarketType.TOTAL: game.total,
        }[mkt]

        old_line = market_data.get(book.value, {}).get(selection)
        movement = None

        if old_line and old_line.american_odds != odds:
            movement = LineMovement(
                movement_id=f"mv-{str(uuid.uuid4())[:8]}",
                game_id=game_id,
                sportsbook=book,
                market=mkt,
                selection=selection,
                old_odds=old_line.american_odds,
                new_odds=odds,
                old_spread=old_line.spread,
                new_spread=spread,
                old_total=old_line.total,
                new_total=total,
            )
            game.movements.append(movement)
            self._movement_log.append(movement)

            # Check for sharp alerts
            if movement.is_significant:
                alerts = self.sharp_detector.analyze_movements(game)
                for alert in alerts:
                    for cb in self._alert_callbacks:
                        cb(alert)

        # Update the line
        if book.value not in market_data:
            market_data[book.value] = {}
        market_data[book.value][selection] = new_line
        game.updated_at = time.time()

        return movement

    def get_game(self, game_id: str) -> Optional[Dict]:
        game = self.games.get(game_id)
        return game.to_dict() if game else None

    def get_all_games(self) -> List[Dict]:
        return [g.to_dict() for g in sorted(self.games.values(), key=lambda g: g.game_time)]

    def get_best_lines(self, game_id: str) -> Dict:
        """Get the best available line at each book for a game"""
        game = self.games.get(game_id)
        if not game:
            return {}

        result = {}
        for market_name, market_data in [
            ("moneyline", game.moneyline),
            ("spread", game.spread),
            ("total", game.total),
        ]:
            result[market_name] = {}
            for book, lines in market_data.items():
                for selection, line in lines.items():
                    if selection not in result[market_name]:
                        result[market_name][selection] = []
                    result[market_name][selection].append(line.to_dict())

            # Sort each selection by best odds
            for selection in result[market_name]:
                result[market_name][selection].sort(key=lambda x: -x["american_odds"])

        return result

    def get_line_history(self, game_id: str, limit: int = 50) -> List[Dict]:
        game = self.games.get(game_id)
        if not game:
            return []
        return [m.to_dict() for m in game.movements[-limit:]]

    def get_sharp_alerts(self, game_id: str = "") -> List[Dict]:
        return self.sharp_detector.get_alerts(game_id)

    def get_movement_summary(self) -> Dict:
        """Summary of all recent movements across all games"""
        now = time.time()
        recent = [m for m in self._movement_log if m.timestamp > now - 3600]

        significant = [m for m in recent if m.is_significant]
        sharp = [m for m in recent if m.is_sharp]

        book_activity = {}
        for m in recent:
            book = m.sportsbook.value
            book_activity[book] = book_activity.get(book, 0) + 1

        return {
            "total_movements_1h": len(recent),
            "significant_movements": len(significant),
            "sharp_movements": len(sharp),
            "book_activity": dict(sorted(book_activity.items(), key=lambda x: -x[1])),
            "active_games": len(set(m.game_id for m in recent)),
            "total_alerts": len(self.sharp_detector.alerts),
        }

    def on_alert(self, callback):
        """Register callback for sharp money alerts"""
        self._alert_callbacks.append(callback)

    def simulate_market(self, game_id: str):
        """Simulate realistic odds updates for testing"""
        game = self.games.get(game_id)
        if not game:
            return

        books = [s.value for s in Sportsbook]
        base_home = random.choice([-150, -130, -120, -110, +100, +110, +120, +140])
        base_away = -(base_home) + random.randint(-20, 20) if base_home < 0 else -(base_home + random.randint(20, 40))
        base_total = random.choice([7.0, 7.5, 8.0, 8.5, 9.0, 9.5])
        base_spread = round(random.uniform(-2.5, 2.5) * 2) / 2

        for book in books:
            # Moneyline with variance
            home_var = random.randint(-10, 10)
            away_var = random.randint(-10, 10)
            self.update_odds(game_id, book, "moneyline", "home", base_home + home_var)
            self.update_odds(game_id, book, "moneyline", "away", base_away + away_var)

            # Total
            total_var = random.choice([-0.5, 0, 0, 0, 0.5])
            juice = random.choice([-110, -110, -110, -115, -105])
            self.update_odds(game_id, book, "total", "over", juice, total=base_total + total_var)
            self.update_odds(game_id, book, "total", "under", juice, total=base_total + total_var)

            # Spread
            spread_var = random.choice([-0.5, 0, 0, 0, 0.5])
            self.update_odds(game_id, book, "spread", "home", -110, spread=base_spread + spread_var)
            self.update_odds(game_id, book, "spread", "away", -110, spread=-(base_spread + spread_var))


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("MLB Predictor - Live Odds WebSocket Feed")
    print("=" * 60)

    feed = LiveOddsFeed()

    # Register games
    games = [
        ("NYY-BOS-20260223", "New York Yankees", "Boston Red Sox", "2026-02-23T19:05:00"),
        ("LAD-SFG-20260223", "Los Angeles Dodgers", "San Francisco Giants", "2026-02-23T22:10:00"),
        ("CHC-STL-20260223", "Chicago Cubs", "St. Louis Cardinals", "2026-02-23T20:15:00"),
        ("HOU-TEX-20260223", "Houston Astros", "Texas Rangers", "2026-02-23T20:05:00"),
    ]

    for gid, home, away, gt in games:
        feed.register_game(gid, home, away, gt)
        feed.simulate_market(gid)

    # Set public betting percentages (for RLM detection)
    feed.sharp_detector.set_public_percentages("NYY-BOS-20260223", 72, 28)
    feed.sharp_detector.set_public_percentages("LAD-SFG-20260223", 65, 35)

    # Simulate line movements
    print(f"\n📊 Simulating odds updates...")
    for _ in range(3):
        gid = random.choice([g[0] for g in games])
        book = random.choice([s.value for s in Sportsbook])
        old_game = feed.games[gid]
        current_ml = old_game.moneyline.get(book, {}).get("home")
        if current_ml:
            new_odds = current_ml.american_odds + random.choice([-20, -15, -10, -5, 5, 10, 15, 20])
            mv = feed.update_odds(gid, book, "moneyline", "home", new_odds)
            if mv:
                print(f"  📈 {book}: {old_game.home_team} ML {mv.old_odds} → {mv.new_odds} "
                      f"({'⚡ SHARP' if mv.is_sharp else '🔄 significant' if mv.is_significant else 'minor'})")

    # Display all games
    print(f"\n🏟️ Today's Games:")
    for game_data in feed.get_all_games():
        print(f"\n  {game_data['away_team']} @ {game_data['home_team']} — {game_data['game_time']}")
        consensus = game_data["consensus"]
        print(f"    Consensus: ML {consensus['home_ml']}/{consensus['away_ml']} | "
              f"Spread {consensus['spread']} | Total {consensus['total']}")

        best = game_data.get("best_odds", {})
        if best.get("home"):
            print(f"    Best Home: {best['home']['american_odds']} @ {best['home']['sportsbook']}")
        if best.get("away"):
            print(f"    Best Away: {best['away']['american_odds']} @ {best['away']['sportsbook']}")

        if game_data["movement_count"] > 0:
            print(f"    Movements: {game_data['movement_count']} total, "
                  f"{game_data['significant_movements']} significant, "
                  f"{game_data['sharp_movements']} sharp")

    # Best lines comparison
    print(f"\n📋 Best Lines (NYY vs BOS):")
    best_lines = feed.get_best_lines("NYY-BOS-20260223")
    for market, selections in best_lines.items():
        print(f"  {market.upper()}:")
        for selection, lines in selections.items():
            if lines:
                top = lines[0]
                print(f"    {selection}: {top['american_odds']} @ {top['sportsbook']} "
                      f"(implied: {top['implied_probability']}%)")

    # Sharp alerts
    print(f"\n⚡ Sharp Money Alerts:")
    alerts = feed.get_sharp_alerts()
    if alerts:
        for alert in alerts[:5]:
            print(f"  🔔 [{alert['alert_type']}] {alert['description']}")
            print(f"     Confidence: {alert['confidence']:.0%} | Side: {alert['side']} ({alert['team']})")
    else:
        print("  No alerts yet (need more movement data)")

    # Movement summary
    print(f"\n📈 Movement Summary:")
    summary = feed.get_movement_summary()
    print(f"  Total movements (1h): {summary['total_movements_1h']}")
    print(f"  Significant: {summary['significant_movements']}")
    print(f"  Sharp: {summary['sharp_movements']}")
    print(f"  Active games: {summary['active_games']}")
    if summary["book_activity"]:
        print(f"  Most active book: {max(summary['book_activity'], key=summary['book_activity'].get)}")

    print(f"\n✅ Live Odds WebSocket Feed ready!")
    print("  • Multi-book odds comparison (8 sportsbooks)")
    print("  • Real-time line movement tracking")
    print("  • Sharp money detection (RLM, steam, divergence)")
    print("  • Consensus odds calculation")
    print("  • Best line finder across all books")
    print("  • Movement history & analytics")
    print("  • WebSocket-ready for live streaming")


if __name__ == "__main__":
    demo()
