#!/usr/bin/env python3
"""
Betting Market Efficiency Analyzer
====================================
Track, analyze, and exploit inefficiencies across sportsbook betting markets.

Features:
- Track opening vs closing line movements across 10+ sportsbooks
- Identify "steam moves" (sharp money indicators)
- Calculate implied probability from odds (American, decimal, fractional)
- Market consensus aggregator (average of all books)
- Closing Line Value (CLV) tracker
- Vig calculator and true probability extractor
- Book-specific bias detection (sharpest books)
- ROI tracker by bet type (ML, spread, total, F5)
- Synthetic hold calculation (find softest books)
- Kelly-optimized bet sizing with market-adjusted edges

Author: MLB Predictor System
Version: 1.0.0
"""

import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants & Enums
# ---------------------------------------------------------------------------

class OddsFormat(Enum):
    AMERICAN = "american"
    DECIMAL = "decimal"
    FRACTIONAL = "fractional"

class BetType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    FIRST_FIVE = "first_five"
    PROP = "prop"

class BetResult(Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"

SPORTSBOOKS = [
    "pinnacle", "circa", "bookmaker", "betcris",    # Sharp books
    "draftkings", "fanduel", "betmgm", "caesars",   # Major US
    "pointsbet", "wynn", "betrivers", "barstool",   # Secondary US
    "bet365", "bovada",                              # International
]

SHARP_BOOKS = {"pinnacle", "circa", "bookmaker", "betcris"}
SOFT_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "pointsbet",
              "wynn", "betrivers", "barstool"}

# Steam move detection thresholds
STEAM_MOVE_THRESHOLD = 15     # cents of line movement
STEAM_TIME_WINDOW_MIN = 10    # minutes for rapid movement
STEAM_BOOK_AGREEMENT = 3      # minimum books moving same direction


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class OddsPoint:
    """A single odds observation at a point in time."""
    sportsbook: str
    timestamp: datetime
    odds_american: int
    odds_decimal: float = 0.0
    odds_fractional: str = ""
    implied_probability: float = 0.0
    is_opening: bool = False
    is_closing: bool = False

    def __post_init__(self):
        if self.odds_decimal == 0.0:
            self.odds_decimal = OddsConverter.american_to_decimal(self.odds_american)
        if not self.odds_fractional:
            self.odds_fractional = OddsConverter.american_to_fractional(self.odds_american)
        if self.implied_probability == 0.0:
            self.implied_probability = OddsConverter.american_to_implied_prob(self.odds_american)


@dataclass
class LineMovement:
    """Tracks line movement for a single market at one sportsbook."""
    sportsbook: str
    game_id: str
    bet_type: BetType
    side: str  # team name or over/under
    opening: OddsPoint = None
    closing: OddsPoint = None
    history: List[OddsPoint] = field(default_factory=list)
    total_movement: int = 0  # in cents

    def compute_movement(self) -> int:
        if self.opening and self.closing:
            self.total_movement = self.closing.odds_american - self.opening.odds_american
        return self.total_movement


@dataclass
class SteamMove:
    """Detected steam move (sharp money indicator)."""
    game_id: str
    bet_type: BetType
    side: str
    direction: str  # "steam_towards" or "steam_away"
    magnitude: int   # cents of movement
    time_window_minutes: float
    books_moving: List[str]
    trigger_book: str
    timestamp: datetime
    confidence: float  # 0-1
    is_sharp: bool = True


@dataclass
class BetRecord:
    """Individual bet tracking record."""
    bet_id: str
    game_id: str
    timestamp: datetime
    bet_type: BetType
    sportsbook: str
    side: str
    odds_placed: int
    closing_odds: int = 0
    stake: float = 0.0
    payout: float = 0.0
    result: BetResult = BetResult.PENDING
    clv: float = 0.0  # Closing Line Value

    def compute_clv(self) -> float:
        """Calculate CLV: difference in implied probability."""
        if self.closing_odds == 0:
            return 0.0
        placed_prob = OddsConverter.american_to_implied_prob(self.odds_placed)
        closing_prob = OddsConverter.american_to_implied_prob(self.closing_odds)
        # Positive CLV means you beat the closing line
        self.clv = closing_prob - placed_prob
        return self.clv

    @property
    def profit(self) -> float:
        if self.result == BetResult.WIN:
            return self.payout - self.stake
        elif self.result == BetResult.LOSS:
            return -self.stake
        return 0.0


@dataclass
class BookBias:
    """Bias profile for a specific sportsbook."""
    sportsbook: str
    total_markets: int = 0
    avg_hold: float = 0.0
    avg_deviation_from_consensus: float = 0.0
    sharpness_score: float = 0.0  # 0-100, higher = sharper
    home_bias: float = 0.0       # positive = favors home
    favorite_bias: float = 0.0   # positive = shades favorites
    total_bias: float = 0.0      # positive = shades totals over
    closing_accuracy: float = 0.0  # how close to "true" closing line


@dataclass
class MarketConsensus:
    """Aggregated market consensus for a matchup."""
    game_id: str
    bet_type: BetType
    side: str
    consensus_odds_american: int = 0
    consensus_implied_prob: float = 0.0
    true_probability: float = 0.0  # vig-removed
    sharp_consensus: float = 0.0   # sharp books only
    soft_consensus: float = 0.0    # soft books only
    num_books: int = 0
    spread: int = 0  # odds spread across books (max - min)
    best_odds: int = 0
    best_book: str = ""
    worst_odds: int = 0
    worst_book: str = ""


# ---------------------------------------------------------------------------
# Odds Conversion Utilities
# ---------------------------------------------------------------------------

class OddsConverter:
    """Convert between American, decimal, and fractional odds formats."""

    @staticmethod
    def american_to_decimal(american: int) -> float:
        if american > 0:
            return 1 + american / 100
        elif american < 0:
            return 1 + 100 / abs(american)
        return 1.0

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1) * 100))
        elif decimal_odds > 1.0:
            return int(round(-100 / (decimal_odds - 1)))
        return 0

    @staticmethod
    def american_to_fractional(american: int) -> str:
        if american > 0:
            from math import gcd
            g = gcd(american, 100)
            return f"{american // g}/{100 // g}"
        elif american < 0:
            from math import gcd
            g = gcd(100, abs(american))
            return f"{100 // g}/{abs(american) // g}"
        return "0/1"

    @staticmethod
    def fractional_to_american(fractional: str) -> int:
        parts = fractional.split("/")
        if len(parts) != 2:
            return 0
        num, den = int(parts[0]), int(parts[1])
        if den == 0:
            return 0
        decimal_val = 1 + num / den
        return OddsConverter.decimal_to_american(decimal_val)

    @staticmethod
    def american_to_implied_prob(american: int) -> float:
        if american > 0:
            return 100 / (american + 100)
        elif american < 0:
            return abs(american) / (abs(american) + 100)
        return 0.5

    @staticmethod
    def implied_prob_to_american(prob: float) -> int:
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return int(round(-100 * prob / (1 - prob)))
        else:
            return int(round(100 * (1 - prob) / prob))

    @staticmethod
    def decimal_to_implied_prob(decimal_odds: float) -> float:
        if decimal_odds <= 0:
            return 0.0
        return 1 / decimal_odds


# ---------------------------------------------------------------------------
# Vig Calculator
# ---------------------------------------------------------------------------

class VigCalculator:
    """Calculate and remove vigorish from odds."""

    @staticmethod
    def calculate_vig(odds_side1: int, odds_side2: int) -> Dict[str, float]:
        """
        Calculate the vig (juice/overround) from a two-way market.
        Returns vig percentage and true probabilities.
        """
        ip1 = OddsConverter.american_to_implied_prob(odds_side1)
        ip2 = OddsConverter.american_to_implied_prob(odds_side2)
        overround = ip1 + ip2
        vig_pct = overround - 1.0

        # True probabilities (multiplicative method)
        true_p1 = ip1 / overround
        true_p2 = ip2 / overround

        # True odds (vig-free)
        true_odds1 = OddsConverter.implied_prob_to_american(true_p1)
        true_odds2 = OddsConverter.implied_prob_to_american(true_p2)

        return {
            "overround": round(overround, 4),
            "vig_pct": round(vig_pct * 100, 2),
            "side1": {
                "raw_implied_prob": round(ip1, 4),
                "true_prob": round(true_p1, 4),
                "raw_odds": odds_side1,
                "true_odds": true_odds1,
            },
            "side2": {
                "raw_implied_prob": round(ip2, 4),
                "true_prob": round(true_p2, 4),
                "raw_odds": odds_side2,
                "true_odds": true_odds2,
            },
        }

    @staticmethod
    def calculate_three_way_vig(odds1: int, odds2: int, odds3: int) -> Dict[str, Any]:
        """Calculate vig for a three-way market (e.g., run line with push)."""
        ip1 = OddsConverter.american_to_implied_prob(odds1)
        ip2 = OddsConverter.american_to_implied_prob(odds2)
        ip3 = OddsConverter.american_to_implied_prob(odds3)
        overround = ip1 + ip2 + ip3
        vig_pct = overround - 1.0
        return {
            "overround": round(overround, 4),
            "vig_pct": round(vig_pct * 100, 2),
            "true_probs": [round(p / overround, 4) for p in [ip1, ip2, ip3]],
        }

    @staticmethod
    def synthetic_hold(book_odds: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """
        Find the lowest synthetic hold (softest combination) by mixing
        the best side from each book.

        book_odds: {sportsbook: (side1_odds, side2_odds)}
        """
        best_side1 = max(book_odds.items(), key=lambda x: x[1][0])
        best_side2 = max(book_odds.items(), key=lambda x: x[1][1])

        best_s1_odds = best_side1[1][0]
        best_s2_odds = best_side2[1][1]

        ip1 = OddsConverter.american_to_implied_prob(best_s1_odds)
        ip2 = OddsConverter.american_to_implied_prob(best_s2_odds)
        synthetic = ip1 + ip2

        return {
            "synthetic_hold": round((synthetic - 1) * 100, 2),
            "best_side1": {
                "book": best_side1[0],
                "odds": best_s1_odds,
                "implied_prob": round(ip1, 4),
            },
            "best_side2": {
                "book": best_side2[0],
                "odds": best_s2_odds,
                "implied_prob": round(ip2, 4),
            },
            "overround": round(synthetic, 4),
            "is_arbitrage": synthetic < 1.0,
        }


# ---------------------------------------------------------------------------
# Betting Market Analyzer
# ---------------------------------------------------------------------------

class BettingMarketAnalyzer:
    """
    Main engine for analyzing betting market efficiency.

    Tracks lines, detects steam moves, calculates CLV, identifies
    book biases, and provides Kelly-optimized bet sizing.
    """

    def __init__(self):
        self.line_movements: Dict[str, List[LineMovement]] = defaultdict(list)
        self.steam_moves: List[SteamMove] = []
        self.bet_history: List[BetRecord] = []
        self.book_biases: Dict[str, BookBias] = {}
        self.vig_calc = VigCalculator()
        self.market_snapshots: Dict[str, Dict[str, OddsPoint]] = {}

    # ---- Line Movement Tracking ----

    def record_odds(self, game_id: str, sportsbook: str,
                    bet_type: BetType, side: str,
                    odds_american: int, timestamp: datetime = None,
                    is_opening: bool = False, is_closing: bool = False) -> OddsPoint:
        """Record a single odds observation."""
        if timestamp is None:
            timestamp = datetime.now()

        point = OddsPoint(
            sportsbook=sportsbook,
            timestamp=timestamp,
            odds_american=odds_american,
            is_opening=is_opening,
            is_closing=is_closing,
        )

        key = f"{game_id}_{bet_type.value}_{side}"

        # Find or create LineMovement
        existing = None
        for lm in self.line_movements[key]:
            if lm.sportsbook == sportsbook:
                existing = lm
                break

        if existing is None:
            existing = LineMovement(
                sportsbook=sportsbook,
                game_id=game_id,
                bet_type=bet_type,
                side=side,
            )
            self.line_movements[key].append(existing)

        existing.history.append(point)
        if is_opening:
            existing.opening = point
        if is_closing:
            existing.closing = point

        existing.compute_movement()
        return point

    def get_line_movement(self, game_id: str, bet_type: BetType,
                          side: str) -> List[Dict[str, Any]]:
        """Get all line movement for a specific market."""
        key = f"{game_id}_{bet_type.value}_{side}"
        movements = self.line_movements.get(key, [])

        result = []
        for lm in movements:
            result.append({
                "sportsbook": lm.sportsbook,
                "opening": lm.opening.odds_american if lm.opening else None,
                "closing": lm.closing.odds_american if lm.closing else None,
                "movement": lm.total_movement,
                "num_changes": len(lm.history),
                "is_sharp": lm.sportsbook in SHARP_BOOKS,
            })

        return sorted(result, key=lambda x: abs(x["movement"] or 0), reverse=True)

    # ---- Steam Move Detection ----

    def detect_steam_moves(self, game_id: str, bet_type: BetType,
                           side: str) -> List[SteamMove]:
        """
        Detect steam moves: rapid, coordinated line movements across
        multiple books, typically initiated by sharp money.
        """
        key = f"{game_id}_{bet_type.value}_{side}"
        movements = self.line_movements.get(key, [])

        if len(movements) < STEAM_BOOK_AGREEMENT:
            return []

        # Collect all timestamped changes
        all_changes = []
        for lm in movements:
            for i in range(1, len(lm.history)):
                prev = lm.history[i - 1]
                curr = lm.history[i]
                change = curr.odds_american - prev.odds_american
                if abs(change) >= 5:  # At least 5 cents movement
                    all_changes.append({
                        "book": lm.sportsbook,
                        "timestamp": curr.timestamp,
                        "change": change,
                        "odds_after": curr.odds_american,
                    })

        all_changes.sort(key=lambda x: x["timestamp"])

        detected = []
        for i, anchor in enumerate(all_changes):
            # Look for correlated moves within the time window
            direction = 1 if anchor["change"] > 0 else -1
            window_end = anchor["timestamp"] + timedelta(minutes=STEAM_TIME_WINDOW_MIN)

            correlated = [anchor]
            books_in_move = {anchor["book"]}

            for j in range(i + 1, len(all_changes)):
                other = all_changes[j]
                if other["timestamp"] > window_end:
                    break
                other_dir = 1 if other["change"] > 0 else -1
                if other_dir == direction and other["book"] not in books_in_move:
                    correlated.append(other)
                    books_in_move.add(other["book"])

            if len(correlated) >= STEAM_BOOK_AGREEMENT:
                total_mag = sum(abs(c["change"]) for c in correlated)
                trigger = correlated[0]
                is_sharp = trigger["book"] in SHARP_BOOKS

                # Confidence based on number of books and sharp involvement
                conf = min(1.0, len(correlated) / 8)
                if is_sharp:
                    conf = min(1.0, conf + 0.2)

                steam = SteamMove(
                    game_id=game_id,
                    bet_type=bet_type,
                    side=side,
                    direction="steam_towards" if direction > 0 else "steam_away",
                    magnitude=total_mag,
                    time_window_minutes=(
                        (correlated[-1]["timestamp"] - correlated[0]["timestamp"]).total_seconds() / 60
                    ),
                    books_moving=[c["book"] for c in correlated],
                    trigger_book=trigger["book"],
                    timestamp=trigger["timestamp"],
                    confidence=round(conf, 3),
                    is_sharp=is_sharp,
                )
                detected.append(steam)

        self.steam_moves.extend(detected)
        return detected

    # ---- Market Consensus ----

    def compute_consensus(self, game_id: str, bet_type: BetType,
                          side: str) -> MarketConsensus:
        """
        Aggregate odds across all books to find market consensus.
        Separates sharp vs soft book consensus.
        """
        key = f"{game_id}_{bet_type.value}_{side}"
        movements = self.line_movements.get(key, [])

        if not movements:
            return MarketConsensus(game_id=game_id, bet_type=bet_type, side=side)

        all_odds = []
        sharp_odds = []
        soft_odds = []

        for lm in movements:
            latest = lm.history[-1] if lm.history else None
            if latest is None:
                continue
            all_odds.append((lm.sportsbook, latest.odds_american))
            if lm.sportsbook in SHARP_BOOKS:
                sharp_odds.append(latest.odds_american)
            else:
                soft_odds.append(latest.odds_american)

        if not all_odds:
            return MarketConsensus(game_id=game_id, bet_type=bet_type, side=side)

        odds_values = [o[1] for o in all_odds]
        avg_odds = int(round(statistics.mean(odds_values)))
        avg_prob = statistics.mean([OddsConverter.american_to_implied_prob(o) for o in odds_values])

        # Best and worst
        best = max(all_odds, key=lambda x: x[1])
        worst = min(all_odds, key=lambda x: x[1])

        consensus = MarketConsensus(
            game_id=game_id,
            bet_type=bet_type,
            side=side,
            consensus_odds_american=avg_odds,
            consensus_implied_prob=round(avg_prob, 4),
            sharp_consensus=round(statistics.mean(
                [OddsConverter.american_to_implied_prob(o) for o in sharp_odds]
            ), 4) if sharp_odds else 0.0,
            soft_consensus=round(statistics.mean(
                [OddsConverter.american_to_implied_prob(o) for o in soft_odds]
            ), 4) if soft_odds else 0.0,
            num_books=len(all_odds),
            spread=max(odds_values) - min(odds_values),
            best_odds=best[1],
            best_book=best[0],
            worst_odds=worst[1],
            worst_book=worst[0],
        )

        return consensus

    # ---- Closing Line Value (CLV) ----

    def track_clv(self, bet: BetRecord) -> float:
        """
        Calculate Closing Line Value for a bet.
        Positive CLV = you beat the closing line (good!).
        """
        return bet.compute_clv()

    def clv_summary(self) -> Dict[str, Any]:
        """Aggregate CLV statistics across all bets."""
        if not self.bet_history:
            return {"total_bets": 0}

        clvs = [b.clv for b in self.bet_history if b.clv != 0]
        if not clvs:
            return {"total_bets": len(self.bet_history), "clv_tracked": 0}

        positive_clv = [c for c in clvs if c > 0]
        negative_clv = [c for c in clvs if c < 0]

        return {
            "total_bets": len(self.bet_history),
            "clv_tracked": len(clvs),
            "avg_clv": round(statistics.mean(clvs) * 100, 2),
            "median_clv": round(statistics.median(clvs) * 100, 2),
            "positive_clv_pct": round(len(positive_clv) / len(clvs) * 100, 1),
            "avg_positive_clv": round(statistics.mean(positive_clv) * 100, 2) if positive_clv else 0,
            "avg_negative_clv": round(statistics.mean(negative_clv) * 100, 2) if negative_clv else 0,
            "interpretation": (
                "Consistently beating the closing line — strong edge"
                if statistics.mean(clvs) > 0.01
                else "Slightly positive CLV — marginal edge"
                if statistics.mean(clvs) > 0
                else "Negative CLV — consider adjusting strategy"
            ),
        }

    # ---- ROI Tracker by Bet Type ----

    def roi_by_bet_type(self) -> Dict[str, Dict[str, Any]]:
        """Calculate ROI broken down by bet type."""
        by_type: Dict[str, List[BetRecord]] = defaultdict(list)
        for bet in self.bet_history:
            by_type[bet.bet_type.value].append(bet)

        results = {}
        for bt, bets in by_type.items():
            total_stake = sum(b.stake for b in bets)
            total_profit = sum(b.profit for b in bets)
            wins = sum(1 for b in bets if b.result == BetResult.WIN)
            losses = sum(1 for b in bets if b.result == BetResult.LOSS)
            pushes = sum(1 for b in bets if b.result == BetResult.PUSH)

            roi = (total_profit / total_stake * 100) if total_stake > 0 else 0

            results[bt] = {
                "total_bets": len(bets),
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
                "total_stake": round(total_stake, 2),
                "total_profit": round(total_profit, 2),
                "roi_pct": round(roi, 2),
                "avg_odds": int(round(statistics.mean(
                    [b.odds_placed for b in bets]
                ))) if bets else 0,
            }

        return results

    # ---- Book Bias Detection ----

    def analyze_book_biases(self) -> Dict[str, Dict[str, Any]]:
        """
        Detect systematic biases per sportsbook.
        Which books are sharpest? Which shade favorites? Home teams?
        """
        book_data: Dict[str, List[Dict]] = defaultdict(list)

        for key, movements in self.line_movements.items():
            for lm in movements:
                if not lm.history:
                    continue
                latest = lm.history[-1]
                book_data[lm.sportsbook].append({
                    "game_id": lm.game_id,
                    "odds": latest.odds_american,
                    "implied_prob": latest.implied_probability,
                    "movement": lm.total_movement,
                })

        results = {}
        # Compute consensus across all books first
        all_probs_by_game: Dict[str, List[float]] = defaultdict(list)
        for book, entries in book_data.items():
            for entry in entries:
                all_probs_by_game[entry["game_id"]].append(entry["implied_prob"])

        consensus_by_game = {
            gid: statistics.mean(probs)
            for gid, probs in all_probs_by_game.items()
        }

        for book, entries in book_data.items():
            if not entries:
                continue

            # Deviation from consensus
            deviations = []
            for entry in entries:
                consensus = consensus_by_game.get(entry["game_id"], 0.5)
                deviations.append(entry["implied_prob"] - consensus)

            avg_dev = statistics.mean(deviations) if deviations else 0
            avg_movement = statistics.mean([abs(e["movement"]) for e in entries])

            # Sharpness: lower deviation from consensus + accurate closing
            sharpness = max(0, 100 - abs(avg_dev) * 1000 - avg_movement * 0.5)

            results[book] = {
                "total_markets": len(entries),
                "avg_deviation_from_consensus": round(avg_dev * 100, 3),
                "avg_line_movement": round(avg_movement, 1),
                "sharpness_score": round(sharpness, 1),
                "is_sharp_book": book in SHARP_BOOKS,
                "classification": (
                    "sharp" if sharpness > 70
                    else "mixed" if sharpness > 40
                    else "soft"
                ),
            }

        return results

    # ---- Synthetic Hold (Softest Books) ----

    def find_softest_lines(self, game_id: str, bet_type: BetType,
                           side1: str, side2: str) -> Dict[str, Any]:
        """
        Find the softest book combination (lowest synthetic hold).
        Can also detect arbitrage if synthetic hold < 0.
        """
        key1 = f"{game_id}_{bet_type.value}_{side1}"
        key2 = f"{game_id}_{bet_type.value}_{side2}"

        book_odds = {}
        for lm in self.line_movements.get(key1, []):
            if lm.history:
                latest = lm.history[-1]
                book_odds[lm.sportsbook] = (latest.odds_american, 0)

        for lm in self.line_movements.get(key2, []):
            if lm.history:
                latest = lm.history[-1]
                book = lm.sportsbook
                if book in book_odds:
                    book_odds[book] = (book_odds[book][0], latest.odds_american)
                else:
                    book_odds[book] = (0, latest.odds_american)

        # Filter books with both sides
        complete = {b: o for b, o in book_odds.items() if o[0] != 0 and o[1] != 0}

        if not complete:
            return {"error": "Insufficient data for synthetic hold calculation"}

        # Per-book hold
        book_holds = {}
        for book, (s1, s2) in complete.items():
            vig_info = self.vig_calc.calculate_vig(s1, s2)
            book_holds[book] = {
                "side1_odds": s1,
                "side2_odds": s2,
                "hold_pct": vig_info["vig_pct"],
            }

        # Synthetic hold (best of each side from any book)
        synthetic = self.vig_calc.synthetic_hold(complete)

        # Rank books by hold
        ranked = sorted(book_holds.items(), key=lambda x: x[1]["hold_pct"])

        return {
            "game_id": game_id,
            "bet_type": bet_type.value,
            "synthetic_hold": synthetic,
            "book_holds": {k: v for k, v in ranked},
            "softest_book": ranked[0][0] if ranked else None,
            "sharpest_book": ranked[-1][0] if ranked else None,
            "arbitrage_exists": synthetic["is_arbitrage"],
        }

    # ---- Kelly Criterion Bet Sizing ----

    def kelly_bet_size(self, edge: float, odds_american: int,
                       bankroll: float, fraction: float = 0.25,
                       max_bet_pct: float = 0.05) -> Dict[str, Any]:
        """
        Calculate Kelly-optimal bet size with fractional Kelly adjustment.

        Args:
            edge: estimated edge (true_prob - implied_prob)
            odds_american: odds being bet at
            bankroll: total bankroll
            fraction: Kelly fraction (0.25 = quarter Kelly, safer)
            max_bet_pct: maximum percentage of bankroll per bet
        """
        if edge <= 0:
            return {
                "recommended_bet": 0,
                "kelly_pct": 0,
                "reason": "No positive edge — do not bet",
            }

        decimal_odds = OddsConverter.american_to_decimal(odds_american)
        b = decimal_odds - 1  # net odds
        p = OddsConverter.american_to_implied_prob(odds_american) + edge
        q = 1 - p

        # Full Kelly: f* = (bp - q) / b
        if b <= 0:
            return {"recommended_bet": 0, "kelly_pct": 0, "reason": "Invalid odds"}

        full_kelly = (b * p - q) / b
        if full_kelly <= 0:
            return {
                "recommended_bet": 0,
                "kelly_pct": 0,
                "reason": "Kelly formula yields non-positive — edge may be insufficient",
            }

        # Apply fractional Kelly
        adjusted_kelly = full_kelly * fraction

        # Cap at max bet percentage
        capped_kelly = min(adjusted_kelly, max_bet_pct)

        bet_amount = round(bankroll * capped_kelly, 2)

        return {
            "full_kelly_pct": round(full_kelly * 100, 2),
            "fractional_kelly_pct": round(adjusted_kelly * 100, 2),
            "capped_kelly_pct": round(capped_kelly * 100, 2),
            "recommended_bet": bet_amount,
            "bankroll": bankroll,
            "fraction_used": fraction,
            "edge_pct": round(edge * 100, 2),
            "expected_value": round(bet_amount * edge / (1 - OddsConverter.american_to_implied_prob(odds_american)), 2),
            "odds": odds_american,
            "risk_level": (
                "conservative" if capped_kelly < 0.01
                else "moderate" if capped_kelly < 0.03
                else "aggressive" if capped_kelly < 0.05
                else "max_risk"
            ),
        }

    def market_adjusted_kelly(self, game_id: str, bet_type: BetType,
                               side: str, model_prob: float,
                               bankroll: float) -> Dict[str, Any]:
        """
        Kelly sizing using market-derived true probability as a check.
        Blends model probability with market consensus for robust edge estimation.
        """
        consensus = self.compute_consensus(game_id, bet_type, side)
        if consensus.num_books == 0:
            return {"error": "No market data available"}

        market_prob = consensus.consensus_implied_prob
        sharp_prob = consensus.sharp_consensus if consensus.sharp_consensus > 0 else market_prob

        # Blend: 60% model, 40% sharp market
        blended_prob = 0.6 * model_prob + 0.4 * sharp_prob

        # Edge vs the best available odds
        best_implied = OddsConverter.american_to_implied_prob(consensus.best_odds)
        edge = blended_prob - best_implied

        kelly = self.kelly_bet_size(edge, consensus.best_odds, bankroll)
        kelly["blended_probability"] = round(blended_prob, 4)
        kelly["model_probability"] = round(model_prob, 4)
        kelly["market_probability"] = round(market_prob, 4)
        kelly["sharp_probability"] = round(sharp_prob, 4)
        kelly["best_book"] = consensus.best_book
        kelly["best_odds"] = consensus.best_odds

        return kelly

    # ---- Bet Recording & Tracking ----

    def record_bet(self, bet_id: str, game_id: str, bet_type: BetType,
                   sportsbook: str, side: str, odds_placed: int,
                   stake: float, timestamp: datetime = None) -> BetRecord:
        """Record a placed bet."""
        if timestamp is None:
            timestamp = datetime.now()
        bet = BetRecord(
            bet_id=bet_id,
            game_id=game_id,
            timestamp=timestamp,
            bet_type=bet_type,
            sportsbook=sportsbook,
            side=side,
            odds_placed=odds_placed,
            stake=stake,
        )
        self.bet_history.append(bet)
        return bet

    def settle_bet(self, bet_id: str, result: BetResult,
                   closing_odds: int = 0, payout: float = 0.0) -> Optional[BetRecord]:
        """Settle a bet with result and closing odds."""
        for bet in self.bet_history:
            if bet.bet_id == bet_id:
                bet.result = result
                bet.closing_odds = closing_odds
                bet.payout = payout
                bet.compute_clv()
                return bet
        return None

    # ---- Full Dashboard Export ----

    def generate_dashboard(self, game_id: str = None) -> Dict[str, Any]:
        """Generate a complete market analysis dashboard."""
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "total_markets_tracked": sum(len(v) for v in self.line_movements.values()),
            "total_bets": len(self.bet_history),
            "steam_moves_detected": len(self.steam_moves),
        }

        if self.bet_history:
            dashboard["roi_by_type"] = self.roi_by_bet_type()
            dashboard["clv_summary"] = self.clv_summary()

        if self.line_movements:
            dashboard["book_analysis"] = self.analyze_book_biases()

        return dashboard


# ---------------------------------------------------------------------------
# Demo / Test
# ---------------------------------------------------------------------------

def demo():
    """Run a comprehensive demo of the Betting Market Analyzer."""
    print("=" * 70)
    print("  BETTING MARKET EFFICIENCY ANALYZER — DEMO")
    print("=" * 70)

    random.seed(42)
    analyzer = BettingMarketAnalyzer()

    # ---- 1. Odds Conversion Demo ----
    print("\n" + "=" * 50)
    print("  1. ODDS CONVERSION")
    print("=" * 50)
    test_odds = [-150, +130, -200, +250, -110, +100]
    for odds in test_odds:
        dec = OddsConverter.american_to_decimal(odds)
        frac = OddsConverter.american_to_fractional(odds)
        prob = OddsConverter.american_to_implied_prob(odds)
        print(f"    {odds:+5d} → Decimal: {dec:.3f}, Fractional: {frac:>6s}, "
              f"Implied: {prob:.4f} ({prob*100:.1f}%)")

    # ---- 2. Vig Calculation ----
    print("\n" + "=" * 50)
    print("  2. VIG CALCULATION")
    print("=" * 50)
    vig = VigCalculator.calculate_vig(-150, +130)
    print(f"    Market: -150 / +130")
    print(f"    Overround: {vig['overround']:.4f}")
    print(f"    Vig: {vig['vig_pct']:.2f}%")
    print(f"    True probs: {vig['side1']['true_prob']:.4f} / {vig['side2']['true_prob']:.4f}")
    print(f"    True odds: {vig['side1']['true_odds']:+d} / {vig['side2']['true_odds']:+d}")

    vig2 = VigCalculator.calculate_vig(-110, -110)
    print(f"\n    Market: -110 / -110 (standard)")
    print(f"    Vig: {vig2['vig_pct']:.2f}%")
    print(f"    True probs: {vig2['side1']['true_prob']:.4f} / {vig2['side2']['true_prob']:.4f}")

    # ---- 3. Line Movement Tracking ----
    print("\n" + "=" * 50)
    print("  3. LINE MOVEMENT TRACKING")
    print("=" * 50)
    game_id = "MLB_2025_NYY_BOS_0801"
    base_time = datetime(2025, 8, 1, 10, 0, 0)

    # Simulate line movements across books
    for i, book in enumerate(SPORTSBOOKS[:10]):
        is_sharp = book in SHARP_BOOKS
        base_odds = -140 + random.randint(-10, 10)

        # Opening line
        analyzer.record_odds(
            game_id, book, BetType.MONEYLINE, "NYY",
            base_odds, base_time + timedelta(hours=i * 0.1),
            is_opening=True,
        )

        # Middle movement — sharp books move first
        move_time = base_time + timedelta(hours=2 + i * 0.3)
        moved_odds = base_odds - random.randint(5, 20)
        analyzer.record_odds(
            game_id, book, BetType.MONEYLINE, "NYY",
            moved_odds, move_time,
        )

        # Final closing line
        closing = moved_odds - random.randint(0, 10)
        analyzer.record_odds(
            game_id, book, BetType.MONEYLINE, "NYY",
            closing, base_time + timedelta(hours=8),
            is_closing=True,
        )

        # Also record the other side
        for odds_am in [base_odds + 20, moved_odds + 20, closing + 20]:
            analyzer.record_odds(
                game_id, book, BetType.MONEYLINE, "BOS",
                odds_am,
                base_time + timedelta(hours=random.uniform(0, 8)),
            )

    movements = analyzer.get_line_movement(game_id, BetType.MONEYLINE, "NYY")
    print(f"\n  Line movements for {game_id} — NYY ML:")
    for m in movements[:6]:
        sharp_tag = " [SHARP]" if m["is_sharp"] else ""
        print(f"    {m['sportsbook']:>12s}: Open={m['opening']:+d}, "
              f"Close={m['closing']:+d}, Move={m['movement']:+d}{sharp_tag}")

    # ---- 4. Steam Move Detection ----
    print("\n" + "=" * 50)
    print("  4. STEAM MOVE DETECTION")
    print("=" * 50)
    steams = analyzer.detect_steam_moves(game_id, BetType.MONEYLINE, "NYY")
    if steams:
        for s in steams[:3]:
            print(f"    🚨 Steam Move: {s.direction}")
            print(f"       Magnitude: {s.magnitude} cents")
            print(f"       Books: {', '.join(s.books_moving[:5])}")
            print(f"       Trigger: {s.trigger_book} {'[SHARP]' if s.is_sharp else '[SOFT]'}")
            print(f"       Confidence: {s.confidence:.1%}")
    else:
        print("    No steam moves detected (simulated data may not trigger)")

    # ---- 5. Market Consensus ----
    print("\n" + "=" * 50)
    print("  5. MARKET CONSENSUS")
    print("=" * 50)
    consensus = analyzer.compute_consensus(game_id, BetType.MONEYLINE, "NYY")
    print(f"    Consensus Odds: {consensus.consensus_odds_american:+d}")
    print(f"    Consensus Prob: {consensus.consensus_implied_prob:.4f} "
          f"({consensus.consensus_implied_prob*100:.1f}%)")
    print(f"    Sharp Consensus: {consensus.sharp_consensus:.4f}")
    print(f"    Soft Consensus: {consensus.soft_consensus:.4f}")
    print(f"    Best Odds: {consensus.best_odds:+d} ({consensus.best_book})")
    print(f"    Worst Odds: {consensus.worst_odds:+d} ({consensus.worst_book})")
    print(f"    Spread: {consensus.spread} cents across {consensus.num_books} books")

    # ---- 6. Synthetic Hold ----
    print("\n" + "=" * 50)
    print("  6. SYNTHETIC HOLD (SOFTEST BOOKS)")
    print("=" * 50)
    soft = analyzer.find_softest_lines(game_id, BetType.MONEYLINE, "NYY", "BOS")
    if "error" not in soft:
        sh = soft["synthetic_hold"]
        print(f"    Synthetic Hold: {sh['synthetic_hold']:.2f}%")
        print(f"    Best Side 1: {sh['best_side1']['book']} @ {sh['best_side1']['odds']:+d}")
        print(f"    Best Side 2: {sh['best_side2']['book']} @ {sh['best_side2']['odds']:+d}")
        print(f"    Arbitrage: {'YES 💰' if sh['is_arbitrage'] else 'No'}")
        print(f"\n    Book holds (ranked softest to sharpest):")
        for book, hold in list(soft["book_holds"].items())[:5]:
            print(f"      {book:>12s}: {hold['hold_pct']:.2f}% hold "
                  f"({hold['side1_odds']:+d} / {hold['side2_odds']:+d})")

    # ---- 7. Kelly Bet Sizing ----
    print("\n" + "=" * 50)
    print("  7. KELLY-OPTIMIZED BET SIZING")
    print("=" * 50)
    bankroll = 10000.0

    # Scenario 1: Good edge
    kelly1 = analyzer.kelly_bet_size(
        edge=0.05, odds_american=+150,
        bankroll=bankroll, fraction=0.25,
    )
    print(f"\n  Scenario 1: 5% edge at +150, $10K bankroll")
    print(f"    Full Kelly: {kelly1['full_kelly_pct']:.2f}%")
    print(f"    Quarter Kelly: {kelly1['fractional_kelly_pct']:.2f}%")
    print(f"    Recommended Bet: ${kelly1['recommended_bet']:.2f}")
    print(f"    Expected Value: ${kelly1['expected_value']:.2f}")
    print(f"    Risk Level: {kelly1['risk_level']}")

    # Scenario 2: Marginal edge
    kelly2 = analyzer.kelly_bet_size(
        edge=0.02, odds_american=-110,
        bankroll=bankroll, fraction=0.25,
    )
    print(f"\n  Scenario 2: 2% edge at -110, $10K bankroll")
    print(f"    Full Kelly: {kelly2['full_kelly_pct']:.2f}%")
    print(f"    Recommended Bet: ${kelly2['recommended_bet']:.2f}")
    print(f"    Risk Level: {kelly2['risk_level']}")

    # Scenario 3: No edge
    kelly3 = analyzer.kelly_bet_size(
        edge=-0.01, odds_american=-110,
        bankroll=bankroll,
    )
    print(f"\n  Scenario 3: No edge (-1%) at -110")
    print(f"    Reason: {kelly3['reason']}")

    # ---- 8. Bet Tracking & ROI ----
    print("\n" + "=" * 50)
    print("  8. BET TRACKING & ROI")
    print("=" * 50)

    # Simulate 50 bets
    bet_types = [BetType.MONEYLINE, BetType.SPREAD, BetType.TOTAL, BetType.FIRST_FIVE]
    for i in range(50):
        bt = random.choice(bet_types)
        odds = random.choice([-150, -130, -120, -110, +100, +110, +130, +150, +200])
        stake = random.uniform(50, 300)
        book = random.choice(SPORTSBOOKS)

        bet = analyzer.record_bet(
            bet_id=f"BET_{i:04d}",
            game_id=f"MLB_2025_G{i:03d}",
            bet_type=bt,
            sportsbook=book,
            side="TeamA",
            odds_placed=odds,
            stake=stake,
        )

        # Settle with random outcome (slight positive edge for realism)
        win_prob = OddsConverter.american_to_implied_prob(odds) + 0.02
        result = BetResult.WIN if random.random() < win_prob else BetResult.LOSS
        closing = odds - random.randint(-10, 15)
        payout = stake * OddsConverter.american_to_decimal(odds) if result == BetResult.WIN else 0

        analyzer.settle_bet(bet.bet_id, result, closing, payout)

    roi = analyzer.roi_by_bet_type()
    for bt, data in roi.items():
        print(f"\n    {bt.upper():>12s}: {data['total_bets']} bets, "
              f"W/L={data['wins']}/{data['losses']}, "
              f"Win%={data['win_rate']:.1f}%, "
              f"ROI={data['roi_pct']:+.2f}%, "
              f"P/L=${data['total_profit']:+.2f}")

    # ---- 9. CLV Summary ----
    print("\n" + "=" * 50)
    print("  9. CLOSING LINE VALUE (CLV)")
    print("=" * 50)
    clv = analyzer.clv_summary()
    print(f"    Total Bets: {clv['total_bets']}")
    print(f"    CLV Tracked: {clv['clv_tracked']}")
    print(f"    Avg CLV: {clv['avg_clv']:+.2f}%")
    print(f"    Median CLV: {clv['median_clv']:+.2f}%")
    print(f"    Positive CLV%: {clv['positive_clv_pct']:.1f}%")
    print(f"    → {clv['interpretation']}")

    # ---- 10. Book Bias Analysis ----
    print("\n" + "=" * 50)
    print("  10. BOOK BIAS ANALYSIS")
    print("=" * 50)
    biases = analyzer.analyze_book_biases()
    sorted_biases = sorted(biases.items(), key=lambda x: x[1]["sharpness_score"], reverse=True)
    for book, data in sorted_biases[:8]:
        sharp_tag = "⚡" if data["is_sharp_book"] else "  "
        print(f"    {sharp_tag} {book:>12s}: Sharpness={data['sharpness_score']:5.1f}, "
              f"Dev={data['avg_deviation_from_consensus']:+.3f}%, "
              f"Class={data['classification']}")

    # ---- 11. Dashboard Export ----
    print("\n" + "=" * 50)
    print("  11. DASHBOARD EXPORT (snippet)")
    print("=" * 50)
    dashboard = analyzer.generate_dashboard()
    print(f"    Markets tracked: {dashboard['total_markets_tracked']}")
    print(f"    Total bets: {dashboard['total_bets']}")
    print(f"    Steam moves: {dashboard['steam_moves_detected']}")

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — All systems operational")
    print("=" * 70)


if __name__ == "__main__":
    demo()
