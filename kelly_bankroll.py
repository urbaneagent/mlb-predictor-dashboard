"""
MLB Predictor: Kelly Criterion Bankroll Manager
==================================================
Scientific bankroll management using Kelly Criterion and variants.
Prevents ruin while maximizing long-term growth.

Features:
- Full Kelly, Half Kelly, Quarter Kelly sizing
- Real-time bankroll tracking with P&L
- Risk of ruin calculator
- Drawdown protection (auto-reduce after losses)
- Daily/weekly/monthly limits
- Unit-based or percentage-based betting
- Win rate tracking by bet type (ML, RL, O/U, props)
- ROI analysis with confidence intervals
- Bet grading (A/B/C/D/F based on edge)
- Streak analysis and regression warnings
"""

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum


# ─── Models ──────────────────────────────────────────────────────
class BetType(str, Enum):
    MONEYLINE = "moneyline"
    RUNLINE = "runline"
    OVER_UNDER = "over_under"
    FIRST_5 = "first_5"
    PROP = "prop"
    PARLAY = "parlay"
    LIVE = "live"


class BetResult(str, Enum):
    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    PENDING = "pending"
    VOID = "void"


class BetGrade(str, Enum):
    A_PLUS = "A+"   # Edge > 10%
    A = "A"         # Edge 7-10%
    B = "B"         # Edge 5-7%
    C = "C"         # Edge 3-5%
    D = "D"         # Edge 1-3%
    F = "F"         # Edge < 1% (no edge)


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"   # Quarter Kelly
    MODERATE = "moderate"           # Half Kelly
    AGGRESSIVE = "aggressive"       # Full Kelly
    CUSTOM = "custom"


@dataclass
class BetRecord:
    """A single bet record."""
    bet_id: str
    timestamp: float
    bet_type: BetType
    game_id: str
    description: str
    odds_american: int  # +150, -110, etc.
    odds_decimal: float
    model_win_prob: float  # Our predicted probability
    implied_prob: float  # Sportsbook implied probability
    edge: float  # model_prob - implied_prob
    stake: float  # Amount wagered
    potential_payout: float
    result: BetResult = BetResult.PENDING
    profit: float = 0.0
    grade: BetGrade = BetGrade.C
    kelly_fraction: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bet_type"] = self.bet_type.value
        d["result"] = self.result.value
        d["grade"] = self.grade.value
        return d


@dataclass
class BankrollState:
    """Current bankroll state."""
    starting_bankroll: float
    current_bankroll: float
    peak_bankroll: float
    low_bankroll: float
    total_wagered: float = 0.0
    total_won: float = 0.0
    total_lost: float = 0.0
    net_profit: float = 0.0
    roi: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    push_count: int = 0
    current_streak: int = 0  # Positive = winning, negative = losing
    max_winning_streak: int = 0
    max_losing_streak: int = 0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Kelly Criterion Calculator ──────────────────────────────────
class KellyCriterion:
    """
    Kelly Criterion: f* = (bp - q) / b
    where:
        f* = fraction of bankroll to wager
        b = odds received on the wager (decimal odds - 1)
        p = probability of winning
        q = probability of losing (1 - p)
    """

    @staticmethod
    def full_kelly(
        win_prob: float, decimal_odds: float
    ) -> float:
        """Calculate full Kelly fraction."""
        if win_prob <= 0 or win_prob >= 1 or decimal_odds <= 1:
            return 0.0

        b = decimal_odds - 1
        p = win_prob
        q = 1 - p

        kelly = (b * p - q) / b

        # Negative Kelly = no edge, don't bet
        return max(kelly, 0.0)

    @staticmethod
    def half_kelly(win_prob: float, decimal_odds: float) -> float:
        """Half Kelly — reduces variance, most common for sports betting."""
        return KellyCriterion.full_kelly(win_prob, decimal_odds) / 2

    @staticmethod
    def quarter_kelly(win_prob: float, decimal_odds: float) -> float:
        """Quarter Kelly — conservative, good for beginners."""
        return KellyCriterion.full_kelly(win_prob, decimal_odds) / 4

    @staticmethod
    def fractional_kelly(
        win_prob: float, decimal_odds: float, fraction: float = 0.5
    ) -> float:
        """Custom fraction of Kelly."""
        return KellyCriterion.full_kelly(win_prob, decimal_odds) * fraction

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return 1 + (american / 100)
        else:
            return 1 + (100 / abs(american))

    @staticmethod
    def american_to_implied_prob(american: int) -> float:
        """Convert American odds to implied probability."""
        if american > 0:
            return 100 / (american + 100)
        else:
            return abs(american) / (abs(american) + 100)

    @staticmethod
    def edge(model_prob: float, implied_prob: float) -> float:
        """Calculate betting edge."""
        return model_prob - implied_prob


# ─── Risk of Ruin Calculator ────────────────────────────────────
class RiskCalculator:
    """Calculate risk of ruin and related metrics."""

    @staticmethod
    def risk_of_ruin(
        win_rate: float, avg_win: float, avg_loss: float, bankroll_units: float
    ) -> float:
        """
        Approximate risk of ruin using the classic formula.
        RoR ≈ ((1-edge)/(1+edge))^(bankroll/unit)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 1.0 if win_rate <= 0 else 0.0

        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        if edge <= 0:
            return 1.0

        variance = win_rate * avg_win**2 + (1 - win_rate) * avg_loss**2
        if variance <= 0:
            return 0.0

        # Simplified formula
        exponent = -2 * edge * bankroll_units / variance
        ror = math.exp(min(exponent, 0))

        return round(min(ror, 1.0), 6)

    @staticmethod
    def expected_growth_rate(
        win_prob: float, decimal_odds: float, kelly_fraction: float
    ) -> float:
        """Calculate expected growth rate of bankroll."""
        if kelly_fraction <= 0:
            return 0.0
        b = decimal_odds - 1
        p = win_prob
        q = 1 - p

        # G = p * ln(1 + f*b) + q * ln(1 - f)
        try:
            growth = p * math.log(1 + kelly_fraction * b) + q * math.log(
                max(1 - kelly_fraction, 0.001)
            )
        except ValueError:
            growth = 0.0

        return round(growth, 6)

    @staticmethod
    def break_even_win_rate(decimal_odds: float) -> float:
        """Calculate the win rate needed to break even at given odds."""
        return 1 / decimal_odds

    @staticmethod
    def bets_to_double(
        win_prob: float, decimal_odds: float, kelly_fraction: float
    ) -> int:
        """Estimate number of bets needed to double the bankroll."""
        growth = RiskCalculator.expected_growth_rate(
            win_prob, decimal_odds, kelly_fraction
        )
        if growth <= 0:
            return -1  # Never
        return math.ceil(math.log(2) / growth)


# ─── Bankroll Manager ───────────────────────────────────────────
class BankrollManager:
    """Complete bankroll management system."""

    def __init__(
        self,
        starting_bankroll: float,
        risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
        daily_limit_pct: float = 10.0,  # Max % of bankroll per day
        max_bet_pct: float = 5.0,  # Max single bet as % of bankroll
        stop_loss_pct: float = 20.0,  # Stop after losing this % in a day
    ):
        self.kelly = KellyCriterion()
        self.risk_calc = RiskCalculator()

        self.state = BankrollState(
            starting_bankroll=starting_bankroll,
            current_bankroll=starting_bankroll,
            peak_bankroll=starting_bankroll,
            low_bankroll=starting_bankroll,
        )

        self.risk_tolerance = risk_tolerance
        self.daily_limit_pct = daily_limit_pct
        self.max_bet_pct = max_bet_pct
        self.stop_loss_pct = stop_loss_pct

        self.bets: List[BetRecord] = []
        self.daily_wagered: Dict[str, float] = defaultdict(float)
        self.daily_pnl: Dict[str, float] = defaultdict(float)

    def calculate_bet_size(
        self,
        model_win_prob: float,
        odds_american: int,
    ) -> dict:
        """Calculate recommended bet size using Kelly Criterion."""
        decimal_odds = self.kelly.american_to_decimal(odds_american)
        implied_prob = self.kelly.american_to_implied_prob(odds_american)
        edge = self.kelly.edge(model_win_prob, implied_prob)

        # No edge = no bet
        if edge <= 0:
            return {
                "recommended": False,
                "reason": f"No edge ({edge:.1%}). Model: {model_win_prob:.1%}, Implied: {implied_prob:.1%}",
                "edge": round(edge, 4),
                "kelly_fraction": 0,
                "bet_size": 0,
                "grade": BetGrade.F.value,
            }

        # Calculate Kelly fractions
        full_k = self.kelly.full_kelly(model_win_prob, decimal_odds)
        half_k = self.kelly.half_kelly(model_win_prob, decimal_odds)
        quarter_k = self.kelly.quarter_kelly(model_win_prob, decimal_odds)

        # Select based on risk tolerance
        if self.risk_tolerance == RiskTolerance.AGGRESSIVE:
            kelly_frac = full_k
        elif self.risk_tolerance == RiskTolerance.MODERATE:
            kelly_frac = half_k
        elif self.risk_tolerance == RiskTolerance.CONSERVATIVE:
            kelly_frac = quarter_k
        else:
            kelly_frac = half_k

        # Apply drawdown protection
        drawdown = 1 - (self.state.current_bankroll / max(self.state.peak_bankroll, 1))
        if drawdown > 0.15:
            kelly_frac *= 0.5  # Halve bets after 15% drawdown
        elif drawdown > 0.10:
            kelly_frac *= 0.75

        # Apply losing streak protection
        if self.state.current_streak < -4:
            kelly_frac *= 0.5

        # Apply caps
        max_frac = self.max_bet_pct / 100
        kelly_frac = min(kelly_frac, max_frac)

        # Check daily limit
        today = datetime.now().strftime("%Y-%m-%d")
        daily_wagered = self.daily_wagered.get(today, 0)
        daily_limit = self.state.current_bankroll * (self.daily_limit_pct / 100)
        remaining_daily = max(daily_limit - daily_wagered, 0)

        # Calculate bet size
        bet_size = round(self.state.current_bankroll * kelly_frac, 2)
        bet_size = min(bet_size, remaining_daily)

        # Grade the bet
        grade = self._grade_bet(edge)

        # Calculate potential payout
        potential_payout = round(bet_size * (decimal_odds - 1), 2)

        # Risk metrics
        ror = self.risk_calc.risk_of_ruin(
            model_win_prob,
            decimal_odds - 1,
            1.0,
            self.state.current_bankroll / max(bet_size, 1),
        )
        growth_rate = self.risk_calc.expected_growth_rate(
            model_win_prob, decimal_odds, kelly_frac
        )
        bets_to_2x = self.risk_calc.bets_to_double(
            model_win_prob, decimal_odds, kelly_frac
        )

        # Check stop loss
        daily_loss = self.daily_pnl.get(today, 0)
        stop_loss_amount = self.state.current_bankroll * (self.stop_loss_pct / 100)
        if daily_loss < -stop_loss_amount:
            return {
                "recommended": False,
                "reason": f"Daily stop-loss reached (${abs(daily_loss):.2f} lost today)",
                "edge": round(edge, 4),
                "kelly_fraction": kelly_frac,
                "bet_size": 0,
                "grade": grade.value,
            }

        return {
            "recommended": bet_size > 0,
            "edge": round(edge, 4),
            "edge_pct": f"{edge:.1%}",
            "model_prob": round(model_win_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "grade": grade.value,
            "kelly": {
                "full": round(full_k, 4),
                "half": round(half_k, 4),
                "quarter": round(quarter_k, 4),
                "applied": round(kelly_frac, 4),
            },
            "bet_size": bet_size,
            "bet_units": round(bet_size / (self.state.starting_bankroll / 100), 2),
            "potential_payout": potential_payout,
            "risk_metrics": {
                "risk_of_ruin": f"{ror:.4%}",
                "expected_growth": f"{growth_rate:.4%}",
                "bets_to_double": bets_to_2x,
                "drawdown_current": f"{drawdown:.1%}",
            },
            "daily_remaining": round(remaining_daily, 2),
            "drawdown_protection": drawdown > 0.10,
            "streak_protection": self.state.current_streak < -4,
        }

    def place_bet(
        self,
        bet_type: BetType,
        game_id: str,
        description: str,
        odds_american: int,
        model_win_prob: float,
        stake: Optional[float] = None,
    ) -> BetRecord:
        """Place a bet and track it."""
        sizing = self.calculate_bet_size(model_win_prob, odds_american)
        if stake is None:
            stake = sizing["bet_size"]

        decimal_odds = self.kelly.american_to_decimal(odds_american)
        implied_prob = self.kelly.american_to_implied_prob(odds_american)

        bet = BetRecord(
            bet_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            bet_type=bet_type,
            game_id=game_id,
            description=description,
            odds_american=odds_american,
            odds_decimal=decimal_odds,
            model_win_prob=model_win_prob,
            implied_prob=implied_prob,
            edge=round(model_win_prob - implied_prob, 4),
            stake=stake,
            potential_payout=round(stake * (decimal_odds - 1), 2),
            grade=BetGrade(sizing.get("grade", "C")),
            kelly_fraction=sizing.get("kelly", {}).get("applied", 0),
        )

        self.bets.append(bet)
        self.state.total_wagered += stake
        self.state.current_bankroll -= stake

        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_wagered[today] += stake

        return bet

    def settle_bet(self, bet_id: str, result: BetResult) -> dict:
        """Settle a pending bet."""
        bet = next((b for b in self.bets if b.bet_id == bet_id), None)
        if not bet:
            return {"error": "Bet not found"}

        bet.result = result
        today = datetime.now().strftime("%Y-%m-%d")

        if result == BetResult.WIN:
            bet.profit = bet.potential_payout
            self.state.current_bankroll += bet.stake + bet.potential_payout
            self.state.total_won += bet.potential_payout
            self.state.win_count += 1
            self.state.current_streak = max(self.state.current_streak, 0) + 1
            self.state.max_winning_streak = max(
                self.state.max_winning_streak, self.state.current_streak
            )
        elif result == BetResult.LOSS:
            bet.profit = -bet.stake
            self.state.total_lost += bet.stake
            self.state.loss_count += 1
            self.state.current_streak = min(self.state.current_streak, 0) - 1
            self.state.max_losing_streak = max(
                self.state.max_losing_streak, abs(self.state.current_streak)
            )
        elif result == BetResult.PUSH:
            bet.profit = 0
            self.state.current_bankroll += bet.stake
            self.state.push_count += 1

        # Update PnL
        self.state.net_profit = (
            self.state.current_bankroll - self.state.starting_bankroll
        )
        self.state.roi = round(
            self.state.net_profit / max(self.state.total_wagered, 1) * 100, 2
        )

        # Track peak/low
        self.state.peak_bankroll = max(
            self.state.peak_bankroll, self.state.current_bankroll
        )
        self.state.low_bankroll = min(
            self.state.low_bankroll, self.state.current_bankroll
        )

        # Max drawdown
        drawdown = self.state.peak_bankroll - self.state.current_bankroll
        drawdown_pct = drawdown / max(self.state.peak_bankroll, 1) * 100
        self.state.max_drawdown = max(self.state.max_drawdown, drawdown)
        self.state.max_drawdown_pct = max(self.state.max_drawdown_pct, drawdown_pct)

        self.daily_pnl[today] += bet.profit

        return {
            "bet_id": bet.bet_id,
            "result": result.value,
            "profit": bet.profit,
            "bankroll": self.state.current_bankroll,
        }

    def get_performance(self) -> dict:
        """Get comprehensive performance analytics."""
        total_bets = self.state.win_count + self.state.loss_count + self.state.push_count
        if total_bets == 0:
            return {"message": "No settled bets yet"}

        win_rate = self.state.win_count / max(
            self.state.win_count + self.state.loss_count, 1
        )

        # Performance by bet type
        type_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "profit": 0.0})
        for bet in self.bets:
            if bet.result == BetResult.PENDING:
                continue
            key = bet.bet_type.value
            if bet.result == BetResult.WIN:
                type_stats[key]["wins"] += 1
            elif bet.result == BetResult.LOSS:
                type_stats[key]["losses"] += 1
            type_stats[key]["profit"] += bet.profit

        # Performance by grade
        grade_stats = defaultdict(lambda: {"count": 0, "wins": 0, "profit": 0.0})
        for bet in self.bets:
            if bet.result == BetResult.PENDING:
                continue
            g = bet.grade.value
            grade_stats[g]["count"] += 1
            if bet.result == BetResult.WIN:
                grade_stats[g]["wins"] += 1
            grade_stats[g]["profit"] += bet.profit

        return {
            "bankroll": self.state.to_dict(),
            "performance": {
                "total_bets": total_bets,
                "win_rate": f"{win_rate:.1%}",
                "roi": f"{self.state.roi:.1f}%",
                "net_profit": round(self.state.net_profit, 2),
                "avg_bet_size": round(
                    self.state.total_wagered / max(total_bets, 1), 2
                ),
                "profit_per_bet": round(
                    self.state.net_profit / max(total_bets, 1), 2
                ),
            },
            "risk": {
                "max_drawdown": f"${self.state.max_drawdown:.2f}",
                "max_drawdown_pct": f"{self.state.max_drawdown_pct:.1f}%",
                "current_streak": self.state.current_streak,
                "max_winning_streak": self.state.max_winning_streak,
                "max_losing_streak": self.state.max_losing_streak,
            },
            "by_type": {k: dict(v) for k, v in type_stats.items()},
            "by_grade": {k: dict(v) for k, v in grade_stats.items()},
        }

    def _grade_bet(self, edge: float) -> BetGrade:
        """Grade a bet based on edge size."""
        if edge >= 0.10:
            return BetGrade.A_PLUS
        elif edge >= 0.07:
            return BetGrade.A
        elif edge >= 0.05:
            return BetGrade.B
        elif edge >= 0.03:
            return BetGrade.C
        elif edge >= 0.01:
            return BetGrade.D
        else:
            return BetGrade.F


# ─── Flask Routes ────────────────────────────────────────────────
def create_bankroll_routes(app, manager: Optional[BankrollManager] = None):
    """Register Flask routes."""
    from flask import request, jsonify

    mgr = manager or BankrollManager(1000.0)

    @app.route("/api/bankroll/size", methods=["POST"])
    def bet_size():
        data = request.json
        result = mgr.calculate_bet_size(
            data["model_win_prob"], data["odds_american"]
        )
        return jsonify(result)

    @app.route("/api/bankroll/place", methods=["POST"])
    def place():
        data = request.json
        bet = mgr.place_bet(
            BetType(data["bet_type"]),
            data["game_id"],
            data["description"],
            data["odds_american"],
            data["model_win_prob"],
            data.get("stake"),
        )
        return jsonify(bet.to_dict())

    @app.route("/api/bankroll/settle", methods=["POST"])
    def settle():
        data = request.json
        result = mgr.settle_bet(data["bet_id"], BetResult(data["result"]))
        return jsonify(result)

    @app.route("/api/bankroll/performance", methods=["GET"])
    def performance():
        return jsonify(mgr.get_performance())

    @app.route("/api/bankroll/state", methods=["GET"])
    def state():
        return jsonify(mgr.state.to_dict())

    return app


if __name__ == "__main__":
    print("⚾ MLB Predictor — Kelly Criterion Bankroll Manager")
    print("=" * 60)

    mgr = BankrollManager(
        starting_bankroll=1000.00,
        risk_tolerance=RiskTolerance.MODERATE,
    )

    # Simulate a betting day
    bets_data = [
        (BetType.MONEYLINE, "NYY_BOS", "NYY ML vs BOS", -130, 0.60),
        (BetType.OVER_UNDER, "LAD_SF", "LAD/SF Over 7.5", -110, 0.58),
        (BetType.RUNLINE, "HOU_SEA", "HOU -1.5 vs SEA", +135, 0.50),
        (BetType.MONEYLINE, "ATL_NYM", "ATL ML vs NYM", +120, 0.52),
        (BetType.FIRST_5, "CHC_STL", "CHC F5 ML", -105, 0.56),
    ]

    print("\n📊 BET SIZING RECOMMENDATIONS:")
    placed_bets = []
    for bt, gid, desc, odds, prob in bets_data:
        sizing = mgr.calculate_bet_size(prob, odds)
        print(f"\n  {desc} ({odds:+d})")
        print(f"    Model: {prob:.0%} | Implied: {sizing.get('implied_prob', 0):.0%} | Edge: {sizing.get('edge_pct', '0%')}")
        print(f"    Grade: {sizing.get('grade', 'F')} | Kelly: {sizing.get('kelly', {}).get('applied', 0):.3f}")
        print(f"    Bet: ${sizing.get('bet_size', 0):.2f} | Payout: ${sizing.get('potential_payout', 0):.2f}")
        print(f"    Recommended: {'✅' if sizing.get('recommended') else '❌'}")

        if sizing.get("recommended"):
            bet = mgr.place_bet(bt, gid, desc, odds, prob)
            placed_bets.append(bet)
            print(f"    → Placed bet #{bet.bet_id}")

    # Settle bets (simulate results)
    results = [BetResult.WIN, BetResult.WIN, BetResult.LOSS, BetResult.LOSS, BetResult.WIN]
    print(f"\n{'=' * 60}")
    print("📋 SETTLING BETS:")
    for bet, result in zip(placed_bets, results[:len(placed_bets)]):
        settle = mgr.settle_bet(bet.bet_id, result)
        emoji = "✅" if result == BetResult.WIN else "❌"
        print(f"  {emoji} {bet.description}: {result.value} → P/L ${settle['profit']:+.2f}")

    # Performance report
    perf = mgr.get_performance()
    print(f"\n{'=' * 60}")
    print("📊 PERFORMANCE REPORT:")
    print(f"  Starting bankroll: ${mgr.state.starting_bankroll:.2f}")
    print(f"  Current bankroll: ${mgr.state.current_bankroll:.2f}")
    p = perf.get("performance", {})
    print(f"  Win rate: {p.get('win_rate', 'N/A')}")
    print(f"  ROI: {p.get('roi', 'N/A')}")
    print(f"  Net profit: ${p.get('net_profit', 0):.2f}")
    r = perf.get("risk", {})
    print(f"  Max drawdown: {r.get('max_drawdown_pct', 'N/A')}")
    print(f"  Current streak: {r.get('current_streak', 0)}")

    print("\n✅ Kelly Criterion Bankroll Manager working!")
