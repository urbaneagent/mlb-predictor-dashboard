"""
MLB Predictor - Bankroll Management System
Implements Kelly Criterion, fractional Kelly, and flat betting strategies.
Tracks risk of ruin, drawdown, and optimal bet sizing.
"""
import math
from dataclasses import dataclass, field, asdict


@dataclass
class BetSizeResult:
    """Optimal bet size calculation."""
    strategy: str
    bankroll: float
    bet_amount: float
    bet_units: float
    unit_size: float
    max_exposure: float
    risk_of_ruin: float
    kelly_fraction: float = 0
    edge: float = 0
    probability: float = 0
    reasoning: str = ""


@dataclass
class BankrollSnapshot:
    """Current bankroll state."""
    starting_bankroll: float
    current_bankroll: float
    peak_bankroll: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    total_bets: int
    wins: int
    losses: int
    pushes: int
    profit_loss: float
    roi: float
    avg_bet_size: float
    unit_size: float
    units_won_lost: float
    win_rate: float
    longest_win_streak: int
    longest_loss_streak: int
    current_streak: int
    current_streak_type: str  # win, loss
    sharpe_ratio: float = 0
    risk_of_ruin: float = 0


class BankrollManager:
    """
    Manages bankroll with multiple staking strategies.
    Protects against ruin while maximizing long-term growth.
    """

    def __init__(self, starting_bankroll: float = 1000, unit_pct: float = 2.0):
        self.starting = starting_bankroll
        self.current = starting_bankroll
        self.peak = starting_bankroll
        self.unit_pct = unit_pct  # Base unit as % of bankroll
        self.bets = []
        self.daily_returns = []

    @property
    def unit_size(self) -> float:
        return round(self.current * (self.unit_pct / 100), 2)

    def kelly_criterion(self, probability: float, odds: int) -> BetSizeResult:
        """
        Full Kelly Criterion bet sizing.
        Kelly% = (bp - q) / b
        where b = decimal odds - 1, p = win prob, q = 1-p
        """
        decimal_odds = self._american_to_decimal(odds)
        b = decimal_odds - 1
        p = probability
        q = 1 - p

        kelly_pct = (b * p - q) / b if b > 0 else 0
        kelly_pct = max(0, kelly_pct)  # Never bet negative

        bet_amount = round(self.current * kelly_pct, 2)
        edge = (probability * decimal_odds) - 1

        return BetSizeResult(
            strategy="full_kelly",
            bankroll=self.current,
            bet_amount=bet_amount,
            bet_units=round(bet_amount / self.unit_size, 1) if self.unit_size > 0 else 0,
            unit_size=self.unit_size,
            max_exposure=round(bet_amount / self.current * 100, 1),
            risk_of_ruin=self._risk_of_ruin(kelly_pct, probability),
            kelly_fraction=round(kelly_pct, 4),
            edge=round(edge, 4),
            probability=probability,
            reasoning=f"Full Kelly: {kelly_pct*100:.1f}% of bankroll. Edge: {edge*100:.1f}%. ⚠️ Full Kelly is aggressive — consider fractional."
        )

    def fractional_kelly(self, probability: float, odds: int, fraction: float = 0.5) -> BetSizeResult:
        """
        Fractional Kelly (recommended: 25-50% Kelly).
        Reduces variance dramatically while keeping ~75% of growth rate.
        """
        full = self.kelly_criterion(probability, odds)
        frac_amount = round(full.bet_amount * fraction, 2)
        frac_pct = full.kelly_fraction * fraction

        return BetSizeResult(
            strategy=f"kelly_{int(fraction*100)}pct",
            bankroll=self.current,
            bet_amount=frac_amount,
            bet_units=round(frac_amount / self.unit_size, 1) if self.unit_size > 0 else 0,
            unit_size=self.unit_size,
            max_exposure=round(frac_amount / self.current * 100, 1),
            risk_of_ruin=self._risk_of_ruin(frac_pct, probability),
            kelly_fraction=round(frac_pct, 4),
            edge=full.edge,
            probability=probability,
            reasoning=f"{int(fraction*100)}% Kelly: ${frac_amount:.2f}. Much safer than full Kelly with ~{int(75 + fraction*10)}% of growth rate."
        )

    def confidence_sizing(self, probability: float, odds: int, confidence: str) -> BetSizeResult:
        """
        Bet sizing based on confidence level.
        HIGH: 2-3 units, MEDIUM: 1-2 units, LOW: 0.5-1 unit
        """
        multipliers = {
            "HIGH": 2.5,
            "MEDIUM": 1.5,
            "LOW": 0.75,
            "MAX": 3.0
        }

        mult = multipliers.get(confidence.upper(), 1.0)
        bet_amount = round(self.unit_size * mult, 2)

        # Cap at 5% of bankroll
        max_bet = self.current * 0.05
        bet_amount = min(bet_amount, max_bet)

        edge = (probability * self._american_to_decimal(odds)) - 1

        return BetSizeResult(
            strategy=f"confidence_{confidence.lower()}",
            bankroll=self.current,
            bet_amount=bet_amount,
            bet_units=mult,
            unit_size=self.unit_size,
            max_exposure=round(bet_amount / self.current * 100, 1),
            risk_of_ruin=self._risk_of_ruin(bet_amount / self.current, probability),
            edge=round(edge, 4),
            probability=probability,
            reasoning=f"{confidence} confidence: {mult:.1f} units (${bet_amount:.2f}). Max 5% bankroll exposure."
        )

    def flat_bet(self, units: float = 1.0) -> BetSizeResult:
        """Simple flat betting (consistent unit size)."""
        bet_amount = round(self.unit_size * units, 2)

        return BetSizeResult(
            strategy="flat",
            bankroll=self.current,
            bet_amount=bet_amount,
            bet_units=units,
            unit_size=self.unit_size,
            max_exposure=round(bet_amount / self.current * 100, 1),
            risk_of_ruin=0.05,  # Approximate for flat betting
            reasoning=f"Flat bet: {units:.1f}u = ${bet_amount:.2f}. Simple, consistent, low variance."
        )

    def record_bet(self, amount: float, odds: int, result: str, units: float = 0):
        """Record a bet result. result: 'win', 'loss', 'push'."""
        decimal_odds = self._american_to_decimal(odds)

        if result == 'win':
            profit = amount * (decimal_odds - 1)
        elif result == 'loss':
            profit = -amount
        else:  # push
            profit = 0

        self.current = round(self.current + profit, 2)
        self.peak = max(self.peak, self.current)

        self.bets.append({
            "amount": amount,
            "odds": odds,
            "result": result,
            "profit": round(profit, 2),
            "bankroll_after": self.current,
            "units": units
        })

    def get_snapshot(self) -> BankrollSnapshot:
        """Get current bankroll state."""
        if not self.bets:
            return BankrollSnapshot(
                starting_bankroll=self.starting, current_bankroll=self.current,
                peak_bankroll=self.peak, max_drawdown_pct=0, current_drawdown_pct=0,
                total_bets=0, wins=0, losses=0, pushes=0,
                profit_loss=0, roi=0, avg_bet_size=0,
                unit_size=self.unit_size, units_won_lost=0, win_rate=0,
                longest_win_streak=0, longest_loss_streak=0,
                current_streak=0, current_streak_type=""
            )

        wins = len([b for b in self.bets if b['result'] == 'win'])
        losses = len([b for b in self.bets if b['result'] == 'loss'])
        pushes = len([b for b in self.bets if b['result'] == 'push'])
        total = len(self.bets)

        total_wagered = sum(b['amount'] for b in self.bets)
        profit = self.current - self.starting
        roi = (profit / total_wagered * 100) if total_wagered > 0 else 0

        # Drawdown
        max_dd = 0
        peak_so_far = self.starting
        for b in self.bets:
            peak_so_far = max(peak_so_far, b['bankroll_after'])
            dd = (peak_so_far - b['bankroll_after']) / peak_so_far * 100
            max_dd = max(max_dd, dd)

        current_dd = (self.peak - self.current) / self.peak * 100 if self.peak > 0 else 0

        # Streaks
        win_streak = loss_streak = max_win = max_loss = curr = 0
        curr_type = ""
        for b in self.bets:
            if b['result'] == 'win':
                if curr_type == 'win':
                    curr += 1
                else:
                    curr = 1
                    curr_type = 'win'
                max_win = max(max_win, curr)
            elif b['result'] == 'loss':
                if curr_type == 'loss':
                    curr += 1
                else:
                    curr = 1
                    curr_type = 'loss'
                max_loss = max(max_loss, curr)

        # Units
        units_wl = sum(b.get('units', 0) * (1 if b['result'] == 'win' else -1)
                      for b in self.bets if b['result'] != 'push')

        return BankrollSnapshot(
            starting_bankroll=self.starting,
            current_bankroll=self.current,
            peak_bankroll=self.peak,
            max_drawdown_pct=round(max_dd, 1),
            current_drawdown_pct=round(current_dd, 1),
            total_bets=total,
            wins=wins, losses=losses, pushes=pushes,
            profit_loss=round(profit, 2),
            roi=round(roi, 1),
            avg_bet_size=round(total_wagered / total, 2) if total > 0 else 0,
            unit_size=self.unit_size,
            units_won_lost=round(units_wl, 1),
            win_rate=round(wins / (wins + losses) * 100, 1) if (wins + losses) > 0 else 0,
            longest_win_streak=max_win,
            longest_loss_streak=max_loss,
            current_streak=curr,
            current_streak_type=curr_type,
            risk_of_ruin=self._portfolio_risk_of_ruin()
        )

    # ============= Internal =============

    def _american_to_decimal(self, odds: int) -> float:
        if odds > 0:
            return 1 + odds / 100
        elif odds < 0:
            return 1 + 100 / abs(odds)
        return 1

    def _risk_of_ruin(self, bet_fraction: float, win_prob: float) -> float:
        """Approximate risk of ruin."""
        if bet_fraction <= 0 or win_prob <= 0:
            return 0
        if win_prob >= 1:
            return 0

        q = 1 - win_prob
        if win_prob <= q:
            return 1.0  # Negative edge = guaranteed ruin eventually

        try:
            n_bets_to_ruin = math.log(0.01) / math.log(1 - bet_fraction)  # Bets to lose 99%
            prob_all_loss = q ** n_bets_to_ruin
            return round(min(1.0, prob_all_loss * 10), 4)  # Rough estimate
        except:
            return 0.5

    def _portfolio_risk_of_ruin(self) -> float:
        """Estimate current risk of ruin based on betting history."""
        if len(self.bets) < 10:
            return 0.1  # Not enough data

        wins = len([b for b in self.bets if b['result'] == 'win'])
        total = len([b for b in self.bets if b['result'] != 'push'])
        if total == 0:
            return 0

        win_rate = wins / total
        avg_bet_pct = sum(b['amount'] for b in self.bets) / (total * self.current) if self.current > 0 else 0

        return self._risk_of_ruin(avg_bet_pct, win_rate)


def create_bankroll_routes(app, manager: BankrollManager):
    """Create FastAPI routes."""

    @app.post("/api/v1/bankroll/kelly")
    async def kelly(probability: float, odds: int, fraction: float = 0.5):
        if fraction >= 1.0:
            return asdict(manager.kelly_criterion(probability, odds))
        return asdict(manager.fractional_kelly(probability, odds, fraction))

    @app.post("/api/v1/bankroll/size")
    async def size_bet(probability: float, odds: int, confidence: str = "MEDIUM"):
        return asdict(manager.confidence_sizing(probability, odds, confidence))

    @app.post("/api/v1/bankroll/record")
    async def record(amount: float, odds: int, result: str, units: float = 1):
        manager.record_bet(amount, odds, result, units)
        return {"recorded": True, "bankroll": manager.current}

    @app.get("/api/v1/bankroll/snapshot")
    async def snapshot():
        return asdict(manager.get_snapshot())

    return manager
