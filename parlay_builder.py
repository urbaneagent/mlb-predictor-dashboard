"""
MLB Predictor - Parlay Builder & Bet Slip Engine
Combine individual picks into parlays, calculate combined odds,
apply correlation adjustments, and manage bet slips.
"""
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Pick:
    """A single betting pick."""
    pick_id: str
    game_id: str
    bet_type: str  # moneyline, spread, total, prop
    selection: str  # e.g., "NYY ML", "Over 8.5", "Judge HR"
    odds: int  # American odds (e.g., -150, +120)
    confidence: float = 0.0  # 0-100
    edge: float = 0.0  # % edge over market
    model_win_prob: float = 0.0
    implied_prob: float = 0.0
    home_team: str = ""
    away_team: str = ""
    umpire: str = ""
    notes: str = ""


@dataclass
class Parlay:
    """A parlay combining multiple picks."""
    parlay_id: str
    picks: list = field(default_factory=list)
    combined_odds: int = 0
    combined_prob: float = 0.0
    correlation_adjusted_prob: float = 0.0
    stake: float = 0.0
    potential_payout: float = 0.0
    status: str = "pending"  # pending, won, lost, push
    created_at: str = ""
    result_notes: str = ""


@dataclass
class BetSlip:
    """A complete bet slip with singles and parlays."""
    slip_id: str
    date: str
    singles: list = field(default_factory=list)
    parlays: list = field(default_factory=list)
    total_stake: float = 0.0
    total_potential_payout: float = 0.0
    status: str = "active"  # active, settled
    created_at: str = ""


class OddsCalculator:
    """Utility for odds calculations."""

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal."""
        if american > 0:
            return (american / 100.0) + 1.0
        else:
            return (100.0 / abs(american)) + 1.0

    @staticmethod
    def decimal_to_american(decimal: float) -> int:
        """Convert decimal odds to American."""
        if decimal >= 2.0:
            return int(round((decimal - 1) * 100))
        else:
            return int(round(-100 / (decimal - 1)))

    @staticmethod
    def american_to_implied(american: int) -> float:
        """Convert American odds to implied probability."""
        if american > 0:
            return 100.0 / (american + 100)
        else:
            return abs(american) / (abs(american) + 100)

    @staticmethod
    def probability_to_american(prob: float) -> int:
        """Convert probability to American odds."""
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return int(round(-100 * prob / (1 - prob)))
        else:
            return int(round(100 * (1 - prob) / prob))

    @staticmethod
    def calculate_parlay_decimal(odds_list: list) -> float:
        """Calculate combined decimal odds for a parlay."""
        result = 1.0
        for odds in odds_list:
            result *= OddsCalculator.american_to_decimal(odds)
        return result

    @staticmethod
    def calculate_parlay_payout(odds_list: list, stake: float) -> float:
        """Calculate potential payout for a parlay."""
        decimal = OddsCalculator.calculate_parlay_decimal(odds_list)
        return round(stake * decimal, 2)

    @staticmethod
    def calculate_ev(probability: float, odds: int, stake: float) -> float:
        """Calculate expected value of a bet."""
        decimal = OddsCalculator.american_to_decimal(odds)
        profit = stake * (decimal - 1)
        ev = (probability * profit) - ((1 - probability) * stake)
        return round(ev, 2)

    @staticmethod
    def kelly_criterion(probability: float, odds: int,
                        fraction: float = 0.25) -> float:
        """
        Calculate Kelly Criterion bet size.
        fraction: Kelly fraction (0.25 = quarter Kelly, safer)
        """
        decimal = OddsCalculator.american_to_decimal(odds)
        b = decimal - 1
        q = 1 - probability
        kelly = (b * probability - q) / b
        return max(0, kelly * fraction)


class ParlayBuilder:
    """
    Build and manage parlays with correlation adjustments.
    """

    # Correlation factors between bet types
    CORRELATION_MATRIX = {
        # Same game correlations
        ("moneyline", "total"): 0.15,      # Moderate negative correlation
        ("moneyline", "spread"): 0.90,     # High correlation
        ("spread", "total"): 0.10,         # Low correlation
        ("moneyline", "prop"): 0.20,       # Some correlation
        ("total", "prop"): 0.25,           # Some correlation
        # Cross-game correlations
        ("cross_game", "cross_game"): 0.0, # Independent
    }

    def __init__(self, storage_dir: str = "./bet_slips"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.calc = OddsCalculator()
        self._slips_file = self.storage_dir / "bet_slips.json"
        self.slips: list = self._load()

    def _load(self) -> list:
        if self._slips_file.exists():
            return json.loads(self._slips_file.read_text())
        return []

    def _save(self):
        self._slips_file.write_text(json.dumps(self.slips, indent=2))

    def _generate_id(self, prefix: str = "PKR") -> str:
        import uuid
        return f"{prefix}-{str(uuid.uuid4())[:6].upper()}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_pick(self, game_id: str, bet_type: str, selection: str,
                    odds: int, model_win_prob: float = 0.0,
                    home_team: str = "", away_team: str = "",
                    umpire: str = "", notes: str = "") -> Pick:
        """Create a single pick."""
        implied = self.calc.american_to_implied(odds)
        edge = (model_win_prob - implied) * 100 if model_win_prob else 0

        return Pick(
            pick_id=self._generate_id("PCK"),
            game_id=game_id,
            bet_type=bet_type,
            selection=selection,
            odds=odds,
            confidence=min(100, max(0, edge * 2 + 50)),
            edge=round(edge, 1),
            model_win_prob=model_win_prob,
            implied_prob=round(implied, 3),
            home_team=home_team,
            away_team=away_team,
            umpire=umpire,
            notes=notes,
        )

    def build_parlay(self, picks: list, stake: float = 10.0) -> Parlay:
        """Build a parlay from a list of picks."""
        if len(picks) < 2:
            raise ValueError("Parlay requires at least 2 picks")
        if len(picks) > 10:
            raise ValueError("Maximum 10 legs per parlay")

        parlay_id = self._generate_id("PKR")
        odds_list = [p.odds for p in picks]

        # Calculate combined odds
        combined_decimal = self.calc.calculate_parlay_decimal(odds_list)
        combined_american = self.calc.decimal_to_american(combined_decimal)

        # Calculate combined probability (naive)
        combined_prob = 1.0
        for p in picks:
            prob = p.model_win_prob if p.model_win_prob else (
                1 - self.calc.american_to_implied(p.odds)
            )
            combined_prob *= max(0.01, min(0.99, prob))

        # Apply correlation adjustments
        adjusted_prob = self._apply_correlation(picks, combined_prob)

        # Calculate payout
        payout = round(stake * combined_decimal, 2)

        parlay = Parlay(
            parlay_id=parlay_id,
            picks=[self._pick_to_dict(p) for p in picks],
            combined_odds=combined_american,
            combined_prob=round(combined_prob, 4),
            correlation_adjusted_prob=round(adjusted_prob, 4),
            stake=stake,
            potential_payout=payout,
            created_at=self._now(),
        )

        return parlay

    def _apply_correlation(self, picks: list, base_prob: float) -> float:
        """Adjust probability based on correlation between picks."""
        if len(picks) < 2:
            return base_prob

        adjustment = 1.0

        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                p1, p2 = picks[i], picks[j]

                if p1.game_id == p2.game_id:
                    # Same game parlay - check correlation
                    key = tuple(sorted([p1.bet_type, p2.bet_type]))
                    correlation = self.CORRELATION_MATRIX.get(key, 0.15)
                    # Higher correlation = picks are less independent
                    # Adjust probability upward (correlated outcomes more likely
                    # to both hit)
                    adjustment *= (1 + correlation * 0.1)
                # Cross-game: no adjustment (independent events)

        return min(0.95, base_prob * adjustment)

    def _pick_to_dict(self, pick: Pick) -> dict:
        return {
            "pick_id": pick.pick_id,
            "game_id": pick.game_id,
            "bet_type": pick.bet_type,
            "selection": pick.selection,
            "odds": pick.odds,
            "confidence": pick.confidence,
            "edge": pick.edge,
            "model_win_prob": pick.model_win_prob,
            "implied_prob": pick.implied_prob,
            "home_team": pick.home_team,
            "away_team": pick.away_team,
        }

    def create_bet_slip(self, singles: list = None,
                        parlays: list = None) -> BetSlip:
        """Create a complete bet slip."""
        slip = BetSlip(
            slip_id=self._generate_id("SLP"),
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            singles=[self._pick_to_dict(p) for p in (singles or [])],
            parlays=[{
                "parlay_id": pk.parlay_id,
                "picks": pk.picks,
                "combined_odds": pk.combined_odds,
                "combined_prob": pk.combined_prob,
                "correlation_adjusted_prob": pk.correlation_adjusted_prob,
                "stake": pk.stake,
                "potential_payout": pk.potential_payout,
                "status": pk.status,
            } for pk in (parlays or [])],
            created_at=self._now(),
        )

        slip.total_stake = sum(
            s.get("stake", 0) if isinstance(s, dict) else 0
            for s in slip.singles
        ) + sum(p.stake for p in (parlays or []))

        slip.total_potential_payout = sum(
            p.potential_payout for p in (parlays or [])
        )

        self.slips.append({
            "slip_id": slip.slip_id,
            "date": slip.date,
            "singles": slip.singles,
            "parlays": slip.parlays,
            "total_stake": slip.total_stake,
            "total_potential_payout": slip.total_potential_payout,
            "status": slip.status,
            "created_at": slip.created_at,
        })
        self._save()

        return slip

    def suggest_parlays(self, picks: list,
                        max_legs: int = 3) -> list:
        """
        Auto-suggest optimal parlays from available picks.
        Prioritizes high-confidence, low-correlation combinations.
        """
        if len(picks) < 2:
            return []

        # Sort by confidence/edge
        sorted_picks = sorted(picks, key=lambda p: p.edge, reverse=True)

        suggestions = []

        # 2-leg parlays with highest edge picks
        for i in range(min(len(sorted_picks), 5)):
            for j in range(i + 1, min(len(sorted_picks), 5)):
                p1, p2 = sorted_picks[i], sorted_picks[j]

                # Prefer cross-game parlays (more independent)
                if p1.game_id != p2.game_id:
                    try:
                        parlay = self.build_parlay([p1, p2])
                        if parlay.correlation_adjusted_prob > 0.15:
                            suggestions.append(parlay)
                    except ValueError:
                        pass

        # 3-leg parlays if enough picks
        if max_legs >= 3 and len(sorted_picks) >= 3:
            for i in range(min(len(sorted_picks), 4)):
                for j in range(i + 1, min(len(sorted_picks), 4)):
                    for k in range(j + 1, min(len(sorted_picks), 4)):
                        p1 = sorted_picks[i]
                        p2 = sorted_picks[j]
                        p3 = sorted_picks[k]
                        game_ids = {p1.game_id, p2.game_id, p3.game_id}
                        if len(game_ids) >= 2:
                            try:
                                parlay = self.build_parlay([p1, p2, p3])
                                if parlay.correlation_adjusted_prob > 0.10:
                                    suggestions.append(parlay)
                            except ValueError:
                                pass

        # Sort by EV
        suggestions.sort(
            key=lambda p: p.correlation_adjusted_prob * (
                self.calc.american_to_decimal(p.combined_odds) - 1
            ),
            reverse=True
        )

        return suggestions[:5]  # Top 5 suggested parlays

    def format_bet_slip(self, slip: BetSlip) -> str:
        """Format a bet slip for display."""
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║              MLB PREDICTOR BET SLIP              ║",
            "╚══════════════════════════════════════════════════╝",
            f"  📅 {slip.date}  |  Slip ID: {slip.slip_id}",
            "",
        ]

        if slip.singles:
            lines.append("━━━ STRAIGHT BETS ━━━")
            for pick in slip.singles:
                lines.append(
                    f"  {pick['selection']} ({pick['odds']:+d}) "
                    f"| Edge: {pick['edge']:+.1f}% "
                    f"| 🎯 {pick['confidence']:.0f}%"
                )
            lines.append("")

        if slip.parlays:
            for i, parlay in enumerate(slip.parlays, 1):
                p = parlay if isinstance(parlay, dict) else {
                    "picks": parlay.picks,
                    "combined_odds": parlay.combined_odds,
                    "stake": parlay.stake,
                    "potential_payout": parlay.potential_payout,
                    "correlation_adjusted_prob": parlay.correlation_adjusted_prob,
                }
                legs = len(p.get("picks", []))
                lines.append(f"━━━ PARLAY #{i} ({legs}-LEG) ━━━")
                for pick in p.get("picks", []):
                    lines.append(
                        f"  ✓ {pick['selection']} ({pick['odds']:+d})"
                    )
                lines.append(
                    f"  Combined: {p.get('combined_odds', 0):+d} | "
                    f"Stake: ${p.get('stake', 0):.2f} | "
                    f"Payout: ${p.get('potential_payout', 0):.2f}"
                )
                adj_prob = p.get("correlation_adjusted_prob", 0)
                lines.append(
                    f"  Win Prob: {adj_prob * 100:.1f}% | "
                    f"EV: ${self.calc.calculate_ev(adj_prob, p.get('combined_odds', 100), p.get('stake', 10)):.2f}"
                )
                lines.append("")

        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  💰 Total Stake: ${slip.total_stake:.2f}",
            f"  🎯 Max Payout: ${slip.total_potential_payout:.2f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ])

        return "\n".join(lines)


class PerformanceTracker:
    """
    Track betting performance over time:
    win rate, ROI, profit/loss, streak tracking.
    """

    def __init__(self, storage_dir: str = "./performance"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._bets_file = self.storage_dir / "bet_history.json"
        self.bets: list = self._load()

    def _load(self) -> list:
        if self._bets_file.exists():
            return json.loads(self._bets_file.read_text())
        return []

    def _save(self):
        self._bets_file.write_text(json.dumps(self.bets, indent=2))

    def record_bet(self, bet_type: str, selection: str, odds: int,
                   stake: float, result: str, payout: float = 0,
                   date: str = "", notes: str = "") -> dict:
        """Record a settled bet."""
        profit = payout - stake if result == "won" else -stake
        if result == "push":
            profit = 0

        bet = {
            "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "bet_type": bet_type,
            "selection": selection,
            "odds": odds,
            "stake": stake,
            "result": result,
            "payout": payout,
            "profit": round(profit, 2),
            "notes": notes,
        }
        self.bets.append(bet)
        self._save()
        return bet

    def get_stats(self, period_days: int = 30) -> dict:
        """Get performance statistics."""
        if not self.bets:
            return self._empty_stats()

        total = len(self.bets)
        won = sum(1 for b in self.bets if b["result"] == "won")
        lost = sum(1 for b in self.bets if b["result"] == "lost")
        pushed = sum(1 for b in self.bets if b["result"] == "push")

        total_staked = sum(b["stake"] for b in self.bets)
        total_profit = sum(b["profit"] for b in self.bets)

        # Streak tracking
        current_streak = 0
        streak_type = ""
        for bet in reversed(self.bets):
            if bet["result"] == "push":
                continue
            if not streak_type:
                streak_type = bet["result"]
                current_streak = 1
            elif bet["result"] == streak_type:
                current_streak += 1
            else:
                break

        # Best/worst day
        daily = {}
        for b in self.bets:
            d = b["date"]
            daily[d] = daily.get(d, 0) + b["profit"]

        best_day = max(daily.items(), key=lambda x: x[1]) if daily else ("N/A", 0)
        worst_day = min(daily.items(), key=lambda x: x[1]) if daily else ("N/A", 0)

        # By bet type
        by_type = {}
        for b in self.bets:
            t = b["bet_type"]
            if t not in by_type:
                by_type[t] = {"total": 0, "won": 0, "profit": 0, "staked": 0}
            by_type[t]["total"] += 1
            if b["result"] == "won":
                by_type[t]["won"] += 1
            by_type[t]["profit"] += b["profit"]
            by_type[t]["staked"] += b["stake"]

        return {
            "total_bets": total,
            "record": f"{won}-{lost}-{pushed}",
            "win_rate": round(won / (won + lost) * 100 if (won + lost) else 0, 1),
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(
                (total_profit / total_staked * 100) if total_staked else 0, 1
            ),
            "avg_odds": round(
                sum(b["odds"] for b in self.bets) / total if total else 0, 0
            ),
            "avg_stake": round(total_staked / total if total else 0, 2),
            "current_streak": f"{current_streak} {streak_type}{'s' if current_streak > 1 else ''}",
            "best_day": {"date": best_day[0], "profit": round(best_day[1], 2)},
            "worst_day": {"date": worst_day[0], "profit": round(worst_day[1], 2)},
            "by_bet_type": {
                t: {
                    "record": f"{d['won']}-{d['total'] - d['won']}",
                    "win_rate": round(d["won"] / d["total"] * 100, 1),
                    "profit": round(d["profit"], 2),
                    "roi": round(d["profit"] / d["staked"] * 100 if d["staked"] else 0, 1),
                }
                for t, d in by_type.items()
            },
        }

    def _empty_stats(self) -> dict:
        return {
            "total_bets": 0, "record": "0-0-0", "win_rate": 0,
            "total_staked": 0, "total_profit": 0, "roi": 0,
            "avg_odds": 0, "avg_stake": 0, "current_streak": "N/A",
            "best_day": {"date": "N/A", "profit": 0},
            "worst_day": {"date": "N/A", "profit": 0},
            "by_bet_type": {},
        }

    def format_dashboard(self) -> str:
        """Format performance dashboard for display."""
        s = self.get_stats()

        dash = f"""
╔══════════════════════════════════════════════════╗
║         MLB PREDICTOR PERFORMANCE                ║
╚══════════════════════════════════════════════════╝

  📊 Record:      {s['record']}
  🎯 Win Rate:    {s['win_rate']}%
  💰 Total P/L:   ${s['total_profit']:+,.2f}
  📈 ROI:         {s['roi']:+.1f}%
  🔥 Streak:      {s['current_streak']}

  💵 Total Staked: ${s['total_staked']:,.2f}
  📊 Avg Stake:    ${s['avg_stake']:.2f}
  📊 Avg Odds:     {s['avg_odds']:+.0f}

  🏆 Best Day:     {s['best_day']['date']} (${s['best_day']['profit']:+,.2f})
  📉 Worst Day:    {s['worst_day']['date']} (${s['worst_day']['profit']:+,.2f})
"""
        if s["by_bet_type"]:
            dash += "\n  BY BET TYPE:\n"
            for t, d in s["by_bet_type"].items():
                dash += (
                    f"    {t:<12} {d['record']:<8} "
                    f"Win: {d['win_rate']}%  "
                    f"P/L: ${d['profit']:+,.2f}  "
                    f"ROI: {d['roi']:+.1f}%\n"
                )

        return dash


if __name__ == "__main__":
    builder = ParlayBuilder(storage_dir="/tmp/mlb_bets")

    # Create picks
    picks = [
        builder.create_pick(
            "NYY_BOS_20260223", "moneyline", "NYY ML",
            -145, model_win_prob=0.62,
            home_team="NYY", away_team="BOS",
        ),
        builder.create_pick(
            "LAD_SF_20260223", "moneyline", "LAD ML",
            -180, model_win_prob=0.65,
            home_team="LAD", away_team="SF",
        ),
        builder.create_pick(
            "HOU_TEX_20260223", "total", "Over 8.5",
            -110, model_win_prob=0.56,
            home_team="HOU", away_team="TEX",
        ),
        builder.create_pick(
            "CHC_STL_20260223", "moneyline", "CHC ML",
            +125, model_win_prob=0.48,
            home_team="STL", away_team="CHC",
        ),
    ]

    # Build a parlay
    parlay = builder.build_parlay(picks[:3], stake=25.0)
    print(f"🎰 Parlay: {parlay.combined_odds:+d}")
    print(f"   Win Prob: {parlay.correlation_adjusted_prob * 100:.1f}%")
    print(f"   Payout: ${parlay.potential_payout}")

    # Suggest parlays
    suggestions = builder.suggest_parlays(picks)
    print(f"\n💡 {len(suggestions)} suggested parlays:")
    for s in suggestions:
        legs = [p["selection"] for p in s.picks]
        print(f"  {' + '.join(legs)} → {s.combined_odds:+d} "
              f"(${s.potential_payout:.2f})")

    # Create bet slip
    slip = builder.create_bet_slip(singles=picks[:2], parlays=[parlay])
    print(builder.format_bet_slip(slip))

    # Performance tracker demo
    tracker = PerformanceTracker(storage_dir="/tmp/mlb_perf")
    tracker.record_bet("moneyline", "NYY ML", -145, 50, "won", payout=84.50)
    tracker.record_bet("moneyline", "LAD ML", -180, 50, "won", payout=77.78)
    tracker.record_bet("total", "Over 8.5", -110, 25, "lost")
    tracker.record_bet("moneyline", "CHC ML", +125, 30, "won", payout=67.50)
    tracker.record_bet("parlay", "3-leg parlay", +450, 25, "lost")

    print(tracker.format_dashboard())
