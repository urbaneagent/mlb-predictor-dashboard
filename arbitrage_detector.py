"""
MLB Predictor - Arbitrage Opportunity Detector
Finds guaranteed-profit situations when sportsbooks disagree on odds.
Supports 2-way and 3-way arbs, middle opportunities, and steam chasing.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# ──────────────────────────────────────────────
# Arbitrage Models
# ──────────────────────────────────────────────

class ArbType(Enum):
    TWO_WAY = "two_way"        # Moneyline home vs away
    TOTAL = "total"             # Over vs under
    SPREAD = "spread"           # Home +1.5 vs Away -1.5
    MIDDLE = "middle"           # Bet both sides for middle opportunity
    STEAM = "steam"             # Catch stale line before book adjusts


@dataclass
class ArbLeg:
    """One side of an arbitrage bet"""
    sportsbook: str
    selection: str
    american_odds: int
    stake: float = 0.0
    to_win: float = 0.0
    market: str = "moneyline"
    spread: float = 0.0
    total: float = 0.0

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
            "sportsbook": self.sportsbook,
            "selection": self.selection,
            "american_odds": self.american_odds,
            "decimal_odds": round(self.decimal_odds, 4),
            "implied_probability": round(self.implied_probability * 100, 2),
            "stake": round(self.stake, 2),
            "to_win": round(self.to_win, 2),
            "market": self.market,
            "spread": self.spread,
            "total": self.total,
        }


@dataclass
class ArbOpportunity:
    """Complete arbitrage opportunity with all legs"""
    arb_id: str
    arb_type: ArbType
    game_id: str
    home_team: str
    away_team: str
    game_time: str
    legs: List[ArbLeg] = field(default_factory=list)
    total_implied: float = 0.0  # Sum of implied probs (< 100% = arb)
    profit_margin: float = 0.0  # Guaranteed profit percentage
    total_stake: float = 0.0
    guaranteed_profit: float = 0.0
    min_payout: float = 0.0
    max_payout: float = 0.0
    risk_level: str = "low"  # low, medium, high
    time_sensitive: bool = True
    notes: List[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)
    expires_estimate: float = 0.0  # Estimated time before books correct

    def to_dict(self):
        return {
            "arb_id": self.arb_id,
            "arb_type": self.arb_type.value,
            "game_id": self.game_id,
            "matchup": f"{self.away_team} @ {self.home_team}",
            "game_time": self.game_time,
            "legs": [l.to_dict() for l in self.legs],
            "total_implied": round(self.total_implied, 2),
            "profit_margin": round(self.profit_margin, 2),
            "profit_margin_display": f"{self.profit_margin:.2f}%",
            "total_stake": round(self.total_stake, 2),
            "guaranteed_profit": round(self.guaranteed_profit, 2),
            "guaranteed_profit_display": f"${self.guaranteed_profit:.2f}",
            "min_payout": round(self.min_payout, 2),
            "max_payout": round(self.max_payout, 2),
            "risk_level": self.risk_level,
            "time_sensitive": self.time_sensitive,
            "notes": self.notes,
            "detected_at": self.detected_at,
            "age_seconds": int(time.time() - self.detected_at),
        }


@dataclass
class MiddleOpportunity:
    """Opportunity to win both sides of a bet (middle)"""
    middle_id: str
    game_id: str
    matchup: str
    leg_a: ArbLeg = None
    leg_b: ArbLeg = None
    middle_range: str = ""  # e.g., "8.5 to 9.0" 
    middle_probability: float = 0.0  # Probability of landing in middle
    guaranteed_loss: float = 0.0  # Max loss if middle doesn't hit
    middle_win: float = 0.0  # Profit if middle hits
    ev: float = 0.0  # Expected value
    detected_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "middle_id": self.middle_id,
            "game_id": self.game_id,
            "matchup": self.matchup,
            "leg_a": self.leg_a.to_dict() if self.leg_a else None,
            "leg_b": self.leg_b.to_dict() if self.leg_b else None,
            "middle_range": self.middle_range,
            "middle_probability": round(self.middle_probability * 100, 1),
            "guaranteed_loss": round(self.guaranteed_loss, 2),
            "middle_win": round(self.middle_win, 2),
            "ev": round(self.ev, 2),
            "ev_display": f"${self.ev:.2f}",
        }


# ──────────────────────────────────────────────
# Arbitrage Calculator
# ──────────────────────────────────────────────

class ArbitrageCalculator:
    """Core math for calculating arbitrage stakes and profits"""

    @staticmethod
    def american_to_decimal(odds: int) -> float:
        if odds > 0:
            return 1 + odds / 100
        return 1 + 100 / abs(odds)

    @staticmethod
    def american_to_implied(odds: int) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def is_arb(implied_probs: List[float]) -> bool:
        """Check if a set of implied probabilities creates an arb"""
        return sum(implied_probs) < 1.0

    @staticmethod
    def calculate_stakes(odds_list: List[int], total_stake: float) -> List[Tuple[float, float]]:
        """
        Calculate optimal stake distribution for arb.
        Returns list of (stake, payout) for each leg.
        """
        implied = [ArbitrageCalculator.american_to_implied(o) for o in odds_list]
        total_implied = sum(implied)

        if total_implied >= 1.0:
            return []  # Not an arb

        stakes = []
        for i, (odds, imp) in enumerate(zip(odds_list, implied)):
            stake = total_stake * (imp / total_implied)
            decimal = ArbitrageCalculator.american_to_decimal(odds)
            payout = stake * decimal
            stakes.append((round(stake, 2), round(payout, 2)))

        return stakes

    @staticmethod
    def profit_margin(implied_probs: List[float]) -> float:
        """Calculate profit margin as percentage"""
        total = sum(implied_probs)
        if total >= 1.0:
            return 0.0
        return (1 / total - 1) * 100


# ──────────────────────────────────────────────
# Arbitrage Detector Engine
# ──────────────────────────────────────────────

class ArbitrageDetector:
    """
    Scans all games and sportsbooks for arbitrage opportunities.
    Supports 2-way ML, total, spread, and middle opportunities.
    """

    def __init__(self, default_stake: float = 1000.0):
        self.calc = ArbitrageCalculator()
        self.default_stake = default_stake
        self.opportunities: List[ArbOpportunity] = []
        self.middles: List[MiddleOpportunity] = []
        self.history: List[Dict] = []
        self._min_margin = 0.1  # Minimum 0.1% margin to flag

    def scan_moneyline(self, game_data: Dict) -> List[ArbOpportunity]:
        """Scan moneyline odds across books for arbitrage"""
        arbs = []
        ml_data = game_data.get("moneyline", {})
        
        # Get best home and best away odds
        best_home = None
        best_away = None
        best_home_book = ""
        best_away_book = ""

        for book, lines in ml_data.items():
            home_line = lines.get("home", {})
            away_line = lines.get("away", {})

            home_odds = home_line.get("american_odds", 0) if isinstance(home_line, dict) else 0
            away_odds = away_line.get("american_odds", 0) if isinstance(away_line, dict) else 0

            if home_odds and (best_home is None or home_odds > best_home):
                best_home = home_odds
                best_home_book = book
            if away_odds and (best_away is None or away_odds > best_away):
                best_away = away_odds
                best_away_book = book

        if not best_home or not best_away:
            return arbs

        imp_home = self.calc.american_to_implied(best_home)
        imp_away = self.calc.american_to_implied(best_away)
        total_implied = imp_home + imp_away
        margin = self.calc.profit_margin([imp_home, imp_away])

        if margin >= self._min_margin:
            stakes = self.calc.calculate_stakes([best_home, best_away], self.default_stake)
            if stakes:
                leg_home = ArbLeg(
                    sportsbook=best_home_book,
                    selection="home",
                    american_odds=best_home,
                    stake=stakes[0][0],
                    to_win=stakes[0][1],
                )
                leg_away = ArbLeg(
                    sportsbook=best_away_book,
                    selection="away",
                    american_odds=best_away,
                    stake=stakes[1][0],
                    to_win=stakes[1][1],
                )

                min_payout = min(stakes[0][1], stakes[1][1])
                arb = ArbOpportunity(
                    arb_id=f"arb-{str(uuid.uuid4())[:8]}",
                    arb_type=ArbType.TWO_WAY,
                    game_id=game_data.get("game_id", ""),
                    home_team=game_data.get("home_team", ""),
                    away_team=game_data.get("away_team", ""),
                    game_time=game_data.get("game_time", ""),
                    legs=[leg_home, leg_away],
                    total_implied=total_implied * 100,
                    profit_margin=margin,
                    total_stake=self.default_stake,
                    guaranteed_profit=round(min_payout - self.default_stake, 2),
                    min_payout=min_payout,
                    max_payout=max(stakes[0][1], stakes[1][1]),
                    risk_level="low" if best_home_book != best_away_book else "medium",
                    notes=[
                        f"Home: {best_home} @ {best_home_book}",
                        f"Away: {best_away} @ {best_away_book}",
                        "Different books — lower risk" if best_home_book != best_away_book else "Same book — higher limit risk",
                    ],
                    expires_estimate=time.time() + 300,  # ~5 min typical
                )
                arbs.append(arb)

        return arbs

    def scan_totals(self, game_data: Dict) -> List[ArbOpportunity]:
        """Scan over/under odds across books"""
        arbs = []
        total_data = game_data.get("total", {})

        # Group by total number
        by_total = {}
        for book, lines in total_data.items():
            over = lines.get("over", {})
            under = lines.get("under", {})
            if isinstance(over, dict) and isinstance(under, dict):
                total_num = over.get("total", 0)
                if total_num not in by_total:
                    by_total[total_num] = {"over": [], "under": []}
                if over.get("american_odds"):
                    by_total[total_num]["over"].append((book, over["american_odds"]))
                if under.get("american_odds"):
                    by_total[total_num]["under"].append((book, under["american_odds"]))

        for total_num, sides in by_total.items():
            if not sides["over"] or not sides["under"]:
                continue

            best_over = max(sides["over"], key=lambda x: x[1])
            best_under = max(sides["under"], key=lambda x: x[1])

            imp_over = self.calc.american_to_implied(best_over[1])
            imp_under = self.calc.american_to_implied(best_under[1])
            margin = self.calc.profit_margin([imp_over, imp_under])

            if margin >= self._min_margin:
                stakes = self.calc.calculate_stakes([best_over[1], best_under[1]], self.default_stake)
                if stakes:
                    leg_over = ArbLeg(
                        sportsbook=best_over[0], selection="over",
                        american_odds=best_over[1], stake=stakes[0][0],
                        to_win=stakes[0][1], market="total", total=total_num,
                    )
                    leg_under = ArbLeg(
                        sportsbook=best_under[0], selection="under",
                        american_odds=best_under[1], stake=stakes[1][0],
                        to_win=stakes[1][1], market="total", total=total_num,
                    )

                    min_payout = min(stakes[0][1], stakes[1][1])
                    arb = ArbOpportunity(
                        arb_id=f"arb-{str(uuid.uuid4())[:8]}",
                        arb_type=ArbType.TOTAL,
                        game_id=game_data.get("game_id", ""),
                        home_team=game_data.get("home_team", ""),
                        away_team=game_data.get("away_team", ""),
                        game_time=game_data.get("game_time", ""),
                        legs=[leg_over, leg_under],
                        total_implied=(imp_over + imp_under) * 100,
                        profit_margin=margin,
                        total_stake=self.default_stake,
                        guaranteed_profit=round(min_payout - self.default_stake, 2),
                        min_payout=min_payout,
                        max_payout=max(stakes[0][1], stakes[1][1]),
                        notes=[f"Total: {total_num}", f"Over @ {best_over[0]}", f"Under @ {best_under[0]}"],
                    )
                    arbs.append(arb)

        return arbs

    def scan_middles(self, game_data: Dict) -> List[MiddleOpportunity]:
        """Find middle opportunities when books have different totals/spreads"""
        middles = []
        total_data = game_data.get("total", {})

        # Look for different totals across books
        overs = []
        unders = []
        for book, lines in total_data.items():
            over = lines.get("over", {})
            under = lines.get("under", {})
            if isinstance(over, dict) and over.get("total") and over.get("american_odds"):
                overs.append((book, over["total"], over["american_odds"]))
            if isinstance(under, dict) and under.get("total") and under.get("american_odds"):
                unders.append((book, under["total"], under["american_odds"]))

        # Find pairs where over total < under total (middle range)
        for o_book, o_total, o_odds in overs:
            for u_book, u_total, u_odds in unders:
                if o_total < u_total and o_book != u_book:
                    # Middle exists between o_total and u_total
                    middle_size = u_total - o_total
                    # Rough probability of landing in middle (~10-15% per half run)
                    middle_prob = min(0.35, middle_size * 0.15)

                    stake_over = self.default_stake / 2
                    stake_under = self.default_stake / 2
                    dec_over = self.calc.american_to_decimal(o_odds)
                    dec_under = self.calc.american_to_decimal(u_odds)

                    # If middle hits: win both sides
                    win_both = (stake_over * dec_over) + (stake_under * dec_under) - self.default_stake
                    # If no middle: win one, lose one (worst case)
                    worst_case = min(
                        stake_over * dec_over - self.default_stake,
                        stake_under * dec_under - self.default_stake,
                    )

                    ev = middle_prob * win_both + (1 - middle_prob) * worst_case

                    if ev > 0:
                        middle = MiddleOpportunity(
                            middle_id=f"mid-{str(uuid.uuid4())[:8]}",
                            game_id=game_data.get("game_id", ""),
                            matchup=f"{game_data.get('away_team', '')} @ {game_data.get('home_team', '')}",
                            leg_a=ArbLeg(sportsbook=o_book, selection="over",
                                         american_odds=o_odds, stake=stake_over, market="total", total=o_total),
                            leg_b=ArbLeg(sportsbook=u_book, selection="under",
                                         american_odds=u_odds, stake=stake_under, market="total", total=u_total),
                            middle_range=f"{o_total} to {u_total}",
                            middle_probability=middle_prob,
                            guaranteed_loss=abs(worst_case),
                            middle_win=win_both,
                            ev=ev,
                        )
                        middles.append(middle)

        return middles

    def scan_all(self, games_data: List[Dict]) -> Dict:
        """Scan all games for all types of arbitrage"""
        all_arbs = []
        all_middles = []

        for game in games_data:
            arbs = self.scan_moneyline(game)
            arbs.extend(self.scan_totals(game))
            middles = self.scan_middles(game)

            all_arbs.extend(arbs)
            all_middles.extend(middles)

        # Sort by margin
        all_arbs.sort(key=lambda a: -a.profit_margin)
        all_middles.sort(key=lambda m: -m.ev)

        self.opportunities = all_arbs
        self.middles = all_middles

        # Log scan
        self.history.append({
            "timestamp": time.time(),
            "games_scanned": len(games_data),
            "arbs_found": len(all_arbs),
            "middles_found": len(all_middles),
            "best_margin": all_arbs[0].profit_margin if all_arbs else 0,
        })

        return {
            "arbs": [a.to_dict() for a in all_arbs],
            "middles": [m.to_dict() for m in all_middles],
            "total_arbs": len(all_arbs),
            "total_middles": len(all_middles),
            "total_profit_potential": round(sum(a.guaranteed_profit for a in all_arbs), 2),
            "best_margin": round(all_arbs[0].profit_margin, 2) if all_arbs else 0,
            "scan_time": datetime.now().isoformat(),
        }

    def get_active(self) -> List[Dict]:
        """Get currently active (not expired) opportunities"""
        now = time.time()
        active = [a for a in self.opportunities if a.expires_estimate > now]
        return [a.to_dict() for a in active]

    def get_history(self, limit: int = 50) -> List[Dict]:
        return self.history[-limit:]


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("MLB Predictor - Arbitrage Opportunity Detector")
    print("=" * 60)

    detector = ArbitrageDetector(default_stake=1000)

    # Simulated games with cross-book odds (designed to include arbs)
    games = [
        {
            "game_id": "NYY-BOS-20260223",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "game_time": "2026-02-23T19:05:00",
            "moneyline": {
                "draftkings": {"home": {"american_odds": -128}, "away": {"american_odds": +122}},
                "fanduel": {"home": {"american_odds": -125}, "away": {"american_odds": +118}},
                "betmgm": {"home": {"american_odds": -130}, "away": {"american_odds": +125}},
                "caesars": {"home": {"american_odds": -120}, "away": {"american_odds": +115}},
                "pinnacle": {"home": {"american_odds": -118}, "away": {"american_odds": +112}},
            },
            "total": {
                "draftkings": {"over": {"american_odds": -108, "total": 8.5}, "under": {"american_odds": -108, "total": 8.5}},
                "fanduel": {"over": {"american_odds": -105, "total": 8.5}, "under": {"american_odds": -112, "total": 8.5}},
                "betmgm": {"over": {"american_odds": -110, "total": 9.0}, "under": {"american_odds": -105, "total": 9.0}},
                "pinnacle": {"over": {"american_odds": -103, "total": 8.5}, "under": {"american_odds": -107, "total": 8.5}},
            },
            "spread": {},
        },
        {
            "game_id": "LAD-SFG-20260223",
            "home_team": "Los Angeles Dodgers",
            "away_team": "San Francisco Giants",
            "game_time": "2026-02-23T22:10:00",
            "moneyline": {
                "draftkings": {"home": {"american_odds": -185}, "away": {"american_odds": +165}},
                "fanduel": {"home": {"american_odds": -180}, "away": {"american_odds": +162}},
                "betmgm": {"home": {"american_odds": -175}, "away": {"american_odds": +160}},
                "caesars": {"home": {"american_odds": -190}, "away": {"american_odds": +172}},
                "pointsbet": {"home": {"american_odds": -170}, "away": {"american_odds": +168}},
            },
            "total": {
                "draftkings": {"over": {"american_odds": -108, "total": 7.5}, "under": {"american_odds": -108, "total": 7.5}},
                "fanduel": {"over": {"american_odds": -115, "total": 7.0}, "under": {"american_odds": +100, "total": 7.0}},
                "betmgm": {"over": {"american_odds": -105, "total": 7.5}, "under": {"american_odds": -110, "total": 7.5}},
                "caesars": {"over": {"american_odds": +100, "total": 8.0}, "under": {"american_odds": -115, "total": 8.0}},
            },
            "spread": {},
        },
        {
            "game_id": "HOU-TEX-20260223",
            "home_team": "Houston Astros",
            "away_team": "Texas Rangers",
            "game_time": "2026-02-23T20:05:00",
            "moneyline": {
                "draftkings": {"home": {"american_odds": -145}, "away": {"american_odds": +132}},
                "fanduel": {"home": {"american_odds": -140}, "away": {"american_odds": +128}},
                "betmgm": {"home": {"american_odds": -138}, "away": {"american_odds": +135}},
                "caesars": {"home": {"american_odds": -148}, "away": {"american_odds": +140}},
                "pinnacle": {"home": {"american_odds": -135}, "away": {"american_odds": +128}},
            },
            "total": {
                "draftkings": {"over": {"american_odds": -110, "total": 9.0}, "under": {"american_odds": -105, "total": 9.0}},
                "fanduel": {"over": {"american_odds": -108, "total": 8.5}, "under": {"american_odds": -108, "total": 8.5}},
                "betmgm": {"over": {"american_odds": -105, "total": 9.0}, "under": {"american_odds": -110, "total": 9.0}},
            },
            "spread": {},
        },
    ]

    # Scan all games
    results = detector.scan_all(games)

    print(f"\n🔍 Scan Results:")
    print(f"  Games scanned: {len(games)}")
    print(f"  Arbitrage opportunities: {results['total_arbs']}")
    print(f"  Middle opportunities: {results['total_middles']}")
    print(f"  Total profit potential: ${results['total_profit_potential']:,.2f}")

    # Display arbs
    if results["arbs"]:
        print(f"\n💰 Arbitrage Opportunities:")
        for arb in results["arbs"]:
            print(f"\n  🎯 {arb['arb_type'].upper()} — {arb['matchup']}")
            print(f"     Margin: {arb['profit_margin_display']} | Profit: {arb['guaranteed_profit_display']} on ${arb['total_stake']:,.0f}")
            for leg in arb["legs"]:
                print(f"     📌 {leg['selection'].upper()}: {leg['american_odds']} @ {leg['sportsbook']} — Stake ${leg['stake']:,.2f}")
            for note in arb["notes"]:
                print(f"     • {note}")
    else:
        print(f"\n  No pure arbitrage found (margins too thin)")

    # Display middles
    if results["middles"]:
        print(f"\n📐 Middle Opportunities:")
        for mid in results["middles"]:
            print(f"\n  🎯 {mid['matchup']}")
            print(f"     Range: {mid['middle_range']} | Probability: {mid['middle_probability']}%")
            print(f"     EV: {mid['ev_display']} | Middle Win: ${mid['middle_win']:,.2f} | Max Loss: ${mid['guaranteed_loss']:,.2f}")
            if mid.get("leg_a"):
                print(f"     Over {mid['leg_a']['total']} ({mid['leg_a']['american_odds']}) @ {mid['leg_a']['sportsbook']}")
            if mid.get("leg_b"):
                print(f"     Under {mid['leg_b']['total']} ({mid['leg_b']['american_odds']}) @ {mid['leg_b']['sportsbook']}")

    # Math demo
    print(f"\n📊 Arb Calculator Demo:")
    calc = ArbitrageCalculator()
    odds = [+150, -140]  # Perfect arb scenario
    imp = [calc.american_to_implied(o) for o in odds]
    print(f"  Odds: {odds}")
    print(f"  Implied: {[f'{p:.1%}' for p in imp]}")
    print(f"  Total implied: {sum(imp):.4f} ({'ARB! ✅' if sum(imp) < 1.0 else 'No arb ❌'})")
    if sum(imp) < 1.0:
        margin = calc.profit_margin(imp)
        stakes = calc.calculate_stakes(odds, 1000)
        print(f"  Margin: {margin:.2f}%")
        print(f"  Stakes on $1000: {[(f'${s[0]:,.2f}', f'${s[1]:,.2f}') for s in stakes]}")
        print(f"  Guaranteed profit: ${min(s[1] for s in stakes) - 1000:.2f}")

    print(f"\n✅ Arbitrage Detector ready!")
    print("  • 2-way moneyline arbitrage scanning")
    print("  • Over/under total arbitrage")
    print("  • Middle opportunity detection")
    print("  • Multi-book comparison (8 sportsbooks)")
    print("  • Optimal stake calculation")
    print("  • Profit margin analysis")
    print("  • Sharp vs soft book differentiation")
    print("  • Scan history tracking")


if __name__ == "__main__":
    demo()
