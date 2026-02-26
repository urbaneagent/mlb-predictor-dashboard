"""
MLB Predictor - Daily Picks Dashboard API
Serves today's top predictions, historical performance, ROI tracking,
and personalized bet recommendations based on user preferences.
"""

import json
import time
import uuid
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# Pick Models
# ──────────────────────────────────────────────

class PickType(Enum):
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    FIRST_5 = "first_5"
    PROP = "prop"
    PARLAY = "parlay"


class PickConfidence(Enum):
    LOCK = "lock"        # 70%+ model edge
    STRONG = "strong"    # 55-70% edge
    LEAN = "lean"        # 52-55% edge
    FADE = "fade"        # Contrarian play


@dataclass
class DailyPick:
    """A single daily prediction/pick"""
    pick_id: str
    game_id: str
    pick_type: PickType
    confidence: PickConfidence
    team: str
    opponent: str
    selection: str  # "home", "away", "over", "under"
    line: float = 0.0
    odds: int = -110
    model_probability: float = 0.0
    implied_probability: float = 0.0
    edge: float = 0.0
    recommended_units: float = 1.0  # 1-5 scale
    # Context
    game_time: str = ""
    pitcher_home: str = ""
    pitcher_away: str = ""
    key_factors: List[str] = field(default_factory=list)
    weather_impact: str = ""
    umpire: str = ""
    stadium: str = ""
    # Tracking
    result: Optional[str] = None  # "win", "loss", "push", "pending"
    actual_score: str = ""
    profit_loss: float = 0.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        imp = abs(self.odds) / (abs(self.odds) + 100) if self.odds < 0 else 100 / (self.odds + 100)
        return {
            "pick_id": self.pick_id,
            "game_id": self.game_id,
            "pick_type": self.pick_type.value,
            "confidence": self.confidence.value,
            "confidence_emoji": {"lock": "🔒", "strong": "💪", "lean": "👀", "fade": "🔄"}.get(self.confidence.value, ""),
            "team": self.team,
            "opponent": self.opponent,
            "selection": self.selection,
            "line": self.line,
            "odds": self.odds,
            "odds_display": f"+{self.odds}" if self.odds > 0 else str(self.odds),
            "model_probability": round(self.model_probability * 100, 1),
            "implied_probability": round(imp * 100, 1),
            "edge": round(self.edge, 1),
            "recommended_units": self.recommended_units,
            "game_time": self.game_time,
            "pitcher_home": self.pitcher_home,
            "pitcher_away": self.pitcher_away,
            "key_factors": self.key_factors,
            "weather_impact": self.weather_impact,
            "umpire": self.umpire,
            "stadium": self.stadium,
            "result": self.result or "pending",
            "actual_score": self.actual_score,
            "profit_loss": round(self.profit_loss, 2),
        }


@dataclass
class DailyCard:
    """Full daily picks card"""
    date: str
    picks: List[DailyPick] = field(default_factory=list)
    locks: int = 0
    strong: int = 0
    leans: int = 0
    total_units: float = 0.0
    weather_alerts: List[str] = field(default_factory=list)
    injury_updates: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "date": self.date,
            "picks": [p.to_dict() for p in sorted(self.picks, key=lambda p: -p.edge)],
            "summary": {
                "total_picks": len(self.picks),
                "locks": sum(1 for p in self.picks if p.confidence == PickConfidence.LOCK),
                "strong": sum(1 for p in self.picks if p.confidence == PickConfidence.STRONG),
                "leans": sum(1 for p in self.picks if p.confidence == PickConfidence.LEAN),
                "total_units": round(sum(p.recommended_units for p in self.picks), 1),
                "avg_edge": round(sum(p.edge for p in self.picks) / max(1, len(self.picks)), 1),
                "best_pick": max(self.picks, key=lambda p: p.edge).to_dict() if self.picks else None,
            },
            "weather_alerts": self.weather_alerts,
            "injury_updates": self.injury_updates,
            "notes": self.notes,
        }


# ──────────────────────────────────────────────
# Performance Tracker
# ──────────────────────────────────────────────

@dataclass
class PerformanceRecord:
    date: str
    picks: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    units_wagered: float = 0.0
    units_won: float = 0.0
    profit_loss: float = 0.0
    roi: float = 0.0

    def to_dict(self):
        return asdict(self)


class PerformanceTracker:
    """Track pick performance over time with detailed analytics"""

    def __init__(self):
        self.daily_records: Dict[str, PerformanceRecord] = {}
        self.all_picks: List[DailyPick] = []
        self._bankroll_start = 10000.0
        self._current_bankroll = 10000.0

    def record_pick(self, pick: DailyPick, result: str, actual_score: str = ""):
        """Record the result of a pick"""
        pick.result = result
        pick.actual_score = actual_score

        # Calculate P/L
        if result == "win":
            if pick.odds > 0:
                pick.profit_loss = pick.recommended_units * (pick.odds / 100)
            else:
                pick.profit_loss = pick.recommended_units * (100 / abs(pick.odds))
        elif result == "loss":
            pick.profit_loss = -pick.recommended_units
        else:
            pick.profit_loss = 0

        self._current_bankroll += pick.profit_loss * 100  # Assume $100 units

        self.all_picks.append(pick)
        self._update_daily(pick)

    def _update_daily(self, pick: DailyPick):
        date = datetime.fromtimestamp(pick.created_at).strftime("%Y-%m-%d")
        if date not in self.daily_records:
            self.daily_records[date] = PerformanceRecord(date=date)

        record = self.daily_records[date]
        record.picks += 1
        record.units_wagered += pick.recommended_units

        if pick.result == "win":
            record.wins += 1
            record.units_won += pick.profit_loss
        elif pick.result == "loss":
            record.losses += 1
        elif pick.result == "push":
            record.pushes += 1

        record.profit_loss += pick.profit_loss
        record.roi = round(record.profit_loss / max(0.1, record.units_wagered) * 100, 1)

    def get_overall_stats(self) -> Dict:
        """Get lifetime performance statistics"""
        if not self.all_picks:
            return {"total_picks": 0}

        settled = [p for p in self.all_picks if p.result in ["win", "loss", "push"]]
        wins = sum(1 for p in settled if p.result == "win")
        losses = sum(1 for p in settled if p.result == "loss")
        pushes = sum(1 for p in settled if p.result == "push")
        total_wagered = sum(p.recommended_units for p in settled)
        total_profit = sum(p.profit_loss for p in settled)

        # By pick type
        by_type = {}
        for pick_type in PickType:
            type_picks = [p for p in settled if p.pick_type == pick_type]
            if type_picks:
                type_wins = sum(1 for p in type_picks if p.result == "win")
                type_profit = sum(p.profit_loss for p in type_picks)
                type_wagered = sum(p.recommended_units for p in type_picks)
                by_type[pick_type.value] = {
                    "picks": len(type_picks),
                    "wins": type_wins,
                    "win_rate": round(type_wins / len(type_picks) * 100, 1),
                    "profit": round(type_profit, 2),
                    "roi": round(type_profit / max(0.1, type_wagered) * 100, 1),
                }

        # By confidence
        by_confidence = {}
        for conf in PickConfidence:
            conf_picks = [p for p in settled if p.confidence == conf]
            if conf_picks:
                conf_wins = sum(1 for p in conf_picks if p.result == "win")
                conf_profit = sum(p.profit_loss for p in conf_picks)
                conf_wagered = sum(p.recommended_units for p in conf_picks)
                by_confidence[conf.value] = {
                    "picks": len(conf_picks),
                    "wins": conf_wins,
                    "win_rate": round(conf_wins / len(conf_picks) * 100, 1),
                    "profit": round(conf_profit, 2),
                    "roi": round(conf_profit / max(0.1, conf_wagered) * 100, 1),
                }

        # Streak tracking
        streak = 0
        streak_type = ""
        for pick in reversed(settled):
            if not streak_type:
                streak_type = pick.result
                streak = 1
            elif pick.result == streak_type:
                streak += 1
            else:
                break

        # Best/worst days
        daily_sorted = sorted(self.daily_records.values(), key=lambda r: r.profit_loss, reverse=True)

        return {
            "total_picks": len(settled),
            "record": f"{wins}-{losses}-{pushes}",
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": round(wins / max(1, wins + losses) * 100, 1),
            "units_wagered": round(total_wagered, 1),
            "units_profit": round(total_profit, 2),
            "roi": round(total_profit / max(0.1, total_wagered) * 100, 1),
            "bankroll": round(self._current_bankroll, 2),
            "bankroll_change": round(self._current_bankroll - self._bankroll_start, 2),
            "bankroll_change_pct": round(
                (self._current_bankroll - self._bankroll_start) / self._bankroll_start * 100, 1
            ),
            "current_streak": f"{streak} {'W' if streak_type == 'win' else 'L' if streak_type == 'loss' else 'P'}",
            "by_type": by_type,
            "by_confidence": by_confidence,
            "best_day": daily_sorted[0].to_dict() if daily_sorted else None,
            "worst_day": daily_sorted[-1].to_dict() if daily_sorted else None,
            "avg_daily_profit": round(total_profit / max(1, len(self.daily_records)), 2),
            "days_tracked": len(self.daily_records),
        }

    def get_recent_performance(self, days: int = 7) -> Dict:
        cutoff = time.time() - (days * 86400)
        recent = [p for p in self.all_picks if p.created_at > cutoff and p.result in ["win", "loss", "push"]]

        if not recent:
            return {"period": f"Last {days} days", "picks": 0}

        wins = sum(1 for p in recent if p.result == "win")
        profit = sum(p.profit_loss for p in recent)
        wagered = sum(p.recommended_units for p in recent)

        return {
            "period": f"Last {days} days",
            "picks": len(recent),
            "record": f"{wins}-{sum(1 for p in recent if p.result == 'loss')}",
            "win_rate": round(wins / max(1, len(recent)) * 100, 1),
            "profit": round(profit, 2),
            "roi": round(profit / max(0.1, wagered) * 100, 1),
        }

    def get_leaderboard(self, period: str = "all") -> List[Dict]:
        """Top picks by ROI (could be multi-user in future)"""
        picks = self.all_picks
        if period == "7d":
            cutoff = time.time() - (7 * 86400)
            picks = [p for p in picks if p.created_at > cutoff]
        elif period == "30d":
            cutoff = time.time() - (30 * 86400)
            picks = [p for p in picks if p.created_at > cutoff]

        winners = [p for p in picks if p.result == "win"]
        winners.sort(key=lambda p: -p.profit_loss)

        return [p.to_dict() for p in winners[:20]]


# ──────────────────────────────────────────────
# Dashboard API
# ──────────────────────────────────────────────

class DashboardAPI:
    """Central API for the daily picks dashboard"""

    def __init__(self):
        self.tracker = PerformanceTracker()
        self.daily_cards: Dict[str, DailyCard] = {}
        self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate realistic sample data for demo"""
        games = [
            ("NYY-BOS", "New York Yankees", "Boston Red Sox", "Gerrit Cole", "Chris Sale", "Yankee Stadium", "Angel Hernandez"),
            ("LAD-SFG", "Los Angeles Dodgers", "San Francisco Giants", "Yoshinobu Yamamoto", "Logan Webb", "Dodger Stadium", "Nic Lentz"),
            ("HOU-TEX", "Houston Astros", "Texas Rangers", "Framber Valdez", "Nathan Eovaldi", "Minute Maid Park", "Ron Kulpa"),
            ("ATL-NYM", "Atlanta Braves", "New York Mets", "Spencer Strider", "Kodai Senga", "Truist Park", "CB Bucknor"),
            ("CHC-STL", "Chicago Cubs", "St. Louis Cardinals", "Justin Steele", "Sonny Gray", "Wrigley Field", "James Hoye"),
        ]

        today = datetime.now().strftime("%Y-%m-%d")
        card = DailyCard(date=today)

        picks_data = [
            (PickConfidence.LOCK, PickType.MONEYLINE, "home", -135, 0.62, 3.0,
             ["Cole dominant at home (3.8 ERA K/9 12.1)", "Red Sox .220 vs RHP last 14d", "Wind blowing in at Yankee Stadium"]),
            (PickConfidence.STRONG, PickType.TOTAL, "over", -110, 0.58, 2.0,
             ["Both bullpens 4.5+ ERA last 7d", "Wind blowing out at Wrigley", "Day game = more runs historically"]),
            (PickConfidence.STRONG, PickType.MONEYLINE, "home", -165, 0.63, 2.5,
             ["Yamamoto 2.1 ERA at home", "Giants .198 team avg vs RHP", "Dodgers 8-2 last 10 at home"]),
            (PickConfidence.LEAN, PickType.SPREAD, "away", +110, 0.54, 1.0,
             ["Eovaldi 3.2 road ERA", "Astros fatigue (4th game in row)", "Rangers 6-4 in last 10 road games"]),
            (PickConfidence.LOCK, PickType.FIRST_5, "under", -115, 0.65, 3.0,
             ["Strider: 1.8 F5 ERA this season", "Senga: 2.4 F5 ERA", "Both elite F5 starters", "Umpire widens zone"]),
            (PickConfidence.LEAN, PickType.TOTAL, "under", -108, 0.53, 1.0,
             ["Low wind, cool night", "Both starters ERA under 3.0", "Pitchers' duel expected"]),
        ]

        for i, (conf, ptype, sel, odds, prob, units, factors) in enumerate(picks_data):
            game = games[i % len(games)]
            team = game[1] if sel in ["home", "over", "under"] else game[2]
            opp = game[2] if sel in ["home", "over", "under"] else game[1]
            imp = abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

            pick = DailyPick(
                pick_id=f"pick-{str(uuid.uuid4())[:8]}",
                game_id=game[0],
                pick_type=ptype,
                confidence=conf,
                team=team,
                opponent=opp,
                selection=sel,
                line=8.5 if ptype in [PickType.TOTAL, PickType.FIRST_5] else -1.5 if ptype == PickType.SPREAD else 0,
                odds=odds,
                model_probability=prob,
                implied_probability=imp,
                edge=round((prob - imp) * 100, 1),
                recommended_units=units,
                game_time=f"2026-02-23T{19 + i}:05:00",
                pitcher_home=game[3],
                pitcher_away=game[4],
                key_factors=factors,
                stadium=game[5],
                umpire=game[6],
            )
            card.picks.append(pick)

        card.weather_alerts = [
            "🌬️ NYY-BOS: Wind 15mph blowing IN — suppresses HR",
            "☀️ CHC-STL: Day game at Wrigley, wind 12mph OUT — boosts offense",
        ]
        card.injury_updates = [
            "⚠️ BOS: Rafael Devers (hamstring) — Game-time decision",
            "✅ LAD: Mookie Betts — Back in lineup after rest day",
        ]

        self.daily_cards[today] = card

        # Simulate historical results
        results = ["win", "win", "loss", "win", "win", "loss"]
        scores = ["5-2", "9-7", "3-4", "6-1", "2-3", "4-5"]
        for pick, result, score in zip(card.picks, results, scores):
            self.tracker.record_pick(pick, result, score)

    def get_todays_card(self) -> Dict:
        today = datetime.now().strftime("%Y-%m-%d")
        card = self.daily_cards.get(today)
        if not card:
            return {"date": today, "picks": [], "message": "No picks for today yet"}
        return card.to_dict()

    def get_performance(self) -> Dict:
        return self.tracker.get_overall_stats()

    def get_recent_performance(self, days: int = 7) -> Dict:
        return self.tracker.get_recent_performance(days)

    def get_leaderboard(self, period: str = "all") -> List[Dict]:
        return self.tracker.get_leaderboard(period)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("MLB Predictor - Daily Picks Dashboard")
    print("=" * 60)

    api = DashboardAPI()

    # Today's card
    card = api.get_todays_card()
    print(f"\n📅 Today's Card ({card['date']}):")
    summary = card.get("summary", {})
    print(f"  Total picks: {summary.get('total_picks', 0)}")
    print(f"  🔒 Locks: {summary.get('locks', 0)} | 💪 Strong: {summary.get('strong', 0)} | 👀 Leans: {summary.get('leans', 0)}")
    print(f"  Total units: {summary.get('total_units', 0)} | Avg edge: {summary.get('avg_edge', 0)}%")

    print(f"\n  🎯 Picks:")
    for pick in card.get("picks", []):
        emoji = pick["confidence_emoji"]
        type_label = pick["pick_type"].upper()
        print(f"\n  {emoji} [{type_label}] {pick['team']} {pick['selection'].upper()} {pick['odds_display']}")
        print(f"     Edge: {pick['edge']}% | Model: {pick['model_probability']}% | Units: {pick['recommended_units']}")
        if pick.get("pitcher_home") and pick.get("pitcher_away"):
            print(f"     {pick['pitcher_home']} vs {pick['pitcher_away']}")
        for factor in pick["key_factors"][:2]:
            print(f"     • {factor}")
        print(f"     Result: {pick['result'].upper()} {'✅' if pick['result'] == 'win' else '❌' if pick['result'] == 'loss' else '⏳'}")

    # Weather & injuries
    if card.get("weather_alerts"):
        print(f"\n  🌤️ Weather Alerts:")
        for alert in card["weather_alerts"]:
            print(f"    {alert}")

    if card.get("injury_updates"):
        print(f"\n  🏥 Injury Updates:")
        for update in card["injury_updates"]:
            print(f"    {update}")

    # Performance
    print(f"\n📊 Performance:")
    perf = api.get_performance()
    print(f"  Record: {perf.get('record', '0-0-0')} ({perf.get('win_rate', 0)}%)")
    print(f"  Units P/L: {perf.get('units_profit', 0):+.2f} | ROI: {perf.get('roi', 0)}%")
    print(f"  Bankroll: ${perf.get('bankroll', 0):,.2f} ({perf.get('bankroll_change_pct', 0):+.1f}%)")
    print(f"  Streak: {perf.get('current_streak', '')}")

    if perf.get("by_confidence"):
        print(f"\n  By Confidence:")
        for conf, stats in perf["by_confidence"].items():
            print(f"    {conf:8s}: {stats['picks']} picks | {stats['win_rate']}% win | ROI: {stats['roi']}%")

    if perf.get("by_type"):
        print(f"\n  By Type:")
        for ptype, stats in perf["by_type"].items():
            print(f"    {ptype:10s}: {stats['picks']} picks | {stats['win_rate']}% win | ROI: {stats['roi']}%")

    print(f"\n✅ Daily Picks Dashboard ready!")
    print("  • Daily picks card with 4 confidence levels")
    print("  • Model probability + edge calculation")
    print("  • Key factors per pick (pitcher, weather, umpire, stadium)")
    print("  • Weather alerts & injury updates")
    print("  • Full performance tracking (by type, confidence)")
    print("  • Bankroll tracking with ROI")
    print("  • Streak tracking")
    print("  • Historical leaderboard")


if __name__ == "__main__":
    demo()
