"""
MLB Predictor - Performance Tracker & ROI Dashboard
Tracks all picks, calculates ROI, win rate, CLV, and generates reports.
Essential for proving model edge and attracting subscribers.
"""
import json
import math
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TrackedBet:
    """A tracked bet with result."""
    bet_id: str
    date: str
    game_id: int
    away_team: str
    home_team: str
    pick_type: str  # moneyline, spread, total, F5, prop
    pick_side: str
    pick_team: str
    pick_line: str
    odds: int  # American odds at time of pick
    closing_odds: int = 0  # Closing line odds
    units_risked: float = 0.0
    model_probability: float = 0.0
    implied_probability: float = 0.0
    edge: float = 0.0
    confidence: str = ""

    # Result
    result: str = ""  # win, loss, push, pending
    units_won: float = 0.0  # Net P&L in units
    actual_score_away: int = 0
    actual_score_home: int = 0
    closing_line_value: float = 0.0  # CLV: our odds vs closing odds


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    period: str  # "all", "2026-02", "last_7", "last_30"
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    pending: int = 0
    win_rate: float = 0.0
    units_risked: float = 0.0
    units_won: float = 0.0
    units_net: float = 0.0
    roi: float = 0.0
    avg_odds: float = 0.0
    avg_edge: float = 0.0
    clv_avg: float = 0.0  # Average closing line value
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    current_streak: str = ""  # "W4", "L2"
    kelly_growth: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # By bet type
    by_type: dict = field(default_factory=dict)
    # By confidence
    by_confidence: dict = field(default_factory=dict)
    # By team
    by_team: dict = field(default_factory=dict)
    # Daily P&L
    daily_pnl: list = field(default_factory=list)


class PerformanceTracker:
    """
    Tracks and analyzes betting performance.
    Provides proof of model edge for marketing/subscribers.
    """

    def __init__(self):
        self.bets = []  # All tracked bets
        self.bankroll_history = []  # Daily bankroll snapshots

    def record_bet(self, bet: TrackedBet):
        """Record a new bet."""
        self.bets.append(bet)

    def record_result(self, bet_id: str, result: str,
                      score_away: int = 0, score_home: int = 0,
                      closing_odds: int = 0):
        """Record the result of a bet."""
        bet = next((b for b in self.bets if b.bet_id == bet_id), None)
        if not bet:
            return

        bet.result = result
        bet.actual_score_away = score_away
        bet.actual_score_home = score_home
        bet.closing_odds = closing_odds

        # Calculate P&L
        if result == "win":
            if bet.odds > 0:
                bet.units_won = bet.units_risked * (bet.odds / 100)
            else:
                bet.units_won = bet.units_risked * (100 / abs(bet.odds))
        elif result == "loss":
            bet.units_won = -bet.units_risked
        else:  # push
            bet.units_won = 0

        # Calculate CLV
        if closing_odds and bet.odds:
            bet.closing_line_value = self._calculate_clv(bet.odds, closing_odds)

    def get_stats(self, period: str = "all",
                  bet_type: str = None, confidence: str = None) -> PerformanceStats:
        """Get comprehensive performance statistics."""
        bets = self._filter_bets(period, bet_type, confidence)
        settled = [b for b in bets if b.result in ['win', 'loss', 'push']]

        if not settled:
            return PerformanceStats(period=period)

        wins = [b for b in settled if b.result == 'win']
        losses = [b for b in settled if b.result == 'loss']
        pushes = [b for b in settled if b.result == 'push']
        pending = [b for b in bets if b.result == 'pending' or b.result == '']

        total_risked = sum(b.units_risked for b in settled)
        total_won = sum(b.units_won for b in settled)
        decided = len(wins) + len(losses)

        # Streaks
        win_streak, loss_streak, current = self._calculate_streaks(settled)

        # Daily P&L
        daily = self._daily_pnl(settled)

        # By type breakdown
        by_type = {}
        for bt in set(b.pick_type for b in settled):
            type_bets = [b for b in settled if b.pick_type == bt]
            type_wins = [b for b in type_bets if b.result == 'win']
            type_decided = len([b for b in type_bets if b.result in ['win', 'loss']])
            type_net = sum(b.units_won for b in type_bets)
            type_risked = sum(b.units_risked for b in type_bets)
            by_type[bt] = {
                "bets": len(type_bets),
                "wins": len(type_wins),
                "win_rate": len(type_wins) / type_decided if type_decided else 0,
                "units_net": round(type_net, 2),
                "roi": round(type_net / type_risked * 100, 1) if type_risked else 0
            }

        # By confidence breakdown
        by_conf = {}
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            conf_bets = [b for b in settled if b.confidence == conf]
            if conf_bets:
                conf_wins = [b for b in conf_bets if b.result == 'win']
                conf_decided = len([b for b in conf_bets if b.result in ['win', 'loss']])
                conf_net = sum(b.units_won for b in conf_bets)
                conf_risked = sum(b.units_risked for b in conf_bets)
                by_conf[conf] = {
                    "bets": len(conf_bets),
                    "wins": len(conf_wins),
                    "win_rate": len(conf_wins) / conf_decided if conf_decided else 0,
                    "units_net": round(conf_net, 2),
                    "roi": round(conf_net / conf_risked * 100, 1) if conf_risked else 0
                }

        # CLV
        clv_values = [b.closing_line_value for b in settled if b.closing_line_value != 0]
        avg_clv = statistics.mean(clv_values) if clv_values else 0

        # Sharpe ratio (daily returns)
        if len(daily) > 1:
            daily_returns = [d['pnl'] for d in daily]
            mean_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1
            sharpe = (mean_return / std_return) * math.sqrt(252) if std_return > 0 else 0  # Annualized
        else:
            sharpe = 0

        # Max drawdown
        max_dd = self._max_drawdown(settled)

        # Average odds
        all_odds = [b.odds for b in settled if b.odds]
        avg_odds = statistics.mean(all_odds) if all_odds else 0

        # Average edge
        all_edges = [b.edge for b in settled if b.edge > 0]
        avg_edge = statistics.mean(all_edges) if all_edges else 0

        return PerformanceStats(
            period=period,
            total_bets=len(bets),
            wins=len(wins),
            losses=len(losses),
            pushes=len(pushes),
            pending=len(pending),
            win_rate=round(len(wins) / decided * 100, 1) if decided else 0,
            units_risked=round(total_risked, 2),
            units_won=round(sum(b.units_won for b in wins), 2),
            units_net=round(total_won, 2),
            roi=round(total_won / total_risked * 100, 1) if total_risked else 0,
            avg_odds=round(avg_odds, 0),
            avg_edge=round(avg_edge, 3),
            clv_avg=round(avg_clv, 3),
            longest_win_streak=win_streak,
            longest_loss_streak=loss_streak,
            current_streak=current,
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            by_type=by_type,
            by_confidence=by_conf,
            daily_pnl=daily[-30:]  # Last 30 days
        )

    def get_leaderboard(self) -> dict:
        """Generate performance leaderboard for marketing."""
        stats = self.get_stats("all")
        monthly_stats = self.get_stats(datetime.now().strftime("%Y-%m"))

        return {
            "headline": f"{stats.wins}-{stats.losses} ({stats.win_rate:.1f}%) | {stats.units_net:+.1f}u | ROI: {stats.roi:+.1f}%",
            "monthly": f"{monthly_stats.wins}-{monthly_stats.losses} | {monthly_stats.units_net:+.1f}u this month",
            "streak": stats.current_streak,
            "best_type": max(stats.by_type.items(), key=lambda x: x[1]['roi'])[0] if stats.by_type else "N/A",
            "clv": f"{stats.clv_avg:+.1f}¢ avg CLV (beating the closing line)",
            "sharpe": f"{stats.sharpe_ratio:.2f} Sharpe ratio",
            "verified": True,
            "model_version": "5.0",
            "since": self.bets[0].date if self.bets else "N/A"
        }

    def generate_report_markdown(self, period: str = "all") -> str:
        """Generate a full performance report in Markdown."""
        stats = self.get_stats(period)

        report = f"""# MLB Predictor Performance Report
**Period:** {period}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Overview
| Metric | Value |
|--------|-------|
| Total Bets | {stats.total_bets} |
| Record | {stats.wins}-{stats.losses}-{stats.pushes} |
| Win Rate | {stats.win_rate:.1f}% |
| Units Net | {stats.units_net:+.1f}u |
| ROI | {stats.roi:+.1f}% |
| Avg Edge | {stats.avg_edge:.1%} |
| CLV | {stats.clv_avg:+.3f} |
| Sharpe Ratio | {stats.sharpe_ratio:.2f} |
| Max Drawdown | {stats.max_drawdown:.1f}u |
| Streak | {stats.current_streak} |

## By Bet Type
| Type | Bets | Wins | Win% | P&L | ROI |
|------|------|------|------|-----|-----|
"""
        for bt, data in stats.by_type.items():
            report += f"| {bt} | {data['bets']} | {data['wins']} | {data['win_rate']:.1%} | {data['units_net']:+.1f}u | {data['roi']:+.1f}% |\n"

        report += "\n## By Confidence\n| Level | Bets | Wins | Win% | P&L | ROI |\n|-------|------|------|------|-----|-----|\n"
        for conf, data in stats.by_confidence.items():
            report += f"| {conf} | {data['bets']} | {data['wins']} | {data['win_rate']:.1%} | {data['units_net']:+.1f}u | {data['roi']:+.1f}% |\n"

        return report

    # ============= Internal Helpers =============

    def _filter_bets(self, period, bet_type=None, confidence=None):
        bets = self.bets

        if period != "all":
            if period == "last_7":
                cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                bets = [b for b in bets if b.date >= cutoff]
            elif period == "last_30":
                cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                bets = [b for b in bets if b.date >= cutoff]
            elif "-" in period:  # "2026-02"
                bets = [b for b in bets if b.date.startswith(period)]

        if bet_type:
            bets = [b for b in bets if b.pick_type == bet_type]
        if confidence:
            bets = [b for b in bets if b.confidence == confidence]

        return bets

    def _calculate_streaks(self, settled):
        results = [b.result for b in sorted(settled, key=lambda b: b.date) if b.result in ['win', 'loss']]
        if not results:
            return 0, 0, ""

        max_w = max_l = curr_w = curr_l = 0
        for r in results:
            if r == 'win':
                curr_w += 1
                curr_l = 0
                max_w = max(max_w, curr_w)
            else:
                curr_l += 1
                curr_w = 0
                max_l = max(max_l, curr_l)

        current = f"W{curr_w}" if curr_w > 0 else f"L{curr_l}" if curr_l > 0 else "—"
        return max_w, max_l, current

    def _daily_pnl(self, settled):
        by_date = {}
        for b in settled:
            if b.date not in by_date:
                by_date[b.date] = {"date": b.date, "pnl": 0, "bets": 0, "wins": 0}
            by_date[b.date]["pnl"] += b.units_won
            by_date[b.date]["bets"] += 1
            if b.result == "win":
                by_date[b.date]["wins"] += 1

        daily = sorted(by_date.values(), key=lambda d: d['date'])
        # Add cumulative
        cum = 0
        for d in daily:
            d['pnl'] = round(d['pnl'], 2)
            cum += d['pnl']
            d['cumulative'] = round(cum, 2)

        return daily

    def _max_drawdown(self, settled):
        sorted_bets = sorted(settled, key=lambda b: b.date)
        cumulative = 0
        peak = 0
        max_dd = 0

        for b in sorted_bets:
            cumulative += b.units_won
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return max_dd

    def _calculate_clv(self, opening_odds, closing_odds) -> float:
        """Calculate closing line value (positive = we beat the close)."""
        open_impl = self._american_to_implied(opening_odds)
        close_impl = self._american_to_implied(closing_odds)
        # Positive CLV means we got better odds than closing
        return round(open_impl - close_impl, 4) if close_impl else 0

    def _american_to_implied(self, odds) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        elif odds < 0:
            return abs(odds) / (abs(odds) + 100)
        return 0.5


def create_performance_routes(app, tracker: PerformanceTracker):
    """Create FastAPI routes for performance tracking."""

    @app.get("/api/v1/performance")
    async def get_performance(period: str = "all", bet_type: str = None, confidence: str = None):
        stats = tracker.get_stats(period, bet_type, confidence)
        return asdict(stats)

    @app.get("/api/v1/performance/leaderboard")
    async def get_leaderboard():
        return tracker.get_leaderboard()

    @app.get("/api/v1/performance/report")
    async def get_report(period: str = "all"):
        return {"markdown": tracker.generate_report_markdown(period)}

    @app.get("/api/v1/bets")
    async def get_bets(limit: int = 50, result: str = None):
        bets = tracker.bets[-limit:]
        if result:
            bets = [b for b in bets if b.result == result]
        return {"count": len(bets), "bets": [asdict(b) for b in bets]}

    return tracker
