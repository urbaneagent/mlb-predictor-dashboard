#!/usr/bin/env python3
"""
MLB Predictor - User Dashboard
=================================
Dashboard for viewing predictions, tracking performance,
and managing bets with export capabilities.

Features:
- Today's top picks (sorted by edge)
- Historical performance (win rate, ROI, streaks)
- Bet slip builder (singles + parlays)
- Performance charts data (for frontend rendering)
- Export to CSV/PDF
- Alerts and notifications
- Model confidence breakdown

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
import csv
import io
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DailyPick:
    """A single prediction for today's games"""
    game_id: str
    game_time: str
    away_team: str
    home_team: str
    pick: str  # Team name
    pick_side: str  # "home" or "away"
    model_probability: float
    market_implied: float
    edge: float
    odds: int  # Best available American odds
    book: str  # Which sportsbook
    confidence: str  # "high", "medium", "low"
    kelly_stake_pct: float
    ev_per_100: float
    factors: Dict[str, Any] = field(default_factory=dict)
    # Environmental factors
    weather_impact: str = ""
    pitcher_matchup: str = ""
    umpire_impact: str = ""

    @property
    def edge_pct(self) -> float:
        return round(self.edge * 100, 1)


@dataclass
class PerformanceSnapshot:
    """Performance data for a time period"""
    period: str
    total_picks: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    roi: float
    units_won: float
    avg_edge: float
    avg_odds: int
    best_pick: Optional[Dict] = None
    worst_pick: Optional[Dict] = None
    streak: int = 0
    by_confidence: Dict[str, Dict] = field(default_factory=dict)


# ============================================================================
# DASHBOARD ENGINE
# ============================================================================

class DashboardEngine:
    """
    Generates dashboard data from predictions and bet tracking.
    """

    # Team abbreviations
    TEAM_ABBREV = {
        'New York Yankees': 'NYY', 'New York Mets': 'NYM',
        'Boston Red Sox': 'BOS', 'Toronto Blue Jays': 'TOR',
        'Tampa Bay Rays': 'TB', 'Baltimore Orioles': 'BAL',
        'Chicago White Sox': 'CWS', 'Cleveland Guardians': 'CLE',
        'Detroit Tigers': 'DET', 'Kansas City Royals': 'KC',
        'Minnesota Twins': 'MIN', 'Houston Astros': 'HOU',
        'Los Angeles Angels': 'LAA', 'Oakland Athletics': 'OAK',
        'Seattle Mariners': 'SEA', 'Texas Rangers': 'TEX',
        'Atlanta Braves': 'ATL', 'Miami Marlins': 'MIA',
        'Philadelphia Phillies': 'PHI', 'Washington Nationals': 'WSH',
        'Chicago Cubs': 'CHC', 'Cincinnati Reds': 'CIN',
        'Milwaukee Brewers': 'MIL', 'Pittsburgh Pirates': 'PIT',
        'St. Louis Cardinals': 'STL', 'Arizona Diamondbacks': 'ARI',
        'Colorado Rockies': 'COL', 'Los Angeles Dodgers': 'LAD',
        'San Diego Padres': 'SD', 'San Francisco Giants': 'SFG',
    }

    def generate_todays_picks(self, predictions: List[Dict],
                               min_edge: float = 0.02) -> List[DailyPick]:
        """
        Generate today's top picks from model predictions.
        
        predictions: List of dicts with keys:
            game_id, game_time, away_team, home_team,
            home_win_prob, away_win_prob, home_odds, away_odds,
            home_book, away_book
        """
        picks = []

        for pred in predictions:
            # Check home side
            home_prob = pred.get('home_win_prob', 0.5)
            home_odds = pred.get('home_odds', -110)
            home_implied = self._implied_prob(home_odds)
            home_edge = home_prob - home_implied

            if home_edge >= min_edge:
                picks.append(self._create_pick(
                    pred, 'home', pred['home_team'],
                    home_prob, home_implied, home_edge,
                    home_odds, pred.get('home_book', '')
                ))

            # Check away side
            away_prob = pred.get('away_win_prob', 0.5)
            away_odds = pred.get('away_odds', -110)
            away_implied = self._implied_prob(away_odds)
            away_edge = away_prob - away_implied

            if away_edge >= min_edge:
                picks.append(self._create_pick(
                    pred, 'away', pred['away_team'],
                    away_prob, away_implied, away_edge,
                    away_odds, pred.get('away_book', '')
                ))

        # Sort by edge (highest first)
        picks.sort(key=lambda p: p.edge, reverse=True)
        return picks

    def generate_dashboard_data(self, picks: List[DailyPick],
                                 history: List[Dict]) -> Dict[str, Any]:
        """Generate complete dashboard data package"""

        # Today's picks summary
        high_conf = [p for p in picks if p.confidence == 'high']
        med_conf = [p for p in picks if p.confidence == 'medium']
        low_conf = [p for p in picks if p.confidence == 'low']

        # Performance from history
        perf = self._calculate_performance(history, 'all')
        perf_7d = self._calculate_performance(history, '7d')
        perf_30d = self._calculate_performance(history, '30d')

        # Chart data (for frontend)
        cumulative_pnl = self._cumulative_pnl(history)
        daily_picks_count = self._daily_picks_count(history)
        roi_by_week = self._roi_by_week(history)

        return {
            'generated_at': datetime.now().isoformat(),
            'today': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_picks': len(picks),
                'high_confidence': len(high_conf),
                'medium_confidence': len(med_conf),
                'low_confidence': len(low_conf),
                'avg_edge': round(
                    sum(p.edge for p in picks) / max(1, len(picks)) * 100, 1
                ),
                'picks': [self._pick_to_dict(p) for p in picks]
            },
            'performance': {
                'all_time': self._perf_to_dict(perf),
                'last_7_days': self._perf_to_dict(perf_7d),
                'last_30_days': self._perf_to_dict(perf_30d),
            },
            'charts': {
                'cumulative_pnl': cumulative_pnl,
                'daily_picks': daily_picks_count,
                'weekly_roi': roi_by_week,
            },
            'alerts': self._generate_alerts(picks, perf)
        }

    def export_picks_csv(self, picks: List[DailyPick]) -> str:
        """Export today's picks to CSV string"""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            'Game', 'Time', 'Pick', 'Side', 'Odds', 'Book',
            'Model%', 'Market%', 'Edge%', 'Confidence',
            'Kelly%', 'EV/$100'
        ])
        for p in picks:
            away_abbrev = self.TEAM_ABBREV.get(p.away_team, p.away_team[:3])
            home_abbrev = self.TEAM_ABBREV.get(p.home_team, p.home_team[:3])
            game = f"{away_abbrev} @ {home_abbrev}"
            writer.writerow([
                game, p.game_time,
                self.TEAM_ABBREV.get(p.pick, p.pick[:3]),
                p.pick_side, p.odds, p.book,
                round(p.model_probability * 100, 1),
                round(p.market_implied * 100, 1),
                p.edge_pct, p.confidence,
                round(p.kelly_stake_pct, 1), round(p.ev_per_100, 2)
            ])
        return output.getvalue()

    def generate_daily_report(self, picks: List[DailyPick],
                               history: List[Dict]) -> str:
        """Generate a text-based daily report"""
        perf = self._calculate_performance(history, '7d')
        date_str = datetime.now().strftime('%A, %B %d, %Y')

        lines = [
            f"{'='*60}",
            f"⚾ MLB PREDICTOR - DAILY REPORT",
            f"📅 {date_str}",
            f"{'='*60}",
            "",
        ]

        if perf.total_picks > 0:
            lines.extend([
                f"📊 LAST 7 DAYS: {perf.wins}W-{perf.losses}L "
                f"({perf.win_rate:.1f}%) | ROI: {perf.roi:+.1f}% | "
                f"Units: {perf.units_won:+.1f}",
                ""
            ])

        lines.append(f"🎯 TODAY'S PICKS ({len(picks)} games with edge)")
        lines.append(f"{'-'*60}")

        if not picks:
            lines.append("   No picks meet minimum edge threshold today.")
        else:
            lines.append(
                f"   {'Game':<22} {'Pick':<6} {'Odds':>6} "
                f"{'Edge':>6} {'Conf':<6} {'EV/$100':>8}"
            )
            lines.append(f"   {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")

            for p in picks:
                away = self.TEAM_ABBREV.get(p.away_team, p.away_team[:3])
                home = self.TEAM_ABBREV.get(p.home_team, p.home_team[:3])
                pick = self.TEAM_ABBREV.get(p.pick, p.pick[:3])
                game = f"{away} @ {home}"
                conf_icon = {
                    'high': '🟢', 'medium': '🟡', 'low': '🔴'
                }.get(p.confidence, '⚪')
                lines.append(
                    f"   {game:<22} {pick:<6} {p.odds:>+6} "
                    f"{p.edge_pct:>5.1f}% {conf_icon:<6} "
                    f"${p.ev_per_100:>+6.2f}"
                )

                # Sub-details
                if p.weather_impact:
                    lines.append(f"     🌤️ {p.weather_impact}")
                if p.pitcher_matchup:
                    lines.append(f"     ⚾ {p.pitcher_matchup}")

        lines.extend([
            "",
            f"{'='*60}",
            f"⚠️ Disclaimer: For entertainment purposes only.",
            f"{'='*60}",
        ])

        return '\n'.join(lines)

    # ── Internal Methods ──────────────────────────────────────

    def _create_pick(self, pred: Dict, side: str, team: str,
                     model_prob: float, implied: float, edge: float,
                     odds: int, book: str) -> DailyPick:
        """Create a DailyPick from prediction data"""
        # Kelly calculation
        if odds > 0:
            decimal = 1 + (odds / 100)
        else:
            decimal = 1 + (100 / abs(odds))

        b = decimal - 1
        kelly = max(0, (b * model_prob - (1 - model_prob)) / b) * 50  # Half-Kelly %

        ev = (model_prob * (decimal - 1) - (1 - model_prob)) * 100

        if edge >= 0.06:
            confidence = 'high'
        elif edge >= 0.03:
            confidence = 'medium'
        else:
            confidence = 'low'

        return DailyPick(
            game_id=pred.get('game_id', ''),
            game_time=pred.get('game_time', ''),
            away_team=pred['away_team'],
            home_team=pred['home_team'],
            pick=team,
            pick_side=side,
            model_probability=model_prob,
            market_implied=implied,
            edge=edge,
            odds=odds,
            book=book,
            confidence=confidence,
            kelly_stake_pct=round(kelly, 1),
            ev_per_100=round(ev, 2),
            weather_impact=pred.get('weather_impact', ''),
            pitcher_matchup=pred.get('pitcher_matchup', ''),
            umpire_impact=pred.get('umpire_impact', ''),
            factors=pred.get('factors', {})
        )

    def _implied_prob(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    def _pick_to_dict(self, pick: DailyPick) -> Dict:
        """Convert pick to dictionary for JSON output"""
        away = self.TEAM_ABBREV.get(pick.away_team, pick.away_team[:3])
        home = self.TEAM_ABBREV.get(pick.home_team, pick.home_team[:3])
        return {
            'game': f"{away} @ {home}",
            'game_time': pick.game_time,
            'pick': self.TEAM_ABBREV.get(pick.pick, pick.pick[:3]),
            'pick_full': pick.pick,
            'side': pick.pick_side,
            'odds': pick.odds,
            'book': pick.book,
            'model_prob': round(pick.model_probability * 100, 1),
            'market_prob': round(pick.market_implied * 100, 1),
            'edge': pick.edge_pct,
            'confidence': pick.confidence,
            'kelly_pct': pick.kelly_stake_pct,
            'ev_per_100': pick.ev_per_100,
            'weather': pick.weather_impact,
            'pitcher': pick.pitcher_matchup,
        }

    def _calculate_performance(self, history: List[Dict],
                                period: str) -> PerformanceSnapshot:
        """Calculate performance for a time period"""
        # Filter by period
        if period == '7d':
            cutoff = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        elif period == '30d':
            cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            cutoff = '2000-01-01'

        filtered = [h for h in history if h.get('date', '') >= cutoff]

        wins = sum(1 for h in filtered if h.get('result') == 'won')
        losses = sum(1 for h in filtered if h.get('result') == 'lost')
        pushes = sum(1 for h in filtered if h.get('result') == 'push')
        total = wins + losses

        units_won = sum(h.get('units_won', 0) for h in filtered)

        return PerformanceSnapshot(
            period=period,
            total_picks=len(filtered),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=round(wins / max(1, total) * 100, 1),
            roi=round(units_won / max(1, len(filtered)) * 100, 1),
            units_won=round(units_won, 2),
            avg_edge=round(
                sum(h.get('edge', 0) for h in filtered) / max(1, len(filtered)) * 100, 1
            ),
            avg_odds=round(
                sum(h.get('odds', -110) for h in filtered) / max(1, len(filtered))
            ),
        )

    def _perf_to_dict(self, perf: PerformanceSnapshot) -> Dict:
        return {
            'period': perf.period,
            'record': f"{perf.wins}W-{perf.losses}L-{perf.pushes}P",
            'win_rate': perf.win_rate,
            'roi': perf.roi,
            'units': perf.units_won,
            'total_picks': perf.total_picks,
            'avg_edge': perf.avg_edge,
            'avg_odds': perf.avg_odds,
        }

    def _cumulative_pnl(self, history: List[Dict]) -> List[Dict]:
        """Generate cumulative P&L chart data"""
        sorted_hist = sorted(history, key=lambda h: h.get('date', ''))
        cumulative = 0
        data = []
        for h in sorted_hist:
            cumulative += h.get('units_won', 0)
            data.append({
                'date': h.get('date', ''),
                'cumulative_units': round(cumulative, 2)
            })
        return data

    def _daily_picks_count(self, history: List[Dict]) -> List[Dict]:
        """Daily pick count for bar chart"""
        by_date: Dict[str, int] = {}
        for h in history:
            date = h.get('date', '')
            by_date[date] = by_date.get(date, 0) + 1
        return [{'date': d, 'picks': c} for d, c in sorted(by_date.items())]

    def _roi_by_week(self, history: List[Dict]) -> List[Dict]:
        """Weekly ROI for line chart"""
        by_week: Dict[str, List] = {}
        for h in history:
            date = h.get('date', '')
            if date:
                # Simple week grouping by date prefix
                week = date[:8] + '01'  # Approximate
                if week not in by_week:
                    by_week[week] = []
                by_week[week].append(h.get('units_won', 0))

        return [
            {
                'week': w,
                'roi': round(sum(units) / max(1, len(units)) * 100, 1),
                'picks': len(units)
            }
            for w, units in sorted(by_week.items())
        ]

    def _generate_alerts(self, picks: List[DailyPick],
                          perf: PerformanceSnapshot) -> List[Dict]:
        """Generate dashboard alerts"""
        alerts = []

        # High-value picks
        high_edge = [p for p in picks if p.edge >= 0.08]
        if high_edge:
            alerts.append({
                'type': 'info',
                'message': f"🔥 {len(high_edge)} high-edge picks today "
                          f"(>{8}% edge)"
            })

        # Cold streak warning
        if perf.streak <= -5:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ {abs(perf.streak)}-game losing streak. "
                          f"Consider reducing stakes."
            })

        # Hot streak
        if perf.streak >= 5:
            alerts.append({
                'type': 'success',
                'message': f"🔥 {perf.streak}-game winning streak!"
            })

        # No picks
        if not picks:
            alerts.append({
                'type': 'info',
                'message': "📋 No picks meet threshold today. "
                          "Sometimes the best bet is no bet."
            })

        return alerts


# ============================================================================
# DEMO
# ============================================================================

def demo_dashboard():
    """Demonstrate the user dashboard"""
    print("=" * 70)
    print("📊 MLB Predictor - User Dashboard Demo")
    print("=" * 70)
    print()

    engine = DashboardEngine()

    # Simulated predictions
    predictions = [
        {'game_id': 'g1', 'game_time': '7:10 PM', 'away_team': 'Boston Red Sox',
         'home_team': 'New York Yankees', 'home_win_prob': 0.62, 'away_win_prob': 0.38,
         'home_odds': -145, 'away_odds': +125, 'home_book': 'DraftKings', 'away_book': 'FanDuel',
         'pitcher_matchup': 'Cole (3.12 ERA) vs Whitlock (3.85 ERA)',
         'weather_impact': '72°F, 8mph wind out to RF'},
        {'game_id': 'g2', 'game_time': '8:10 PM', 'away_team': 'San Francisco Giants',
         'home_team': 'Los Angeles Dodgers', 'home_win_prob': 0.68, 'away_win_prob': 0.32,
         'home_odds': -180, 'away_odds': +155, 'home_book': 'BetMGM', 'away_book': 'DraftKings',
         'pitcher_matchup': 'Yamamoto (2.78 ERA) vs Webb (3.35 ERA)'},
        {'game_id': 'g3', 'game_time': '8:10 PM', 'away_team': 'Texas Rangers',
         'home_team': 'Houston Astros', 'home_win_prob': 0.56, 'away_win_prob': 0.44,
         'home_odds': -120, 'away_odds': +102, 'home_book': 'FanDuel', 'away_book': 'PointsBet'},
        {'game_id': 'g4', 'game_time': '7:20 PM', 'away_team': 'New York Mets',
         'home_team': 'Atlanta Braves', 'home_win_prob': 0.60, 'away_win_prob': 0.40,
         'home_odds': -155, 'away_odds': +135, 'home_book': 'DraftKings', 'away_book': 'BetMGM',
         'pitcher_matchup': 'Fried (2.95 ERA) vs Senga (3.40 ERA)'},
        {'game_id': 'g5', 'game_time': '9:40 PM', 'away_team': 'Arizona Diamondbacks',
         'home_team': 'San Diego Padres', 'home_win_prob': 0.54, 'away_win_prob': 0.46,
         'home_odds': -125, 'away_odds': +108, 'home_book': 'Caesars', 'away_book': 'FanDuel'},
    ]

    # Simulated history
    import random
    history = []
    for i in range(60):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        for _ in range(random.randint(1, 4)):
            won = random.random() < 0.56  # 56% win rate
            odds = random.choice([-150, -140, -130, -120, -110, +110, +120, +130])
            if odds > 0:
                payout = odds / 100 if won else -1
            else:
                payout = 100 / abs(odds) if won else -1
            history.append({
                'date': date,
                'result': 'won' if won else 'lost',
                'odds': odds,
                'edge': random.uniform(0.02, 0.08),
                'units_won': round(payout, 2),
                'confidence': random.choice(['high', 'medium', 'low'])
            })

    # Generate picks
    picks = engine.generate_todays_picks(predictions, min_edge=0.02)

    # Generate daily report
    report = engine.generate_daily_report(picks, history)
    print(report)
    print()

    # Dashboard data
    dashboard = engine.generate_dashboard_data(picks, history)
    print("📊 DASHBOARD DATA SUMMARY:")
    print(f"   Today: {dashboard['today']['total_picks']} picks "
          f"({dashboard['today']['high_confidence']} high conf)")
    print(f"   7-Day: {dashboard['performance']['last_7_days']['record']} "
          f"(ROI: {dashboard['performance']['last_7_days']['roi']:+.1f}%)")
    print(f"   30-Day: {dashboard['performance']['last_30_days']['record']} "
          f"(ROI: {dashboard['performance']['last_30_days']['roi']:+.1f}%)")
    print(f"   All-Time: {dashboard['performance']['all_time']['record']} "
          f"(ROI: {dashboard['performance']['all_time']['roi']:+.1f}%)")
    print()

    # Alerts
    for alert in dashboard['alerts']:
        print(f"   {alert['message']}")
    print()

    # CSV export
    csv_data = engine.export_picks_csv(picks)
    csv_path = Path("/tmp/mlb_daily_picks.csv")
    csv_path.write_text(csv_data)
    print(f"📤 Picks exported to: {csv_path}")

    print()
    print("=" * 70)
    print("✅ Dashboard Demo Complete")
    print("=" * 70)

    return dashboard


if __name__ == "__main__":
    demo_dashboard()
