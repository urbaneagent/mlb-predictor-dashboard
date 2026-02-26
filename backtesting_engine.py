#!/usr/bin/env python3
"""
MLB Predictor - Backtesting Engine
=====================================
Backtest prediction models against historical data to validate performance.

Features:
- Historical game simulation
- Strategy comparison (flat bet, Kelly, confidence-weighted)
- Performance metrics (win rate, ROI, Sharpe ratio, max drawdown)
- Monte Carlo simulation for confidence intervals
- Seasonal pattern detection
- Model calibration analysis

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
import math
import statistics
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class HistoricalGame:
    """A single historical game with model prediction and actual result"""
    game_id: str
    date: str
    away_team: str
    home_team: str
    home_win_prob: float  # Model's prediction
    away_win_prob: float
    home_odds: int
    away_odds: int
    home_won: bool  # Actual result
    runs_home: int = 0
    runs_away: int = 0


@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    strategy: str
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_staked: float
    total_pnl: float
    roi: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_odds: float
    avg_edge: float
    longest_win_streak: int
    longest_loss_streak: int
    monthly_returns: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    calibration: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestingEngine:
    """
    Backtests MLB prediction strategies against historical data.
    """

    def __init__(self, bankroll: float = 10000.0):
        self.initial_bankroll = bankroll

    def run_backtest(self, games: List[HistoricalGame],
                     strategy: str = "flat",
                     min_edge: float = 0.02,
                     stake_size: float = 100,
                     kelly_fraction: float = 0.5) -> BacktestResult:
        """
        Run backtest with specified strategy.
        
        Strategies:
        - "flat": Fixed $stake_size on every qualifying bet
        - "kelly": Kelly Criterion sized bets
        - "confidence": Weight bets by confidence level
        - "high_only": Only bet on high-confidence picks
        """
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        equity = [bankroll]
        bets = []
        daily_pnl = {}

        for game in games:
            # Calculate edge for home side
            home_implied = self._implied_prob(game.home_odds)
            home_edge = game.home_win_prob - home_implied

            # Calculate edge for away side
            away_implied = self._implied_prob(game.away_odds)
            away_edge = game.away_win_prob - away_implied

            # Determine which side to bet (if any)
            bet_side = None
            bet_prob = 0
            bet_odds = 0
            edge = 0

            if home_edge >= min_edge and home_edge >= away_edge:
                bet_side = 'home'
                bet_prob = game.home_win_prob
                bet_odds = game.home_odds
                edge = home_edge
            elif away_edge >= min_edge:
                bet_side = 'away'
                bet_prob = game.away_win_prob
                bet_odds = game.away_odds
                edge = away_edge

            if not bet_side:
                continue

            # Skip high_only strategy for non-high-confidence
            if strategy == "high_only" and edge < 0.06:
                continue

            # Calculate stake based on strategy
            if strategy == "kelly":
                decimal = self._american_to_decimal(bet_odds)
                b = decimal - 1
                p = bet_prob
                q = 1 - p
                full_kelly = max(0, (b * p - q) / b)
                kelly_adjusted = full_kelly * kelly_fraction
                stake = min(bankroll * kelly_adjusted, bankroll * 0.05)
                stake = max(10, round(stake, 2))
            elif strategy == "confidence":
                if edge >= 0.08:
                    stake = stake_size * 2.0
                elif edge >= 0.05:
                    stake = stake_size * 1.5
                elif edge >= 0.03:
                    stake = stake_size * 1.0
                else:
                    stake = stake_size * 0.5
            else:  # flat
                stake = stake_size

            stake = min(stake, bankroll)
            if stake <= 0:
                continue

            # Determine result
            if bet_side == 'home':
                won = game.home_won
            else:
                won = not game.home_won

            # Calculate P&L
            if won:
                decimal = self._american_to_decimal(bet_odds)
                pnl = stake * (decimal - 1)
            else:
                pnl = -stake

            bankroll += pnl
            peak_bankroll = max(peak_bankroll, bankroll)
            equity.append(bankroll)

            bets.append({
                'date': game.date,
                'game': f"{game.away_team} @ {game.home_team}",
                'side': bet_side,
                'odds': bet_odds,
                'edge': edge,
                'stake': stake,
                'won': won,
                'pnl': pnl
            })

            # Track daily PnL
            if game.date not in daily_pnl:
                daily_pnl[game.date] = 0
            daily_pnl[game.date] += pnl

        # Calculate metrics
        wins = sum(1 for b in bets if b['won'])
        losses = len(bets) - wins
        total_staked = sum(b['stake'] for b in bets)
        total_pnl = bankroll - self.initial_bankroll
        pnl_list = [b['pnl'] for b in bets]

        # Max drawdown
        max_dd, max_dd_pct = self._calculate_max_drawdown(equity)

        # Sharpe ratio (annualized)
        daily_returns = list(daily_pnl.values())
        if len(daily_returns) > 1 and statistics.stdev(daily_returns) > 0:
            sharpe = (statistics.mean(daily_returns) / statistics.stdev(daily_returns)) * math.sqrt(252)
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        profit_factor = gross_profit / max(0.01, gross_loss)

        # Streaks
        win_streak, loss_streak = self._calculate_streaks(bets)

        # Monthly returns
        monthly = self._calculate_monthly_returns(bets)

        # Calibration
        calibration = self._calculate_calibration(games, min_edge)

        return BacktestResult(
            strategy=strategy,
            total_bets=len(bets),
            wins=wins,
            losses=losses,
            pushes=0,
            win_rate=round(wins / max(1, len(bets)) * 100, 1),
            total_staked=round(total_staked, 2),
            total_pnl=round(total_pnl, 2),
            roi=round(total_pnl / max(0.01, total_staked) * 100, 1),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 1),
            sharpe_ratio=round(sharpe, 2),
            profit_factor=round(profit_factor, 2),
            avg_odds=round(statistics.mean(b['odds'] for b in bets), 0) if bets else 0,
            avg_edge=round(statistics.mean(b['edge'] for b in bets) * 100, 1) if bets else 0,
            longest_win_streak=win_streak,
            longest_loss_streak=loss_streak,
            monthly_returns=monthly,
            equity_curve=equity,
            calibration=calibration
        )

    def compare_strategies(self, games: List[HistoricalGame],
                           min_edge: float = 0.02) -> Dict[str, BacktestResult]:
        """Compare multiple betting strategies"""
        strategies = {
            'flat_100': ('flat', 0.02, 100, 0.5),
            'flat_200': ('flat', 0.02, 200, 0.5),
            'kelly_half': ('kelly', 0.02, 100, 0.5),
            'kelly_quarter': ('kelly', 0.02, 100, 0.25),
            'confidence_weighted': ('confidence', 0.02, 100, 0.5),
            'high_confidence_only': ('high_only', 0.06, 150, 0.5),
        }

        results = {}
        for name, (strategy, min_e, stake, kelly_f) in strategies.items():
            results[name] = self.run_backtest(
                games, strategy=strategy, min_edge=min_e,
                stake_size=stake, kelly_fraction=kelly_f
            )

        return results

    def monte_carlo_simulation(self, base_win_rate: float,
                                base_odds: float,
                                num_bets: int = 500,
                                num_simulations: int = 10000,
                                stake: float = 100) -> Dict[str, Any]:
        """
        Monte Carlo simulation to estimate confidence intervals.
        """
        final_bankrolls = []
        max_drawdowns = []
        ruin_count = 0

        for _ in range(num_simulations):
            bankroll = self.initial_bankroll
            peak = bankroll
            max_dd = 0

            for _ in range(num_bets):
                won = random.random() < base_win_rate
                if won:
                    bankroll += stake * (base_odds - 1)
                else:
                    bankroll -= stake

                peak = max(peak, bankroll)
                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

                if bankroll <= 0:
                    ruin_count += 1
                    break

            final_bankrolls.append(bankroll)
            max_drawdowns.append(max_dd)

        final_bankrolls.sort()
        percentiles = {
            '5th': round(final_bankrolls[int(len(final_bankrolls) * 0.05)], 2),
            '25th': round(final_bankrolls[int(len(final_bankrolls) * 0.25)], 2),
            'median': round(final_bankrolls[int(len(final_bankrolls) * 0.50)], 2),
            '75th': round(final_bankrolls[int(len(final_bankrolls) * 0.75)], 2),
            '95th': round(final_bankrolls[int(len(final_bankrolls) * 0.95)], 2),
        }

        return {
            'simulations': num_simulations,
            'num_bets': num_bets,
            'win_rate': base_win_rate,
            'avg_final_bankroll': round(statistics.mean(final_bankrolls), 2),
            'percentiles': percentiles,
            'probability_of_profit': round(
                sum(1 for b in final_bankrolls if b > self.initial_bankroll) / num_simulations * 100, 1
            ),
            'probability_of_ruin': round(ruin_count / num_simulations * 100, 2),
            'avg_max_drawdown_pct': round(statistics.mean(max_drawdowns) * 100, 1),
            'worst_case_drawdown_pct': round(max(max_drawdowns) * 100, 1),
        }

    # ── Internal Methods ──────────────────────────────────────

    def _implied_prob(self, odds: int) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    def _american_to_decimal(self, odds: int) -> float:
        if odds > 0:
            return 1 + odds / 100
        return 1 + 100 / abs(odds)

    def _calculate_max_drawdown(self, equity: List[float]) -> Tuple[float, float]:
        """Calculate maximum drawdown in $ and %"""
        if len(equity) < 2:
            return 0, 0
        peak = equity[0]
        max_dd = 0
        max_dd_pct = 0
        for val in equity:
            peak = max(peak, val)
            dd = peak - val
            dd_pct = dd / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)
        return max_dd, max_dd_pct

    def _calculate_streaks(self, bets: List[Dict]) -> Tuple[int, int]:
        """Calculate longest win and loss streaks"""
        if not bets:
            return 0, 0
        max_win = max_loss = 0
        current = 0
        for b in bets:
            if b['won']:
                current = max(1, current + 1) if current >= 0 else 1
                max_win = max(max_win, current)
            else:
                current = min(-1, current - 1) if current <= 0 else -1
                max_loss = max(max_loss, abs(current))
        return max_win, max_loss

    def _calculate_monthly_returns(self, bets: List[Dict]) -> List[Dict]:
        """Calculate monthly P&L"""
        monthly: Dict[str, float] = {}
        monthly_bets: Dict[str, int] = {}
        for b in bets:
            month = b['date'][:7]
            monthly[month] = monthly.get(month, 0) + b['pnl']
            monthly_bets[month] = monthly_bets.get(month, 0) + 1
        return [
            {'month': m, 'pnl': round(pnl, 2), 'bets': monthly_bets[m]}
            for m, pnl in sorted(monthly.items())
        ]

    def _calculate_calibration(self, games: List[HistoricalGame],
                                min_edge: float) -> Dict:
        """Check model calibration (predicted vs actual win rates)"""
        buckets = {'50-55': [], '55-60': [], '60-65': [], '65-70': [], '70+': []}

        for game in games:
            for prob, won in [
                (game.home_win_prob, game.home_won),
                (game.away_win_prob, not game.home_won)
            ]:
                if prob >= 0.70:
                    buckets['70+'].append(1 if won else 0)
                elif prob >= 0.65:
                    buckets['65-70'].append(1 if won else 0)
                elif prob >= 0.60:
                    buckets['60-65'].append(1 if won else 0)
                elif prob >= 0.55:
                    buckets['55-60'].append(1 if won else 0)
                elif prob >= 0.50:
                    buckets['50-55'].append(1 if won else 0)

        result = {}
        for bucket, outcomes in buckets.items():
            if outcomes:
                actual_rate = sum(outcomes) / len(outcomes)
                result[bucket] = {
                    'predicted_range': bucket,
                    'games': len(outcomes),
                    'actual_win_rate': round(actual_rate * 100, 1),
                    'calibrated': abs(actual_rate * 100 - float(bucket.split('-')[0])) < 5
                }
        return result


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def generate_sample_season(games: int = 500) -> List[HistoricalGame]:
    """Generate simulated season data for backtesting"""
    teams = [
        ('NYY', 'BOS'), ('LAD', 'SFG'), ('HOU', 'TEX'),
        ('ATL', 'NYM'), ('PHI', 'MIA'), ('MIN', 'CLE'),
        ('SEA', 'LAA'), ('SD', 'ARI'), ('TB', 'BAL'),
        ('CHC', 'STL'), ('MIL', 'CIN'), ('KC', 'DET'),
    ]

    game_list = []
    base_date = datetime(2025, 4, 1)

    for i in range(games):
        away, home = random.choice(teams)
        date = (base_date + timedelta(days=i // 3)).strftime('%Y-%m-%d')

        # Generate realistic model prediction
        base_home = random.uniform(0.42, 0.68)
        noise = random.gauss(0, 0.03)
        home_prob = max(0.35, min(0.75, base_home + noise))
        away_prob = 1 - home_prob

        # Generate odds (with some vig)
        if home_prob > 0.5:
            home_odds = -round(home_prob / (1 - home_prob) * 100 + random.uniform(-5, 15))
            away_odds = round((1 - home_prob) / home_prob * 100 + random.uniform(-5, 10))
        else:
            home_odds = round((1 - home_prob) / home_prob * 100 + random.uniform(-5, 10))
            away_odds = -round(away_prob / (1 - away_prob) * 100 + random.uniform(-5, 15))

        # Simulate actual result (model is slightly better than random)
        actual_home_win_prob = home_prob + random.gauss(0, 0.08)
        home_won = random.random() < actual_home_win_prob

        game_list.append(HistoricalGame(
            game_id=f"g_{i:04d}",
            date=date,
            away_team=away,
            home_team=home,
            home_win_prob=round(home_prob, 3),
            away_win_prob=round(away_prob, 3),
            home_odds=home_odds,
            away_odds=away_odds,
            home_won=home_won
        ))

    return game_list


# ============================================================================
# DEMO
# ============================================================================

def demo_backtesting():
    """Demonstrate the backtesting engine"""
    print("=" * 70)
    print("🔬 MLB Predictor - Backtesting Engine Demo")
    print("=" * 70)
    print()

    # Generate sample data
    print("📊 Generating 500-game simulated season...")
    games = generate_sample_season(500)
    print(f"   Generated {len(games)} games")
    print()

    engine = BacktestingEngine(bankroll=10000)

    # Compare strategies
    print("1️⃣  STRATEGY COMPARISON")
    print("-" * 60)
    results = engine.compare_strategies(games)

    print(f"   {'Strategy':<25} {'Bets':>5} {'Win%':>6} {'ROI':>7} "
          f"{'PnL':>10} {'MaxDD%':>7} {'Sharpe':>7}")
    print(f"   {'-'*25} {'-'*5} {'-'*6} {'-'*7} {'-'*10} {'-'*7} {'-'*7}")

    for name, result in sorted(results.items(), key=lambda x: x[1].roi, reverse=True):
        print(f"   {name:<25} {result.total_bets:>5} "
              f"{result.win_rate:>5.1f}% {result.roi:>+6.1f}% "
              f"${result.total_pnl:>+9,.2f} {result.max_drawdown_pct:>6.1f}% "
              f"{result.sharpe_ratio:>6.2f}")
    print()

    # Best strategy details
    best_name = max(results, key=lambda k: results[k].roi)
    best = results[best_name]
    print(f"2️⃣  BEST STRATEGY: {best_name}")
    print("-" * 60)
    print(f"   Record: {best.wins}W-{best.losses}L ({best.win_rate}%)")
    print(f"   ROI: {best.roi:+.1f}% | PnL: ${best.total_pnl:+,.2f}")
    print(f"   Avg Edge: {best.avg_edge}% | Avg Odds: {best.avg_odds:+.0f}")
    print(f"   Profit Factor: {best.profit_factor:.2f}")
    print(f"   Best Streak: {best.longest_win_streak}W | "
          f"Worst: {best.longest_loss_streak}L")
    print()

    # Monthly returns
    if best.monthly_returns:
        print("   Monthly Returns:")
        for month in best.monthly_returns[:6]:
            icon = '✅' if month['pnl'] > 0 else '❌'
            print(f"     {icon} {month['month']}: ${month['pnl']:+,.2f} "
                  f"({month['bets']} bets)")
    print()

    # Calibration
    if best.calibration:
        print("3️⃣  MODEL CALIBRATION")
        print("-" * 60)
        for bucket, data in sorted(best.calibration.items()):
            cal_icon = '✅' if data['calibrated'] else '⚠️'
            print(f"   {cal_icon} Predicted {data['predicted_range']}%: "
                  f"Actual {data['actual_win_rate']}% "
                  f"({data['games']} games)")
    print()

    # Monte Carlo
    print("4️⃣  MONTE CARLO SIMULATION (10,000 runs)")
    print("-" * 60)
    mc = engine.monte_carlo_simulation(
        base_win_rate=best.win_rate / 100,
        base_odds=1.0 + (100 / abs(best.avg_odds)) if best.avg_odds < 0 else 1.0 + best.avg_odds / 100,
        num_bets=500,
        num_simulations=10000,
        stake=100
    )
    print(f"   Probability of Profit: {mc['probability_of_profit']}%")
    print(f"   Probability of Ruin: {mc['probability_of_ruin']}%")
    print(f"   Avg Final Bankroll: ${mc['avg_final_bankroll']:,.2f}")
    print(f"   Percentiles:")
    for pct, val in mc['percentiles'].items():
        print(f"     {pct}: ${val:,.2f}")
    print(f"   Avg Max Drawdown: {mc['avg_max_drawdown_pct']}%")
    print()

    print("=" * 70)
    print("✅ Backtesting Demo Complete")
    print(f"   Best Strategy: {best_name} ({best.roi:+.1f}% ROI)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    demo_backtesting()
