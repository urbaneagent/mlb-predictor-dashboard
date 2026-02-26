#!/usr/bin/env python3
"""
ROI Backtester Dashboard for MLB Predictor

Comprehensive backtesting and visualization system for analyzing historical
betting performance, strategy optimization, and risk-adjusted returns.

Features:
- Historical strategy backtesting
- Interactive strategy builder
- ROI analysis by time periods
- Drawdown and risk analysis
- Sharpe ratio and risk-adjusted returns
- Win rate analysis by confidence buckets
- Strategy comparison tools
- Streak analysis
- Monte Carlo future projections
- HTML dashboard export

Author: MLB Predictor Team
Version: 2.0
License: MIT
"""

import os
import json
import sqlite3
import logging
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('roi_backtester.log'),
        logging.StreamHandler()
    ]
)

class BetType(Enum):
    """Types of bets"""
    MONEYLINE = "moneyline"
    RUN_LINE = "run_line"
    TOTAL = "total"
    PROPS = "props"

class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class StrategyType(Enum):
    """Types of betting strategies"""
    FLAT_BET = "flat_bet"
    KELLY_CRITERION = "kelly_criterion"
    PROPORTIONAL = "proportional"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"

@dataclass
class BetResult:
    """Individual bet result"""
    bet_id: str
    date: datetime
    home_team: str
    away_team: str
    bet_type: BetType
    bet_side: str
    amount: float
    odds: float
    predicted_probability: float
    confidence: ConfidenceLevel
    actual_outcome: str
    won: bool
    profit: float
    roi: float
    model_version: str
    factors: List[str] = field(default_factory=list)
    
    @property
    def implied_probability(self) -> float:
        """Calculate implied probability from odds"""
        if self.odds > 0:
            return 100 / (self.odds + 100)
        else:
            return abs(self.odds) / (abs(self.odds) + 100)

@dataclass
class StrategyFilter:
    """Strategy filter configuration"""
    name: str
    confidence_min: Optional[ConfidenceLevel] = None
    confidence_max: Optional[ConfidenceLevel] = None
    odds_min: Optional[float] = None
    odds_max: Optional[float] = None
    teams: Optional[List[str]] = None
    bet_types: Optional[List[BetType]] = None
    home_only: bool = False
    away_only: bool = False
    value_threshold: Optional[float] = None
    pitcher_names: Optional[List[str]] = None
    month_filter: Optional[List[int]] = None
    day_of_week: Optional[List[int]] = None
    
    def matches(self, bet: BetResult) -> bool:
        """Check if bet matches filter criteria"""
        # Confidence filter
        confidence_values = {'low': 1, 'medium': 2, 'high': 3, 'very_high': 4}
        bet_conf_val = confidence_values.get(bet.confidence.value, 0)
        
        if self.confidence_min:
            min_val = confidence_values.get(self.confidence_min.value, 0)
            if bet_conf_val < min_val:
                return False
                
        if self.confidence_max:
            max_val = confidence_values.get(self.confidence_max.value, 4)
            if bet_conf_val > max_val:
                return False
        
        # Odds filter
        if self.odds_min is not None and bet.odds < self.odds_min:
            return False
        if self.odds_max is not None and bet.odds > self.odds_max:
            return False
            
        # Team filter
        if self.teams and bet.home_team not in self.teams and bet.away_team not in self.teams:
            return False
            
        # Bet type filter
        if self.bet_types and bet.bet_type not in self.bet_types:
            return False
            
        # Home/away filter
        if self.home_only and not bet.bet_side.startswith(bet.home_team):
            return False
        if self.away_only and not bet.bet_side.startswith(bet.away_team):
            return False
            
        # Value threshold filter
        if self.value_threshold is not None:
            edge = bet.predicted_probability - bet.implied_probability
            if edge < self.value_threshold:
                return False
        
        # Month filter
        if self.month_filter and bet.date.month not in self.month_filter:
            return False
            
        # Day of week filter
        if self.day_of_week and bet.date.weekday() not in self.day_of_week:
            return False
            
        return True

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    strategy_type: StrategyType
    base_bet_size: float
    max_bet_size: float
    bankroll_percentage: float
    kelly_fraction: float
    filters: List[StrategyFilter]
    
    def calculate_bet_size(self, bankroll: float, bet: BetResult, 
                          recent_results: List[BetResult]) -> float:
        """Calculate bet size based on strategy"""
        if self.strategy_type == StrategyType.FLAT_BET:
            return min(self.base_bet_size, self.max_bet_size)
            
        elif self.strategy_type == StrategyType.KELLY_CRITERION:
            edge = bet.predicted_probability - bet.implied_probability
            if edge <= 0:
                return 0
            
            kelly_size = (edge * bet.predicted_probability) / bet.implied_probability
            kelly_bet = bankroll * kelly_size * self.kelly_fraction
            return min(kelly_bet, self.max_bet_size, bankroll * self.bankroll_percentage)
            
        elif self.strategy_type == StrategyType.PROPORTIONAL:
            confidence_multiplier = {
                ConfidenceLevel.LOW: 0.5,
                ConfidenceLevel.MEDIUM: 1.0,
                ConfidenceLevel.HIGH: 1.5,
                ConfidenceLevel.VERY_HIGH: 2.0
            }
            
            multiplier = confidence_multiplier.get(bet.confidence, 1.0)
            prop_bet = self.base_bet_size * multiplier
            return min(prop_bet, self.max_bet_size, bankroll * self.bankroll_percentage)
            
        elif self.strategy_type == StrategyType.MARTINGALE:
            if not recent_results or recent_results[-1].won:
                return self.base_bet_size
            
            # Double bet after loss
            last_bet_size = recent_results[-1].amount
            return min(last_bet_size * 2, self.max_bet_size, bankroll * self.bankroll_percentage)
            
        elif self.strategy_type == StrategyType.ANTI_MARTINGALE:
            if not recent_results or not recent_results[-1].won:
                return self.base_bet_size
                
            # Double bet after win
            last_bet_size = recent_results[-1].amount
            return min(last_bet_size * 2, self.max_bet_size, bankroll * self.bankroll_percentage)
            
        return self.base_bet_size

@dataclass
class BacktestResult:
    """Backtesting results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: float
    total_staked: float
    total_profit: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    longest_winning_streak: int
    longest_losing_streak: int
    current_streak: Tuple[str, int]
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    monthly_results: List[Dict[str, Any]] = field(default_factory=list)
    daily_results: List[Dict[str, Any]] = field(default_factory=list)
    confidence_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    drawdown_periods: List[Dict[str, Any]] = field(default_factory=list)

class ROIBacktesterDashboard:
    """
    Comprehensive ROI backtesting and dashboard system
    
    Features:
    - Strategy backtesting with multiple filters
    - Performance visualization and analytics
    - Risk analysis including drawdowns and Sharpe ratio
    - Strategy comparison and optimization
    - Monte Carlo simulation for future projections
    - Interactive HTML dashboard generation
    """
    
    def __init__(self, data_source: str = None):
        """Initialize the backtester"""
        self.bet_results: List[BetResult] = []
        self.strategies: Dict[str, StrategyConfig] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize database if needed
        if data_source:
            self.db_path = data_source
            self._init_database()
        else:
            self.db_path = 'roi_backtester.db'
            self._init_database()
            
        # Load historical data
        self.load_historical_data()
        
        # Create default strategies
        self._create_default_strategies()

    def _init_database(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Bet results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bet_results (
                    bet_id TEXT PRIMARY KEY,
                    date TIMESTAMP NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    bet_type TEXT NOT NULL,
                    bet_side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    odds REAL NOT NULL,
                    predicted_probability REAL NOT NULL,
                    confidence TEXT NOT NULL,
                    actual_outcome TEXT NOT NULL,
                    won BOOLEAN NOT NULL,
                    profit REAL NOT NULL,
                    roi REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    factors TEXT
                )
            ''')
            
            # Strategies table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Backtest results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    backtest_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()

    def load_historical_data(self, start_date: str = None, end_date: str = None) -> None:
        """Load historical betting data"""
        try:
            # Try to load from database first
            with sqlite3.connect(self.db_path) as conn:
                query = 'SELECT * FROM bet_results'
                params = []
                
                if start_date or end_date:
                    conditions = []
                    if start_date:
                        conditions.append('date >= ?')
                        params.append(start_date)
                    if end_date:
                        conditions.append('date <= ?')
                        params.append(end_date)
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                query += ' ORDER BY date'
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if len(df) > 0:
                    self._load_from_dataframe(df)
                    self.logger.info(f"Loaded {len(self.bet_results)} bet results from database")
                    return
                    
        except Exception as e:
            self.logger.warning(f"Could not load from database: {e}")
        
        # Generate mock historical data if no database data
        self._generate_mock_historical_data()
        self.logger.info(f"Generated {len(self.bet_results)} mock bet results")

    def _load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load bet results from pandas DataFrame"""
        for _, row in df.iterrows():
            bet = BetResult(
                bet_id=row['bet_id'],
                date=pd.to_datetime(row['date']),
                home_team=row['home_team'],
                away_team=row['away_team'],
                bet_type=BetType(row['bet_type']),
                bet_side=row['bet_side'],
                amount=row['amount'],
                odds=row['odds'],
                predicted_probability=row['predicted_probability'],
                confidence=ConfidenceLevel(row['confidence']),
                actual_outcome=row['actual_outcome'],
                won=bool(row['won']),
                profit=row['profit'],
                roi=row['roi'],
                model_version=row['model_version'],
                factors=json.loads(row['factors']) if row['factors'] else []
            )
            self.bet_results.append(bet)

    def _generate_mock_historical_data(self) -> None:
        """Generate mock historical betting data"""
        teams = [
            'NYY', 'BOS', 'TB', 'TOR', 'BAL', 'HOU', 'SEA', 'TEX', 'LAA', 'OAK',
            'CLE', 'DET', 'KC', 'MIN', 'CWS', 'ATL', 'PHI', 'NYM', 'MIA', 'WSN',
            'MIL', 'STL', 'CHC', 'CIN', 'PIT', 'LAD', 'SD', 'SF', 'COL', 'AZ'
        ]
        
        # Generate data for past 2 seasons
        start_date = datetime.now() - timedelta(days=500)
        
        bet_id = 1
        for day_offset in range(400):  # 400 days of data
            current_date = start_date + timedelta(days=day_offset)
            
            # Skip some days (no games every day)
            if random.random() < 0.3:
                continue
            
            # Generate 5-15 games per day
            num_games = random.randint(5, 15)
            
            for game_num in range(num_games):
                home_team = random.choice(teams)
                away_team = random.choice([t for t in teams if t != home_team])
                
                # Generate multiple bet types per game
                for bet_type in [BetType.MONEYLINE, BetType.RUN_LINE, BetType.TOTAL]:
                    if random.random() < 0.7:  # 70% chance of betting each type
                        
                        # Determine bet side
                        if bet_type == BetType.MONEYLINE:
                            bet_side = random.choice([home_team, away_team])
                        elif bet_type == BetType.RUN_LINE:
                            spread = random.choice([-1.5, 1.5])
                            team = random.choice([home_team, away_team])
                            bet_side = f"{team} {spread:+.1f}"
                        else:  # TOTAL
                            total = round(random.uniform(8.0, 11.5), 1)
                            bet_side = f"{'Over' if random.random() > 0.5 else 'Under'} {total}"
                        
                        # Generate odds
                        odds = round(random.uniform(-150, 200), 0)
                        if odds > 0:
                            odds = int(odds)
                        else:
                            odds = int(odds)
                        
                        # Generate predicted probability (model prediction)
                        predicted_prob = random.uniform(0.45, 0.75)
                        
                        # Generate confidence level
                        if predicted_prob > 0.65:
                            confidence = random.choice([ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH])
                        elif predicted_prob > 0.55:
                            confidence = ConfidenceLevel.MEDIUM
                        else:
                            confidence = ConfidenceLevel.LOW
                        
                        # Calculate if bet won (simulate with slight edge for higher confidence)
                        win_probability = predicted_prob
                        if confidence == ConfidenceLevel.HIGH:
                            win_probability += 0.03
                        elif confidence == ConfidenceLevel.VERY_HIGH:
                            win_probability += 0.05
                        
                        won = random.random() < win_probability
                        
                        # Generate bet amount (flat betting for mock data)
                        amount = random.choice([25, 50, 100, 200])
                        
                        # Calculate profit
                        if won:
                            if odds > 0:
                                profit = amount * (odds / 100)
                            else:
                                profit = amount * (100 / abs(odds))
                        else:
                            profit = -amount
                        
                        roi = (profit / amount) * 100
                        
                        bet = BetResult(
                            bet_id=f"bet_{bet_id:06d}",
                            date=current_date,
                            home_team=home_team,
                            away_team=away_team,
                            bet_type=bet_type,
                            bet_side=bet_side,
                            amount=amount,
                            odds=odds,
                            predicted_probability=predicted_prob,
                            confidence=confidence,
                            actual_outcome=bet_side if won else "opposite",
                            won=won,
                            profit=profit,
                            roi=roi,
                            model_version=random.choice(['v1.2', 'v1.3', 'v2.0']),
                            factors=random.sample([
                                'Pitcher advantage', 'Home field', 'Recent form',
                                'Weather', 'Bullpen rest', 'Lineup changes',
                                'Historical matchup', 'Injuries'
                            ], random.randint(2, 5))
                        )
                        
                        self.bet_results.append(bet)
                        bet_id += 1
        
        # Store to database
        self._store_bet_results()

    def _store_bet_results(self) -> None:
        """Store bet results to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for bet in self.bet_results:
                    cursor.execute('''
                        INSERT OR REPLACE INTO bet_results 
                        (bet_id, date, home_team, away_team, bet_type, bet_side, amount,
                         odds, predicted_probability, confidence, actual_outcome, won,
                         profit, roi, model_version, factors)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        bet.bet_id, bet.date.isoformat(), bet.home_team, bet.away_team,
                        bet.bet_type.value, bet.bet_side, bet.amount, bet.odds,
                        bet.predicted_probability, bet.confidence.value, bet.actual_outcome,
                        bet.won, bet.profit, bet.roi, bet.model_version,
                        json.dumps(bet.factors)
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing bet results: {e}")

    def _create_default_strategies(self) -> None:
        """Create default betting strategies"""
        
        # Conservative strategy
        conservative_filter = StrategyFilter(
            name="Conservative",
            confidence_min=ConfidenceLevel.MEDIUM,
            odds_min=-120,
            odds_max=110,
            value_threshold=0.05
        )
        
        conservative_strategy = StrategyConfig(
            name="Conservative",
            strategy_type=StrategyType.FLAT_BET,
            base_bet_size=50,
            max_bet_size=100,
            bankroll_percentage=0.02,
            kelly_fraction=0.5,
            filters=[conservative_filter]
        )
        
        self.strategies["Conservative"] = conservative_strategy
        
        # Aggressive strategy
        aggressive_filter = StrategyFilter(
            name="Aggressive",
            confidence_min=ConfidenceLevel.HIGH,
            value_threshold=0.08
        )
        
        aggressive_strategy = StrategyConfig(
            name="Aggressive",
            strategy_type=StrategyType.KELLY_CRITERION,
            base_bet_size=100,
            max_bet_size=500,
            bankroll_percentage=0.05,
            kelly_fraction=0.75,
            filters=[aggressive_filter]
        )
        
        self.strategies["Aggressive"] = aggressive_strategy
        
        # Value betting strategy
        value_filter = StrategyFilter(
            name="Value",
            value_threshold=0.10,
            odds_min=100  # Only positive odds (underdogs)
        )
        
        value_strategy = StrategyConfig(
            name="Value Betting",
            strategy_type=StrategyType.PROPORTIONAL,
            base_bet_size=75,
            max_bet_size=200,
            bankroll_percentage=0.03,
            kelly_fraction=1.0,
            filters=[value_filter]
        )
        
        self.strategies["Value Betting"] = value_strategy
        
        # Favorites only strategy
        favorites_filter = StrategyFilter(
            name="Favorites",
            odds_max=-110,  # Only negative odds (favorites)
            confidence_min=ConfidenceLevel.MEDIUM
        )
        
        favorites_strategy = StrategyConfig(
            name="Favorites Only",
            strategy_type=StrategyType.FLAT_BET,
            base_bet_size=100,
            max_bet_size=200,
            bankroll_percentage=0.04,
            kelly_fraction=1.0,
            filters=[favorites_filter]
        )
        
        self.strategies["Favorites Only"] = favorites_strategy

    def create_custom_strategy(self, config: StrategyConfig) -> None:
        """Create a custom strategy"""
        self.strategies[config.name] = config
        self.logger.info(f"Created custom strategy: {config.name}")

    def backtest_strategy(self, strategy_name: str, start_date: datetime = None, 
                         end_date: datetime = None, initial_bankroll: float = 10000) -> BacktestResult:
        """Backtest a strategy against historical data"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        
        # Filter data by date range
        filtered_bets = self.bet_results
        if start_date:
            filtered_bets = [bet for bet in filtered_bets if bet.date >= start_date]
        if end_date:
            filtered_bets = [bet for bet in filtered_bets if bet.date <= end_date]
        
        # Sort by date
        filtered_bets.sort(key=lambda x: x.date)
        
        # Initialize tracking variables
        bankroll = initial_bankroll
        placed_bets: List[BetResult] = []
        daily_bankrolls: List[Tuple[datetime, float]] = []
        monthly_results: List[Dict[str, Any]] = []
        current_month_profit = 0
        current_month = None
        
        # Streak tracking
        current_streak_type = None
        current_streak_length = 0
        longest_winning_streak = 0
        longest_losing_streak = 0
        
        # Drawdown tracking
        peak_bankroll = initial_bankroll
        max_drawdown = 0
        max_drawdown_duration = 0
        drawdown_start = None
        drawdown_periods: List[Dict[str, Any]] = []
        
        # Process each bet
        for bet in filtered_bets:
            # Check if bet passes strategy filters
            bet_qualifies = False
            for strategy_filter in strategy.filters:
                if strategy_filter.matches(bet):
                    bet_qualifies = True
                    break
            
            if not bet_qualifies:
                continue
            
            # Calculate bet size
            recent_bets = placed_bets[-10:] if len(placed_bets) >= 10 else placed_bets
            bet_size = strategy.calculate_bet_size(bankroll, bet, recent_bets)
            
            if bet_size <= 0 or bet_size > bankroll:
                continue
            
            # Create actual bet with calculated size
            actual_bet = BetResult(
                bet_id=bet.bet_id,
                date=bet.date,
                home_team=bet.home_team,
                away_team=bet.away_team,
                bet_type=bet.bet_type,
                bet_side=bet.bet_side,
                amount=bet_size,
                odds=bet.odds,
                predicted_probability=bet.predicted_probability,
                confidence=bet.confidence,
                actual_outcome=bet.actual_outcome,
                won=bet.won,
                profit=(bet.profit / bet.amount) * bet_size,  # Scale profit by bet size
                roi=(bet.profit / bet.amount) * 100,
                model_version=bet.model_version,
                factors=bet.factors
            )
            
            placed_bets.append(actual_bet)
            bankroll += actual_bet.profit
            
            # Track daily bankroll
            daily_bankrolls.append((actual_bet.date, bankroll))
            
            # Update peak and drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
                if drawdown_start:
                    # End of drawdown period
                    drawdown_periods.append({
                        'start_date': drawdown_start,
                        'end_date': actual_bet.date,
                        'duration_days': (actual_bet.date - drawdown_start).days,
                        'max_drawdown_pct': max_drawdown,
                        'recovery_date': actual_bet.date
                    })
                    drawdown_start = None
            else:
                current_drawdown = (peak_bankroll - bankroll) / peak_bankroll
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                if not drawdown_start:
                    drawdown_start = actual_bet.date
            
            # Track streaks
            if actual_bet.won:
                if current_streak_type == 'win':
                    current_streak_length += 1
                else:
                    current_streak_type = 'win'
                    current_streak_length = 1
                longest_winning_streak = max(longest_winning_streak, current_streak_length)
            else:
                if current_streak_type == 'loss':
                    current_streak_length += 1
                else:
                    current_streak_type = 'loss'
                    current_streak_length = 1
                longest_losing_streak = max(longest_losing_streak, current_streak_length)
            
            # Track monthly results
            bet_month = actual_bet.date.strftime('%Y-%m')
            if current_month != bet_month:
                if current_month is not None:
                    monthly_results.append({
                        'month': current_month,
                        'profit': current_month_profit,
                        'roi': (current_month_profit / initial_bankroll) * 100 if initial_bankroll > 0 else 0
                    })
                current_month = bet_month
                current_month_profit = actual_bet.profit
            else:
                current_month_profit += actual_bet.profit
        
        # Add final month
        if current_month:
            monthly_results.append({
                'month': current_month,
                'profit': current_month_profit,
                'roi': (current_month_profit / initial_bankroll) * 100 if initial_bankroll > 0 else 0
            })
        
        # Calculate results
        if not placed_bets:
            raise ValueError("No bets matched the strategy criteria")
        
        winning_bets = sum(1 for bet in placed_bets if bet.won)
        losing_bets = len(placed_bets) - winning_bets
        win_rate = winning_bets / len(placed_bets)
        
        total_staked = sum(bet.amount for bet in placed_bets)
        total_profit = sum(bet.profit for bet in placed_bets)
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        # Calculate Sharpe ratio
        daily_returns = self._calculate_daily_returns(daily_bankrolls, initial_bankroll)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        
        # Calculate profit factor
        total_wins = sum(bet.profit for bet in placed_bets if bet.won)
        total_losses = abs(sum(bet.profit for bet in placed_bets if not bet.won))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Calculate average win/loss
        avg_win = total_wins / winning_bets if winning_bets > 0 else 0
        avg_loss = total_losses / losing_bets if losing_bets > 0 else 0
        
        # Find largest win/loss
        largest_win = max((bet.profit for bet in placed_bets if bet.won), default=0)
        largest_loss = min((bet.profit for bet in placed_bets if not bet.won), default=0)
        
        # Generate confidence breakdown
        confidence_breakdown = self._analyze_confidence_breakdown(placed_bets)
        
        # Create backtest result
        result = BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date or filtered_bets[0].date,
            end_date=end_date or filtered_bets[-1].date,
            total_bets=len(placed_bets),
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            win_rate=win_rate,
            total_staked=total_staked,
            total_profit=total_profit,
            roi=roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            longest_winning_streak=longest_winning_streak,
            longest_losing_streak=longest_losing_streak,
            current_streak=(current_streak_type or 'none', current_streak_length),
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            monthly_results=monthly_results,
            daily_results=[{'date': date.isoformat(), 'bankroll': bankroll} for date, bankroll in daily_bankrolls],
            confidence_breakdown=confidence_breakdown,
            drawdown_periods=drawdown_periods
        )
        
        self.backtest_results[strategy_name] = result
        self.logger.info(f"Backtested strategy '{strategy_name}': {roi:.2f}% ROI, {win_rate:.1%} win rate")
        
        return result

    def _calculate_daily_returns(self, daily_bankrolls: List[Tuple[datetime, float]], 
                                initial_bankroll: float) -> List[float]:
        """Calculate daily returns from bankroll history"""
        if not daily_bankrolls:
            return []
        
        returns = []
        prev_bankroll = initial_bankroll
        
        for date, bankroll in daily_bankrolls:
            daily_return = (bankroll - prev_bankroll) / prev_bankroll if prev_bankroll > 0 else 0
            returns.append(daily_return)
            prev_bankroll = bankroll
        
        return returns

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns"""
        if not returns or len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        sharpe = (avg_return / std_return) * np.sqrt(365)
        return sharpe

    def _analyze_confidence_breakdown(self, bets: List[BetResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by confidence level"""
        breakdown = {}
        
        for confidence in ConfidenceLevel:
            conf_bets = [bet for bet in bets if bet.confidence == confidence]
            
            if conf_bets:
                winning_bets = sum(1 for bet in conf_bets if bet.won)
                total_profit = sum(bet.profit for bet in conf_bets)
                total_staked = sum(bet.amount for bet in conf_bets)
                
                breakdown[confidence.value] = {
                    'bet_count': len(conf_bets),
                    'win_rate': winning_bets / len(conf_bets),
                    'total_profit': total_profit,
                    'roi': (total_profit / total_staked) * 100 if total_staked > 0 else 0
                }
        
        return breakdown

    def compare_strategies(self, strategy_names: List[str], 
                          start_date: datetime = None, end_date: datetime = None,
                          initial_bankroll: float = 10000) -> Dict[str, Any]:
        """Compare multiple strategies"""
        comparison_results = {}
        
        # Backtest each strategy
        for strategy_name in strategy_names:
            if strategy_name not in self.strategies:
                self.logger.warning(f"Strategy '{strategy_name}' not found, skipping")
                continue
                
            try:
                result = self.backtest_strategy(strategy_name, start_date, end_date, initial_bankroll)
                comparison_results[strategy_name] = result
            except Exception as e:
                self.logger.error(f"Error backtesting strategy '{strategy_name}': {e}")
                continue
        
        if not comparison_results:
            raise ValueError("No strategies could be backtested successfully")
        
        # Generate comparison summary
        comparison_summary = {
            'strategies_compared': len(comparison_results),
            'comparison_period': {
                'start_date': start_date.isoformat() if start_date else 'N/A',
                'end_date': end_date.isoformat() if end_date else 'N/A'
            },
            'performance_ranking': {
                'by_roi': sorted(comparison_results.items(), key=lambda x: x[1].roi, reverse=True),
                'by_sharpe_ratio': sorted(comparison_results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True),
                'by_win_rate': sorted(comparison_results.items(), key=lambda x: x[1].win_rate, reverse=True),
                'by_profit_factor': sorted(comparison_results.items(), key=lambda x: x[1].profit_factor, reverse=True)
            },
            'risk_analysis': {
                'lowest_drawdown': min(comparison_results.items(), key=lambda x: x[1].max_drawdown),
                'highest_sharpe': max(comparison_results.items(), key=lambda x: x[1].sharpe_ratio),
                'most_consistent': min(comparison_results.items(), key=lambda x: abs(x[1].roi - np.mean([r.roi for r in comparison_results.values()])))
            }
        }
        
        return {
            'results': comparison_results,
            'summary': comparison_summary
        }

    def analyze_drawdowns(self, strategy_name: str) -> Dict[str, Any]:
        """Analyze drawdown patterns for a strategy"""
        if strategy_name not in self.backtest_results:
            raise ValueError(f"No backtest results found for strategy '{strategy_name}'")
        
        result = self.backtest_results[strategy_name]
        
        if not result.drawdown_periods:
            return {
                'total_drawdown_periods': 0,
                'max_drawdown_percent': result.max_drawdown * 100,
                'avg_drawdown_duration': 0,
                'longest_drawdown_duration': 0,
                'drawdown_frequency': 0,
                'recovery_analysis': {}
            }
        
        # Analyze drawdown periods
        drawdown_durations = [dd['duration_days'] for dd in result.drawdown_periods]
        avg_duration = np.mean(drawdown_durations)
        longest_duration = max(drawdown_durations)
        
        # Calculate drawdown frequency (drawdowns per year)
        total_days = (result.end_date - result.start_date).days
        frequency_per_year = (len(result.drawdown_periods) / total_days) * 365 if total_days > 0 else 0
        
        # Recovery analysis
        recovery_times = [dd['duration_days'] for dd in result.drawdown_periods]
        
        drawdown_analysis = {
            'total_drawdown_periods': len(result.drawdown_periods),
            'max_drawdown_percent': result.max_drawdown * 100,
            'avg_drawdown_duration': avg_duration,
            'longest_drawdown_duration': longest_duration,
            'drawdown_frequency_per_year': frequency_per_year,
            'recovery_analysis': {
                'avg_recovery_days': np.mean(recovery_times) if recovery_times else 0,
                'fastest_recovery_days': min(recovery_times) if recovery_times else 0,
                'slowest_recovery_days': max(recovery_times) if recovery_times else 0
            },
            'drawdown_periods': result.drawdown_periods
        }
        
        return drawdown_analysis

    def monte_carlo_projection(self, strategy_name: str, num_simulations: int = 1000,
                             future_days: int = 365, confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """Run Monte Carlo simulation for future performance projection"""
        if strategy_name not in self.backtest_results:
            raise ValueError(f"No backtest results found for strategy '{strategy_name}'")
        
        result = self.backtest_results[strategy_name]
        
        # Extract historical daily returns
        if not result.daily_results or len(result.daily_results) < 30:
            raise ValueError("Insufficient historical data for Monte Carlo simulation")
        
        historical_bankrolls = [day['bankroll'] for day in result.daily_results]
        daily_returns = []
        
        for i in range(1, len(historical_bankrolls)):
            daily_return = (historical_bankrolls[i] - historical_bankrolls[i-1]) / historical_bankrolls[i-1]
            daily_returns.append(daily_return)
        
        if not daily_returns:
            raise ValueError("Could not calculate daily returns for simulation")
        
        # Calculate return statistics
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        # Run simulations
        final_bankrolls = []
        simulation_paths = []
        
        starting_bankroll = historical_bankrolls[-1]  # Start from last known bankroll
        
        for simulation in range(num_simulations):
            bankroll_path = [starting_bankroll]
            current_bankroll = starting_bankroll
            
            for day in range(future_days):
                # Generate random return from normal distribution
                daily_return = np.random.normal(mean_return, std_return)
                current_bankroll *= (1 + daily_return)
                
                # Prevent bankroll from going negative
                current_bankroll = max(current_bankroll, 0)
                bankroll_path.append(current_bankroll)
            
            final_bankrolls.append(current_bankroll)
            
            # Store selected simulation paths for visualization
            if simulation < 50:  # Store first 50 paths
                simulation_paths.append(bankroll_path)
        
        # Calculate statistics
        final_bankrolls.sort()
        
        percentiles = {}
        for conf_level in confidence_levels:
            percentile_index = int(conf_level * num_simulations)
            percentiles[f'p{int(conf_level * 100)}'] = final_bankrolls[percentile_index]
        
        # Add median and mean
        percentiles['median'] = np.median(final_bankrolls)
        percentiles['mean'] = np.mean(final_bankrolls)
        
        # Calculate probability of profit
        profitable_simulations = sum(1 for bankroll in final_bankrolls if bankroll > starting_bankroll)
        probability_of_profit = profitable_simulations / num_simulations
        
        # Calculate expected ROI
        expected_roi = ((np.mean(final_bankrolls) - starting_bankroll) / starting_bankroll) * 100
        
        monte_carlo_results = {
            'simulation_parameters': {
                'num_simulations': num_simulations,
                'future_days': future_days,
                'starting_bankroll': starting_bankroll,
                'historical_data_points': len(daily_returns)
            },
            'return_statistics': {
                'mean_daily_return': mean_return,
                'daily_return_volatility': std_return,
                'annualized_return': mean_return * 365,
                'annualized_volatility': std_return * np.sqrt(365)
            },
            'projections': {
                'percentiles': percentiles,
                'probability_of_profit': probability_of_profit,
                'expected_roi': expected_roi,
                'worst_case_loss': ((min(final_bankrolls) - starting_bankroll) / starting_bankroll) * 100,
                'best_case_gain': ((max(final_bankrolls) - starting_bankroll) / starting_bankroll) * 100
            },
            'simulation_paths': simulation_paths[:10],  # Store first 10 paths for visualization
            'risk_metrics': {
                'value_at_risk_5pct': ((percentiles['p5'] - starting_bankroll) / starting_bankroll) * 100,
                'expected_shortfall': np.mean([
                    ((bankroll - starting_bankroll) / starting_bankroll) * 100 
                    for bankroll in final_bankrolls[:int(0.05 * num_simulations)]
                ])
            }
        }
        
        return monte_carlo_results

    def generate_html_dashboard(self, output_file: str = 'roi_dashboard.html') -> str:
        """Generate interactive HTML dashboard"""
        if not self.backtest_results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        # Create plots
        plots = self._create_dashboard_plots()
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MLB Predictor ROI Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .dashboard-header {{
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .dashboard-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    color: #666;
                    margin-top: 5px;
                }}
                .plot-container {{
                    background: white;
                    margin-bottom: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    padding: 20px;
                }}
                .plot-title {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 15px;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>🏗️ MLB Predictor ROI Dashboard</h1>
                <p>Comprehensive Backtesting Results & Performance Analytics</p>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="dashboard-stats">
                {self._generate_summary_stats_html()}
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Strategy Performance Comparison</div>
                <div id="performance-comparison"></div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Cumulative Returns Over Time</div>
                <div id="cumulative-returns"></div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Win Rate by Confidence Level</div>
                <div id="confidence-analysis"></div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Monthly ROI Distribution</div>
                <div id="monthly-performance"></div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Drawdown Analysis</div>
                <div id="drawdown-analysis"></div>
            </div>
            
            <div class="plot-container">
                <div class="plot-title">Monte Carlo Projections (Sample Strategy)</div>
                <div id="monte-carlo"></div>
            </div>
            
            <script>
                // Plot data and configurations
                {self._generate_plotly_javascript(plots)}
            </script>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML dashboard generated: {output_file}")
        return output_file

    def _generate_summary_stats_html(self) -> str:
        """Generate summary statistics HTML"""
        if not self.backtest_results:
            return "<p>No backtest results available</p>"
        
        # Calculate overall statistics
        total_bets = sum(result.total_bets for result in self.backtest_results.values())
        avg_roi = np.mean([result.roi for result in self.backtest_results.values()])
        avg_win_rate = np.mean([result.win_rate for result in self.backtest_results.values()])
        best_strategy = max(self.backtest_results.items(), key=lambda x: x[1].roi)
        
        return f"""
        <div class="stat-card">
            <div class="stat-value">{len(self.backtest_results)}</div>
            <div class="stat-label">Strategies Tested</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{total_bets:,}</div>
            <div class="stat-label">Total Bets Analyzed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_roi:.1f}%</div>
            <div class="stat-label">Average ROI</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_win_rate:.1%}</div>
            <div class="stat-label">Average Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best_strategy[0]}</div>
            <div class="stat-label">Best Strategy<br>({best_strategy[1].roi:.1f}% ROI)</div>
        </div>
        """

    def _create_dashboard_plots(self) -> Dict[str, Any]:
        """Create all dashboard plots"""
        plots = {}
        
        if not self.backtest_results:
            return plots
        
        # Performance comparison plot
        strategies = list(self.backtest_results.keys())
        roi_values = [self.backtest_results[s].roi for s in strategies]
        win_rates = [self.backtest_results[s].win_rate * 100 for s in strategies]
        sharpe_ratios = [self.backtest_results[s].sharpe_ratio for s in strategies]
        
        plots['performance_comparison'] = {
            'data': [
                {
                    'x': strategies,
                    'y': roi_values,
                    'type': 'bar',
                    'name': 'ROI (%)',
                    'marker': {'color': '#667eea'}
                }
            ],
            'layout': {
                'title': 'Strategy ROI Comparison',
                'xaxis': {'title': 'Strategy'},
                'yaxis': {'title': 'ROI (%)'},
                'height': 400
            }
        }
        
        # Cumulative returns (mock data for first strategy)
        if self.backtest_results:
            first_strategy = list(self.backtest_results.keys())[0]
            daily_results = self.backtest_results[first_strategy].daily_results
            
            if daily_results:
                dates = [datetime.fromisoformat(day['date']) for day in daily_results[:100]]  # Limit for demo
                bankrolls = [day['bankroll'] for day in daily_results[:100]]
                
                plots['cumulative_returns'] = {
                    'data': [
                        {
                            'x': [d.isoformat() for d in dates],
                            'y': bankrolls,
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': first_strategy,
                            'line': {'color': '#667eea'}
                        }
                    ],
                    'layout': {
                        'title': f'Cumulative Returns - {first_strategy}',
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Bankroll ($)'},
                        'height': 400
                    }
                }
        
        # Confidence analysis
        confidence_data = []
        for strategy_name, result in self.backtest_results.items():
            for conf_level, stats in result.confidence_breakdown.items():
                confidence_data.append({
                    'strategy': strategy_name,
                    'confidence': conf_level,
                    'win_rate': stats['win_rate'] * 100,
                    'roi': stats['roi']
                })
        
        if confidence_data:
            confidence_levels = ['low', 'medium', 'high', 'very_high']
            win_rates_by_conf = []
            
            for conf in confidence_levels:
                conf_win_rates = [d['win_rate'] for d in confidence_data if d['confidence'] == conf]
                win_rates_by_conf.append(np.mean(conf_win_rates) if conf_win_rates else 0)
            
            plots['confidence_analysis'] = {
                'data': [
                    {
                        'x': confidence_levels,
                        'y': win_rates_by_conf,
                        'type': 'bar',
                        'name': 'Win Rate by Confidence',
                        'marker': {'color': '#764ba2'}
                    }
                ],
                'layout': {
                    'title': 'Win Rate by Confidence Level',
                    'xaxis': {'title': 'Confidence Level'},
                    'yaxis': {'title': 'Win Rate (%)'},
                    'height': 400
                }
            }
        
        return plots

    def _generate_plotly_javascript(self, plots: Dict[str, Any]) -> str:
        """Generate JavaScript for Plotly charts"""
        js_code = ""
        
        for plot_id, plot_data in plots.items():
            js_code += f"""
            Plotly.newPlot('{plot_id.replace('_', '-')}', 
                {json.dumps(plot_data.get('data', []))}, 
                {json.dumps(plot_data.get('layout', {}))});
            """
        
        return js_code

    def export_results(self, format_type: str = 'json', filename: str = None) -> str:
        """Export backtest results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.{format_type}"
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_strategies': len(self.backtest_results),
            'strategies': {}
        }
        
        # Convert results to serializable format
        for strategy_name, result in self.backtest_results.items():
            export_data['strategies'][strategy_name] = {
                'strategy_name': result.strategy_name,
                'start_date': result.start_date.isoformat() if result.start_date else None,
                'end_date': result.end_date.isoformat() if result.end_date else None,
                'total_bets': result.total_bets,
                'winning_bets': result.winning_bets,
                'losing_bets': result.losing_bets,
                'win_rate': result.win_rate,
                'total_staked': result.total_staked,
                'total_profit': result.total_profit,
                'roi': result.roi,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'longest_winning_streak': result.longest_winning_streak,
                'longest_losing_streak': result.longest_losing_streak,
                'current_streak': result.current_streak,
                'profit_factor': result.profit_factor,
                'monthly_results': result.monthly_results,
                'confidence_breakdown': result.confidence_breakdown
            }
        
        if format_type.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format_type.lower() == 'csv':
            # Create summary CSV
            summary_data = []
            for strategy_name, result in self.backtest_results.items():
                summary_data.append({
                    'Strategy': strategy_name,
                    'Total Bets': result.total_bets,
                    'Win Rate': f"{result.win_rate:.1%}",
                    'ROI': f"{result.roi:.2f}%",
                    'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                    'Max Drawdown': f"{result.max_drawdown:.1%}",
                    'Profit Factor': f"{result.profit_factor:.2f}"
                })
            
            df = pd.DataFrame(summary_data)
            df.to_csv(filename, index=False)
        else:
            raise ValueError("Supported formats: 'json', 'csv'")
        
        self.logger.info(f"Results exported to {filename}")
        return filename

    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed summary for a specific strategy"""
        if strategy_name not in self.backtest_results:
            raise ValueError(f"No backtest results found for strategy '{strategy_name}'")
        
        result = self.backtest_results[strategy_name]
        
        summary = {
            'strategy_name': strategy_name,
            'performance_overview': {
                'total_bets': result.total_bets,
                'win_rate': f"{result.win_rate:.1%}",
                'roi': f"{result.roi:.2f}%",
                'total_profit': f"${result.total_profit:,.2f}",
                'sharpe_ratio': f"{result.sharpe_ratio:.2f}"
            },
            'risk_metrics': {
                'max_drawdown': f"{result.max_drawdown:.1%}",
                'longest_losing_streak': result.longest_losing_streak,
                'profit_factor': f"{result.profit_factor:.2f}",
                'largest_loss': f"${result.largest_loss:,.2f}"
            },
            'consistency_metrics': {
                'longest_winning_streak': result.longest_winning_streak,
                'current_streak': f"{result.current_streak[1]} {result.current_streak[0]}s" if result.current_streak[0] else "No streak",
                'avg_win': f"${result.avg_win:,.2f}",
                'avg_loss': f"${abs(result.avg_loss):,.2f}"
            },
            'confidence_breakdown': result.confidence_breakdown,
            'monthly_performance': result.monthly_results[-12:],  # Last 12 months
            'recommendation': self._generate_strategy_recommendation(result)
        }
        
        return summary

    def _generate_strategy_recommendation(self, result: BacktestResult) -> Dict[str, Any]:
        """Generate strategy recommendation based on results"""
        score = 0
        factors = []
        
        # ROI scoring
        if result.roi > 15:
            score += 25
            factors.append("Excellent ROI")
        elif result.roi > 8:
            score += 20
            factors.append("Good ROI")
        elif result.roi > 3:
            score += 10
            factors.append("Moderate ROI")
        else:
            score -= 10
            factors.append("Low ROI")
        
        # Win rate scoring
        if result.win_rate > 0.60:
            score += 20
            factors.append("High win rate")
        elif result.win_rate > 0.55:
            score += 15
            factors.append("Good win rate")
        elif result.win_rate > 0.50:
            score += 10
            factors.append("Moderate win rate")
        else:
            score -= 5
            factors.append("Below average win rate")
        
        # Sharpe ratio scoring
        if result.sharpe_ratio > 1.5:
            score += 20
            factors.append("Excellent risk-adjusted returns")
        elif result.sharpe_ratio > 1.0:
            score += 15
            factors.append("Good risk-adjusted returns")
        elif result.sharpe_ratio > 0.5:
            score += 10
            factors.append("Moderate risk-adjusted returns")
        else:
            score -= 5
            factors.append("Poor risk-adjusted returns")
        
        # Drawdown scoring
        if result.max_drawdown < 0.10:
            score += 15
            factors.append("Low drawdown risk")
        elif result.max_drawdown < 0.20:
            score += 10
            factors.append("Moderate drawdown risk")
        else:
            score -= 10
            factors.append("High drawdown risk")
        
        # Generate recommendation
        if score >= 70:
            recommendation = "Highly Recommended"
            risk_level = "Low to Moderate"
        elif score >= 50:
            recommendation = "Recommended"
            risk_level = "Moderate"
        elif score >= 30:
            recommendation = "Consider with Caution"
            risk_level = "Moderate to High"
        else:
            recommendation = "Not Recommended"
            risk_level = "High"
        
        return {
            'recommendation': recommendation,
            'risk_level': risk_level,
            'score': score,
            'key_factors': factors,
            'suggested_improvements': self._suggest_improvements(result)
        }

    def _suggest_improvements(self, result: BacktestResult) -> List[str]:
        """Suggest improvements based on results"""
        suggestions = []
        
        if result.win_rate < 0.55:
            suggestions.append("Consider tightening filters to focus on higher-confidence bets")
        
        if result.max_drawdown > 0.20:
            suggestions.append("Implement position sizing adjustments to reduce drawdown")
        
        if result.longest_losing_streak > 8:
            suggestions.append("Consider adding streak-breaking mechanisms")
        
        if result.sharpe_ratio < 1.0:
            suggestions.append("Focus on improving risk-adjusted returns through better risk management")
        
        if not suggestions:
            suggestions.append("Strategy is performing well - consider increasing position sizes cautiously")
        
        return suggestions

def main():
    """Demo script for ROI Backtester Dashboard"""
    print("📊 MLB ROI Backtester Dashboard v2.0")
    print("=" * 50)
    
    # Initialize backtester
    print("🔄 Initializing backtester...")
    backtester = ROIBacktesterDashboard()
    
    print(f"✅ Loaded {len(backtester.bet_results)} historical bet results")
    print(f"📋 Created {len(backtester.strategies)} default strategies")
    
    # Run backtests
    print("\n🧪 Running backtests...")
    start_date = datetime.now() - timedelta(days=300)
    end_date = datetime.now() - timedelta(days=30)
    
    backtest_results = {}
    for strategy_name in backtester.strategies.keys():
        try:
            print(f"  Testing {strategy_name}...")
            result = backtester.backtest_strategy(strategy_name, start_date, end_date)
            backtest_results[strategy_name] = result
            print(f"    ✅ {result.roi:.2f}% ROI, {result.win_rate:.1%} win rate")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Display results summary
    print(f"\n📈 Backtest Results Summary:")
    print("-" * 70)
    print(f"{'Strategy':<20} {'ROI':<8} {'Win Rate':<10} {'Sharpe':<8} {'Max DD':<8}")
    print("-" * 70)
    
    for strategy_name, result in backtest_results.items():
        print(f"{strategy_name:<20} {result.roi:>6.1f}% {result.win_rate:>8.1%} {result.sharpe_ratio:>7.2f} {result.max_drawdown:>6.1%}")
    
    # Compare strategies
    print(f"\n🔄 Running strategy comparison...")
    comparison = backtester.compare_strategies(list(backtest_results.keys()), start_date, end_date)
    
    best_roi = comparison['summary']['performance_ranking']['by_roi'][0]
    best_sharpe = comparison['summary']['performance_ranking']['by_sharpe_ratio'][0]
    
    print(f"🏆 Best ROI: {best_roi[0]} ({best_roi[1].roi:.2f}%)")
    print(f"📊 Best Sharpe: {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
    
    # Analyze drawdowns
    print(f"\n📉 Analyzing drawdowns...")
    for strategy_name in list(backtest_results.keys())[:2]:  # Analyze first 2 strategies
        try:
            drawdown_analysis = backtester.analyze_drawdowns(strategy_name)
            print(f"  {strategy_name}:")
            print(f"    Max Drawdown: {drawdown_analysis['max_drawdown_percent']:.1f}%")
            print(f"    Avg Recovery: {drawdown_analysis['recovery_analysis']['avg_recovery_days']:.0f} days")
        except Exception as e:
            print(f"    ❌ Error analyzing {strategy_name}: {e}")
    
    # Monte Carlo projection
    print(f"\n🎲 Running Monte Carlo simulation...")
    first_strategy = list(backtest_results.keys())[0]
    try:
        monte_carlo = backtester.monte_carlo_projection(first_strategy, num_simulations=500, future_days=180)
        
        print(f"  Strategy: {first_strategy}")
        print(f"  Expected ROI (6 months): {monte_carlo['projections']['expected_roi']:.1f}%")
        print(f"  Probability of Profit: {monte_carlo['projections']['probability_of_profit']:.1%}")
        print(f"  5% VaR: {monte_carlo['risk_metrics']['value_at_risk_5pct']:.1f}%")
        
    except Exception as e:
        print(f"    ❌ Monte Carlo error: {e}")
    
    # Generate detailed strategy summary
    print(f"\n📋 Strategy Summary for {first_strategy}:")
    try:
        summary = backtester.get_strategy_summary(first_strategy)
        
        print(f"  Performance Overview:")
        for key, value in summary['performance_overview'].items():
            print(f"    {key.replace('_', ' ').title()}: {value}")
        
        print(f"  Recommendation: {summary['recommendation']['recommendation']} (Score: {summary['recommendation']['score']})")
        
    except Exception as e:
        print(f"    ❌ Summary error: {e}")
    
    # Export results
    print(f"\n💾 Exporting results...")
    try:
        json_file = backtester.export_results('json')
        csv_file = backtester.export_results('csv')
        print(f"  ✅ JSON: {json_file}")
        print(f"  ✅ CSV: {csv_file}")
    except Exception as e:
        print(f"    ❌ Export error: {e}")
    
    # Generate HTML dashboard
    print(f"\n🌐 Generating HTML dashboard...")
    try:
        dashboard_file = backtester.generate_html_dashboard()
        print(f"  ✅ Dashboard: {dashboard_file}")
        print(f"  📱 Open in browser to view interactive charts")
    except Exception as e:
        print(f"    ❌ Dashboard error: {e}")
    
    print(f"\n🎉 ROI Backtester Demo Complete!")
    print(f"📊 Analyzed {len(backtest_results)} strategies")
    print(f"📈 Best performing strategy: {best_roi[0]} ({best_roi[1].roi:.2f}% ROI)")
    print(f"💡 Interactive dashboard available at roi_dashboard.html")

if __name__ == "__main__":
    main()