#!/usr/bin/env python3
"""
DFS Lineup Optimizer for MLB Daily Fantasy Sports

Advanced lineup optimization for DraftKings, FanDuel, and other DFS platforms.
Features salary cap optimization, player projections, correlation matrices,
ownership projections, and multi-lineup generation capabilities.

Author: MLB Predictor Team
Version: 2.0
License: MIT
"""

import json
import csv
import math
import random
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dfs_optimizer.log'),
        logging.StreamHandler()
    ]
)

class Platform(Enum):
    """DFS platform types"""
    DRAFTKINGS = "DraftKings"
    FANDUEL = "FanDuel"
    YAHOO = "Yahoo"
    SUPERDRAFT = "SuperDraft"

class GameType(Enum):
    """DFS game types"""
    CASH = "Cash"
    GPP = "GPP"  # Guaranteed Prize Pool
    TOURNAMENT = "Tournament"
    H2H = "Head-to-Head"

class Position(Enum):
    """Baseball positions for DFS"""
    C = "C"       # Catcher
    FB = "1B"     # First Base
    SB = "2B"     # Second Base
    TB = "3B"     # Third Base
    SS = "SS"     # Shortstop
    OF = "OF"     # Outfield
    P = "P"       # Pitcher
    UTIL = "UTIL" # Utility
    SP = "SP"     # Starting Pitcher
    RP = "RP"     # Relief Pitcher

@dataclass
class Player:
    """Player data for DFS optimization"""
    id: str
    name: str
    team: str
    position: List[str]
    salary: int
    platform: Platform
    projected_points: float
    ownership_projection: float = 0.0
    ceiling: float = 0.0
    floor: float = 0.0
    stdev: float = 0.0
    game_id: str = ""
    opponent: str = ""
    home_away: str = ""
    batting_order: int = 0
    confirmed_starter: bool = True
    injury_status: str = "Healthy"
    weather_factor: float = 1.0
    park_factor: float = 1.0
    platoon_advantage: float = 1.0
    recent_form: float = 1.0

    @property
    def value(self) -> float:
        """Points per $1000 of salary"""
        if self.salary == 0:
            return 0.0
        return (self.projected_points / self.salary) * 1000

    @property
    def adjusted_points(self) -> float:
        """Projected points adjusted for various factors"""
        return (self.projected_points * 
                self.weather_factor * 
                self.park_factor * 
                self.platoon_advantage * 
                self.recent_form)

@dataclass
class Lineup:
    """DFS lineup representation"""
    players: List[Player]
    total_salary: int = 0
    projected_points: float = 0.0
    ownership: float = 0.0
    variance: float = 0.0
    correlation_boost: float = 0.0
    position_map: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate lineup metrics after initialization"""
        self.total_salary = sum(p.salary for p in self.players)
        self.projected_points = sum(p.adjusted_points for p in self.players)
        self.ownership = sum(p.ownership_projection for p in self.players) / len(self.players)
        self.variance = sum(p.stdev ** 2 for p in self.players) ** 0.5

class DFSLineupOptimizer:
    """
    Advanced DFS lineup optimizer for MLB
    
    Features:
    - Multi-platform support (DK, FD, Yahoo)
    - Salary cap optimization
    - Player projection engine with multiple sources
    - Correlation matrices for team stacking
    - Ownership projection and contrarian plays
    - Multi-lineup generation (up to 150 lineups)
    - GPP vs Cash game strategies
    - Late swap support
    - Historical contest backtesting
    """
    
    def __init__(self, platform: Platform = Platform.DRAFTKINGS):
        """Initialize the DFS optimizer"""
        self.platform = platform
        self.players: List[Player] = []
        self.lineups: List[Lineup] = []
        self.settings = self._load_settings()
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.ownership_projections: Dict[str, float] = {}
        self.game_stacks: Dict[str, List[str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Platform-specific configurations
        self.platform_configs = {
            Platform.DRAFTKINGS: {
                'salary_cap': 50000,
                'roster_size': 10,
                'position_limits': {
                    'P': (2, 2), 'C': (1, 1), '1B': (1, 1), '2B': (1, 1),
                    '3B': (1, 1), 'SS': (1, 1), 'OF': (3, 3), 'UTIL': (1, 1)
                }
            },
            Platform.FANDUEL: {
                'salary_cap': 35000,
                'roster_size': 9,
                'position_limits': {
                    'P': (1, 1), 'C/1B': (1, 1), '2B': (1, 1), '3B': (1, 1),
                    'SS': (1, 1), 'OF': (3, 3), 'UTIL': (1, 1)
                }
            }
        }

    def _load_settings(self) -> Dict[str, Any]:
        """Load optimizer settings from config file or use defaults"""
        default_settings = {
            'max_exposure': 0.3,
            'min_exposure': 0.05,
            'correlation_threshold': 0.1,
            'ownership_penalty': 0.5,
            'variance_target': 0.2,
            'stack_sizes': [4, 5],
            'max_players_per_team': 5,
            'min_salary_usage': 0.95,
            'weather_importance': 0.1,
            'projection_sources': ['steamer', 'zips', 'atc', 'marcel']
        }
        
        try:
            with open('dfs_config.json', 'r') as f:
                settings = json.load(f)
                default_settings.update(settings)
        except FileNotFoundError:
            self.logger.info("No config file found, using default settings")
            
        return default_settings

    def load_player_pool(self, csv_file: str = None, api_data: Dict = None) -> None:
        """Load player pool from CSV file or API data"""
        if csv_file:
            self._load_from_csv(csv_file)
        elif api_data:
            self._load_from_api(api_data)
        else:
            self._generate_sample_players()
            
        self.logger.info(f"Loaded {len(self.players)} players for {self.platform.value}")

    def _load_from_csv(self, csv_file: str) -> None:
        """Load players from DraftKings/FanDuel CSV export"""
        try:
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                positions = row['Roster Position'].split('/')
                
                player = Player(
                    id=row.get('ID', str(row.name)),
                    name=row['Name'],
                    team=row['TeamAbbrev'],
                    position=positions,
                    salary=int(row['Salary']),
                    platform=self.platform,
                    projected_points=float(row.get('FPPG', 0)),
                    opponent=row.get('Opponent', ''),
                    game_id=row.get('Game Info', '').split('@')[0] if '@' in row.get('Game Info', '') else ''
                )
                
                self.players.append(player)
                
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise

    def _load_from_api(self, api_data: Dict) -> None:
        """Load players from API response"""
        for player_data in api_data.get('players', []):
            player = Player(
                id=player_data['id'],
                name=player_data['name'],
                team=player_data['team'],
                position=player_data['positions'],
                salary=player_data['salary'],
                platform=self.platform,
                projected_points=player_data['projection'],
                ownership_projection=player_data.get('ownership', 0),
                opponent=player_data.get('opponent', ''),
                game_id=player_data.get('game_id', '')
            )
            self.players.append(player)

    def _generate_sample_players(self) -> None:
        """Generate sample players for testing"""
        teams = ['NYY', 'BOS', 'TOR', 'TB', 'BAL', 'HOU', 'SEA', 'TEX', 'LAA', 'OAK']
        positions_pool = {
            'P': 20, 'C': 10, '1B': 10, '2B': 10, 
            '3B': 10, 'SS': 10, 'OF': 30
        }
        
        player_id = 1
        for pos, count in positions_pool.items():
            for i in range(count):
                salary_range = {
                    'P': (7000, 12000), 'C': (3000, 5500), '1B': (3500, 6000),
                    '2B': (3000, 5500), '3B': (3500, 6000), 'SS': (3000, 5500),
                    'OF': (3500, 6000)
                }
                
                min_sal, max_sal = salary_range[pos]
                salary = random.randint(min_sal, max_sal)
                projection = max(0, random.normalvariate(
                    salary / 1000 * 0.8, salary / 1000 * 0.2
                ))
                
                player = Player(
                    id=str(player_id),
                    name=f"{pos}_{i+1}",
                    team=random.choice(teams),
                    position=[pos],
                    salary=salary,
                    platform=self.platform,
                    projected_points=round(projection, 2),
                    ownership_projection=random.uniform(5, 25),
                    ceiling=projection * 1.5,
                    floor=max(0, projection * 0.5),
                    stdev=projection * 0.3
                )
                
                self.players.append(player)
                player_id += 1

    def calculate_projections(self, sources: List[str] = None) -> None:
        """
        Merge projections from multiple sources using weighted average
        
        Args:
            sources: List of projection source names to include
        """
        if not sources:
            sources = self.settings['projection_sources']
            
        weights = {
            'steamer': 0.3,
            'zips': 0.25,
            'atc': 0.25,
            'marcel': 0.2
        }
        
        for player in self.players:
            projections = []
            
            # Simulate fetching from different sources
            for source in sources:
                base_projection = player.projected_points
                variance = base_projection * 0.15  # 15% variance between sources
                source_projection = max(0, random.normalvariate(base_projection, variance))
                projections.append(source_projection * weights.get(source, 0.25))
            
            # Weighted average of all sources
            player.projected_points = sum(projections)
            
            # Calculate ceiling and floor based on projection variance
            player.ceiling = player.projected_points * random.uniform(1.3, 1.8)
            player.floor = max(0, player.projected_points * random.uniform(0.3, 0.7))
            player.stdev = player.projected_points * random.uniform(0.2, 0.4)

    def build_correlation_matrix(self) -> None:
        """Build correlation matrix for team stacking"""
        team_players = defaultdict(list)
        
        for player in self.players:
            team_players[player.team].append(player)
            
        for team, players in team_players.items():
            self.correlation_matrix[team] = {}
            
            # Batting order correlation (consecutive batters more correlated)
            for i, player1 in enumerate(players):
                for j, player2 in enumerate(players):
                    if i == j:
                        continue
                        
                    # Higher correlation for same team, especially consecutive batters
                    base_correlation = 0.15
                    if abs(player1.batting_order - player2.batting_order) <= 2:
                        base_correlation = 0.25
                    
                    # Pitcher correlation with own hitters (negative)
                    if ('P' in player1.position and 'P' not in player2.position) or \
                       ('P' not in player1.position and 'P' in player2.position):
                        base_correlation = -0.1
                        
                    self.correlation_matrix[team][f"{player1.id}_{player2.id}"] = base_correlation

    def project_ownership(self, game_type: GameType = GameType.GPP) -> None:
        """Project player ownership percentages"""
        if game_type == GameType.CASH:
            # Cash games: ownership more predictable, value-based
            for player in self.players:
                value_factor = min(2.0, player.value / 3.0)
                base_ownership = 5 + (value_factor * 15)
                player.ownership_projection = min(50, max(1, base_ownership))
                
        else:  # GPP/Tournament
            # Tournament: more chalky stars, lower mid-tier ownership
            for player in self.players:
                if player.salary > 9000:  # Stars
                    player.ownership_projection = random.uniform(15, 35)
                elif player.salary > 6000:  # Mid-tier
                    player.ownership_projection = random.uniform(8, 20)
                else:  # Value
                    player.ownership_projection = random.uniform(5, 15)

    def generate_lineups(self, 
                        count: int = 20,
                        game_type: GameType = GameType.GPP,
                        unique_players: int = None) -> List[Lineup]:
        """
        Generate multiple optimal lineups using different strategies
        
        Args:
            count: Number of lineups to generate (max 150)
            game_type: Cash or GPP strategy
            unique_players: Minimum unique players across lineups
            
        Returns:
            List of generated lineups
        """
        self.lineups = []
        count = min(count, 150)  # Platform limit
        
        strategies = self._get_strategies(game_type)
        used_lineups = set()
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            
            # Generate lineup with current strategy
            lineup = self._optimize_single_lineup(strategy, used_lineups)
            
            if lineup and self._is_valid_lineup(lineup):
                self.lineups.append(lineup)
                used_lineups.add(self._lineup_hash(lineup))
                
            if len(self.lineups) >= count:
                break
                
        # Apply unique player constraints if specified
        if unique_players:
            self.lineups = self._enforce_uniqueness(self.lineups, unique_players)
            
        self.logger.info(f"Generated {len(self.lineups)} lineups using {game_type.value} strategy")
        return self.lineups

    def _get_strategies(self, game_type: GameType) -> List[Dict[str, Any]]:
        """Get optimization strategies based on game type"""
        base_strategies = [
            {'name': 'optimal', 'ownership_penalty': 0, 'correlation_boost': 0},
            {'name': 'balanced', 'ownership_penalty': 0.3, 'correlation_boost': 0.1},
            {'name': 'contrarian', 'ownership_penalty': 0.8, 'correlation_boost': 0},
            {'name': 'correlation', 'ownership_penalty': 0.2, 'correlation_boost': 0.3},
            {'name': 'stars_studs', 'ownership_penalty': 0, 'correlation_boost': 0, 'min_salary': 8000},
            {'name': 'value_heavy', 'ownership_penalty': 0.5, 'correlation_boost': 0, 'max_salary': 6000}
        ]
        
        if game_type == GameType.CASH:
            # Cash games favor safety and high floor
            return [
                {'name': 'cash_optimal', 'ownership_penalty': 0, 'floor_weight': 0.3},
                {'name': 'cash_safe', 'ownership_penalty': 0.1, 'floor_weight': 0.5},
                {'name': 'cash_balanced', 'ownership_penalty': 0.2, 'floor_weight': 0.4}
            ]
        else:
            return base_strategies

    def _optimize_single_lineup(self, strategy: Dict[str, Any], used_lineups: Set[str]) -> Optional[Lineup]:
        """Optimize a single lineup using given strategy"""
        config = self.platform_configs[self.platform]
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            lineup_players = []
            total_salary = 0
            position_counts = defaultdict(int)
            
            # Apply strategy modifications to player scores
            weighted_players = self._apply_strategy_weights(self.players, strategy)
            
            # Greedy selection with position constraints
            available_positions = list(config['position_limits'].keys())
            random.shuffle(available_positions)
            
            for position in available_positions:
                min_count, max_count = config['position_limits'][position]
                
                eligible_players = [
                    p for p in weighted_players 
                    if any(pos in position or position in pos for pos in p.position)
                    and total_salary + p.salary <= config['salary_cap']
                    and p not in lineup_players
                ]
                
                if not eligible_players:
                    break
                    
                # Sort by weighted score
                eligible_players.sort(key=lambda x: x.projected_points, reverse=True)
                
                # Add some randomness for variety
                top_choices = min(len(eligible_players), 5)
                selected = random.choice(eligible_players[:top_choices])
                
                lineup_players.append(selected)
                total_salary += selected.salary
                position_counts[position] += 1
                
                if len(lineup_players) >= config['roster_size']:
                    break
                    
            # Create lineup if valid
            if len(lineup_players) == config['roster_size']:
                lineup = Lineup(players=lineup_players)
                lineup_hash = self._lineup_hash(lineup)
                
                if lineup_hash not in used_lineups:
                    return lineup
                    
        self.logger.warning(f"Failed to generate unique lineup after {max_attempts} attempts")
        return None

    def _apply_strategy_weights(self, players: List[Player], strategy: Dict[str, Any]) -> List[Player]:
        """Apply strategy-specific weights to player projections"""
        weighted_players = []
        
        for player in players:
            weighted_projection = player.adjusted_points
            
            # Ownership penalty
            if 'ownership_penalty' in strategy:
                ownership_penalty = strategy['ownership_penalty'] * (player.ownership_projection / 100)
                weighted_projection *= (1 - ownership_penalty)
            
            # Floor weighting for cash games
            if 'floor_weight' in strategy:
                floor_bonus = player.floor * strategy['floor_weight']
                weighted_projection += floor_bonus
            
            # Salary filters
            if 'min_salary' in strategy and player.salary < strategy['min_salary']:
                continue
            if 'max_salary' in strategy and player.salary > strategy['max_salary']:
                continue
                
            # Create copy with weighted projection
            weighted_player = Player(
                id=player.id,
                name=player.name,
                team=player.team,
                position=player.position,
                salary=player.salary,
                platform=player.platform,
                projected_points=weighted_projection,
                ownership_projection=player.ownership_projection,
                ceiling=player.ceiling,
                floor=player.floor,
                stdev=player.stdev
            )
            
            weighted_players.append(weighted_player)
            
        return weighted_players

    def _is_valid_lineup(self, lineup: Lineup) -> bool:
        """Validate lineup meets platform constraints"""
        config = self.platform_configs[self.platform]
        
        # Salary cap check
        if lineup.total_salary > config['salary_cap']:
            return False
            
        # Position requirements check
        position_counts = defaultdict(int)
        for player in lineup.players:
            for position in player.position:
                position_counts[position] += 1
                
        for position, (min_count, max_count) in config['position_limits'].items():
            actual_count = position_counts.get(position, 0)
            if actual_count < min_count or actual_count > max_count:
                return False
                
        return True

    def _lineup_hash(self, lineup: Lineup) -> str:
        """Generate unique hash for lineup"""
        player_ids = sorted([p.id for p in lineup.players])
        return '|'.join(player_ids)

    def _enforce_uniqueness(self, lineups: List[Lineup], min_unique: int) -> List[Lineup]:
        """Enforce minimum unique players across lineups"""
        if len(lineups) <= 1:
            return lineups
            
        filtered_lineups = [lineups[0]]  # Keep first lineup
        
        for lineup in lineups[1:]:
            is_unique = True
            
            for existing in filtered_lineups:
                shared_players = set(p.id for p in lineup.players) & set(p.id for p in existing.players)
                if len(shared_players) > len(lineup.players) - min_unique:
                    is_unique = False
                    break
                    
            if is_unique:
                filtered_lineups.append(lineup)
                
        return filtered_lineups

    def late_swap_update(self, updated_projections: Dict[str, float]) -> None:
        """Update projections for late swap before lineup lock"""
        swap_count = 0
        
        for player_id, new_projection in updated_projections.items():
            # Update player projection
            for player in self.players:
                if player.id == player_id:
                    old_projection = player.projected_points
                    player.projected_points = new_projection
                    
                    self.logger.info(f"Late swap: {player.name} {old_projection:.1f} -> {new_projection:.1f}")
                    break
            
            # Update existing lineups if improvement is significant
            for lineup in self.lineups:
                for i, player in enumerate(lineup.players):
                    if player.id == player_id and new_projection > player.projected_points * 1.1:
                        # Find better replacement
                        better_players = [
                            p for p in self.players
                            if p.projected_points > new_projection
                            and p.salary <= player.salary + 500  # Small salary buffer
                            and any(pos in player.position for pos in p.position)
                            and p not in lineup.players
                        ]
                        
                        if better_players:
                            best_replacement = max(better_players, key=lambda x: x.projected_points)
                            lineup.players[i] = best_replacement
                            swap_count += 1
                            
                            self.logger.info(f"Late swap: {player.name} -> {best_replacement.name}")
                            
        self.logger.info(f"Completed late swap: {swap_count} player swaps made")

    def export_lineups(self, filename: str = None, format_type: str = 'csv') -> str:
        """Export lineups to DraftKings/FanDuel format"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dfs_lineups_{self.platform.value.lower()}_{timestamp}.{format_type}"
            
        if format_type.lower() == 'csv':
            return self._export_csv(filename)
        elif format_type.lower() == 'json':
            return self._export_json(filename)
        else:
            raise ValueError("Supported formats: csv, json")

    def _export_csv(self, filename: str) -> str:
        """Export lineups to platform-specific CSV format"""
        config = self.platform_configs[self.platform]
        
        with open(filename, 'w', newline='') as csvfile:
            if self.platform == Platform.DRAFTKINGS:
                fieldnames = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']
            else:  # FanDuel
                fieldnames = ['P', 'C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']
                
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for lineup in self.lineups:
                row = {}
                position_assignments = self._assign_positions(lineup.players, fieldnames)
                
                for pos, player in position_assignments.items():
                    row[pos] = f"{player.name} ({player.id})" if player else ""
                    
                writer.writerow(row)
                
        self.logger.info(f"Exported {len(self.lineups)} lineups to {filename}")
        return filename

    def _export_json(self, filename: str) -> str:
        """Export lineups to JSON format with detailed metadata"""
        export_data = {
            'platform': self.platform.value,
            'export_time': datetime.now().isoformat(),
            'lineup_count': len(self.lineups),
            'lineups': []
        }
        
        for i, lineup in enumerate(self.lineups):
            lineup_data = {
                'lineup_id': i + 1,
                'total_salary': lineup.total_salary,
                'projected_points': round(lineup.projected_points, 2),
                'projected_ownership': round(lineup.ownership, 2),
                'variance': round(lineup.variance, 2),
                'players': [
                    {
                        'id': player.id,
                        'name': player.name,
                        'position': player.position,
                        'team': player.team,
                        'salary': player.salary,
                        'projection': round(player.projected_points, 2),
                        'ownership': round(player.ownership_projection, 2)
                    }
                    for player in lineup.players
                ]
            }
            export_data['lineups'].append(lineup_data)
            
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        return filename

    def _assign_positions(self, players: List[Player], fieldnames: List[str]) -> Dict[str, Player]:
        """Assign players to specific lineup positions"""
        assignments = {pos: None for pos in fieldnames}
        unassigned_players = players.copy()
        
        # First pass: assign players to their primary positions
        for position in fieldnames:
            if not unassigned_players:
                break
                
            for player in unassigned_players:
                if position in player.position or any(pos in position for pos in player.position):
                    assignments[position] = player
                    unassigned_players.remove(player)
                    break
                    
        # Second pass: assign remaining players to UTIL or flexible positions
        for player in unassigned_players:
            for position in ['UTIL', 'C/1B']:
                if position in assignments and assignments[position] is None:
                    assignments[position] = player
                    break
                    
        return assignments

    def backtest_optimizer(self, 
                          historical_contests: List[Dict],
                          start_date: str,
                          end_date: str) -> Dict[str, Any]:
        """
        Backtest the optimizer against historical contest results
        
        Args:
            historical_contests: List of past contest data
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            
        Returns:
            Backtesting results and performance metrics
        """
        results = {
            'contests_analyzed': 0,
            'total_lineups': 0,
            'avg_score': 0.0,
            'top_percentile_finishes': 0,
            'profitable_contests': 0,
            'roi': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'contest_breakdown': []
        }
        
        total_invested = 0
        total_winnings = 0
        daily_returns = []
        peak_bankroll = 0
        current_bankroll = 10000  # Starting bankroll
        
        for contest in historical_contests:
            contest_date = contest['date']
            if start_date <= contest_date <= end_date:
                # Simulate lineup generation for historical contest
                historical_players = contest.get('players', [])
                
                # Mock optimization using historical projections
                lineups_generated = min(20, contest.get('max_entries', 20))
                entry_fee = contest.get('entry_fee', 25)
                
                contest_results = {
                    'date': contest_date,
                    'contest_name': contest['name'],
                    'lineups_entered': lineups_generated,
                    'entry_fee': entry_fee,
                    'total_invested': lineups_generated * entry_fee,
                    'scores': [],
                    'winnings': 0,
                    'net_profit': 0
                }
                
                # Simulate lineup scores based on actual results
                for lineup_idx in range(lineups_generated):
                    # Mock score based on contest difficulty and randomness
                    base_score = contest.get('winning_score', 150) * random.uniform(0.6, 1.1)
                    noise = random.normalvariate(0, base_score * 0.1)
                    lineup_score = max(0, base_score + noise)
                    
                    contest_results['scores'].append(round(lineup_score, 2))
                    
                    # Calculate winnings based on percentile finish
                    percentile = random.uniform(0, 1)
                    if percentile > 0.9:  # Top 10%
                        winnings = entry_fee * random.uniform(2, 10)
                        contest_results['winnings'] += winnings
                        results['top_percentile_finishes'] += 1
                
                contest_results['net_profit'] = contest_results['winnings'] - contest_results['total_invested']
                total_invested += contest_results['total_invested']
                total_winnings += contest_results['winnings']
                
                # Track bankroll
                current_bankroll += contest_results['net_profit']
                peak_bankroll = max(peak_bankroll, current_bankroll)
                daily_returns.append(contest_results['net_profit'] / contest_results['total_invested'])
                
                if contest_results['net_profit'] > 0:
                    results['profitable_contests'] += 1
                    
                results['contest_breakdown'].append(contest_results)
                results['contests_analyzed'] += 1
                results['total_lineups'] += lineups_generated
        
        # Calculate summary statistics
        if results['contests_analyzed'] > 0:
            all_scores = [score for contest in results['contest_breakdown'] for score in contest['scores']]
            results['avg_score'] = sum(all_scores) / len(all_scores) if all_scores else 0
            
            results['roi'] = ((total_winnings - total_invested) / total_invested * 100) if total_invested > 0 else 0
            
            # Sharpe ratio calculation
            if daily_returns:
                avg_return = np.mean(daily_returns)
                return_std = np.std(daily_returns) if len(daily_returns) > 1 else 0
                results['sharpe_ratio'] = (avg_return / return_std) if return_std > 0 else 0
            
            # Maximum drawdown
            if peak_bankroll > 0:
                results['max_drawdown'] = ((peak_bankroll - min(current_bankroll, peak_bankroll)) / peak_bankroll * 100)
        
        self.logger.info(f"Backtest complete: {results['contests_analyzed']} contests, {results['roi']:.1f}% ROI")
        return results

    def analyze_ownership_patterns(self, contest_results: List[Dict]) -> Dict[str, Any]:
        """Analyze ownership patterns from past contests"""
        ownership_analysis = {
            'high_owned_performance': {},
            'low_owned_gems': [],
            'ownership_correlation': 0.0,
            'optimal_ownership_range': (0, 100),
            'chalk_avoidance_spots': []
        }
        
        high_owned = []
        low_owned = []
        performances = []
        ownerships = []
        
        for contest in contest_results:
            for player_result in contest.get('player_results', []):
                ownership = player_result.get('ownership', 0)
                points = player_result.get('points', 0)
                
                ownerships.append(ownership)
                performances.append(points)
                
                if ownership > 25:  # High owned
                    high_owned.append({'ownership': ownership, 'points': points, 'name': player_result.get('name', '')})
                elif ownership < 10:  # Low owned
                    low_owned.append({'ownership': ownership, 'points': points, 'name': player_result.get('name', '')})
        
        # Find correlation between ownership and performance
        if len(ownerships) > 1 and len(performances) > 1:
            correlation_matrix = np.corrcoef(ownerships, performances)
            ownership_analysis['ownership_correlation'] = correlation_matrix[0, 1]
        
        # Find low-owned gems (high points, low ownership)
        low_owned_gems = [
            player for player in low_owned 
            if player['points'] > np.percentile([p['points'] for p in low_owned], 80)
        ]
        ownership_analysis['low_owned_gems'] = sorted(low_owned_gems, key=lambda x: x['points'], reverse=True)[:10]
        
        # Analyze high-owned performance
        if high_owned:
            avg_high_owned_points = np.mean([p['points'] for p in high_owned])
            ownership_analysis['high_owned_performance']['avg_points'] = avg_high_owned_points
            ownership_analysis['high_owned_performance']['sample_size'] = len(high_owned)
        
        return ownership_analysis

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of current optimization results"""
        if not self.lineups:
            return {'error': 'No lineups generated'}
            
        summary = {
            'total_lineups': len(self.lineups),
            'avg_projection': np.mean([l.projected_points for l in self.lineups]),
            'avg_salary': np.mean([l.total_salary for l in self.lineups]),
            'avg_ownership': np.mean([l.ownership for l in self.lineups]),
            'salary_usage': np.mean([l.total_salary / self.platform_configs[self.platform]['salary_cap'] for l in self.lineups]),
            'top_lineup': max(self.lineups, key=lambda x: x.projected_points),
            'most_contrarian': min(self.lineups, key=lambda x: x.ownership),
            'player_exposures': self._calculate_exposures(),
            'team_stacks': self._analyze_team_stacks(),
            'position_analysis': self._analyze_positions()
        }
        
        return summary

    def _calculate_exposures(self) -> Dict[str, float]:
        """Calculate player exposure percentages across lineups"""
        exposures = defaultdict(int)
        total_lineups = len(self.lineups)
        
        for lineup in self.lineups:
            for player in lineup.players:
                exposures[player.name] += 1
                
        return {
            name: (count / total_lineups * 100) 
            for name, count in exposures.items()
        }

    def _analyze_team_stacks(self) -> Dict[str, Any]:
        """Analyze team stacking patterns in generated lineups"""
        team_stacks = defaultdict(list)
        
        for i, lineup in enumerate(self.lineups):
            team_counts = Counter(player.team for player in lineup.players)
            
            for team, count in team_counts.items():
                if count >= 3:  # Consider 3+ players a stack
                    team_stacks[team].append({'lineup_id': i, 'stack_size': count})
        
        stack_analysis = {}
        for team, stacks in team_stacks.items():
            stack_analysis[team] = {
                'total_stacks': len(stacks),
                'avg_stack_size': np.mean([s['stack_size'] for s in stacks]),
                'max_stack_size': max([s['stack_size'] for s in stacks]) if stacks else 0
            }
            
        return stack_analysis

    def _analyze_positions(self) -> Dict[str, Any]:
        """Analyze position usage and salary allocation"""
        position_data = defaultdict(list)
        
        for lineup in self.lineups:
            for player in lineup.players:
                primary_position = player.position[0] if player.position else 'Unknown'
                position_data[primary_position].append({
                    'salary': player.salary,
                    'projection': player.projected_points,
                    'value': player.value
                })
        
        position_analysis = {}
        for position, players in position_data.items():
            if players:
                position_analysis[position] = {
                    'avg_salary': np.mean([p['salary'] for p in players]),
                    'avg_projection': np.mean([p['projection'] for p in players]),
                    'avg_value': np.mean([p['value'] for p in players]),
                    'salary_range': (min([p['salary'] for p in players]), max([p['salary'] for p in players]))
                }
        
        return position_analysis

def main():
    """Demo script for DFS Lineup Optimizer"""
    print("🏗️ MLB DFS Lineup Optimizer v2.0")
    print("=" * 50)
    
    # Initialize optimizer
    optimizer = DFSLineupOptimizer(Platform.DRAFTKINGS)
    
    # Load sample player pool
    print("📊 Loading player pool...")
    optimizer.load_player_pool()
    
    # Calculate projections
    print("🎯 Calculating projections...")
    optimizer.calculate_projections()
    
    # Build correlation matrix
    print("🔗 Building correlation matrix...")
    optimizer.build_correlation_matrix()
    
    # Project ownership
    print("📈 Projecting ownership...")
    optimizer.project_ownership(GameType.GPP)
    
    # Generate lineups
    print("⚡ Generating lineups...")
    lineups = optimizer.generate_lineups(count=20, game_type=GameType.GPP)
    
    # Display results
    print(f"\n✅ Generated {len(lineups)} optimal lineups")
    
    if lineups:
        best_lineup = max(lineups, key=lambda x: x.projected_points)
        print(f"\n🏆 Best Lineup ({best_lineup.projected_points:.1f} points, ${best_lineup.total_salary}):")
        
        for player in best_lineup.players:
            print(f"  {player.position[0]:>2} | {player.name:<15} | {player.team} | ${player.salary:>4} | {player.projected_points:.1f}pts")
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print(f"\n📊 Optimization Summary:")
    print(f"  Average Projection: {summary['avg_projection']:.1f} points")
    print(f"  Average Salary: ${summary['avg_salary']:,.0f}")
    print(f"  Salary Usage: {summary['salary_usage']:.1%}")
    print(f"  Average Ownership: {summary['avg_ownership']:.1f}%")
    
    # Export lineups
    print("\n💾 Exporting lineups...")
    csv_file = optimizer.export_lineups(format_type='csv')
    json_file = optimizer.export_lineups(format_type='json')
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")
    
    # Simulate late swap
    print("\n🔄 Simulating late swap...")
    late_updates = {
        optimizer.players[0].id: optimizer.players[0].projected_points * 1.2,
        optimizer.players[1].id: optimizer.players[1].projected_points * 0.8
    }
    optimizer.late_swap_update(late_updates)
    
    # Mock backtest
    print("\n📈 Running backtest simulation...")
    mock_contests = [
        {
            'date': '2024-07-15',
            'name': 'GPP Main',
            'entry_fee': 25,
            'max_entries': 20,
            'winning_score': 180,
            'players': []
        }
        for _ in range(10)
    ]
    
    backtest_results = optimizer.backtest_optimizer(
        mock_contests, 
        '2024-07-01', 
        '2024-07-31'
    )
    
    print(f"  Contests Analyzed: {backtest_results['contests_analyzed']}")
    print(f"  ROI: {backtest_results['roi']:.1f}%")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest_results['max_drawdown']:.1f}%")
    
    print(f"\n🎉 Demo complete! Generated {len(lineups)} lineups for {optimizer.platform.value}")

if __name__ == "__main__":
    main()