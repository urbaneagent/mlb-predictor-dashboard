"""
Deep Historical Matchup Analysis Engine

This module provides comprehensive historical matchup analysis for MLB games,
including team vs team records, pitcher vs lineup matchups, venue-specific
performance, and situational splits analysis.

Author: MLB Predictor System
Created: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, Counter
import statistics
from scipy.stats import binomial, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GameSituation(Enum):
    """Different game situations for analysis."""
    AHEAD_EARLY = "ahead_early"  # Leading in innings 1-6
    AHEAD_LATE = "ahead_late"    # Leading in innings 7+
    BEHIND_EARLY = "behind_early"
    BEHIND_LATE = "behind_late"
    TIED_EARLY = "tied_early"
    TIED_LATE = "tied_late"
    BASES_EMPTY = "bases_empty"
    RUNNERS_ON = "runners_on"
    SCORING_POSITION = "scoring_position"
    BASES_LOADED = "bases_loaded"


class WeatherCondition(Enum):
    """Weather conditions affecting play."""
    CLEAR = "clear"
    OVERCAST = "overcast"
    LIGHT_RAIN = "light_rain"
    WINDY = "windy"
    COLD = "cold"
    HOT = "hot"


@dataclass
class TeamRecord:
    """Historical record between two teams."""
    wins: int = 0
    losses: int = 0
    runs_scored: int = 0
    runs_allowed: int = 0
    games_played: int = 0
    
    @property
    def win_percentage(self) -> float:
        """Calculate win percentage."""
        return self.wins / max(self.games_played, 1)
    
    @property
    def avg_runs_scored(self) -> float:
        """Average runs scored per game."""
        return self.runs_scored / max(self.games_played, 1)
    
    @property
    def avg_runs_allowed(self) -> float:
        """Average runs allowed per game."""
        return self.runs_allowed / max(self.games_played, 1)
    
    @property
    def run_differential(self) -> float:
        """Average run differential per game."""
        return self.avg_runs_scored - self.avg_runs_allowed


@dataclass
class VenueStats:
    """Team performance at a specific venue."""
    venue_name: str
    team_name: str
    record: TeamRecord = field(default_factory=TeamRecord)
    total_games: int = 0
    home_games: int = 0
    away_games: int = 0
    day_games: int = 0
    night_games: int = 0
    weather_performance: Dict[WeatherCondition, TeamRecord] = field(default_factory=dict)
    
    @property
    def home_win_percentage(self) -> float:
        """Win percentage at this venue as home team."""
        return self.record.win_percentage if self.home_games > 0 else 0.0


@dataclass
class PitcherVsLineup:
    """Pitcher performance against specific lineup/team."""
    pitcher_id: str
    pitcher_name: str
    opponent_team: str
    career_record: TeamRecord = field(default_factory=TeamRecord)
    batter_matchups: Dict[str, Dict] = field(default_factory=dict)  # batter_id -> stats
    recent_performance: List[Dict] = field(default_factory=list)  # Last 10 starts vs team
    venue_splits: Dict[str, TeamRecord] = field(default_factory=dict)
    
    def get_batter_stats(self, batter_id: str) -> Dict[str, Any]:
        """Get specific batter vs pitcher stats."""
        return self.batter_matchups.get(batter_id, {
            "at_bats": 0, "hits": 0, "home_runs": 0, "rbis": 0,
            "walks": 0, "strikeouts": 0, "avg": 0.000, "obp": 0.000, "slg": 0.000
        })


@dataclass
class DivisionRivalry:
    """Division rivalry adjustments and historical data."""
    team1: str
    team2: str
    is_division_rival: bool
    all_time_record: TeamRecord = field(default_factory=TeamRecord)
    last_season_record: TeamRecord = field(default_factory=TeamRecord)
    playoff_implications: Dict[str, Any] = field(default_factory=dict)
    rivalry_intensity: float = 1.0  # Multiplier for rivalry games
    
    @property
    def rivalry_factor(self) -> float:
        """Calculate rivalry adjustment factor."""
        base_factor = 1.1 if self.is_division_rival else 1.0
        intensity_bonus = min(0.2, self.rivalry_intensity * 0.1)
        return base_factor + intensity_bonus


@dataclass
class SituationalSplit:
    """Performance in specific game situations."""
    situation: GameSituation
    team_name: str
    record: TeamRecord = field(default_factory=TeamRecord)
    batting_stats: Dict[str, float] = field(default_factory=dict)
    pitching_stats: Dict[str, float] = field(default_factory=dict)
    clutch_performance: float = 1.0  # Multiplier for clutch situations


@dataclass
class DayOfWeekPattern:
    """Team performance patterns by day of week."""
    team_name: str
    day_records: Dict[str, TeamRecord] = field(default_factory=dict)
    travel_fatigue_impact: Dict[str, float] = field(default_factory=dict)
    pitcher_rest_patterns: Dict[str, List[int]] = field(default_factory=dict)
    
    def get_day_performance(self, day: str) -> TeamRecord:
        """Get performance for specific day of week."""
        return self.day_records.get(day, TeamRecord())


@dataclass
class HistoricalMatchup:
    """Comprehensive historical matchup analysis."""
    team1: str
    team2: str
    venue: str
    analysis_date: datetime
    
    # Core matchup data
    h2h_record: TeamRecord
    venue_advantage: float
    recent_trend: str  # "team1_hot", "team2_hot", "neutral"
    
    # Detailed breakdowns
    pitcher_matchups: List[PitcherVsLineup]
    division_rivalry: DivisionRivalry
    situational_splits: List[SituationalSplit]
    day_patterns: List[DayOfWeekPattern]
    
    # Statistical significance
    sample_size: int
    confidence_interval: Tuple[float, float]
    statistical_significance: bool
    
    # Predictive factors
    key_advantages: List[str]
    predicted_impact: float  # -1 to 1 (team2 favored to team1 favored)


class HistoricalMatchupEngine:
    """
    Deep historical matchup analysis engine for MLB games.
    
    This class provides comprehensive analysis of team vs team historical
    performance, incorporating venue effects, pitcher matchups, situational
    performance, and various contextual factors.
    """
    
    def __init__(
        self,
        historical_data_path: Optional[str] = None,
        min_sample_size: int = 10,
        significance_level: float = 0.05,
        lookback_years: int = 5
    ):
        """
        Initialize the historical matchup engine.
        
        Args:
            historical_data_path: Path to historical game data
            min_sample_size: Minimum games for statistical significance
            significance_level: Statistical significance threshold
            lookback_years: Years of historical data to consider
        """
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.lookback_years = lookback_years
        
        # Data storage
        self.game_history: List[Dict] = []
        self.team_records: Dict[str, Dict[str, TeamRecord]] = defaultdict(lambda: defaultdict(TeamRecord))
        self.venue_stats: Dict[str, Dict[str, VenueStats]] = defaultdict(lambda: defaultdict(VenueStats))
        self.pitcher_vs_team: Dict[str, Dict[str, PitcherVsLineup]] = defaultdict(lambda: defaultdict(PitcherVsLineup))
        self.division_rivalries: Dict[Tuple[str, str], DivisionRivalry] = {}
        self.situational_data: Dict[str, Dict[GameSituation, SituationalSplit]] = defaultdict(dict)
        self.day_patterns: Dict[str, DayOfWeekPattern] = {}
        
        # MLB divisions for rivalry detection
        self.divisions = {
            'AL East': ['NYY', 'BOS', 'TB', 'TOR', 'BAL'],
            'AL Central': ['CLE', 'MIN', 'CWS', 'DET', 'KC'],
            'AL West': ['HOU', 'SEA', 'TEX', 'LAA', 'OAK'],
            'NL East': ['ATL', 'PHI', 'NYM', 'MIA', 'WSH'],
            'NL Central': ['MIL', 'STL', 'CHC', 'CIN', 'PIT'],
            'NL West': ['LAD', 'SD', 'SF', 'COL', 'ARI']
        }
        
        # Load historical data if provided
        if historical_data_path:
            self._load_historical_data(historical_data_path)
            self._process_historical_data()
        
        logger.info(f"Historical Matchup Engine initialized with {lookback_years} year lookback")
    
    def get_h2h_record(
        self,
        team1: str,
        team2: str,
        venue: Optional[str] = None,
        years_back: Optional[int] = None,
        include_playoffs: bool = True
    ) -> TeamRecord:
        """
        Get head-to-head record between two teams.
        
        Args:
            team1: First team code
            team2: Second team code
            venue: Specific venue filter (optional)
            years_back: Years to look back (defaults to engine setting)
            include_playoffs: Include playoff games
            
        Returns:
            TeamRecord object with head-to-head statistics
        """
        years_back = years_back or self.lookback_years
        cutoff_date = datetime.now() - timedelta(days=years_back * 365)
        
        # Get games between these teams
        relevant_games = [
            game for game in self.game_history
            if ((game['home_team'] == team1 and game['away_team'] == team2) or
                (game['home_team'] == team2 and game['away_team'] == team1))
            and datetime.strptime(game['date'], '%Y-%m-%d') >= cutoff_date
            and (not venue or game.get('venue') == venue)
            and (include_playoffs or not game.get('is_playoff', False))
        ]
        
        record = TeamRecord()
        
        for game in relevant_games:
            record.games_played += 1
            
            # Determine if team1 won
            if ((game['home_team'] == team1 and game['home_score'] > game['away_score']) or
                (game['away_team'] == team1 and game['away_score'] > game['home_score'])):
                record.wins += 1
            else:
                record.losses += 1
            
            # Add run statistics
            if game['home_team'] == team1:
                record.runs_scored += game['home_score']
                record.runs_allowed += game['away_score']
            else:
                record.runs_scored += game['away_score']
                record.runs_allowed += game['home_score']
        
        return record
    
    def pitcher_vs_lineup(
        self,
        pitcher_id: str,
        opponent_team: str,
        venue: Optional[str] = None,
        years_back: int = 3
    ) -> PitcherVsLineup:
        """
        Analyze pitcher performance against specific team's lineup.
        
        Args:
            pitcher_id: Pitcher identifier
            opponent_team: Opposing team code
            venue: Specific venue (optional)
            years_back: Years of data to analyze
            
        Returns:
            PitcherVsLineup object with detailed matchup analysis
        """
        cutoff_date = datetime.now() - timedelta(days=years_back * 365)
        
        # Get pitcher's games against this team
        pitcher_games = [
            game for game in self.game_history
            if game.get('starting_pitcher_home') == pitcher_id and game['away_team'] == opponent_team
            or game.get('starting_pitcher_away') == pitcher_id and game['home_team'] == opponent_team
            if datetime.strptime(game['date'], '%Y-%m-%d') >= cutoff_date
            and (not venue or game.get('venue') == venue)
        ]
        
        matchup = PitcherVsLineup(
            pitcher_id=pitcher_id,
            pitcher_name=f"Pitcher {pitcher_id}",  # Would fetch from database
            opponent_team=opponent_team
        )
        
        # Calculate career record
        record = TeamRecord()
        for game in pitcher_games:
            record.games_played += 1
            
            # Determine if pitcher's team won
            pitcher_home = game.get('starting_pitcher_home') == pitcher_id
            if pitcher_home:
                if game['home_score'] > game['away_score']:
                    record.wins += 1
                else:
                    record.losses += 1
                record.runs_scored += game['home_score']
                record.runs_allowed += game['away_score']
            else:
                if game['away_score'] > game['home_score']:
                    record.wins += 1
                else:
                    record.losses += 1
                record.runs_scored += game['away_score']
                record.runs_allowed += game['home_score']
        
        matchup.career_record = record
        
        # Add recent performance (last 10 starts)
        recent_games = sorted(pitcher_games, key=lambda x: x['date'], reverse=True)[:10]
        matchup.recent_performance = [
            {
                'date': game['date'],
                'result': 'W' if self._pitcher_won_game(game, pitcher_id) else 'L',
                'runs_allowed': self._get_pitcher_runs_allowed(game, pitcher_id),
                'innings': game.get('pitcher_innings', 6.0)
            }
            for game in recent_games
        ]
        
        # Calculate venue splits
        venue_games = defaultdict(list)
        for game in pitcher_games:
            venue_games[game.get('venue', 'Unknown')].append(game)
        
        for venue_name, games in venue_games.items():
            venue_record = TeamRecord()
            for game in games:
                venue_record.games_played += 1
                if self._pitcher_won_game(game, pitcher_id):
                    venue_record.wins += 1
                else:
                    venue_record.losses += 1
            matchup.venue_splits[venue_name] = venue_record
        
        return matchup
    
    def get_situational_splits(
        self,
        team: str,
        situation: GameSituation,
        years_back: int = 2
    ) -> SituationalSplit:
        """
        Get team performance in specific game situations.
        
        Args:
            team: Team code
            situation: Game situation to analyze
            years_back: Years of data to consider
            
        Returns:
            SituationalSplit object with situational performance
        """
        cutoff_date = datetime.now() - timedelta(days=years_back * 365)
        
        # Filter games by situation
        situational_games = []
        for game in self.game_history:
            if datetime.strptime(game['date'], '%Y-%m-%d') < cutoff_date:
                continue
            
            if not (game['home_team'] == team or game['away_team'] == team):
                continue
            
            # Check if game matches the situation
            if self._game_matches_situation(game, team, situation):
                situational_games.append(game)
        
        split = SituationalSplit(situation=situation, team_name=team)
        
        if not situational_games:
            return split
        
        # Calculate record
        record = TeamRecord()
        for game in situational_games:
            record.games_played += 1
            
            team_won = (
                (game['home_team'] == team and game['home_score'] > game['away_score']) or
                (game['away_team'] == team and game['away_score'] > game['home_score'])
            )
            
            if team_won:
                record.wins += 1
            else:
                record.losses += 1
            
            # Add runs
            if game['home_team'] == team:
                record.runs_scored += game['home_score']
                record.runs_allowed += game['away_score']
            else:
                record.runs_scored += game['away_score']
                record.runs_allowed += game['home_score']
        
        split.record = record
        
        # Calculate batting and pitching stats (simplified)
        split.batting_stats = {
            'avg': 0.250 + (record.win_percentage - 0.5) * 0.1,  # Simplified
            'obp': 0.320 + (record.win_percentage - 0.5) * 0.08,
            'slg': 0.400 + (record.win_percentage - 0.5) * 0.12
        }
        
        split.pitching_stats = {
            'era': 4.50 - (record.win_percentage - 0.5) * 2.0,
            'whip': 1.35 - (record.win_percentage - 0.5) * 0.4,
            'k_per_9': 8.0 + (record.win_percentage - 0.5) * 2.0
        }
        
        # Calculate clutch performance factor
        if situation in [GameSituation.BEHIND_LATE, GameSituation.TIED_LATE, GameSituation.SCORING_POSITION]:
            split.clutch_performance = max(0.5, min(1.5, record.win_percentage * 2))
        
        return split
    
    def analyze_division_rivalry(
        self,
        team1: str,
        team2: str
    ) -> DivisionRivalry:
        """
        Analyze division rivalry between two teams.
        
        Args:
            team1: First team code
            team2: Second team code
            
        Returns:
            DivisionRivalry object with rivalry analysis
        """
        # Check if teams are division rivals
        is_rival = False
        for division_teams in self.divisions.values():
            if team1 in division_teams and team2 in division_teams:
                is_rival = True
                break
        
        rivalry = DivisionRivalry(
            team1=team1,
            team2=team2,
            is_division_rival=is_rival
        )
        
        # Get all-time record
        rivalry.all_time_record = self.get_h2h_record(team1, team2, years_back=10)
        
        # Get last season record
        rivalry.last_season_record = self.get_h2h_record(team1, team2, years_back=1)
        
        # Calculate rivalry intensity
        total_games = rivalry.all_time_record.games_played
        if total_games > 0:
            # More games played = higher intensity
            intensity_base = min(2.0, total_games / 50.0)
            
            # Close record = higher intensity
            win_pct = rivalry.all_time_record.win_percentage
            competitiveness = 1.0 - abs(win_pct - 0.5) * 2  # Higher when close to .500
            
            rivalry.rivalry_intensity = intensity_base * (0.5 + competitiveness * 0.5)
        
        return rivalry
    
    def get_day_of_week_performance(
        self,
        team: str,
        day_of_week: str,
        years_back: int = 3
    ) -> DayOfWeekPattern:
        """
        Analyze team performance patterns by day of week.
        
        Args:
            team: Team code
            day_of_week: Day of week ("Monday", "Tuesday", etc.)
            years_back: Years of data to consider
            
        Returns:
            DayOfWeekPattern object
        """
        if team in self.day_patterns:
            return self.day_patterns[team]
        
        cutoff_date = datetime.now() - timedelta(days=years_back * 365)
        
        # Group games by day of week
        day_games = defaultdict(list)
        for game in self.game_history:
            if datetime.strptime(game['date'], '%Y-%m-%d') < cutoff_date:
                continue
            
            if not (game['home_team'] == team or game['away_team'] == team):
                continue
            
            game_day = datetime.strptime(game['date'], '%Y-%m-%d').strftime('%A')
            day_games[game_day].append(game)
        
        pattern = DayOfWeekPattern(team_name=team)
        
        # Calculate record for each day
        for day, games in day_games.items():
            record = TeamRecord()
            for game in games:
                record.games_played += 1
                
                team_won = (
                    (game['home_team'] == team and game['home_score'] > game['away_score']) or
                    (game['away_team'] == team and game['away_score'] > game['home_score'])
                )
                
                if team_won:
                    record.wins += 1
                else:
                    record.losses += 1
                
                # Add runs
                if game['home_team'] == team:
                    record.runs_scored += game['home_score']
                    record.runs_allowed += game['away_score']
                else:
                    record.runs_scored += game['away_score']
                    record.runs_allowed += game['home_score']
            
            pattern.day_records[day] = record
        
        # Analyze travel fatigue impact (simplified)
        for day in pattern.day_records:
            # Teams typically perform worse on Mondays (travel day) and getaway days
            if day == 'Monday':
                pattern.travel_fatigue_impact[day] = -0.02  # 2% penalty
            elif day == 'Thursday':
                pattern.travel_fatigue_impact[day] = -0.01  # 1% penalty  
            else:
                pattern.travel_fatigue_impact[day] = 0.0
        
        self.day_patterns[team] = pattern
        return pattern
    
    def generate_comprehensive_matchup(
        self,
        team1: str,
        team2: str,
        venue: str,
        game_date: datetime,
        starting_pitcher1: Optional[str] = None,
        starting_pitcher2: Optional[str] = None
    ) -> HistoricalMatchup:
        """
        Generate comprehensive historical matchup analysis.
        
        Args:
            team1: First team (home team)
            team2: Second team (away team)
            venue: Game venue
            game_date: Date of the game
            starting_pitcher1: Team1's starting pitcher
            starting_pitcher2: Team2's starting pitcher
            
        Returns:
            HistoricalMatchup object with complete analysis
        """
        # Get head-to-head record
        h2h_record = self.get_h2h_record(team1, team2, venue)
        
        # Calculate venue advantage
        team1_home_record = self.get_h2h_record(team1, team2, venue)
        team1_all_record = self.get_h2h_record(team1, team2)
        
        if team1_all_record.games_played > 0:
            venue_advantage = (team1_home_record.win_percentage - 
                             team1_all_record.win_percentage)
        else:
            venue_advantage = 0.0
        
        # Determine recent trend
        recent_games = self._get_recent_h2h_games(team1, team2, 10)
        recent_trend = self._analyze_recent_trend(recent_games, team1, team2)
        
        # Get pitcher matchups
        pitcher_matchups = []
        if starting_pitcher1:
            pitcher_matchups.append(
                self.pitcher_vs_lineup(starting_pitcher1, team2, venue)
            )
        if starting_pitcher2:
            pitcher_matchups.append(
                self.pitcher_vs_lineup(starting_pitcher2, team1, venue)
            )
        
        # Analyze division rivalry
        division_rivalry = self.analyze_division_rivalry(team1, team2)
        
        # Get situational splits
        key_situations = [
            GameSituation.AHEAD_LATE,
            GameSituation.BEHIND_LATE,
            GameSituation.SCORING_POSITION,
            GameSituation.BASES_LOADED
        ]
        
        situational_splits = []
        for situation in key_situations:
            split1 = self.get_situational_splits(team1, situation)
            split2 = self.get_situational_splits(team2, situation)
            situational_splits.extend([split1, split2])
        
        # Get day of week patterns
        game_day = game_date.strftime('%A')
        day_pattern1 = self.get_day_of_week_performance(team1, game_day)
        day_pattern2 = self.get_day_of_week_performance(team2, game_day)
        day_patterns = [day_pattern1, day_pattern2]
        
        # Calculate statistical significance
        sample_size = h2h_record.games_played
        statistical_significance = sample_size >= self.min_sample_size
        
        # Calculate confidence interval for win probability
        if sample_size > 0:
            win_rate = h2h_record.win_percentage
            std_error = np.sqrt(win_rate * (1 - win_rate) / sample_size)
            ci_lower = max(0, win_rate - 1.96 * std_error)
            ci_upper = min(1, win_rate + 1.96 * std_error)
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = (0.0, 1.0)
        
        # Identify key advantages
        key_advantages = self._identify_key_advantages(
            team1, team2, h2h_record, venue_advantage, division_rivalry,
            situational_splits, day_patterns, pitcher_matchups
        )
        
        # Calculate predicted impact
        predicted_impact = self._calculate_predicted_impact(
            h2h_record, venue_advantage, division_rivalry, situational_splits,
            day_patterns, pitcher_matchups
        )
        
        return HistoricalMatchup(
            team1=team1,
            team2=team2,
            venue=venue,
            analysis_date=datetime.now(),
            h2h_record=h2h_record,
            venue_advantage=venue_advantage,
            recent_trend=recent_trend,
            pitcher_matchups=pitcher_matchups,
            division_rivalry=division_rivalry,
            situational_splits=situational_splits,
            day_patterns=day_patterns,
            sample_size=sample_size,
            confidence_interval=confidence_interval,
            statistical_significance=statistical_significance,
            key_advantages=key_advantages,
            predicted_impact=predicted_impact
        )
    
    def _load_historical_data(self, path: str):
        """Load historical game data from file."""
        try:
            with open(path, 'r') as f:
                self.game_history = json.load(f)
            logger.info(f"Loaded {len(self.game_history)} historical games")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            # Generate sample data for demonstration
            self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample historical data for demonstration."""
        teams = ['NYY', 'BOS', 'TB', 'TOR', 'BAL', 'HOU', 'LAA', 'OAK', 'SEA', 'TEX']
        venues = ['Yankee Stadium', 'Fenway Park', 'Tropicana Field', 'Rogers Centre', 'Oriole Park']
        
        sample_games = []
        for i in range(1000):  # Generate 1000 sample games
            date = (datetime.now() - timedelta(days=np.random.randint(1, 1800))).strftime('%Y-%m-%d')
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            home_score = np.random.poisson(4.5)
            away_score = np.random.poisson(4.2)
            
            game = {
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'venue': np.random.choice(venues),
                'starting_pitcher_home': f"{home_team}_pitcher_{np.random.randint(1, 6)}",
                'starting_pitcher_away': f"{away_team}_pitcher_{np.random.randint(1, 6)}",
                'is_playoff': np.random.random() < 0.05  # 5% playoff games
            }
            sample_games.append(game)
        
        self.game_history = sample_games
        logger.info("Generated sample historical data")
    
    def _process_historical_data(self):
        """Process historical data to build internal structures."""
        logger.info("Processing historical data...")
        
        for game in self.game_history:
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Update team records
            if home_team not in self.team_records:
                self.team_records[home_team] = defaultdict(TeamRecord)
            if away_team not in self.team_records:
                self.team_records[away_team] = defaultdict(TeamRecord)
    
    def _game_matches_situation(self, game: Dict, team: str, situation: GameSituation) -> bool:
        """Check if a game matches the specified situation."""
        # Simplified implementation - in practice would need detailed play-by-play data
        if situation == GameSituation.AHEAD_LATE:
            return (game['home_team'] == team and game['home_score'] > game['away_score']) or \
                   (game['away_team'] == team and game['away_score'] > game['home_score'])
        elif situation == GameSituation.BEHIND_LATE:
            return (game['home_team'] == team and game['home_score'] < game['away_score']) or \
                   (game['away_team'] == team and game['away_score'] < game['home_score'])
        elif situation == GameSituation.SCORING_POSITION:
            return True  # Simplified - would need play-by-play data
        else:
            return True  # Default to include game
    
    def _pitcher_won_game(self, game: Dict, pitcher_id: str) -> bool:
        """Determine if pitcher won the game."""
        pitcher_home = game.get('starting_pitcher_home') == pitcher_id
        if pitcher_home:
            return game['home_score'] > game['away_score']
        else:
            return game['away_score'] > game['home_score']
    
    def _get_pitcher_runs_allowed(self, game: Dict, pitcher_id: str) -> int:
        """Get runs allowed by pitcher (simplified)."""
        # In practice would need detailed pitching stats
        return np.random.randint(0, 6)  # Placeholder
    
    def _get_recent_h2h_games(self, team1: str, team2: str, num_games: int) -> List[Dict]:
        """Get recent head-to-head games."""
        h2h_games = [
            game for game in self.game_history
            if ((game['home_team'] == team1 and game['away_team'] == team2) or
                (game['home_team'] == team2 and game['away_team'] == team1))
        ]
        
        # Sort by date descending and take most recent
        sorted_games = sorted(h2h_games, key=lambda x: x['date'], reverse=True)
        return sorted_games[:num_games]
    
    def _analyze_recent_trend(self, recent_games: List[Dict], team1: str, team2: str) -> str:
        """Analyze recent trend between teams."""
        if len(recent_games) < 5:
            return "neutral"
        
        team1_wins = sum(1 for game in recent_games[:5] 
                        if (game['home_team'] == team1 and game['home_score'] > game['away_score']) or
                           (game['away_team'] == team1 and game['away_score'] > game['home_score']))
        
        if team1_wins >= 4:
            return "team1_hot"
        elif team1_wins <= 1:
            return "team2_hot"
        else:
            return "neutral"
    
    def _identify_key_advantages(
        self,
        team1: str,
        team2: str,
        h2h_record: TeamRecord,
        venue_advantage: float,
        division_rivalry: DivisionRivalry,
        situational_splits: List[SituationalSplit],
        day_patterns: List[DayOfWeekPattern],
        pitcher_matchups: List[PitcherVsLineup]
    ) -> List[str]:
        """Identify key advantages for each team."""
        advantages = []
        
        # Head-to-head advantage
        if h2h_record.win_percentage > 0.6:
            advantages.append(f"{team1} dominates historically ({h2h_record.win_percentage:.1%})")
        elif h2h_record.win_percentage < 0.4:
            advantages.append(f"{team2} dominates historically ({1-h2h_record.win_percentage:.1%})")
        
        # Venue advantage
        if venue_advantage > 0.1:
            advantages.append(f"{team1} strong home venue advantage (+{venue_advantage:.1%})")
        elif venue_advantage < -0.1:
            advantages.append(f"{team2} performs well at this venue (+{-venue_advantage:.1%})")
        
        # Division rivalry
        if division_rivalry.is_division_rival and division_rivalry.rivalry_intensity > 1.5:
            advantages.append("Intense division rivalry - expect close game")
        
        # Pitcher advantages
        for matchup in pitcher_matchups:
            if matchup.career_record.games_played >= 5:
                if matchup.career_record.win_percentage > 0.7:
                    advantages.append(f"Pitcher {matchup.pitcher_name} dominates this matchup")
                elif matchup.career_record.win_percentage < 0.3:
                    advantages.append(f"Pitcher {matchup.pitcher_name} struggles vs this team")
        
        return advantages[:5]  # Return top 5 advantages
    
    def _calculate_predicted_impact(
        self,
        h2h_record: TeamRecord,
        venue_advantage: float,
        division_rivalry: DivisionRivalry,
        situational_splits: List[SituationalSplit],
        day_patterns: List[DayOfWeekPattern],
        pitcher_matchups: List[PitcherVsLineup]
    ) -> float:
        """Calculate predicted impact score (-1 to 1)."""
        impact = 0.0
        
        # Head-to-head impact (40% weight)
        if h2h_record.games_played > 0:
            h2h_impact = (h2h_record.win_percentage - 0.5) * 2  # Scale to -1 to 1
            impact += h2h_impact * 0.4
        
        # Venue impact (20% weight)
        impact += venue_advantage * 0.2
        
        # Division rivalry impact (10% weight)
        if division_rivalry.is_division_rival:
            rivalry_impact = min(0.1, division_rivalry.rivalry_intensity * 0.05)
            impact += rivalry_impact * 0.1
        
        # Pitcher impact (20% weight)
        pitcher_impact = 0.0
        if pitcher_matchups:
            for matchup in pitcher_matchups:
                if matchup.career_record.games_played > 0:
                    pitcher_impact += (matchup.career_record.win_percentage - 0.5) * 2
            pitcher_impact /= len(pitcher_matchups)
            impact += pitcher_impact * 0.2
        
        # Day pattern impact (10% weight)
        day_impact = 0.0
        for pattern in day_patterns:
            for day, record in pattern.day_records.items():
                if record.games_played > 0:
                    day_impact += (record.win_percentage - 0.5) * 2
                    break
        if len(day_patterns) > 0:
            day_impact /= len(day_patterns)
            impact += day_impact * 0.1
        
        # Clamp to -1 to 1 range
        return max(-1.0, min(1.0, impact))


def main():
    """Example usage of the HistoricalMatchupEngine."""
    
    # Initialize engine
    engine = HistoricalMatchupEngine()
    
    # Example matchup analysis
    team1, team2 = "NYY", "BOS"
    venue = "Yankee Stadium"
    game_date = datetime(2026, 7, 15)
    
    print(f"Historical Matchup Analysis: {team1} vs {team2}")
    print("=" * 50)
    
    # Head-to-head record
    h2h_record = engine.get_h2h_record(team1, team2, venue)
    print(f"H2H at {venue}: {h2h_record.wins}-{h2h_record.losses} "
          f"({h2h_record.win_percentage:.1%})")
    print(f"Avg runs: {h2h_record.avg_runs_scored:.1f} - {h2h_record.avg_runs_allowed:.1f}")
    print()
    
    # Pitcher vs lineup
    pitcher_matchup = engine.pitcher_vs_lineup("NYY_pitcher_1", team2, venue)
    print(f"Pitcher vs {team2}: {pitcher_matchup.career_record.wins}-{pitcher_matchup.career_record.losses}")
    print(f"Recent form: {len(pitcher_matchup.recent_performance)} starts")
    print()
    
    # Situational splits
    clutch_split = engine.get_situational_splits(team1, GameSituation.SCORING_POSITION)
    print(f"{team1} w/ RISP: {clutch_split.record.wins}-{clutch_split.record.losses} "
          f"({clutch_split.record.win_percentage:.1%})")
    print()
    
    # Division rivalry
    rivalry = engine.analyze_division_rivalry(team1, team2)
    print(f"Division rivals: {rivalry.is_division_rival}")
    print(f"Rivalry intensity: {rivalry.rivalry_intensity:.2f}")
    print()
    
    # Day of week performance
    day_pattern = engine.get_day_of_week_performance(team1, "Monday")
    monday_record = day_pattern.get_day_performance("Monday")
    print(f"{team1} on Mondays: {monday_record.wins}-{monday_record.losses} "
          f"({monday_record.win_percentage:.1%})")
    print()
    
    # Comprehensive analysis
    matchup = engine.generate_comprehensive_matchup(
        team1, team2, venue, game_date, "NYY_pitcher_1", "BOS_pitcher_1"
    )
    
    print("Comprehensive Analysis:")
    print(f"Predicted impact: {matchup.predicted_impact:.2f}")
    print(f"Statistical significance: {matchup.statistical_significance}")
    print(f"Sample size: {matchup.sample_size} games")
    print()
    
    print("Key advantages:")
    for advantage in matchup.key_advantages:
        print(f"  • {advantage}")


if __name__ == "__main__":
    main()