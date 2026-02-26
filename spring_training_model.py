#!/usr/bin/env python3
"""
Spring Training Model for MLB Predictions

Advanced spring training analysis system that adjusts regular season predictions
based on spring performance, roster battles, velocity changes, new pitches,
injury recovery, and team chemistry dynamics.

Author: MLB Predictor Team
Version: 2.0
License: MIT
"""

import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import requests
import math
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spring_training.log'),
        logging.StreamHandler()
    ]
)

class PlayerStatus(Enum):
    """Player roster status"""
    ESTABLISHED = "Established"
    COMPETING = "Competing"
    BUBBLE = "Bubble"
    PROSPECT = "Prospect"
    VETERAN_TRYOUT = "Veteran_Tryout"
    INJURED = "Injured"

class InjuryStatus(Enum):
    """Injury recovery status"""
    HEALTHY = "Healthy"
    RECOVERING = "Recovering"
    LIMITED = "Limited"
    SETBACK = "Setback"
    UNKNOWN = "Unknown"

class Position(Enum):
    """Baseball positions"""
    P = "Pitcher"
    C = "Catcher"
    FB = "First_Base"
    SB = "Second_Base"
    TB = "Third_Base"
    SS = "Shortstop"
    LF = "Left_Field"
    CF = "Center_Field"
    RF = "Right_Field"
    DH = "Designated_Hitter"

@dataclass
class VelocityTracking:
    """Pitcher velocity tracking data"""
    player_id: str
    date: datetime
    pitch_type: str
    velocity: float
    spin_rate: int
    movement_horizontal: float
    movement_vertical: float
    location_x: float
    location_z: float
    usage_rate: float
    whiff_rate: float
    called_strike_rate: float
    
    @property
    def velocity_percentile(self) -> float:
        """Calculate velocity percentile for pitch type"""
        # Mock percentile calculation - in real implementation would use league data
        league_averages = {
            'Fastball': 93.5, 'Sinker': 92.8, 'Cutter': 88.2,
            'Slider': 84.1, 'Curveball': 78.9, 'Changeup': 84.7,
            'Splitter': 85.6, 'Knuckleball': 68.2
        }
        
        avg = league_averages.get(self.pitch_type, 90.0)
        return max(0, min(100, 50 + (self.velocity - avg) * 3))

@dataclass  
class SpringStats:
    """Spring training statistics"""
    player_id: str
    games: int = 0
    at_bats: int = 0
    hits: int = 0
    doubles: int = 0
    triples: int = 0
    home_runs: int = 0
    rbis: int = 0
    walks: int = 0
    strikeouts: int = 0
    stolen_bases: int = 0
    caught_stealing: int = 0
    innings_pitched: float = 0.0
    earned_runs: int = 0
    wins: int = 0
    losses: int = 0
    saves: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    strikeouts_pitched: int = 0
    
    @property
    def batting_average(self) -> float:
        return self.hits / self.at_bats if self.at_bats > 0 else 0.0
    
    @property
    def on_base_percentage(self) -> float:
        total_pa = self.at_bats + self.walks
        return (self.hits + self.walks) / total_pa if total_pa > 0 else 0.0
    
    @property
    def slugging_percentage(self) -> float:
        if self.at_bats == 0:
            return 0.0
        total_bases = (self.hits - self.doubles - self.triples - self.home_runs + 
                      self.doubles * 2 + self.triples * 3 + self.home_runs * 4)
        return total_bases / self.at_bats
    
    @property
    def era(self) -> float:
        return (self.earned_runs * 9) / self.innings_pitched if self.innings_pitched > 0 else 0.0
    
    @property
    def whip(self) -> float:
        return (self.hits_allowed + self.walks_allowed) / self.innings_pitched if self.innings_pitched > 0 else 0.0

@dataclass
class RosterBattle:
    """Roster battle information"""
    position: str
    team: str
    candidates: List[str]
    favorite: str
    competition_level: float  # 0-1, higher = more competitive
    decision_timeline: str
    factors: List[str]
    current_leader: str
    probability_distribution: Dict[str, float]

@dataclass
class SpringPlayer:
    """Spring training player data"""
    player_id: str
    name: str
    team: str
    primary_position: str
    eligible_positions: List[str]
    age: int
    experience_years: int
    status: PlayerStatus
    injury_status: InjuryStatus
    spring_stats: SpringStats
    velocity_data: List[VelocityTracking] = field(default_factory=list)
    roster_battle: Optional[RosterBattle] = None
    contract_status: str = ""
    minor_league_options: int = 0
    arbitration_eligible: bool = False
    spring_notes: List[str] = field(default_factory=list)
    acquisition_date: Optional[datetime] = None
    previous_team: Optional[str] = None

class SpringTrainingModel:
    """
    Comprehensive Spring Training Analysis Model
    
    Features:
    - Velocity tracking and trend analysis
    - New pitch detection and development
    - Roster battle prediction and probability
    - Lineup projection based on spring performance
    - Injury recovery assessment and timeline
    - Team chemistry analysis for new acquisitions
    - Minor league call-up probability modeling
    - Spring-to-regular season correlation modeling
    - Breakout and regression candidate detection
    """
    
    def __init__(self):
        """Initialize the spring training model"""
        self.players: Dict[str, SpringPlayer] = {}
        self.velocity_baselines: Dict[str, Dict[str, float]] = {}
        self.roster_battles: List[RosterBattle] = []
        self.team_acquisitions: Dict[str, List[str]] = defaultdict(list)
        self.injury_timelines: Dict[str, Dict] = {}
        self.correlation_factors: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load historical correlation data
        self._load_correlation_factors()
        self._initialize_velocity_baselines()

    def _load_correlation_factors(self) -> None:
        """Load spring-to-regular season correlation factors"""
        # Default correlation factors based on historical analysis
        self.correlation_factors = {
            'batting_average': 0.35,
            'on_base_percentage': 0.42,
            'slugging_percentage': 0.38,
            'home_runs': 0.28,
            'strikeout_rate': 0.55,
            'walk_rate': 0.48,
            'era': 0.25,
            'whip': 0.31,
            'strikeout_rate_pitcher': 0.52,
            'walk_rate_pitcher': 0.44,
            'velocity_fastball': 0.85,
            'velocity_secondary': 0.78,
            'new_pitch_success': 0.42
        }

    def _initialize_velocity_baselines(self) -> None:
        """Initialize velocity baseline data from previous seasons"""
        # Mock baseline data - in real implementation would load from database
        sample_players = ['player_001', 'player_002', 'player_003']
        
        for player_id in sample_players:
            self.velocity_baselines[player_id] = {
                'Fastball': random.uniform(90.0, 98.0),
                'Slider': random.uniform(82.0, 88.0),
                'Changeup': random.uniform(82.0, 87.0),
                'Curveball': random.uniform(75.0, 82.0),
                'Cutter': random.uniform(86.0, 92.0)
            }

    def add_player(self, player: SpringPlayer) -> None:
        """Add a player to the spring training model"""
        self.players[player.player_id] = player
        self.logger.info(f"Added player {player.name} ({player.team}) to spring model")

    def load_spring_data(self, data_source: str = 'api') -> None:
        """Load spring training data from various sources"""
        if data_source == 'api':
            self._load_from_api()
        elif data_source == 'csv':
            self._load_from_csv('spring_data.csv')
        else:
            self._generate_mock_data()
            
        self.logger.info(f"Loaded {len(self.players)} players from {data_source}")

    def _load_from_api(self) -> None:
        """Load spring training data from API"""
        # Mock API call - in real implementation would call actual API
        mock_response = {
            'players': [
                {
                    'id': f'player_{i:03d}',
                    'name': f'Player {i}',
                    'team': random.choice(['NYY', 'BOS', 'TB', 'TOR', 'BAL']),
                    'position': random.choice(['P', 'C', '1B', '2B', '3B', 'SS', 'OF']),
                    'age': random.randint(20, 35),
                    'experience': random.randint(0, 12),
                    'spring_stats': self._generate_mock_spring_stats()
                }
                for i in range(50)
            ]
        }
        
        for player_data in mock_response['players']:
            player = SpringPlayer(
                player_id=player_data['id'],
                name=player_data['name'],
                team=player_data['team'],
                primary_position=player_data['position'],
                eligible_positions=[player_data['position']],
                age=player_data['age'],
                experience_years=player_data['experience'],
                status=random.choice(list(PlayerStatus)),
                injury_status=random.choice(list(InjuryStatus)),
                spring_stats=SpringStats(**player_data['spring_stats'])
            )
            self.add_player(player)

    def _load_from_csv(self, filename: str) -> None:
        """Load spring training data from CSV file"""
        try:
            df = pd.read_csv(filename)
            
            for _, row in df.iterrows():
                spring_stats = SpringStats(
                    player_id=row['player_id'],
                    games=row.get('games', 0),
                    at_bats=row.get('at_bats', 0),
                    hits=row.get('hits', 0),
                    home_runs=row.get('home_runs', 0),
                    rbis=row.get('rbis', 0),
                    walks=row.get('walks', 0),
                    strikeouts=row.get('strikeouts', 0)
                )
                
                player = SpringPlayer(
                    player_id=row['player_id'],
                    name=row['name'],
                    team=row['team'],
                    primary_position=row['position'],
                    eligible_positions=[row['position']],
                    age=int(row['age']),
                    experience_years=int(row.get('experience', 0)),
                    status=PlayerStatus(row.get('status', 'Established')),
                    injury_status=InjuryStatus(row.get('injury_status', 'Healthy')),
                    spring_stats=spring_stats
                )
                self.add_player(player)
                
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            self._generate_mock_data()

    def _generate_mock_data(self) -> None:
        """Generate mock spring training data for testing"""
        teams = ['NYY', 'BOS', 'TB', 'TOR', 'BAL', 'HOU', 'SEA', 'TEX', 'LAA', 'OAK']
        positions = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']
        
        for i in range(100):
            spring_stats = SpringStats(
                player_id=f'spring_player_{i:03d}',
                **self._generate_mock_spring_stats()
            )
            
            player = SpringPlayer(
                player_id=f'spring_player_{i:03d}',
                name=f'Spring Player {i}',
                team=random.choice(teams),
                primary_position=random.choice(positions),
                eligible_positions=[random.choice(positions)],
                age=random.randint(19, 38),
                experience_years=random.randint(0, 15),
                status=random.choice(list(PlayerStatus)),
                injury_status=random.choice(list(InjuryStatus)),
                spring_stats=spring_stats,
                contract_status=random.choice(['Rookie', 'Pre-Arb', 'Arb-Eligible', 'Free Agent']),
                minor_league_options=random.randint(0, 3)
            )
            
            # Add velocity data for pitchers
            if player.primary_position == 'P':
                for _ in range(random.randint(5, 15)):
                    velocity_data = VelocityTracking(
                        player_id=player.player_id,
                        date=datetime.now() - timedelta(days=random.randint(1, 30)),
                        pitch_type=random.choice(['Fastball', 'Slider', 'Changeup', 'Curveball', 'Cutter']),
                        velocity=random.uniform(85.0, 100.0),
                        spin_rate=random.randint(2000, 3000),
                        movement_horizontal=random.uniform(-15.0, 15.0),
                        movement_vertical=random.uniform(-15.0, 15.0),
                        location_x=random.uniform(-2.0, 2.0),
                        location_z=random.uniform(1.0, 4.0),
                        usage_rate=random.uniform(0.1, 0.4),
                        whiff_rate=random.uniform(0.1, 0.4),
                        called_strike_rate=random.uniform(0.1, 0.3)
                    )
                    player.velocity_data.append(velocity_data)
            
            self.add_player(player)

    def _generate_mock_spring_stats(self) -> Dict[str, int]:
        """Generate mock spring training statistics"""
        is_pitcher = random.choice([True, False])
        
        if is_pitcher:
            innings = random.uniform(5.0, 25.0)
            return {
                'games': random.randint(3, 12),
                'innings_pitched': round(innings, 1),
                'earned_runs': random.randint(0, int(innings * 0.8)),
                'hits_allowed': random.randint(int(innings * 0.5), int(innings * 1.5)),
                'walks_allowed': random.randint(0, int(innings * 0.5)),
                'strikeouts_pitched': random.randint(int(innings * 0.5), int(innings * 1.2)),
                'wins': random.randint(0, 3),
                'losses': random.randint(0, 2),
                'saves': random.randint(0, 2)
            }
        else:
            at_bats = random.randint(15, 60)
            hits = random.randint(0, int(at_bats * 0.4))
            return {
                'games': random.randint(8, 20),
                'at_bats': at_bats,
                'hits': hits,
                'doubles': random.randint(0, max(1, hits // 4)),
                'triples': random.randint(0, max(1, hits // 10)),
                'home_runs': random.randint(0, max(1, hits // 5)),
                'rbis': random.randint(0, hits + 5),
                'walks': random.randint(0, at_bats // 4),
                'strikeouts': random.randint(0, at_bats // 2),
                'stolen_bases': random.randint(0, 5),
                'caught_stealing': random.randint(0, 2)
            }

    def track_velocity_changes(self, player_id: str) -> Dict[str, Any]:
        """Track velocity changes from previous season to spring training"""
        if player_id not in self.players:
            return {'error': 'Player not found'}
            
        player = self.players[player_id]
        if not player.velocity_data:
            return {'error': 'No velocity data available'}
            
        velocity_analysis = {
            'player_id': player_id,
            'name': player.name,
            'team': player.team,
            'pitch_types': {},
            'overall_change': 0.0,
            'significant_changes': [],
            'new_pitches': [],
            'trends': {}
        }
        
        # Group velocity data by pitch type
        pitch_velocities = defaultdict(list)
        for vdata in player.velocity_data:
            pitch_velocities[vdata.pitch_type].append(vdata.velocity)
        
        # Compare to baselines
        baseline_data = self.velocity_baselines.get(player_id, {})
        total_change = 0.0
        significant_changes = []
        
        for pitch_type, velocities in pitch_velocities.items():
            current_avg = np.mean(velocities)
            current_max = max(velocities)
            baseline_avg = baseline_data.get(pitch_type, current_avg)
            
            change = current_avg - baseline_avg
            change_pct = (change / baseline_avg * 100) if baseline_avg > 0 else 0
            
            velocity_analysis['pitch_types'][pitch_type] = {
                'current_avg': round(current_avg, 1),
                'current_max': round(current_max, 1),
                'baseline_avg': round(baseline_avg, 1),
                'change_mph': round(change, 1),
                'change_percent': round(change_pct, 1),
                'readings': len(velocities)
            }
            
            total_change += abs(change)
            
            # Flag significant changes (> 1.5 mph)
            if abs(change) > 1.5:
                significance = "Major" if abs(change) > 3.0 else "Moderate"
                direction = "increase" if change > 0 else "decrease"
                
                significant_changes.append({
                    'pitch_type': pitch_type,
                    'change': round(change, 1),
                    'significance': significance,
                    'direction': direction,
                    'implications': self._interpret_velocity_change(pitch_type, change)
                })
        
        velocity_analysis['overall_change'] = round(total_change / len(pitch_velocities), 1)
        velocity_analysis['significant_changes'] = significant_changes
        
        # Check for new pitches (not in baseline data)
        new_pitches = []
        for pitch_type in pitch_velocities:
            if pitch_type not in baseline_data:
                new_pitches.append({
                    'pitch_type': pitch_type,
                    'avg_velocity': round(np.mean(pitch_velocities[pitch_type]), 1),
                    'usage_estimate': self._estimate_pitch_usage(player_id, pitch_type),
                    'development_stage': self._assess_pitch_development(player_id, pitch_type)
                })
        
        velocity_analysis['new_pitches'] = new_pitches
        
        return velocity_analysis

    def _interpret_velocity_change(self, pitch_type: str, change: float) -> List[str]:
        """Interpret the implications of velocity changes"""
        implications = []
        
        if change > 2.0:  # Significant increase
            implications.extend([
                "Improved conditioning or mechanics",
                "Potential for increased strikeout rate",
                "Watch for command issues with added velocity"
            ])
            if pitch_type == 'Fastball':
                implications.append("Could elevate overall stuff ratings")
        elif change < -2.0:  # Significant decrease
            implications.extend([
                "Possible injury concern or fatigue",
                "May impact strikeout ability",
                "Could indicate mechanical adjustment"
            ])
            if pitch_type == 'Fastball':
                implications.append("Monitor closely for underlying issues")
        elif -2.0 <= change <= 2.0:  # Normal variation
            implications.append("Within normal spring training variation")
            
        return implications

    def _estimate_pitch_usage(self, player_id: str, pitch_type: str) -> float:
        """Estimate usage rate for a pitch type"""
        player = self.players[player_id]
        pitch_data = [v for v in player.velocity_data if v.pitch_type == pitch_type]
        
        if pitch_data:
            return np.mean([p.usage_rate for p in pitch_data])
        return 0.0

    def _assess_pitch_development(self, player_id: str, pitch_type: str) -> str:
        """Assess the development stage of a new pitch"""
        player = self.players[player_id]
        pitch_data = [v for v in player.velocity_data if v.pitch_type == pitch_type]
        
        if not pitch_data:
            return "Unknown"
            
        avg_whiff_rate = np.mean([p.whiff_rate for p in pitch_data])
        avg_called_strike_rate = np.mean([p.called_strike_rate for p in pitch_data])
        consistency = 1.0 - np.std([p.velocity for p in pitch_data]) / np.mean([p.velocity for p in pitch_data])
        
        overall_effectiveness = (avg_whiff_rate + avg_called_strike_rate + consistency) / 3
        
        if overall_effectiveness > 0.6:
            return "Advanced"
        elif overall_effectiveness > 0.4:
            return "Developing"
        else:
            return "Experimental"

    def detect_new_pitches(self) -> Dict[str, List[Dict]]:
        """Detect players who have added new pitches"""
        new_pitch_report = {}
        
        for player_id, player in self.players.items():
            if player.primary_position != 'P':
                continue
                
            velocity_analysis = self.track_velocity_changes(player_id)
            new_pitches = velocity_analysis.get('new_pitches', [])
            
            if new_pitches:
                new_pitch_report[player.name] = [{
                    'team': player.team,
                    'experience': player.experience_years,
                    'new_pitches': new_pitches,
                    'impact_rating': self._rate_new_pitch_impact(new_pitches),
                    'fantasy_implications': self._assess_fantasy_impact(player, new_pitches)
                }]
        
        return new_pitch_report

    def _rate_new_pitch_impact(self, new_pitches: List[Dict]) -> str:
        """Rate the potential impact of new pitches"""
        if not new_pitches:
            return "None"
            
        high_impact_pitches = ['Slider', 'Changeup', 'Cutter']
        advanced_pitches = [p for p in new_pitches if p['development_stage'] == 'Advanced']
        high_impact = [p for p in new_pitches if p['pitch_type'] in high_impact_pitches]
        
        if advanced_pitches and high_impact:
            return "High"
        elif advanced_pitches or len(new_pitches) > 1:
            return "Moderate"
        else:
            return "Low"

    def _assess_fantasy_impact(self, player: SpringPlayer, new_pitches: List[Dict]) -> List[str]:
        """Assess fantasy baseball implications of new pitches"""
        implications = []
        
        for pitch in new_pitches:
            if pitch['development_stage'] == 'Advanced':
                if pitch['pitch_type'] in ['Slider', 'Changeup']:
                    implications.append(f"New {pitch['pitch_type']} could boost strikeout rate significantly")
                elif pitch['pitch_type'] == 'Cutter':
                    implications.append("New cutter may improve command and reduce hard contact")
                    
            elif pitch['development_stage'] == 'Developing':
                implications.append(f"{pitch['pitch_type']} in development - monitor for mid-season breakthrough")
                
        if not implications:
            implications.append("New pitch still experimental - limited immediate impact expected")
            
        return implications

    def predict_roster_battles(self) -> List[Dict[str, Any]]:
        """Predict outcomes of spring training roster battles"""
        roster_battles = []
        
        # Group players by team and position
        team_positions = defaultdict(lambda: defaultdict(list))
        for player in self.players.values():
            if player.status in [PlayerStatus.COMPETING, PlayerStatus.BUBBLE]:
                team_positions[player.team][player.primary_position].append(player)
        
        for team, positions in team_positions.items():
            for position, candidates in positions.items():
                if len(candidates) > 1:  # Only battles with multiple candidates
                    battle = self._analyze_roster_battle(team, position, candidates)
                    if battle:
                        roster_battles.append(battle)
        
        return sorted(roster_battles, key=lambda x: x['competition_level'], reverse=True)

    def _analyze_roster_battle(self, team: str, position: str, candidates: List[SpringPlayer]) -> Optional[Dict[str, Any]]:
        """Analyze a specific roster battle"""
        if len(candidates) < 2:
            return None
            
        battle_analysis = {
            'team': team,
            'position': position,
            'candidates': [],
            'favorite': '',
            'competition_level': 0.0,
            'key_factors': [],
            'timeline': 'Late March',
            'confidence': 0.0
        }
        
        candidate_scores = []
        
        for candidate in candidates:
            score_factors = {
                'spring_performance': self._evaluate_spring_performance(candidate),
                'experience': min(1.0, candidate.experience_years / 10.0),
                'contract_status': self._evaluate_contract_leverage(candidate),
                'age_factor': self._calculate_age_factor(candidate.age),
                'injury_risk': self._assess_injury_risk(candidate),
                'defensive_ability': random.uniform(0.4, 1.0),  # Mock defensive rating
                'team_fit': random.uniform(0.5, 1.0)  # Mock team fit rating
            }
            
            # Weighted score calculation
            weights = {
                'spring_performance': 0.25,
                'experience': 0.15,
                'contract_status': 0.20,
                'age_factor': 0.10,
                'injury_risk': 0.10,
                'defensive_ability': 0.10,
                'team_fit': 0.10
            }
            
            total_score = sum(score_factors[factor] * weights[factor] for factor in weights)
            
            candidate_info = {
                'player_id': candidate.player_id,
                'name': candidate.name,
                'age': candidate.age,
                'experience': candidate.experience_years,
                'total_score': round(total_score, 3),
                'score_factors': score_factors,
                'spring_stats': {
                    'games': candidate.spring_stats.games,
                    'batting_avg': candidate.spring_stats.batting_average,
                    'ops': candidate.spring_stats.on_base_percentage + candidate.spring_stats.slugging_percentage
                } if candidate.primary_position != 'P' else {
                    'games': candidate.spring_stats.games,
                    'era': candidate.spring_stats.era,
                    'whip': candidate.spring_stats.whip
                },
                'win_probability': 0.0  # Will calculate after all candidates
            }
            
            candidate_scores.append(candidate_info)
            battle_analysis['candidates'].append(candidate_info)
        
        # Calculate win probabilities
        total_scores = sum(c['total_score'] for c in candidate_scores)
        for candidate in candidate_scores:
            candidate['win_probability'] = round(candidate['total_score'] / total_scores, 3) if total_scores > 0 else 0.0
        
        # Determine favorite and competition level
        candidate_scores.sort(key=lambda x: x['total_score'], reverse=True)
        battle_analysis['favorite'] = candidate_scores[0]['name']
        
        # Competition level based on score differential
        top_score = candidate_scores[0]['total_score']
        second_score = candidate_scores[1]['total_score'] if len(candidate_scores) > 1 else 0
        score_diff = top_score - second_score
        battle_analysis['competition_level'] = max(0.1, 1.0 - (score_diff * 2))  # Closer scores = higher competition
        
        battle_analysis['confidence'] = 1.0 - battle_analysis['competition_level']  # Inverse relationship
        
        # Identify key factors
        key_factors = []
        if any(c['score_factors']['spring_performance'] > 0.7 for c in candidate_scores):
            key_factors.append("Spring training performance")
        if any(c['experience'] < 2 for c in candidates):
            key_factors.append("Experience vs. youth")
        if any(c.contract_status == 'Free Agent' for c in candidates):
            key_factors.append("Contract considerations")
        if any(c.injury_status != InjuryStatus.HEALTHY for c in candidates):
            key_factors.append("Health and availability")
            
        battle_analysis['key_factors'] = key_factors
        
        return battle_analysis

    def _evaluate_spring_performance(self, player: SpringPlayer) -> float:
        """Evaluate spring training performance"""
        stats = player.spring_stats
        
        if player.primary_position == 'P':
            if stats.innings_pitched == 0:
                return 0.5  # Neutral for no data
                
            # Pitcher evaluation
            era_score = max(0, 1 - (stats.era / 6.0))  # Better ERA = higher score
            whip_score = max(0, 1 - (stats.whip / 2.0))  # Better WHIP = higher score
            k_rate = stats.strikeouts_pitched / stats.innings_pitched if stats.innings_pitched > 0 else 0
            k_score = min(1.0, k_rate / 1.2)  # 1.2 K/IP = max score
            
            return (era_score + whip_score + k_score) / 3
        else:
            if stats.at_bats == 0:
                return 0.5  # Neutral for no data
                
            # Hitter evaluation
            ba_score = min(1.0, stats.batting_average / 0.350)  # .350 BA = max score
            obp_score = min(1.0, stats.on_base_percentage / 0.450)  # .450 OBP = max score
            slg_score = min(1.0, stats.slugging_percentage / 0.600)  # .600 SLG = max score
            
            return (ba_score + obp_score + slg_score) / 3

    def _evaluate_contract_leverage(self, player: SpringPlayer) -> float:
        """Evaluate contract-based leverage in roster decisions"""
        if player.contract_status == 'Rookie':
            return 0.3  # Low leverage, can be sent down easily
        elif player.contract_status == 'Pre-Arb':
            return 0.4 + (0.1 * player.minor_league_options)  # Some leverage
        elif player.contract_status == 'Arb-Eligible':
            return 0.7  # High leverage, expensive to release
        elif player.contract_status == 'Free Agent':
            return 0.9  # Very high leverage, no cost to release
        else:
            return 0.5  # Default

    def _calculate_age_factor(self, age: int) -> float:
        """Calculate age factor (peak around 27-29)"""
        if 25 <= age <= 29:
            return 1.0  # Peak years
        elif 22 <= age <= 24 or 30 <= age <= 32:
            return 0.8  # Good years
        elif age < 22 or 33 <= age <= 35:
            return 0.6  # Developing or declining
        else:
            return 0.3  # Very young or old

    def _assess_injury_risk(self, player: SpringPlayer) -> float:
        """Assess injury risk factor"""
        if player.injury_status == InjuryStatus.HEALTHY:
            return 1.0
        elif player.injury_status == InjuryStatus.RECOVERING:
            return 0.7
        elif player.injury_status == InjuryStatus.LIMITED:
            return 0.5
        elif player.injury_status == InjuryStatus.SETBACK:
            return 0.2
        else:
            return 0.6  # Unknown

    def project_opening_day_lineups(self) -> Dict[str, Dict[str, Any]]:
        """Project opening day lineups based on spring performance"""
        team_lineups = {}
        
        # Group players by team
        team_players = defaultdict(list)
        for player in self.players.values():
            if player.status in [PlayerStatus.ESTABLISHED, PlayerStatus.COMPETING] and \
               player.injury_status in [InjuryStatus.HEALTHY, InjuryStatus.RECOVERING]:
                team_players[player.team].append(player)
        
        for team, players in team_players.items():
            lineup_projection = self._project_team_lineup(team, players)
            team_lineups[team] = lineup_projection
            
        return team_lineups

    def _project_team_lineup(self, team: str, players: List[SpringPlayer]) -> Dict[str, Any]:
        """Project lineup for a specific team"""
        # Separate pitchers and position players
        pitchers = [p for p in players if p.primary_position == 'P']
        hitters = [p for p in players if p.primary_position != 'P']
        
        # Project starting rotation (top 5 pitchers)
        starting_rotation = self._select_starting_rotation(pitchers)
        
        # Project everyday lineup
        everyday_lineup = self._select_everyday_lineup(hitters)
        
        # Project bench players
        bench = self._select_bench_players(hitters, everyday_lineup)
        
        # Project bullpen
        bullpen = self._select_bullpen(pitchers, starting_rotation)
        
        lineup_projection = {
            'team': team,
            'starting_rotation': starting_rotation,
            'everyday_lineup': everyday_lineup,
            'bench': bench,
            'bullpen': bullpen,
            'lineup_confidence': self._calculate_lineup_confidence(everyday_lineup),
            'rotation_confidence': self._calculate_rotation_confidence(starting_rotation),
            'key_battles': self._identify_key_position_battles(players),
            'spring_standouts': self._identify_spring_standouts(players),
            'injury_concerns': self._identify_injury_concerns(players)
        }
        
        return lineup_projection

    def _select_starting_rotation(self, pitchers: List[SpringPlayer]) -> List[Dict[str, Any]]:
        """Select starting rotation based on spring performance and status"""
        starters = [p for p in pitchers if 'starter' in str(p.spring_notes).lower() or 
                   p.experience_years > 2 or p.status == PlayerStatus.ESTABLISHED]
        
        # Score each potential starter
        starter_scores = []
        for pitcher in starters[:8]:  # Consider top 8 candidates
            score = (
                self._evaluate_spring_performance(pitcher) * 0.4 +
                min(1.0, pitcher.experience_years / 6.0) * 0.3 +
                self._calculate_age_factor(pitcher.age) * 0.2 +
                self._assess_injury_risk(pitcher) * 0.1
            )
            
            starter_scores.append({
                'player': pitcher,
                'score': score,
                'projected_role': self._project_rotation_spot(pitcher, score)
            })
        
        # Sort by score and take top 5
        starter_scores.sort(key=lambda x: x['score'], reverse=True)
        
        rotation = []
        for i, starter in enumerate(starter_scores[:5]):
            rotation.append({
                'name': starter['player'].name,
                'player_id': starter['player'].player_id,
                'rotation_spot': i + 1,
                'projected_role': starter['projected_role'],
                'spring_era': starter['player'].spring_stats.era,
                'spring_whip': starter['player'].spring_stats.whip,
                'confidence': round(starter['score'], 2),
                'key_metrics': self._extract_pitcher_metrics(starter['player'])
            })
            
        return rotation

    def _project_rotation_spot(self, pitcher: SpringPlayer, score: float) -> str:
        """Project rotation spot based on pitcher profile"""
        if score > 0.8 and pitcher.experience_years > 3:
            return "Ace/Top of rotation"
        elif score > 0.7:
            return "Mid-rotation starter"
        elif score > 0.6:
            return "Back-end starter"
        else:
            return "Rotation candidate"

    def _extract_pitcher_metrics(self, pitcher: SpringPlayer) -> Dict[str, Any]:
        """Extract key metrics for pitcher evaluation"""
        velocity_data = pitcher.velocity_data
        
        metrics = {
            'innings_pitched': pitcher.spring_stats.innings_pitched,
            'strikeouts_per_9': (pitcher.spring_stats.strikeouts_pitched * 9 / pitcher.spring_stats.innings_pitched) 
                               if pitcher.spring_stats.innings_pitched > 0 else 0,
            'walks_per_9': (pitcher.spring_stats.walks_allowed * 9 / pitcher.spring_stats.innings_pitched)
                          if pitcher.spring_stats.innings_pitched > 0 else 0
        }
        
        if velocity_data:
            fastball_velocities = [v.velocity for v in velocity_data if 'fastball' in v.pitch_type.lower()]
            if fastball_velocities:
                metrics['avg_fastball_velocity'] = round(np.mean(fastball_velocities), 1)
                metrics['max_fastball_velocity'] = round(max(fastball_velocities), 1)
        
        return metrics

    def _select_everyday_lineup(self, hitters: List[SpringPlayer]) -> List[Dict[str, Any]]:
        """Select everyday lineup based on spring performance"""
        position_assignments = {}
        positions = ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH']
        
        for position in positions:
            candidates = [h for h in hitters if position in h.eligible_positions or h.primary_position == position]
            
            if candidates:
                best_candidate = max(candidates, key=lambda x: (
                    self._evaluate_spring_performance(x) * 0.4 +
                    self._calculate_age_factor(x.age) * 0.2 +
                    min(1.0, x.experience_years / 8.0) * 0.2 +
                    self._assess_injury_risk(x) * 0.2
                ))
                
                position_assignments[position] = {
                    'name': best_candidate.name,
                    'player_id': best_candidate.player_id,
                    'position': position,
                    'batting_average': best_candidate.spring_stats.batting_average,
                    'on_base_percentage': best_candidate.spring_stats.on_base_percentage,
                    'slugging_percentage': best_candidate.spring_stats.slugging_percentage,
                    'spring_games': best_candidate.spring_stats.games,
                    'experience_years': best_candidate.experience_years,
                    'age': best_candidate.age,
                    'status': best_candidate.status.value
                }
                
                # Remove assigned player from future consideration
                hitters = [h for h in hitters if h.player_id != best_candidate.player_id]
        
        # Convert to batting order (simplified logic)
        lineup = []
        batting_order_positions = ['CF', 'SS', '2B', '1B', 'DH', '3B', 'LF', 'RF', 'C']
        
        for i, position in enumerate(batting_order_positions):
            if position in position_assignments:
                player_info = position_assignments[position].copy()
                player_info['batting_order'] = i + 1
                lineup.append(player_info)
        
        return lineup

    def _select_bench_players(self, hitters: List[SpringPlayer], everyday_lineup: List[Dict]) -> List[Dict[str, Any]]:
        """Select bench players"""
        # Get players not in everyday lineup
        everyday_ids = {player['player_id'] for player in everyday_lineup}
        bench_candidates = [h for h in hitters if h.player_id not in everyday_ids]
        
        # Select top bench candidates (utility players, backup catcher, etc.)
        bench_scores = []
        for candidate in bench_candidates:
            versatility_bonus = len(candidate.eligible_positions) * 0.1
            score = (
                self._evaluate_spring_performance(candidate) * 0.3 +
                versatility_bonus +
                self._calculate_age_factor(candidate.age) * 0.2 +
                min(1.0, candidate.experience_years / 6.0) * 0.2 +
                self._assess_injury_risk(candidate) * 0.2
            )
            
            bench_scores.append({
                'player': candidate,
                'score': score
            })
        
        bench_scores.sort(key=lambda x: x['score'], reverse=True)
        
        bench = []
        for bench_player in bench_scores[:5]:  # Top 5 bench players
            bench.append({
                'name': bench_player['player'].name,
                'player_id': bench_player['player'].player_id,
                'primary_position': bench_player['player'].primary_position,
                'eligible_positions': bench_player['player'].eligible_positions,
                'role': self._project_bench_role(bench_player['player']),
                'spring_stats_summary': f".{int(bench_player['player'].spring_stats.batting_average * 1000):03d}/.{int(bench_player['player'].spring_stats.on_base_percentage * 1000):03d}",
                'experience_years': bench_player['player'].experience_years
            })
        
        return bench

    def _project_bench_role(self, player: SpringPlayer) -> str:
        """Project bench role for a player"""
        if player.primary_position == 'C':
            return "Backup Catcher"
        elif len(player.eligible_positions) >= 3:
            return "Super Utility"
        elif 'OF' in player.eligible_positions or any(pos in player.eligible_positions for pos in ['LF', 'CF', 'RF']):
            return "Fourth Outfielder"
        elif any(pos in player.eligible_positions for pos in ['1B', '2B', '3B', 'SS']):
            return "Infield Utility"
        else:
            return "Bench Player"

    def _select_bullpen(self, pitchers: List[SpringPlayer], rotation: List[Dict]) -> List[Dict[str, Any]]:
        """Select bullpen based on spring performance"""
        # Get pitchers not in rotation
        rotation_ids = {starter['player_id'] for starter in rotation}
        bullpen_candidates = [p for p in pitchers if p.player_id not in rotation_ids]
        
        # Score relievers
        reliever_scores = []
        for pitcher in bullpen_candidates:
            score = (
                self._evaluate_spring_performance(pitcher) * 0.4 +
                self._calculate_reliever_profile_score(pitcher) * 0.3 +
                self._calculate_age_factor(pitcher.age) * 0.2 +
                self._assess_injury_risk(pitcher) * 0.1
            )
            
            reliever_scores.append({
                'player': pitcher,
                'score': score
            })
        
        reliever_scores.sort(key=lambda x: x['score'], reverse=True)
        
        bullpen = []
        roles = ['Closer', 'Setup', 'Setup', 'Middle Relief', 'Middle Relief', 'Long Relief', 'Swing Man']
        
        for i, reliever in enumerate(reliever_scores[:7]):
            role = roles[i] if i < len(roles) else "Relief Pitcher"
            
            bullpen.append({
                'name': reliever['player'].name,
                'player_id': reliever['player'].player_id,
                'projected_role': role,
                'spring_era': reliever['player'].spring_stats.era,
                'spring_whip': reliever['player'].spring_stats.whip,
                'experience_years': reliever['player'].experience_years,
                'key_attributes': self._identify_reliever_attributes(reliever['player'])
            })
        
        return bullpen

    def _calculate_reliever_profile_score(self, pitcher: SpringPlayer) -> float:
        """Calculate reliever-specific profile score"""
        velocity_data = pitcher.velocity_data
        score = 0.5  # Base score
        
        # Velocity bonus for relievers
        if velocity_data:
            fastball_velos = [v.velocity for v in velocity_data if 'fastball' in v.pitch_type.lower()]
            if fastball_velos:
                avg_velo = np.mean(fastball_velos)
                if avg_velo > 95:
                    score += 0.3
                elif avg_velo > 92:
                    score += 0.2
                elif avg_velo > 89:
                    score += 0.1
        
        # Strikeout ability
        if pitcher.spring_stats.innings_pitched > 0:
            k_per_9 = (pitcher.spring_stats.strikeouts_pitched * 9) / pitcher.spring_stats.innings_pitched
            if k_per_9 > 10:
                score += 0.2
            elif k_per_9 > 8:
                score += 0.1
        
        return min(1.0, score)

    def _identify_reliever_attributes(self, pitcher: SpringPlayer) -> List[str]:
        """Identify key attributes for a reliever"""
        attributes = []
        velocity_data = pitcher.velocity_data
        
        if velocity_data:
            fastball_velos = [v.velocity for v in velocity_data if 'fastball' in v.pitch_type.lower()]
            if fastball_velos and max(fastball_velos) > 97:
                attributes.append("Power arm")
            
            pitch_types = set(v.pitch_type for v in velocity_data)
            if len(pitch_types) >= 3:
                attributes.append("Multiple pitches")
        
        if pitcher.spring_stats.innings_pitched > 0:
            k_per_9 = (pitcher.spring_stats.strikeouts_pitched * 9) / pitcher.spring_stats.innings_pitched
            if k_per_9 > 10:
                attributes.append("High strikeout rate")
                
            bb_per_9 = (pitcher.spring_stats.walks_allowed * 9) / pitcher.spring_stats.innings_pitched
            if bb_per_9 < 3:
                attributes.append("Good command")
        
        if pitcher.experience_years < 3:
            attributes.append("Young/developing")
        elif pitcher.experience_years > 8:
            attributes.append("Veteran presence")
        
        return attributes if attributes else ["Standard reliever profile"]

    def _calculate_lineup_confidence(self, lineup: List[Dict]) -> float:
        """Calculate confidence level in lineup projection"""
        if not lineup:
            return 0.0
            
        # Base confidence on spring sample sizes and performance consistency
        total_confidence = 0.0
        for player in lineup:
            games_factor = min(1.0, player['spring_games'] / 15.0)  # More games = higher confidence
            performance_factor = (player['batting_average'] + player['on_base_percentage']) / 2
            experience_factor = min(1.0, player['experience_years'] / 5.0)
            
            player_confidence = (games_factor * 0.4 + performance_factor * 0.3 + experience_factor * 0.3)
            total_confidence += player_confidence
            
        return round(total_confidence / len(lineup), 2)

    def _calculate_rotation_confidence(self, rotation: List[Dict]) -> float:
        """Calculate confidence level in rotation projection"""
        if not rotation:
            return 0.0
            
        total_confidence = sum(starter['confidence'] for starter in rotation)
        return round(total_confidence / len(rotation), 2)

    def _identify_key_position_battles(self, players: List[SpringPlayer]) -> List[str]:
        """Identify key position battles still ongoing"""
        battles = []
        
        # Count competitors by position
        position_competitors = defaultdict(int)
        for player in players:
            if player.status in [PlayerStatus.COMPETING, PlayerStatus.BUBBLE]:
                position_competitors[player.primary_position] += 1
        
        for position, count in position_competitors.items():
            if count > 1:
                battles.append(f"{position}: {count} candidates competing")
        
        return battles

    def _identify_spring_standouts(self, players: List[SpringPlayer]) -> List[Dict[str, Any]]:
        """Identify spring training standouts"""
        standouts = []
        
        for player in players:
            performance_score = self._evaluate_spring_performance(player)
            
            if performance_score > 0.8 and player.spring_stats.games >= 8:
                standouts.append({
                    'name': player.name,
                    'position': player.primary_position,
                    'performance_score': round(performance_score, 2),
                    'key_stats': self._get_standout_stats(player)
                })
        
        return sorted(standouts, key=lambda x: x['performance_score'], reverse=True)

    def _get_standout_stats(self, player: SpringPlayer) -> Dict[str, Any]:
        """Get key stats for standout players"""
        if player.primary_position == 'P':
            return {
                'ERA': round(player.spring_stats.era, 2),
                'WHIP': round(player.spring_stats.whip, 2),
                'K/9': round((player.spring_stats.strikeouts_pitched * 9 / player.spring_stats.innings_pitched), 1) 
                       if player.spring_stats.innings_pitched > 0 else 0,
                'IP': round(player.spring_stats.innings_pitched, 1)
            }
        else:
            return {
                'AVG': round(player.spring_stats.batting_average, 3),
                'OBP': round(player.spring_stats.on_base_percentage, 3),
                'SLG': round(player.spring_stats.slugging_percentage, 3),
                'HR': player.spring_stats.home_runs,
                'RBI': player.spring_stats.rbis
            }

    def _identify_injury_concerns(self, players: List[SpringPlayer]) -> List[Dict[str, Any]]:
        """Identify players with injury concerns"""
        concerns = []
        
        for player in players:
            if player.injury_status != InjuryStatus.HEALTHY:
                concerns.append({
                    'name': player.name,
                    'position': player.primary_position,
                    'injury_status': player.injury_status.value,
                    'impact_level': self._assess_injury_impact_level(player)
                })
        
        return concerns

    def _assess_injury_impact_level(self, player: SpringPlayer) -> str:
        """Assess the impact level of an injury"""
        if player.injury_status == InjuryStatus.SETBACK:
            return "High"
        elif player.injury_status == InjuryStatus.LIMITED:
            return "Moderate"
        elif player.injury_status == InjuryStatus.RECOVERING:
            return "Low"
        else:
            return "Unknown"

    def analyze_team_chemistry(self, team: str) -> Dict[str, Any]:
        """Analyze team chemistry changes from acquisitions and trades"""
        team_players = [p for p in self.players.values() if p.team == team]
        new_acquisitions = [p for p in team_players if p.acquisition_date and 
                           p.acquisition_date > datetime.now() - timedelta(days=180)]
        
        chemistry_analysis = {
            'team': team,
            'new_acquisitions': len(new_acquisitions),
            'veteran_presence': len([p for p in team_players if p.experience_years > 8]),
            'young_core': len([p for p in team_players if p.age < 26 and p.experience_years < 4]),
            'chemistry_factors': [],
            'integration_assessment': '',
            'clubhouse_impact': self._assess_clubhouse_impact(team_players, new_acquisitions)
        }
        
        # Analyze chemistry factors
        if new_acquisitions:
            chemistry_analysis['chemistry_factors'].append(f"{len(new_acquisitions)} new players integrating")
            
            high_profile_additions = [p for p in new_acquisitions if p.experience_years > 5]
            if high_profile_additions:
                chemistry_analysis['chemistry_factors'].append(f"{len(high_profile_additions)} veteran additions")
        
        international_players = len([p for p in team_players if 'international' in str(p.spring_notes).lower()])
        if international_players > 3:
            chemistry_analysis['chemistry_factors'].append(f"Strong international contingent ({international_players} players)")
        
        # Integration assessment
        if len(new_acquisitions) > 5:
            chemistry_analysis['integration_assessment'] = "Major roster turnover - chemistry development needed"
        elif len(new_acquisitions) > 2:
            chemistry_analysis['integration_assessment'] = "Moderate changes - good spring chemistry crucial"
        else:
            chemistry_analysis['integration_assessment'] = "Stable roster - chemistry should be strong"
            
        return chemistry_analysis

    def _assess_clubhouse_impact(self, team_players: List[SpringPlayer], new_players: List[SpringPlayer]) -> Dict[str, Any]:
        """Assess clubhouse impact of roster changes"""
        impact_analysis = {
            'leadership_changes': False,
            'cultural_fit_concerns': False,
            'positive_additions': [],
            'integration_challenges': [],
            'overall_rating': 'Neutral'
        }
        
        # Check for veteran leadership additions
        veteran_additions = [p for p in new_players if p.experience_years > 8 and p.age > 30]
        if veteran_additions:
            impact_analysis['positive_additions'].append(f"Veteran leadership: {[p.name for p in veteran_additions]}")
            impact_analysis['leadership_changes'] = True
        
        # Check for potential chemistry challenges
        high_maintenance_acquisitions = [p for p in new_players if 'behavior' in str(p.spring_notes).lower()]
        if high_maintenance_acquisitions:
            impact_analysis['integration_challenges'].append("Potential character concerns with new acquisitions")
            impact_analysis['cultural_fit_concerns'] = True
        
        # Overall rating
        positive_factors = len(impact_analysis['positive_additions'])
        negative_factors = len(impact_analysis['integration_challenges'])
        
        if positive_factors > negative_factors:
            impact_analysis['overall_rating'] = 'Positive'
        elif negative_factors > positive_factors:
            impact_analysis['overall_rating'] = 'Concerning'
        else:
            impact_analysis['overall_rating'] = 'Neutral'
            
        return impact_analysis

    def calculate_callup_probability(self, player_id: str) -> Dict[str, Any]:
        """Calculate probability of minor league call-up during season"""
        if player_id not in self.players:
            return {'error': 'Player not found'}
            
        player = self.players[player_id]
        
        callup_analysis = {
            'player_id': player_id,
            'name': player.name,
            'current_level': self._estimate_current_level(player),
            'mlb_readiness_score': 0.0,
            'callup_probability': 0.0,
            'projected_timeline': '',
            'key_factors': [],
            'development_areas': []
        }
        
        # Calculate MLB readiness score
        readiness_factors = {
            'age_factor': self._calculate_prospect_age_factor(player.age),
            'performance_factor': self._evaluate_spring_performance(player),
            'experience_factor': min(1.0, player.experience_years / 3.0),
            'position_need': self._assess_positional_need(player),
            'service_time': self._calculate_service_time_factor(player),
            'injury_status': self._assess_injury_risk(player),
            'spring_showing': self._evaluate_spring_showing(player)
        }
        
        # Weighted readiness score
        weights = {
            'age_factor': 0.15,
            'performance_factor': 0.25,
            'experience_factor': 0.15,
            'position_need': 0.20,
            'service_time': 0.10,
            'injury_status': 0.05,
            'spring_showing': 0.10
        }
        
        readiness_score = sum(readiness_factors[factor] * weights[factor] for factor in weights)
        callup_analysis['mlb_readiness_score'] = round(readiness_score, 3)
        
        # Convert to probability
        if readiness_score > 0.8:
            callup_analysis['callup_probability'] = 0.85
            callup_analysis['projected_timeline'] = "Opening Day or April"
        elif readiness_score > 0.65:
            callup_analysis['callup_probability'] = 0.65
            callup_analysis['projected_timeline'] = "Early season (April-May)"
        elif readiness_score > 0.5:
            callup_analysis['callup_probability'] = 0.45
            callup_analysis['projected_timeline'] = "Mid-season (June-July)"
        elif readiness_score > 0.35:
            callup_analysis['callup_probability'] = 0.25
            callup_analysis['projected_timeline'] = "Late season (August-September)"
        else:
            callup_analysis['callup_probability'] = 0.10
            callup_analysis['projected_timeline'] = "Unlikely in 2024"
        
        # Identify key factors
        key_factors = []
        if readiness_factors['position_need'] > 0.7:
            key_factors.append("High organizational need at position")
        if readiness_factors['spring_showing'] > 0.8:
            key_factors.append("Excellent spring training performance")
        if readiness_factors['age_factor'] > 0.8:
            key_factors.append("Optimal prospect age")
        if player.minor_league_options == 0:
            key_factors.append("Out of options - must be on roster or DFA")
            
        callup_analysis['key_factors'] = key_factors
        
        # Development areas
        development_areas = []
        if readiness_factors['performance_factor'] < 0.6:
            development_areas.append("Needs to improve statistical performance")
        if readiness_factors['experience_factor'] < 0.5:
            development_areas.append("Limited professional experience")
        if player.primary_position == 'P' and not player.velocity_data:
            development_areas.append("Velocity/stuff development needed")
            
        callup_analysis['development_areas'] = development_areas
        
        return callup_analysis

    def _estimate_current_level(self, player: SpringPlayer) -> str:
        """Estimate player's current minor league level"""
        if player.status == PlayerStatus.ESTABLISHED:
            return "MLB"
        elif player.experience_years > 3:
            return "AAA"
        elif player.experience_years > 1:
            return "AA"
        elif player.age > 23:
            return "A+"
        else:
            return "A"

    def _calculate_prospect_age_factor(self, age: int) -> float:
        """Calculate age factor for prospect evaluation"""
        if 21 <= age <= 24:
            return 1.0  # Ideal prospect age
        elif 20 <= age <= 25:
            return 0.8  # Good prospect age
        elif 19 <= age <= 26:
            return 0.6  # Acceptable age
        else:
            return 0.3  # Too young or old

    def _assess_positional_need(self, player: SpringPlayer) -> float:
        """Assess team's need at player's position"""
        # Mock assessment - in real implementation would analyze MLB roster depth
        team_players = [p for p in self.players.values() if p.team == player.team]
        position_players = [p for p in team_players if p.primary_position == player.primary_position]
        
        established_players = [p for p in position_players if p.status == PlayerStatus.ESTABLISHED]
        
        if len(established_players) == 0:
            return 1.0  # High need
        elif len(established_players) == 1:
            return 0.7  # Moderate need
        else:
            return 0.3  # Low need

    def _calculate_service_time_factor(self, player: SpringPlayer) -> float:
        """Calculate service time considerations"""
        if player.minor_league_options == 0:
            return 1.0  # Must be kept on roster
        elif player.minor_league_options == 1:
            return 0.7  # Limited flexibility
        else:
            return 0.4  # Plenty of flexibility

    def _evaluate_spring_showing(self, player: SpringPlayer) -> float:
        """Evaluate how player performed in spring training"""
        base_performance = self._evaluate_spring_performance(player)
        
        # Bonus for exceeding expectations
        if player.status == PlayerStatus.PROSPECT and base_performance > 0.7:
            return min(1.0, base_performance + 0.2)  # Bonus for prospects performing well
        
        return base_performance

    def detect_breakout_candidates(self) -> List[Dict[str, Any]]:
        """Detect potential breakout candidates based on spring indicators"""
        breakout_candidates = []
        
        for player in self.players.values():
            breakout_score = self._calculate_breakout_score(player)
            
            if breakout_score > 0.65:  # Threshold for breakout candidate
                breakout_analysis = {
                    'name': player.name,
                    'team': player.team,
                    'position': player.primary_position,
                    'age': player.age,
                    'experience_years': player.experience_years,
                    'breakout_score': round(breakout_score, 3),
                    'breakout_type': self._classify_breakout_type(player),
                    'supporting_evidence': self._collect_breakout_evidence(player),
                    'upside_projection': self._project_breakout_upside(player),
                    'risk_factors': self._identify_breakout_risks(player)
                }
                
                breakout_candidates.append(breakout_analysis)
        
        return sorted(breakout_candidates, key=lambda x: x['breakout_score'], reverse=True)

    def _calculate_breakout_score(self, player: SpringPlayer) -> float:
        """Calculate breakout potential score"""
        factors = {
            'age_factor': self._calculate_breakout_age_factor(player.age),
            'spring_performance': self._evaluate_spring_performance(player),
            'development_curve': self._assess_development_trajectory(player),
            'physical_changes': self._detect_physical_improvements(player),
            'approach_changes': self._detect_approach_changes(player),
            'health_factor': self._assess_injury_risk(player),
            'opportunity_factor': self._assess_opportunity_for_breakout(player)
        }
        
        weights = {
            'age_factor': 0.15,
            'spring_performance': 0.20,
            'development_curve': 0.15,
            'physical_changes': 0.15,
            'approach_changes': 0.15,
            'health_factor': 0.10,
            'opportunity_factor': 0.10
        }
        
        return sum(factors[factor] * weights[factor] for factor in weights)

    def _calculate_breakout_age_factor(self, age: int) -> float:
        """Calculate age factor for breakout potential"""
        if 23 <= age <= 26:
            return 1.0  # Prime breakout age
        elif 21 <= age <= 27:
            return 0.8  # Good breakout age
        elif 20 <= age <= 28:
            return 0.6  # Possible breakout age
        else:
            return 0.2  # Less likely breakout age

    def _assess_development_trajectory(self, player: SpringPlayer) -> float:
        """Assess player's development trajectory"""
        # Mock assessment based on experience progression
        if player.experience_years < 2:
            return 0.8  # High development potential
        elif player.experience_years < 4:
            return 0.6  # Moderate development potential
        else:
            return 0.3  # Lower development potential

    def _detect_physical_improvements(self, player: SpringPlayer) -> float:
        """Detect physical improvements (velocity, strength, etc.)"""
        if player.primary_position == 'P' and player.velocity_data:
            velocity_analysis = self.track_velocity_changes(player.player_id)
            significant_changes = velocity_analysis.get('significant_changes', [])
            
            improvements = [change for change in significant_changes if change['direction'] == 'increase']
            if improvements:
                return min(1.0, len(improvements) * 0.3 + 0.4)
        
        # For hitters, assess power indicators
        power_indicators = player.spring_stats.home_runs + player.spring_stats.doubles
        at_bats = player.spring_stats.at_bats
        
        if at_bats > 0:
            power_rate = power_indicators / at_bats
            return min(1.0, power_rate * 3)  # Scale power rate
            
        return 0.5  # Neutral when no data

    def _detect_approach_changes(self, player: SpringPlayer) -> float:
        """Detect changes in player approach"""
        # Mock detection based on spring notes and performance patterns
        approach_improvements = 0.5  # Base score
        
        # Look for improved plate discipline
        if player.spring_stats.at_bats > 0:
            walk_rate = player.spring_stats.walks / (player.spring_stats.at_bats + player.spring_stats.walks)
            strikeout_rate = player.spring_stats.strikeouts / player.spring_stats.at_bats
            
            if walk_rate > 0.12:  # Good walk rate
                approach_improvements += 0.2
            if strikeout_rate < 0.20:  # Good contact rate
                approach_improvements += 0.2
                
        return min(1.0, approach_improvements)

    def _assess_opportunity_for_breakout(self, player: SpringPlayer) -> float:
        """Assess opportunity for player to break out"""
        # Consider playing time opportunity and role changes
        if player.status in [PlayerStatus.ESTABLISHED, PlayerStatus.COMPETING]:
            return 0.8  # Good opportunity
        elif player.status == PlayerStatus.BUBBLE:
            return 0.6  # Moderate opportunity
        else:
            return 0.3  # Limited opportunity

    def _classify_breakout_type(self, player: SpringPlayer) -> str:
        """Classify the type of potential breakout"""
        if player.age < 24:
            return "Development Breakout"
        elif player.experience_years < 3:
            return "Adjustment Breakout"
        elif 'injury' in str(player.spring_notes).lower():
            return "Health Comeback"
        else:
            return "Late Bloomer"

    def _collect_breakout_evidence(self, player: SpringPlayer) -> List[str]:
        """Collect evidence supporting breakout potential"""
        evidence = []
        
        spring_performance = self._evaluate_spring_performance(player)
        if spring_performance > 0.7:
            evidence.append("Strong spring training performance")
            
        if player.primary_position == 'P' and player.velocity_data:
            velocity_analysis = self.track_velocity_changes(player.player_id)
            if velocity_analysis.get('new_pitches'):
                evidence.append("Added new pitch(es)")
            if velocity_analysis.get('overall_change', 0) > 1.0:
                evidence.append("Velocity increase across multiple pitches")
        
        if player.age in range(23, 27):
            evidence.append("Prime breakout age range")
            
        if player.spring_stats.games >= 12:
            evidence.append("Substantial spring playing time")
            
        return evidence if evidence else ["Limited supporting evidence"]

    def _project_breakout_upside(self, player: SpringPlayer) -> Dict[str, str]:
        """Project potential upside if breakout occurs"""
        if player.primary_position == 'P':
            return {
                'ceiling': "Mid-rotation starter" if player.experience_years < 3 else "Back-end starter",
                'statistical_targets': "200+ IP, 3.50 ERA, 8+ K/9",
                'fantasy_value': "SP3-SP4 potential"
            }
        else:
            return {
                'ceiling': "Everyday player" if player.age < 26 else "Quality regular",
                'statistical_targets': ".275+ AVG, 20+ HR, 80+ RBI",
                'fantasy_value': "Top 15 at position potential"
            }

    def _identify_breakout_risks(self, player: SpringPlayer) -> List[str]:
        """Identify risks to breakout potential"""
        risks = []
        
        if player.spring_stats.games < 10:
            risks.append("Limited spring sample size")
            
        if player.injury_status != InjuryStatus.HEALTHY:
            risks.append("Health concerns")
            
        if player.status == PlayerStatus.BUBBLE:
            risks.append("Uncertain roster spot")
            
        if player.experience_years > 5:
            risks.append("Limited track record of improvement")
            
        return risks if risks else ["Low risk profile"]

    def detect_regression_candidates(self) -> List[Dict[str, Any]]:
        """Detect players likely to regress from previous season performance"""
        regression_candidates = []
        
        for player in self.players.values():
            regression_score = self._calculate_regression_score(player)
            
            if regression_score > 0.60:  # Threshold for regression candidate
                regression_analysis = {
                    'name': player.name,
                    'team': player.team,
                    'position': player.primary_position,
                    'age': player.age,
                    'regression_score': round(regression_score, 3),
                    'regression_type': self._classify_regression_type(player),
                    'warning_signs': self._collect_regression_warning_signs(player),
                    'projected_decline': self._project_regression_magnitude(player),
                    'mitigating_factors': self._identify_mitigating_factors(player)
                }
                
                regression_candidates.append(regression_analysis)
        
        return sorted(regression_candidates, key=lambda x: x['regression_score'], reverse=True)

    def _calculate_regression_score(self, player: SpringPlayer) -> float:
        """Calculate regression risk score"""
        factors = {
            'age_factor': self._calculate_regression_age_factor(player.age),
            'spring_performance': 1.0 - self._evaluate_spring_performance(player),  # Poor spring = higher regression risk
            'injury_concerns': 1.0 - self._assess_injury_risk(player),
            'velocity_decline': self._assess_velocity_decline_risk(player),
            'peripheral_concerns': self._assess_peripheral_regression(player),
            'usage_concerns': self._assess_usage_regression_risk(player)
        }
        
        weights = {
            'age_factor': 0.20,
            'spring_performance': 0.25,
            'injury_concerns': 0.20,
            'velocity_decline': 0.15,
            'peripheral_concerns': 0.10,
            'usage_concerns': 0.10
        }
        
        return sum(factors[factor] * weights[factor] for factor in weights)

    def _calculate_regression_age_factor(self, age: int) -> float:
        """Calculate age-based regression risk"""
        if age >= 34:
            return 1.0  # High regression risk
        elif age >= 31:
            return 0.7  # Moderate regression risk
        elif age >= 28:
            return 0.4  # Some regression risk
        else:
            return 0.1  # Low age-based regression risk

    def _assess_velocity_decline_risk(self, player: SpringPlayer) -> float:
        """Assess risk of velocity decline for pitchers"""
        if player.primary_position != 'P' or not player.velocity_data:
            return 0.0
            
        velocity_analysis = self.track_velocity_changes(player.player_id)
        significant_changes = velocity_analysis.get('significant_changes', [])
        
        declines = [change for change in significant_changes if change['direction'] == 'decrease']
        if declines:
            major_declines = [change for change in declines if change['significance'] == 'Major']
            return min(1.0, len(declines) * 0.3 + len(major_declines) * 0.4)
            
        return 0.0

    def _assess_peripheral_regression(self, player: SpringPlayer) -> float:
        """Assess peripheral stats suggesting regression"""
        # Mock assessment - in real implementation would compare to expected stats
        if player.primary_position == 'P':
            # Look for unsustainable ERA vs peripherals
            if player.spring_stats.era < 2.0 and player.spring_stats.whip > 1.3:
                return 0.8  # ERA looks unsustainable
        else:
            # Look for unsustainable batting average
            if (player.spring_stats.batting_average > 0.350 and 
                player.spring_stats.strikeouts / player.spring_stats.at_bats > 0.25):
                return 0.7  # High average with high strikeout rate
                
        return 0.3  # Default moderate concern

    def _assess_usage_regression_risk(self, player: SpringPlayer) -> float:
        """Assess risk from usage changes or workload concerns"""
        # Mock assessment for usage/workload regression risk
        if player.age > 32 and player.primary_position == 'P':
            return 0.6  # Older pitchers at higher risk
        elif player.experience_years > 8:
            return 0.4  # Veterans may see reduced roles
        else:
            return 0.2  # Lower risk for younger players

    def _classify_regression_type(self, player: SpringPlayer) -> str:
        """Classify the type of potential regression"""
        if player.age >= 33:
            return "Age-Related Decline"
        elif player.injury_status != InjuryStatus.HEALTHY:
            return "Injury-Related Regression"
        elif player.primary_position == 'P':
            return "Workload/Velocity Regression"
        else:
            return "Performance Regression"

    def _collect_regression_warning_signs(self, player: SpringPlayer) -> List[str]:
        """Collect warning signs for potential regression"""
        warning_signs = []
        
        spring_performance = self._evaluate_spring_performance(player)
        if spring_performance < 0.4:
            warning_signs.append("Poor spring training performance")
            
        if player.injury_status in [InjuryStatus.LIMITED, InjuryStatus.RECOVERING]:
            warning_signs.append("Coming off injury")
            
        if player.age >= 32:
            warning_signs.append("Age-related decline window")
            
        if player.primary_position == 'P' and player.velocity_data:
            velocity_analysis = self.track_velocity_changes(player.player_id)
            if velocity_analysis.get('overall_change', 0) < -1.0:
                warning_signs.append("Velocity decline from previous season")
        
        return warning_signs if warning_signs else ["No major warning signs identified"]

    def _project_regression_magnitude(self, player: SpringPlayer) -> Dict[str, str]:
        """Project magnitude of potential regression"""
        regression_score = self._calculate_regression_score(player)
        
        if regression_score > 0.8:
            return {
                'severity': "Major",
                'description': "Significant decline expected",
                'fantasy_impact': "Avoid or target late in drafts"
            }
        elif regression_score > 0.65:
            return {
                'severity': "Moderate",
                'description': "Noticeable decline likely",
                'fantasy_impact': "Discount previous season performance"
            }
        else:
            return {
                'severity': "Minor",
                'description': "Slight decline possible",
                'fantasy_impact': "Monitor closely but not necessarily avoidable"
            }

    def _identify_mitigating_factors(self, player: SpringPlayer) -> List[str]:
        """Identify factors that might mitigate regression risk"""
        factors = []
        
        if player.experience_years < 5:
            factors.append("Still in development phase")
            
        if self._evaluate_spring_performance(player) > 0.7:
            factors.append("Strong spring training showing")
            
        if player.injury_status == InjuryStatus.HEALTHY:
            factors.append("Currently healthy")
            
        if player.primary_position == 'P' and player.velocity_data:
            new_pitches = self.detect_new_pitches().get(player.name, [])
            if new_pitches:
                factors.append("Added new pitch(es) to arsenal")
        
        return factors if factors else ["Limited mitigating factors"]

    def generate_spring_report(self) -> Dict[str, Any]:
        """Generate comprehensive spring training report"""
        report = {
            'report_date': datetime.now().isoformat(),
            'total_players_analyzed': len(self.players),
            'velocity_analysis': self._summarize_velocity_trends(),
            'roster_battles': self.predict_roster_battles()[:10],  # Top 10 battles
            'lineup_projections': self.project_opening_day_lineups(),
            'breakout_candidates': self.detect_breakout_candidates()[:15],  # Top 15
            'regression_candidates': self.detect_regression_candidates()[:10],  # Top 10
            'new_pitch_report': self.detect_new_pitches(),
            'injury_updates': self._summarize_injury_landscape(),
            'prospect_callups': self._summarize_callup_probabilities(),
            'team_chemistry_notes': self._summarize_team_chemistry_changes(),
            'key_storylines': self._identify_key_spring_storylines()
        }
        
        return report

    def _summarize_velocity_trends(self) -> Dict[str, Any]:
        """Summarize velocity trends across all pitchers"""
        velocity_summary = {
            'pitchers_tracked': 0,
            'avg_fastball_change': 0.0,
            'significant_increases': [],
            'significant_decreases': [],
            'new_pitches_count': 0
        }
        
        velocity_changes = []
        significant_increases = []
        significant_decreases = []
        
        for player in self.players.values():
            if player.primary_position == 'P' and player.velocity_data:
                velocity_summary['pitchers_tracked'] += 1
                
                velocity_analysis = self.track_velocity_changes(player.player_id)
                
                # Track fastball changes
                for pitch_type, data in velocity_analysis.get('pitch_types', {}).items():
                    if 'fastball' in pitch_type.lower():
                        velocity_changes.append(data['change_mph'])
                
                # Track significant changes
                for change in velocity_analysis.get('significant_changes', []):
                    if change['direction'] == 'increase' and change['change'] > 2.0:
                        significant_increases.append(f"{player.name}: +{change['change']} mph {change['pitch_type']}")
                    elif change['direction'] == 'decrease' and change['change'] < -2.0:
                        significant_decreases.append(f"{player.name}: {change['change']} mph {change['pitch_type']}")
                
                # Count new pitches
                velocity_summary['new_pitches_count'] += len(velocity_analysis.get('new_pitches', []))
        
        if velocity_changes:
            velocity_summary['avg_fastball_change'] = round(np.mean(velocity_changes), 1)
            
        velocity_summary['significant_increases'] = significant_increases[:10]  # Top 10
        velocity_summary['significant_decreases'] = significant_decreases[:10]  # Top 10
        
        return velocity_summary

    def _summarize_injury_landscape(self) -> Dict[str, Any]:
        """Summarize injury landscape across all players"""
        injury_summary = {
            'total_injuries': 0,
            'by_status': {},
            'position_impact': {},
            'recovery_timelines': [],
            'concern_level': 'Low'
        }
        
        status_counts = Counter()
        position_impacts = defaultdict(int)
        
        for player in self.players.values():
            if player.injury_status != InjuryStatus.HEALTHY:
                injury_summary['total_injuries'] += 1
                status_counts[player.injury_status.value] += 1
                position_impacts[player.primary_position] += 1
                
                if player.injury_status in [InjuryStatus.LIMITED, InjuryStatus.SETBACK]:
                    injury_summary['recovery_timelines'].append({
                        'player': player.name,
                        'position': player.primary_position,
                        'team': player.team,
                        'status': player.injury_status.value
                    })
        
        injury_summary['by_status'] = dict(status_counts)
        injury_summary['position_impact'] = dict(position_impacts)
        
        # Assess overall concern level
        if injury_summary['total_injuries'] > 20:
            injury_summary['concern_level'] = 'High'
        elif injury_summary['total_injuries'] > 10:
            injury_summary['concern_level'] = 'Moderate'
        else:
            injury_summary['concern_level'] = 'Low'
            
        return injury_summary

    def _summarize_callup_probabilities(self) -> List[Dict[str, Any]]:
        """Summarize call-up probabilities for prospects"""
        callup_prospects = []
        
        prospects = [p for p in self.players.values() if p.status == PlayerStatus.PROSPECT]
        
        for prospect in prospects[:20]:  # Top 20 prospects
            callup_analysis = self.calculate_callup_probability(prospect.player_id)
            if callup_analysis.get('callup_probability', 0) > 0.3:
                callup_prospects.append({
                    'name': prospect.name,
                    'team': prospect.team,
                    'position': prospect.primary_position,
                    'probability': callup_analysis['callup_probability'],
                    'timeline': callup_analysis['projected_timeline']
                })
        
        return sorted(callup_prospects, key=lambda x: x['probability'], reverse=True)

    def _summarize_team_chemistry_changes(self) -> Dict[str, Dict[str, Any]]:
        """Summarize team chemistry changes across all teams"""
        teams = set(player.team for player in self.players.values())
        chemistry_summary = {}
        
        for team in teams:
            team_analysis = self.analyze_team_chemistry(team)
            if team_analysis['new_acquisitions'] > 0:
                chemistry_summary[team] = {
                    'new_acquisitions': team_analysis['new_acquisitions'],
                    'integration_assessment': team_analysis['integration_assessment'],
                    'impact_rating': team_analysis['clubhouse_impact']['overall_rating']
                }
        
        return chemistry_summary

    def _identify_key_spring_storylines(self) -> List[str]:
        """Identify key storylines emerging from spring training"""
        storylines = []
        
        # Velocity increases
        velocity_summary = self._summarize_velocity_trends()
        if len(velocity_summary['significant_increases']) > 5:
            storylines.append(f"Velocity surge: {len(velocity_summary['significant_increases'])} pitchers showing significant gains")
        
        # Roster battles
        battles = self.predict_roster_battles()
        competitive_battles = [b for b in battles if b['competition_level'] > 0.7]
        if competitive_battles:
            storylines.append(f"Tight roster battles: {len(competitive_battles)} highly competitive position battles")
        
        # Breakout candidates
        breakouts = self.detect_breakout_candidates()
        high_potential_breakouts = [b for b in breakouts if b['breakout_score'] > 0.75]
        if high_potential_breakouts:
            storylines.append(f"Breakout watch: {len(high_potential_breakouts)} players showing high breakout potential")
        
        # Injury concerns
        injury_summary = self._summarize_injury_landscape()
        if injury_summary['concern_level'] == 'High':
            storylines.append(f"Injury concerns mounting: {injury_summary['total_injuries']} players dealing with health issues")
        
        # New pitches
        new_pitches = self.detect_new_pitches()
        if len(new_pitches) > 8:
            storylines.append(f"Pitch development: {len(new_pitches)} pitchers working on new offerings")
        
        return storylines[:10]  # Top 10 storylines

def main():
    """Demo script for Spring Training Model"""
    print("🌸 MLB Spring Training Model v2.0")
    print("=" * 50)
    
    # Initialize model
    model = SpringTrainingModel()
    
    # Load spring training data
    print("📊 Loading spring training data...")
    model.load_spring_data('mock')
    
    print(f"✅ Loaded {len(model.players)} players")
    
    # Analyze velocity changes
    print("\n🎯 Analyzing velocity changes...")
    sample_pitchers = [p.player_id for p in model.players.values() if p.primary_position == 'P'][:3]
    
    for pitcher_id in sample_pitchers:
        velocity_analysis = model.track_velocity_changes(pitcher_id)
        player_name = velocity_analysis.get('name', pitcher_id)
        
        print(f"\n{player_name} ({velocity_analysis.get('team', 'UNK')}):")
        print(f"  Overall Change: {velocity_analysis.get('overall_change', 0):.1f} mph")
        
        for change in velocity_analysis.get('significant_changes', [])[:2]:
            print(f"  {change['pitch_type']}: {change['change']:+.1f} mph ({change['significance']})")
    
    # Detect new pitches
    print("\n🆕 Detecting new pitches...")
    new_pitches = model.detect_new_pitches()
    
    for player, data in list(new_pitches.items())[:3]:
        print(f"{player}: {len(data[0]['new_pitches'])} new pitch(es)")
        for pitch in data[0]['new_pitches']:
            print(f"  {pitch['pitch_type']} ({pitch['development_stage']})")
    
    # Predict roster battles
    print("\n⚔️ Predicting roster battles...")
    battles = model.predict_roster_battles()
    
    for battle in battles[:3]:
        print(f"\n{battle['team']} {battle['position']}:")
        print(f"  Favorite: {battle['favorite']} ({battle['confidence']:.0%} confidence)")
        print(f"  Competition Level: {battle['competition_level']:.1%}")
    
    # Project opening day lineups
    print("\n📋 Projecting opening day lineups...")
    lineups = model.project_opening_day_lineups()
    
    sample_team = list(lineups.keys())[0]
    team_lineup = lineups[sample_team]
    
    print(f"\n{sample_team} Projected Lineup:")
    for i, player in enumerate(team_lineup['everyday_lineup'][:3]):  # Show top 3
        print(f"  {i+1}. {player['name']} ({player['position']}) - {player['batting_average']:.3f}")
    
    print(f"\nStarting Rotation:")
    for starter in team_lineup['starting_rotation']:
        print(f"  {starter['rotation_spot']}. {starter['name']} ({starter['spring_era']:.2f} ERA)")
    
    # Identify breakout candidates
    print("\n📈 Identifying breakout candidates...")
    breakouts = model.detect_breakout_candidates()
    
    for candidate in breakouts[:3]:
        print(f"{candidate['name']} ({candidate['team']}) - {candidate['breakout_score']:.3f}")
        print(f"  Type: {candidate['breakout_type']}")
        print(f"  Evidence: {', '.join(candidate['supporting_evidence'][:2])}")
    
    # Detect regression candidates
    print("\n📉 Detecting regression candidates...")
    regressions = model.detect_regression_candidates()
    
    for candidate in regressions[:3]:
        print(f"{candidate['name']} ({candidate['team']}) - {candidate['regression_score']:.3f}")
        print(f"  Type: {candidate['regression_type']}")
        print(f"  Severity: {candidate['projected_decline']['severity']}")
    
    # Calculate call-up probabilities
    print("\n📞 Calculating call-up probabilities...")
    prospects = [p for p in model.players.values() if p.status == PlayerStatus.PROSPECT][:3]
    
    for prospect in prospects:
        callup_analysis = model.calculate_callup_probability(prospect.player_id)
        print(f"{prospect.name}: {callup_analysis['callup_probability']:.0%} ({callup_analysis['projected_timeline']})")
    
    # Generate comprehensive report
    print("\n📄 Generating spring training report...")
    report = model.generate_spring_report()
    
    print(f"\n🎉 Spring Training Analysis Complete!")
    print(f"  Players Analyzed: {report['total_players_analyzed']}")
    print(f"  Roster Battles: {len(report['roster_battles'])}")
    print(f"  Breakout Candidates: {len(report['breakout_candidates'])}")
    print(f"  Regression Candidates: {len(report['regression_candidates'])}")
    print(f"  Key Storylines: {len(report['key_storylines'])}")
    
    print(f"\n🔥 Top Storyline: {report['key_storylines'][0] if report['key_storylines'] else 'No major storylines identified'}")

if __name__ == "__main__":
    main()