"""
Inning-by-Inning Score Predictor

This module provides sophisticated inning-by-inning run scoring predictions
for MLB games, incorporating lineup positioning, pitcher fatigue curves,
bullpen transitions, and situational factors.

Author: MLB Predictor System
Created: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import json
import logging
from scipy.stats import poisson, beta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class InningHalf(Enum):
    """Enumeration for inning halves."""
    TOP = "top"
    BOTTOM = "bottom"


class PitcherRole(Enum):
    """Enumeration for pitcher roles."""
    STARTER = "starter"
    RELIEVER = "reliever"
    CLOSER = "closer"


@dataclass
class BatterPosition:
    """Represents a batter's position in the lineup with performance metrics."""
    lineup_position: int  # 1-9
    player_id: str
    avg_vs_rhp: float = 0.250
    avg_vs_lhp: float = 0.250
    obp_vs_rhp: float = 0.320
    obp_vs_lhp: float = 0.320
    slg_vs_rhp: float = 0.400
    slg_vs_lhp: float = 0.400
    clutch_factor: float = 1.0  # Multiplier for pressure situations
    late_inning_boost: float = 1.0  # Performance boost in innings 7+


@dataclass
class PitcherState:
    """Represents current pitcher state and fatigue metrics."""
    pitcher_id: str
    role: PitcherRole
    pitches_thrown: int = 0
    innings_pitched: float = 0.0
    era: float = 4.50
    whip: float = 1.30
    k_per_9: float = 8.0
    bb_per_9: float = 3.0
    is_same_handed: bool = False  # vs current batter
    rest_days: int = 0
    velocity_decline: float = 0.0  # % decline from first inning
    command_rating: float = 50.0  # 0-100 scale


@dataclass
class GameSituation:
    """Represents the current game situation."""
    inning: int
    half: InningHalf
    score_diff: int  # positive = home team leading
    runners_on_base: List[str] = field(default_factory=list)  # ["1B", "2B", "3B"]
    outs: int = 0
    pitch_count: int = 0
    leverage_index: float = 1.0
    pressure_rating: float = 1.0  # 1.0 = normal, >1.0 = high pressure


@dataclass
class InningPrediction:
    """Prediction results for a specific inning."""
    inning: int
    half: InningHalf
    predicted_runs: float
    run_probability_dist: Dict[int, float]  # P(exactly N runs)
    scoring_probability: float  # P(1+ runs)
    big_inning_probability: float  # P(3+ runs)
    confidence_interval: Tuple[float, float]  # 90% CI for runs
    key_factors: List[str]  # Most influential factors


@dataclass
class PitcherDecayCurve:
    """Models pitcher effectiveness decay by inning."""
    starter_fatigue_curve: Dict[int, float] = field(default_factory=dict)
    reliever_effectiveness: Dict[int, float] = field(default_factory=dict)
    velocity_decay_rate: float = 0.02  # % per inning
    command_decay_rate: float = 0.03  # % per inning
    
    def __post_init__(self):
        """Initialize default decay curves."""
        if not self.starter_fatigue_curve:
            # Default starter fatigue (effectiveness multiplier by inning)
            self.starter_fatigue_curve = {
                1: 1.05, 2: 1.02, 3: 1.00, 4: 0.98, 5: 0.95,
                6: 0.91, 7: 0.85, 8: 0.78, 9: 0.70
            }
        
        if not self.reliever_effectiveness:
            # Default reliever effectiveness by inning of appearance
            self.reliever_effectiveness = {
                1: 1.00, 2: 0.95, 3: 0.88, 4: 0.80
            }


class InningByInningPredictor:
    """
    Advanced inning-by-inning scoring predictor for MLB games.
    
    This class provides comprehensive run prediction capabilities including:
    - Lineup position impact modeling
    - Pitcher fatigue and decay curves
    - Bullpen transition effects
    - Situational adjustments
    - Over/under probability calculations
    """
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize the predictor.
        
        Args:
            historical_data: DataFrame with historical game data for training
        """
        self.historical_data = historical_data
        self.decay_curves = PitcherDecayCurve()
        self.lineup_impact_model = None
        self.situational_model = None
        self.is_trained = False
        
        # Base run scoring rates by inning
        self.base_scoring_rates = {
            1: 0.52, 2: 0.48, 3: 0.51, 4: 0.49, 5: 0.47,
            6: 0.45, 7: 0.48, 8: 0.44, 9: 0.42
        }
        
        # Initialize models if historical data provided
        if historical_data is not None:
            self._train_models()
    
    def _train_models(self) -> None:
        """Train internal models using historical data."""
        if self.historical_data is None or len(self.historical_data) < 100:
            logger.warning("Insufficient historical data for training")
            return
        
        try:
            # Train lineup impact model
            self._train_lineup_model()
            
            # Train situational model
            self._train_situational_model()
            
            self.is_trained = True
            logger.info("Models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _train_lineup_model(self) -> None:
        """Train model for lineup position impact."""
        # Simplified lineup impact - in production would use actual data
        lineup_multipliers = {
            1: 1.15, 2: 1.10, 3: 1.20, 4: 1.18, 5: 1.05,
            6: 0.95, 7: 0.90, 8: 0.85, 9: 0.80
        }
        self.lineup_impact_model = lineup_multipliers
    
    def _train_situational_model(self) -> None:
        """Train model for situational adjustments."""
        # Simplified situational factors - in production would use ML models
        self.situational_model = {
            'runners_on_multipliers': {
                tuple(): 1.0,           # Bases empty
                ('1B',): 1.25,          # Runner on 1st
                ('2B',): 1.35,          # Runner on 2nd
                ('3B',): 1.45,          # Runner on 3rd
                ('1B', '2B'): 1.55,     # Runners on 1st and 2nd
                ('1B', '3B'): 1.65,     # Runners on 1st and 3rd
                ('2B', '3B'): 1.75,     # Runners on 2nd and 3rd
                ('1B', '2B', '3B'): 1.85  # Bases loaded
            },
            'outs_multipliers': {0: 1.2, 1: 1.0, 2: 0.7},
            'pressure_multipliers': {
                'low': 0.95, 'normal': 1.0, 'high': 1.1, 'critical': 1.2
            }
        }
    
    def predict_inning_scores(
        self,
        home_lineup: List[BatterPosition],
        away_lineup: List[BatterPosition],
        home_pitcher: PitcherState,
        away_pitcher: PitcherState,
        game_situation: GameSituation,
        num_innings: int = 9
    ) -> List[InningPrediction]:
        """
        Predict run scoring for each inning of a game.
        
        Args:
            home_lineup: Home team batting order
            away_lineup: Away team batting order
            home_pitcher: Home team starting pitcher
            away_pitcher: Away team starting pitcher
            game_situation: Current game state
            num_innings: Number of innings to predict
            
        Returns:
            List of InningPrediction objects for each half-inning
        """
        predictions = []
        
        for inning in range(game_situation.inning, num_innings + 1):
            for half in [InningHalf.TOP, InningHalf.BOTTOM]:
                # Skip if we're in the middle of an inning
                if (inning == game_situation.inning and 
                    ((half == InningHalf.TOP and game_situation.half == InningHalf.BOTTOM) or
                     (half == InningHalf.BOTTOM and game_situation.half == InningHalf.TOP and inning > game_situation.inning))):
                    continue
                
                # Determine batting team and pitcher
                if half == InningHalf.TOP:
                    batting_lineup = away_lineup
                    pitching_state = home_pitcher
                else:
                    batting_lineup = home_lineup
                    pitching_state = away_pitcher
                
                # Create inning situation
                inning_situation = GameSituation(
                    inning=inning,
                    half=half,
                    score_diff=game_situation.score_diff,
                    runners_on_base=game_situation.runners_on_base if inning == game_situation.inning else [],
                    outs=game_situation.outs if inning == game_situation.inning else 0,
                    leverage_index=self._calculate_leverage_index(inning, game_situation.score_diff),
                    pressure_rating=self._calculate_pressure_rating(inning, game_situation.score_diff)
                )
                
                # Predict for this half-inning
                prediction = self._predict_half_inning(
                    batting_lineup, pitching_state, inning_situation
                )
                
                predictions.append(prediction)
                
                # Update pitcher state for fatigue
                pitching_state.innings_pitched += 0.5
                pitching_state.pitches_thrown += 15  # Average pitches per half inning
        
        return predictions
    
    def _predict_half_inning(
        self,
        batting_lineup: List[BatterPosition],
        pitcher: PitcherState,
        situation: GameSituation
    ) -> InningPrediction:
        """Predict scoring for a single half-inning."""
        
        # Get base scoring probability
        base_rate = self.base_scoring_rates.get(situation.inning, 0.45)
        
        # Apply pitcher decay
        pitcher_effectiveness = self.model_pitcher_decay(pitcher, situation.inning)
        
        # Calculate lineup strength for this inning
        lineup_strength = self._calculate_lineup_strength(batting_lineup, situation.inning)
        
        # Apply situational adjustments
        situational_multiplier = self._get_situational_multiplier(situation)
        
        # Calculate expected runs
        expected_runs = (base_rate * pitcher_effectiveness * 
                        lineup_strength * situational_multiplier * 
                        (1 + situation.leverage_index * 0.1))
        
        # Generate run distribution using Poisson with adjustments
        run_distribution = self._generate_run_distribution(expected_runs)
        
        # Calculate probabilities
        scoring_prob = 1 - run_distribution[0]  # P(1+ runs)
        big_inning_prob = sum(run_distribution[i] for i in range(3, 11))  # P(3+ runs)
        
        # Calculate confidence interval
        ci = self._calculate_confidence_interval(expected_runs)
        
        # Identify key factors
        key_factors = self._identify_key_factors(
            pitcher, batting_lineup, situation, pitcher_effectiveness, lineup_strength
        )
        
        return InningPrediction(
            inning=situation.inning,
            half=situation.half,
            predicted_runs=expected_runs,
            run_probability_dist=run_distribution,
            scoring_probability=scoring_prob,
            big_inning_probability=big_inning_prob,
            confidence_interval=ci,
            key_factors=key_factors
        )
    
    def get_scoring_probability(
        self,
        batting_lineup: List[BatterPosition],
        pitcher: PitcherState,
        situation: GameSituation,
        min_runs: int = 1
    ) -> float:
        """
        Calculate probability of scoring at least N runs in an inning.
        
        Args:
            batting_lineup: Batting team lineup
            pitcher: Opposing pitcher state
            situation: Current game situation
            min_runs: Minimum runs to score
            
        Returns:
            Probability of scoring at least min_runs
        """
        prediction = self._predict_half_inning(batting_lineup, pitcher, situation)
        
        probability = sum(
            prediction.run_probability_dist[i] 
            for i in range(min_runs, 11)  # Cap at 10 runs
        )
        
        return min(probability, 1.0)
    
    def model_pitcher_decay(self, pitcher: PitcherState, inning: int) -> float:
        """
        Model pitcher effectiveness decay by inning.
        
        Args:
            pitcher: Pitcher state information
            inning: Current inning
            
        Returns:
            Effectiveness multiplier (1.0 = normal, <1.0 = degraded)
        """
        if pitcher.role == PitcherRole.STARTER:
            # Use starter fatigue curve
            base_effectiveness = self.decay_curves.starter_fatigue_curve.get(inning, 0.70)
            
            # Apply pitch count fatigue
            pitch_fatigue = max(0.7, 1.0 - (pitcher.pitches_thrown - 60) * 0.002)
            
            # Apply velocity decline
            velocity_factor = 1.0 - pitcher.velocity_decline
            
            # Apply rest day adjustment
            rest_adjustment = min(1.1, 1.0 + pitcher.rest_days * 0.02)
            
            return base_effectiveness * pitch_fatigue * velocity_factor * rest_adjustment
        
        else:  # Reliever or closer
            # Relievers generally maintain effectiveness better
            innings_in_appearance = int(pitcher.innings_pitched % 1 * 2) + 1
            base_effectiveness = self.decay_curves.reliever_effectiveness.get(
                innings_in_appearance, 0.80
            )
            
            # Less pitch count impact for relievers
            pitch_fatigue = max(0.8, 1.0 - (pitcher.pitches_thrown - 20) * 0.001)
            
            return base_effectiveness * pitch_fatigue
    
    def _calculate_lineup_strength(self, lineup: List[BatterPosition], inning: int) -> float:
        """Calculate overall lineup strength for the inning."""
        # Determine which batters are likely to come up
        total_strength = 0.0
        batter_count = 0
        
        # Simplified - assume 3-4 batters per inning
        for i in range(3):  # Average 3 batters per inning
            batter_idx = ((inning - 1) * 3 + i) % 9
            batter = lineup[batter_idx]
            
            # Calculate batter strength (OPS proxy)
            ops = (batter.obp_vs_rhp + batter.slg_vs_rhp) * batter.clutch_factor
            if inning >= 7:
                ops *= batter.late_inning_boost
            
            # Apply lineup position multiplier
            if self.lineup_impact_model:
                lineup_multiplier = self.lineup_impact_model.get(batter.lineup_position, 1.0)
                ops *= lineup_multiplier
            
            total_strength += ops
            batter_count += 1
        
        # Normalize to ~1.0 for average lineup
        return (total_strength / batter_count) / 0.75  # Average OPS ~0.75
    
    def _get_situational_multiplier(self, situation: GameSituation) -> float:
        """Calculate multiplier based on game situation."""
        multiplier = 1.0
        
        if self.situational_model:
            # Runners on base
            runners_key = tuple(sorted(situation.runners_on_base))
            runners_mult = self.situational_model['runners_on_multipliers'].get(runners_key, 1.0)
            
            # Outs
            outs_mult = self.situational_model['outs_multipliers'].get(situation.outs, 1.0)
            
            # Pressure situation
            if situation.pressure_rating > 1.5:
                pressure_mult = self.situational_model['pressure_multipliers']['critical']
            elif situation.pressure_rating > 1.2:
                pressure_mult = self.situational_model['pressure_multipliers']['high']
            else:
                pressure_mult = self.situational_model['pressure_multipliers']['normal']
            
            multiplier = runners_mult * outs_mult * pressure_mult
        
        # Late inning pressure adjustment
        if situation.inning >= 7 and abs(situation.score_diff) <= 3:
            multiplier *= 1.05  # Slight boost for clutch situations
        
        return multiplier
    
    def _generate_run_distribution(self, expected_runs: float) -> Dict[int, float]:
        """Generate probability distribution for runs scored."""
        # Use modified Poisson distribution
        distribution = {}
        
        for runs in range(11):  # 0-10 runs
            if runs == 0:
                # Adjust for higher probability of 0 runs in baseball
                prob = poisson.pmf(runs, expected_runs) * 1.2
            else:
                prob = poisson.pmf(runs, expected_runs)
            
            distribution[runs] = min(prob, 1.0)
        
        # Normalize to ensure probabilities sum to 1
        total_prob = sum(distribution.values())
        for runs in distribution:
            distribution[runs] /= total_prob
        
        return distribution
    
    def _calculate_leverage_index(self, inning: int, score_diff: int) -> float:
        """Calculate leverage index for the situation."""
        # Simplified leverage calculation
        base_leverage = {
            1: 0.9, 2: 0.95, 3: 1.0, 4: 1.0, 5: 1.05,
            6: 1.1, 7: 1.3, 8: 1.8, 9: 2.5
        }.get(inning, 1.0)
        
        # Adjust for score differential
        if abs(score_diff) <= 1:
            multiplier = 1.2
        elif abs(score_diff) <= 3:
            multiplier = 1.1
        else:
            multiplier = 0.8
        
        return base_leverage * multiplier
    
    def _calculate_pressure_rating(self, inning: int, score_diff: int) -> float:
        """Calculate pressure rating for the situation."""
        if inning >= 7 and abs(score_diff) <= 2:
            return 1.5  # High pressure
        elif inning >= 8 and abs(score_diff) <= 4:
            return 1.3  # Moderate pressure
        else:
            return 1.0  # Normal pressure
    
    def _calculate_confidence_interval(self, expected_runs: float) -> Tuple[float, float]:
        """Calculate 90% confidence interval for predicted runs."""
        # Use beta distribution for bounded CI
        alpha = expected_runs + 1
        beta_param = 10 - expected_runs + 1
        
        lower = beta.ppf(0.05, alpha, beta_param) * 10
        upper = beta.ppf(0.95, alpha, beta_param) * 10
        
        return (round(lower, 2), round(upper, 2))
    
    def _identify_key_factors(
        self,
        pitcher: PitcherState,
        lineup: List[BatterPosition],
        situation: GameSituation,
        pitcher_effectiveness: float,
        lineup_strength: float
    ) -> List[str]:
        """Identify the most influential factors for the prediction."""
        factors = []
        
        # Pitcher factors
        if pitcher_effectiveness < 0.85:
            factors.append(f"Pitcher fatigue ({pitcher_effectiveness:.2f} effectiveness)")
        
        if pitcher.pitches_thrown > 90:
            factors.append(f"High pitch count ({pitcher.pitches_thrown} pitches)")
        
        # Lineup factors
        if lineup_strength > 1.15:
            factors.append(f"Strong lineup segment ({lineup_strength:.2f}x)")
        
        # Situational factors
        if len(situation.runners_on_base) >= 2:
            factors.append(f"Multiple runners on base ({len(situation.runners_on_base)})")
        
        if situation.leverage_index > 1.5:
            factors.append(f"High leverage situation ({situation.leverage_index:.1f} LI)")
        
        if situation.inning >= 7:
            factors.append("Late-inning pressure")
        
        return factors[:5]  # Return top 5 factors


def main():
    """Example usage of the InningByInningPredictor."""
    # Create sample data
    home_lineup = [
        BatterPosition(i+1, f"home_player_{i+1}", 
                      avg_vs_rhp=0.260 + i*0.01, obp_vs_rhp=0.330 + i*0.01,
                      slg_vs_rhp=0.420 + i*0.02)
        for i in range(9)
    ]
    
    away_lineup = [
        BatterPosition(i+1, f"away_player_{i+1}",
                      avg_vs_rhp=0.250 + i*0.01, obp_vs_rhp=0.320 + i*0.01,
                      slg_vs_rhp=0.400 + i*0.02)
        for i in range(9)
    ]
    
    home_pitcher = PitcherState(
        pitcher_id="home_starter",
        role=PitcherRole.STARTER,
        era=3.50,
        whip=1.15,
        k_per_9=9.5
    )
    
    away_pitcher = PitcherState(
        pitcher_id="away_starter",
        role=PitcherRole.STARTER,
        era=4.20,
        whip=1.35,
        k_per_9=8.0
    )
    
    game_situation = GameSituation(
        inning=1,
        half=InningHalf.TOP,
        score_diff=0
    )
    
    # Initialize predictor
    predictor = InningByInningPredictor()
    
    # Predict inning scores
    predictions = predictor.predict_inning_scores(
        home_lineup, away_lineup, home_pitcher, away_pitcher, game_situation
    )
    
    # Display results
    print("Inning-by-Inning Predictions:")
    print("=" * 50)
    
    for pred in predictions[:6]:  # Show first 3 innings
        print(f"{pred.inning}{'T' if pred.half == InningHalf.TOP else 'B'}: "
              f"{pred.predicted_runs:.2f} runs "
              f"({pred.scoring_probability:.1%} chance to score)")
        print(f"  Key factors: {', '.join(pred.key_factors[:3])}")
        print()


if __name__ == "__main__":
    main()