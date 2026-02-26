"""
Model Confidence Calibration System

This module provides comprehensive prediction confidence calibration for MLB
betting models, ensuring predicted probabilities accurately reflect actual
outcomes through various calibration techniques and reliability analysis.

Author: MLB Predictor System
Created: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Available calibration methods."""
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    BETA_CALIBRATION = "beta_calibration"
    HISTOGRAM_BINNING = "histogram_binning"
    TEMPERATURE_SCALING = "temperature_scaling"


class PredictionType(Enum):
    """Types of predictions to calibrate."""
    WIN_PROBABILITY = "win_probability"
    TOTAL_PROBABILITY = "total_probability"
    SPREAD_PROBABILITY = "spread_probability"
    PROP_PROBABILITY = "prop_probability"


@dataclass
class PredictionRecord:
    """Record of a prediction and its outcome."""
    prediction_id: str
    predicted_probability: float
    actual_outcome: bool  # True if prediction was correct
    prediction_type: PredictionType
    confidence_score: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def prediction_error(self) -> float:
        """Calculate absolute prediction error."""
        return abs(self.predicted_probability - (1.0 if self.actual_outcome else 0.0))


@dataclass
class CalibrationBin:
    """Represents a bin in the calibration analysis."""
    bin_id: int
    min_probability: float
    max_probability: float
    predicted_probabilities: List[float] = field(default_factory=list)
    actual_outcomes: List[bool] = field(default_factory=list)
    
    @property
    def bin_center(self) -> float:
        """Center point of the bin."""
        return (self.min_probability + self.max_probability) / 2
    
    @property
    def mean_predicted_probability(self) -> float:
        """Mean predicted probability in this bin."""
        return np.mean(self.predicted_probabilities) if self.predicted_probabilities else 0.0
    
    @property
    def actual_frequency(self) -> float:
        """Actual frequency of positive outcomes in this bin."""
        return np.mean(self.actual_outcomes) if self.actual_outcomes else 0.0
    
    @property
    def sample_size(self) -> int:
        """Number of predictions in this bin."""
        return len(self.predicted_probabilities)
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval for actual frequency."""
        if self.sample_size < 5:
            return (0.0, 1.0)
        
        p = self.actual_frequency
        n = self.sample_size
        z = 1.96  # 95% CI
        
        margin = z * np.sqrt(p * (1 - p) / n)
        return (max(0, p - margin), min(1, p + margin))


@dataclass
class ReliabilityDiagram:
    """Reliability diagram data and statistics."""
    bins: List[CalibrationBin]
    calibration_error: float  # Expected Calibration Error (ECE)
    max_calibration_error: float  # Maximum Calibration Error (MCE)
    brier_score: float
    log_loss_score: float
    reliability_score: float  # Overall reliability (0-1, higher is better)
    overconfidence_ratio: float  # Ratio of overconfident predictions
    diagram_path: Optional[str] = None


@dataclass
class CalibrationModel:
    """Trained calibration model."""
    method: CalibrationMethod
    model_object: Any  # Sklearn model or custom calibrator
    training_data_size: int
    calibration_score: float  # Performance on validation set
    creation_timestamp: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class ConfidenceInterval:
    """Confidence interval for a prediction."""
    prediction_id: str
    lower_bound: float
    upper_bound: float
    confidence_level: float  # 0.90, 0.95, etc.
    method: str  # Method used to calculate interval
    
    @property
    def interval_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound


@dataclass
class CalibrationDrift:
    """Tracks calibration performance over time."""
    time_period: str  # "daily", "weekly", "monthly"
    timestamps: List[datetime] = field(default_factory=list)
    calibration_errors: List[float] = field(default_factory=list)
    brier_scores: List[float] = field(default_factory=list)
    sample_sizes: List[int] = field(default_factory=list)
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    last_recalibration: Optional[datetime] = None


class ModelConfidenceCalibrator:
    """
    Comprehensive model confidence calibration system.
    
    This class provides multiple calibration methods to ensure that predicted
    probabilities accurately reflect the true likelihood of events, with
    ongoing monitoring of calibration performance and automatic recalibration
    when drift is detected.
    """
    
    def __init__(
        self,
        default_bins: int = 10,
        min_bin_size: int = 20,
        calibration_threshold: float = 0.05,  # ECE threshold for recalibration
        drift_window_days: int = 30,
        auto_recalibrate: bool = True
    ):
        """
        Initialize the confidence calibrator.
        
        Args:
            default_bins: Default number of bins for reliability analysis
            min_bin_size: Minimum sample size per bin
            calibration_threshold: ECE threshold triggering recalibration
            drift_window_days: Days to look back for drift detection
            auto_recalibrate: Automatically recalibrate when drift detected
        """
        self.default_bins = default_bins
        self.min_bin_size = min_bin_size
        self.calibration_threshold = calibration_threshold
        self.drift_window_days = drift_window_days
        self.auto_recalibrate = auto_recalibrate
        
        # Data storage
        self.prediction_history: List[PredictionRecord] = []
        self.calibration_models: Dict[PredictionType, List[CalibrationModel]] = defaultdict(list)
        self.reliability_diagrams: Dict[PredictionType, ReliabilityDiagram] = {}
        self.drift_monitoring: Dict[PredictionType, CalibrationDrift] = {}
        
        # Performance tracking
        self.calibration_metrics: Dict[str, Dict] = defaultdict(dict)
        self.recalibration_history: List[Dict] = []
        
        logger.info("Model Confidence Calibrator initialized")
    
    def add_prediction_record(
        self,
        prediction_id: str,
        predicted_probability: float,
        actual_outcome: bool,
        prediction_type: PredictionType,
        confidence_score: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a prediction record for future calibration analysis.
        
        Args:
            prediction_id: Unique identifier for the prediction
            predicted_probability: Model's predicted probability (0-1)
            actual_outcome: Whether the prediction was correct
            prediction_type: Type of prediction
            confidence_score: Optional confidence score
            metadata: Additional metadata about the prediction
        """
        record = PredictionRecord(
            prediction_id=prediction_id,
            predicted_probability=predicted_probability,
            actual_outcome=actual_outcome,
            prediction_type=prediction_type,
            confidence_score=confidence_score,
            metadata=metadata or {}
        )
        
        self.prediction_history.append(record)
        
        # Check for drift if enough data accumulated
        if len(self.prediction_history) % 100 == 0:  # Check every 100 predictions
            self._check_calibration_drift(prediction_type)
    
    def calibrate_predictions(
        self,
        prediction_type: PredictionType,
        method: CalibrationMethod = CalibrationMethod.PLATT_SCALING,
        min_samples: int = 100,
        validation_split: float = 0.2
    ) -> CalibrationModel:
        """
        Train a calibration model on historical predictions.
        
        Args:
            prediction_type: Type of predictions to calibrate
            method: Calibration method to use
            min_samples: Minimum number of samples required
            validation_split: Fraction of data for validation
            
        Returns:
            Trained calibration model
        """
        # Get relevant prediction records
        relevant_records = [
            record for record in self.prediction_history
            if record.prediction_type == prediction_type
        ]
        
        if len(relevant_records) < min_samples:
            raise ValueError(f"Insufficient data: {len(relevant_records)} samples, need {min_samples}")
        
        # Prepare training data
        probabilities = np.array([r.predicted_probability for r in relevant_records])
        outcomes = np.array([r.actual_outcome for r in relevant_records])
        
        # Split data
        split_idx = int(len(relevant_records) * (1 - validation_split))
        train_probs, val_probs = probabilities[:split_idx], probabilities[split_idx:]
        train_outcomes, val_outcomes = outcomes[:split_idx], outcomes[split_idx:]
        
        # Train calibration model
        if method == CalibrationMethod.PLATT_SCALING:
            calibrator = self._train_platt_scaling(train_probs, train_outcomes)
        elif method == CalibrationMethod.ISOTONIC_REGRESSION:
            calibrator = self._train_isotonic_regression(train_probs, train_outcomes)
        elif method == CalibrationMethod.BETA_CALIBRATION:
            calibrator = self._train_beta_calibration(train_probs, train_outcomes)
        elif method == CalibrationMethod.HISTOGRAM_BINNING:
            calibrator = self._train_histogram_binning(train_probs, train_outcomes)
        elif method == CalibrationMethod.TEMPERATURE_SCALING:
            calibrator = self._train_temperature_scaling(train_probs, train_outcomes)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Validate calibration model
        calibrated_probs = self._apply_calibration(calibrator, val_probs, method)
        validation_score = self._calculate_calibration_score(calibrated_probs, val_outcomes)
        
        # Create calibration model object
        calibration_model = CalibrationModel(
            method=method,
            model_object=calibrator,
            training_data_size=len(train_probs),
            calibration_score=validation_score
        )
        
        # Store calibration model
        self.calibration_models[prediction_type].append(calibration_model)
        
        # Keep only the best 3 models per type
        self.calibration_models[prediction_type] = sorted(
            self.calibration_models[prediction_type],
            key=lambda x: x.calibration_score,
            reverse=True
        )[:3]
        
        logger.info(f"Calibration model trained: {method.value} for {prediction_type.value}")
        logger.info(f"Validation score: {validation_score:.4f}")
        
        return calibration_model
    
    def get_confidence_interval(
        self,
        predicted_probability: float,
        prediction_type: PredictionType,
        confidence_level: float = 0.90,
        method: str = "bootstrap"
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for a prediction.
        
        Args:
            predicted_probability: Raw model prediction
            prediction_type: Type of prediction
            confidence_level: Confidence level (0.90, 0.95, etc.)
            method: Method for calculating interval
            
        Returns:
            ConfidenceInterval object
        """
        # Get relevant historical data
        relevant_records = [
            record for record in self.prediction_history
            if record.prediction_type == prediction_type
        ]
        
        if len(relevant_records) < 50:
            # Insufficient data - use wide interval
            margin = (1 - confidence_level) / 2
            return ConfidenceInterval(
                prediction_id=f"ci_{int(datetime.now().timestamp())}",
                lower_bound=max(0, predicted_probability - margin),
                upper_bound=min(1, predicted_probability + margin),
                confidence_level=confidence_level,
                method="default_wide"
            )
        
        if method == "bootstrap":
            return self._bootstrap_confidence_interval(
                predicted_probability, relevant_records, confidence_level
            )
        elif method == "bayesian":
            return self._bayesian_confidence_interval(
                predicted_probability, relevant_records, confidence_level
            )
        elif method == "empirical":
            return self._empirical_confidence_interval(
                predicted_probability, relevant_records, confidence_level
            )
        else:
            raise ValueError(f"Unknown confidence interval method: {method}")
    
    def generate_reliability_diagram(
        self,
        prediction_type: PredictionType,
        n_bins: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> ReliabilityDiagram:
        """
        Generate reliability diagram for a prediction type.
        
        Args:
            prediction_type: Type of predictions to analyze
            n_bins: Number of bins (defaults to class setting)
            save_path: Path to save diagram image
            
        Returns:
            ReliabilityDiagram object with analysis results
        """
        n_bins = n_bins or self.default_bins
        
        # Get relevant records
        relevant_records = [
            record for record in self.prediction_history
            if record.prediction_type == prediction_type
        ]
        
        if len(relevant_records) < self.min_bin_size:
            raise ValueError(f"Insufficient data for reliability analysis: {len(relevant_records)} samples")
        
        # Create bins
        bins = self._create_calibration_bins(relevant_records, n_bins)
        
        # Calculate metrics
        predicted_probs = np.array([r.predicted_probability for r in relevant_records])
        actual_outcomes = np.array([r.actual_outcome for r in relevant_records])
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_expected_calibration_error(bins)
        
        # Maximum Calibration Error (MCE)
        mce = max(abs(bin.mean_predicted_probability - bin.actual_frequency) 
                 for bin in bins if bin.sample_size > 0)
        
        # Brier Score
        brier_score = brier_score_loss(actual_outcomes, predicted_probs)
        
        # Log Loss
        # Clip probabilities to avoid log(0)
        clipped_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)
        log_loss_score = log_loss(actual_outcomes, clipped_probs)
        
        # Reliability Score (1 - normalized ECE)
        reliability_score = max(0, 1 - ece / 0.1)  # Normalize by 10% ECE
        
        # Overconfidence Ratio
        overconfident_predictions = sum(
            1 for r in relevant_records 
            if r.predicted_probability > 0.5 and not r.actual_outcome
        )
        total_confident_predictions = sum(
            1 for r in relevant_records if r.predicted_probability > 0.5
        )
        overconfidence_ratio = (overconfident_predictions / max(total_confident_predictions, 1))
        
        # Create reliability diagram
        reliability_diagram = ReliabilityDiagram(
            bins=bins,
            calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier_score,
            log_loss_score=log_loss_score,
            reliability_score=reliability_score,
            overconfidence_ratio=overconfidence_ratio
        )
        
        # Generate and save plot if requested
        if save_path:
            self._plot_reliability_diagram(reliability_diagram, save_path)
            reliability_diagram.diagram_path = save_path
        
        # Store diagram
        self.reliability_diagrams[prediction_type] = reliability_diagram
        
        return reliability_diagram
    
    def apply_calibration(
        self,
        raw_predictions: List[float],
        prediction_type: PredictionType
    ) -> List[float]:
        """
        Apply calibration to raw model predictions.
        
        Args:
            raw_predictions: List of raw model predictions (0-1)
            prediction_type: Type of predictions
            
        Returns:
            List of calibrated predictions
        """
        # Get best calibration model for this type
        if prediction_type not in self.calibration_models or not self.calibration_models[prediction_type]:
            logger.warning(f"No calibration model available for {prediction_type.value}")
            return raw_predictions
        
        best_model = self.calibration_models[prediction_type][0]  # Sorted by score
        
        # Apply calibration
        calibrated_predictions = self._apply_calibration(
            best_model.model_object,
            np.array(raw_predictions),
            best_model.method
        )
        
        return calibrated_predictions.tolist()
    
    def monitor_calibration_drift(
        self,
        prediction_type: PredictionType,
        time_window_days: Optional[int] = None
    ) -> CalibrationDrift:
        """
        Monitor calibration performance over time to detect drift.
        
        Args:
            prediction_type: Type of predictions to monitor
            time_window_days: Time window for analysis
            
        Returns:
            CalibrationDrift object with drift analysis
        """
        time_window_days = time_window_days or self.drift_window_days
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        # Get recent records
        recent_records = [
            record for record in self.prediction_history
            if (record.prediction_type == prediction_type and 
                record.timestamp >= cutoff_date)
        ]
        
        if len(recent_records) < 50:
            # Insufficient data for drift analysis
            return CalibrationDrift(time_period=f"{time_window_days}_days")
        
        # Group records by time periods (weekly)
        time_groups = defaultdict(list)
        for record in recent_records:
            # Week key: YYYY-WW
            week_key = record.timestamp.strftime("%Y-%W")
            time_groups[week_key].append(record)
        
        # Calculate calibration metrics for each time period
        timestamps = []
        calibration_errors = []
        brier_scores = []
        sample_sizes = []
        
        for week_key in sorted(time_groups.keys()):
            week_records = time_groups[week_key]
            
            if len(week_records) < 10:  # Skip weeks with too few samples
                continue
            
            # Calculate ECE for this week
            bins = self._create_calibration_bins(week_records, 5)  # Fewer bins for weekly data
            ece = self._calculate_expected_calibration_error(bins)
            
            # Calculate Brier score
            predicted_probs = np.array([r.predicted_probability for r in week_records])
            actual_outcomes = np.array([r.actual_outcome for r in week_records])
            brier_score = brier_score_loss(actual_outcomes, predicted_probs)
            
            # Add to tracking
            timestamps.append(datetime.strptime(week_key + '-1', "%Y-%W-%w"))
            calibration_errors.append(ece)
            brier_scores.append(brier_score)
            sample_sizes.append(len(week_records))
        
        # Detect drift
        drift_detected = False
        drift_magnitude = 0.0
        
        if len(calibration_errors) >= 4:  # Need at least 4 weeks
            # Check if recent calibration error significantly worse than earlier
            recent_error = np.mean(calibration_errors[-2:])  # Last 2 weeks
            earlier_error = np.mean(calibration_errors[:2])   # First 2 weeks
            
            drift_magnitude = recent_error - earlier_error
            if drift_magnitude > self.calibration_threshold:
                drift_detected = True
        
        # Create drift object
        drift = CalibrationDrift(
            time_period="weekly",
            timestamps=timestamps,
            calibration_errors=calibration_errors,
            brier_scores=brier_scores,
            sample_sizes=sample_sizes,
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude
        )
        
        # Update monitoring
        self.drift_monitoring[prediction_type] = drift
        
        # Auto-recalibrate if drift detected
        if drift_detected and self.auto_recalibrate:
            logger.warning(f"Calibration drift detected for {prediction_type.value}: {drift_magnitude:.4f}")
            try:
                self.calibrate_predictions(prediction_type)
                drift.last_recalibration = datetime.now()
                logger.info(f"Auto-recalibration completed for {prediction_type.value}")
            except Exception as e:
                logger.error(f"Auto-recalibration failed: {e}")
        
        return drift
    
    def get_calibration_summary(self, prediction_type: PredictionType) -> Dict[str, Any]:
        """Get comprehensive calibration summary for a prediction type."""
        
        relevant_records = [
            record for record in self.prediction_history
            if record.prediction_type == prediction_type
        ]
        
        if not relevant_records:
            return {"error": "No prediction records found"}
        
        summary = {
            "prediction_type": prediction_type.value,
            "total_predictions": len(relevant_records),
            "date_range": {
                "start": min(r.timestamp for r in relevant_records).isoformat(),
                "end": max(r.timestamp for r in relevant_records).isoformat()
            }
        }
        
        # Basic performance metrics
        predicted_probs = np.array([r.predicted_probability for r in relevant_records])
        actual_outcomes = np.array([r.actual_outcome for r in relevant_records])
        
        summary["performance_metrics"] = {
            "accuracy": np.mean(
                (predicted_probs > 0.5) == actual_outcomes
            ),
            "brier_score": brier_score_loss(actual_outcomes, predicted_probs),
            "log_loss": log_loss(actual_outcomes, np.clip(predicted_probs, 1e-15, 1-1e-15)),
            "mean_prediction": np.mean(predicted_probs),
            "actual_rate": np.mean(actual_outcomes)
        }
        
        # Calibration models
        if prediction_type in self.calibration_models:
            models_info = []
            for model in self.calibration_models[prediction_type]:
                models_info.append({
                    "method": model.method.value,
                    "score": model.calibration_score,
                    "training_size": model.training_data_size,
                    "created": model.creation_timestamp.isoformat(),
                    "active": model.is_active
                })
            summary["calibration_models"] = models_info
        
        # Reliability diagram
        if prediction_type in self.reliability_diagrams:
            diagram = self.reliability_diagrams[prediction_type]
            summary["reliability"] = {
                "calibration_error": diagram.calibration_error,
                "max_calibration_error": diagram.max_calibration_error,
                "reliability_score": diagram.reliability_score,
                "overconfidence_ratio": diagram.overconfidence_ratio
            }
        
        # Drift monitoring
        if prediction_type in self.drift_monitoring:
            drift = self.drift_monitoring[prediction_type]
            summary["drift_monitoring"] = {
                "drift_detected": drift.drift_detected,
                "drift_magnitude": drift.drift_magnitude,
                "monitoring_periods": len(drift.timestamps),
                "last_recalibration": drift.last_recalibration.isoformat() if drift.last_recalibration else None
            }
        
        return summary
    
    def _train_platt_scaling(self, probabilities: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Train Platt scaling calibration."""
        # Convert probabilities to logits
        eps = 1e-15
        logits = np.log(np.clip(probabilities, eps, 1-eps) / np.clip(1-probabilities, eps, 1-eps))
        
        # Fit sigmoid: P = 1 / (1 + exp(-(A * logit + B)))
        def objective(params):
            A, B = params
            calibrated_probs = 1 / (1 + np.exp(-(A * logits + B)))
            return log_loss(outcomes, np.clip(calibrated_probs, eps, 1-eps))
        
        result = minimize(objective, [1.0, 0.0], method='BFGS')
        
        return {"A": result.x[0], "B": result.x[1]}
    
    def _train_isotonic_regression(self, probabilities: np.ndarray, outcomes: np.ndarray):
        """Train isotonic regression calibration."""
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(probabilities, outcomes)
        return calibrator
    
    def _train_beta_calibration(self, probabilities: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Train beta calibration."""
        # Fit beta distribution parameters
        def objective(params):
            a, b, c = params
            if a <= 0 or b <= 0:
                return np.inf
            
            # Beta calibration: f(p) = p^a / (p^a + (1-p)^b)^c
            calibrated_probs = np.power(probabilities, a) / np.power(
                np.power(probabilities, a) + np.power(1 - probabilities, b), c
            )
            calibrated_probs = np.clip(calibrated_probs, 1e-15, 1-1e-15)
            return log_loss(outcomes, calibrated_probs)
        
        result = minimize(objective, [1.0, 1.0, 1.0], method='L-BFGS-B',
                         bounds=[(0.1, 10), (0.1, 10), (0.1, 2)])
        
        return {"a": result.x[0], "b": result.x[1], "c": result.x[2]}
    
    def _train_histogram_binning(self, probabilities: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Train histogram binning calibration."""
        n_bins = min(10, len(probabilities) // 20)  # Adaptive number of bins
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_calibration = {}
        for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (probabilities >= lower) & (probabilities < upper)
            if i == len(bin_lowers) - 1:  # Last bin includes upper boundary
                in_bin = (probabilities >= lower) & (probabilities <= upper)
            
            if np.sum(in_bin) > 0:
                bin_calibration[i] = {
                    'lower': lower,
                    'upper': upper,
                    'calibrated_prob': np.mean(outcomes[in_bin])
                }
            else:
                bin_calibration[i] = {
                    'lower': lower,
                    'upper': upper,
                    'calibrated_prob': (lower + upper) / 2
                }
        
        return bin_calibration
    
    def _train_temperature_scaling(self, probabilities: np.ndarray, outcomes: np.ndarray) -> Dict:
        """Train temperature scaling calibration."""
        eps = 1e-15
        logits = np.log(np.clip(probabilities, eps, 1-eps) / np.clip(1-probabilities, eps, 1-eps))
        
        def objective(temperature):
            if temperature <= 0:
                return np.inf
            
            calibrated_logits = logits / temperature
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
            return log_loss(outcomes, np.clip(calibrated_probs, eps, 1-eps))
        
        result = minimize(objective, [1.0], method='BFGS', bounds=[(0.1, 10)])
        
        return {"temperature": result.x[0]}
    
    def _apply_calibration(
        self,
        calibrator: Any,
        probabilities: np.ndarray,
        method: CalibrationMethod
    ) -> np.ndarray:
        """Apply calibration to probabilities."""
        eps = 1e-15
        
        if method == CalibrationMethod.PLATT_SCALING:
            A, B = calibrator["A"], calibrator["B"]
            logits = np.log(np.clip(probabilities, eps, 1-eps) / np.clip(1-probabilities, eps, 1-eps))
            calibrated_probs = 1 / (1 + np.exp(-(A * logits + B)))
            
        elif method == CalibrationMethod.ISOTONIC_REGRESSION:
            calibrated_probs = calibrator.predict(probabilities)
            
        elif method == CalibrationMethod.BETA_CALIBRATION:
            a, b, c = calibrator["a"], calibrator["b"], calibrator["c"]
            calibrated_probs = np.power(probabilities, a) / np.power(
                np.power(probabilities, a) + np.power(1 - probabilities, b), c
            )
            
        elif method == CalibrationMethod.HISTOGRAM_BINNING:
            calibrated_probs = np.zeros_like(probabilities)
            for i, bin_info in calibrator.items():
                in_bin = (probabilities >= bin_info['lower']) & (probabilities < bin_info['upper'])
                if i == len(calibrator) - 1:  # Last bin
                    in_bin = (probabilities >= bin_info['lower']) & (probabilities <= bin_info['upper'])
                calibrated_probs[in_bin] = bin_info['calibrated_prob']
                
        elif method == CalibrationMethod.TEMPERATURE_SCALING:
            temperature = calibrator["temperature"]
            logits = np.log(np.clip(probabilities, eps, 1-eps) / np.clip(1-probabilities, eps, 1-eps))
            calibrated_logits = logits / temperature
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
            
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return np.clip(calibrated_probs, eps, 1-eps)
    
    def _calculate_calibration_score(self, calibrated_probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate calibration performance score."""
        # Combine multiple metrics for overall score
        brier = brier_score_loss(outcomes, calibrated_probs)
        log_loss_score = log_loss(outcomes, np.clip(calibrated_probs, 1e-15, 1-1e-15))
        
        # Create bins for ECE calculation
        bins = self._create_calibration_bins_from_arrays(calibrated_probs, outcomes, 10)
        ece = self._calculate_expected_calibration_error(bins)
        
        # Composite score (lower is better, so we use 1 - normalized_score)
        normalized_brier = brier  # Already 0-1
        normalized_log_loss = min(1.0, log_loss_score / 2.0)  # Cap at 2.0
        normalized_ece = ece  # Already 0-1
        
        composite_score = 1.0 - (normalized_brier * 0.4 + normalized_log_loss * 0.3 + normalized_ece * 0.3)
        
        return max(0.0, composite_score)
    
    def _create_calibration_bins(self, records: List[PredictionRecord], n_bins: int) -> List[CalibrationBin]:
        """Create calibration bins from prediction records."""
        probabilities = np.array([r.predicted_probability for r in records])
        outcomes = np.array([r.actual_outcome for r in records])
        
        return self._create_calibration_bins_from_arrays(probabilities, outcomes, n_bins)
    
    def _create_calibration_bins_from_arrays(
        self,
        probabilities: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int
    ) -> List[CalibrationBin]:
        """Create calibration bins from arrays."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bins = []
        
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            if i == n_bins - 1:  # Last bin includes upper boundary
                in_bin = (probabilities >= lower) & (probabilities <= upper)
            else:
                in_bin = (probabilities >= lower) & (probabilities < upper)
            
            bin_probs = probabilities[in_bin].tolist()
            bin_outcomes = outcomes[in_bin].tolist()
            
            calibration_bin = CalibrationBin(
                bin_id=i,
                min_probability=lower,
                max_probability=upper,
                predicted_probabilities=bin_probs,
                actual_outcomes=bin_outcomes
            )
            
            bins.append(calibration_bin)
        
        return bins
    
    def _calculate_expected_calibration_error(self, bins: List[CalibrationBin]) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        total_samples = sum(bin.sample_size for bin in bins)
        
        if total_samples == 0:
            return 0.0
        
        ece = 0.0
        for bin in bins:
            if bin.sample_size > 0:
                bin_weight = bin.sample_size / total_samples
                calibration_error = abs(bin.mean_predicted_probability - bin.actual_frequency)
                ece += bin_weight * calibration_error
        
        return ece
    
    def _bootstrap_confidence_interval(
        self,
        predicted_probability: float,
        records: List[PredictionRecord],
        confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate bootstrap confidence interval."""
        # Find similar predictions (within 10% probability range)
        similar_records = [
            r for r in records
            if abs(r.predicted_probability - predicted_probability) <= 0.1
        ]
        
        if len(similar_records) < 10:
            # Expand search range
            similar_records = [
                r for r in records
                if abs(r.predicted_probability - predicted_probability) <= 0.2
            ]
        
        if len(similar_records) < 5:
            # Default wide interval
            margin = (1 - confidence_level) / 2
            return ConfidenceInterval(
                prediction_id=f"bootstrap_{int(datetime.now().timestamp())}",
                lower_bound=max(0, predicted_probability - margin),
                upper_bound=min(1, predicted_probability + margin),
                confidence_level=confidence_level,
                method="bootstrap_default"
            )
        
        # Bootstrap resampling
        n_bootstrap = 1000
        outcomes = [r.actual_outcome for r in similar_records]
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return ConfidenceInterval(
            prediction_id=f"bootstrap_{int(datetime.now().timestamp())}",
            lower_bound=max(0, lower_bound),
            upper_bound=min(1, upper_bound),
            confidence_level=confidence_level,
            method="bootstrap"
        )
    
    def _bayesian_confidence_interval(
        self,
        predicted_probability: float,
        records: List[PredictionRecord],
        confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate Bayesian confidence interval using Beta distribution."""
        # Find similar predictions
        similar_records = [
            r for r in records
            if abs(r.predicted_probability - predicted_probability) <= 0.15
        ]
        
        if len(similar_records) < 5:
            return self._bootstrap_confidence_interval(predicted_probability, records, confidence_level)
        
        # Count successes and failures
        successes = sum(1 for r in similar_records if r.actual_outcome)
        failures = len(similar_records) - successes
        
        # Beta distribution parameters (with weak prior)
        alpha = successes + 1
        beta = failures + 1
        
        # Calculate confidence interval
        alpha_level = 1 - confidence_level
        lower_bound = stats.beta.ppf(alpha_level / 2, alpha, beta)
        upper_bound = stats.beta.ppf(1 - alpha_level / 2, alpha, beta)
        
        return ConfidenceInterval(
            prediction_id=f"bayesian_{int(datetime.now().timestamp())}",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method="bayesian"
        )
    
    def _empirical_confidence_interval(
        self,
        predicted_probability: float,
        records: List[PredictionRecord],
        confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate empirical confidence interval."""
        # Use actual observed frequencies in probability bins
        bin_width = 0.1
        bin_center = round(predicted_probability / bin_width) * bin_width
        
        bin_records = [
            r for r in records
            if abs(r.predicted_probability - bin_center) <= bin_width / 2
        ]
        
        if len(bin_records) < 10:
            return self._bootstrap_confidence_interval(predicted_probability, records, confidence_level)
        
        # Calculate empirical frequency and confidence interval
        success_rate = np.mean([r.actual_outcome for r in bin_records])
        n = len(bin_records)
        
        # Wilson score interval
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center = (success_rate + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((success_rate * (1 - success_rate) + z**2 / (4 * n)) / n) / denominator
        
        lower_bound = max(0, center - margin)
        upper_bound = min(1, center + margin)
        
        return ConfidenceInterval(
            prediction_id=f"empirical_{int(datetime.now().timestamp())}",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            method="empirical"
        )
    
    def _plot_reliability_diagram(self, diagram: ReliabilityDiagram, save_path: str):
        """Plot and save reliability diagram."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Reliability plot
            bin_centers = [bin.bin_center for bin in diagram.bins if bin.sample_size > 0]
            mean_predicted = [bin.mean_predicted_probability for bin in diagram.bins if bin.sample_size > 0]
            actual_freq = [bin.actual_frequency for bin in diagram.bins if bin.sample_size > 0]
            
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            ax1.scatter(mean_predicted, actual_freq, s=50, alpha=0.7, label='Observed')
            ax1.set_xlabel('Mean Predicted Probability')
            ax1.set_ylabel('Actual Frequency')
            ax1.set_title(f'Reliability Diagram\nECE: {diagram.calibration_error:.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histogram
            sample_sizes = [bin.sample_size for bin in diagram.bins]
            ax2.bar(range(len(sample_sizes)), sample_sizes, alpha=0.7)
            ax2.set_xlabel('Bin')
            ax2.set_ylabel('Number of Predictions')
            ax2.set_title('Prediction Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot reliability diagram: {e}")
    
    def _check_calibration_drift(self, prediction_type: PredictionType):
        """Check for calibration drift and trigger recalibration if needed."""
        try:
            drift = self.monitor_calibration_drift(prediction_type)
            
            if drift.drift_detected:
                logger.warning(f"Calibration drift detected for {prediction_type.value}")
                
                # Log drift event
                drift_event = {
                    "timestamp": datetime.now().isoformat(),
                    "prediction_type": prediction_type.value,
                    "drift_magnitude": drift.drift_magnitude,
                    "auto_recalibrated": self.auto_recalibrate
                }
                self.recalibration_history.append(drift_event)
        
        except Exception as e:
            logger.error(f"Error checking calibration drift: {e}")


def main():
    """Example usage of the ModelConfidenceCalibrator."""
    
    # Initialize calibrator
    calibrator = ModelConfidenceCalibrator(
        default_bins=10,
        calibration_threshold=0.03,
        auto_recalibrate=True
    )
    
    # Generate sample prediction data
    np.random.seed(42)
    n_predictions = 1000
    
    print("Generating sample predictions...")
    
    for i in range(n_predictions):
        # Simulate predictions with systematic bias (overconfident)
        true_prob = np.random.beta(2, 2)  # True underlying probability
        predicted_prob = min(0.95, true_prob * 1.2 + 0.1)  # Overconfident predictions
        actual_outcome = np.random.random() < true_prob
        
        calibrator.add_prediction_record(
            prediction_id=f"pred_{i}",
            predicted_probability=predicted_prob,
            actual_outcome=actual_outcome,
            prediction_type=PredictionType.WIN_PROBABILITY,
            metadata={"model_version": "v1.0"}
        )
    
    print(f"Added {n_predictions} prediction records")
    
    # Train calibration model
    print("\nTraining calibration models...")
    
    # Try different calibration methods
    methods_to_test = [
        CalibrationMethod.PLATT_SCALING,
        CalibrationMethod.ISOTONIC_REGRESSION,
        CalibrationMethod.HISTOGRAM_BINNING
    ]
    
    for method in methods_to_test:
        try:
            model = calibrator.calibrate_predictions(
                PredictionType.WIN_PROBABILITY,
                method=method,
                min_samples=100
            )
            print(f"  {method.value}: Score = {model.calibration_score:.4f}")
        except Exception as e:
            print(f"  {method.value}: Failed - {e}")
    
    # Generate reliability diagram
    print("\nGenerating reliability diagram...")
    
    try:
        reliability_diagram = calibrator.generate_reliability_diagram(
            PredictionType.WIN_PROBABILITY,
            save_path="/tmp/reliability_diagram.png"
        )
        
        print(f"Reliability metrics:")
        print(f"  Expected Calibration Error: {reliability_diagram.calibration_error:.4f}")
        print(f"  Reliability Score: {reliability_diagram.reliability_score:.4f}")
        print(f"  Brier Score: {reliability_diagram.brier_score:.4f}")
        print(f"  Overconfidence Ratio: {reliability_diagram.overconfidence_ratio:.4f}")
        
    except Exception as e:
        print(f"Reliability diagram failed: {e}")
    
    # Test confidence intervals
    print("\nTesting confidence intervals...")
    
    test_probabilities = [0.3, 0.5, 0.7, 0.9]
    for prob in test_probabilities:
        ci = calibrator.get_confidence_interval(
            predicted_probability=prob,
            prediction_type=PredictionType.WIN_PROBABILITY,
            confidence_level=0.90
        )
        print(f"  P={prob:.1f}: CI = [{ci.lower_bound:.3f}, {ci.upper_bound:.3f}] "
              f"(width: {ci.interval_width:.3f})")
    
    # Test calibration application
    print("\nTesting calibration application...")
    
    raw_predictions = [0.2, 0.4, 0.6, 0.8]
    calibrated_predictions = calibrator.apply_calibration(
        raw_predictions,
        PredictionType.WIN_PROBABILITY
    )
    
    print("Raw -> Calibrated:")
    for raw, calibrated in zip(raw_predictions, calibrated_predictions):
        print(f"  {raw:.2f} -> {calibrated:.3f}")
    
    # Get calibration summary
    print("\nCalibration Summary:")
    summary = calibrator.get_calibration_summary(PredictionType.WIN_PROBABILITY)
    
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"Accuracy: {summary['performance_metrics']['accuracy']:.3f}")
    print(f"Brier Score: {summary['performance_metrics']['brier_score']:.4f}")
    
    if 'calibration_models' in summary:
        print(f"Calibration models: {len(summary['calibration_models'])}")
        for model in summary['calibration_models']:
            print(f"  {model['method']}: {model['score']:.4f}")


if __name__ == "__main__":
    main()