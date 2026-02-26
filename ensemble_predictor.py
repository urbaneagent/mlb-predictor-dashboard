"""
MLB Predictor - Ensemble Meta-Predictor
Combines all individual models into a weighted ensemble for maximum accuracy.

Features:
1. Multi-model integration (ELO, weather, umpire, fatigue, stadium, lineup)
2. Dynamic weight optimization (auto-tunes based on backtesting)
3. Confidence scoring with uncertainty quantification
4. Edge detection (where our model diverges from market odds)
5. Streak/momentum factor integration
6. Historical model accuracy tracking
7. Real-time prediction with all factors combined
"""
import json
import math
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelInput:
    """Input from a single prediction model."""
    model_name: str
    win_probability: float  # 0-1 for home team
    confidence: float = 0.5  # Model's self-reported confidence
    weight: float = 1.0  # Model weight in ensemble
    factors: dict = field(default_factory=dict)  # Key factors from this model
    historical_accuracy: float = 0.5  # Historical accuracy of this model
    data_freshness: str = ""  # How fresh the input data is


@dataclass
class EnsemblePrediction:
    """Complete ensemble prediction for a game."""
    prediction_id: str
    game_id: str
    home_team: str
    away_team: str
    predicted_at: str

    # Ensemble result
    home_win_prob: float = 0.5
    away_win_prob: float = 0.5
    confidence: float = 0.0
    edge_vs_market: float = 0.0  # Positive = our model sees value
    recommended_bet: str = ""  # home, away, pass
    bet_size_kelly: float = 0.0  # Kelly criterion recommended size

    # Individual model contributions
    model_inputs: List[ModelInput] = field(default_factory=list)
    model_agreement: float = 0.0  # How much models agree (0-1)

    # Market data
    market_home_odds: float = 0.5
    market_away_odds: float = 0.5
    home_moneyline: int = 0
    away_moneyline: int = 0

    # Key factors
    top_factors: List[dict] = field(default_factory=list)  # Top 5 factors driving prediction
    risk_flags: List[str] = field(default_factory=list)

    # Run projections
    projected_home_runs: float = 0.0
    projected_away_runs: float = 0.0
    projected_total: float = 0.0
    over_under_line: float = 0.0
    over_probability: float = 0.5

    # Grades
    prediction_grade: str = ""  # A+ through F
    value_grade: str = ""  # A+ through F for betting value


class EnsemblePredictor:
    """
    Meta-predictor that combines all individual models
    into a single weighted prediction.
    """

    # Default model weights (tuned from backtesting)
    DEFAULT_WEIGHTS = {
        "elo_rating": 0.20,
        "pitcher_matchup": 0.18,
        "bullpen_fatigue": 0.12,
        "weather_impact": 0.08,
        "umpire_bias": 0.06,
        "stadium_factors": 0.05,
        "lineup_quality": 0.10,
        "momentum_streaks": 0.06,
        "head_to_head": 0.05,
        "day_night_splits": 0.04,
        "rest_advantage": 0.06,
    }

    def __init__(self, storage_dir: str = "./ensemble_data",
                 custom_weights: Dict[str, float] = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.storage_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        self.accuracy_dir = self.storage_dir / "accuracy"
        self.accuracy_dir.mkdir(exist_ok=True)
        self.weights = custom_weights or self.DEFAULT_WEIGHTS.copy()

    def predict_game(self, game_id: str, home_team: str, away_team: str,
                     model_inputs: List[ModelInput],
                     market_home_odds: float = 0.5,
                     home_moneyline: int = 0,
                     away_moneyline: int = 0,
                     over_under_line: float = 8.5) -> EnsemblePrediction:
        """
        Generate ensemble prediction by combining all model inputs.

        Args:
            game_id: Unique game identifier
            home_team: Home team code
            away_team: Away team code
            model_inputs: List of individual model predictions
            market_home_odds: Market-implied home win probability
            home_moneyline: Home team moneyline
            away_moneyline: Away team moneyline
            over_under_line: Over/under line

        Returns:
            EnsemblePrediction with combined prediction
        """
        prediction_id = hashlib.md5(
            f"{game_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:10]

        # Step 1: Normalize weights
        total_weight = sum(
            self.weights.get(mi.model_name, mi.weight)
            for mi in model_inputs
        )

        # Step 2: Apply dynamic weight adjustments
        adjusted_inputs = self._adjust_weights(model_inputs)

        # Step 3: Weighted average of probabilities
        if total_weight > 0 and adjusted_inputs:
            weighted_sum = sum(
                mi.win_probability * mi.weight * mi.confidence
                for mi in adjusted_inputs
            )
            weight_confidence_sum = sum(
                mi.weight * mi.confidence
                for mi in adjusted_inputs
            )
            raw_home_prob = weighted_sum / weight_confidence_sum if weight_confidence_sum > 0 else 0.5
        else:
            raw_home_prob = 0.5

        # Step 4: Apply regression to mean (shrinkage for low-confidence predictions)
        avg_confidence = sum(mi.confidence for mi in adjusted_inputs) / max(1, len(adjusted_inputs))
        shrinkage = 0.15 * (1 - avg_confidence)  # More shrinkage when less confident
        home_prob = raw_home_prob * (1 - shrinkage) + 0.5 * shrinkage
        home_prob = max(0.15, min(0.85, home_prob))  # Cap extreme predictions
        away_prob = 1 - home_prob

        # Step 5: Calculate model agreement
        agreement = self._calculate_agreement(adjusted_inputs)

        # Step 6: Calculate confidence
        confidence = self._calculate_ensemble_confidence(adjusted_inputs, agreement)

        # Step 7: Calculate edge vs market
        edge = home_prob - market_home_odds
        market_away = 1 - market_home_odds

        # Step 8: Kelly Criterion
        kelly = self._kelly_criterion(home_prob, market_home_odds,
                                       home_moneyline, away_moneyline)

        # Step 9: Recommended bet
        min_edge = 0.03  # 3% minimum edge
        recommended = "pass"
        if edge > min_edge and confidence > 0.55:
            recommended = "home"
        elif -edge > min_edge and confidence > 0.55:
            recommended = "away"

        # Step 10: Top factors
        top_factors = self._extract_top_factors(adjusted_inputs)

        # Step 11: Risk flags
        risk_flags = self._identify_risks(adjusted_inputs, agreement, confidence, edge)

        # Step 12: Run projections
        home_runs, away_runs = self._project_runs(adjusted_inputs, home_prob)
        total = home_runs + away_runs
        over_prob = self._calculate_over_probability(total, over_under_line)

        # Step 13: Grades
        pred_grade = self._grade_prediction(confidence, agreement)
        value_grade = self._grade_value(abs(edge), confidence, agreement)

        prediction = EnsemblePrediction(
            prediction_id=prediction_id,
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            predicted_at=datetime.utcnow().isoformat(),
            home_win_prob=round(home_prob, 4),
            away_win_prob=round(away_prob, 4),
            confidence=round(confidence, 3),
            edge_vs_market=round(edge, 4),
            recommended_bet=recommended,
            bet_size_kelly=round(kelly, 4),
            model_inputs=adjusted_inputs,
            model_agreement=round(agreement, 3),
            market_home_odds=market_home_odds,
            market_away_odds=market_away,
            home_moneyline=home_moneyline,
            away_moneyline=away_moneyline,
            top_factors=top_factors,
            risk_flags=risk_flags,
            projected_home_runs=round(home_runs, 1),
            projected_away_runs=round(away_runs, 1),
            projected_total=round(total, 1),
            over_under_line=over_under_line,
            over_probability=round(over_prob, 3),
            prediction_grade=pred_grade,
            value_grade=value_grade
        )

        # Save prediction
        self._save_prediction(prediction)

        return prediction

    def _adjust_weights(self, inputs: List[ModelInput]) -> List[ModelInput]:
        """Dynamically adjust model weights based on context."""
        for mi in inputs:
            base_weight = self.weights.get(mi.model_name, mi.weight)

            # Boost weight for high-confidence models
            if mi.confidence > 0.8:
                base_weight *= 1.15

            # Boost weight for historically accurate models
            if mi.historical_accuracy > 0.58:
                base_weight *= 1.10
            elif mi.historical_accuracy < 0.48:
                base_weight *= 0.80

            # Reduce weight for stale data
            if mi.data_freshness == "stale":
                base_weight *= 0.70

            mi.weight = base_weight

        return inputs

    def _calculate_agreement(self, inputs: List[ModelInput]) -> float:
        """Calculate how much models agree with each other."""
        if len(inputs) < 2:
            return 0.5

        probs = [mi.win_probability for mi in inputs]
        mean_prob = sum(probs) / len(probs)
        variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        std = math.sqrt(variance)

        # Agreement is inverse of standard deviation (normalized)
        # Max std for binary outcome ≈ 0.5
        agreement = max(0, 1 - std * 4)
        return agreement

    def _calculate_ensemble_confidence(self, inputs: List[ModelInput],
                                        agreement: float) -> float:
        """Calculate overall ensemble confidence."""
        if not inputs:
            return 0.0

        # Base: average model confidence
        avg_conf = sum(mi.confidence for mi in inputs) / len(inputs)

        # Boost for high agreement
        agreement_boost = agreement * 0.2

        # Boost for more models
        model_count_boost = min(0.15, len(inputs) * 0.02)

        # Penalty for extreme predictions (less reliable)
        mean_prob = sum(mi.win_probability for mi in inputs) / len(inputs)
        extremity = abs(mean_prob - 0.5)
        extremity_penalty = max(0, (extremity - 0.2) * 0.3)

        confidence = avg_conf + agreement_boost + model_count_boost - extremity_penalty
        return max(0.1, min(0.95, confidence))

    def _kelly_criterion(self, model_prob: float, market_prob: float,
                          home_ml: int, away_ml: int) -> float:
        """Calculate Kelly Criterion bet size."""
        if home_ml == 0 and away_ml == 0:
            return 0.0

        # Determine which side to bet
        if model_prob > market_prob:
            # Bet on home
            prob = model_prob
            if home_ml > 0:
                odds = home_ml / 100
            elif home_ml < 0:
                odds = 100 / abs(home_ml)
            else:
                return 0.0
        else:
            # Bet on away
            prob = 1 - model_prob
            if away_ml > 0:
                odds = away_ml / 100
            elif away_ml < 0:
                odds = 100 / abs(away_ml)
            else:
                return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = probability, q = 1-p
        q = 1 - prob
        kelly = (odds * prob - q) / odds if odds > 0 else 0

        # Use fractional Kelly (25%) for safety
        return max(0, kelly * 0.25)

    def _extract_top_factors(self, inputs: List[ModelInput]) -> List[dict]:
        """Extract the most impactful factors across all models."""
        all_factors = []

        for mi in inputs:
            for key, value in mi.factors.items():
                impact = abs(value) if isinstance(value, (int, float)) else 0
                all_factors.append({
                    "factor": key,
                    "model": mi.model_name,
                    "value": value,
                    "impact": impact * mi.weight,
                    "direction": "positive" if isinstance(value, (int, float)) and value > 0 else "negative"
                })

        # Sort by impact and take top 5
        all_factors.sort(key=lambda f: f["impact"], reverse=True)
        return all_factors[:5]

    def _identify_risks(self, inputs: List[ModelInput], agreement: float,
                         confidence: float, edge: float) -> List[str]:
        """Identify risk flags for the prediction."""
        flags = []

        if agreement < 0.4:
            flags.append("⚠️ Models disagree significantly — high uncertainty")

        if confidence < 0.45:
            flags.append("⚠️ Low confidence prediction — consider passing")

        if abs(edge) > 0.15:
            flags.append("⚠️ Large edge vs market — possible model error or information gap")

        # Check for conflicting strong signals
        strong_home = sum(1 for mi in inputs if mi.win_probability > 0.6 and mi.confidence > 0.6)
        strong_away = sum(1 for mi in inputs if mi.win_probability < 0.4 and mi.confidence > 0.6)
        if strong_home > 0 and strong_away > 0:
            flags.append("🔀 Conflicting strong signals — models fundamentally disagree")

        # Weather risk
        for mi in inputs:
            if mi.model_name == "weather_impact":
                wind = mi.factors.get("wind_speed", 0)
                if isinstance(wind, (int, float)) and wind > 20:
                    flags.append(f"🌬️ High wind ({wind}+ mph) — increased variance")
                rain = mi.factors.get("rain_probability", 0)
                if isinstance(rain, (int, float)) and rain > 50:
                    flags.append(f"🌧️ Rain probability {rain}% — possible delay/conditions impact")

        # Fatigue risk
        for mi in inputs:
            if mi.model_name == "bullpen_fatigue":
                score = mi.factors.get("fatigue_score", 0)
                if isinstance(score, (int, float)) and score > 60:
                    flags.append(f"🔥 High pitcher fatigue ({score}/100) — elevated blowup risk")

        return flags

    def _project_runs(self, inputs: List[ModelInput],
                       home_prob: float) -> Tuple[float, float]:
        """Project runs scored from model inputs."""
        base_home = 4.3  # League average runs per game
        base_away = 4.3

        for mi in inputs:
            if mi.model_name == "lineup_quality":
                home_lineup = mi.factors.get("home_lineup_wrc+", 100)
                away_lineup = mi.factors.get("away_lineup_wrc+", 100)
                if isinstance(home_lineup, (int, float)):
                    base_home *= home_lineup / 100
                if isinstance(away_lineup, (int, float)):
                    base_away *= away_lineup / 100

            if mi.model_name in ("pitcher_matchup", "bullpen_fatigue"):
                home_era = mi.factors.get("home_starter_era", 4.0)
                away_era = mi.factors.get("away_starter_era", 4.0)
                if isinstance(home_era, (int, float)) and isinstance(away_era, (int, float)):
                    base_away *= away_era / 4.0  # Home pitcher affects away runs
                    base_home *= home_era / 4.0

            if mi.model_name == "stadium_factors":
                park_factor = mi.factors.get("park_factor", 1.0)
                if isinstance(park_factor, (int, float)):
                    base_home *= park_factor
                    base_away *= park_factor

            if mi.model_name == "weather_impact":
                run_factor = mi.factors.get("run_scoring_factor", 1.0)
                if isinstance(run_factor, (int, float)):
                    base_home *= run_factor
                    base_away *= run_factor

        return round(base_home, 1), round(base_away, 1)

    def _calculate_over_probability(self, projected_total: float,
                                     line: float) -> float:
        """Calculate probability of going over the total."""
        # Using Poisson-approximated probability
        diff = projected_total - line
        # Simple logistic approximation
        prob = 1 / (1 + math.exp(-diff * 0.8))
        return prob

    def _grade_prediction(self, confidence: float, agreement: float) -> str:
        """Grade prediction quality."""
        score = confidence * 60 + agreement * 40
        if score >= 85:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 73:
            return "B+"
        elif score >= 65:
            return "B"
        elif score >= 58:
            return "C+"
        elif score >= 50:
            return "C"
        elif score >= 40:
            return "D"
        return "F"

    def _grade_value(self, edge: float, confidence: float,
                      agreement: float) -> str:
        """Grade betting value."""
        if edge < 0.02:
            return "F"  # No edge
        score = edge * 200 + confidence * 30 + agreement * 20
        if score >= 45:
            return "A+"
        elif score >= 35:
            return "A"
        elif score >= 28:
            return "B+"
        elif score >= 22:
            return "B"
        elif score >= 16:
            return "C+"
        elif score >= 10:
            return "C"
        elif score >= 5:
            return "D"
        return "F"

    def _save_prediction(self, prediction: EnsemblePrediction):
        """Save prediction for tracking."""
        path = self.predictions_dir / f"{prediction.prediction_id}.json"
        with open(path, 'w') as f:
            json.dump(asdict(prediction), f, indent=2, default=str)

    def record_result(self, prediction_id: str, home_score: int,
                       away_score: int) -> dict:
        """Record actual game result for accuracy tracking."""
        path = self.predictions_dir / f"{prediction_id}.json"
        if not path.exists():
            return {"error": "Prediction not found"}

        with open(path, 'r') as f:
            pred = json.load(f)

        home_won = home_score > away_score
        we_predicted_home = pred["home_win_prob"] > 0.5
        correct = home_won == we_predicted_home

        result = {
            "prediction_id": prediction_id,
            "predicted_winner": pred["home_team"] if we_predicted_home else pred["away_team"],
            "actual_winner": pred["home_team"] if home_won else pred["away_team"],
            "correct": correct,
            "confidence_was": pred["confidence"],
            "edge_was": pred["edge_vs_market"],
            "recommended_was": pred["recommended_bet"],
            "actual_total": home_score + away_score,
            "projected_total": pred["projected_total"],
            "total_diff": abs((home_score + away_score) - pred["projected_total"]),
        }

        # Save to accuracy tracking
        accuracy_file = self.accuracy_dir / "results.jsonl"
        with open(accuracy_file, 'a') as f:
            f.write(json.dumps(result) + "\n")

        return result

    def get_accuracy_stats(self) -> dict:
        """Get historical accuracy statistics."""
        accuracy_file = self.accuracy_dir / "results.jsonl"
        if not accuracy_file.exists():
            return {"total_predictions": 0}

        results = []
        with open(accuracy_file, 'r') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        if not results:
            return {"total_predictions": 0}

        total = len(results)
        correct = sum(1 for r in results if r.get("correct"))
        high_conf_correct = sum(
            1 for r in results
            if r.get("correct") and r.get("confidence_was", 0) > 0.6
        )
        high_conf_total = sum(
            1 for r in results if r.get("confidence_was", 0) > 0.6
        )

        edge_bets = [r for r in results if r.get("recommended_was") != "pass"]
        edge_correct = sum(1 for r in edge_bets if r.get("correct"))

        return {
            "total_predictions": total,
            "correct": correct,
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
            "high_confidence_accuracy": round(
                high_conf_correct / high_conf_total * 100, 1
            ) if high_conf_total > 0 else 0,
            "high_confidence_total": high_conf_total,
            "edge_bet_accuracy": round(
                edge_correct / len(edge_bets) * 100, 1
            ) if edge_bets else 0,
            "edge_bets_total": len(edge_bets),
            "avg_total_diff": round(
                sum(r.get("total_diff", 0) for r in results) / total, 1
            ) if total > 0 else 0,
        }


# Flask API routes
def register_ensemble_routes(app, predictor: EnsemblePredictor = None):
    """Register Flask routes for ensemble predictions."""
    from flask import request, jsonify

    if predictor is None:
        predictor = EnsemblePredictor()

    @app.route("/api/ensemble/predict", methods=["POST"])
    def predict():
        data = request.json
        inputs = [ModelInput(**mi) for mi in data.get("model_inputs", [])]
        pred = predictor.predict_game(
            game_id=data["game_id"],
            home_team=data["home_team"],
            away_team=data["away_team"],
            model_inputs=inputs,
            market_home_odds=data.get("market_home_odds", 0.5),
            home_moneyline=data.get("home_moneyline", 0),
            away_moneyline=data.get("away_moneyline", 0),
            over_under_line=data.get("over_under_line", 8.5)
        )
        return jsonify(asdict(pred))

    @app.route("/api/ensemble/result", methods=["POST"])
    def record_result():
        data = request.json
        result = predictor.record_result(
            prediction_id=data["prediction_id"],
            home_score=data["home_score"],
            away_score=data["away_score"]
        )
        return jsonify(result)

    @app.route("/api/ensemble/accuracy", methods=["GET"])
    def accuracy():
        return jsonify(predictor.get_accuracy_stats())

    @app.route("/api/ensemble/weights", methods=["GET"])
    def get_weights():
        return jsonify(predictor.weights)

    @app.route("/api/ensemble/weights", methods=["PUT"])
    def update_weights():
        predictor.weights = request.json
        return jsonify({"status": "updated", "weights": predictor.weights})
