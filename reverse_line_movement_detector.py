"""
Reverse Line Movement (RLM) Detector

This module detects reverse line movement in MLB betting markets, identifying
instances where betting lines move contrary to public betting percentages,
indicating sharp money action and professional betting influence.

Author: MLB Predictor System
Created: February 2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque
import statistics
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BetType(Enum):
    """Types of bets to monitor for RLM."""
    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    FIRST_FIVE = "first_five"


class LineDirection(Enum):
    """Direction of line movement."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class SharpnessLevel(Enum):
    """Classification of betting action sharpness."""
    PUBLIC = "public"
    MIXED = "mixed"
    SEMI_SHARP = "semi_sharp"
    SHARP = "sharp"
    SUPER_SHARP = "super_sharp"


@dataclass
class BookmakerLine:
    """Represents a betting line from a specific bookmaker."""
    book_id: str
    book_name: str
    line_value: float  # Odds, spread, or total
    juice: float = -110.0  # Vigorish/commission
    timestamp: datetime = field(default_factory=datetime.now)
    volume_indicator: Optional[float] = None  # Betting volume if available


@dataclass
class PublicBettingData:
    """Public betting percentage data."""
    bet_type: BetType
    team_or_side: str
    public_percentage: float  # 0-100
    ticket_percentage: float  # Percentage of tickets
    money_percentage: float  # Percentage of money
    sharp_percentage: Optional[float] = None  # If available
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LineMovement:
    """Represents a line movement event."""
    bet_type: BetType
    team_or_side: str
    opening_line: float
    current_line: float
    movement_amount: float
    movement_percentage: float
    direction: LineDirection
    book_id: str
    timestamp: datetime
    steam_move: bool = False  # Rapid, synchronized movement


@dataclass
class RLMEvent:
    """Represents a detected reverse line movement event."""
    game_id: str
    bet_type: BetType
    team_or_side: str
    line_movements: List[LineMovement]
    public_data: PublicBettingData
    rlm_strength: float  # 0-100 scale
    sharpness_level: SharpnessLevel
    detection_timestamp: datetime
    books_involved: Set[str]
    movement_velocity: float  # Speed of movement
    confidence_score: float  # 0-1
    historical_success_rate: Optional[float] = None


@dataclass
class SteamMove:
    """Represents a steam move (rapid line movement across books)."""
    bet_type: BetType
    team_or_side: str
    books_moved: List[str]
    movement_timespan: timedelta
    average_movement: float
    max_movement: float
    synchronization_score: float  # How synchronized the movement was
    timestamp: datetime


@dataclass
class SharpMoneyIndicator:
    """Indicators of sharp money betting."""
    reverse_line_movement: bool = False
    steam_move: bool = False
    low_public_high_money: bool = False  # Low public %, high money %
    line_origination: bool = False  # Did this book originate the move?
    professional_book_movement: bool = False  # Movement at sharp books
    overnight_movement: bool = False  # Movement during off-hours
    injury_news_correlation: bool = False
    weather_correlation: bool = False


@dataclass
class RLMAlert:
    """Alert for significant RLM events."""
    alert_id: str
    game_id: str
    rlm_event: RLMEvent
    alert_level: str  # "low", "medium", "high", "critical"
    recommended_action: str
    expected_value: Optional[float] = None
    historical_roi: Optional[float] = None
    alert_timestamp: datetime = field(default_factory=datetime.now)


class ReverseLineMovementDetector:
    """
    Advanced detector for reverse line movement in MLB betting markets.
    
    This class monitors betting lines across multiple sportsbooks and identifies
    instances where lines move contrary to public betting patterns, indicating
    sharp money influence and potential betting opportunities.
    """
    
    def __init__(
        self,
        books_to_monitor: List[str] = None,
        rlm_threshold: float = 15.0,  # Minimum % difference for RLM
        steam_threshold: float = 0.5,  # Hours for steam detection
        min_books_for_steam: int = 3,
        historical_data_path: Optional[str] = None
    ):
        """
        Initialize the RLM detector.
        
        Args:
            books_to_monitor: List of bookmaker IDs to monitor
            rlm_threshold: Minimum percentage difference to flag RLM
            steam_threshold: Maximum time for steam move detection (hours)
            min_books_for_steam: Minimum books required for steam detection
            historical_data_path: Path to historical RLM performance data
        """
        self.books_to_monitor = books_to_monitor or [
            "pinnacle", "betmgm", "fanduel", "draftkings", "caesars",
            "betrivers", "pointsbet", "barstool", "unibet", "bet365"
        ]
        self.rlm_threshold = rlm_threshold
        self.steam_threshold = steam_threshold
        self.min_books_for_steam = min_books_for_steam
        
        # Data storage
        self.current_lines: Dict[str, Dict[str, List[BookmakerLine]]] = defaultdict(lambda: defaultdict(list))
        self.public_betting_data: Dict[str, Dict[str, PublicBettingData]] = defaultdict(dict)
        self.line_history: Dict[str, List[LineMovement]] = defaultdict(list)
        self.detected_rlm_events: List[RLMEvent] = []
        self.steam_moves: List[SteamMove] = []
        
        # Alert system
        self.active_alerts: List[RLMAlert] = []
        self.alert_thresholds = {
            "low": 20.0,
            "medium": 35.0,
            "high": 50.0,
            "critical": 70.0
        }
        
        # Historical performance tracking
        self.historical_performance: Dict[str, Dict] = defaultdict(dict)
        if historical_data_path:
            self._load_historical_data(historical_data_path)
        
        # Sharp book identification (books known for sharp action)
        self.sharp_books = {"pinnacle", "bookmaker", "heritage", "betcris"}
        self.public_books = {"fanduel", "draftkings", "betmgm", "caesars"}
        
        logger.info(f"RLM Detector initialized monitoring {len(self.books_to_monitor)} books")
    
    def detect_rlm(
        self,
        game_id: str,
        current_lines: Dict[str, List[BookmakerLine]],
        public_data: Dict[str, PublicBettingData],
        lookback_hours: int = 24
    ) -> List[RLMEvent]:
        """
        Detect reverse line movement for a specific game.
        
        Args:
            game_id: Unique game identifier
            current_lines: Current lines by bet type
            public_data: Public betting data by bet type
            lookback_hours: Hours to look back for line movements
            
        Returns:
            List of detected RLM events
        """
        rlm_events = []
        
        # Update internal data
        self._update_line_data(game_id, current_lines)
        self._update_public_data(game_id, public_data)
        
        for bet_type, lines in current_lines.items():
            if bet_type in public_data:
                # Detect RLM for this bet type
                rlm_event = self._analyze_rlm_for_bet_type(
                    game_id, bet_type, lines, public_data[bet_type], lookback_hours
                )
                
                if rlm_event:
                    rlm_events.append(rlm_event)
                    self.detected_rlm_events.append(rlm_event)
                    
                    # Generate alert if significant
                    if rlm_event.rlm_strength >= self.alert_thresholds["low"]:
                        alert = self._generate_rlm_alert(game_id, rlm_event)
                        self.active_alerts.append(alert)
        
        return rlm_events
    
    def track_sharp_money(
        self,
        game_id: str,
        bet_type: BetType,
        time_window: int = 2  # hours
    ) -> SharpMoneyIndicator:
        """
        Track indicators of sharp money betting for a specific bet.
        
        Args:
            game_id: Game identifier
            bet_type: Type of bet to analyze
            time_window: Time window for analysis (hours)
            
        Returns:
            SharpMoneyIndicator with various sharp money signals
        """
        indicator = SharpMoneyIndicator()
        
        # Get recent line movements
        cutoff_time = datetime.now() - timedelta(hours=time_window)
        recent_movements = [
            movement for movement in self.line_history[game_id]
            if movement.bet_type == bet_type and movement.timestamp >= cutoff_time
        ]
        
        if not recent_movements:
            return indicator
        
        # Check for reverse line movement
        indicator.reverse_line_movement = self._has_reverse_line_movement(
            game_id, bet_type, recent_movements
        )
        
        # Check for steam moves
        indicator.steam_move = self._has_steam_move(recent_movements)
        
        # Check public vs money percentage discrepancy
        if game_id in self.public_betting_data and bet_type.value in self.public_betting_data[game_id]:
            public_data = self.public_betting_data[game_id][bet_type.value]
            indicator.low_public_high_money = (
                public_data.public_percentage < 40 and 
                public_data.money_percentage > 60
            )
        
        # Check line origination (did sharp books move first?)
        indicator.line_origination = self._check_line_origination(recent_movements)
        
        # Check professional book movement
        indicator.professional_book_movement = any(
            movement.book_id in self.sharp_books for movement in recent_movements
        )
        
        # Check for overnight movement (sharp money often bets off-hours)
        indicator.overnight_movement = any(
            movement.timestamp.hour < 8 or movement.timestamp.hour > 22
            for movement in recent_movements
        )
        
        return indicator
    
    def get_rlm_alerts(
        self,
        alert_level: Optional[str] = None,
        active_only: bool = True,
        lookback_hours: int = 24
    ) -> List[RLMAlert]:
        """
        Get RLM alerts based on criteria.
        
        Args:
            alert_level: Filter by alert level ("low", "medium", "high", "critical")
            active_only: Only return active alerts
            lookback_hours: Hours to look back for alerts
            
        Returns:
            Filtered list of RLM alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        filtered_alerts = []
        for alert in self.active_alerts:
            # Time filter
            if alert.alert_timestamp < cutoff_time and active_only:
                continue
            
            # Level filter
            if alert_level and alert.alert_level != alert_level:
                continue
            
            filtered_alerts.append(alert)
        
        # Sort by alert level priority and timestamp
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        filtered_alerts.sort(
            key=lambda x: (priority_order.get(x.alert_level, 4), -x.alert_timestamp.timestamp())
        )
        
        return filtered_alerts
    
    def analyze_historical_rlm_performance(
        self,
        lookback_days: int = 30,
        bet_type: Optional[BetType] = None
    ) -> Dict[str, Any]:
        """
        Analyze historical performance of RLM signals.
        
        Args:
            lookback_days: Days to analyze
            bet_type: Specific bet type to analyze (optional)
            
        Returns:
            Performance analysis dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        # Filter historical events
        relevant_events = [
            event for event in self.detected_rlm_events
            if event.detection_timestamp >= cutoff_date
            and (not bet_type or event.bet_type == bet_type)
        ]
        
        if not relevant_events:
            return {"error": "No historical data available"}
        
        # Calculate performance metrics
        win_rate = self._calculate_win_rate(relevant_events)
        roi = self._calculate_roi(relevant_events)
        avg_line_movement = self._calculate_avg_line_movement(relevant_events)
        strength_correlation = self._analyze_strength_correlation(relevant_events)
        
        performance = {
            "total_events": len(relevant_events),
            "win_rate": win_rate,
            "roi": roi,
            "average_line_movement": avg_line_movement,
            "strength_correlation": strength_correlation,
            "events_by_sharpness": self._group_events_by_sharpness(relevant_events),
            "events_by_bet_type": self._group_events_by_bet_type(relevant_events),
            "book_performance": self._analyze_book_performance(relevant_events)
        }
        
        return performance
    
    def _analyze_rlm_for_bet_type(
        self,
        game_id: str,
        bet_type: str,
        lines: List[BookmakerLine],
        public_data: PublicBettingData,
        lookback_hours: int
    ) -> Optional[RLMEvent]:
        """Analyze RLM for a specific bet type."""
        
        # Get line movements for this bet type
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        movements = [
            movement for movement in self.line_history[game_id]
            if movement.bet_type.value == bet_type and movement.timestamp >= cutoff_time
        ]
        
        if len(movements) < 2:  # Need at least 2 movements to detect RLM
            return None
        
        # Calculate average line movement
        avg_movement = statistics.mean([abs(m.movement_amount) for m in movements])
        
        # Determine primary movement direction
        total_movement = sum(m.movement_amount for m in movements)
        primary_direction = LineDirection.UP if total_movement > 0 else LineDirection.DOWN
        
        # Check for RLM - line moving against public
        rlm_detected = False
        rlm_strength = 0.0
        
        if public_data.public_percentage > 60 and primary_direction == LineDirection.DOWN:
            # Public on one side, line moving other way
            rlm_detected = True
            rlm_strength = min(100.0, (public_data.public_percentage - 50) * 2 + avg_movement * 10)
        elif public_data.public_percentage < 40 and primary_direction == LineDirection.UP:
            # Public fading, line still moving against public
            rlm_detected = True
            rlm_strength = min(100.0, (50 - public_data.public_percentage) * 2 + avg_movement * 10)
        
        if not rlm_detected or rlm_strength < self.rlm_threshold:
            return None
        
        # Classify sharpness level
        sharpness_level = self._classify_sharpness(rlm_strength, movements, public_data)
        
        # Calculate movement velocity
        time_span = max(movements, key=lambda x: x.timestamp).timestamp - min(movements, key=lambda x: x.timestamp).timestamp
        movement_velocity = avg_movement / max(time_span.total_seconds() / 3600, 0.1)  # per hour
        
        # Calculate confidence score
        confidence_score = min(1.0, (
            rlm_strength / 100 * 0.4 +
            len(set(m.book_id for m in movements)) / len(self.books_to_monitor) * 0.3 +
            movement_velocity / 10 * 0.3
        ))
        
        # Get historical success rate
        historical_success_rate = self._get_historical_success_rate(bet_type, rlm_strength)
        
        return RLMEvent(
            game_id=game_id,
            bet_type=BetType(bet_type),
            team_or_side=public_data.team_or_side,
            line_movements=movements,
            public_data=public_data,
            rlm_strength=rlm_strength,
            sharpness_level=sharpness_level,
            detection_timestamp=datetime.now(),
            books_involved=set(m.book_id for m in movements),
            movement_velocity=movement_velocity,
            confidence_score=confidence_score,
            historical_success_rate=historical_success_rate
        )
    
    def _has_reverse_line_movement(
        self,
        game_id: str,
        bet_type: BetType,
        movements: List[LineMovement]
    ) -> bool:
        """Check if there's reverse line movement."""
        if not movements or game_id not in self.public_betting_data:
            return False
        
        public_data = self.public_betting_data[game_id].get(bet_type.value)
        if not public_data:
            return False
        
        # Calculate net movement direction
        net_movement = sum(m.movement_amount for m in movements)
        
        # Check if movement is against public
        if public_data.public_percentage > 55 and net_movement < -0.5:
            return True
        elif public_data.public_percentage < 45 and net_movement > 0.5:
            return True
        
        return False
    
    def _has_steam_move(self, movements: List[LineMovement]) -> bool:
        """Check for steam move (rapid synchronized movement)."""
        if len(movements) < self.min_books_for_steam:
            return False
        
        # Group movements by time windows
        time_windows = defaultdict(list)
        for movement in movements:
            # 30-minute windows
            window_key = movement.timestamp.replace(minute=movement.timestamp.minute // 30 * 30, second=0, microsecond=0)
            time_windows[window_key].append(movement)
        
        # Look for synchronized movement
        for window_movements in time_windows.values():
            if len(window_movements) >= self.min_books_for_steam:
                # Check if movements are in same direction and significant
                directions = [m.direction for m in window_movements]
                avg_movement = statistics.mean([abs(m.movement_amount) for m in window_movements])
                
                if len(set(directions)) == 1 and avg_movement > 1.0:  # All same direction, significant movement
                    return True
        
        return False
    
    def _check_line_origination(self, movements: List[LineMovement]) -> bool:
        """Check if sharp books originated the line movement."""
        if not movements:
            return False
        
        # Sort movements by timestamp
        sorted_movements = sorted(movements, key=lambda x: x.timestamp)
        
        # Check if first few movements came from sharp books
        first_movements = sorted_movements[:3]
        sharp_moves = [m for m in first_movements if m.book_id in self.sharp_books]
        
        return len(sharp_moves) >= len(first_movements) * 0.6  # 60% from sharp books
    
    def _classify_sharpness(
        self,
        rlm_strength: float,
        movements: List[LineMovement],
        public_data: PublicBettingData
    ) -> SharpnessLevel:
        """Classify the sharpness level of the betting action."""
        
        # Factor in multiple indicators
        sharp_indicators = 0
        
        # RLM strength
        if rlm_strength > 60:
            sharp_indicators += 2
        elif rlm_strength > 40:
            sharp_indicators += 1
        
        # Public vs money percentage discrepancy
        if abs(public_data.public_percentage - public_data.money_percentage) > 20:
            sharp_indicators += 1
        
        # Sharp book involvement
        sharp_books_involved = len([m for m in movements if m.book_id in self.sharp_books])
        if sharp_books_involved >= 2:
            sharp_indicators += 1
        
        # Steam move indicator
        if len(set(m.book_id for m in movements)) >= self.min_books_for_steam:
            sharp_indicators += 1
        
        # Classify based on indicators
        if sharp_indicators >= 4:
            return SharpnessLevel.SUPER_SHARP
        elif sharp_indicators >= 3:
            return SharpnessLevel.SHARP
        elif sharp_indicators >= 2:
            return SharpnessLevel.SEMI_SHARP
        elif sharp_indicators >= 1:
            return SharpnessLevel.MIXED
        else:
            return SharpnessLevel.PUBLIC
    
    def _generate_rlm_alert(self, game_id: str, rlm_event: RLMEvent) -> RLMAlert:
        """Generate an alert for an RLM event."""
        
        # Determine alert level
        if rlm_event.rlm_strength >= self.alert_thresholds["critical"]:
            alert_level = "critical"
        elif rlm_event.rlm_strength >= self.alert_thresholds["high"]:
            alert_level = "high"
        elif rlm_event.rlm_strength >= self.alert_thresholds["medium"]:
            alert_level = "medium"
        else:
            alert_level = "low"
        
        # Generate recommended action
        recommended_action = self._generate_recommendation(rlm_event)
        
        # Calculate expected value if available
        expected_value = self._calculate_expected_value(rlm_event)
        
        # Get historical ROI for similar events
        historical_roi = self._get_historical_roi(rlm_event)
        
        alert_id = f"rlm_{game_id}_{rlm_event.bet_type.value}_{int(datetime.now().timestamp())}"
        
        return RLMAlert(
            alert_id=alert_id,
            game_id=game_id,
            rlm_event=rlm_event,
            alert_level=alert_level,
            recommended_action=recommended_action,
            expected_value=expected_value,
            historical_roi=historical_roi
        )
    
    def _generate_recommendation(self, rlm_event: RLMEvent) -> str:
        """Generate betting recommendation based on RLM event."""
        
        # Determine which side the sharp money is on
        net_movement = sum(m.movement_amount for m in rlm_event.line_movements)
        
        if rlm_event.bet_type == BetType.MONEYLINE:
            if net_movement < 0:  # Line moving down, bet the favorite
                return f"Consider betting {rlm_event.team_or_side} (favorite) - sharp money indication"
            else:  # Line moving up, bet the underdog
                return f"Consider betting {rlm_event.team_or_side} (underdog) - sharp money indication"
        
        elif rlm_event.bet_type == BetType.TOTAL:
            if net_movement < 0:  # Total moving down
                return f"Consider UNDER {rlm_event.current_line} - sharp money on under"
            else:  # Total moving up
                return f"Consider OVER {rlm_event.current_line} - sharp money on over"
        
        else:  # Spread
            if net_movement < 0:  # Spread moving down (more points)
                return f"Consider {rlm_event.team_or_side} +{abs(rlm_event.current_line)} - sharp money indication"
            else:  # Spread moving up (fewer points)
                return f"Consider {rlm_event.team_or_side} {rlm_event.current_line} - sharp money indication"
    
    def _calculate_expected_value(self, rlm_event: RLMEvent) -> Optional[float]:
        """Calculate expected value of the RLM signal."""
        # This would require access to actual outcomes and closing lines
        # For now, return an estimate based on RLM strength
        if rlm_event.historical_success_rate:
            return (rlm_event.historical_success_rate - 0.5) * 2  # Rough EV estimate
        return None
    
    def _get_historical_roi(self, rlm_event: RLMEvent) -> Optional[float]:
        """Get historical ROI for similar RLM events."""
        # Look for similar events in historical data
        similar_events = [
            event for event in self.detected_rlm_events
            if (event.bet_type == rlm_event.bet_type and
                abs(event.rlm_strength - rlm_event.rlm_strength) < 10 and
                event.sharpness_level == rlm_event.sharpness_level)
        ]
        
        if len(similar_events) >= 10:  # Need reasonable sample size
            # Calculate average ROI (simplified)
            return self._calculate_roi(similar_events)
        
        return None
    
    def _update_line_data(self, game_id: str, lines: Dict[str, List[BookmakerLine]]):
        """Update internal line data storage."""
        for bet_type, book_lines in lines.items():
            for line in book_lines:
                # Check for line movement
                previous_lines = self.current_lines[game_id][bet_type]
                previous_line = next(
                    (l for l in previous_lines if l.book_id == line.book_id),
                    None
                )
                
                if previous_line and abs(previous_line.line_value - line.line_value) > 0.01:
                    # Record line movement
                    movement = LineMovement(
                        bet_type=BetType(bet_type),
                        team_or_side="",  # Would need to determine from context
                        opening_line=previous_line.line_value,
                        current_line=line.line_value,
                        movement_amount=line.line_value - previous_line.line_value,
                        movement_percentage=((line.line_value - previous_line.line_value) / previous_line.line_value) * 100,
                        direction=LineDirection.UP if line.line_value > previous_line.line_value else LineDirection.DOWN,
                        book_id=line.book_id,
                        timestamp=line.timestamp
                    )
                    
                    self.line_history[game_id].append(movement)
                
                # Update current line
                # Remove old line from same book
                self.current_lines[game_id][bet_type] = [
                    l for l in self.current_lines[game_id][bet_type]
                    if l.book_id != line.book_id
                ]
                # Add new line
                self.current_lines[game_id][bet_type].append(line)
    
    def _update_public_data(self, game_id: str, public_data: Dict[str, PublicBettingData]):
        """Update public betting data storage."""
        self.public_betting_data[game_id].update(public_data)
    
    def _load_historical_data(self, path: str):
        """Load historical performance data."""
        try:
            with open(path, 'r') as f:
                self.historical_performance = json.load(f)
            logger.info(f"Loaded historical data from {path}")
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
    
    def _calculate_win_rate(self, events: List[RLMEvent]) -> float:
        """Calculate win rate for RLM events."""
        # Simplified calculation - would need actual outcomes
        return 0.58  # Placeholder
    
    def _calculate_roi(self, events: List[RLMEvent]) -> float:
        """Calculate ROI for RLM events."""
        # Simplified calculation - would need actual outcomes
        return 0.12  # Placeholder 12% ROI
    
    def _calculate_avg_line_movement(self, events: List[RLMEvent]) -> float:
        """Calculate average line movement for events."""
        movements = []
        for event in events:
            movements.extend([abs(m.movement_amount) for m in event.line_movements])
        return statistics.mean(movements) if movements else 0.0
    
    def _analyze_strength_correlation(self, events: List[RLMEvent]) -> float:
        """Analyze correlation between RLM strength and success."""
        # Would need actual outcomes for real correlation
        return 0.45  # Placeholder correlation coefficient
    
    def _group_events_by_sharpness(self, events: List[RLMEvent]) -> Dict[str, int]:
        """Group events by sharpness level."""
        groups = defaultdict(int)
        for event in events:
            groups[event.sharpness_level.value] += 1
        return dict(groups)
    
    def _group_events_by_bet_type(self, events: List[RLMEvent]) -> Dict[str, int]:
        """Group events by bet type."""
        groups = defaultdict(int)
        for event in events:
            groups[event.bet_type.value] += 1
        return dict(groups)
    
    def _analyze_book_performance(self, events: List[RLMEvent]) -> Dict[str, Dict]:
        """Analyze performance by bookmaker."""
        book_stats = defaultdict(lambda: {"events": 0, "avg_strength": 0.0})
        
        for event in events:
            for book in event.books_involved:
                book_stats[book]["events"] += 1
                book_stats[book]["avg_strength"] += event.rlm_strength
        
        # Calculate averages
        for book in book_stats:
            if book_stats[book]["events"] > 0:
                book_stats[book]["avg_strength"] /= book_stats[book]["events"]
        
        return dict(book_stats)
    
    def _get_historical_success_rate(self, bet_type: str, strength: float) -> Optional[float]:
        """Get historical success rate for similar RLM strength."""
        # Placeholder - would use actual historical data
        base_rate = 0.55
        strength_bonus = min(0.15, strength / 100 * 0.2)
        return base_rate + strength_bonus


def main():
    """Example usage of the ReverseLineMovementDetector."""
    
    # Initialize detector
    detector = ReverseLineMovementDetector()
    
    # Example game lines
    game_id = "mlb_2026_02_24_nyy_bos"
    
    current_lines = {
        "moneyline": [
            BookmakerLine("fanduel", "FanDuel", -120, -110, datetime.now()),
            BookmakerLine("draftkings", "DraftKings", -115, -110, datetime.now()),
            BookmakerLine("pinnacle", "Pinnacle", -118, -105, datetime.now()),
        ],
        "total": [
            BookmakerLine("fanduel", "FanDuel", 8.5, -110, datetime.now()),
            BookmakerLine("draftkings", "DraftKings", 8.0, -110, datetime.now()),
            BookmakerLine("pinnacle", "Pinnacle", 8.0, -108, datetime.now()),
        ]
    }
    
    public_data = {
        "moneyline": PublicBettingData(
            bet_type=BetType.MONEYLINE,
            team_or_side="Yankees",
            public_percentage=75.0,
            ticket_percentage=78.0,
            money_percentage=45.0  # RLM signal - public on Yankees but money on Red Sox
        ),
        "total": PublicBettingData(
            bet_type=BetType.TOTAL,
            team_or_side="Over",
            public_percentage=65.0,
            ticket_percentage=68.0,
            money_percentage=52.0
        )
    }
    
    # Detect RLM
    rlm_events = detector.detect_rlm(game_id, current_lines, public_data)
    
    # Display results
    print("Reverse Line Movement Detection Results:")
    print("=" * 50)
    
    for event in rlm_events:
        print(f"RLM Event: {event.bet_type.value}")
        print(f"Strength: {event.rlm_strength:.1f}")
        print(f"Sharpness: {event.sharpness_level.value}")
        print(f"Confidence: {event.confidence_score:.2f}")
        print(f"Books involved: {', '.join(event.books_involved)}")
        print()
    
    # Get alerts
    alerts = detector.get_rlm_alerts("medium")
    print(f"Active alerts: {len(alerts)}")
    
    # Track sharp money indicators
    sharp_indicators = detector.track_sharp_money(game_id, BetType.MONEYLINE)
    print(f"Sharp money indicators:")
    print(f"  RLM: {sharp_indicators.reverse_line_movement}")
    print(f"  Steam: {sharp_indicators.steam_move}")
    print(f"  Low public/High money: {sharp_indicators.low_public_high_money}")
    

if __name__ == "__main__":
    main()