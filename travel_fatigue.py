"""
MLB Predictor - Travel Fatigue & Schedule Advantage Model
Quantifies travel, timezone, and schedule effects on team performance.

Features:
1. Travel distance calculation between stadiums
2. Timezone crossing penalty (jet lag effect)
3. Back-to-back game fatigue
4. Day game after night game penalty
5. West-to-East travel impact (harder)
6. Homestand vs road trip length effects
7. Off-day advantage quantification
8. Cross-country series travel burden
"""
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# Stadium locations (lat, lon, timezone offset from ET)
STADIUMS = {
    "NYY": {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "tz": 0, "city": "New York"},
    "NYM": {"name": "Citi Field", "lat": 40.7571, "lon": -73.8458, "tz": 0, "city": "New York"},
    "BOS": {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "tz": 0, "city": "Boston"},
    "BAL": {"name": "Camden Yards", "lat": 39.2838, "lon": -76.6216, "tz": 0, "city": "Baltimore"},
    "TB":  {"name": "Tropicana Field", "lat": 27.7682, "lon": -82.6534, "tz": 0, "city": "St. Petersburg"},
    "TOR": {"name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "tz": 0, "city": "Toronto"},
    "CLE": {"name": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "tz": 0, "city": "Cleveland"},
    "DET": {"name": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "tz": 0, "city": "Detroit"},
    "KC":  {"name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "tz": -1, "city": "Kansas City"},
    "MIN": {"name": "Target Field", "lat": 44.9818, "lon": -93.2775, "tz": -1, "city": "Minneapolis"},
    "CWS": {"name": "Guaranteed Rate", "lat": 41.8299, "lon": -87.6338, "tz": -1, "city": "Chicago"},
    "HOU": {"name": "Minute Maid Park", "lat": 29.7573, "lon": -95.3555, "tz": -1, "city": "Houston"},
    "TEX": {"name": "Globe Life Field", "lat": 32.7473, "lon": -97.0845, "tz": -1, "city": "Arlington"},
    "LAA": {"name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "tz": -3, "city": "Anaheim"},
    "OAK": {"name": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005, "tz": -3, "city": "Oakland"},
    "SEA": {"name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "tz": -3, "city": "Seattle"},
    "ATL": {"name": "Truist Park", "lat": 33.8908, "lon": -84.4678, "tz": 0, "city": "Atlanta"},
    "MIA": {"name": "loanDepot Park", "lat": 25.7781, "lon": -80.2197, "tz": 0, "city": "Miami"},
    "PHI": {"name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "tz": 0, "city": "Philadelphia"},
    "WSH": {"name": "Nationals Park", "lat": 38.8731, "lon": -77.0074, "tz": 0, "city": "Washington"},
    "CHC": {"name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "tz": -1, "city": "Chicago"},
    "CIN": {"name": "Great American", "lat": 39.0974, "lon": -84.5085, "tz": 0, "city": "Cincinnati"},
    "MIL": {"name": "American Family", "lat": 43.0280, "lon": -87.9712, "tz": -1, "city": "Milwaukee"},
    "PIT": {"name": "PNC Park", "lat": 40.4469, "lon": -80.0057, "tz": 0, "city": "Pittsburgh"},
    "STL": {"name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "tz": -1, "city": "St. Louis"},
    "ARI": {"name": "Chase Field", "lat": 33.4454, "lon": -112.0667, "tz": -2, "city": "Phoenix"},
    "COL": {"name": "Coors Field", "lat": 39.7559, "lon": -104.9942, "tz": -2, "city": "Denver"},
    "LAD": {"name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "tz": -3, "city": "Los Angeles"},
    "SD":  {"name": "Petco Park", "lat": 32.7076, "lon": -117.1570, "tz": -3, "city": "San Diego"},
    "SF":  {"name": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "tz": -3, "city": "San Francisco"},
}


@dataclass
class TravelInfo:
    """Travel information between two locations."""
    from_team: str
    to_team: str
    distance_miles: float
    timezone_change: int  # Positive = eastward
    direction: str  # east, west, same_tz
    estimated_flight_hours: float
    travel_burden_score: float  # 0-100


@dataclass
class ScheduleContext:
    """Schedule context for a team heading into a game."""
    team: str
    games_last_7_days: int = 0
    games_last_10_days: int = 0
    off_days_last_7: int = 0
    consecutive_games: int = 0
    is_day_after_night: bool = False
    is_back_to_back: bool = False
    homestand_length: int = 0  # 0 = road game
    road_trip_length: int = 0  # 0 = home game
    total_miles_last_7_days: float = 0
    timezone_changes_last_7: int = 0
    arrived_city_hours_ago: float = 24


@dataclass
class TravelFatigueAdjustment:
    """Win probability adjustment based on travel/schedule factors."""
    team: str
    adjustment: float  # Percentage adjustment to win probability
    travel_factor: float = 0.0
    timezone_factor: float = 0.0
    schedule_factor: float = 0.0
    rest_factor: float = 0.0
    daynight_factor: float = 0.0
    homestand_factor: float = 0.0
    composite_score: float = 0.0  # 0-100, lower = more fatigued
    details: List[str] = field(default_factory=list)
    confidence: float = 0.7


class TravelFatigueModel:
    """
    Models the impact of travel, timezone changes, and schedule
    density on MLB team performance.
    """

    # Research-based impact factors
    TIMEZONE_PENALTY_PER_HOUR = 0.008  # ~0.8% per timezone crossed
    WEST_TO_EAST_MULTIPLIER = 1.4      # Eastward travel is harder
    DISTANCE_PENALTY_PER_1000MI = 0.005  # 0.5% per 1000 miles
    DAY_AFTER_NIGHT_PENALTY = 0.015    # 1.5% for day game after night game
    BACK_TO_BACK_PENALTY = 0.005       # 0.5% for back-to-back
    NO_OFF_DAY_PENALTY_PER_GAME = 0.002  # Cumulative per consecutive game
    LONG_ROAD_TRIP_PENALTY = 0.003      # Per game on road trip > 6 games
    SHORT_ARRIVAL_PENALTY = 0.012       # Arrived < 12 hours before game
    HOME_ADVANTAGE_BASE = 0.04          # ~4% home field advantage

    def calculate_travel(self, from_team: str, to_team: str) -> TravelInfo:
        """Calculate travel distance and burden between two stadiums."""
        from_info = STADIUMS.get(from_team)
        to_info = STADIUMS.get(to_team)

        if not from_info or not to_info:
            return TravelInfo(from_team=from_team, to_team=to_team,
                              distance_miles=0, timezone_change=0,
                              direction="unknown", estimated_flight_hours=0,
                              travel_burden_score=0)

        # Haversine distance
        distance = self._haversine(
            from_info["lat"], from_info["lon"],
            to_info["lat"], to_info["lon"]
        )

        # Timezone change
        tz_change = to_info["tz"] - from_info["tz"]
        direction = "east" if tz_change > 0 else "west" if tz_change < 0 else "same_tz"

        # Flight time estimate (avg 500 mph + 1hr overhead)
        flight_hours = distance / 500 + 1 if distance > 200 else distance / 60  # Drive if close

        # Travel burden score
        burden = min(100, (
            distance / 30 +
            abs(tz_change) * 15 +
            (10 if direction == "east" else 0)
        ))

        return TravelInfo(
            from_team=from_team,
            to_team=to_team,
            distance_miles=round(distance, 0),
            timezone_change=tz_change,
            direction=direction,
            estimated_flight_hours=round(flight_hours, 1),
            travel_burden_score=round(burden, 1)
        )

    def calculate_adjustment(self, team: str, opponent: str,
                              schedule: ScheduleContext,
                              is_home: bool = True,
                              travel: TravelInfo = None) -> TravelFatigueAdjustment:
        """
        Calculate composite travel fatigue adjustment.

        Args:
            team: Team code
            opponent: Opponent team code
            schedule: Schedule context for the team
            is_home: Whether this team is at home
            travel: Travel info (calculated if not provided)
        """
        adjustment = 0.0
        details = []

        # Home advantage
        homestand_factor = 0.0
        if is_home:
            homestand_factor = self.HOME_ADVANTAGE_BASE
            if schedule.homestand_length >= 7:
                homestand_factor += 0.005  # Long homestand bonus
                details.append(f"Long homestand ({schedule.homestand_length} games) +0.5%")
        else:
            homestand_factor = -self.HOME_ADVANTAGE_BASE * 0.3  # Partial road penalty
            if schedule.road_trip_length >= 7:
                homestand_factor -= self.LONG_ROAD_TRIP_PENALTY * (schedule.road_trip_length - 6)
                details.append(f"Long road trip ({schedule.road_trip_length} games)")

        # Travel distance factor
        travel_factor = 0.0
        timezone_factor = 0.0
        if travel and not is_home:
            # Distance penalty
            travel_factor = -(travel.distance_miles / 1000) * self.DISTANCE_PENALTY_PER_1000MI
            if travel.distance_miles > 1500:
                details.append(f"Long travel: {travel.distance_miles:.0f} miles")

            # Timezone penalty
            tz_hours = abs(travel.timezone_change)
            if tz_hours > 0:
                tz_penalty = tz_hours * self.TIMEZONE_PENALTY_PER_HOUR
                if travel.direction == "east":
                    tz_penalty *= self.WEST_TO_EAST_MULTIPLIER
                    details.append(f"Eastward travel ({tz_hours}h timezone change)")
                else:
                    details.append(f"Westward travel ({tz_hours}h timezone change)")
                timezone_factor = -tz_penalty

            # Short arrival penalty
            if schedule.arrived_city_hours_ago < 12:
                travel_factor -= self.SHORT_ARRIVAL_PENALTY
                details.append(f"Arrived only {schedule.arrived_city_hours_ago:.0f}h before game")

        # Schedule density factor
        schedule_factor = 0.0
        if schedule.consecutive_games > 10:
            penalty = (schedule.consecutive_games - 10) * self.NO_OFF_DAY_PENALTY_PER_GAME
            schedule_factor -= penalty
            details.append(f"No off-day in {schedule.consecutive_games} games")

        if schedule.off_days_last_7 >= 2:
            schedule_factor += 0.005
            details.append("Well-rested (2+ off-days in last week)")

        # Rest factor
        rest_factor = 0.0
        if schedule.is_back_to_back:
            rest_factor -= self.BACK_TO_BACK_PENALTY
            details.append("Back-to-back games")

        # Day-after-night penalty
        daynight_factor = 0.0
        if schedule.is_day_after_night:
            daynight_factor -= self.DAY_AFTER_NIGHT_PENALTY
            details.append("Day game after night game (-1.5%)")

        # High mileage week
        if schedule.total_miles_last_7_days > 5000:
            travel_factor -= 0.008
            details.append(f"Heavy travel week ({schedule.total_miles_last_7_days:.0f} miles)")

        # Composite
        adjustment = (travel_factor + timezone_factor + schedule_factor +
                       rest_factor + daynight_factor + homestand_factor)

        # Composite score (0-100, higher = better shape)
        composite = 75  # Baseline
        composite += adjustment * 500  # Scale adjustment to score
        composite = max(20, min(100, composite))

        return TravelFatigueAdjustment(
            team=team,
            adjustment=round(adjustment * 100, 2),  # As percentage
            travel_factor=round(travel_factor * 100, 2),
            timezone_factor=round(timezone_factor * 100, 2),
            schedule_factor=round(schedule_factor * 100, 2),
            rest_factor=round(rest_factor * 100, 2),
            daynight_factor=round(daynight_factor * 100, 2),
            homestand_factor=round(homestand_factor * 100, 2),
            composite_score=round(composite, 1),
            details=details,
            confidence=0.72
        )

    def compare_matchup(self, home_team: str, away_team: str,
                         home_schedule: ScheduleContext,
                         away_schedule: ScheduleContext,
                         away_previous_city: str = None) -> dict:
        """
        Compare travel/schedule advantage for a matchup.
        """
        # Calculate travel for away team
        from_city = away_previous_city or away_team
        travel = self.calculate_travel(from_city, home_team)

        home_adj = self.calculate_adjustment(
            home_team, away_team, home_schedule, is_home=True
        )
        away_adj = self.calculate_adjustment(
            away_team, home_team, away_schedule, is_home=False, travel=travel
        )

        net_advantage = home_adj.adjustment - away_adj.adjustment

        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_adjustment": asdict(home_adj),
            "away_adjustment": asdict(away_adj),
            "travel_info": asdict(travel),
            "net_home_advantage": round(net_advantage, 2),
            "advantage_team": home_team if net_advantage > 0 else away_team,
            "advantage_description": (
                f"{home_team if net_advantage > 0 else away_team} has "
                f"{abs(net_advantage):.1f}% schedule/travel advantage"
            ),
            "key_factors": (home_adj.details + away_adj.details)[:5],
        }

    def _haversine(self, lat1: float, lon1: float,
                    lat2: float, lon2: float) -> float:
        """Calculate great-circle distance in miles."""
        R = 3959  # Earth radius in miles
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def get_all_distances(self) -> dict:
        """Get distance matrix between all stadiums."""
        distances = {}
        teams = list(STADIUMS.keys())
        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
                info = self.calculate_travel(team_a, team_b)
                distances[f"{team_a}-{team_b}"] = {
                    "distance": info.distance_miles,
                    "timezone_change": info.timezone_change,
                    "burden": info.travel_burden_score,
                }
        return distances


# Flask API routes
def register_travel_routes(app, model: TravelFatigueModel = None):
    from flask import request, jsonify

    if model is None:
        model = TravelFatigueModel()

    @app.route("/api/travel/distance", methods=["GET"])
    def travel_distance():
        from_team = request.args.get("from")
        to_team = request.args.get("to")
        info = model.calculate_travel(from_team, to_team)
        return jsonify(asdict(info))

    @app.route("/api/travel/matchup", methods=["POST"])
    def matchup_analysis():
        data = request.json
        home_sched = ScheduleContext(**data.get("home_schedule", {}))
        away_sched = ScheduleContext(**data.get("away_schedule", {}))
        result = model.compare_matchup(
            data["home_team"], data["away_team"],
            home_sched, away_sched,
            data.get("away_previous_city")
        )
        return jsonify(result)

    @app.route("/api/travel/distances", methods=["GET"])
    def all_distances():
        return jsonify(model.get_all_distances())
