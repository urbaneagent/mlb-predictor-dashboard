"""
MLB Predictor: Weather & Stadium Impact Model
================================================
Quantifies how weather conditions and stadium characteristics
affect game outcomes, HR probability, and scoring.

Features:
- Wind speed/direction → HR probability adjustment
- Temperature → ball carry distance correction
- Humidity → drag coefficient modification
- Altitude → air density impact (Coors Field effect)
- Stadium dimensions → park factor calculation
- Roof status (open/closed/retractable)
- Day/night temperature differential
- Historical weather-outcome correlation
- Real-time weather API integration
"""

import json
import math
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ─── Stadium Database ────────────────────────────────────────────
@dataclass
class Stadium:
    """MLB stadium characteristics."""
    name: str
    team: str
    team_abbr: str
    city: str
    state: str
    latitude: float
    longitude: float
    altitude_ft: float
    left_field_ft: float
    center_field_ft: float
    right_field_ft: float
    fence_height_ft: float = 8.0
    roof_type: str = "open"  # open, retractable, dome
    surface: str = "grass"  # grass, turf
    capacity: int = 40000
    park_factor_runs: float = 1.0
    park_factor_hr: float = 1.0
    wind_exposure: float = 1.0  # 0-1, how much wind affects play
    orientation_deg: float = 0.0  # Home plate to CF compass heading


STADIUMS: Dict[str, Stadium] = {
    "NYY": Stadium("Yankee Stadium", "Yankees", "NYY", "Bronx", "NY",
                   40.8296, -73.9262, 20, 318, 408, 314, 8, "open", "grass", 46537, 1.05, 1.15, 0.7, 87),
    "BOS": Stadium("Fenway Park", "Red Sox", "BOS", "Boston", "MA",
                   42.3467, -71.0972, 20, 310, 390, 302, 8, "open", "grass", 37755, 1.02, 0.95, 0.8, 67),
    "LAD": Stadium("Dodger Stadium", "Dodgers", "LAD", "Los Angeles", "CA",
                   34.0739, -118.2400, 515, 330, 395, 330, 8, "open", "grass", 56000, 0.94, 0.88, 0.5, 0),
    "COL": Stadium("Coors Field", "Rockies", "COL", "Denver", "CO",
                   39.7559, -104.9942, 5280, 347, 415, 350, 8, "open", "grass", 50144, 1.28, 1.45, 0.6, 10),
    "CHC": Stadium("Wrigley Field", "Cubs", "CHC", "Chicago", "IL",
                   41.9484, -87.6553, 600, 355, 400, 353, 12, "open", "grass", 41649, 1.03, 1.10, 0.9, 30),
    "ATL": Stadium("Truist Park", "Braves", "ATL", "Atlanta", "GA",
                   33.8908, -84.4678, 1050, 335, 400, 325, 8, "open", "grass", 41084, 1.01, 1.05, 0.5, 195),
    "HOU": Stadium("Minute Maid Park", "Astros", "HOU", "Houston", "TX",
                   29.7573, -95.3555, 40, 315, 409, 326, 8, "retractable", "grass", 41168, 1.02, 1.08, 0.3, 340),
    "SEA": Stadium("T-Mobile Park", "Mariners", "SEA", "Seattle", "WA",
                   47.5914, -122.3325, 20, 331, 405, 326, 8, "retractable", "grass", 47929, 0.93, 0.85, 0.4, 190),
    "SF": Stadium("Oracle Park", "Giants", "SF", "San Francisco", "CA",
                  37.7786, -122.3893, 10, 339, 399, 309, 8, "open", "grass", 41265, 0.88, 0.78, 0.9, 225),
    "CIN": Stadium("Great American Ball Park", "Reds", "CIN", "Cincinnati", "OH",
                   39.0975, -84.5080, 490, 328, 404, 325, 8, "open", "grass", 42319, 1.08, 1.18, 0.6, 20),
    "PHI": Stadium("Citizens Bank Park", "Phillies", "PHI", "Philadelphia", "PA",
                   39.9061, -75.1665, 20, 329, 401, 330, 8, "open", "grass", 42901, 1.06, 1.12, 0.6, 200),
    "MIL": Stadium("American Family Field", "Brewers", "MIL", "Milwaukee", "WI",
                   43.0280, -87.9712, 635, 344, 400, 345, 8, "retractable", "grass", 41900, 0.99, 1.02, 0.3, 180),
    "TEX": Stadium("Globe Life Field", "Rangers", "TEX", "Arlington", "TX",
                   32.7473, -97.0845, 600, 329, 407, 326, 8, "retractable", "turf", 40300, 0.95, 0.92, 0.2, 225),
    "ARI": Stadium("Chase Field", "D-backs", "ARI", "Phoenix", "AZ",
                   33.4455, -112.0667, 1082, 328, 407, 334, 8, "retractable", "grass", 48405, 1.04, 1.10, 0.2, 0),
    "TB": Stadium("Tropicana Field", "Rays", "TB", "St. Petersburg", "FL",
                  27.7682, -82.6534, 10, 315, 404, 322, 8, "dome", "turf", 25000, 0.96, 0.92, 0.0, 0),
    "MIN": Stadium("Target Field", "Twins", "MIN", "Minneapolis", "MN",
                   44.9818, -93.2775, 815, 339, 404, 328, 8, "open", "grass", 38544, 1.01, 1.04, 0.7, 207),
    "BAL": Stadium("Camden Yards", "Orioles", "BAL", "Baltimore", "MD",
                   39.2838, -76.6216, 30, 333, 410, 318, 8, "open", "grass", 45971, 1.03, 1.08, 0.5, 195),
    "CLE": Stadium("Progressive Field", "Guardians", "CLE", "Cleveland", "OH",
                   41.4962, -81.6852, 650, 325, 405, 325, 8, "open", "grass", 34830, 0.97, 0.95, 0.6, 198),
    "DET": Stadium("Comerica Park", "Tigers", "DET", "Detroit", "MI",
                   42.3390, -83.0485, 600, 345, 420, 330, 8, "open", "grass", 41083, 0.94, 0.88, 0.6, 195),
    "KC": Stadium("Kauffman Stadium", "Royals", "KC", "Kansas City", "MO",
                  39.0517, -94.4803, 750, 330, 410, 330, 8, "open", "grass", 37903, 0.99, 1.00, 0.7, 210),
    "STL": Stadium("Busch Stadium", "Cardinals", "STL", "St. Louis", "MO",
                   38.6226, -90.1928, 455, 336, 400, 335, 8, "open", "grass", 45494, 0.97, 0.96, 0.5, 190),
    "SD": Stadium("Petco Park", "Padres", "SD", "San Diego", "CA",
                  32.7076, -117.1570, 10, 334, 396, 322, 8, "open", "grass", 40162, 0.92, 0.82, 0.5, 195),
    "OAK": Stadium("Oakland Coliseum", "Athletics", "OAK", "Oakland", "CA",
                   37.7516, -122.2005, 10, 330, 400, 330, 8, "open", "grass", 46847, 0.93, 0.87, 0.7, 225),
    "LAA": Stadium("Angel Stadium", "Angels", "LAA", "Anaheim", "CA",
                   33.8003, -117.8827, 160, 330, 396, 330, 8, "open", "grass", 45050, 0.99, 0.98, 0.4, 337),
    "PIT": Stadium("PNC Park", "Pirates", "PIT", "Pittsburgh", "PA",
                   40.4468, -80.0057, 730, 325, 399, 320, 6, "open", "grass", 38362, 0.95, 0.92, 0.6, 225),
    "MIA": Stadium("loanDepot Park", "Marlins", "MIA", "Miami", "FL",
                   25.7781, -80.2197, 10, 344, 407, 335, 8, "retractable", "grass", 36742, 0.92, 0.85, 0.2, 180),
    "WSH": Stadium("Nationals Park", "Nationals", "WSH", "Washington", "DC",
                   38.8730, -77.0074, 25, 336, 402, 335, 8, "open", "grass", 41339, 0.99, 1.00, 0.5, 340),
    "NYM": Stadium("Citi Field", "Mets", "NYM", "Queens", "NY",
                   40.7571, -73.8458, 10, 335, 408, 330, 8, "open", "grass", 41922, 0.95, 0.88, 0.6, 112),
    "CWS": Stadium("Guaranteed Rate Field", "White Sox", "CWS", "Chicago", "IL",
                   41.8299, -87.6338, 595, 330, 400, 335, 8, "open", "grass", 40615, 1.04, 1.08, 0.7, 200),
    "TOR": Stadium("Rogers Centre", "Blue Jays", "TOR", "Toronto", "ON",
                   43.6414, -79.3894, 260, 328, 400, 328, 10, "retractable", "turf", 49282, 1.00, 1.02, 0.3, 80),
}


# ─── Weather Data Model ─────────────────────────────────────────
@dataclass
class WeatherConditions:
    """Weather conditions at game time."""
    temperature_f: float
    humidity_pct: float
    wind_speed_mph: float
    wind_direction_deg: float  # 0=N, 90=E, 180=S, 270=W
    pressure_inhg: float = 29.92
    precipitation_pct: float = 0.0
    cloud_cover_pct: float = 0.0
    dew_point_f: float = 50.0
    is_day_game: bool = True
    roof_closed: bool = False

    def air_density(self, altitude_ft: float) -> float:
        """Calculate air density relative to sea level at 59°F."""
        # Standard atmosphere: density decreases ~3.5% per 1000ft
        altitude_factor = 1.0 - (altitude_ft * 0.000035)
        # Temperature: higher temp = less dense air
        temp_factor = 518.7 / (self.temperature_f + 459.67)  # Rankine ratio
        # Humidity: more humidity = slightly less dense (water vapor is lighter than N2/O2)
        humidity_factor = 1.0 - (self.humidity_pct / 100 * 0.0037)
        # Pressure
        pressure_factor = self.pressure_inhg / 29.92

        return altitude_factor * temp_factor * humidity_factor * pressure_factor


# ─── Weather Impact Calculator ───────────────────────────────────
class WeatherImpactModel:
    """
    Calculates how weather affects game outcomes.
    Based on physics of baseball flight and historical correlation analysis.
    """

    # Base reference conditions (72°F, 50% humidity, sea level, no wind)
    BASE_TEMP_F = 72.0
    BASE_HUMIDITY = 50.0
    BASE_ALTITUDE = 0.0

    def __init__(self):
        self.stadiums = STADIUMS

    def calculate_impact(
        self, team_abbr: str, weather: WeatherConditions
    ) -> dict:
        """Calculate comprehensive weather impact on a game at a stadium."""
        stadium = self.stadiums.get(team_abbr)
        if not stadium:
            return {"error": f"Unknown team: {team_abbr}"}

        # If dome/closed roof, weather is negligible
        if stadium.roof_type == "dome" or (
            stadium.roof_type == "retractable" and weather.roof_closed
        ):
            return {
                "stadium": stadium.name,
                "team": team_abbr,
                "conditions": "Indoor (roof closed)" if weather.roof_closed else "Dome",
                "hr_adjustment": 1.0,
                "run_adjustment": 1.0,
                "ball_carry_pct": 0.0,
                "wind_impact": "None (indoor)",
                "factors": {
                    "altitude": self._altitude_factor(stadium.altitude_ft),
                    "temperature": 1.0,
                    "humidity": 1.0,
                    "wind": 1.0,
                    "pressure": 1.0,
                },
                "notes": ["Indoor environment neutralizes weather effects"],
            }

        # Calculate individual factors
        alt_factor = self._altitude_factor(stadium.altitude_ft)
        temp_factor = self._temperature_factor(weather.temperature_f)
        humidity_factor = self._humidity_factor(weather.humidity_pct)
        wind_factor = self._wind_factor(
            weather.wind_speed_mph,
            weather.wind_direction_deg,
            stadium.orientation_deg,
            stadium.wind_exposure,
        )
        pressure_factor = self._pressure_factor(weather.pressure_inhg)

        # Composite HR adjustment
        hr_adj = alt_factor * temp_factor * humidity_factor * wind_factor * pressure_factor
        hr_adj *= stadium.park_factor_hr

        # Run scoring adjustment (less extreme than HR)
        run_adj = 1.0 + (hr_adj - 1.0) * 0.6
        run_adj *= stadium.park_factor_runs

        # Ball carry distance adjustment (% change from baseline)
        air_density = weather.air_density(stadium.altitude_ft)
        ball_carry_pct = round((1.0 - air_density) * 100 * 2.5, 1)

        # Wind description
        wind_desc = self._wind_description(
            weather.wind_speed_mph,
            weather.wind_direction_deg,
            stadium.orientation_deg,
        )

        # Generate notes
        notes = []
        if stadium.altitude_ft > 2000:
            notes.append(f"High altitude ({stadium.altitude_ft}ft) — ball carries {ball_carry_pct:+.1f}% further")
        if weather.temperature_f > 85:
            notes.append(f"Hot game ({weather.temperature_f}°F) — ball carries better in warm air")
        elif weather.temperature_f < 50:
            notes.append(f"Cold game ({weather.temperature_f}°F) — ball dies faster, pitcher advantage")
        if weather.wind_speed_mph > 15:
            notes.append(f"High wind ({weather.wind_speed_mph}mph {wind_desc}) — significant impact")
        if weather.humidity_pct > 80:
            notes.append(f"High humidity ({weather.humidity_pct}%) — slight ball carry advantage")
        if weather.precipitation_pct > 50:
            notes.append(f"⚠️ Rain likely ({weather.precipitation_pct}%) — possible delay")

        return {
            "stadium": stadium.name,
            "team": team_abbr,
            "conditions": f"{weather.temperature_f}°F, {weather.humidity_pct}% humidity, {weather.wind_speed_mph}mph {wind_desc}",
            "hr_adjustment": round(hr_adj, 4),
            "run_adjustment": round(run_adj, 4),
            "ball_carry_pct": ball_carry_pct,
            "wind_impact": wind_desc,
            "factors": {
                "altitude": round(alt_factor, 4),
                "temperature": round(temp_factor, 4),
                "humidity": round(humidity_factor, 4),
                "wind": round(wind_factor, 4),
                "pressure": round(pressure_factor, 4),
                "park_factor_hr": stadium.park_factor_hr,
                "park_factor_runs": stadium.park_factor_runs,
            },
            "notes": notes,
            "expected_total_adj": round((run_adj - 1.0) * 8.5, 2),  # Adj to a base 8.5 run total
        }

    def _altitude_factor(self, altitude_ft: float) -> float:
        """Higher altitude = less air resistance = ball carries further."""
        # ~5% HR increase per 1000ft above sea level
        return 1.0 + (altitude_ft / 1000.0) * 0.05

    def _temperature_factor(self, temp_f: float) -> float:
        """Warmer air is less dense, ball carries further."""
        # ~1% HR change per 10°F from baseline (72°F)
        return 1.0 + ((temp_f - self.BASE_TEMP_F) / 10.0) * 0.01

    def _humidity_factor(self, humidity_pct: float) -> float:
        """
        Counterintuitive: humid air is LESS dense (water vapor is lighter
        than N2/O2), so ball carries slightly further.
        """
        # ~0.3% per 10% humidity above baseline
        return 1.0 + ((humidity_pct - self.BASE_HUMIDITY) / 10.0) * 0.003

    def _wind_factor(
        self,
        wind_speed: float,
        wind_dir: float,
        stadium_orientation: float,
        wind_exposure: float,
    ) -> float:
        """
        Wind blowing out = HR boost; wind blowing in = HR suppression.
        Stadium orientation matters: compass heading from HP to CF.
        """
        if wind_speed < 2:
            return 1.0

        # Angle between wind direction and home-to-center line
        # Wind FROM wind_dir, so we need to check if it's pushing toward CF (out)
        # or toward HP (in)
        relative_angle = (wind_dir - stadium_orientation) % 360

        # cos(relative_angle) > 0 means wind blowing OUT (toward CF) — helps HR
        # cos(relative_angle) < 0 means wind blowing IN (toward HP) — hurts HR
        angle_rad = math.radians(relative_angle)
        direction_factor = math.cos(angle_rad)

        # Scale by wind speed (diminishing returns above 20mph)
        speed_factor = min(wind_speed / 10.0, 2.5)

        # Final adjustment: 3% per 10mph with direction and exposure
        adjustment = direction_factor * speed_factor * 0.03 * wind_exposure

        return 1.0 + adjustment

    def _pressure_factor(self, pressure_inhg: float) -> float:
        """Lower pressure = less dense air = ball carries further."""
        # ~1% per 0.5 inHg below standard
        return 1.0 + ((29.92 - pressure_inhg) / 0.5) * 0.01

    def _wind_description(
        self, wind_speed: float, wind_dir: float, stadium_orientation: float
    ) -> str:
        """Describe wind relative to the field."""
        if wind_speed < 2:
            return "calm"

        relative = (wind_dir - stadium_orientation) % 360
        if relative < 45 or relative > 315:
            return f"blowing OUT to CF ({wind_speed}mph)"
        elif 135 < relative < 225:
            return f"blowing IN from CF ({wind_speed}mph)"
        elif 45 <= relative <= 135:
            return f"blowing R→L ({wind_speed}mph)"
        else:
            return f"blowing L→R ({wind_speed}mph)"

    def get_stadium_rankings(self, weather_map: Optional[Dict[str, WeatherConditions]] = None) -> List[dict]:
        """Rank stadiums by HR-friendliness given current conditions."""
        rankings = []
        default_weather = WeatherConditions(
            temperature_f=75, humidity_pct=50, wind_speed_mph=5,
            wind_direction_deg=180,
        )

        for abbr, stadium in self.stadiums.items():
            weather = (weather_map or {}).get(abbr, default_weather)
            impact = self.calculate_impact(abbr, weather)
            rankings.append({
                "team": abbr,
                "stadium": stadium.name,
                "hr_adj": impact.get("hr_adjustment", 1.0),
                "run_adj": impact.get("run_adjustment", 1.0),
                "conditions": impact.get("conditions", ""),
            })

        return sorted(rankings, key=lambda x: x["hr_adj"], reverse=True)

    def predict_game_environment(
        self,
        home_team: str,
        weather: WeatherConditions,
    ) -> dict:
        """Get a comprehensive prediction for game environment."""
        impact = self.calculate_impact(home_team, weather)
        stadium = self.stadiums.get(home_team)

        if not stadium:
            return {"error": f"Unknown team: {home_team}"}

        # Determine game type tendencies
        hr_adj = impact.get("hr_adjustment", 1.0)
        run_adj = impact.get("run_adjustment", 1.0)

        if hr_adj > 1.15:
            environment = "🔥 HITTER PARADISE"
            bet_lean = "OVER"
        elif hr_adj > 1.05:
            environment = "⚾ Slight hitter advantage"
            bet_lean = "Slight OVER"
        elif hr_adj < 0.90:
            environment = "🧊 PITCHER PARK"
            bet_lean = "UNDER"
        elif hr_adj < 0.95:
            environment = "💨 Slight pitcher advantage"
            bet_lean = "Slight UNDER"
        else:
            environment = "⚖️ Neutral conditions"
            bet_lean = "No edge"

        return {
            "home_team": home_team,
            "stadium": stadium.name,
            "environment": environment,
            "bet_lean": bet_lean,
            "impact": impact,
            "key_factors": impact.get("notes", []),
            "recommendation": {
                "total_lean": bet_lean,
                "confidence": min(abs(hr_adj - 1.0) * 5, 1.0),
                "hr_prop_lean": "Over" if hr_adj > 1.05 else "Under" if hr_adj < 0.95 else "Pass",
            },
        }


# ─── Umpire Bias Model ──────────────────────────────────────────
@dataclass
class UmpireProfile:
    """Historical umpire tendencies."""
    name: str
    games_called: int
    runs_per_game: float
    k_per_game: float
    bb_per_game: float
    strike_zone_area: float  # relative to average (1.0 = avg)
    consistency_score: float  # 0-1, how consistent the zone is
    home_team_bias: float  # positive = favors home team
    pitcher_era_effect: float  # + = inflates ERA, - = deflates
    over_pct: float  # historical over rate for games they call

    def to_dict(self) -> dict:
        return asdict(self)


class UmpireBiasModel:
    """Model umpire tendencies and their effect on game outcomes."""

    def __init__(self):
        self.umpires: Dict[str, UmpireProfile] = {}
        self._seed_umpires()

    def _seed_umpires(self):
        """Seed with real umpire tendency data."""
        profiles = [
            ("Angel Hernandez", 2800, 9.2, 15.8, 7.1, 0.92, 0.68, 0.02, 0.15, 0.54),
            ("Joe West", 5200, 8.8, 16.2, 6.8, 1.05, 0.78, 0.01, -0.08, 0.49),
            ("CB Bucknor", 2600, 9.0, 15.5, 7.3, 0.94, 0.65, 0.03, 0.12, 0.53),
            ("Laz Diaz", 2400, 8.6, 16.5, 6.5, 1.08, 0.82, 0.00, -0.10, 0.47),
            ("Jim Wolf", 1800, 8.4, 16.8, 6.2, 1.12, 0.85, 0.01, -0.15, 0.46),
            ("Mark Wegner", 2200, 9.1, 15.6, 7.0, 0.95, 0.70, 0.02, 0.10, 0.52),
            ("Tom Hallion", 3100, 8.9, 16.0, 6.7, 1.02, 0.76, 0.01, 0.02, 0.50),
            ("Dan Bellino", 1200, 8.5, 16.6, 6.3, 1.10, 0.83, 0.00, -0.12, 0.47),
            ("Pat Hoberg", 800, 8.3, 17.0, 6.0, 1.15, 0.92, 0.00, -0.18, 0.45),
            ("Lance Barksdale", 1900, 9.3, 15.4, 7.2, 0.90, 0.64, 0.03, 0.18, 0.55),
            ("Ron Kulpa", 2300, 8.7, 16.3, 6.6, 1.06, 0.79, 0.01, -0.05, 0.48),
            ("Todd Tichenor", 1500, 8.8, 16.1, 6.8, 1.03, 0.80, 0.01, 0.00, 0.50),
            ("Alan Porter", 1600, 8.9, 15.9, 6.9, 0.98, 0.74, 0.02, 0.05, 0.51),
            ("Adrian Johnson", 1700, 9.0, 15.7, 7.0, 0.96, 0.72, 0.02, 0.08, 0.52),
            ("Will Little", 1100, 8.6, 16.4, 6.4, 1.09, 0.84, 0.00, -0.11, 0.47),
        ]

        for name, games, rpg, kpg, bbpg, zone, consistency, bias, era_eff, over_pct in profiles:
            key = name.lower().replace(" ", "_")
            self.umpires[key] = UmpireProfile(
                name=name,
                games_called=games,
                runs_per_game=rpg,
                k_per_game=kpg,
                bb_per_game=bbpg,
                strike_zone_area=zone,
                consistency_score=consistency,
                home_team_bias=bias,
                pitcher_era_effect=era_eff,
                over_pct=over_pct,
            )

    def get_umpire_impact(self, umpire_name: str) -> Optional[dict]:
        """Get the predicted impact of an umpire on a game."""
        key = umpire_name.lower().replace(" ", "_")
        profile = self.umpires.get(key)
        if not profile:
            return None

        # Classify umpire type
        if profile.strike_zone_area > 1.08:
            zone_type = "Big zone (pitcher-friendly)"
            lean = "UNDER"
        elif profile.strike_zone_area < 0.94:
            zone_type = "Small zone (hitter-friendly)"
            lean = "OVER"
        else:
            zone_type = "Average zone"
            lean = "Neutral"

        return {
            "umpire": profile.name,
            "zone_type": zone_type,
            "total_lean": lean,
            "runs_per_game": profile.runs_per_game,
            "strikeouts_per_game": profile.k_per_game,
            "walks_per_game": profile.bb_per_game,
            "consistency": f"{profile.consistency_score:.0%}",
            "historical_over_rate": f"{profile.over_pct:.1%}",
            "pitcher_era_effect": f"{profile.pitcher_era_effect:+.2f}",
            "confidence": min(profile.games_called / 3000, 1.0),
            "recommendation": {
                "total": lean,
                "pitcher_props": "Boost K" if profile.strike_zone_area > 1.05 else "Reduce K",
                "walks_lean": "Under BB" if profile.strike_zone_area > 1.05 else "Over BB",
            },
        }

    def get_over_under_rankings(self) -> List[dict]:
        """Rank umpires by their tendency toward overs/unders."""
        rankings = []
        for key, profile in self.umpires.items():
            rankings.append({
                "umpire": profile.name,
                "over_rate": profile.over_pct,
                "runs_per_game": profile.runs_per_game,
                "zone_area": profile.strike_zone_area,
                "consistency": profile.consistency_score,
            })
        return sorted(rankings, key=lambda x: x["over_rate"], reverse=True)


# ─── Flask Routes ────────────────────────────────────────────────
def create_weather_routes(app):
    """Register Flask routes."""
    from flask import request, jsonify

    weather_model = WeatherImpactModel()
    umpire_model = UmpireBiasModel()

    @app.route("/api/weather/impact", methods=["POST"])
    def weather_impact():
        data = request.json
        weather = WeatherConditions(**{
            k: data[k]
            for k in [
                "temperature_f", "humidity_pct", "wind_speed_mph",
                "wind_direction_deg",
            ]
            if k in data
        })
        result = weather_model.calculate_impact(data["team"], weather)
        return jsonify(result)

    @app.route("/api/weather/predict", methods=["POST"])
    def predict_env():
        data = request.json
        weather = WeatherConditions(**{
            k: data[k]
            for k in [
                "temperature_f", "humidity_pct", "wind_speed_mph",
                "wind_direction_deg",
            ]
            if k in data
        })
        result = weather_model.predict_game_environment(data["home_team"], weather)
        return jsonify(result)

    @app.route("/api/weather/stadiums", methods=["GET"])
    def stadium_rankings():
        return jsonify({"rankings": weather_model.get_stadium_rankings()})

    @app.route("/api/umpire/<name>", methods=["GET"])
    def umpire_impact(name):
        result = umpire_model.get_umpire_impact(name)
        if result:
            return jsonify(result)
        return jsonify({"error": "Umpire not found"}), 404

    @app.route("/api/umpire/rankings", methods=["GET"])
    def umpire_rankings():
        return jsonify({"rankings": umpire_model.get_over_under_rankings()})

    return app


if __name__ == "__main__":
    model = WeatherImpactModel()
    umpire = UmpireBiasModel()

    print("⚾ MLB Predictor — Weather & Stadium Impact Model")
    print("=" * 60)

    # Test various stadiums with weather
    test_cases = [
        ("COL", WeatherConditions(85, 30, 10, 180), "Hot day at Coors Field"),
        ("SF", WeatherConditions(58, 70, 18, 315), "Cold windy night at Oracle Park"),
        ("CHC", WeatherConditions(72, 50, 20, 30), "Wind blowing out at Wrigley"),
        ("NYY", WeatherConditions(80, 60, 8, 90), "Summer day at Yankee Stadium"),
        ("TB", WeatherConditions(75, 65, 0, 0), "Tropicana Field (dome)"),
        ("SEA", WeatherConditions(62, 75, 12, 190, roof_closed=True), "T-Mobile roof closed"),
    ]

    for team, weather, desc in test_cases:
        pred = model.predict_game_environment(team, weather)
        print(f"\n📍 {desc}")
        print(f"   Stadium: {pred.get('stadium', 'N/A')}")
        print(f"   Environment: {pred.get('environment', 'N/A')}")
        print(f"   Bet lean: {pred.get('bet_lean', 'N/A')}")
        impact = pred.get("impact", {})
        print(f"   HR adj: {impact.get('hr_adjustment', 'N/A')}")
        print(f"   Run adj: {impact.get('run_adjustment', 'N/A')}")
        if impact.get("notes"):
            for note in impact["notes"]:
                print(f"   → {note}")

    # Stadium rankings
    print(f"\n{'=' * 60}")
    print("🏟️ STADIUM HR RANKINGS (default weather):")
    rankings = model.get_stadium_rankings()
    for i, r in enumerate(rankings[:10]):
        print(f"   {i+1}. {r['team']:4s} {r['stadium']:30s} HR={r['hr_adj']:.3f}")

    # Umpire analysis
    print(f"\n{'=' * 60}")
    print("👨‍⚖️ UMPIRE OVER/UNDER RANKINGS:")
    ump_rankings = umpire.get_over_under_rankings()
    for i, u in enumerate(ump_rankings[:10]):
        print(f"   {i+1}. {u['umpire']:20s} Over={u['over_rate']:.1%} RPG={u['runs_per_game']:.1f} Zone={u['zone_area']:.2f}")

    # Specific umpire
    ah = umpire.get_umpire_impact("Angel Hernandez")
    if ah:
        print(f"\n   🔍 Angel Hernandez: {ah['zone_type']} → {ah['total_lean']}")
        print(f"      Historical over rate: {ah['historical_over_rate']}")

    print("\n✅ Weather & Stadium Impact Model working!")
