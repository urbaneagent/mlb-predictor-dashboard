"""
MLB Predictor - Weather Impact Model
Models the effect of weather on MLB game outcomes and totals.
Wind, temperature, humidity, and altitude are key factors.
"""
import math
from dataclasses import dataclass, field, asdict


@dataclass
class StadiumProfile:
    """Stadium characteristics that interact with weather."""
    name: str
    team: str
    city: str
    roof: str  # open, retractable, dome
    altitude_ft: int
    outfield_avg_ft: int  # Average outfield fence distance
    capacity: int
    park_factor_runs: float  # 1.0 = neutral, >1 = hitter-friendly
    park_factor_hr: float
    wind_exposure: str  # high, medium, low, none
    latitude: float = 0


@dataclass
class WeatherConditions:
    """Current or forecast weather for a game."""
    temperature_f: float
    humidity_pct: float
    wind_speed_mph: float
    wind_direction: str  # N, NE, E, SE, S, SW, W, NW
    wind_toward_field: str  # in, out, cross, none
    precipitation_pct: float
    barometric_pressure_inhg: float = 29.92
    dew_point_f: float = 0


@dataclass
class WeatherImpact:
    """Quantified weather impact on game metrics."""
    total_adjustment: float  # +/- runs on game total
    hr_factor: float  # Multiplier on HR probability
    fly_ball_carry: str  # enhanced, neutral, reduced
    pitching_advantage: bool
    scoring_environment: str  # elevated, neutral, depressed
    temperature_impact: float
    wind_impact: float
    humidity_impact: float
    altitude_impact: float
    details: list = field(default_factory=list)
    betting_angle: str = ""


# All 30 MLB stadiums
MLB_STADIUMS = {
    "COL": StadiumProfile("Coors Field", "COL", "Denver", "open", 5280, 347, 50144, 1.28, 1.39, "medium", 39.76),
    "CIN": StadiumProfile("Great American Ball Park", "CIN", "Cincinnati", "open", 490, 325, 42319, 1.08, 1.18, "high", 39.10),
    "TEX": StadiumProfile("Globe Life Field", "TEX", "Arlington", "retractable", 596, 333, 40300, 0.96, 0.95, "none", 32.75),
    "HOU": StadiumProfile("Minute Maid Park", "HOU", "Houston", "retractable", 50, 326, 41168, 1.01, 1.05, "none", 29.76),
    "NYY": StadiumProfile("Yankee Stadium", "NYY", "New York", "open", 55, 318, 46537, 1.08, 1.18, "medium", 40.83),
    "BOS": StadiumProfile("Fenway Park", "BOS", "Boston", "open", 20, 310, 37755, 1.05, 0.96, "high", 42.35),
    "CHC": StadiumProfile("Wrigley Field", "CHC", "Chicago", "open", 597, 353, 41649, 1.04, 1.10, "high", 41.95),
    "MIL": StadiumProfile("American Family Field", "MIL", "Milwaukee", "retractable", 600, 344, 41900, 0.98, 0.96, "none", 43.04),
    "LAD": StadiumProfile("Dodger Stadium", "LAD", "Los Angeles", "open", 515, 330, 56000, 0.96, 0.92, "low", 34.07),
    "SF": StadiumProfile("Oracle Park", "SF", "San Francisco", "open", 0, 339, 41265, 0.88, 0.78, "high", 37.78),
    "SEA": StadiumProfile("T-Mobile Park", "SEA", "Seattle", "retractable", 20, 331, 47929, 0.93, 0.88, "none", 47.61),
    "STL": StadiumProfile("Busch Stadium", "STL", "St. Louis", "open", 455, 335, 45494, 0.97, 0.96, "medium", 38.63),
    "ATL": StadiumProfile("Truist Park", "ATL", "Atlanta", "open", 1050, 335, 41084, 0.99, 1.02, "medium", 33.89),
    "PHI": StadiumProfile("Citizens Bank Park", "PHI", "Philadelphia", "open", 20, 329, 42792, 1.04, 1.12, "medium", 39.91),
    "WSH": StadiumProfile("Nationals Park", "WSH", "Washington", "open", 25, 336, 41339, 0.98, 0.99, "medium", 38.91),
    "NYM": StadiumProfile("Citi Field", "NYM", "New York", "open", 15, 335, 41922, 0.92, 0.88, "medium", 40.76),
    "MIA": StadiumProfile("LoanDepot Park", "MIA", "Miami", "retractable", 10, 335, 36742, 0.92, 0.86, "none", 25.78),
    "TB": StadiumProfile("Tropicana Field", "TB", "St. Petersburg", "dome", 45, 322, 25000, 0.91, 0.82, "none", 27.77),
    "MIN": StadiumProfile("Target Field", "MIN", "Minneapolis", "open", 841, 339, 38544, 1.02, 1.05, "high", 44.98),
    "DET": StadiumProfile("Comerica Park", "DET", "Detroit", "open", 585, 345, 41083, 0.94, 0.90, "medium", 42.33),
    "CLE": StadiumProfile("Progressive Field", "CLE", "Cleveland", "open", 653, 325, 34830, 0.95, 0.94, "high", 41.50),
    "CWS": StadiumProfile("Guaranteed Rate Field", "CWS", "Chicago", "open", 595, 330, 40615, 1.03, 1.08, "high", 41.83),
    "KC": StadiumProfile("Kauffman Stadium", "KC", "Kansas City", "open", 750, 340, 37903, 0.98, 0.97, "medium", 39.10),
    "OAK": StadiumProfile("Oakland Coliseum", "OAK", "Oakland", "open", 25, 330, 46847, 0.88, 0.80, "high", 37.75),
    "ARI": StadiumProfile("Chase Field", "ARI", "Phoenix", "retractable", 1082, 334, 48519, 1.05, 1.10, "none", 33.45),
    "SD": StadiumProfile("Petco Park", "SD", "San Diego", "open", 19, 336, 40209, 0.91, 0.85, "low", 32.71),
    "LAA": StadiumProfile("Angel Stadium", "LAA", "Anaheim", "open", 160, 330, 45517, 0.97, 0.96, "low", 33.80),
    "PIT": StadiumProfile("PNC Park", "PIT", "Pittsburgh", "open", 730, 325, 38362, 0.94, 0.90, "medium", 40.44),
    "BAL": StadiumProfile("Camden Yards", "BAL", "Baltimore", "open", 130, 333, 45971, 1.02, 1.06, "medium", 39.28),
    "TOR": StadiumProfile("Rogers Centre", "TOR", "Toronto", "retractable", 269, 328, 49286, 0.97, 0.94, "none", 43.65),
}


class WeatherImpactModel:
    """
    Quantifies how weather conditions affect MLB game outcomes.

    Key findings from research:
    - 10°F increase → +0.35 runs on game total
    - Wind out 10+ mph → +0.8 runs, 25% more HR
    - Wind in 10+ mph → -0.5 runs, 15% fewer HR
    - Humidity >80% → slight increase in fly ball carry
    - Coors Field altitude → +1.5 runs baseline, 30% more HR
    """

    def __init__(self):
        self.stadiums = MLB_STADIUMS

    def analyze_impact(self, home_team: str, weather: WeatherConditions) -> WeatherImpact:
        """Calculate weather impact on a game."""
        stadium = self.stadiums.get(home_team)
        if not stadium:
            return WeatherImpact(
                total_adjustment=0, hr_factor=1.0, fly_ball_carry="neutral",
                pitching_advantage=False, scoring_environment="neutral",
                temperature_impact=0, wind_impact=0, humidity_impact=0, altitude_impact=0,
                details=["Stadium not found"]
            )

        # Skip weather for dome/closed roof
        if stadium.roof == "dome" or (stadium.roof == "retractable" and weather.temperature_f > 95):
            return WeatherImpact(
                total_adjustment=stadium.park_factor_runs - 1.0,
                hr_factor=stadium.park_factor_hr,
                fly_ball_carry="neutral",
                pitching_advantage=False,
                scoring_environment="neutral (controlled environment)",
                temperature_impact=0, wind_impact=0, humidity_impact=0,
                altitude_impact=self._altitude_impact(stadium.altitude_ft),
                details=["Controlled environment — weather has no effect"]
            )

        temp_impact = self._temperature_impact(weather.temperature_f)
        wind_impact = self._wind_impact(weather.wind_speed_mph, weather.wind_toward_field)
        humidity_impact = self._humidity_impact(weather.humidity_pct)
        altitude_impact = self._altitude_impact(stadium.altitude_ft)
        pressure_impact = self._pressure_impact(weather.barometric_pressure_inhg)

        total = temp_impact + wind_impact + humidity_impact + altitude_impact + pressure_impact

        # HR factor
        hr_factor = stadium.park_factor_hr
        if weather.wind_toward_field == "out":
            hr_factor *= (1 + weather.wind_speed_mph * 0.02)
        elif weather.wind_toward_field == "in":
            hr_factor *= (1 - weather.wind_speed_mph * 0.015)
        if weather.temperature_f > 85:
            hr_factor *= 1.05
        elif weather.temperature_f < 50:
            hr_factor *= 0.90

        # Fly ball carry
        if total > 0.5:
            carry = "enhanced"
        elif total < -0.3:
            carry = "reduced"
        else:
            carry = "neutral"

        # Scoring environment
        if total > 1.0:
            env = "elevated"
        elif total < -0.5:
            env = "depressed"
        else:
            env = "neutral"

        # Details
        details = []
        if abs(temp_impact) > 0.1:
            details.append(f"🌡️ Temperature ({weather.temperature_f:.0f}°F): {'+' if temp_impact > 0 else ''}{temp_impact:.2f} runs")
        if abs(wind_impact) > 0.1:
            details.append(f"🌬️ Wind ({weather.wind_speed_mph:.0f} mph {weather.wind_toward_field}): {'+' if wind_impact > 0 else ''}{wind_impact:.2f} runs")
        if abs(humidity_impact) > 0.05:
            details.append(f"💧 Humidity ({weather.humidity_pct:.0f}%): {'+' if humidity_impact > 0 else ''}{humidity_impact:.2f} runs")
        if abs(altitude_impact) > 0.1:
            details.append(f"⛰️ Altitude ({stadium.altitude_ft}ft): +{altitude_impact:.2f} runs")

        # Betting angle
        angle = ""
        if total > 0.8 and weather.wind_toward_field == "out" and weather.wind_speed_mph > 10:
            angle = "🎯 STRONG OVER: Wind blowing out + warm temps. Look at game total OVER."
        elif total < -0.5 and weather.wind_toward_field == "in":
            angle = "🎯 STRONG UNDER: Wind blowing in. Look at game total UNDER."
        elif weather.temperature_f < 45:
            angle = "🎯 LEAN UNDER: Cold game. Balls don't carry. Pitchers have advantage."
        elif total > 0.5:
            angle = "💡 Slight OVER lean. Weather favors hitters."

        return WeatherImpact(
            total_adjustment=round(total, 2),
            hr_factor=round(hr_factor, 2),
            fly_ball_carry=carry,
            pitching_advantage=total < -0.3,
            scoring_environment=env,
            temperature_impact=round(temp_impact, 3),
            wind_impact=round(wind_impact, 3),
            humidity_impact=round(humidity_impact, 3),
            altitude_impact=round(altitude_impact, 3),
            details=details,
            betting_angle=angle
        )

    def get_stadium_report(self, team: str) -> dict:
        """Get stadium profile with weather impact ranges."""
        stadium = self.stadiums.get(team)
        if not stadium:
            return {"error": "Stadium not found"}

        return {
            "stadium": asdict(stadium),
            "weather_sensitivity": "high" if stadium.wind_exposure == "high" and stadium.roof == "open" else
                                  "none" if stadium.roof == "dome" else "medium",
            "typical_adjustments": {
                "hot_day_95F": round(self._temperature_impact(95), 2),
                "cold_day_40F": round(self._temperature_impact(40), 2),
                "wind_out_15mph": round(self._wind_impact(15, "out"), 2),
                "wind_in_15mph": round(self._wind_impact(15, "in"), 2),
                "altitude": round(self._altitude_impact(stadium.altitude_ft), 2)
            }
        }

    # ============= Impact Calculations =============

    def _temperature_impact(self, temp_f: float) -> float:
        """
        Temperature impact on scoring.
        Baseline: 72°F. Every 10°F above → +0.35 runs.
        Research: warmer air is less dense → balls carry farther.
        """
        baseline = 72
        return (temp_f - baseline) / 10 * 0.35

    def _wind_impact(self, wind_mph: float, direction: str) -> float:
        """
        Wind impact on scoring.
        Out: +0.08 runs per mph
        In: -0.05 runs per mph
        Cross: minimal
        """
        if direction == "out":
            return wind_mph * 0.08
        elif direction == "in":
            return wind_mph * -0.05
        elif direction == "cross":
            return wind_mph * 0.02  # Slight increase due to unpredictability
        return 0

    def _humidity_impact(self, humidity_pct: float) -> float:
        """
        Humidity impact. Counterintuitive: humid air is LESS dense
        (water vapor is lighter than N2/O2), so balls carry slightly farther.
        """
        baseline = 50
        return (humidity_pct - baseline) / 100 * 0.15

    def _altitude_impact(self, altitude_ft: int) -> float:
        """
        Altitude impact. Less dense air = less drag = farther fly balls.
        Coors Field (5280ft) is the extreme case.
        """
        # Air density decreases ~3.5% per 1000ft
        density_reduction = altitude_ft / 1000 * 0.035
        return density_reduction * 4.5  # 4.5 runs adjustment at max density reduction

    def _pressure_impact(self, pressure_inhg: float) -> float:
        """Barometric pressure: lower pressure = less dense air = more carry."""
        baseline = 29.92
        return (baseline - pressure_inhg) * 0.5


def create_weather_routes(app, model: WeatherImpactModel):
    """Create FastAPI routes."""

    @app.post("/api/v1/weather/impact")
    async def get_impact(home_team: str, weather: dict):
        w = WeatherConditions(**weather)
        impact = model.analyze_impact(home_team, w)
        return asdict(impact)

    @app.get("/api/v1/weather/stadium/{team}")
    async def stadium_report(team: str):
        return model.get_stadium_report(team)

    @app.get("/api/v1/weather/stadiums")
    async def all_stadiums():
        return {k: asdict(v) for k, v in model.stadiums.items()}

    return model
