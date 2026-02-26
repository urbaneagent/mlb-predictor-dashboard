#!/usr/bin/env python3
"""
MLB Predictor - Environmental Factors Engine
===============================================
Weather, umpire, stadium, and situational factors
that impact game outcomes.

Features:
- Weather impact (wind speed/direction → HR probability)
- Umpire strike zone tendencies (wide/tight zone scoring)
- Stadium factors (altitude, dimensions, roof status)
- Pitcher fatigue (days rest, pitch count, innings load)
- Travel fatigue (timezone changes, road trip length)
- Day/night performance splits
- Platoon advantages (L/R matchups)

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum


# ============================================================================
# WEATHER IMPACT ENGINE
# ============================================================================

class WindDirection(Enum):
    OUT_TO_LF = "out_to_lf"
    OUT_TO_CF = "out_to_cf"
    OUT_TO_RF = "out_to_rf"
    IN_FROM_LF = "in_from_lf"
    IN_FROM_CF = "in_from_cf"
    IN_FROM_RF = "in_from_rf"
    LEFT_TO_RIGHT = "l_to_r"
    RIGHT_TO_LEFT = "r_to_l"
    CALM = "calm"


@dataclass
class GameWeather:
    """Weather conditions for a game"""
    temperature_f: float
    wind_speed_mph: float
    wind_direction: WindDirection
    humidity_pct: float
    precipitation_chance: float
    is_dome: bool = False
    roof_status: str = "open"  # "open", "closed", "retractable_open", "retractable_closed"

    @property
    def is_indoor(self) -> bool:
        return self.is_dome or self.roof_status in ("closed", "retractable_closed")


class WeatherEngine:
    """
    Calculates weather impact on game scoring and HR probability.
    
    Key factors:
    - Temperature: Every 10°F increase adds ~2% to HR probability
    - Wind out (10+ mph): Significant HR boost
    - Wind in (10+ mph): Significant HR reduction
    - Humidity: Higher humidity slightly reduces HR distance
    - Altitude: 5,280ft (Coors) adds ~25% to HR probability
    """

    # Temperature baseline: 72°F = neutral
    TEMP_BASELINE = 72

    # Wind impact multipliers
    WIND_HR_FACTORS = {
        WindDirection.OUT_TO_CF: 0.035,    # +3.5% per mph
        WindDirection.OUT_TO_LF: 0.025,    # +2.5% per mph
        WindDirection.OUT_TO_RF: 0.025,
        WindDirection.IN_FROM_CF: -0.03,   # -3.0% per mph
        WindDirection.IN_FROM_LF: -0.02,
        WindDirection.IN_FROM_RF: -0.02,
        WindDirection.LEFT_TO_RIGHT: 0.005, # Minimal effect
        WindDirection.RIGHT_TO_LEFT: 0.005,
        WindDirection.CALM: 0.0,
    }

    def calculate_weather_impact(self, weather: GameWeather,
                                  stadium_altitude: int = 0
                                  ) -> Dict[str, Any]:
        """
        Calculate comprehensive weather impact on scoring.
        
        Returns multipliers for runs and HRs.
        """
        if weather.is_indoor:
            return {
                'run_multiplier': 1.0,
                'hr_multiplier': 1.0,
                'scoring_environment': 'neutral',
                'factors': {'indoor': True},
                'summary': 'Indoor/dome game - weather neutral'
            }

        factors = {}

        # Temperature factor
        temp_diff = weather.temperature_f - self.TEMP_BASELINE
        temp_hr_factor = 1.0 + (temp_diff * 0.002)  # +0.2% per degree
        temp_run_factor = 1.0 + (temp_diff * 0.001)
        factors['temperature'] = {
            'value': weather.temperature_f,
            'hr_impact': round((temp_hr_factor - 1) * 100, 1),
            'note': f"{'Above' if temp_diff > 0 else 'Below'} baseline "
                    f"by {abs(temp_diff):.0f}°F"
        }

        # Wind factor
        wind_rate = self.WIND_HR_FACTORS.get(weather.wind_direction, 0)
        wind_hr_factor = 1.0 + (weather.wind_speed_mph * wind_rate)
        wind_hr_factor = max(0.75, min(1.40, wind_hr_factor))  # Cap at ±40%

        wind_run_factor = 1.0 + (weather.wind_speed_mph * wind_rate * 0.3)
        factors['wind'] = {
            'speed': weather.wind_speed_mph,
            'direction': weather.wind_direction.value,
            'hr_impact': round((wind_hr_factor - 1) * 100, 1),
            'note': f"{weather.wind_speed_mph:.0f} mph "
                    f"{weather.wind_direction.value.replace('_', ' ')}"
        }

        # Humidity factor (slight negative on HR distance)
        humidity_factor = 1.0 - (weather.humidity_pct - 50) * 0.0005
        humidity_factor = max(0.97, min(1.03, humidity_factor))
        factors['humidity'] = {
            'value': weather.humidity_pct,
            'hr_impact': round((humidity_factor - 1) * 100, 1)
        }

        # Altitude factor
        altitude_factor = 1.0 + (stadium_altitude / 5280) * 0.25
        factors['altitude'] = {
            'feet': stadium_altitude,
            'hr_impact': round((altitude_factor - 1) * 100, 1)
        }

        # Combined
        hr_multiplier = round(
            temp_hr_factor * wind_hr_factor * humidity_factor * altitude_factor,
            3
        )
        run_multiplier = round(
            temp_run_factor * wind_run_factor * altitude_factor * 0.5 +
            hr_multiplier * 0.5,
            3
        )

        # Classify environment
        if hr_multiplier >= 1.15:
            environment = "hitter_friendly"
        elif hr_multiplier <= 0.90:
            environment = "pitcher_friendly"
        else:
            environment = "neutral"

        return {
            'run_multiplier': run_multiplier,
            'hr_multiplier': hr_multiplier,
            'scoring_environment': environment,
            'factors': factors,
            'summary': self._summarize(hr_multiplier, factors)
        }

    def _summarize(self, hr_mult: float, factors: Dict) -> str:
        """Generate human-readable summary"""
        parts = []
        temp = factors.get('temperature', {}).get('value', 72)
        wind = factors.get('wind', {})

        if temp >= 85:
            parts.append(f"Hot ({temp:.0f}°F) favors hitters")
        elif temp <= 55:
            parts.append(f"Cold ({temp:.0f}°F) favors pitchers")

        if wind.get('speed', 0) >= 10:
            direction = wind.get('direction', '')
            if 'out' in direction:
                parts.append(f"Wind out at {wind['speed']:.0f} mph = HR boost")
            elif 'in' in direction:
                parts.append(f"Wind in at {wind['speed']:.0f} mph = HR suppression")

        alt = factors.get('altitude', {}).get('feet', 0)
        if alt >= 3000:
            parts.append(f"High altitude ({alt}ft) boosts offense")

        if not parts:
            parts.append("Neutral conditions")

        return ". ".join(parts)


# ============================================================================
# UMPIRE IMPACT ENGINE
# ============================================================================

@dataclass
class UmpireProfile:
    """Umpire strike zone and tendencies"""
    name: str
    umpire_id: str
    games_called: int = 0
    # Strike zone metrics (relative to league average)
    zone_width: float = 0.0      # + = wider (pitcher-friendly), - = tighter
    zone_height: float = 0.0     # + = taller zone, - = shorter
    k_rate_impact: float = 0.0   # + = more K's than average
    bb_rate_impact: float = 0.0  # + = more BB's than average
    run_impact: float = 0.0      # + = more runs than average
    # Consistency
    consistency_score: float = 0.5  # 0-1, 1 = perfectly consistent
    # Tendencies
    favors: str = "neutral"      # "pitcher", "hitter", "neutral"


class UmpireEngine:
    """
    Umpire strike zone analysis and game impact prediction.
    
    Data sources: Baseball Savant umpire scorecards, UmpScorecards.com
    """

    # Top MLB umpires with tendencies (based on historical data)
    UMPIRE_DATABASE = {
        'angel_hernandez': UmpireProfile(
            "Angel Hernandez", "ah001", 2500,
            zone_width=0.15, zone_height=-0.05,
            k_rate_impact=-0.02, bb_rate_impact=0.03,
            run_impact=0.15, consistency_score=0.35,
            favors="hitter"
        ),
        'pat_hoberg': UmpireProfile(
            "Pat Hoberg", "ph001", 800,
            zone_width=-0.02, zone_height=0.01,
            k_rate_impact=0.01, bb_rate_impact=-0.01,
            run_impact=-0.03, consistency_score=0.92,
            favors="neutral"
        ),
        'cb_bucknor': UmpireProfile(
            "CB Bucknor", "cb001", 2200,
            zone_width=0.20, zone_height=0.10,
            k_rate_impact=0.04, bb_rate_impact=-0.01,
            run_impact=-0.08, consistency_score=0.42,
            favors="pitcher"
        ),
        'joe_west': UmpireProfile(
            "Joe West", "jw001", 5400,
            zone_width=0.25, zone_height=0.05,
            k_rate_impact=0.06, bb_rate_impact=-0.02,
            run_impact=-0.12, consistency_score=0.48,
            favors="pitcher"
        ),
        'mark_carlson': UmpireProfile(
            "Mark Carlson", "mc001", 1500,
            zone_width=-0.10, zone_height=-0.05,
            k_rate_impact=-0.03, bb_rate_impact=0.04,
            run_impact=0.20, consistency_score=0.55,
            favors="hitter"
        ),
        'dan_bellino': UmpireProfile(
            "Dan Bellino", "db001", 1100,
            zone_width=0.05, zone_height=0.02,
            k_rate_impact=0.02, bb_rate_impact=0.0,
            run_impact=-0.02, consistency_score=0.78,
            favors="neutral"
        ),
        'jim_wolf': UmpireProfile(
            "Jim Wolf", "jw002", 1800,
            zone_width=-0.05, zone_height=0.0,
            k_rate_impact=-0.01, bb_rate_impact=0.02,
            run_impact=0.08, consistency_score=0.72,
            favors="neutral"
        ),
        'lance_barksdale': UmpireProfile(
            "Lance Barksdale", "lb001", 1600,
            zone_width=0.18, zone_height=0.08,
            k_rate_impact=0.05, bb_rate_impact=-0.02,
            run_impact=-0.10, consistency_score=0.50,
            favors="pitcher"
        ),
    }

    def get_umpire_impact(self, umpire_name: str) -> Dict[str, Any]:
        """Get the predicted impact of an umpire on a game"""
        # Find umpire
        ump = None
        name_lower = umpire_name.lower().replace(' ', '_')
        for key, profile in self.UMPIRE_DATABASE.items():
            if name_lower in key or name_lower in profile.name.lower():
                ump = profile
                break

        if not ump:
            return {
                'umpire': umpire_name,
                'found': False,
                'impact': 'unknown',
                'note': 'Umpire not in database - using league average'
            }

        # Calculate impact
        run_adj = ump.run_impact
        if run_adj > 0.10:
            scoring_impact = "high_scoring"
            scoring_note = f"Expect +{run_adj:.1f} runs above average"
        elif run_adj < -0.08:
            scoring_impact = "low_scoring"
            scoring_note = f"Expect {run_adj:.1f} runs below average"
        else:
            scoring_impact = "neutral"
            scoring_note = "Near league-average scoring expected"

        # Strategy recommendations
        recommendations = []
        if ump.favors == "pitcher":
            recommendations.append("Under is historically profitable with this ump")
            recommendations.append("Pitchers with good command benefit most")
        elif ump.favors == "hitter":
            recommendations.append("Over is historically profitable with this ump")
            recommendations.append("Aggressive hitters benefit from tight zone")
        
        if ump.consistency_score < 0.50:
            recommendations.append("⚠️ Low consistency - expect controversial calls")

        return {
            'umpire': ump.name,
            'found': True,
            'games_called': ump.games_called,
            'zone_tendency': ump.favors,
            'zone_width': f"{'Wider' if ump.zone_width > 0 else 'Tighter'} "
                          f"than average by {abs(ump.zone_width)*100:.0f}%",
            'consistency': round(ump.consistency_score * 100, 0),
            'run_impact': round(ump.run_impact, 2),
            'k_rate_impact': round(ump.k_rate_impact, 3),
            'bb_rate_impact': round(ump.bb_rate_impact, 3),
            'scoring_impact': scoring_impact,
            'scoring_note': scoring_note,
            'recommendations': recommendations,
            'over_under_lean': 'over' if ump.run_impact > 0.05 else
                               'under' if ump.run_impact < -0.05 else 'neutral'
        }


# ============================================================================
# PITCHER FATIGUE ENGINE
# ============================================================================

@dataclass
class PitcherWorkload:
    """Pitcher's recent workload data"""
    name: str
    days_rest: int
    last_start_pitches: int
    last_3_starts_avg_pitches: float
    season_innings: float
    career_avg_innings_per_season: float
    pitch_count_trend: str = ""  # "increasing", "stable", "decreasing"
    bullpen_usage_last_3_days: int = 0  # Total pitches from bullpen


class FatigueEngine:
    """
    Predicts pitcher effectiveness based on workload and rest.
    
    Key factors:
    - Days rest: 4 days = optimal for starters, 5+ = rust potential
    - Pitch count: >100 in last start = higher fatigue
    - Season workload: Innings vs career average
    - Bullpen: Heavy recent use = lower effectiveness
    """

    def calculate_fatigue_score(self, pitcher: PitcherWorkload
                                 ) -> Dict[str, Any]:
        """
        Calculate fatigue score (0-100, where 100 = fully rested).
        """
        scores = {}

        # Days rest (optimal: 4 for SP, 1-2 for RP)
        if pitcher.days_rest == 4:
            rest_score = 100
        elif pitcher.days_rest == 5:
            rest_score = 95
        elif pitcher.days_rest == 3:
            rest_score = 80
        elif pitcher.days_rest >= 6:
            rest_score = 85  # Rust factor
        elif pitcher.days_rest <= 2:
            rest_score = 60  # Short rest
        else:
            rest_score = 70
        scores['rest'] = rest_score

        # Pitch count from last start
        if pitcher.last_start_pitches <= 85:
            pitch_score = 100
        elif pitcher.last_start_pitches <= 100:
            pitch_score = 90
        elif pitcher.last_start_pitches <= 110:
            pitch_score = 75
        else:
            pitch_score = 60
        scores['last_start'] = pitch_score

        # Season workload
        if pitcher.career_avg_innings_per_season > 0:
            workload_ratio = pitcher.season_innings / pitcher.career_avg_innings_per_season
            if workload_ratio <= 0.8:
                workload_score = 95
            elif workload_ratio <= 1.0:
                workload_score = 85
            elif workload_ratio <= 1.1:
                workload_score = 70
            else:
                workload_score = 55  # Overworked
        else:
            workload_score = 85
        scores['season_workload'] = workload_score

        # Bullpen fatigue
        if pitcher.bullpen_usage_last_3_days <= 40:
            bullpen_score = 95
        elif pitcher.bullpen_usage_last_3_days <= 70:
            bullpen_score = 80
        elif pitcher.bullpen_usage_last_3_days <= 100:
            bullpen_score = 65
        else:
            bullpen_score = 50
        scores['bullpen'] = bullpen_score

        # Weighted composite
        composite = (
            rest_score * 0.30 +
            pitch_score * 0.25 +
            workload_score * 0.25 +
            bullpen_score * 0.20
        )

        # Performance adjustment
        if composite >= 90:
            performance_adj = 1.02  # Slight boost
            status = "fresh"
        elif composite >= 75:
            performance_adj = 1.00  # Neutral
            status = "normal"
        elif composite >= 60:
            performance_adj = 0.96  # Slight decline
            status = "fatigued"
        else:
            performance_adj = 0.90  # Significant decline
            status = "exhausted"

        return {
            'pitcher': pitcher.name,
            'fatigue_score': round(composite, 1),
            'status': status,
            'performance_multiplier': round(performance_adj, 3),
            'scores': scores,
            'days_rest': pitcher.days_rest,
            'last_pitches': pitcher.last_start_pitches,
            'season_innings': pitcher.season_innings,
            'warnings': self._generate_warnings(pitcher, scores),
            'recommendation': self._get_recommendation(status)
        }

    def _generate_warnings(self, pitcher: PitcherWorkload,
                            scores: Dict) -> List[str]:
        """Generate fatigue warnings"""
        warnings = []
        if pitcher.days_rest <= 3:
            warnings.append(f"⚠️ Short rest ({pitcher.days_rest} days)")
        if pitcher.last_start_pitches > 105:
            warnings.append(f"⚠️ High pitch count last start ({pitcher.last_start_pitches})")
        if scores.get('season_workload', 100) < 70:
            warnings.append("⚠️ Approaching season workload limit")
        if pitcher.bullpen_usage_last_3_days > 80:
            warnings.append(f"⚠️ Heavy bullpen use ({pitcher.bullpen_usage_last_3_days} pitches in 3 days)")
        return warnings

    def _get_recommendation(self, status: str) -> str:
        recs = {
            'fresh': "✅ Pitcher is well-rested. Expect normal or above-average performance.",
            'normal': "✅ Standard workload. No concerns.",
            'fatigued': "⚠️ Some fatigue indicators. May see declining velocity in later innings.",
            'exhausted': "🔴 High fatigue risk. Consider fade or under on this pitcher."
        }
        return recs.get(status, "Unknown")


# ============================================================================
# TRAVEL FATIGUE ENGINE
# ============================================================================

class TravelEngine:
    """Calculates travel fatigue impact on team performance"""

    # Timezone zones for MLB cities
    TEAM_TIMEZONES = {
        'NYY': -5, 'NYM': -5, 'BOS': -5, 'PHI': -5, 'PIT': -5,
        'BAL': -5, 'TB': -5, 'MIA': -5, 'ATL': -5, 'WSH': -5,
        'TOR': -5, 'CLE': -5, 'DET': -5, 'CIN': -5,
        'MIL': -6, 'CHC': -6, 'CWS': -6, 'STL': -6, 'MIN': -6,
        'KC': -6, 'HOU': -6, 'TEX': -6,
        'COL': -7, 'ARI': -7,
        'LAD': -8, 'LAA': -8, 'SD': -8, 'SFG': -8, 'OAK': -8,
        'SEA': -8,
    }

    def calculate_travel_impact(self, team: str,
                                 games_on_road: int,
                                 timezone_changes: int,
                                 day_game_after_night: bool = False
                                 ) -> Dict[str, Any]:
        """Calculate travel fatigue impact"""
        fatigue_score = 100  # Start at 100 (fully rested)

        # Road trip length
        if games_on_road <= 3:
            road_penalty = 1
        elif games_on_road <= 6:
            road_penalty = 3
        elif games_on_road <= 9:
            road_penalty = 7
        else:
            road_penalty = 12
        fatigue_score -= road_penalty

        # Timezone changes
        tz_penalty = timezone_changes * 4  # 4 points per timezone change
        fatigue_score -= tz_penalty

        # Day game after night game
        if day_game_after_night:
            fatigue_score -= 8

        fatigue_score = max(50, fatigue_score)

        # Performance multiplier
        if fatigue_score >= 90:
            multiplier = 1.0
            status = "rested"
        elif fatigue_score >= 80:
            multiplier = 0.98
            status = "mild_fatigue"
        elif fatigue_score >= 70:
            multiplier = 0.96
            status = "fatigued"
        else:
            multiplier = 0.93
            status = "exhausted"

        return {
            'team': team,
            'fatigue_score': fatigue_score,
            'performance_multiplier': multiplier,
            'status': status,
            'games_on_road': games_on_road,
            'timezone_changes': timezone_changes,
            'day_after_night': day_game_after_night,
            'note': self._describe(status, games_on_road, timezone_changes)
        }

    def _describe(self, status: str, road_games: int, tz: int) -> str:
        if status == "exhausted":
            return f"Long road trip ({road_games} games, {tz} TZ changes). Expect diminished performance."
        elif status == "fatigued":
            return f"Moderate travel fatigue ({road_games} road games)."
        elif status == "mild_fatigue":
            return "Minor travel factor."
        return "Team is well-rested."


# ============================================================================
# COMBINED ENVIRONMENTAL FACTOR CALCULATOR
# ============================================================================

class EnvironmentalFactorCalculator:
    """Combines all environmental factors into a single game adjustment"""

    def __init__(self):
        self.weather_engine = WeatherEngine()
        self.umpire_engine = UmpireEngine()
        self.fatigue_engine = FatigueEngine()
        self.travel_engine = TravelEngine()

    def calculate_game_environment(
        self,
        weather: Optional[GameWeather] = None,
        umpire_name: Optional[str] = None,
        home_pitcher: Optional[PitcherWorkload] = None,
        away_pitcher: Optional[PitcherWorkload] = None,
        home_road_games: int = 0,
        away_road_games: int = 0,
        stadium_altitude: int = 0
    ) -> Dict[str, Any]:
        """
        Calculate combined environmental impact on a game.
        Returns adjustments for scoring, HR rate, and team performance.
        """
        result = {
            'scoring_adjustment': 1.0,
            'hr_adjustment': 1.0,
            'home_performance_adj': 1.0,
            'away_performance_adj': 1.0,
            'factors': {},
            'summary': []
        }

        # Weather
        if weather:
            weather_impact = self.weather_engine.calculate_weather_impact(
                weather, stadium_altitude
            )
            result['scoring_adjustment'] *= weather_impact['run_multiplier']
            result['hr_adjustment'] *= weather_impact['hr_multiplier']
            result['factors']['weather'] = weather_impact
            if weather_impact['scoring_environment'] != 'neutral':
                result['summary'].append(weather_impact['summary'])

        # Umpire
        if umpire_name:
            ump_impact = self.umpire_engine.get_umpire_impact(umpire_name)
            if ump_impact.get('found'):
                run_adj = 1.0 + ump_impact.get('run_impact', 0) * 0.05
                result['scoring_adjustment'] *= run_adj
                result['factors']['umpire'] = ump_impact
                if ump_impact.get('scoring_impact') != 'neutral':
                    result['summary'].append(
                        f"Umpire {ump_impact['umpire']}: "
                        f"{ump_impact['scoring_note']}"
                    )

        # Pitcher fatigue
        if home_pitcher:
            home_fatigue = self.fatigue_engine.calculate_fatigue_score(home_pitcher)
            result['away_performance_adj'] *= (
                2 - home_fatigue['performance_multiplier']
            )
            result['factors']['home_pitcher_fatigue'] = home_fatigue
            if home_fatigue['status'] in ('fatigued', 'exhausted'):
                result['summary'].append(
                    f"Home P ({home_pitcher.name}): {home_fatigue['status']}"
                )

        if away_pitcher:
            away_fatigue = self.fatigue_engine.calculate_fatigue_score(away_pitcher)
            result['home_performance_adj'] *= (
                2 - away_fatigue['performance_multiplier']
            )
            result['factors']['away_pitcher_fatigue'] = away_fatigue
            if away_fatigue['status'] in ('fatigued', 'exhausted'):
                result['summary'].append(
                    f"Away P ({away_pitcher.name}): {away_fatigue['status']}"
                )

        # Round results
        result['scoring_adjustment'] = round(result['scoring_adjustment'], 3)
        result['hr_adjustment'] = round(result['hr_adjustment'], 3)
        result['home_performance_adj'] = round(result['home_performance_adj'], 3)
        result['away_performance_adj'] = round(result['away_performance_adj'], 3)

        # Overall assessment
        if result['scoring_adjustment'] >= 1.10:
            result['game_environment'] = "high_scoring"
        elif result['scoring_adjustment'] <= 0.92:
            result['game_environment'] = "low_scoring"
        else:
            result['game_environment'] = "neutral"

        return result


# ============================================================================
# DEMO
# ============================================================================

def demo_environmental_factors():
    """Demonstrate environmental factors engine"""
    print("=" * 70)
    print("🌤️ MLB Predictor - Environmental Factors Demo")
    print("=" * 70)
    print()

    calc = EnvironmentalFactorCalculator()

    # Scenario 1: Hot day, wind blowing out at Coors
    print("1️⃣  SCENARIO: Coors Field (Hot, Wind Out)")
    print("-" * 60)
    weather1 = GameWeather(
        temperature_f=92, wind_speed_mph=12,
        wind_direction=WindDirection.OUT_TO_CF,
        humidity_pct=25, precipitation_chance=0
    )
    result1 = calc.calculate_game_environment(
        weather=weather1,
        umpire_name="Mark Carlson",
        stadium_altitude=5280
    )
    print(f"   Scoring Adj: {result1['scoring_adjustment']:.3f}x")
    print(f"   HR Adj: {result1['hr_adjustment']:.3f}x")
    print(f"   Environment: {result1['game_environment']}")
    for s in result1['summary']:
        print(f"   📝 {s}")
    print()

    # Scenario 2: Cold night, wind blowing in
    print("2️⃣  SCENARIO: Wrigley Field (Cold, Wind In)")
    print("-" * 60)
    weather2 = GameWeather(
        temperature_f=48, wind_speed_mph=15,
        wind_direction=WindDirection.IN_FROM_CF,
        humidity_pct=65, precipitation_chance=0.1
    )
    result2 = calc.calculate_game_environment(
        weather=weather2,
        umpire_name="Joe West",
        stadium_altitude=595
    )
    print(f"   Scoring Adj: {result2['scoring_adjustment']:.3f}x")
    print(f"   HR Adj: {result2['hr_adjustment']:.3f}x")
    print(f"   Environment: {result2['game_environment']}")
    for s in result2['summary']:
        print(f"   📝 {s}")
    print()

    # Scenario 3: Dome game with fatigued pitcher
    print("3️⃣  SCENARIO: Tropicana Field (Dome, Fatigued Away SP)")
    print("-" * 60)
    weather3 = GameWeather(
        temperature_f=72, wind_speed_mph=0,
        wind_direction=WindDirection.CALM,
        humidity_pct=50, precipitation_chance=0,
        is_dome=True
    )
    away_pitcher = PitcherWorkload(
        name="Gerrit Cole", days_rest=3,
        last_start_pitches=112, last_3_starts_avg_pitches=104,
        season_innings=165.2, career_avg_innings_per_season=185,
        bullpen_usage_last_3_days=65
    )
    result3 = calc.calculate_game_environment(
        weather=weather3,
        umpire_name="Pat Hoberg",
        away_pitcher=away_pitcher
    )
    print(f"   Scoring Adj: {result3['scoring_adjustment']:.3f}x")
    print(f"   Home Perf Adj: {result3['home_performance_adj']:.3f}x")
    print(f"   Away Perf Adj: {result3['away_performance_adj']:.3f}x")
    fatigue = result3['factors'].get('away_pitcher_fatigue', {})
    if fatigue:
        print(f"   Pitcher Fatigue: {fatigue.get('fatigue_score', 0):.0f}/100 "
              f"({fatigue.get('status', '?')})")
        for w in fatigue.get('warnings', []):
            print(f"   {w}")
    print()

    # Umpire database
    print("4️⃣  UMPIRE DATABASE LOOKUP")
    print("-" * 60)
    ump_engine = UmpireEngine()
    for name in ["Angel Hernandez", "Pat Hoberg", "CB Bucknor", "Joe West"]:
        impact = ump_engine.get_umpire_impact(name)
        if impact.get('found'):
            print(f"   {impact['umpire']}: Zone={impact['zone_tendency']} | "
                  f"Consistency={impact['consistency']}% | "
                  f"O/U lean={impact['over_under_lean']}")
    print()

    print("=" * 70)
    print("✅ Environmental Factors Demo Complete")
    print("=" * 70)

    return calc


if __name__ == "__main__":
    demo_environmental_factors()
