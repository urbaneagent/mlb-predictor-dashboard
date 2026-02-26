#!/usr/bin/env python3
"""
MLB Predictor v121 - Temperature & Humidity Carry Distance Physics

WHAT'S NEW:
- Temperature-based ball carry adjustment (+4 ft per 10°F for HR balls)
- Humidity-based carry adjustment (+1 ft per 50% humidity)
- Physics-backed HR probability boosts (research: Weather Applied Metrics)
- Combined with existing park factors + spray angles for maximum edge

RESEARCH CITATIONS:
- Weather Applied Metrics: HR ball gains ~4 ft per 10°F temperature rise
- Alan Nathan (Baseball Physics): Average fly ball gains ~3 ft per 10°F
- Humidity effect: +1 ft per 50% RH (minor but measurable)

HOW IT WORKS:
1. Fetch game-time weather (temp, humidity) from Open-Meteo API
2. Calculate carry distance boost from baseline (70°F, 50% RH)
3. Convert distance boost to HR probability adjustment
4. Apply to batters based on power profile (higher power = more temp sensitivity)

EDGE CASES THIS EXPLOITS:
- 90°F game at Coors Field = +8 ft carry (temp) + altitude boost = MASSIVE
- 50°F game at Oracle Park = -8 ft carry (temp) + marine layer = HR KILLER
- High-power pull hitters benefit most from hot weather (exit velo + temp = max carry)

Author: Mike Ross (The Architect)
Date: February 24, 2026
Version: 121
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ============================================================================
# WEATHER API INTEGRATION
# ============================================================================

def get_game_weather(lat: float, lon: float, game_datetime: str) -> Dict[str, float]:
    """
    Fetch game-time weather from Open-Meteo API.
    
    Args:
        lat: Stadium latitude
        lon: Stadium longitude
        game_datetime: ISO format datetime (e.g., "2026-07-15T19:00")
    
    Returns:
        {
            'temperature_f': 85.0,
            'humidity_pct': 65.0,
            'wind_speed_mph': 12.0,
            'wind_direction_deg': 180.0
        }
    """
    try:
        # Parse game datetime
        dt = datetime.fromisoformat(game_datetime.replace('Z', '+00:00'))
        date_str = dt.strftime('%Y-%m-%d')
        
        # Open-Meteo API (free, no key required)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'timezone': 'America/New_York',
            'start_date': date_str,
            'end_date': date_str
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Find closest hour to game time
        hourly = data['hourly']
        game_hour = dt.hour
        
        return {
            'temperature_f': hourly['temperature_2m'][game_hour],
            'humidity_pct': hourly['relative_humidity_2m'][game_hour],
            'wind_speed_mph': hourly['wind_speed_10m'][game_hour],
            'wind_direction_deg': hourly['wind_direction_10m'][game_hour]
        }
    
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        # Return neutral defaults
        return {
            'temperature_f': 70.0,
            'humidity_pct': 50.0,
            'wind_speed_mph': 5.0,
            'wind_direction_deg': 0.0
        }


# ============================================================================
# TEMPERATURE & HUMIDITY PHYSICS
# ============================================================================

def calculate_temp_carry_boost(temp_f: float, baseline_temp: float = 70.0) -> float:
    """
    Calculate ball carry distance adjustment based on temperature.
    
    Research (Weather Applied Metrics):
    - Home run-like fly ball gains ~4 ft per 10°F temperature rise
    - Average fly ball gains ~3 ft per 10°F
    - Using 4 ft for power hitters (exit velo >95 mph)
    
    Args:
        temp_f: Current temperature (°F)
        baseline_temp: Neutral baseline (70°F)
    
    Returns:
        Distance boost in feet (positive = helps hitters)
    
    Examples:
        85°F: +6 ft carry (vs 70°F baseline)
        50°F: -8 ft carry (cold air = more drag)
        95°F: +10 ft carry (scorching hot day)
    """
    temp_diff = temp_f - baseline_temp
    carry_boost_ft = (temp_diff / 10.0) * 4.0
    return carry_boost_ft


def calculate_humidity_carry_boost(humidity_pct: float, baseline_humidity: float = 50.0) -> float:
    """
    Calculate ball carry distance adjustment based on humidity.
    
    Research (Weather Applied Metrics):
    - Humid air is less dense (water vapor lighter than dry air)
    - ~1 ft carry boost per 50% humidity increase
    - Effect is small but measurable
    
    Args:
        humidity_pct: Current relative humidity (0-100)
        baseline_humidity: Neutral baseline (50%)
    
    Returns:
        Distance boost in feet (positive = helps hitters)
    
    Examples:
        100% humidity: +1 ft carry
        0% humidity: -1 ft carry
        75% humidity: +0.5 ft carry
    """
    humidity_diff = humidity_pct - baseline_humidity
    carry_boost_ft = (humidity_diff / 50.0) * 1.0
    return carry_boost_ft


def calculate_total_weather_carry_boost(
    temp_f: float,
    humidity_pct: float,
    batter_power: float
) -> Tuple[float, float]:
    """
    Calculate total weather-driven carry distance boost.
    
    Args:
        temp_f: Temperature (°F)
        humidity_pct: Relative humidity (%)
        batter_power: Power index (0-100, based on exit velo + barrel rate)
    
    Returns:
        (total_carry_boost_ft, hr_prob_adjustment_pct)
    
    Logic:
    - Temperature effect scales with batter power (high power = more temp sensitivity)
    - Humidity effect is constant (doesn't scale with power)
    - Every 5 ft of carry ≈ 1% HR probability swing (based on park dimensions)
    """
    # Temperature boost (scaled by batter power)
    temp_boost_ft = calculate_temp_carry_boost(temp_f)
    power_scalar = 0.5 + (batter_power / 100.0) * 0.5  # 0.5-1.0 range
    temp_boost_ft *= power_scalar
    
    # Humidity boost (not power-scaled)
    humidity_boost_ft = calculate_humidity_carry_boost(humidity_pct)
    
    # Total carry boost
    total_boost_ft = temp_boost_ft + humidity_boost_ft
    
    # Convert to HR probability adjustment
    # Rule of thumb: 5 ft carry ≈ 1% HR probability swing
    # (Based on typical fence distances being 380-410 ft)
    hr_prob_adjustment_pct = (total_boost_ft / 5.0) * 1.0
    
    return total_boost_ft, hr_prob_adjustment_pct


# ============================================================================
# BATTER POWER INDEX (for temp scaling)
# ============================================================================

def calculate_batter_power_index(
    avg_exit_velo: float,
    barrel_rate: float,
    hard_contact_rate: float
) -> float:
    """
    Calculate batter power index (0-100 scale).
    
    High-power batters benefit more from hot weather because:
    - Hard contact + hot air = maximum carry distance
    - Temperature effect compounds with exit velocity
    
    Args:
        avg_exit_velo: Average exit velocity (mph, typical range 85-95)
        barrel_rate: Barrel rate (%, typical range 5-15)
        hard_contact_rate: Hard contact rate (%, typical range 30-50)
    
    Returns:
        Power index (0-100)
    
    Examples:
        Elite power hitter (93 mph, 12% barrel, 48% hard): ~85-90
        Average hitter (89 mph, 8% barrel, 40% hard): ~50-60
        Light hitter (85 mph, 4% barrel, 32% hard): ~20-30
    """
    # Normalize each component to 0-100 scale
    velo_score = ((avg_exit_velo - 80) / 15.0) * 100  # 80-95 mph range
    barrel_score = (barrel_rate / 0.15) * 100  # 0-15% range
    hard_contact_score = ((hard_contact_rate - 25) / 30.0) * 100  # 25-55% range
    
    # Weighted average (exit velo is most important)
    power_index = (
        velo_score * 0.5 +
        barrel_score * 0.3 +
        hard_contact_score * 0.2
    )
    
    # Clamp to 0-100
    power_index = max(0, min(100, power_index))
    
    return power_index


# ============================================================================
# WEATHER-ADJUSTED HR PROBABILITY
# ============================================================================

def apply_weather_adjustment(
    base_hr_prob: float,
    temp_f: float,
    humidity_pct: float,
    batter_power: float,
    park_name: str
) -> Tuple[float, Dict[str, float]]:
    """
    Apply temperature + humidity adjustments to HR probability.
    
    Args:
        base_hr_prob: Base HR probability (0-1 scale)
        temp_f: Temperature (°F)
        humidity_pct: Relative humidity (%)
        batter_power: Batter power index (0-100)
        park_name: Stadium name (for context)
    
    Returns:
        (adjusted_hr_prob, adjustment_breakdown)
    
    Examples:
        90°F, 70% humidity, power=85:
            - Temp boost: +8 ft carry → +1.6% HR prob
            - Humidity boost: +0.4 ft carry → +0.08% HR prob
            - Total: +1.68% HR prob boost
        
        50°F, 30% humidity, power=60:
            - Temp penalty: -6 ft carry → -1.2% HR prob
            - Humidity penalty: -0.4 ft carry → -0.08% HR prob
            - Total: -1.28% HR prob penalty
    """
    # Calculate weather carry boost
    carry_boost_ft, hr_prob_adj_pct = calculate_total_weather_carry_boost(
        temp_f, humidity_pct, batter_power
    )
    
    # Convert percentage to decimal adjustment
    hr_prob_adj = hr_prob_adj_pct / 100.0
    
    # Apply adjustment (additive)
    adjusted_hr_prob = base_hr_prob + hr_prob_adj
    
    # Clamp to valid probability range
    adjusted_hr_prob = max(0.0, min(1.0, adjusted_hr_prob))
    
    # Breakdown for debugging/reporting
    breakdown = {
        'base_hr_prob': base_hr_prob,
        'temp_f': temp_f,
        'humidity_pct': humidity_pct,
        'batter_power': batter_power,
        'carry_boost_ft': carry_boost_ft,
        'hr_prob_adj_pct': hr_prob_adj_pct,
        'adjusted_hr_prob': adjusted_hr_prob,
        'park_name': park_name
    }
    
    return adjusted_hr_prob, breakdown


# ============================================================================
# DEMO: EXTREME WEATHER SCENARIOS
# ============================================================================

def demo_weather_scenarios():
    """
    Demonstrate weather impact on HR probability across extreme scenarios.
    """
    print("=" * 80)
    print("MLB PREDICTOR v121 - TEMPERATURE & HUMIDITY CARRY DISTANCE PHYSICS")
    print("=" * 80)
    print()
    
    # Define test scenarios
    scenarios = [
        {
            'name': 'Scorching Summer Day (Coors Field)',
            'temp_f': 95,
            'humidity_pct': 30,
            'power': 90,
            'base_hr_prob': 0.12,
            'park': 'Coors Field'
        },
        {
            'name': 'Cold Spring Night (Oracle Park)',
            'temp_f': 52,
            'humidity_pct': 80,
            'power': 85,
            'base_hr_prob': 0.08,
            'park': 'Oracle Park'
        },
        {
            'name': 'Humid Summer Evening (Cincinnati)',
            'temp_f': 88,
            'humidity_pct': 85,
            'power': 75,
            'base_hr_prob': 0.15,
            'park': 'Great American Ball Park'
        },
        {
            'name': 'Neutral Conditions (Baseline)',
            'temp_f': 70,
            'humidity_pct': 50,
            'power': 70,
            'base_hr_prob': 0.10,
            'park': 'Generic Stadium'
        },
        {
            'name': 'Late Season Cold (Detroit)',
            'temp_f': 48,
            'humidity_pct': 60,
            'power': 60,
            'base_hr_prob': 0.09,
            'park': 'Comerica Park'
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        print(f"\n{'─' * 80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'─' * 80}")
        print(f"Park: {scenario['park']}")
        print(f"Temperature: {scenario['temp_f']}°F")
        print(f"Humidity: {scenario['humidity_pct']}%")
        print(f"Batter Power Index: {scenario['power']}/100")
        print()
        
        # Calculate adjustment
        adj_prob, breakdown = apply_weather_adjustment(
            scenario['base_hr_prob'],
            scenario['temp_f'],
            scenario['humidity_pct'],
            scenario['power'],
            scenario['park']
        )
        
        # Display results
        print(f"Base HR Probability: {breakdown['base_hr_prob']:.1%}")
        print(f"Carry Boost from Weather: {breakdown['carry_boost_ft']:+.1f} ft")
        print(f"HR Probability Adjustment: {breakdown['hr_prob_adj_pct']:+.2f}%")
        print(f"Weather-Adjusted HR Prob: {breakdown['adjusted_hr_prob']:.1%}")
        
        # Impact assessment
        prob_change_pct = ((adj_prob - scenario['base_hr_prob']) / scenario['base_hr_prob']) * 100
        if abs(prob_change_pct) > 5:
            impact = "🔥 MASSIVE EDGE" if prob_change_pct > 0 else "❌ MAJOR PENALTY"
        elif abs(prob_change_pct) > 2:
            impact = "⚠️ SIGNIFICANT" if prob_change_pct > 0 else "⚠️ NOTABLE PENALTY"
        else:
            impact = "✅ MINOR" if prob_change_pct > 0 else "✅ MINOR PENALTY"
        
        print(f"\nIMPACT: {impact} ({prob_change_pct:+.1f}% relative change)")
    
    print()
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. Hot weather (90°F+) can boost HR probability by 1-2% (EXPLOITABLE)")
    print("2. Cold weather (50°F-) can reduce HR probability by 1-2% (FADE POWER HITTERS)")
    print("3. High-power batters (90+ power index) benefit most from hot weather")
    print("4. Humidity effect is small (~0.5% max) but still measurable")
    print("5. Combined with park factors + spray angles = MAXIMUM EDGE")
    print()
    print("NEXT STEPS:")
    print("1. Integrate into main predictor for live game predictions")
    print("2. Backtest on 2024/2025 data to validate edge size")
    print("3. Add pressure-based adjustments (-2 ft per 0.3 inHg drop)")
    print("=" * 80)
    print()


# ============================================================================
# INTEGRATION WITH PARK FACTORS
# ============================================================================

def get_combined_adjustments(
    temp_f: float,
    humidity_pct: float,
    batter_power: float,
    park_hr_factor: float,
    park_name: str,
    base_hr_prob: float
) -> Dict[str, float]:
    """
    Combine weather + park factor adjustments.
    
    This is the master function that integrates:
    - Park dimensions (v120 spray angle fit)
    - Park HR factors (v120 park factors)
    - Temperature carry boost (v121 NEW)
    - Humidity carry boost (v121 NEW)
    
    Args:
        temp_f: Temperature (°F)
        humidity_pct: Relative humidity (%)
        batter_power: Power index (0-100)
        park_hr_factor: Park HR factor (0.85-1.25 range)
        park_name: Stadium name
        base_hr_prob: Base HR probability before adjustments
    
    Returns:
        {
            'base_hr_prob': 0.10,
            'park_adjusted_prob': 0.11,  # After park factor
            'weather_adjusted_prob': 0.12,  # After temp + humidity
            'final_hr_prob': 0.12,
            'park_boost_pct': 10.0,
            'weather_boost_pct': 9.1,
            'total_boost_pct': 20.0
        }
    """
    # Step 1: Apply park factor
    park_adjusted_prob = base_hr_prob * park_hr_factor
    
    # Step 2: Apply weather adjustment
    weather_adjusted_prob, weather_breakdown = apply_weather_adjustment(
        park_adjusted_prob,
        temp_f,
        humidity_pct,
        batter_power,
        park_name
    )
    
    # Calculate boost percentages
    park_boost_pct = ((park_adjusted_prob - base_hr_prob) / base_hr_prob) * 100
    weather_boost_pct = ((weather_adjusted_prob - park_adjusted_prob) / park_adjusted_prob) * 100
    total_boost_pct = ((weather_adjusted_prob - base_hr_prob) / base_hr_prob) * 100
    
    return {
        'base_hr_prob': base_hr_prob,
        'park_adjusted_prob': park_adjusted_prob,
        'weather_adjusted_prob': weather_adjusted_prob,
        'final_hr_prob': weather_adjusted_prob,
        'park_boost_pct': park_boost_pct,
        'weather_boost_pct': weather_boost_pct,
        'total_boost_pct': total_boost_pct,
        'carry_boost_ft': weather_breakdown['carry_boost_ft']
    }


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    demo_weather_scenarios()
