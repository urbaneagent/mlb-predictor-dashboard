#!/usr/bin/env python3
"""
MLB Weather Integration - Open-Meteo API
========================================
Fetches weather forecasts for MLB stadiums
Free API, no key required

Author: Mike Ross
Date: 2026-02-21
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# ============================================================================
# STADIUM LOCATIONS (lat/lon for all 30 MLB teams)
# ============================================================================

STADIUMS = {
    'ARI': {'name': 'Chase Field', 'lat': 33.4455, 'lon': -112.0667},
    'ATL': {'name': 'Truist Park', 'lat': 33.8908, 'lon': -84.4679},
    'BAL': {'name': 'Camden Yards', 'lat': 39.2839, 'lon': -76.6219},
    'BOS': {'name': 'Fenway Park', 'lat': 42.3467, 'lon': -71.0972},
    'CHC': {'name': 'Wrigley Field', 'lat': 41.9484, 'lon': -87.6553},
    'CWS': {'name': 'Guaranteed Rate Field', 'lat': 41.8300, 'lon': -87.6339},
    'CIN': {'name': 'Great American Ball Park', 'lat': 39.0975, 'lon': -84.5070},
    'CLE': {'name': 'Progressive Field', 'lat': 41.4962, 'lon': -81.6852},
    'COL': {'name': 'Coors Field', 'lat': 39.7562, 'lon': -104.9942},
    'DET': {'name': 'Comerica Park', 'lat': 42.3390, 'lon': -83.0485},
    'HOU': {'name': 'Minute Maid Park', 'lat': 29.7573, 'lon': -95.3555},
    'KC': {'name': 'Kauffman Stadium', 'lat': 39.0517, 'lon': -94.4803},
    'LAA': {'name': 'Angel Stadium', 'lat': 33.8003, 'lon': -117.8827},
    'LAD': {'name': 'Dodger Stadium', 'lat': 34.0739, 'lon': -118.2400},
    'MIA': {'name': 'loanDepot Park', 'lat': 25.7780, 'lon': -80.2190},
    'MIL': {'name': 'American Family Field', 'lat': 43.0280, 'lon': -87.9712},
    'MIN': {'name': 'Target Field', 'lat': 44.9817, 'lon': -93.2778},
    'NYM': {'name': 'Citi Field', 'lat': 40.7571, 'lon': -73.8458},
    'NYY': {'name': 'Yankee Stadium', 'lat': 40.8296, 'lon': -73.9262},
    'OAK': {'name': 'Oakland Coliseum', 'lat': 37.7516, 'lon': -122.2005},
    'PHI': {'name': 'Citizens Bank Park', 'lat': 39.9061, 'lon': -75.1665},
    'PIT': {'name': 'PNC Park', 'lat': 40.4469, 'lon': -80.0057},
    'SD': {'name': 'Petco Park', 'lat': 32.7073, 'lon': -117.1566},
    'SEA': {'name': 'T-Mobile Park', 'lat': 47.5914, 'lon': -122.3325},
    'SF': {'name': 'Oracle Park', 'lat': 37.7786, 'lon': -122.3893},
    'STL': {'name': 'Busch Stadium', 'lat': 38.6226, 'lon': -90.1928},
    'TB': {'name': 'Tropicana Field', 'lat': 27.7683, 'lon': -82.6534},
    'TEX': {'name': 'Globe Life Field', 'lat': 32.7473, 'lon': -97.0823},
    'TOR': {'name': 'Rogers Centre', 'lat': 43.6414, 'lon': -79.3894},
    'WSH': {'name': 'Nationals Park', 'lat': 38.8729, 'lon': -77.0074}
}

def get_weather_forecast(team_abbr, date=None):
    """
    Get weather forecast for a team/stadium
    
    Args:
        team_abbr: 3-letter team code (e.g., 'NYY')
        date: datetime object or string 'YYYY-MM-DD'
    
    Returns:
        dict with weather data
    """
    if team_abbr not in STADIUMS:
        return None
    
    stadium = STADIUMS[team_abbr]
    lat, lon = stadium['lat'], stadium['lon']
    
    # Parse date
    if date is None:
        target_date = datetime.now()
    elif isinstance(date, str):
        target_date = datetime.strptime(date, '%Y-%m-%d')
    else:
        target_date = date
    
    # Open-Meteo API (free, no key)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation_probability',
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_probability_max',
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'America/New_York',
        'forecast_days': 7
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        # Find the target date's hourly data
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        
        # Find closest hour to typical game time (7pm local)
        game_hour = 19
        target_str = target_date.strftime('%Y-%m-%d')
        
        weather = {
            'team': team_abbr,
            'stadium': stadium['name'],
            'date': target_date.strftime('%Y-%m-%d'),
            'temperature': None,
            'humidity': None,
            'wind_speed': None,
            'wind_direction': None,
            'precipitation_prob': None,
            'forecast_max': None,
            'forecast_min': None
        }
        
        # Find hourly data for target date at ~7pm
        for i, t in enumerate(times):
            if target_str in t:
                # Look for hour around 7pm (19:00)
                if 'T19:' in t or 'T18:' in t or 'T20:' in t:
                    weather['temperature'] = hourly['temperature_2m'][i]
                    weather['humidity'] = hourly['relative_humidity_2m'][i]
                    weather['wind_speed'] = hourly['wind_speed_10m'][i]
                    weather['wind_direction'] = hourly['wind_direction_10m'][i]
                    weather['precipitation_prob'] = hourly['precipitation_probability'][i]
                    break
        
        # Get daily high/low
        daily = data.get('daily', {})
        daily_times = daily.get('time', [])
        for i, t in enumerate(daily_times):
            if target_str in t:
                weather['forecast_max'] = daily['temperature_2m_max'][i]
                weather['forecast_min'] = daily['temperature_2m_min'][i]
                weather['precipitation_prob'] = daily['precipitation_probability_max'][i]
                break
        
        return weather
        
    except Exception as e:
        return {'error': str(e), 'team': team_abbr}

def calculate_weather_impact(weather):
    """
    Calculate HR/Hit probability adjustment based on weather
    
    Returns:
        dict with impact factors (1.0 = no change)
    """
    if not weather or 'error' in weather:
        return {'hr_impact': 1.0, 'hit_impact': 1.0}
    
    hr_impact = 1.0
    hit_impact = 1.0
    
    temp = weather.get('temperature', 70)
    wind = weather.get('wind_speed', 10)
    wind_dir = weather.get('wind_direction', 180)
    precip = weather.get('precipitation_prob', 0)
    
    # Temperature impact (optimal ~75°F)
    if temp > 85:
        hr_impact *= 1.10  # Hot air = more carry
        hit_impact *= 1.05
    elif temp < 55:
        hr_impact *= 0.90  # Cold air = less carry
        hit_impact *= 0.95
    
    # Wind impact (wind blowing out = more HRs)
    # Wind direction: 0=North, 90=East, 180=South, 270=West
    # HR friendly: wind from LF to RF (春风 = out to center/right for pull hitters)
    if wind > 15:
        # Strong wind
        if 135 <= wind_dir <= 225:  # Southerly wind (out to CF/RF)
            hr_impact *= 1.15
        elif 315 <= wind_dir <= 45:  # Northerly wind (in from CF/LF)
            hr_impact *= 0.90
    
    # Precipitation
    if precip > 50:
        hr_impact *= 0.80
        hit_impact *= 0.90
    
    return {
        'hr_impact': round(hr_impact, 3),
        'hit_impact': round(hit_impact, 3),
        'temp': temp,
        'wind_speed': wind,
        'wind_direction': wind_dir,
        'precipitation_prob': precip
    }

def get_all_team_weather(date=None):
    """Get weather for all 30 teams"""
    results = []
    for team in STADIUMS:
        w = get_weather_forecast(team, date)
        if w:
            w['impact'] = calculate_weather_impact(w)
            results.append(w)
        time.sleep(0.1)  # Rate limit
    return results

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("⚾ Testing Weather API...")
    
    # Test a few stadiums
    test_teams = ['NYY', 'COL', 'BOS', 'LAD']
    
    for team in test_teams:
        w = get_weather_forecast(team)
        if w and 'error' not in w:
            impact = calculate_weather_impact(w)
            print(f"\n{team} ({w.get('stadium')}):")
            print(f"   Temp: {w.get('temperature')}°F")
            print(f"   Wind: {w.get('wind_speed')} mph from {w.get('wind_direction')}°")
            print(f"   Precip: {w.get('precipitation_prob')}%")
            print(f"   HR Impact: {impact['hr_impact']:.1%}")
        else:
            print(f"\n{team}: Error getting weather")
    
    print("\n✅ Weather integration complete!")
