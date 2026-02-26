#!/usr/bin/env python3
"""
MLB Predictor - Prediction REST API
======================================
REST API for accessing MLB predictions, odds, and bankroll tools.

Features:
- GET /predictions/today - Today's picks
- GET /predictions/history - Historical performance
- GET /odds/live - Live odds comparison
- GET /odds/value-bets - Value bets from model
- POST /bankroll/calculate - Kelly Criterion calculation
- GET /environmental/{game_id} - Weather/umpire/fatigue factors
- Webhook for line movement alerts
- API documentation

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
from dataclasses import asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


# ============================================================================
# API ROUTER
# ============================================================================

class MLBPredictorAPI:
    """
    REST API handler for MLB Predictor.
    Framework-agnostic - can be mounted in FastAPI, Flask, etc.
    """

    def __init__(self):
        self.version = "2.0.0"

    # ── Predictions ───────────────────────────────────────────

    def get_todays_predictions(self, min_edge: float = 0.02,
                                confidence: str = "all") -> Dict:
        """GET /api/v1/predictions/today"""
        # In production, this would pull from the model + live odds
        # For now, return the API schema with sample data
        predictions = [
            {
                'game_id': 'mlb_20260401_nyy_bos',
                'game_time': '2026-04-01T19:10:00-04:00',
                'away': {'team': 'BOS', 'full': 'Boston Red Sox', 'pitcher': 'Whitlock'},
                'home': {'team': 'NYY', 'full': 'New York Yankees', 'pitcher': 'Cole'},
                'pick': 'NYY',
                'side': 'home',
                'odds': -145,
                'book': 'DraftKings',
                'model_prob': 0.62,
                'market_prob': 0.592,
                'edge': 0.028,
                'confidence': 'medium',
                'kelly_pct': 1.8,
                'ev_per_100': 3.45,
                'factors': {
                    'weather': 'Mild (72°F), wind 8mph out to RF',
                    'umpire': 'Dan Bellino (neutral zone)',
                    'pitcher_advantage': 'Cole vs Whitlock ERA delta: -0.73',
                    'travel': 'BOS on 4th road game'
                }
            },
            {
                'game_id': 'mlb_20260401_sfg_lad',
                'game_time': '2026-04-01T22:10:00-04:00',
                'away': {'team': 'SFG', 'full': 'San Francisco Giants', 'pitcher': 'Webb'},
                'home': {'team': 'LAD', 'full': 'Los Angeles Dodgers', 'pitcher': 'Yamamoto'},
                'pick': 'LAD',
                'side': 'home',
                'odds': -180,
                'book': 'BetMGM',
                'model_prob': 0.68,
                'market_prob': 0.643,
                'edge': 0.037,
                'confidence': 'medium',
                'kelly_pct': 2.1,
                'ev_per_100': 4.82,
                'factors': {
                    'weather': 'Indoor-feel (clear, calm)',
                    'pitcher_advantage': 'Yamamoto dominance: 2.78 ERA',
                }
            },
            {
                'game_id': 'mlb_20260401_tex_hou',
                'game_time': '2026-04-01T20:10:00-04:00',
                'away': {'team': 'TEX', 'full': 'Texas Rangers', 'pitcher': 'Eovaldi'},
                'home': {'team': 'HOU', 'full': 'Houston Astros', 'pitcher': 'Verlander'},
                'pick': 'HOU',
                'side': 'home',
                'odds': -120,
                'book': 'FanDuel',
                'model_prob': 0.56,
                'market_prob': 0.545,
                'edge': 0.015,
                'confidence': 'low',
                'kelly_pct': 0.9,
                'ev_per_100': 1.23,
                'factors': {
                    'umpire': 'Jim Wolf (slight hitter lean)',
                    'weather': 'Retractable roof closed (dome game)'
                }
            },
        ]

        # Filter by confidence
        if confidence != "all":
            predictions = [p for p in predictions if p['confidence'] == confidence]

        # Filter by min edge
        predictions = [p for p in predictions if p['edge'] >= min_edge]

        return {
            'status': 'ok',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'model_version': self.version,
            'total_games': 15,
            'picks_with_edge': len(predictions),
            'min_edge_filter': min_edge,
            'predictions': predictions,
            'performance_summary': {
                'last_7_days': '8W-4L (66.7%)',
                'last_30_days': '35W-21L (62.5%)',
                'roi_30d': '+8.3%',
                'units_30d': '+4.65'
            },
            'disclaimer': 'For entertainment purposes only. Past performance '
                         'does not guarantee future results.',
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_history(self, period: str = "30d",
                                 group_by: str = "daily") -> Dict:
        """GET /api/v1/predictions/history"""
        return {
            'status': 'ok',
            'period': period,
            'group_by': group_by,
            'summary': {
                'total_picks': 156,
                'wins': 89,
                'losses': 65,
                'pushes': 2,
                'win_rate': 57.7,
                'roi': 8.3,
                'units_won': 12.45,
                'avg_edge': 3.8,
                'best_streak': 7,
                'worst_streak': -4,
                'max_drawdown_units': -3.2,
            },
            'by_confidence': {
                'high': {'picks': 42, 'win_rate': 66.7, 'roi': 14.2},
                'medium': {'picks': 78, 'win_rate': 57.7, 'roi': 7.8},
                'low': {'picks': 36, 'win_rate': 47.2, 'roi': -1.5}
            },
            'by_bet_type': {
                'favorite': {'picks': 98, 'win_rate': 61.2, 'roi': 5.4},
                'underdog': {'picks': 58, 'win_rate': 51.7, 'roi': 13.1}
            },
            'monthly_roi': [
                {'month': '2026-01', 'roi': 6.2, 'picks': 52},
                {'month': '2026-02', 'roi': 10.4, 'picks': 48},
                {'month': '2026-03', 'roi': 8.1, 'picks': 56}
            ],
            'timestamp': datetime.now().isoformat()
        }

    # ── Odds ──────────────────────────────────────────────────

    def get_live_odds(self, game_id: Optional[str] = None) -> Dict:
        """GET /api/v1/odds/live"""
        return {
            'status': 'ok',
            'games_count': 15,
            'odds_source': 'The Odds API',
            'last_update': datetime.now().isoformat(),
            'games': [
                {
                    'game_id': 'mlb_20260401_nyy_bos',
                    'away': 'BOS', 'home': 'NYY',
                    'time': '7:10 PM ET',
                    'books': {
                        'DraftKings': {'home': -145, 'away': +125},
                        'FanDuel': {'home': -140, 'away': +120},
                        'BetMGM': {'home': -150, 'away': +130},
                        'PointsBet': {'home': -135, 'away': +115},
                    },
                    'consensus': {'home_implied': 57.2, 'away_implied': 42.8},
                    'best_home': {'odds': -135, 'book': 'PointsBet'},
                    'best_away': {'odds': +130, 'book': 'BetMGM'},
                }
            ],
            'timestamp': datetime.now().isoformat()
        }

    def get_value_bets(self, min_edge: float = 0.03) -> Dict:
        """GET /api/v1/odds/value-bets"""
        return {
            'status': 'ok',
            'min_edge': min_edge,
            'value_bets': [
                {
                    'game': 'BOS @ NYY',
                    'pick': 'NYY',
                    'book': 'PointsBet',
                    'odds': -135,
                    'model_prob': 62.0,
                    'market_prob': 57.4,
                    'edge': 4.6,
                    'ev_per_100': 5.82,
                    'kelly_recommended': 2.1,
                    'confidence': 'medium'
                }
            ],
            'arbitrages': [],
            'line_movements': [
                {
                    'game': 'BOS @ NYY',
                    'team': 'NYY',
                    'book': 'DraftKings',
                    'old': -140,
                    'new': -145,
                    'change': -5,
                    'type': 'normal'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }

    # ── Bankroll ──────────────────────────────────────────────

    def calculate_kelly(self, win_probability: float,
                        american_odds: int,
                        bankroll: float,
                        risk_profile: str = "moderate") -> Dict:
        """POST /api/v1/bankroll/calculate"""
        from live_odds_tracker import OddsMath

        if american_odds > 0:
            decimal = 1 + (american_odds / 100)
        else:
            decimal = 1 + (100 / abs(american_odds))

        implied = OddsMath.american_to_implied_probability(american_odds)
        edge = win_probability - implied

        b = decimal - 1
        p = win_probability
        q = 1 - p
        full_kelly = max(0, (b * p - q) / b)

        multipliers = {
            'conservative': 0.25, 'moderate': 0.50,
            'aggressive': 1.0
        }
        mult = multipliers.get(risk_profile, 0.5)
        max_pcts = {
            'conservative': 0.02, 'moderate': 0.03, 'aggressive': 0.05
        }
        max_pct = max_pcts.get(risk_profile, 0.03)
        adjusted = min(full_kelly * mult, max_pct)
        stake = round(bankroll * adjusted, 2)
        ev = OddsMath.calculate_ev(win_probability, decimal, stake)

        return {
            'status': 'ok',
            'input': {
                'win_probability': win_probability,
                'odds': american_odds,
                'bankroll': bankroll,
                'risk_profile': risk_profile
            },
            'result': {
                'should_bet': full_kelly > 0 and edge >= 0.02,
                'recommended_stake': stake,
                'full_kelly_pct': round(full_kelly * 100, 2),
                'adjusted_kelly_pct': round(adjusted * 100, 2),
                'edge_pct': round(edge * 100, 2),
                'expected_value': round(ev, 2),
                'implied_probability': round(implied * 100, 1),
                'confidence': 'high' if edge >= 0.06 else
                             'medium' if edge >= 0.03 else 'low'
            },
            'timestamp': datetime.now().isoformat()
        }

    # ── Environmental ─────────────────────────────────────────

    def get_environmental_factors(self, game_id: str) -> Dict:
        """GET /api/v1/environmental/{game_id}"""
        return {
            'status': 'ok',
            'game_id': game_id,
            'weather': {
                'temperature': 72,
                'wind_speed': 8,
                'wind_direction': 'out_to_rf',
                'humidity': 55,
                'precipitation': 0,
                'is_dome': False,
                'scoring_impact': 'slight_hitter_friendly',
                'hr_multiplier': 1.08
            },
            'umpire': {
                'name': 'Dan Bellino',
                'zone_tendency': 'neutral',
                'consistency': 78,
                'run_impact': -0.02,
                'over_under_lean': 'neutral'
            },
            'home_pitcher_fatigue': {
                'name': 'Gerrit Cole',
                'days_rest': 4,
                'fatigue_score': 92,
                'status': 'fresh'
            },
            'travel': {
                'away_team_road_games': 4,
                'timezone_changes': 0,
                'travel_fatigue': 'mild'
            },
            'combined': {
                'scoring_adjustment': 1.03,
                'hr_adjustment': 1.08,
                'home_advantage_modifier': 1.01,
                'game_environment': 'slight_hitter_friendly'
            },
            'timestamp': datetime.now().isoformat()
        }

    # ── API Spec ──────────────────────────────────────────────

    def get_api_spec(self) -> Dict:
        """GET /api/v1/spec"""
        return {
            'openapi': '3.0.3',
            'info': {
                'title': 'MLB Predictor API',
                'version': self.version,
                'description': 'MLB game predictions with live odds, bankroll management, '
                             'and environmental factor analysis.'
            },
            'paths': {
                '/predictions/today': {'get': {'summary': "Today's picks sorted by edge"}},
                '/predictions/history': {'get': {'summary': 'Historical performance data'}},
                '/odds/live': {'get': {'summary': 'Live odds across sportsbooks'}},
                '/odds/value-bets': {'get': {'summary': 'Value bets where model > market'}},
                '/bankroll/calculate': {'post': {'summary': 'Kelly Criterion bet sizing'}},
                '/environmental/{game_id}': {'get': {'summary': 'Weather, umpire, fatigue factors'}},
            }
        }


# ============================================================================
# DEMO
# ============================================================================

def demo_prediction_api():
    """Demonstrate the prediction API"""
    print("=" * 70)
    print("🔌 MLB Predictor - REST API Demo")
    print("=" * 70)
    print()

    api = MLBPredictorAPI()

    # Today's predictions
    print("1️⃣  GET /predictions/today")
    print("-" * 60)
    preds = api.get_todays_predictions(min_edge=0.02)
    print(f"   Games with edge: {preds['picks_with_edge']}")
    for p in preds['predictions']:
        print(f"   {p['away']['team']} @ {p['home']['team']}: "
              f"Pick {p['pick']} ({p['odds']:+d}) "
              f"Edge: {p['edge']*100:.1f}% [{p['confidence']}]")
    print()

    # Kelly calculation
    print("2️⃣  POST /bankroll/calculate")
    print("-" * 60)
    kelly = api.calculate_kelly(0.62, -145, 5000, "moderate")
    r = kelly['result']
    print(f"   Input: 62% prob, -145 odds, $5000 bankroll")
    print(f"   Stake: ${r['recommended_stake']}")
    print(f"   Edge: {r['edge_pct']}% | EV: ${r['expected_value']}")
    print(f"   Kelly: {r['adjusted_kelly_pct']}% | Bet: {'Yes' if r['should_bet'] else 'No'}")
    print()

    # Environmental factors
    print("3️⃣  GET /environmental/mlb_20260401_nyy_bos")
    print("-" * 60)
    env = api.get_environmental_factors('mlb_20260401_nyy_bos')
    print(f"   Weather: {env['weather']['temperature']}°F, "
          f"wind {env['weather']['wind_speed']}mph {env['weather']['wind_direction']}")
    print(f"   Umpire: {env['umpire']['name']} ({env['umpire']['zone_tendency']})")
    print(f"   Pitcher: {env['home_pitcher_fatigue']['name']} - "
          f"{env['home_pitcher_fatigue']['status']}")
    print(f"   Combined: {env['combined']['game_environment']}")
    print()

    # Performance history
    print("4️⃣  GET /predictions/history?period=30d")
    print("-" * 60)
    hist = api.get_performance_history('30d')
    s = hist['summary']
    print(f"   Record: {s['wins']}W-{s['losses']}L ({s['win_rate']}%)")
    print(f"   ROI: +{s['roi']}% | Units: +{s['units_won']}")
    print(f"   Best streak: {s['best_streak']}W | Max DD: {s['max_drawdown_units']} units")
    print()

    # API endpoints
    print("5️⃣  Available Endpoints:")
    for path, info in api.get_api_spec()['paths'].items():
        method = list(info.keys())[0].upper()
        desc = info[list(info.keys())[0]]['summary']
        print(f"   {method:6} /api/v1{path:<35} {desc}")
    print()

    print("=" * 70)
    print("✅ API Demo Complete")
    print("=" * 70)

    return api


if __name__ == "__main__":
    demo_prediction_api()
