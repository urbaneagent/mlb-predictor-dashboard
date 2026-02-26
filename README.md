# MLB Predictor ⚾
### The Best MLB Prediction System Ever Built

**Built by Mike Ross (The Architect) | February 2026**

---

## 🚀 What is MLB Predictor?

An ML-powered MLB prediction system with live odds tracking, bankroll management, environmental factor analysis, and a user-facing dashboard. Built on real Statcast data (2023-2025).

## ✨ Features

### 📊 Live Odds Tracker (`live_odds_tracker.py`)
- **The Odds API** integration (8 sportsbooks: DraftKings, FanDuel, BetMGM, PointsBet, Caesars, BetRivers, Unibet, Bovada)
- **Sportsbook comparison** — find best odds across books
- **Line movement tracker** — detect steam moves, significant shifts
- **Arbitrage detector** — find guaranteed profit when books disagree
- **Value bet finder** — model probability vs market probability
- **Odds math utilities** — American/Decimal conversion, Kelly, EV, implied probability
- **Historical odds caching** for backtesting

### 💰 Bankroll Manager (`bankroll_manager.py`)
- **Kelly Criterion calculator** — Full, Half, Quarter Kelly
- **4 risk profiles** — Conservative (2% max), Moderate (3%), Aggressive (5%), Degen (10%)
- **Bet sizing recommendations** with confidence ratings
- **Win/loss tracking** with ROI, streaks, and max drawdown
- **Drawdown protection** — auto-stop at 25% drawdown
- **Parlay builder** with Kelly-sized stakes
- **Performance by confidence** level analysis
- **CSV export** for all bet history

### 🌤️ Environmental Factors (`environmental_factors.py`)
- **Weather impact engine** — temperature, wind speed/direction, humidity, altitude
  - Wind out to CF = +3.5% HR probability per mph
  - Every 10°F above baseline = +2% HR probability
  - Coors Field altitude = +25% HR boost
- **Umpire database** — 8 MLB umpires with zone tendencies
  - Strike zone width, consistency score, run impact
  - O/U lean recommendations
- **Pitcher fatigue engine** — days rest, pitch count, season workload
  - Weighted composite score (rest 30%, last start 25%, season 25%, bullpen 20%)
  - Performance multiplier: fresh (+2%) to exhausted (-10%)
- **Travel fatigue** — road trip length, timezone changes, day-after-night

### 📊 User Dashboard (`user_dashboard.py`)
- **Today's top picks** — sorted by edge with full details
- **Historical performance** tracking (7d, 30d, all-time)
- **Daily report generator** — formatted text report with all picks
- **Chart data** — cumulative PnL, weekly ROI, daily picks
- **Alerts engine** — hot streaks, cold streaks, high-edge picks
- **CSV export** for all daily picks
- **Confidence breakdown** — high/medium/low performance tracking

### 🔌 Prediction API (`prediction_api.py`)
- **REST API** with 6 endpoint categories
- **Today's predictions** with model/market probability comparison
- **Performance history** with confidence and bet-type breakdown
- **Live odds** across sportsbooks
- **Value bets** where model > market
- **Kelly calculator** endpoint
- **Environmental factors** per game
- **OpenAPI specification** included

### 🧠 Core ML Model (`mlb_predictor_v5.0.py` / `v2.0_ML_READY.py`)
- **XGBoost models** trained on Statcast data (2023-2025)
- **270M+ Statcast datapoints** (2023-2025 raw parquet files)
- **Feature engineering**: batting stats, pitcher metrics, park factors
- **Batter-pitcher matchup analysis** (H2H scoring)
- **Day/night performance splits**
- **30 MLB ballparks** with park factors (run factor, HR factor, altitude, dimensions)

## 📁 Project Structure

```
mlb-predictor/
├── live_odds_tracker.py         # Odds API + arbitrage + line movements
├── bankroll_manager.py          # Kelly Criterion + risk management
├── environmental_factors.py     # Weather + umpire + fatigue engines
├── user_dashboard.py            # Dashboard data + daily reports
├── prediction_api.py            # REST API endpoints
├── mlb_predictor_v5.0.py        # Latest ML model
├── mlb_predictor_v2.0_ML_READY.py # XGBoost production model
├── batter_pitcher_matchups.py   # H2H matchup analysis
├── day_night_splits.py          # Performance splits
├── weather_integration.py       # Legacy weather module
├── fetch_statcast.py            # Statcast data fetcher
├── train_model.py               # Model training pipeline
├── statcast_2023.parquet        # 92MB Statcast data
├── statcast_2024.parquet        # 98MB Statcast data
├── statcast_2025.parquet        # 98MB Statcast data
├── statcast_2023_2025_RAW.parquet # 286MB combined raw data
└── README.md
```

## 🏗️ Architecture

```
[Statcast Data (2023-2025)] → [Feature Engineering] → [XGBoost Models]
              ↓                                              ↓
    [Park Factors (30 stadiums)]                    [Hit/HR Predictions]
              ↓                                              ↓
    [Matchup Analysis]                              [Win Probabilities]
              ↓                                              ↓
    [Day/Night Splits]          [The Odds API] → [Market Probabilities]
                                       ↓                     ↓
                              [Odds Comparison]      [Edge Calculation]
                                       ↓                     ↓
                              [Line Movements]        [Value Bets]
                                       ↓                     ↓
[Weather API] → [Environmental]   [Arbitrage]    [Kelly Criterion]
      ↓              ↓                                   ↓
[Umpire DB]    [Fatigue Engine]              [Bankroll Management]
      ↓              ↓                                   ↓
[Combined Adjustments]           →           [Dashboard + Reports]
                                                        ↓
                                                   [REST API]
```

## 📊 Data Sources

| Source | Data | Size |
|--------|------|------|
| **Statcast (Baseball Savant)** | Pitch-level data 2023-2025 | 286MB |
| **The Odds API** | Live sportsbook odds | Real-time |
| **CMS NADAC (park factors)** | Stadium dimensions, altitude | Static |
| **UmpScorecards** | Umpire strike zone data | 8 umpires |

## 🎯 Performance Targets

- **Win Rate**: 56-60% on moneyline picks
- **ROI**: 8-12% on 1-unit flat bets  
- **Edge Threshold**: Minimum 2% edge to recommend
- **Kelly Sizing**: Half-Kelly (moderate risk) as default
