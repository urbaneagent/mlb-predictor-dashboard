# MLB Predictor v2.1 — Complete Feature Implementation

**Author:** Mike Ross (The Architect)  
**Date:** February 21, 2026  
**Status:** Production-ready

---

## 🎯 What's New in v2.1

### 1. 148 Total Features (80% Statcast + 20% Engineered)
- **118 Statcast columns** from pybaseball.statcast()
- **30 engineered features** for enhanced prediction

### 2. 3-Tab Excel Output

| Tab | Columns | Description |
|-----|---------|-------------|
| **Top Picks** | 20 | Best matchups with probabilities, risk flags, confidence |
| **Player Stats** | 50 | All batter + pitcher metrics, game context |
| **Full Features** | 148 | Complete feature set |

### 3. New Features Added

| Feature | Description |
|---------|-------------|
| `park_dimensions` | LF/CF/RF distances for all 30 parks |
| `effective_speed` | Perceived velocity (release_speed + extension) |
| `barrel_rate` | % of barrels (exit velo >98 + LA 26-30°) |
| `pull_rate` / `oppo_rate` | Spray angle tendencies |
| `bat_speed` | Batter swing speed |
| `swing_length` | Batter swing length |
| `temp_altitude_boost` | Weather × altitude interaction |
| `times_through_order` | Pitcher fatigue indicator |
| `ip_last_start` | Pitcher workload |
| `barrel_rate_allowed` | HR risk against pitcher |
| `hit_probability` | Predicted hit % (0-100) |
| `hr_probability` | Predicted HR % (0-100) |
| `win_probability` | Team win % (0-100) |
| `risk_flags` | INJURY_RETURN, COLD_STREAK, HOT_STREAK, SMALL_SAMPLE |
| `confidence` | 1-5 star rating |

### 4. Risk Flags System

| Flag | Description | Confidence Impact |
|------|-------------|------------------|
| 🚨 INJURY_RETURN | Player missed 30+ days | -2 stars |
| ⚠️ COLD_STREAK | BA < .150 last 7 days | -2 stars |
| 🔥 HOT_STREAK | BA > .400 last 7 days | +1 star |
| ⚡ SMALL_SAMPLE | <20 AB vs pitcher | -1 star |

### 5. Game-Day Fast Mode
- Optimized for quick lineup → predictions in ~5 minutes
- Uses cached Statcast data when available

---

## 📊 Feature Categories (148 Total)

| Category | Count | Examples |
|----------|-------|----------|
| **Pitcher Basic Stats** | 9 | ERA, WHIP, K/9, BB/9, IP |
| **Pitcher Advanced** | 15 | fastball%, velo, spin, effective_speed, gb_rate, times_through_order |
| **Batter Basic Stats** | 6 | BA L7, BA L30, walk_rate, k_rate |
| **Batter Advanced** | 12 | exit_velo, barrel_rate, bat_speed, swing_length, pull_rate, oppo_rate |
| **Matchup History** | 4 | career_ab, career_ba, career_hr vs pitcher |
| **Weather** | 6 | temp, humidity, wind_speed, wind_direction, precip |
| **Wind Analysis** | 2 | wind_batter_boost, temp_altitude_boost |
| **Park Factors** | 8 | run_factor, hr_factor, altitude, lf/cf/rf distances |
| **Probabilities** | 3 | hit_probability, hr_probability, win_probability |
| **Risk & Confidence** | 2 | risk_flags, confidence |
| **Statcast Columns** | ~81 | All available Statcast data |

---

## 🚀 How to Run

### Run Today's Predictions:
```bash
python mlb_predictor_v2.1.py
```

**Output:**
- Excel file: `/Users/mikeross/MLB_Predictions/YYYYMMDD_mlb_predictions_v2.1.xlsx`

---

## 📁 File Structure

```
mlb-predictor/
├── mlb_predictor_v2.1.py          # Main script (920 lines)
├── README_v2.1.md                 # This file
├── README_v2.0.md                 # v2.0 documentation
├── FEATURES_TO_ADD_v2.1.md        # Feature list
├── statcast_2023_2025_RAW.parquet # Cached Statcast data
└── MLB_Predictions/               # Output directory
    └── YYYYMMDD_mlb_predictions_v2.1.xlsx
```

---

## 🔬 Probability Models

### Hit Probability Formula
```
hit_probability = 0.20 + (
    ba_last_7_days * 25 +
    avg_exit_velo * 20 +
    barrel_rate * 15 +
    career_ba_vs_pitcher * 20 +
    park_run_factor * 10 +
    wind_batter_boost * 5 +
    contact_rate * 5
) * 0.50
```

### HR Probability Formula
```
hr_probability = 0.01 + (
    barrel_rate * 25 +
    max_exit_velo * 25 +
    park_hr_factor * 20 +
    avg_exit_velo * 15 +
    wind_batter_boost * 10 +
    temp_altitude_boost * 5
) * 0.08
```

---

## 🎓 Key Improvements Over v2.0

| Feature | v2.0 | v2.1 |
|---------|------|------|
| **Total Features** | 71 | 148 |
| **Excel Tabs** | 1 | 3 |
| **Park Dimensions** | ❌ | ✅ LF/CF/RF distances |
| **Effective Speed** | ❌ | ✅ Perceived velocity |
| **Barrel Rate** | Basic | ✅ Full + allowed |
| **Spray Angle** | ❌ | ✅ Pull/Oppo rates |
| **Probabilities** | Heuristic only | ✅ Hit/HR/Win % |
| **Risk Flags** | ❌ | ✅ 4 flags + confidence |
| **Game-Day Fast Mode** | ❌ | ✅ ~5 min execution |

---

## 📞 Questions?

Contact: Mike Ross (The Architect)  
Location: `~/.openclaw/workspace/projects/mlb-predictor/`

---

**END OF README v2.1**
