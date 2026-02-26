# MLB Predictor - Feature Documentation

## Recent Additions (2026-02-22)

### 🆕 Batter vs Pitcher Head-to-Head Matchup Analysis

**File:** `batter_pitcher_matchups.py`

Analyzes historical performance in specific batter vs pitcher matchups using 549,410 plate appearances from 2023-2025 Statcast data.

**Features:**
- Direct H2H statistics (AVG, OPS, HR rate, K rate)
- Platoon split analysis (L vs R, R vs L)
- Recent form tracking (last 50 PA in matchup)
- Confidence scoring based on sample size (5-100+ PA)
- Matchup favorability score (0-100 scale)

**Real Data:**
- 910 unique batters
- 1,301 unique pitchers
- 2.1M+ pitch-level data points

**Usage:**
```python
from batter_pitcher_matchups import MatchupAnalyzer

analyzer = MatchupAnalyzer()
analyzer.load_data()

# Get H2H performance
h2h = analyzer.get_h2h_matchup(batter_id=656941, pitcher_id=500779)
# Returns: {'avg': 0.292, 'ops': 0.792, 'hr_rate': 0.042, ...}

# Get platoon splits
platoon = analyzer.get_platoon_stats(batter_id=656941, pitcher_handedness='R')
# Returns: {'avg': 0.178, 'slg': 0.419, 'pa': 1399, ...}

# Get matchup score (0-100, higher = batter favored)
score = analyzer.get_matchup_score(batter_id=656941, pitcher_id=500779)
# Returns: 66 (batter favored)
```

**Integration:**
Use in pre-game analysis to adjust prediction confidence based on historical matchup performance.

---

### 🆕 Day/Night Performance Splits

**File:** `day_night_splits.py`

Analyzes performance differences between day games (before 5 PM) and night games (5 PM+) using 2.1M+ pitches.

**Features:**
- Day vs night performance comparison for batters and pitchers
- Split strength classification (strong/moderate/neutral)
- Matchup adjustment calculator (-10 to +10 scale)
- Confidence scoring based on sample sizes

**Game Time Classification:**
- **Day game:** Starts before 5 PM local time
- **Night game:** Starts at 5 PM or later

**Usage:**
```python
from day_night_splits import DayNightAnalyzer

analyzer = DayNightAnalyzer()
analyzer.load_data()

# Get batter's day/night splits
batter_splits = analyzer.get_batter_splits(batter_id=656941)
# Returns: {
#   'day': {'pa': 450, 'avg': 0.245, 'ops': 0.720},
#   'night': {'pa': 820, 'avg': 0.268, 'ops': 0.785},
#   'differential': {'preference': 'night', 'strength': 'moderate'}
# }

# Get pitcher's day/night splits
pitcher_splits = analyzer.get_pitcher_splits(pitcher_id=500779)
# Returns: {
#   'day': {'bf': 380, 'avg_against': 0.235},
#   'night': {'bf': 710, 'avg_against': 0.252},
#   'differential': {'preference': 'day', 'strength': 'moderate'}
# }

# Get matchup adjustment for specific game time
adjustment = analyzer.get_matchup_adjustment(
    batter_id=656941,
    pitcher_id=500779,
    game_time='night'
)
# Returns: {
#   'adjustment': +3,  # Favors batter
#   'explanation': 'Batter moderately prefers night games | Pitcher slightly worse in night games',
#   'batter_pref': 'night (moderate)',
#   'pitcher_pref': 'day (moderate)'
# }
```

**Integration:**
Apply adjustment to prediction score based on game start time. For example:
- Night game, batter prefers night (+3 to batter score)
- Day game, pitcher prefers day (-3 to batter score)

---

## Data Sources

All features use the same Statcast parquet file:
- **File:** `statcast_2023_2025_RAW.parquet`
- **Size:** 2,141,060 pitches
- **Years:** 2023-2025 seasons
- **Source:** pybaseball/MLB Statcast

---

## Next Steps

1. **Integrate into main predictor:**
   - Add H2H matchup score to feature set
   - Apply day/night adjustments based on game time
   - Weight matchup confidence in final prediction

2. **Model training:**
   - Include matchup features in ML model
   - Test impact on prediction accuracy
   - Tune weighting for matchup vs other features

3. **Real-time updates:**
   - Fetch today's matchups
   - Calculate scores for all games
   - Display in prediction output

---

## Performance

- **Matchup analysis:** ~2 seconds to load 2.1M pitches, instant lookups
- **Day/night analysis:** ~2 seconds to load and classify all games
- **Memory usage:** ~500MB for full dataset in memory

---

**Last Updated:** 2026-02-22  
**Author:** Mike Ross (The Architect)
