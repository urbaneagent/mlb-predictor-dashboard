# MLB Predictor v2.1 - Risk Flags

## Overview
Risk flags identify players with higher prediction uncertainty due to injuries, role changes, streaks, or limited data.

**Purpose:** Help users avoid unpredictable bets and identify high-confidence picks.

---

## Flag Types

### 🚨 **INJURY_RETURN** (Critical!)
**Trigger:** Player missed 30+ consecutive days

**Logic:**
```python
days_since_last_game = (today - last_game_date).days

if days_since_last_game >= 30:
    risk_flag = '🚨 INJURY_RETURN'
    confidence_penalty = -15%
    note = f"Missed {days_since_last_game} days (last game: {last_game_date})"
```

**Why it matters:**
- Players coming off Injured List (IL) = rust, timing off, limited conditioning
- First 5-10 games back = unpredictable (some bounce back, some struggle)
- MLB teams often limit PA/innings for players returning from injury

**Examples:**
- Mike Trout misses 6 weeks (hamstring) → First game back = 🚨 INJURY_RETURN
- Gerrit Cole misses 45 days (elbow) → First 3 starts = 🚨 INJURY_RETURN

**User action:** Consider avoiding these players for first week back.

---

### 🔄 **ROLE_CHANGE**
**Trigger:** Player's role changed from last season

**Logic:**
```python
# Pitcher role change:
if pitcher_role_2025 != pitcher_role_2024:
    risk_flag = '🔄 ROLE_CHANGE'
    note = f"{pitcher_role_2024} → {pitcher_role_2025}"

# Batter role change:
if batter_lineup_position_2025 != batter_lineup_position_2024:
    risk_flag = '🔄 ROLE_CHANGE'
    note = f"Moved from {batter_lineup_position_2024} to {batter_lineup_position_2025}"
```

**Examples:**
- Starter → Reliever (shorter outings, different pitch mix)
- Reliever → Closer (high leverage, new pressure)
- #7 hitter → #3 hitter (more PA, better pitches)
- Bench → Everyday starter (more volume, less rest)

**Why it matters:**
- Role changes = adjustment period (mechanics, approach, routine)
- Historical stats may not reflect new role

**User action:** Monitor for 2-3 weeks, adjust expectations.

---

### 🆕 **ROOKIE**
**Trigger:** Player has fewer than 50 career MLB games

**Logic:**
```python
if career_games < 50:
    risk_flag = '🆕 ROOKIE'
    confidence_penalty = -10%
    note = f"Only {career_games} MLB games (limited data)"
```

**Why it matters:**
- Small sample size = less reliable predictions
- MLB pitchers adjust after seeing rookie 2-3 times ("book" on the player)
- "Sophomore slump" is real (pitchers/batters adjust, rookie struggles)

**Examples:**
- Rookie hits .350 in first month → Pitchers learn weaknesses → Drops to .250
- Rookie pitcher dominates first 5 starts → Hitters adjust → ERA doubles

**User action:** Use caution, especially after first month.

---

### ⚠️ **COLD_STREAK**
**Trigger:** Batter hitting under .150 in last 7 days OR pitcher ERA >6.00 in last 3 starts

**Logic:**
```python
# Batter cold streak:
if ba_last_7_days < 0.150:
    risk_flag = '⚠️ COLD_STREAK'
    note = f"Hitting {ba_last_7_days:.3f} in last 7 days ({strikeouts_last_7} K)"

# Pitcher cold streak:
if era_last_3_starts > 6.00:
    risk_flag = '⚠️ COLD_STREAK'
    note = f"ERA {era_last_3_starts:.2f} in last 3 starts ({hr_allowed_last_3} HR)"
```

**Why it matters:**
- Players pressing, chasing bad pitches, confidence shaken
- May be playing through minor injury (not on IL yet)
- Mechanics issues (timing, release point)

**User action:** Avoid until streak breaks (2-3 good games).

---

### 🔥 **HOT_STREAK** (Positive!)
**Trigger:** Batter hitting over .400 in last 7 days OR pitcher ERA <2.00 in last 3 starts

**Logic:**
```python
# Batter hot streak:
if ba_last_7_days > 0.400:
    risk_flag = '🔥 HOT_STREAK'
    confidence_boost = +5%
    note = f"Hitting {ba_last_7_days:.3f} in last 7 days ({hr_last_7} HR, {xbh_last_7} XBH)"

# Pitcher hot streak:
if era_last_3_starts < 2.00:
    risk_flag = '🔥 HOT_STREAK'
    confidence_boost = +5%
    note = f"ERA {era_last_3_starts:.2f} in last 3 starts ({k_last_3} K, {bb_last_3} BB)"
```

**Why it matters:**
- Locked in, seeing ball well, high confidence
- Pitchers avoiding (intentional walks) or respecting (better pitches)
- "Hot hand" effect is real in baseball (confidence → better approach)

**User action:** Target these players! Higher upside.

---

### ⚡ **SMALL_SAMPLE**
**Trigger:** Batter has fewer than 20 AB vs this specific pitcher

**Logic:**
```python
if career_ab_vs_pitcher < 20:
    risk_flag = '⚡ SMALL_SAMPLE'
    note = f"Only {career_ab_vs_pitcher} AB vs {pitcher_name} (BA: {career_ba_vs_pitcher:.3f})"
```

**Why it matters:**
- 5-for-5 vs pitcher (1.000 BA!) but only 5 AB = not statistically significant
- 0-for-10 vs pitcher (.000 BA) but small sample = may just be variance
- Need 20+ AB for reliable career matchup stats

**User action:** Ignore career BA vs pitcher if sample is small. Use overall BA instead.

---

### 🌟 **PLATOON_ADVANTAGE** (Positive!)
**Trigger:** Batter has strong platoon split advantage

**Logic:**
```python
# Left-handed batter vs right-handed pitcher:
if batter_stand == 'L' and pitcher_throws == 'R':
    platoon_advantage = batter_ba_vs_rhp - batter_ba_vs_lhp
    
    if platoon_advantage > 0.050:  # Hits 50+ points better vs RHP
        risk_flag = '🌟 PLATOON_ADVANTAGE'
        note = f"Hits {batter_ba_vs_rhp:.3f} vs RHP (vs {batter_ba_vs_lhp:.3f} vs LHP)"
```

**Why it matters:**
- Some batters DOMINATE opposite-hand pitchers (+50-80 points BA)
- Huge edge for DFS/betting

**User action:** Prioritize these matchups!

---

### 🏟️ **PARK_CRUSHER** (Positive!)
**Trigger:** Batter has strong career stats at this specific park

**Logic:**
```python
if career_ba_at_park > (career_ba_overall + 0.050):
    risk_flag = '🏟️ PARK_CRUSHER'
    note = f"Hits {career_ba_at_park:.3f} at {park_name} (vs {career_ba_overall:.3f} overall)"
```

**Examples:**
- David Ortiz at Yankee Stadium (.350+ career, loved short RF porch)
- Left-handed power hitters at Fenway (Green Monster LF)

**User action:** Huge edge! Target these players when playing at their favorite park.

---

## Flag Priority (Display Order)

**Critical (Red):**
1. 🚨 INJURY_RETURN

**Caution (Yellow):**
2. 🔄 ROLE_CHANGE
3. 🆕 ROOKIE
4. ⚠️ COLD_STREAK
5. ⚡ SMALL_SAMPLE

**Positive (Green):**
6. 🔥 HOT_STREAK
7. 🌟 PLATOON_ADVANTAGE
8. 🏟️ PARK_CRUSHER

---

## Confidence Scoring

**Base confidence:** 5 stars (★★★★★)

**Penalties:**
- 🚨 INJURY_RETURN: -2 stars (★★★☆☆)
- 🔄 ROLE_CHANGE: -1 star (★★★★☆)
- 🆕 ROOKIE: -1 star (★★★★☆)
- ⚠️ COLD_STREAK: -2 stars (★★★☆☆)
- ⚡ SMALL_SAMPLE: -1 star (★★★★☆)

**Boosts:**
- 🔥 HOT_STREAK: +1 star (★★★★★★ — max 5 stars displayed)
- 🌟 PLATOON_ADVANTAGE: +0.5 stars
- 🏟️ PARK_CRUSHER: +0.5 stars

**Multiple flags:** Cumulative (e.g., INJURY_RETURN + COLD_STREAK = 1 star, avoid!)

---

## Tab 1 Display Format

```
| Rank | Batter | Pitcher | Hit% | HR% | Risk Flags | Confidence | Notes |
|------|--------|---------|------|-----|------------|------------|-------|
| 1 | Aaron Judge | Chris Sale | 68% | 12% | 🔥 🌟 | ★★★★★ | Hot streak + platoon advantage |
| 2 | Juan Soto | Shane Bieber | 65% | 10% | - | ★★★★★ | Clean, high confidence |
| 3 | Mike Trout | Lucas Giolito | 55% | 9% | 🚨 | ★★☆☆☆ | First game back from IL |
| 4 | Rafael Devers | Gerrit Cole | 58% | 8% | ⚠️ | ★★★☆☆ | 3-for-20 last 7 days |
```

---

## Implementation Notes

**Data sources:**
- `days_since_last_game`: Calculate from Statcast game logs
- `career_games`: Count unique `game_pk` for player ID
- `ba_last_7_days`: Aggregate last 7 days of Statcast data
- `career_ba_vs_pitcher`: Filter Statcast by batter + pitcher IDs
- `career_ba_at_park`: Filter Statcast by batter + venue

**Storage:**
- Pre-calculate flags during feature engineering
- Store in `predictions_with_flags.xlsx` (Tab 1)
- Include flag details in Tab 2 ("Player Stats")

**Future enhancements (v3.0):**
- 🔬 **LINEUP_CHANGE** (batting order moved significantly)
- 🏥 **MINOR_INJURY** (questionable status, not on IL)
- 📉 **DECLINING_VELO** (pitcher fastball velo down 2+ mph from career avg)
- 🎓 **PROSPECT_HYPE** (top prospect, unproven but high upside)

---

**END OF RISK FLAGS DOCUMENTATION**
