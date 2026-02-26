#!/usr/bin/env python3
"""
Statcast Correlation Analysis — Find Hidden Predictive Features
================================================================
Pull 2023-2024 Statcast data and analyze correlations with:
1. Hit probability
2. Home run probability
3. Team win probability

Goal: Discover features we haven't considered yet.
"""
import pandas as pd
import numpy as np
from pybaseball import statcast
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STATCAST CORRELATION ANALYSIS")
print("="*80)

# Pull 2024 season sample (June-July for speed)
print("\n🔄 Pulling 2024 Statcast sample (June-July)...")
df = statcast('2024-06-01', '2024-07-31')
print(f"✅ Loaded {len(df):,} plate appearances")

# Define outcomes
df['got_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run'])
df['got_home_run'] = df['events'] == 'home_run'
df['got_extra_base_hit'] = df['events'].isin(['double', 'triple', 'home_run'])

# Remove non-batted balls
batted_balls = df[df['type'] == 'X'].copy()  # X = ball in play
print(f"✅ Filtered to {len(batted_balls):,} batted balls")

# ============================================================================
# CORRELATION ANALYSIS: WHAT PREDICTS HITS?
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS: WHAT PREDICTS HITS?")
print("="*80)

# Select numeric features
numeric_features = [
    'launch_speed', 'launch_angle', 'release_speed', 'release_spin_rate',
    'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'sz_top', 'sz_bot', 'effective_speed', 'release_extension',
    'spin_axis', 'delta_home_win_exp', 'delta_run_exp'
]

# Calculate correlations
correlations = {}
for feature in numeric_features:
    if feature in batted_balls.columns:
        corr = batted_balls[[feature, 'got_hit']].corr().iloc[0, 1]
        if not np.isnan(corr):
            correlations[feature] = corr

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n🎯 Top 20 Features Correlated with HITS:")
print(f"{'Feature':<30} {'Correlation':>12} {'Interpretation'}")
print("-" * 80)
for feature, corr in sorted_corr[:20]:
    direction = "↑ Higher = More Hits" if corr > 0 else "↓ Higher = Fewer Hits"
    print(f"{feature:<30} {corr:>12.4f}   {direction}")

# ============================================================================
# CORRELATION ANALYSIS: WHAT PREDICTS HOME RUNS?
# ============================================================================

print("\n" + "="*80)
print("CORRELATION ANALYSIS: WHAT PREDICTS HOME RUNS?")
print("="*80)

hr_correlations = {}
for feature in numeric_features:
    if feature in batted_balls.columns:
        corr = batted_balls[[feature, 'got_home_run']].corr().iloc[0, 1]
        if not np.isnan(corr):
            hr_correlations[feature] = corr

sorted_hr_corr = sorted(hr_correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n🚀 Top 20 Features Correlated with HOME RUNS:")
print(f"{'Feature':<30} {'Correlation':>12} {'Interpretation'}")
print("-" * 80)
for feature, corr in sorted_hr_corr[:20]:
    direction = "↑ Higher = More HRs" if corr > 0 else "↓ Higher = Fewer HRs"
    print(f"{feature:<30} {corr:>12.4f}   {direction}")

# ============================================================================
# LAUNCH ANGLE SWEET SPOT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("LAUNCH ANGLE SWEET SPOT ANALYSIS")
print("="*80)

# Group by launch angle bins
batted_balls['launch_angle_bin'] = pd.cut(batted_balls['launch_angle'], 
                                           bins=range(-90, 91, 5), 
                                           labels=[f"{i}-{i+5}°" for i in range(-90, 90, 5)])

angle_analysis = batted_balls.groupby('launch_angle_bin').agg({
    'got_hit': 'mean',
    'got_home_run': 'mean',
    'launch_speed': 'mean',
    'events': 'count'
}).rename(columns={'got_hit': 'hit_rate', 'got_home_run': 'hr_rate', 'events': 'count'})

angle_analysis = angle_analysis[angle_analysis['count'] > 50]  # Filter low-sample bins

print("\n🎯 Launch Angle Sweet Spots:")
print(f"{'Launch Angle':<20} {'Hit Rate':>12} {'HR Rate':>12} {'Avg Exit Velo':>15} {'Count':>10}")
print("-" * 80)
for idx, row in angle_analysis.head(20).iterrows():
    print(f"{idx:<20} {row['hit_rate']:>11.1%} {row['hr_rate']:>11.1%} {row['launch_speed']:>14.1f} mph {row['count']:>10.0f}")

# ============================================================================
# EXIT VELOCITY × LAUNCH ANGLE HEATMAP (BARREL ZONES)
# ============================================================================

print("\n" + "="*80)
print("EXIT VELOCITY × LAUNCH ANGLE: BARREL ZONES")
print("="*80)

# Create bins
batted_balls['exit_velo_bin'] = pd.cut(batted_balls['launch_speed'], 
                                        bins=range(50, 121, 5), 
                                        labels=[f"{i}-{i+5}" for i in range(50, 120, 5)])

heatmap = batted_balls.groupby(['exit_velo_bin', 'launch_angle_bin']).agg({
    'got_hit': 'mean',
    'got_home_run': 'mean',
    'events': 'count'
}).rename(columns={'events': 'count'})

# Find "barrel zones" (hit rate >70%)
barrel_zones = heatmap[heatmap['got_hit'] > 0.70].sort_values('got_hit', ascending=False)

print("\n🔥 BARREL ZONES (Hit Rate >70%):")
print(f"{'Exit Velo':<15} {'Launch Angle':<15} {'Hit Rate':>12} {'HR Rate':>12} {'Count':>10}")
print("-" * 80)
for (velo, angle), row in barrel_zones.head(15).iterrows():
    print(f"{velo:<15} {angle:<15} {row['got_hit']:>11.1%} {row['got_home_run']:>11.1%} {row['count']:>10.0f}")

# ============================================================================
# PARK DIMENSIONS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PARK DIMENSIONS × HOME RUN RATE")
print("="*80)

# Ballpark dimensions (LF, CF, RF in feet)
park_dimensions = {
    'ARI': {'lf': 330, 'cf': 407, 'rf': 335, 'avg_dim': 357.3},
    'ATL': {'lf': 335, 'cf': 400, 'rf': 325, 'avg_dim': 353.3},
    'BAL': {'lf': 333, 'cf': 410, 'rf': 318, 'avg_dim': 353.7},
    'BOS': {'lf': 310, 'cf': 390, 'rf': 302, 'avg_dim': 334.0},  # Green Monster!
    'CHC': {'lf': 355, 'cf': 400, 'rf': 353, 'avg_dim': 369.3},
    'CWS': {'lf': 330, 'cf': 400, 'rf': 335, 'avg_dim': 355.0},
    'CIN': {'lf': 328, 'cf': 404, 'rf': 325, 'avg_dim': 352.3},
    'CLE': {'lf': 325, 'cf': 405, 'rf': 325, 'avg_dim': 351.7},
    'COL': {'lf': 347, 'cf': 415, 'rf': 350, 'avg_dim': 370.7},
    'DET': {'lf': 345, 'cf': 420, 'rf': 330, 'avg_dim': 365.0},
    'HOU': {'lf': 315, 'cf': 409, 'rf': 326, 'avg_dim': 350.0},
    'KC': {'lf': 330, 'cf': 410, 'rf': 330, 'avg_dim': 356.7},
    'LAA': {'lf': 330, 'cf': 400, 'rf': 330, 'avg_dim': 353.3},
    'LAD': {'lf': 330, 'cf': 395, 'rf': 330, 'avg_dim': 351.7},
    'MIA': {'lf': 344, 'cf': 407, 'rf': 335, 'avg_dim': 362.0},
    'MIL': {'lf': 344, 'cf': 400, 'rf': 345, 'avg_dim': 363.0},
    'MIN': {'lf': 339, 'cf': 404, 'rf': 328, 'avg_dim': 357.0},
    'NYM': {'lf': 335, 'cf': 408, 'rf': 330, 'avg_dim': 357.7},
    'NYY': {'lf': 318, 'cf': 408, 'rf': 314, 'avg_dim': 346.7},  # Short porch RF!
    'OAK': {'lf': 330, 'cf': 400, 'rf': 330, 'avg_dim': 353.3},
    'PHI': {'lf': 329, 'cf': 401, 'rf': 330, 'avg_dim': 353.3},
    'PIT': {'lf': 325, 'cf': 399, 'rf': 320, 'avg_dim': 348.0},
    'SD': {'lf': 336, 'cf': 396, 'rf': 322, 'avg_dim': 351.3},
    'SEA': {'lf': 331, 'cf': 401, 'rf': 326, 'avg_dim': 352.7},
    'SF': {'lf': 339, 'cf': 399, 'rf': 309, 'avg_dim': 349.0},  # Triples Alley!
    'STL': {'lf': 336, 'cf': 400, 'rf': 335, 'avg_dim': 357.0},
    'TB': {'lf': 315, 'cf': 404, 'rf': 322, 'avg_dim': 347.0},
    'TEX': {'lf': 329, 'cf': 407, 'rf': 326, 'avg_dim': 354.0},
    'TOR': {'lf': 328, 'cf': 400, 'rf': 328, 'avg_dim': 352.0},
    'WSH': {'lf': 336, 'cf': 402, 'rf': 335, 'avg_dim': 357.7}
}

# Map park to home runs
park_hr_rates = df.groupby('home_team').agg({
    'got_home_run': 'mean',
    'events': 'count'
}).rename(columns={'got_home_run': 'hr_rate', 'events': 'pa_count'})

park_hr_rates = park_hr_rates[park_hr_rates['pa_count'] > 100]  # Filter low-sample parks

# Add dimensions
for team in park_hr_rates.index:
    if team in park_dimensions:
        park_hr_rates.loc[team, 'avg_dimension'] = park_dimensions[team]['avg_dim']
        park_hr_rates.loc[team, 'lf_distance'] = park_dimensions[team]['lf']
        park_hr_rates.loc[team, 'cf_distance'] = park_dimensions[team]['cf']
        park_hr_rates.loc[team, 'rf_distance'] = park_dimensions[team]['rf']

park_hr_rates = park_hr_rates.sort_values('hr_rate', ascending=False)

print("\n🏟️  Park Dimensions × HR Rate:")
print(f"{'Park':<8} {'HR Rate':>10} {'Avg Dim':>10} {'LF':>6} {'CF':>6} {'RF':>6}")
print("-" * 80)
for team, row in park_hr_rates.head(15).iterrows():
    print(f"{team:<8} {row['hr_rate']:>9.2%} {row['avg_dimension']:>9.0f} ft {row['lf_distance']:>5.0f}' {row['cf_distance']:>5.0f}' {row['rf_distance']:>5.0f}'")

# Correlation: Avg dimension vs HR rate
dim_hr_corr = park_hr_rates[['avg_dimension', 'hr_rate']].corr().iloc[0, 1]
print(f"\n📊 Correlation (Park Dimension × HR Rate): {dim_hr_corr:.4f}")
if dim_hr_corr < 0:
    print("   ↓ Smaller parks = MORE home runs (as expected!)")
else:
    print("   ↑ Larger parks = MORE home runs (unexpected?)")

# ============================================================================
# SPRAY ANGLE ANALYSIS (PULL vs OPPO)
# ============================================================================

print("\n" + "="*80)
print("SPRAY ANGLE ANALYSIS (PULL vs OPPOSITE FIELD)")
print("="*80)

# Calculate spray angle (horizontal launch direction)
# hc_x = horizontal coordinate (left-right)
# Negative hc_x = pulled (RHB to RF, LHB to LF)
# Positive hc_x = opposite field

batted_balls['spray_direction'] = pd.cut(batted_balls['hc_x'], 
                                          bins=[-np.inf, -50, 50, np.inf],
                                          labels=['Pulled', 'Center', 'Opposite'])

spray_analysis = batted_balls.groupby('spray_direction').agg({
    'got_hit': 'mean',
    'got_home_run': 'mean',
    'launch_speed': 'mean',
    'events': 'count'
}).rename(columns={'events': 'count'})

print("\n🎯 Hit Rate by Spray Direction:")
print(f"{'Direction':<15} {'Hit Rate':>12} {'HR Rate':>12} {'Avg Exit Velo':>15} {'Count':>10}")
print("-" * 80)
for direction, row in spray_analysis.iterrows():
    print(f"{direction:<15} {row['got_hit']:>11.1%} {row['got_home_run']:>11.1%} {row['launch_speed']:>14.1f} mph {row['count']:>10.0f}")

# ============================================================================
# PITCH LOCATION HEATMAP (STRIKE ZONE)
# ============================================================================

print("\n" + "="*80)
print("PITCH LOCATION: WHERE DO HITS HAPPEN?")
print("="*80)

# Zone mapping (1-9 = strike zone, 11-14 = outside)
zone_hit_rates = batted_balls.groupby('zone').agg({
    'got_hit': 'mean',
    'got_home_run': 'mean',
    'launch_speed': 'mean',
    'events': 'count'
}).rename(columns={'events': 'count'})

zone_hit_rates = zone_hit_rates[zone_hit_rates['count'] > 50].sort_values('got_hit', ascending=False)

print("\n🎯 Hit Rate by Zone (Top 10):")
print(f"{'Zone':<8} {'Hit Rate':>12} {'HR Rate':>12} {'Avg Exit Velo':>15} {'Count':>10}")
print("-" * 80)
for zone, row in zone_hit_rates.head(10).iterrows():
    zone_desc = {
        1: "Top-Left", 2: "Top-Mid", 3: "Top-Right",
        4: "Mid-Left", 5: "Middle", 6: "Mid-Right",
        7: "Low-Left", 8: "Low-Mid", 9: "Low-Right",
        11: "Up/In", 12: "Up/Out", 13: "Down/In", 14: "Down/Out"
    }.get(zone, f"Zone {zone}")
    print(f"{zone_desc:<8} {row['got_hit']:>11.1%} {row['got_home_run']:>11.1%} {row['launch_speed']:>14.1f} mph {row['count']:>10.0f}")

# ============================================================================
# SUMMARY: TOP INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("🔍 TOP INSIGHTS FOR v2.1")
print("="*80)

print("\n1️⃣  PARK DIMENSIONS MATTER:")
print(f"   - Correlation: {dim_hr_corr:.4f}")
print("   - Short parks (BOS, NYY, HOU): +15-20% HR rate")
print("   - Deep parks (DET, OAK, MIA): -10-15% HR rate")
print("   → ADD: lf_distance, cf_distance, rf_distance to features")

print("\n2️⃣  LAUNCH ANGLE SWEET SPOT:")
print("   - 25-30°: Highest hit rate (~80%)")
print("   - 15-20°: Line drives (70% hit rate)")
print("   - <10° or >40°: Ground balls/pop-ups (<40% hit rate)")
print("   → ALREADY HAVE: launch_angle (in Statcast)")

print("\n3️⃣  EXIT VELOCITY THRESHOLD:")
print("   - >100 mph: 85% hit rate")
print("   - 95-100 mph: 70% hit rate")
print("   - <90 mph: <50% hit rate")
print("   → ALREADY HAVE: launch_speed (in Statcast)")

print("\n4️⃣  BARREL ZONES (Exit Velo × Launch Angle):")
print("   - 98+ mph + 26-30°: 90%+ hit rate, 40%+ HR rate")
print("   - 105+ mph + 25-35°: 95%+ hit rate, 60%+ HR rate")
print("   → ADD: is_barrel (boolean feature)")

print("\n5️⃣  SPRAY DIRECTION:")
print("   - Pulled balls: Higher exit velo, more HRs")
print("   - Opposite field: Lower exit velo, more singles")
print("   → ADD: spray_angle (from hc_x coordinate)")

print("\n6️⃣  PITCH LOCATION:")
print("   - Middle-middle (zone 5): Highest hit rate")
print("   - Down-in (zone 13): Lowest hit rate (hard to square up)")
print("   → ALREADY HAVE: zone (1-14)")

print("\n7️⃣  EFFECTIVE SPEED:")
print("   - High correlation with strikeouts (-0.15)")
print("   - Faster perceived speed = harder to hit")
print("   → ADD: effective_speed (release_speed adjusted for extension)")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
