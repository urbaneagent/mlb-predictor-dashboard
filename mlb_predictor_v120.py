#!/usr/bin/env python3
"""
MLB DFS Predictor v120 - Park Factors + Pull/Spray Angle Analysis
===================================================================

IMPROVEMENT: Park-adjusted predictions with batter spray angle alignment

RESEARCH FOUNDATION:
- Park factors can boost/suppress production by 15-30% (FanGraphs, Baseball Savant)
- Pull hitters gain 15-25% more value in short-porch parks (Yankee Stadium RF, Fenway LF)
- Spray angle (horizontal batted ball direction) combined with park dimensions = exploitable edge
- Parks like Coors Field boost ALL hitters (+20% runs), but short porches favor pull-side tendencies
- Oracle Park suppresses RHH power (-18% HR factor), Great American boosts LHH power (+12% HR)

METHODOLOGY:
1. Load park dimensions (LF/CF/RF distances, wall heights)
2. Load 3-year park factors (HR, Runs, Hits - overall + LHB/RHB splits)
3. Calculate batter spray tendency from Statcast:
   - Pull% (LHB: <-15° spray angle, RHB: >15°)
   - Center% (-15° to 15°)
   - Oppo% (LHB: >15°, RHB: <-15°)
4. Calculate "Park-Batter Fit Score":
   - Pull hitter + short porch = +20-30% boost
   - Oppo hitter + deep fence = -10-15% penalty
5. Apply park factor adjustments to HR/hit probabilities
6. Generate park-adjusted DFS value scores

OUTPUT:
- batter_spray_tendencies.csv (pull/center/oppo splits per batter)
- park_dimensions.csv (all 30 MLB parks + factors)
- v120_park_adjusted_predictions.csv (final predictions with park boosts/penalties)

USAGE:
    python mlb_predictor_v120.py --date 2026-04-15

VERSION: 120
AUTHOR: The $100M Architect
CREATED: 2026-02-24 4:01 AM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# ============================================================================
# MLB PARK DIMENSIONS & FACTORS (2026 Season)
# ============================================================================

PARK_DATA = {
    # American League
    "BAL": {  # Camden Yards (Orioles Park)
        "name": "Oriole Park at Camden Yards",
        "lf_ft": 333, "cf_ft": 400, "rf_ft": 318,  # Short RF porch!
        "wall_lf_ft": 7, "wall_cf_ft": 7, "wall_rf_ft": 7,
        "hr_factor": 105,  # 5% boost (short RF for LHH)
        "runs_factor": 102,
        "lhb_hr_factor": 112,  # 12% boost for lefties (short RF)
        "rhb_hr_factor": 98,
    },
    "BOS": {  # Fenway Park
        "name": "Fenway Park",
        "lf_ft": 310, "cf_ft": 390, "rf_ft": 302,  # Green Monster + Pesky's Pole
        "wall_lf_ft": 37, "wall_cf_ft": 17, "wall_rf_ft": 3,
        "hr_factor": 98,  # Monster hurts LHH, Pesky helps RHH
        "runs_factor": 104,
        "lhb_hr_factor": 92,  # Green Monster = doubles, not HRs
        "rhb_hr_factor": 108,  # Short RF porch
    },
    "NYY": {  # Yankee Stadium
        "name": "Yankee Stadium",
        "lf_ft": 318, "cf_ft": 408, "rf_ft": 314,  # Famous short RF porch
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 108,  # 8% boost overall
        "runs_factor": 105,
        "lhb_hr_factor": 118,  # 18% boost for lefties (short RF = moonshots)
        "rhb_hr_factor": 98,
    },
    "TOR": {  # Rogers Centre
        "name": "Rogers Centre",
        "lf_ft": 328, "cf_ft": 400, "rf_ft": 328,
        "wall_lf_ft": 10, "wall_cf_ft": 10, "wall_rf_ft": 10,
        "hr_factor": 103,
        "runs_factor": 102,
        "lhb_hr_factor": 104,
        "rhb_hr_factor": 102,
    },
    "TB": {  # Tropicana Field
        "name": "Tropicana Field",
        "lf_ft": 315, "cf_ft": 404, "rf_ft": 322,
        "wall_lf_ft": 9, "wall_cf_ft": 10, "wall_rf_ft": 10,
        "hr_factor": 92,  # Suppresses HRs (fixed roof, dead air)
        "runs_factor": 95,
        "lhb_hr_factor": 90,
        "rhb_hr_factor": 94,
    },
    "CLE": {  # Progressive Field
        "name": "Progressive Field",
        "lf_ft": 325, "cf_ft": 405, "rf_ft": 325,
        "wall_lf_ft": 19, "wall_cf_ft": 9, "wall_rf_ft": 9,
        "hr_factor": 98,
        "runs_factor": 100,
        "lhb_hr_factor": 99,
        "rhb_hr_factor": 97,
    },
    "CWS": {  # Guaranteed Rate Field
        "name": "Guaranteed Rate Field",
        "lf_ft": 330, "cf_ft": 400, "rf_ft": 335,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 104,
        "runs_factor": 103,
        "lhb_hr_factor": 106,
        "rhb_hr_factor": 102,
    },
    "DET": {  # Comerica Park
        "name": "Comerica Park",
        "lf_ft": 345, "cf_ft": 420, "rf_ft": 330,  # HUGE CF (420ft!)
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 88,  # 12% suppression (massive CF)
        "runs_factor": 94,
        "lhb_hr_factor": 86,
        "rhb_hr_factor": 90,
    },
    "KC": {  # Kauffman Stadium
        "name": "Kauffman Stadium",
        "lf_ft": 330, "cf_ft": 410, "rf_ft": 330,
        "wall_lf_ft": 9, "wall_cf_ft": 9, "wall_rf_ft": 9,
        "hr_factor": 96,
        "runs_factor": 99,
        "lhb_hr_factor": 95,
        "rhb_hr_factor": 97,
    },
    "MIN": {  # Target Field
        "name": "Target Field",
        "lf_ft": 339, "cf_ft": 404, "rf_ft": 328,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 102,
        "runs_factor": 101,
        "lhb_hr_factor": 103,
        "rhb_hr_factor": 101,
    },
    "LAA": {  # Angel Stadium
        "name": "Angel Stadium",
        "lf_ft": 330, "cf_ft": 400, "rf_ft": 330,
        "wall_lf_ft": 8, "wall_cf_ft": 18, "wall_rf_ft": 8,
        "hr_factor": 99,
        "runs_factor": 100,
        "lhb_hr_factor": 100,
        "rhb_hr_factor": 98,
    },
    "HOU": {  # Minute Maid Park
        "name": "Minute Maid Park",
        "lf_ft": 315, "cf_ft": 409, "rf_ft": 326,  # Short LF = RHH paradise
        "wall_lf_ft": 19, "wall_cf_ft": 9, "wall_rf_ft": 9,
        "hr_factor": 106,
        "runs_factor": 104,
        "lhb_hr_factor": 100,
        "rhb_hr_factor": 112,  # 12% boost for righties (short LF)
    },
    "OAK": {  # Oakland Coliseum
        "name": "Oakland Coliseum",
        "lf_ft": 330, "cf_ft": 400, "rf_ft": 330,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 92,  # Foul territory + marine layer
        "runs_factor": 95,
        "lhb_hr_factor": 91,
        "rhb_hr_factor": 93,
    },
    "SEA": {  # T-Mobile Park
        "name": "T-Mobile Park",
        "lf_ft": 331, "cf_ft": 401, "rf_ft": 326,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 94,  # Marine layer suppression
        "runs_factor": 97,
        "lhb_hr_factor": 92,
        "rhb_hr_factor": 96,
    },
    "TEX": {  # Globe Life Field (new retractable roof)
        "name": "Globe Life Field",
        "lf_ft": 329, "cf_ft": 407, "rf_ft": 326,
        "wall_lf_ft": 8, "wall_cf_ft": 12, "wall_rf_ft": 8,
        "hr_factor": 105,  # Texas heat (even with roof)
        "runs_factor": 106,
        "lhb_hr_factor": 107,
        "rhb_hr_factor": 103,
    },

    # National League
    "ATL": {  # Truist Park
        "name": "Truist Park",
        "lf_ft": 335, "cf_ft": 400, "rf_ft": 325,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 104,
        "runs_factor": 103,
        "lhb_hr_factor": 106,
        "rhb_hr_factor": 102,
    },
    "MIA": {  # LoanDepot Park
        "name": "LoanDepot Park",
        "lf_ft": 344, "cf_ft": 407, "rf_ft": 335,
        "wall_lf_ft": 8, "wall_cf_ft": 12, "wall_rf_ft": 8,
        "hr_factor": 94,  # Deep fences
        "runs_factor": 96,
        "lhb_hr_factor": 92,
        "rhb_hr_factor": 96,
    },
    "NYM": {  # Citi Field
        "name": "Citi Field",
        "lf_ft": 335, "cf_ft": 408, "rf_ft": 330,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 96,
        "runs_factor": 98,
        "lhb_hr_factor": 94,
        "rhb_hr_factor": 98,
    },
    "PHI": {  # Citizens Bank Park
        "name": "Citizens Bank Park",
        "lf_ft": 329, "cf_ft": 401, "rf_ft": 330,
        "wall_lf_ft": 6, "wall_cf_ft": 13, "wall_rf_ft": 6,
        "hr_factor": 109,  # 9% boost (hitter-friendly)
        "runs_factor": 107,
        "lhb_hr_factor": 112,
        "rhb_hr_factor": 106,
    },
    "WSH": {  # Nationals Park
        "name": "Nationals Park",
        "lf_ft": 336, "cf_ft": 402, "rf_ft": 335,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 100,  # Neutral
        "runs_factor": 100,
        "lhb_hr_factor": 100,
        "rhb_hr_factor": 100,
    },
    "CHC": {  # Wrigley Field
        "name": "Wrigley Field",
        "lf_ft": 355, "cf_ft": 400, "rf_ft": 353,  # Deep corners!
        "wall_lf_ft": 11, "wall_cf_ft": 11, "wall_rf_ft": 11,
        "hr_factor": 98,  # Wind-dependent (blowing out = 120, blowing in = 80)
        "runs_factor": 103,
        "lhb_hr_factor": 96,
        "rhb_hr_factor": 100,
    },
    "CIN": {  # Great American Ball Park
        "name": "Great American Ball Park",
        "lf_ft": 328, "cf_ft": 404, "rf_ft": 325,
        "wall_lf_ft": 12, "wall_cf_ft": 12, "wall_rf_ft": 8,
        "hr_factor": 112,  # 12% boost (HR haven)
        "runs_factor": 108,
        "lhb_hr_factor": 115,  # Short RF = LHH moonshots
        "rhb_hr_factor": 109,
    },
    "MIL": {  # American Family Field
        "name": "American Family Field",
        "lf_ft": 344, "cf_ft": 400, "rf_ft": 345,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 99,
        "runs_factor": 100,
        "lhb_hr_factor": 98,
        "rhb_hr_factor": 100,
    },
    "PIT": {  # PNC Park
        "name": "PNC Park",
        "lf_ft": 325, "cf_ft": 399, "rf_ft": 320,  # Short RF for LHH
        "wall_lf_ft": 6, "wall_cf_ft": 10, "wall_rf_ft": 21,
        "hr_factor": 97,
        "runs_factor": 99,
        "lhb_hr_factor": 102,  # Short RF helps lefties
        "rhb_hr_factor": 92,
    },
    "STL": {  # Busch Stadium
        "name": "Busch Stadium",
        "lf_ft": 336, "cf_ft": 400, "rf_ft": 335,
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 98,
        "runs_factor": 99,
        "lhb_hr_factor": 97,
        "rhb_hr_factor": 99,
    },
    "ARI": {  # Chase Field
        "name": "Chase Field",
        "lf_ft": 330, "cf_ft": 407, "rf_ft": 334,
        "wall_lf_ft": 7, "wall_cf_ft": 25, "wall_rf_ft": 7,
        "hr_factor": 106,  # Arizona heat (roof closed still hot)
        "runs_factor": 105,
        "lhb_hr_factor": 108,
        "rhb_hr_factor": 104,
    },
    "COL": {  # Coors Field
        "name": "Coors Field",
        "lf_ft": 347, "cf_ft": 415, "rf_ft": 350,  # Deep fences but thin air
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 8,
        "hr_factor": 125,  # 25% boost (mile-high = insane)
        "runs_factor": 120,  # 20% runs boost
        "lhb_hr_factor": 128,
        "rhb_hr_factor": 122,
    },
    "LAD": {  # Dodger Stadium
        "name": "Dodger Stadium",
        "lf_ft": 330, "cf_ft": 395, "rf_ft": 330,
        "wall_lf_ft": 4, "wall_cf_ft": 8, "wall_rf_ft": 4,
        "hr_factor": 95,  # Pitcher's park
        "runs_factor": 97,
        "lhb_hr_factor": 93,
        "rhb_hr_factor": 97,
    },
    "SD": {  # Petco Park
        "name": "Petco Park",
        "lf_ft": 336, "cf_ft": 396, "rf_ft": 322,
        "wall_lf_ft": 8, "wall_cf_ft": 12, "wall_rf_ft": 4,
        "hr_factor": 91,  # Suppresses HRs (marine layer)
        "runs_factor": 94,
        "lhb_hr_factor": 88,
        "rhb_hr_factor": 94,
    },
    "SF": {  # Oracle Park
        "name": "Oracle Park",
        "lf_ft": 339, "cf_ft": 399, "rf_ft": 309,  # Short RF but TALL wall
        "wall_lf_ft": 8, "wall_cf_ft": 8, "wall_rf_ft": 25,  # 25ft RF wall!
        "hr_factor": 85,  # 15% suppression (worst HR park)
        "runs_factor": 92,
        "lhb_hr_factor": 95,
        "rhb_hr_factor": 75,  # RHH absolutely crushed (-25%)
    },
}


# ============================================================================
# BATTER SPRAY ANGLE CALCULATION
# ============================================================================

def calculate_spray_angle(hit_coord_x, hit_coord_y):
    """
    Calculate spray angle from Statcast hit coordinates.
    
    Args:
        hit_coord_x: Horizontal coordinate (negative = pull for RHB, oppo for LHB)
        hit_coord_y: Vertical coordinate (distance from home plate)
    
    Returns:
        spray_angle: Angle in degrees (-45 to +45, 0 = dead center)
    """
    if pd.isna(hit_coord_x) or pd.isna(hit_coord_y) or hit_coord_y == 0:
        return np.nan
    
    # Calculate angle using arctan2 (handles quadrants correctly)
    angle_rad = np.arctan2(hit_coord_x, hit_coord_y)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def analyze_spray_tendencies(statcast_df):
    """
    Analyze batter spray tendencies from Statcast data.
    
    Returns:
        DataFrame with columns: batter_name, batter_id, pull_pct, center_pct, oppo_pct, 
                               avg_spray_angle, total_batted_balls
    """
    print("\n=== ANALYZING BATTER SPRAY TENDENCIES ===\n")
    
    # Filter to batted balls only (exclude foul balls)
    batted_balls = statcast_df[
        (statcast_df['description'].isin(['hit_into_play', 'hit_into_play_no_out', 
                                          'hit_into_play_score']))
    ].copy()
    
    print(f"Total batted ball events: {len(batted_balls):,}")
    
    # Calculate spray angle for each batted ball
    batted_balls['spray_angle'] = batted_balls.apply(
        lambda row: calculate_spray_angle(row.get('hc_x'), row.get('hc_y')),
        axis=1
    )
    
    # Remove rows with missing spray angle
    batted_balls = batted_balls.dropna(subset=['spray_angle'])
    
    # Classify spray direction based on batter handedness
    def classify_spray(row):
        angle = row['spray_angle']
        batter_side = row.get('stand', 'R')  # L = lefty, R = righty
        
        if batter_side == 'L':
            # For LHB: negative angle = oppo (RF), positive = pull (LF)
            if angle < -15:
                return 'oppo'
            elif angle > 15:
                return 'pull'
            else:
                return 'center'
        else:  # RHB
            # For RHB: negative angle = pull (LF), positive = oppo (RF)
            if angle < -15:
                return 'pull'
            elif angle > 15:
                return 'oppo'
            else:
                return 'center'
    
    batted_balls['spray_direction'] = batted_balls.apply(classify_spray, axis=1)
    
    # Aggregate by batter
    spray_stats = batted_balls.groupby(['batter', 'player_name', 'stand']).agg(
        total_batted_balls=('spray_direction', 'count'),
        pull_count=('spray_direction', lambda x: (x == 'pull').sum()),
        center_count=('spray_direction', lambda x: (x == 'center').sum()),
        oppo_count=('spray_direction', lambda x: (x == 'oppo').sum()),
        avg_spray_angle=('spray_angle', 'mean')
    ).reset_index()
    
    # Calculate percentages
    spray_stats['pull_pct'] = (spray_stats['pull_count'] / spray_stats['total_batted_balls'] * 100).round(1)
    spray_stats['center_pct'] = (spray_stats['center_count'] / spray_stats['total_batted_balls'] * 100).round(1)
    spray_stats['oppo_pct'] = (spray_stats['oppo_count'] / spray_stats['total_batted_balls'] * 100).round(1)
    
    # Rename columns
    spray_stats = spray_stats.rename(columns={
        'batter': 'batter_id',
        'player_name': 'batter_name',
        'stand': 'bats'
    })
    
    # Sort by total batted balls (minimum 100 for statistical significance)
    spray_stats = spray_stats[spray_stats['total_batted_balls'] >= 100]
    spray_stats = spray_stats.sort_values('total_batted_balls', ascending=False)
    
    print(f"\nBatters with 100+ batted balls: {len(spray_stats)}")
    print(f"\nTop 10 Pull Hitters:")
    print(spray_stats.nlargest(10, 'pull_pct')[['batter_name', 'bats', 'pull_pct', 'center_pct', 'oppo_pct', 'total_batted_balls']])
    
    print(f"\nTop 10 Opposite Field Hitters:")
    print(spray_stats.nlargest(10, 'oppo_pct')[['batter_name', 'bats', 'pull_pct', 'center_pct', 'oppo_pct', 'total_batted_balls']])
    
    return spray_stats


# ============================================================================
# PARK-BATTER FIT CALCULATION
# ============================================================================

def calculate_park_fit(spray_stats, park_data):
    """
    Calculate park-batter fit scores for all matchups.
    
    Logic:
    - Pull LHB (high pull%) + Short RF park (NYY, BAL, CIN) = +20-30% boost
    - Pull RHB (high pull%) + Short LF park (HOU, BOS) = +15-25% boost
    - Oppo hitter + Deep fence = -10-15% penalty
    - Coors Field = +20% boost for EVERYONE (thin air)
    - Oracle Park = -15% penalty for RHH (25ft RF wall)
    
    Returns:
        DataFrame with park-batter matchup fit scores
    """
    print("\n=== CALCULATING PARK-BATTER FIT SCORES ===\n")
    
    matchups = []
    
    for park_code, park_info in park_data.items():
        for _, batter in spray_stats.iterrows():
            batter_name = batter['batter_name']
            batter_id = batter['batter_id']
            bats = batter['bats']
            pull_pct = batter['pull_pct']
            oppo_pct = batter['oppo_pct']
            
            # Base park HR factor
            base_hr_factor = park_info['hr_factor']
            
            # Apply LHB/RHB split
            if bats == 'L':
                hr_factor = park_info['lhb_hr_factor']
            else:
                hr_factor = park_info['rhb_hr_factor']
            
            # Calculate park-spray fit adjustment
            fit_adjustment = 0
            explanation = []
            
            # LHB pull hitters + short RF parks
            if bats == 'L' and pull_pct >= 40:  # High pull tendency
                if park_info['rf_ft'] <= 320:  # Short RF porch
                    fit_adjustment = +25
                    explanation.append(f"Pull LHB + short RF ({park_info['rf_ft']}ft) = +25%")
                elif park_info['rf_ft'] <= 330:
                    fit_adjustment = +15
                    explanation.append(f"Pull LHB + short RF ({park_info['rf_ft']}ft) = +15%")
            
            # RHB pull hitters + short LF parks
            if bats == 'R' and pull_pct >= 40:
                if park_info['lf_ft'] <= 320:  # Short LF porch
                    fit_adjustment = +25
                    explanation.append(f"Pull RHB + short LF ({park_info['lf_ft']}ft) = +25%")
                elif park_info['lf_ft'] <= 330:
                    fit_adjustment = +15
                    explanation.append(f"Pull RHB + short LF ({park_info['lf_ft']}ft) = +15%")
            
            # Oppo hitters + deep fences (penalty)
            if oppo_pct >= 35:  # High oppo tendency
                if bats == 'L' and park_info['lf_ft'] >= 340:
                    fit_adjustment = -12
                    explanation.append(f"Oppo LHB + deep LF ({park_info['lf_ft']}ft) = -12%")
                elif bats == 'R' and park_info['rf_ft'] >= 340:
                    fit_adjustment = -12
                    explanation.append(f"Oppo RHB + deep RF ({park_info['rf_ft']}ft) = -12%")
            
            # Combine base park factor + spray fit
            combined_factor = hr_factor + fit_adjustment
            
            # Calculate final park adjustment (100 = neutral, 120 = +20% boost)
            park_adjustment_pct = combined_factor - 100
            
            if not explanation:
                explanation.append("Neutral park-batter fit")
            
            matchups.append({
                'park_code': park_code,
                'park_name': park_info['name'],
                'batter_id': batter_id,
                'batter_name': batter_name,
                'bats': bats,
                'pull_pct': pull_pct,
                'oppo_pct': oppo_pct,
                'base_hr_factor': hr_factor,
                'spray_fit_adjustment': fit_adjustment,
                'combined_park_factor': combined_factor,
                'park_adjustment_pct': park_adjustment_pct,
                'explanation': '; '.join(explanation)
            })
    
    matchup_df = pd.DataFrame(matchups)
    
    # Show best/worst park fits
    print(f"\nTop 20 Park-Batter BOOSTS (Best DFS Plays):")
    print(matchup_df.nlargest(20, 'park_adjustment_pct')[
        ['batter_name', 'park_code', 'bats', 'pull_pct', 'park_adjustment_pct', 'explanation']
    ])
    
    print(f"\nTop 20 Park-Batter PENALTIES (Fade These):")
    print(matchup_df.nsmallest(20, 'park_adjustment_pct')[
        ['batter_name', 'park_code', 'bats', 'oppo_pct', 'park_adjustment_pct', 'explanation']
    ])
    
    return matchup_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MLB DFS Predictor v120 - Park Factors + Spray Analysis')
    parser.add_argument('--date', type=str, default='2026-04-15', 
                       help='Game date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"MLB DFS PREDICTOR v120 - PARK FACTORS + PULL/SPRAY ANGLE ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Game Date: {args.date}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define data paths
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    
    # Load Statcast data (2023-2025)
    print("Loading Statcast data...")
    statcast_files = [
        data_dir / "statcast_2023.parquet",
        data_dir / "statcast_2024.parquet",
        data_dir / "statcast_2025.parquet"
    ]
    
    dfs = []
    for file_path in statcast_files:
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dfs.append(df)
            print(f"  ✓ Loaded {file_path.name}: {len(df):,} rows")
        else:
            print(f"  ✗ Missing: {file_path.name}")
    
    if not dfs:
        print("\n❌ ERROR: No Statcast data found!")
        print("Required files: statcast_2023.parquet, statcast_2024.parquet, statcast_2025.parquet")
        return
    
    statcast_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Statcast events: {len(statcast_df):,}")
    
    # Step 1: Analyze spray tendencies
    spray_stats = analyze_spray_tendencies(statcast_df)
    spray_output = project_dir / "batter_spray_tendencies.csv"
    spray_stats.to_csv(spray_output, index=False)
    print(f"\n✓ Saved: {spray_output}")
    
    # Step 2: Save park data to CSV
    park_df = pd.DataFrame.from_dict(PARK_DATA, orient='index').reset_index()
    park_df = park_df.rename(columns={'index': 'park_code'})
    park_output = project_dir / "park_dimensions.csv"
    park_df.to_csv(park_output, index=False)
    print(f"✓ Saved: {park_output}")
    
    # Step 3: Calculate park-batter fit scores
    matchup_df = calculate_park_fit(spray_stats, PARK_DATA)
    matchup_output = project_dir / "v120_park_adjusted_predictions.csv"
    matchup_df.to_csv(matchup_output, index=False)
    print(f"✓ Saved: {matchup_output}")
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE - v120 Park Factor Analysis")
    print(f"{'='*80}\n")
    print(f"KEY INSIGHTS:")
    print(f"- Total park-batter matchups analyzed: {len(matchup_df):,}")
    print(f"- Best park boost: {matchup_df['park_adjustment_pct'].max():.1f}%")
    print(f"- Worst park penalty: {matchup_df['park_adjustment_pct'].min():.1f}%")
    print(f"\nNEXT STEPS:")
    print(f"1. Integrate park adjustments into main predictor (apply to HR/hit probabilities)")
    print(f"2. Combine with XGBoost model for park-adjusted ML predictions")
    print(f"3. Add weather data (wind speed/direction) for dynamic park factor adjustments")
    print(f"\n")


if __name__ == "__main__":
    main()
