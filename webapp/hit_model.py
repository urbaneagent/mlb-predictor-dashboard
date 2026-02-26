"""
Hit Probability Model
=======================
Calculates the probability that each batter in today's lineup
records at least 1 hit, based on multiple factors:

1. Season batting average (baseline)
2. Platoon splits (batter hand vs pitcher hand)
3. Home/away adjustment
4. Park factor
5. Pitcher quality (ERA, WHIP, avg against)
6. Recent performance proxy

Output: ranked list of "Today's Top Hitters" with hit probability %.
"""

import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────

# Average MLB batting average (2025 approx)
LEAGUE_AVG_BA = 0.248

# Platoon advantage multipliers
# When batter has platoon advantage (L vs R or R vs L), they hit better
PLATOON_ADVANTAGE_BOOST = 1.08  # +8% to BA
PLATOON_DISADVANTAGE_PENALTY = 0.92  # -8% to BA

# Home/away split
HOME_BOOST = 1.03  # Batters hit ~3% better at home
AWAY_PENALTY = 0.97

# Pitcher quality buckets (ERA-based adjustment to opponent BA)
def pitcher_quality_factor(era: float) -> float:
    """
    Ace pitchers suppress hits, bad pitchers inflate them.
    Returns multiplier for batter's BA against this pitcher.
    """
    if era <= 0:
        return 1.0
    # League avg ERA ~4.00. Lower ERA = tougher to hit.
    # ERA 2.0 → 0.88 multiplier (suppress 12%)
    # ERA 4.0 → 1.00 (neutral)
    # ERA 6.0 → 1.12 (inflate 12%)
    return 1.0 + (era - 4.0) * 0.06


def whip_adjustment(whip: float) -> float:
    """Additional adjustment based on WHIP."""
    if whip <= 0:
        return 1.0
    # League avg WHIP ~1.28
    # Low WHIP (0.90) → harder to hit → 0.95
    # High WHIP (1.50) → easier → 1.06
    return 1.0 + (whip - 1.28) * 0.16


# ── Core Model ────────────────────────────────────────────────

def calculate_single_ab_hit_probability(
    batter_avg: float,
    pitcher_era: float = 4.0,
    pitcher_whip: float = 1.28,
    pitcher_hand: str = "R",
    batter_hand: str = "R",
    is_home: bool = True,
    park_factor: float = 1.0,
    pitcher_avg_against: float = 0.248,
) -> float:
    """
    Estimate probability of a hit in a single at-bat.

    Combines:
    - Batter's season BA (regressed toward mean)
    - Pitcher quality (ERA + WHIP adjustment)
    - Platoon matchup
    - Home/away
    - Park factor
    """
    if batter_avg <= 0:
        batter_avg = LEAGUE_AVG_BA

    # Start with batter's BA, regressed slightly toward league mean
    # (prevents extreme outliers early in season)
    regressed_avg = batter_avg * 0.85 + LEAGUE_AVG_BA * 0.15

    # Pitcher quality adjustment
    pq = pitcher_quality_factor(pitcher_era)
    wq = whip_adjustment(pitcher_whip)
    pitcher_adj = (pq + wq) / 2  # Average the two signals

    # If we have the pitcher's avg against, blend it in
    if pitcher_avg_against > 0:
        # Pitcher with .200 avg against is harder; .280 is easier
        pitcher_quality_from_avg = pitcher_avg_against / LEAGUE_AVG_BA
        pitcher_adj = pitcher_adj * 0.6 + pitcher_quality_from_avg * 0.4

    # Platoon split
    has_platoon_advantage = (
        (batter_hand == "L" and pitcher_hand == "R") or
        (batter_hand == "R" and pitcher_hand == "L")
    )
    if batter_hand == "S":
        # Switch hitters always bat from advantaged side
        platoon = PLATOON_ADVANTAGE_BOOST
    elif has_platoon_advantage:
        platoon = PLATOON_ADVANTAGE_BOOST
    else:
        platoon = PLATOON_DISADVANTAGE_PENALTY

    # Home/away
    location = HOME_BOOST if is_home else AWAY_PENALTY

    # Park factor
    park = park_factor if park_factor > 0 else 1.0

    # Final single-AB hit probability
    p_hit = regressed_avg * pitcher_adj * platoon * location * park

    # Clamp to reasonable range [0.10, 0.42]
    return max(0.10, min(0.42, p_hit))


def calculate_game_hit_probability(
    single_ab_prob: float,
    expected_abs: float = 3.8,
) -> float:
    """
    P(at least 1 hit in game) = 1 - P(0 hits in all ABs)
    = 1 - (1 - p)^n

    Average batter gets ~3.8 AB per game.
    Leadoff hitters get ~4.5, bottom of order ~3.3.
    """
    if single_ab_prob <= 0:
        return 0.0
    if single_ab_prob >= 1:
        return 1.0

    p_no_hit = (1 - single_ab_prob) ** expected_abs
    return 1 - p_no_hit


def lineup_position_abs(position: int) -> float:
    """Expected at-bats based on lineup position (1-9)."""
    # Leadoff gets most ABs, 9-hole gets fewest
    abs_by_position = {
        1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 4.0,
        6: 3.8, 7: 3.6, 8: 3.5, 9: 3.3,
    }
    return abs_by_position.get(position, 3.8)


# ── Top Hitters Generator ────────────────────────────────────

def generate_top_hitters(
    games: List[Dict],
    batter_stats: Dict[str, Dict],
    pitcher_stats: Dict[str, Dict],
    demo_lineups: Optional[Dict[int, Dict]] = None,
) -> List[Dict]:
    """
    Generate today's top hitters with hit probabilities.

    Args:
        games: List of game dicts from schedule
        batter_stats: {player_name: {avg, ops, bat_side, ...}}
        pitcher_stats: {pitcher_name: {era, whip, pitch_hand, avg_against, ...}}
        demo_lineups: Optional pre-set lineups for demo mode

    Returns:
        Sorted list of hitter predictions
    """
    all_hitters = []

    for game in games:
        game_pk = game.get("game_pk")
        park_factor = game.get("park_factor", 1.0)
        is_demo = game.get("game_type") == "DEMO"

        for side in ("away", "home"):
            is_home = (side == "home")
            opposing_side = "home" if side == "away" else "away"

            # Get opposing pitcher info
            opp_pitcher_name = game[opposing_side]["probable_pitcher"]["name"]
            opp_pitcher = pitcher_stats.get(opp_pitcher_name, {})
            pitcher_era = opp_pitcher.get("era", 4.0)
            pitcher_whip = opp_pitcher.get("whip", 1.28)
            pitcher_hand = opp_pitcher.get("pitch_hand", "R")
            pitcher_avg_against = opp_pitcher.get("avg_against", LEAGUE_AVG_BA)

            # Get lineup
            lineup = []
            if demo_lineups and game_pk in demo_lineups:
                lineup = demo_lineups[game_pk].get(side, [])
            else:
                lineup = game.get(f"{side}_lineup", [])

            team_abbr = game[side]["abbr"]
            team_name = game[side]["name"]

            for i, batter in enumerate(lineup):
                batter_name = batter["name"]
                batter_hand = batter.get("bat_side", "R")
                lineup_pos = i + 1

                # Get batter stats
                bstats = batter_stats.get(batter_name, {})
                batter_avg = bstats.get("avg", LEAGUE_AVG_BA)
                batter_ops = bstats.get("ops", 0.720)

                # Calculate hit probability
                single_ab_p = calculate_single_ab_hit_probability(
                    batter_avg=batter_avg,
                    pitcher_era=pitcher_era,
                    pitcher_whip=pitcher_whip,
                    pitcher_hand=pitcher_hand,
                    batter_hand=batter_hand,
                    is_home=is_home,
                    park_factor=park_factor,
                    pitcher_avg_against=pitcher_avg_against,
                )

                expected_abs = lineup_position_abs(lineup_pos)
                game_hit_prob = calculate_game_hit_probability(single_ab_p, expected_abs)

                # Confidence tier
                if game_hit_prob >= 0.82:
                    confidence = "🔒 Lock"
                elif game_hit_prob >= 0.75:
                    confidence = "💪 Strong"
                elif game_hit_prob >= 0.68:
                    confidence = "👀 Lean"
                else:
                    confidence = "📊 Value"

                # Platoon indicator
                has_platoon = (
                    batter_hand == "S" or
                    (batter_hand == "L" and pitcher_hand == "R") or
                    (batter_hand == "R" and pitcher_hand == "L")
                )

                all_hitters.append({
                    "batter": batter_name,
                    "team": team_abbr,
                    "team_name": team_name,
                    "position": batter.get("position", ""),
                    "lineup_pos": lineup_pos,
                    "bat_side": batter_hand,
                    "vs_pitcher": opp_pitcher_name,
                    "pitcher_team": game[opposing_side]["abbr"],
                    "pitcher_hand": pitcher_hand,
                    "batter_avg": round(batter_avg, 3),
                    "batter_ops": round(batter_ops, 3),
                    "pitcher_era": round(pitcher_era, 2),
                    "single_ab_prob": round(single_ab_p, 3),
                    "hit_probability": round(game_hit_prob, 3),
                    "expected_abs": round(expected_abs, 1),
                    "park_factor": round(park_factor, 2),
                    "platoon_advantage": has_platoon,
                    "is_home": is_home,
                    "confidence": confidence,
                    "game_pk": game_pk,
                    "game_label": f"{game['away']['abbr']} @ {game['home']['abbr']}",
                    "is_demo": is_demo,
                })

    # Sort by hit probability (descending)
    all_hitters.sort(key=lambda x: x["hit_probability"], reverse=True)

    return all_hitters
