"""
Win Probability Model
=======================
Calculates win probability for each team in today's games.

Factors:
1. Starting pitcher quality (ERA, FIP proxy, WHIP, K/9)
2. Team lineup strength (OPS vs pitcher handedness)
3. Bullpen strength (team bullpen ERA proxy)
4. Home field advantage
5. Park factor

Output: win probability for each team, confidence tier.
"""

import logging
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────

# Home field advantage: home teams win ~54% of the time in MLB
HOME_WIN_RATE = 0.540

# League averages (2025)
LEAGUE_AVG_ERA = 4.02
LEAGUE_AVG_WHIP = 1.28
LEAGUE_AVG_K9 = 8.6
LEAGUE_AVG_OPS = 0.720
LEAGUE_AVG_BULLPEN_ERA = 3.85

# Weight for each factor in the model
WEIGHTS = {
    "starter_era": 0.25,
    "starter_whip": 0.10,
    "starter_k9": 0.10,
    "lineup_ops": 0.25,
    "bullpen": 0.10,
    "home_field": 0.10,
    "park_factor": 0.05,
    "record": 0.05,
}


# ── Factor Functions ──────────────────────────────────────────

def starter_era_factor(era: float) -> float:
    """
    Convert ERA to a 0-1 quality score.
    ERA 1.5 → ~0.85 (elite)
    ERA 4.0 → ~0.50 (average)
    ERA 6.5 → ~0.15 (poor)
    """
    if era <= 0:
        return 0.50
    # Sigmoid-like transformation centered at league avg
    score = 1 / (1 + math.exp(0.5 * (era - LEAGUE_AVG_ERA)))
    return max(0.10, min(0.90, score))


def starter_whip_factor(whip: float) -> float:
    """WHIP quality score."""
    if whip <= 0:
        return 0.50
    score = 1 / (1 + math.exp(2.0 * (whip - LEAGUE_AVG_WHIP)))
    return max(0.10, min(0.90, score))


def starter_k9_factor(k_per_9: float) -> float:
    """Strikeout rate quality score. Higher K/9 = better."""
    if k_per_9 <= 0:
        return 0.50
    score = 1 / (1 + math.exp(-0.4 * (k_per_9 - LEAGUE_AVG_K9)))
    return max(0.10, min(0.90, score))


def lineup_ops_factor(team_ops: float, pitcher_hand: str = "R") -> float:
    """
    Team batting quality based on OPS.
    In practice, should use OPS vs LHP or vs RHP based on opposing starter.
    """
    if team_ops <= 0:
        team_ops = LEAGUE_AVG_OPS
    score = 1 / (1 + math.exp(-5.0 * (team_ops - LEAGUE_AVG_OPS)))
    return max(0.10, min(0.90, score))


def bullpen_factor(bullpen_era: float) -> float:
    """Bullpen quality. Lower ERA = better."""
    if bullpen_era <= 0:
        return 0.50
    score = 1 / (1 + math.exp(0.4 * (bullpen_era - LEAGUE_AVG_BULLPEN_ERA)))
    return max(0.10, min(0.90, score))


def home_field_factor(is_home: bool) -> float:
    """Home field advantage."""
    return HOME_WIN_RATE if is_home else (1 - HOME_WIN_RATE)


def park_factor_score(park_factor: float) -> float:
    """Park factor for runs. Higher park factor slightly favors the home team offense."""
    return 0.50 + (park_factor - 1.0) * 0.3


def record_factor(wins: int, losses: int) -> float:
    """Team quality from win-loss record."""
    total = wins + losses
    if total == 0:
        return 0.50
    return wins / total


# ── FIP Proxy ─────────────────────────────────────────────────

def estimate_fip(era: float, k_per_9: float, bb_per_9: float,
                  hr_per_9: float) -> float:
    """
    Estimate FIP (Fielding Independent Pitching).
    FIP = ((13*HR + 3*BB - 2*K) / IP) + constant
    We approximate since we work in per-9 rates.
    """
    if k_per_9 <= 0:
        return era  # fallback

    # Per-9 FIP approximation
    fip = ((13 * hr_per_9 + 3 * bb_per_9 - 2 * k_per_9) / 9) + 3.10
    # Constant is usually ~3.10 for modern MLB
    return max(1.0, min(7.0, fip))


# ── Core Model ────────────────────────────────────────────────

def calculate_team_win_probability(
    team_starter_era: float,
    team_starter_whip: float,
    team_starter_k9: float,
    team_lineup_ops: float,
    team_bullpen_era: float,
    team_wins: int,
    team_losses: int,
    is_home: bool,
    park_factor: float,
    opp_starter_era: float,
    opp_starter_whip: float,
    opp_starter_k9: float,
    opp_lineup_ops: float,
    opp_bullpen_era: float,
    opp_wins: int,
    opp_losses: int,
) -> float:
    """
    Calculate win probability for a team.
    Returns probability between 0.0 and 1.0.
    """
    w = WEIGHTS

    # Team factors — how good things are FOR this team
    # Good starter ERA (low) = good for team → use team's own pitcher
    # Bad opponent starter (high ERA) = good for team → invert opponent's factor
    team_score = (
        w["starter_era"] * starter_era_factor(team_starter_era) +  # Our ace helps us
        w["starter_whip"] * starter_whip_factor(team_starter_whip) +
        w["starter_k9"] * starter_k9_factor(team_starter_k9) +  # Our K rate helps us
        w["lineup_ops"] * lineup_ops_factor(team_lineup_ops) +
        w["bullpen"] * bullpen_factor(team_bullpen_era) +
        w["home_field"] * home_field_factor(is_home) +
        w["park_factor"] * park_factor_score(park_factor) +
        w["record"] * record_factor(team_wins, team_losses)
    )

    # Opponent factors (mirror — how good things are for the opponent)
    opp_score = (
        w["starter_era"] * starter_era_factor(opp_starter_era) +
        w["starter_whip"] * starter_whip_factor(opp_starter_whip) +
        w["starter_k9"] * starter_k9_factor(opp_starter_k9) +
        w["lineup_ops"] * lineup_ops_factor(opp_lineup_ops) +
        w["bullpen"] * bullpen_factor(opp_bullpen_era) +
        w["home_field"] * home_field_factor(not is_home) +
        w["park_factor"] * park_factor_score(1.0 / max(park_factor, 0.5)) +
        w["record"] * record_factor(opp_wins, opp_losses)
    )

    # Convert to probability using logistic
    total = team_score + opp_score
    if total <= 0:
        return 0.50

    raw_prob = team_score / total

    # Slight regression toward 0.50 (no game is ever truly 70-30)
    regression = 0.08
    regressed = raw_prob * (1 - regression) + 0.50 * regression

    return max(0.32, min(0.68, regressed))


# ── Demo Team Data ────────────────────────────────────────────

DEMO_TEAM_OPS = {
    "NYY": 0.752, "BOS": 0.738, "LAD": 0.768, "SF": 0.701,
    "HOU": 0.745, "TEX": 0.718, "PHI": 0.755, "ATL": 0.742,
    "MIN": 0.722, "DET": 0.698, "CLE": 0.710, "BAL": 0.748,
    "KC": 0.715, "SEA": 0.695, "NYM": 0.725, "SD": 0.718,
    "MIL": 0.728, "CHC": 0.712, "CIN": 0.735, "STL": 0.705,
    "MIA": 0.680, "CWS": 0.665, "COL": 0.710, "TB": 0.708,
    "PIT": 0.698, "WSH": 0.705, "TOR": 0.715, "LAA": 0.705,
    "AZ": 0.732, "ATH": 0.678,
}

DEMO_BULLPEN_ERA = {
    "NYY": 3.55, "BOS": 3.72, "LAD": 3.38, "SF": 3.85,
    "HOU": 3.48, "TEX": 3.92, "PHI": 3.42, "ATL": 3.65,
    "MIN": 3.95, "DET": 3.58, "CLE": 3.32, "BAL": 3.78,
    "KC": 3.55, "SEA": 3.68, "NYM": 3.72, "SD": 3.82,
    "MIL": 3.45, "CHC": 3.88, "CIN": 4.05, "STL": 3.75,
    "MIA": 4.15, "CWS": 4.55, "COL": 4.42, "TB": 3.52,
    "PIT": 3.92, "WSH": 4.08, "TOR": 3.85, "LAA": 4.12,
    "AZ": 3.68, "ATH": 4.28,
}


# ── Game Predictions Generator ────────────────────────────────

def generate_win_predictions(
    games: List[Dict],
    pitcher_stats: Dict[str, Dict],
    team_ops: Optional[Dict[str, float]] = None,
    bullpen_era: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Generate win probability predictions for today's games.

    Args:
        games: List of game dicts from schedule
        pitcher_stats: {pitcher_name: {era, whip, k_per_9, ...}}
        team_ops: {team_abbr: float} — team OPS
        bullpen_era: {team_abbr: float} — team bullpen ERA

    Returns:
        List of game prediction dicts
    """
    if team_ops is None:
        team_ops = DEMO_TEAM_OPS
    if bullpen_era is None:
        bullpen_era = DEMO_BULLPEN_ERA

    predictions = []

    for game in games:
        is_demo = game.get("game_type") == "DEMO"
        away = game["away"]
        home = game["home"]

        # Get pitcher stats
        away_pitcher_name = away["probable_pitcher"]["name"]
        home_pitcher_name = home["probable_pitcher"]["name"]
        away_pstats = pitcher_stats.get(away_pitcher_name, {})
        home_pstats = pitcher_stats.get(home_pitcher_name, {})

        away_era = away_pstats.get("era", LEAGUE_AVG_ERA)
        away_whip = away_pstats.get("whip", LEAGUE_AVG_WHIP)
        away_k9 = away_pstats.get("k_per_9", LEAGUE_AVG_K9)
        home_era = home_pstats.get("era", LEAGUE_AVG_ERA)
        home_whip = home_pstats.get("whip", LEAGUE_AVG_WHIP)
        home_k9 = home_pstats.get("k_per_9", LEAGUE_AVG_K9)

        # Team stats
        away_ops = team_ops.get(away["abbr"], LEAGUE_AVG_OPS)
        home_ops = team_ops.get(home["abbr"], LEAGUE_AVG_OPS)
        away_bp = bullpen_era.get(away["abbr"], LEAGUE_AVG_BULLPEN_ERA)
        home_bp = bullpen_era.get(home["abbr"], LEAGUE_AVG_BULLPEN_ERA)

        # Records
        away_wins = away.get("record", {}).get("wins", 81)
        away_losses = away.get("record", {}).get("losses", 81)
        home_wins = home.get("record", {}).get("wins", 81)
        home_losses = home.get("record", {}).get("losses", 81)

        park_factor = game.get("park_factor", 1.0)

        # Calculate home win probability
        home_wp = calculate_team_win_probability(
            team_starter_era=home_era,
            team_starter_whip=home_whip,
            team_starter_k9=home_k9,
            team_lineup_ops=home_ops,
            team_bullpen_era=home_bp,
            team_wins=home_wins,
            team_losses=home_losses,
            is_home=True,
            park_factor=park_factor,
            opp_starter_era=away_era,
            opp_starter_whip=away_whip,
            opp_starter_k9=away_k9,
            opp_lineup_ops=away_ops,
            opp_bullpen_era=away_bp,
            opp_wins=away_wins,
            opp_losses=away_losses,
        )
        away_wp = 1 - home_wp

        # Determine pick and confidence
        if home_wp >= away_wp:
            pick = home["abbr"]
            pick_name = home["name"]
            pick_prob = home_wp
        else:
            pick = away["abbr"]
            pick_name = away["name"]
            pick_prob = away_wp

        # Confidence tier
        edge = abs(home_wp - 0.50)
        if edge >= 0.12:
            confidence = "🔒 Lock"
        elif edge >= 0.08:
            confidence = "💪 Strong"
        elif edge >= 0.04:
            confidence = "👀 Lean"
        else:
            confidence = "📊 Toss-up"

        # FIP estimates
        away_fip = estimate_fip(
            away_era, away_k9,
            away_pstats.get("bb_per_9", 3.0),
            away_pstats.get("hr_per_9", 1.0)
        )
        home_fip = estimate_fip(
            home_era, home_k9,
            home_pstats.get("bb_per_9", 3.0),
            home_pstats.get("hr_per_9", 1.0)
        )

        predictions.append({
            "game_pk": game["game_pk"],
            "game_label": f"{away['abbr']} @ {home['abbr']}",
            "game_time": game.get("game_time", ""),
            "venue": game.get("venue", ""),
            "status": game.get("status", "Scheduled"),
            "away": {
                "abbr": away["abbr"],
                "name": away["name"],
                "record": f"{away_wins}-{away_losses}",
                "probable_pitcher": away_pitcher_name,
                "pitcher_era": round(away_era, 2),
                "pitcher_whip": round(away_whip, 2),
                "pitcher_k9": round(away_k9, 1),
                "pitcher_fip": round(away_fip, 2),
                "team_ops": round(away_ops, 3),
                "bullpen_era": round(away_bp, 2),
                "win_probability": round(away_wp, 3),
            },
            "home": {
                "abbr": home["abbr"],
                "name": home["name"],
                "record": f"{home_wins}-{home_losses}",
                "probable_pitcher": home_pitcher_name,
                "pitcher_era": round(home_era, 2),
                "pitcher_whip": round(home_whip, 2),
                "pitcher_k9": round(home_k9, 1),
                "pitcher_fip": round(home_fip, 2),
                "team_ops": round(home_ops, 3),
                "bullpen_era": round(home_bp, 2),
                "win_probability": round(home_wp, 3),
            },
            "pick": pick,
            "pick_name": pick_name,
            "pick_probability": round(pick_prob, 3),
            "confidence": confidence,
            "park_factor": round(park_factor, 2),
            "is_demo": is_demo,
            "factors": {
                "pitcher_matchup": _pitcher_matchup_summary(
                    home_pitcher_name, home_era, away_pitcher_name, away_era
                ),
                "home_field": f"{home['abbr']} has home field advantage",
                "park": f"Park factor: {park_factor:.2f}",
            }
        })

    # Sort by pick probability (most confident first)
    predictions.sort(key=lambda x: x["pick_probability"], reverse=True)

    return predictions


def _pitcher_matchup_summary(home_p: str, home_era: float,
                              away_p: str, away_era: float) -> str:
    """Generate a human-readable pitcher matchup summary."""
    delta = away_era - home_era
    if abs(delta) < 0.3:
        return f"{home_p} vs {away_p}: Even matchup"
    elif delta > 0:
        return f"{home_p} ({home_era:.2f} ERA) has edge over {away_p} ({away_era:.2f} ERA)"
    else:
        return f"{away_p} ({away_era:.2f} ERA) has edge over {home_p} ({home_era:.2f} ERA)"
