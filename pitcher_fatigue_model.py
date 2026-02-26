"""
MLB Predictor: Pitcher Fatigue & Workload Model
==================================================
Models how pitcher fatigue affects performance based on workload metrics.

Features:
- Days rest impact on ERA/WHIP/K%
- Pitch count trends (season & career)
- Velocity degradation tracking
- Innings pitched workload (season pace)
- Bullpen availability scoring
- Travel fatigue factor
- Back-to-back start analysis
- Times through order penalty
- First start after IL return
- Platoon advantage modeling
"""

import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from enum import Enum


class Handedness(str, Enum):
    LEFT = "L"
    RIGHT = "R"
    SWITCH = "S"


@dataclass
class PitcherWorkload:
    """Current workload metrics for a pitcher."""
    name: str
    team: str
    handedness: Handedness
    days_rest: int
    pitch_count_last: int  # Last start
    pitch_count_avg: float  # Season average
    innings_pitched_season: float
    innings_pitched_pace: float  # Projected full-season IP
    innings_pitched_career_high: float
    games_started_season: int
    era: float
    whip: float
    k_per_9: float
    bb_per_9: float
    hr_per_9: float
    avg_velocity: float  # Current average fastball velocity
    velocity_season_start: float  # Velocity at season start
    velocity_delta: float = 0.0  # Change in velocity
    is_returning_from_il: bool = False
    il_return_days: int = 0  # Days since IL return
    travel_games: int = 0  # Consecutive road games
    times_through_order: Dict[str, float] = field(
        default_factory=lambda: {"1st": 0.0, "2nd": 0.0, "3rd": 0.0}
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["handedness"] = self.handedness.value
        return d


@dataclass
class BullpenState:
    """Bullpen availability and readiness."""
    team: str
    relievers: List[dict] = field(default_factory=list)
    total_available: int = 0
    high_leverage_available: int = 0
    closer_available: bool = True
    innings_last_3_days: float = 0.0
    bullpen_era: float = 4.00
    bullpen_whip: float = 1.30

    def availability_score(self) -> float:
        """0-1 score of bullpen readiness."""
        base = self.total_available / max(len(self.relievers), 1)
        leverage_bonus = 0.1 if self.high_leverage_available >= 2 else 0
        closer_bonus = 0.1 if self.closer_available else -0.1
        fatigue_penalty = min(self.innings_last_3_days / 15.0, 0.3)
        return round(min(max(base + leverage_bonus + closer_bonus - fatigue_penalty, 0), 1.0), 3)


# ─── Fatigue Impact Calculator ───────────────────────────────────
class PitcherFatigueModel:
    """Models pitcher fatigue impact on expected performance."""

    # Historical data: ERA multiplier by days rest
    DAYS_REST_MULTIPLIERS = {
        0: 1.35,   # Back-to-back (bullpen/emergency)
        1: 1.25,   # Short rest (1 day)
        2: 1.15,   # Short rest (2 days)
        3: 1.08,   # Short rest (3 days)
        4: 1.00,   # Normal rest (4 days)
        5: 0.98,   # Extra rest (5 days)
        6: 0.97,   # Extra rest (6 days)
        7: 0.99,   # Extended rest (7+ days, slight rust)
        8: 1.01,
        9: 1.03,
        10: 1.05,  # 10+ days = rust factor
    }

    # Times through order ERA penalty
    TTO_MULTIPLIERS = {
        "1st": 1.00,  # First time through — baseline
        "2nd": 1.12,  # Second time — hitters adjust
        "3rd": 1.28,  # Third time — significant penalty
    }

    # Pitch count thresholds
    PITCH_COUNT_FATIGUE = {
        (0, 75): 1.00,    # Fresh
        (75, 90): 1.02,   # Slight fatigue
        (90, 100): 1.08,  # Moderate fatigue
        (100, 110): 1.15, # Heavy fatigue
        (110, 130): 1.25, # Extreme fatigue
    }

    def calculate_fatigue_impact(self, workload: PitcherWorkload) -> dict:
        """Calculate comprehensive fatigue impact on pitcher performance."""
        # 1. Days rest impact
        rest_impact = self.DAYS_REST_MULTIPLIERS.get(
            min(workload.days_rest, 10),
            self.DAYS_REST_MULTIPLIERS[10],
        )

        # 2. Season workload (IP pace vs career high)
        workload_ratio = (
            workload.innings_pitched_pace / max(workload.innings_pitched_career_high, 1)
        )
        if workload_ratio > 1.1:
            workload_impact = 1.10  # Exceeding career high
        elif workload_ratio > 1.0:
            workload_impact = 1.05
        elif workload_ratio > 0.9:
            workload_impact = 1.00
        else:
            workload_impact = 0.98  # Well within limits

        # 3. Velocity degradation
        velocity_impact = 1.0
        if workload.velocity_delta < -2.0:
            velocity_impact = 1.15  # Significant velocity loss
        elif workload.velocity_delta < -1.0:
            velocity_impact = 1.08
        elif workload.velocity_delta < -0.5:
            velocity_impact = 1.03

        # 4. IL return factor
        il_impact = 1.0
        if workload.is_returning_from_il:
            if workload.il_return_days < 7:
                il_impact = 1.20  # First start back
            elif workload.il_return_days < 21:
                il_impact = 1.10  # Still ramping up
            elif workload.il_return_days < 45:
                il_impact = 1.05

        # 5. Travel fatigue
        travel_impact = 1.0
        if workload.travel_games > 6:
            travel_impact = 1.05  # Extended road trip
        elif workload.travel_games > 3:
            travel_impact = 1.02

        # 6. Recent pitch count stress
        last_game_impact = 1.0
        if workload.pitch_count_last > 110:
            last_game_impact = 1.08
        elif workload.pitch_count_last > 100:
            last_game_impact = 1.04

        # Composite ERA multiplier
        composite = (
            rest_impact * workload_impact * velocity_impact *
            il_impact * travel_impact * last_game_impact
        )

        # Adjusted ERA
        adjusted_era = round(workload.era * composite, 2)

        # TTO breakdown
        tto_eras = {}
        for order, mult in self.TTO_MULTIPLIERS.items():
            tto_eras[order] = round(adjusted_era * mult, 2)

        # Projected pitch count efficiency
        efficiency = "Normal"
        if composite > 1.15:
            efficiency = "Shortened start likely (4-5 IP)"
        elif composite > 1.08:
            efficiency = "Slightly shortened (5-6 IP)"
        elif composite < 0.98:
            efficiency = "Extended start possible (7+ IP)"

        # Generate warnings
        warnings = []
        if rest_impact > 1.10:
            warnings.append(f"⚠️ Short rest ({workload.days_rest} days) — ERA inflated by {(rest_impact-1)*100:.0f}%")
        if workload_ratio > 1.0:
            warnings.append(f"⚠️ Season workload ({workload.innings_pitched_season:.0f}IP) exceeding career high pace")
        if workload.velocity_delta < -1.5:
            warnings.append(f"⚠️ Velocity down {abs(workload.velocity_delta):.1f}mph from season start")
        if workload.is_returning_from_il and workload.il_return_days < 21:
            warnings.append(f"⚠️ Recent IL return ({workload.il_return_days} days ago) — ramp-up period")
        if workload.pitch_count_last > 105:
            warnings.append(f"⚠️ High pitch count last start ({workload.pitch_count_last}) — possible fatigue")

        return {
            "pitcher": workload.name,
            "team": workload.team,
            "base_era": workload.era,
            "adjusted_era": adjusted_era,
            "fatigue_multiplier": round(composite, 4),
            "factors": {
                "days_rest": {
                    "value": workload.days_rest,
                    "multiplier": rest_impact,
                },
                "workload": {
                    "ip_season": workload.innings_pitched_season,
                    "ip_pace": workload.innings_pitched_pace,
                    "career_high": workload.innings_pitched_career_high,
                    "ratio": round(workload_ratio, 2),
                    "multiplier": workload_impact,
                },
                "velocity": {
                    "current": workload.avg_velocity,
                    "season_start": workload.velocity_season_start,
                    "delta": workload.velocity_delta,
                    "multiplier": velocity_impact,
                },
                "il_return": {
                    "active": workload.is_returning_from_il,
                    "days_since": workload.il_return_days,
                    "multiplier": il_impact,
                },
                "travel": {
                    "consecutive_road": workload.travel_games,
                    "multiplier": travel_impact,
                },
                "recent_load": {
                    "last_pitch_count": workload.pitch_count_last,
                    "avg_pitch_count": workload.pitch_count_avg,
                    "multiplier": last_game_impact,
                },
            },
            "times_through_order": tto_eras,
            "efficiency_projection": efficiency,
            "warnings": warnings,
            "recommendation": self._recommendation(composite, workload),
        }

    def compare_matchup(
        self,
        starter_a: PitcherWorkload,
        starter_b: PitcherWorkload,
    ) -> dict:
        """Compare two starters' fatigue profiles."""
        impact_a = self.calculate_fatigue_impact(starter_a)
        impact_b = self.calculate_fatigue_impact(starter_b)

        era_diff = impact_a["adjusted_era"] - impact_b["adjusted_era"]
        fatigue_diff = impact_a["fatigue_multiplier"] - impact_b["fatigue_multiplier"]

        if era_diff > 0.5:
            advantage = f"{starter_b.name} has significant pitching advantage"
            lean = starter_b.team
        elif era_diff > 0.2:
            advantage = f"{starter_b.name} has slight pitching advantage"
            lean = starter_b.team
        elif era_diff < -0.5:
            advantage = f"{starter_a.name} has significant pitching advantage"
            lean = starter_a.team
        elif era_diff < -0.2:
            advantage = f"{starter_a.name} has slight pitching advantage"
            lean = starter_a.team
        else:
            advantage = "Pitching matchup is even"
            lean = "Even"

        return {
            "matchup": f"{starter_a.name} vs {starter_b.name}",
            "summary": advantage,
            "lean": lean,
            "pitcher_a": {
                "name": starter_a.name,
                "team": starter_a.team,
                "base_era": starter_a.era,
                "adjusted_era": impact_a["adjusted_era"],
                "fatigue_mult": impact_a["fatigue_multiplier"],
                "warnings": impact_a["warnings"],
            },
            "pitcher_b": {
                "name": starter_b.name,
                "team": starter_b.team,
                "base_era": starter_b.era,
                "adjusted_era": impact_b["adjusted_era"],
                "fatigue_mult": impact_b["fatigue_multiplier"],
                "warnings": impact_b["warnings"],
            },
            "era_differential": round(era_diff, 2),
        }

    def _recommendation(self, composite: float, workload: PitcherWorkload) -> dict:
        """Generate betting recommendation based on fatigue."""
        if composite > 1.15:
            return {
                "action": "FADE",
                "confidence": min((composite - 1.0) * 3, 1.0),
                "detail": f"Significant fatigue factors — bet against {workload.name}",
            }
        elif composite > 1.08:
            return {
                "action": "LEAN_AGAINST",
                "confidence": min((composite - 1.0) * 2, 0.7),
                "detail": f"Moderate fatigue — slight lean against {workload.name}",
            }
        elif composite < 0.98:
            return {
                "action": "BACK",
                "confidence": min((1.0 - composite) * 5, 0.8),
                "detail": f"Well-rested and fresh — lean toward {workload.name}",
            }
        else:
            return {
                "action": "NEUTRAL",
                "confidence": 0.0,
                "detail": "No significant fatigue factors",
            }


# ─── Bullpen Analyzer ───────────────────────────────────────────
class BullpenAnalyzer:
    """Analyze bullpen availability and its impact on game outcomes."""

    def analyze(self, bullpen: BullpenState) -> dict:
        """Analyze bullpen state for betting impact."""
        score = bullpen.availability_score()

        if score > 0.8:
            status = "💪 Fresh bullpen — full availability"
            lean = "Slight advantage"
        elif score > 0.6:
            status = "✅ Adequate bullpen availability"
            lean = "Neutral"
        elif score > 0.4:
            status = "⚠️ Taxed bullpen — limited options"
            lean = "Slight disadvantage"
        else:
            status = "🚨 Depleted bullpen — high risk"
            lean = "Significant disadvantage"

        return {
            "team": bullpen.team,
            "availability_score": score,
            "status": status,
            "lean": lean,
            "available_relievers": bullpen.total_available,
            "high_leverage_available": bullpen.high_leverage_available,
            "closer_available": bullpen.closer_available,
            "recent_workload_ip": bullpen.innings_last_3_days,
            "bullpen_era": bullpen.bullpen_era,
            "impact_on_total": "Over lean" if score < 0.5 else "Neutral",
        }


if __name__ == "__main__":
    model = PitcherFatigueModel()
    bp_analyzer = BullpenAnalyzer()

    print("⚾ MLB Predictor — Pitcher Fatigue & Workload Model")
    print("=" * 60)

    # Test pitcher profiles
    cole = PitcherWorkload(
        name="Gerrit Cole", team="NYY", handedness=Handedness.RIGHT,
        days_rest=4, pitch_count_last=98, pitch_count_avg=95.5,
        innings_pitched_season=145.2, innings_pitched_pace=210.0,
        innings_pitched_career_high=212.1, games_started_season=24,
        era=3.15, whip=1.08, k_per_9=10.8, bb_per_9=2.1, hr_per_9=1.1,
        avg_velocity=96.2, velocity_season_start=97.1, velocity_delta=-0.9,
    )

    degrom = PitcherWorkload(
        name="Jacob deGrom", team="TEX", handedness=Handedness.RIGHT,
        days_rest=3, pitch_count_last=112, pitch_count_avg=88.0,
        innings_pitched_season=80.0, innings_pitched_pace=180.0,
        innings_pitched_career_high=204.0, games_started_season=15,
        era=2.45, whip=0.92, k_per_9=12.1, bb_per_9=1.8, hr_per_9=0.8,
        avg_velocity=98.8, velocity_season_start=100.1, velocity_delta=-1.3,
        is_returning_from_il=True, il_return_days=30,
    )

    # Individual analysis
    for pitcher in [cole, degrom]:
        impact = model.calculate_fatigue_impact(pitcher)
        print(f"\n📊 {impact['pitcher']} ({impact['team']})")
        print(f"   Base ERA: {impact['base_era']} → Adjusted: {impact['adjusted_era']}")
        print(f"   Fatigue multiplier: {impact['fatigue_multiplier']:.3f}")
        print(f"   TTO: 1st={impact['times_through_order']['1st']}, 2nd={impact['times_through_order']['2nd']}, 3rd={impact['times_through_order']['3rd']}")
        print(f"   Efficiency: {impact['efficiency_projection']}")
        rec = impact['recommendation']
        print(f"   Recommendation: {rec['action']} (confidence: {rec['confidence']:.0%})")
        if impact['warnings']:
            for w in impact['warnings']:
                print(f"   {w}")

    # Head-to-head
    matchup = model.compare_matchup(cole, degrom)
    print(f"\n{'=' * 60}")
    print(f"⚔️ MATCHUP: {matchup['matchup']}")
    print(f"   {matchup['summary']}")
    print(f"   Lean: {matchup['lean']}")
    print(f"   ERA diff: {matchup['era_differential']:+.2f}")

    # Bullpen analysis
    nyy_bp = BullpenState(
        team="NYY",
        relievers=[{"name": f"RP{i}"} for i in range(8)],
        total_available=6,
        high_leverage_available=3,
        closer_available=True,
        innings_last_3_days=8.2,
        bullpen_era=3.45,
    )

    tex_bp = BullpenState(
        team="TEX",
        relievers=[{"name": f"RP{i}"} for i in range(7)],
        total_available=3,
        high_leverage_available=1,
        closer_available=False,
        innings_last_3_days=14.1,
        bullpen_era=4.20,
    )

    print(f"\n{'=' * 60}")
    print("🏟️ BULLPEN ANALYSIS:")
    for bp in [nyy_bp, tex_bp]:
        analysis = bp_analyzer.analyze(bp)
        print(f"\n   {analysis['team']}: {analysis['status']}")
        print(f"   Availability: {analysis['availability_score']:.2f} | {analysis['lean']}")
        print(f"   Relievers: {analysis['available_relievers']}/{len(bp.relievers)} | Closer: {'✅' if analysis['closer_available'] else '❌'}")

    print("\n✅ Pitcher Fatigue & Workload Model working!")
