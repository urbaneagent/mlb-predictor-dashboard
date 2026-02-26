"""
MLB Predictor - Stadium Factors Engine
Comprehensive park factor analysis including dimensions, altitude,
wind patterns, roof status, and their impact on scoring/home runs.
"""
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StadiumProfile:
    """Comprehensive stadium profile with park factors."""
    name: str
    team: str
    city: str
    state: str
    capacity: int = 0
    year_opened: int = 0
    surface: str = "grass"  # grass, turf
    roof: str = "open"  # open, retractable, dome
    altitude_feet: int = 0
    # Park factors (100 = league average)
    run_factor: float = 100.0
    hr_factor: float = 100.0
    doubles_factor: float = 100.0
    triples_factor: float = 100.0
    bb_factor: float = 100.0
    so_factor: float = 100.0
    # Dimensions
    lf_distance: int = 0   # left field
    lcf_distance: int = 0  # left-center
    cf_distance: int = 0   # center field
    rcf_distance: int = 0  # right-center
    rf_distance: int = 0   # right field
    wall_height_lf: int = 0
    wall_height_cf: int = 0
    wall_height_rf: int = 0
    # Wind
    avg_wind_speed_mph: float = 0.0
    predominant_wind_dir: str = ""  # "in", "out", "cross_LR", "cross_RL"
    # Splits
    lhb_hr_factor: float = 100.0  # left-handed batter HR factor
    rhb_hr_factor: float = 100.0  # right-handed batter HR factor
    day_run_factor: float = 100.0
    night_run_factor: float = 100.0


# ─── STADIUM DATABASE (2025 season) ─────────────────────────

STADIUMS = {
    "COL": StadiumProfile(
        name="Coors Field", team="Colorado Rockies",
        city="Denver", state="CO", capacity=50144, year_opened=1995,
        surface="grass", roof="open", altitude_feet=5280,
        run_factor=118, hr_factor=114, doubles_factor=125,
        triples_factor=145, bb_factor=102, so_factor=94,
        lf_distance=347, lcf_distance=390, cf_distance=415,
        rcf_distance=375, rf_distance=350,
        wall_height_lf=8, wall_height_cf=8, wall_height_rf=14,
        avg_wind_speed_mph=6.5, predominant_wind_dir="out",
        lhb_hr_factor=118, rhb_hr_factor=112,
        day_run_factor=122, night_run_factor=115,
    ),
    "CIN": StadiumProfile(
        name="Great American Ball Park", team="Cincinnati Reds",
        city="Cincinnati", state="OH", capacity=42319, year_opened=2003,
        surface="grass", roof="open", altitude_feet=490,
        run_factor=110, hr_factor=115, doubles_factor=105,
        triples_factor=95, bb_factor=100, so_factor=98,
        lf_distance=328, lcf_distance=379, cf_distance=404,
        rcf_distance=370, rf_distance=325,
        wall_height_lf=12, wall_height_cf=8, wall_height_rf=8,
        avg_wind_speed_mph=7.0, predominant_wind_dir="out",
        lhb_hr_factor=108, rhb_hr_factor=120,
        day_run_factor=112, night_run_factor=108,
    ),
    "TEX": StadiumProfile(
        name="Globe Life Field", team="Texas Rangers",
        city="Arlington", state="TX", capacity=40300, year_opened=2020,
        surface="turf", roof="retractable", altitude_feet=600,
        run_factor=105, hr_factor=108, doubles_factor=102,
        triples_factor=90, bb_factor=100, so_factor=100,
        lf_distance=329, lcf_distance=372, cf_distance=407,
        rcf_distance=374, rf_distance=326,
        wall_height_lf=8, wall_height_cf=8, wall_height_rf=8,
        avg_wind_speed_mph=3.0, predominant_wind_dir="cross_LR",
        lhb_hr_factor=110, rhb_hr_factor=106,
        day_run_factor=103, night_run_factor=107,
    ),
    "NYY": StadiumProfile(
        name="Yankee Stadium", team="New York Yankees",
        city="Bronx", state="NY", capacity=46537, year_opened=2009,
        surface="grass", roof="open", altitude_feet=55,
        run_factor=107, hr_factor=112, doubles_factor=95,
        triples_factor=85, bb_factor=101, so_factor=99,
        lf_distance=318, lcf_distance=399, cf_distance=408,
        rcf_distance=385, rf_distance=314,
        wall_height_lf=8, wall_height_cf=8, wall_height_rf=8,
        avg_wind_speed_mph=8.5, predominant_wind_dir="cross_LR",
        lhb_hr_factor=105, rhb_hr_factor=118,
        day_run_factor=110, night_run_factor=105,
    ),
    "BOS": StadiumProfile(
        name="Fenway Park", team="Boston Red Sox",
        city="Boston", state="MA", capacity=37755, year_opened=1912,
        surface="grass", roof="open", altitude_feet=20,
        run_factor=106, hr_factor=96, doubles_factor=130,
        triples_factor=60, bb_factor=100, so_factor=100,
        lf_distance=310, lcf_distance=379, cf_distance=390,
        rcf_distance=380, rf_distance=302,
        wall_height_lf=37, wall_height_cf=17, wall_height_rf=5,
        avg_wind_speed_mph=9.0, predominant_wind_dir="in",
        lhb_hr_factor=92, rhb_hr_factor=105,
        day_run_factor=108, night_run_factor=104,
    ),
    "CHC": StadiumProfile(
        name="Wrigley Field", team="Chicago Cubs",
        city="Chicago", state="IL", capacity=41649, year_opened=1914,
        surface="grass", roof="open", altitude_feet=600,
        run_factor=105, hr_factor=104, doubles_factor=108,
        triples_factor=90, bb_factor=100, so_factor=99,
        lf_distance=355, lcf_distance=368, cf_distance=400,
        rcf_distance=368, rf_distance=353,
        wall_height_lf=15, wall_height_cf=11, wall_height_rf=15,
        avg_wind_speed_mph=11.0, predominant_wind_dir="variable",
        lhb_hr_factor=102, rhb_hr_factor=106,
        day_run_factor=110, night_run_factor=100,
    ),
    "SF": StadiumProfile(
        name="Oracle Park", team="San Francisco Giants",
        city="San Francisco", state="CA", capacity=41915, year_opened=2000,
        surface="grass", roof="open", altitude_feet=5,
        run_factor=91, hr_factor=85, doubles_factor=98,
        triples_factor=120, bb_factor=100, so_factor=101,
        lf_distance=339, lcf_distance=364, cf_distance=399,
        rcf_distance=365, rf_distance=309,
        wall_height_lf=8, wall_height_cf=8, wall_height_rf=24,
        avg_wind_speed_mph=12.0, predominant_wind_dir="in",
        lhb_hr_factor=82, rhb_hr_factor=88,
        day_run_factor=88, night_run_factor=93,
    ),
    "LAD": StadiumProfile(
        name="Dodger Stadium", team="Los Angeles Dodgers",
        city="Los Angeles", state="CA", capacity=56000, year_opened=1962,
        surface="grass", roof="open", altitude_feet=515,
        run_factor=96, hr_factor=94, doubles_factor=98,
        triples_factor=105, bb_factor=100, so_factor=100,
        lf_distance=330, lcf_distance=385, cf_distance=395,
        rcf_distance=385, rf_distance=330,
        wall_height_lf=8, wall_height_cf=8, wall_height_rf=8,
        avg_wind_speed_mph=5.0, predominant_wind_dir="in",
        lhb_hr_factor=95, rhb_hr_factor=93,
        day_run_factor=94, night_run_factor=97,
    ),
    "HOU": StadiumProfile(
        name="Minute Maid Park", team="Houston Astros",
        city="Houston", state="TX", capacity=41168, year_opened=2000,
        surface="grass", roof="retractable", altitude_feet=40,
        run_factor=104, hr_factor=105, doubles_factor=105,
        triples_factor=95, bb_factor=100, so_factor=100,
        lf_distance=315, lcf_distance=362, cf_distance=409,
        rcf_distance=373, rf_distance=326,
        wall_height_lf=19, wall_height_cf=8, wall_height_rf=8,
        avg_wind_speed_mph=2.0, predominant_wind_dir="none",
        lhb_hr_factor=100, rhb_hr_factor=110,
        day_run_factor=102, night_run_factor=105,
    ),
    "MIA": StadiumProfile(
        name="LoanDepot Park", team="Miami Marlins",
        city="Miami", state="FL", capacity=36742, year_opened=2012,
        surface="grass", roof="retractable", altitude_feet=10,
        run_factor=90, hr_factor=87, doubles_factor=95,
        triples_factor=100, bb_factor=100, so_factor=101,
        lf_distance=344, lcf_distance=386, cf_distance=400,
        rcf_distance=392, rf_distance=335,
        wall_height_lf=7, wall_height_cf=7, wall_height_rf=7,
        avg_wind_speed_mph=1.5, predominant_wind_dir="none",
        lhb_hr_factor=85, rhb_hr_factor=88,
        day_run_factor=88, night_run_factor=91,
    ),
}


class StadiumFactorsEngine:
    """
    Analyzes how stadium characteristics impact game outcomes.
    """

    # League average park factor
    LEAGUE_AVG = 100

    def __init__(self):
        self.stadiums = STADIUMS

    def get_stadium(self, team_code: str) -> Optional[StadiumProfile]:
        """Get stadium by team code."""
        return self.stadiums.get(team_code.upper())

    def get_run_adjustment(self, team_code: str,
                            time_of_day: str = "night",
                            wind_condition: Optional[str] = None) -> dict:
        """
        Calculate run total adjustments for a game at this stadium.
        Returns adjustment factors for prediction models.
        """
        stadium = self.get_stadium(team_code)
        if not stadium:
            return {"error": f"Stadium not found for {team_code}"}

        # Base run factor
        base_factor = stadium.run_factor / self.LEAGUE_AVG

        # Time of day adjustment
        if time_of_day == "day":
            time_factor = stadium.day_run_factor / self.LEAGUE_AVG
        else:
            time_factor = stadium.night_run_factor / self.LEAGUE_AVG

        # Altitude factor (every 1000ft adds ~5% to run scoring)
        altitude_factor = 1.0 + (stadium.altitude_feet / 1000 * 0.005)

        # Wind adjustment
        wind_factor = 1.0
        if wind_condition:
            if wind_condition == "out" and stadium.avg_wind_speed_mph > 8:
                wind_factor = 1.05
            elif wind_condition == "in" and stadium.avg_wind_speed_mph > 8:
                wind_factor = 0.95

        # Composite
        composite = base_factor * altitude_factor * wind_factor

        return {
            "stadium": stadium.name,
            "team": team_code,
            "base_run_factor": round(base_factor, 3),
            "time_factor": round(time_factor, 3),
            "altitude_factor": round(altitude_factor, 3),
            "wind_factor": round(wind_factor, 3),
            "composite_factor": round(composite, 3),
            "run_adjustment_pct": round((composite - 1) * 100, 1),
            "predicted_impact": (
                f"{'+'if composite > 1 else ''}"
                f"{(composite - 1) * 8.5:.1f} runs vs. average"
            ),
            "hr_factor": stadium.hr_factor,
            "is_hitter_park": stadium.run_factor > 103,
            "is_pitcher_park": stadium.run_factor < 97,
        }

    def get_hr_advantage(self, team_code: str,
                          batter_hand: str = "R") -> dict:
        """Get home run advantage/disadvantage for a batter handedness."""
        stadium = self.get_stadium(team_code)
        if not stadium:
            return {"error": f"Stadium not found for {team_code}"}

        if batter_hand.upper() == "L":
            hr_factor = stadium.lhb_hr_factor
            target_wall = "right field"
            distance = stadium.rf_distance
            wall_height = stadium.wall_height_rf
        else:
            hr_factor = stadium.rhb_hr_factor
            target_wall = "left field"
            distance = stadium.lf_distance
            wall_height = stadium.wall_height_lf

        return {
            "stadium": stadium.name,
            "batter_hand": batter_hand.upper(),
            "hr_factor": hr_factor,
            "hr_advantage": round((hr_factor / 100 - 1) * 100, 1),
            "target_wall": target_wall,
            "target_distance": distance,
            "wall_height": wall_height,
            "assessment": (
                "HR-FRIENDLY" if hr_factor >= 110 else
                "NEUTRAL" if hr_factor >= 95 else
                "HR-SUPPRESSING"
            ),
        }

    def rank_parks(self, metric: str = "run_factor") -> list:
        """Rank all parks by a given metric."""
        parks = list(self.stadiums.items())
        parks.sort(key=lambda x: getattr(x[1], metric, 100), reverse=True)

        return [
            {
                "rank": i + 1,
                "team": code,
                "stadium": park.name,
                metric: getattr(park, metric),
                "assessment": (
                    "HITTER-FRIENDLY" if getattr(park, metric) > 103 else
                    "NEUTRAL" if getattr(park, metric) >= 97 else
                    "PITCHER-FRIENDLY"
                ),
            }
            for i, (code, park) in enumerate(parks)
        ]

    def get_matchup_factors(self, home_team: str, away_team: str,
                             time_of_day: str = "night") -> dict:
        """Get comprehensive park factors for a specific matchup."""
        stadium = self.get_stadium(home_team)
        if not stadium:
            return {"error": f"Stadium not found for {home_team}"}

        run_adj = self.get_run_adjustment(home_team, time_of_day)
        lhb_hr = self.get_hr_advantage(home_team, "L")
        rhb_hr = self.get_hr_advantage(home_team, "R")

        return {
            "matchup": f"{away_team} @ {home_team}",
            "stadium": stadium.name,
            "surface": stadium.surface,
            "roof": stadium.roof,
            "altitude": stadium.altitude_feet,
            "avg_wind": f"{stadium.avg_wind_speed_mph} mph {stadium.predominant_wind_dir}",
            "run_adjustment": run_adj,
            "lhb_hr_analysis": lhb_hr,
            "rhb_hr_analysis": rhb_hr,
            "key_factors": self._get_key_factors(stadium),
            "betting_implications": self._get_betting_implications(stadium),
        }

    def _get_key_factors(self, stadium: StadiumProfile) -> list:
        """Get the most notable factors about a stadium."""
        factors = []

        if stadium.altitude_feet > 1000:
            factors.append(
                f"🏔️ High altitude ({stadium.altitude_feet}ft) — "
                f"ball carries further, expect more runs"
            )

        if stadium.avg_wind_speed_mph > 10:
            factors.append(
                f"💨 High wind ({stadium.avg_wind_speed_mph}mph {stadium.predominant_wind_dir}) — "
                f"{'favors hitters' if stadium.predominant_wind_dir == 'out' else 'suppresses scoring'}"
            )

        if max(stadium.wall_height_lf, stadium.wall_height_cf,
               stadium.wall_height_rf) > 15:
            factors.append(
                f"🧱 Tall wall ({max(stadium.wall_height_lf, stadium.wall_height_cf, stadium.wall_height_rf)}ft) — "
                f"suppresses home runs, more doubles"
            )

        if stadium.doubles_factor > 115:
            factors.append(
                f"2️⃣ Doubles-heavy park (factor: {stadium.doubles_factor}) — "
                f"favor teams with speed"
            )

        if stadium.roof == "retractable":
            factors.append(
                "🏠 Retractable roof — check weather; closed roof reduces wind impact"
            )

        if stadium.surface == "turf":
            factors.append(
                "🟢 Turf surface — faster infield, more infield hits, favors speed"
            )

        return factors

    def _get_betting_implications(self, stadium: StadiumProfile) -> list:
        """Get betting-relevant implications."""
        implications = []

        if stadium.run_factor >= 108:
            implications.append("📈 LEAN OVER: High-run park, totals regularly hit overs")
        elif stadium.run_factor <= 93:
            implications.append("📉 LEAN UNDER: Low-run park, totals regularly hit unders")

        if stadium.hr_factor >= 110:
            implications.append("💣 HR PROP PLAYS: Look for HR props on power hitters")
        elif stadium.hr_factor <= 90:
            implications.append("🛡️ AVOID HR PROPS: HR-suppressing park")

        if abs(stadium.lhb_hr_factor - stadium.rhb_hr_factor) > 10:
            if stadium.lhb_hr_factor > stadium.rhb_hr_factor:
                implications.append("🔄 LEFT-HANDED ADVANTAGE: LHB HRs significantly boosted")
            else:
                implications.append("🔄 RIGHT-HANDED ADVANTAGE: RHB HRs significantly boosted")

        return implications


if __name__ == "__main__":
    engine = StadiumFactorsEngine()

    # Coors Field analysis
    coors = engine.get_run_adjustment("COL", time_of_day="day")
    print(f"🏟️ {coors['stadium']}:")
    print(f"   Run Adjustment: {coors['run_adjustment_pct']:+.1f}%")
    print(f"   Impact: {coors['predicted_impact']}")

    # Oracle Park
    oracle = engine.get_run_adjustment("SF", time_of_day="night")
    print(f"\n🏟️ {oracle['stadium']}:")
    print(f"   Run Adjustment: {oracle['run_adjustment_pct']:+.1f}%")
    print(f"   Impact: {oracle['predicted_impact']}")

    # HR advantage for LHB at Yankee Stadium
    yankee_lhb = engine.get_hr_advantage("NYY", "L")
    print(f"\n⚾ {yankee_lhb['stadium']} (LHB):")
    print(f"   HR Factor: {yankee_lhb['hr_factor']}")
    print(f"   Assessment: {yankee_lhb['assessment']}")

    # Full matchup
    matchup = engine.get_matchup_factors("COL", "LAD", "day")
    print(f"\n🎮 {matchup['matchup']} at {matchup['stadium']}:")
    for f in matchup['key_factors']:
        print(f"   {f}")
    for b in matchup['betting_implications']:
        print(f"   {b}")

    # Rankings
    print("\n📊 Parks Ranked by Run Factor:")
    for park in engine.rank_parks("run_factor"):
        print(f"   #{park['rank']} {park['team']} ({park['stadium']}): "
              f"{park['run_factor']} [{park['assessment']}]")
