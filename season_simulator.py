"""
MLB Predictor - Season Simulator & Playoff Predictor
Monte Carlo simulation engine for season outcomes, playoff probabilities,
division winners, wild card races, and World Series predictions.
"""

import json
import time
import uuid
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


# ──────────────────────────────────────────────
# Team Models
# ──────────────────────────────────────────────

@dataclass
class TeamProfile:
    team_id: str
    name: str
    abbreviation: str
    league: str  # AL or NL
    division: str  # East, Central, West
    # Ratings (0-100)
    overall: float = 50.0
    offense: float = 50.0
    pitching: float = 50.0
    bullpen: float = 50.0
    defense: float = 50.0
    # Current record
    wins: int = 0
    losses: int = 0
    run_differential: int = 0
    # Projections
    pythagorean_wins: float = 0.0
    elo_rating: float = 1500.0
    # Strength of schedule
    remaining_sos: float = 0.500

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.500

    @property
    def games_played(self) -> int:
        return self.wins + self.losses

    @property
    def games_remaining(self) -> int:
        return 162 - self.games_played


TEAMS = {
    # AL East
    "NYY": TeamProfile("NYY", "New York Yankees", "NYY", "AL", "East", 88, 90, 86, 82, 85, 72, 48, 95, 1585),
    "BOS": TeamProfile("BOS", "Boston Red Sox", "BOS", "AL", "East", 80, 84, 76, 74, 78, 64, 56, -12, 1545),
    "BAL": TeamProfile("BAL", "Baltimore Orioles", "BAL", "AL", "East", 85, 86, 84, 80, 82, 70, 50, 68, 1575),
    "TBR": TeamProfile("TBR", "Tampa Bay Rays", "TBR", "AL", "East", 82, 78, 85, 84, 83, 66, 54, 42, 1555),
    "TOR": TeamProfile("TOR", "Toronto Blue Jays", "TOR", "AL", "East", 78, 82, 74, 72, 76, 62, 58, -8, 1535),
    # AL Central
    "CLE": TeamProfile("CLE", "Cleveland Guardians", "CLE", "AL", "Central", 83, 76, 88, 86, 85, 68, 52, 55, 1560),
    "MIN": TeamProfile("MIN", "Minnesota Twins", "MIN", "AL", "Central", 80, 82, 78, 76, 75, 65, 55, 25, 1548),
    "DET": TeamProfile("DET", "Detroit Tigers", "DET", "AL", "Central", 76, 72, 80, 78, 79, 60, 60, -15, 1520),
    "KCR": TeamProfile("KCR", "Kansas City Royals", "KCR", "AL", "Central", 75, 74, 76, 72, 74, 58, 62, -28, 1510),
    "CWS": TeamProfile("CWS", "Chicago White Sox", "CWS", "AL", "Central", 62, 64, 60, 58, 62, 42, 78, -105, 1420),
    # AL West
    "HOU": TeamProfile("HOU", "Houston Astros", "HOU", "AL", "West", 87, 88, 86, 84, 84, 71, 49, 80, 1580),
    "TEX": TeamProfile("TEX", "Texas Rangers", "TEX", "AL", "West", 84, 85, 82, 80, 81, 68, 52, 52, 1565),
    "SEA": TeamProfile("SEA", "Seattle Mariners", "SEA", "AL", "West", 82, 76, 88, 85, 84, 67, 53, 35, 1555),
    "LAA": TeamProfile("LAA", "Los Angeles Angels", "LAA", "AL", "West", 74, 80, 68, 66, 72, 56, 64, -35, 1500),
    "OAK": TeamProfile("OAK", "Oakland Athletics", "OAK", "AL", "West", 60, 58, 62, 60, 60, 40, 80, -120, 1400),
    # NL East
    "ATL": TeamProfile("ATL", "Atlanta Braves", "ATL", "NL", "East", 90, 92, 88, 84, 86, 74, 46, 110, 1595),
    "PHI": TeamProfile("PHI", "Philadelphia Phillies", "PHI", "NL", "East", 86, 88, 84, 82, 80, 70, 50, 72, 1578),
    "NYM": TeamProfile("NYM", "New York Mets", "NYM", "NL", "East", 82, 84, 80, 76, 78, 66, 54, 35, 1555),
    "MIA": TeamProfile("MIA", "Miami Marlins", "MIA", "NL", "East", 72, 68, 76, 74, 75, 55, 65, -40, 1490),
    "WSN": TeamProfile("WSN", "Washington Nationals", "WSN", "NL", "East", 68, 66, 70, 68, 70, 50, 70, -60, 1470),
    # NL Central
    "MIL": TeamProfile("MIL", "Milwaukee Brewers", "MIL", "NL", "Central", 84, 80, 86, 84, 83, 69, 51, 55, 1568),
    "CHC": TeamProfile("CHC", "Chicago Cubs", "CHC", "NL", "Central", 80, 82, 78, 76, 77, 64, 56, 18, 1545),
    "STL": TeamProfile("STL", "St. Louis Cardinals", "STL", "NL", "Central", 78, 76, 80, 78, 79, 62, 58, 5, 1535),
    "PIT": TeamProfile("PIT", "Pittsburgh Pirates", "PIT", "NL", "Central", 72, 70, 74, 72, 73, 55, 65, -30, 1495),
    "CIN": TeamProfile("CIN", "Cincinnati Reds", "CIN", "NL", "Central", 76, 78, 74, 70, 72, 58, 62, -10, 1515),
    # NL West
    "LAD": TeamProfile("LAD", "Los Angeles Dodgers", "LAD", "NL", "West", 92, 94, 90, 88, 88, 76, 44, 125, 1605),
    "SDP": TeamProfile("SDP", "San Diego Padres", "SDP", "NL", "West", 84, 82, 86, 82, 82, 68, 52, 48, 1565),
    "ARI": TeamProfile("ARI", "Arizona Diamondbacks", "ARI", "NL", "West", 82, 80, 82, 80, 80, 66, 54, 30, 1552),
    "SFG": TeamProfile("SFG", "San Francisco Giants", "SFG", "NL", "West", 76, 74, 78, 76, 78, 60, 60, -15, 1520),
    "COL": TeamProfile("COL", "Colorado Rockies", "COL", "NL", "West", 64, 70, 58, 56, 62, 44, 76, -85, 1430),
}


# ──────────────────────────────────────────────
# Simulation Engine
# ──────────────────────────────────────────────

class SimulationEngine:
    """Monte Carlo season simulation"""

    def __init__(self, teams: Dict[str, TeamProfile] = None):
        self.teams = teams or TEAMS

    def simulate_game(self, home: TeamProfile, away: TeamProfile) -> Tuple[str, str]:
        """Simulate a single game, return (winner_id, loser_id)"""
        # Base win probability from Elo-like rating
        home_advantage = 0.04  # ~54% home win rate in MLB
        rating_diff = (home.elo_rating - away.elo_rating) / 400
        home_win_prob = 1 / (1 + 10 ** (-rating_diff)) + home_advantage

        # Adjust for team strengths
        off_diff = (home.offense - away.pitching) / 200
        pitch_diff = (home.pitching - away.offense) / 200
        home_win_prob += (off_diff + pitch_diff) / 2

        # Clamp
        home_win_prob = max(0.25, min(0.75, home_win_prob))

        if random.random() < home_win_prob:
            return home.team_id, away.team_id
        return away.team_id, home.team_id

    def simulate_remaining_season(self, n_sims: int = 10000) -> Dict:
        """Run Monte Carlo simulation for remaining games"""
        division_winners = defaultdict(int)
        wild_card_teams = defaultdict(int)
        playoff_teams = defaultdict(int)
        pennant_winners = defaultdict(int)
        ws_winners = defaultdict(int)
        projected_wins = defaultdict(list)

        for _ in range(n_sims):
            # Copy current records
            sim_records = {}
            for tid, team in self.teams.items():
                sim_records[tid] = {"wins": team.wins, "losses": team.losses}

            # Simulate remaining games (simplified: each team plays remaining against random opponents)
            for tid, team in self.teams.items():
                remaining = team.games_remaining
                for _ in range(remaining):
                    # Pick random opponent
                    opp_id = random.choice([t for t in self.teams if t != tid])
                    opp = self.teams[opp_id]

                    # Home/away random
                    if random.random() < 0.5:
                        winner, _ = self.simulate_game(team, opp)
                    else:
                        winner, _ = self.simulate_game(opp, team)

                    if winner == tid:
                        sim_records[tid]["wins"] += 1
                    else:
                        sim_records[tid]["losses"] += 1

            # Record projected wins
            for tid in self.teams:
                projected_wins[tid].append(sim_records[tid]["wins"])

            # Determine division winners and wild cards
            for league in ["AL", "NL"]:
                league_teams = [(tid, sim_records[tid]) for tid, t in self.teams.items() if t.league == league]

                for division in ["East", "Central", "West"]:
                    div_teams = [(tid, rec) for tid, rec in league_teams
                                 if self.teams[tid].division == division]
                    div_winner = max(div_teams, key=lambda x: x[1]["wins"])
                    division_winners[div_winner[0]] += 1
                    playoff_teams[div_winner[0]] += 1

                # Wild cards (top 3 non-division winners)
                div_winner_ids = set()
                for division in ["East", "Central", "West"]:
                    div_teams = [(tid, rec) for tid, rec in league_teams
                                 if self.teams[tid].division == division]
                    div_winner = max(div_teams, key=lambda x: x[1]["wins"])
                    div_winner_ids.add(div_winner[0])

                non_winners = [(tid, rec) for tid, rec in league_teams if tid not in div_winner_ids]
                non_winners.sort(key=lambda x: -x[1]["wins"])
                for tid, _ in non_winners[:3]:
                    wild_card_teams[tid] += 1
                    playoff_teams[tid] += 1

            # Simulate playoffs (simplified)
            for league in ["AL", "NL"]:
                league_playoff = [(tid, sim_records[tid]) for tid, t in self.teams.items()
                                  if t.league == league and tid in playoff_teams and playoff_teams[tid] > 0]
                league_playoff.sort(key=lambda x: -x[1]["wins"])

                if len(league_playoff) >= 4:
                    # ALCS/NLCS (top 2 vs next 2, simplified)
                    semifinal_winners = []
                    matchups = [(league_playoff[0][0], league_playoff[3][0]),
                                (league_playoff[1][0], league_playoff[2][0])]
                    for home_id, away_id in matchups:
                        home = self.teams[home_id]
                        away = self.teams[away_id]
                        wins_needed = 4
                        h_wins = a_wins = 0
                        for _ in range(7):
                            winner, _ = self.simulate_game(home, away)
                            if winner == home_id:
                                h_wins += 1
                            else:
                                a_wins += 1
                            if h_wins >= wins_needed or a_wins >= wins_needed:
                                break
                        semifinal_winners.append(home_id if h_wins > a_wins else away_id)

                    # Championship series
                    if len(semifinal_winners) >= 2:
                        h = self.teams[semifinal_winners[0]]
                        a = self.teams[semifinal_winners[1]]
                        h_w = a_w = 0
                        for _ in range(7):
                            w, _ = self.simulate_game(h, a)
                            if w == semifinal_winners[0]:
                                h_w += 1
                            else:
                                a_w += 1
                            if h_w >= 4 or a_w >= 4:
                                break
                        pennant_winner = semifinal_winners[0] if h_w > a_w else semifinal_winners[1]
                        pennant_winners[pennant_winner] += 1

            # World Series
            al_champ = max(
                [(tid, pennant_winners.get(tid, 0)) for tid, t in self.teams.items() if t.league == "AL"],
                key=lambda x: x[1], default=("", 0)
            )[0]
            nl_champ = max(
                [(tid, pennant_winners.get(tid, 0)) for tid, t in self.teams.items() if t.league == "NL"],
                key=lambda x: x[1], default=("", 0)
            )[0]

            if al_champ and nl_champ:
                h = self.teams[al_champ]
                a = self.teams[nl_champ]
                h_w = a_w = 0
                for _ in range(7):
                    w, _ = self.simulate_game(h, a)
                    if w == al_champ:
                        h_w += 1
                    else:
                        a_w += 1
                    if h_w >= 4 or a_w >= 4:
                        break
                ws_winner = al_champ if h_w > a_w else nl_champ
                ws_winners[ws_winner] += 1

        # Compile results
        results = {}
        for tid, team in self.teams.items():
            wins_list = projected_wins.get(tid, [])
            results[tid] = {
                "team": team.name,
                "abbreviation": team.abbreviation,
                "league": team.league,
                "division": team.division,
                "current_record": f"{team.wins}-{team.losses}",
                "projected_wins": round(sum(wins_list) / max(1, len(wins_list)), 1) if wins_list else team.wins,
                "win_range": f"{min(wins_list, default=0)}-{max(wins_list, default=0)}",
                "division_pct": round(division_winners.get(tid, 0) / n_sims * 100, 1),
                "wild_card_pct": round(wild_card_teams.get(tid, 0) / n_sims * 100, 1),
                "playoff_pct": round(playoff_teams.get(tid, 0) / n_sims * 100, 1),
                "pennant_pct": round(pennant_winners.get(tid, 0) / n_sims * 100, 1),
                "ws_pct": round(ws_winners.get(tid, 0) / n_sims * 100, 1),
            }

        return {
            "simulations": n_sims,
            "teams": dict(sorted(results.items(), key=lambda x: -x[1]["projected_wins"])),
            "division_favorites": self._get_favorites(results, "division_pct"),
            "ws_favorites": self._get_favorites(results, "ws_pct", limit=10),
            "generated_at": datetime.now().isoformat(),
        }

    def _get_favorites(self, results: Dict, key: str, limit: int = 6) -> List[Dict]:
        sorted_teams = sorted(results.items(), key=lambda x: -x[1][key])
        return [{"team": v["team"], "abbreviation": v["abbreviation"], key: v[key]}
                for _, v in sorted_teams[:limit] if v[key] > 0]


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("MLB Predictor - Season Simulator & Playoff Predictor")
    print("=" * 60)

    engine = SimulationEngine()

    # Run simulation (fewer for demo speed)
    print(f"\n🎲 Running 1,000 season simulations...")
    results = engine.simulate_remaining_season(n_sims=1000)
    print(f"  ✅ Complete!")

    # Division favorites
    print(f"\n🏆 Division Favorites:")
    for fav in results["division_favorites"]:
        print(f"  {fav['abbreviation']:4s} {fav['team']:28s} {fav['division_pct']:5.1f}%")

    # World Series favorites
    print(f"\n🏆 World Series Favorites:")
    for i, fav in enumerate(results["ws_favorites"], 1):
        print(f"  {i}. {fav['abbreviation']:4s} {fav['team']:28s} {fav['ws_pct']:5.1f}%")

    # Full standings projection
    print(f"\n📊 Projected Standings:")
    for league in ["AL", "NL"]:
        print(f"\n  {'='*60}")
        print(f"  {league}")
        for division in ["East", "Central", "West"]:
            print(f"\n  {league} {division}:")
            div_teams = [
                (tid, data) for tid, data in results["teams"].items()
                if data["league"] == league and data["division"] == division
            ]
            div_teams.sort(key=lambda x: -x[1]["projected_wins"])
            for tid, data in div_teams:
                print(f"    {data['abbreviation']:4s} {data['projected_wins']:5.1f}W "
                      f"| Div: {data['division_pct']:5.1f}% | PO: {data['playoff_pct']:5.1f}% "
                      f"| WS: {data['ws_pct']:4.1f}%")

    print(f"\n✅ Season Simulator ready!")
    print("  • 30 MLB team profiles with ratings")
    print("  • Monte Carlo season simulation (10K+ sims)")
    print("  • Division winner probabilities")
    print("  • Wild card race tracking")
    print("  • Pennant & World Series predictions")
    print("  • Projected win totals with ranges")
    print("  • Elo-based game simulation")


if __name__ == "__main__":
    demo()
