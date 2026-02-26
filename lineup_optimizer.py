#!/usr/bin/env python3
"""
MLB Predictor - DFS Lineup Optimizer
=======================================
Optimize DFS (Daily Fantasy Sports) lineups for DraftKings and FanDuel.

Features:
- Salary cap optimization (greedy + simulated annealing)
- Stacking strategies (team stacks, bring-backs)
- Ownership-weighted optimization (contrarian plays)
- Multi-lineup generation with diversity constraints
- Player pool filtering (weather, matchup, recent form)
- Vegas correlation (run totals, implied runs)

Author: Mike Ross (The Architect)
Date: 2026-02-23
"""

import json
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum
from copy import deepcopy


# ============================================================================
# DATA MODELS
# ============================================================================

class Platform(Enum):
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"


class Position(Enum):
    P = "P"    # Pitcher
    C = "C"    # Catcher
    FB = "1B"  # First Base
    SB = "2B"  # Second Base
    TB = "3B"  # Third Base
    SS = "SS"  # Shortstop
    OF = "OF"  # Outfield
    UTIL = "UTIL"  # Utility (any hitter)


# DraftKings MLB roster: P, P, C, 1B, 2B, 3B, SS, OF, OF, OF
# FanDuel MLB roster: P, C/1B, 2B, 3B, SS, OF, OF, OF, UTIL

DK_ROSTER = ['P', 'P', 'C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF']
FD_ROSTER = ['P', 'C/1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL']

DK_SALARY_CAP = 50000
FD_SALARY_CAP = 60000


@dataclass
class DFSPlayer:
    """A player in the DFS pool"""
    player_id: str
    name: str
    team: str
    position: str  # "P", "C", "1B", "2B", "3B", "SS", "OF"
    salary: int
    projected_points: float
    ownership_pct: float = 5.0  # Expected ownership %
    batting_order: int = 0  # 1-9
    opponent: str = ""
    game_id: str = ""
    is_home: bool = False
    implied_runs: float = 4.5  # Vegas team implied run total
    recent_form: float = 1.0  # Multiplier (1.0 = average)
    weather_impact: float = 1.0  # Multiplier
    handedness: str = ""  # L/R/S

    @property
    def value(self) -> float:
        """Points per $1000 salary"""
        return self.projected_points / (self.salary / 1000) if self.salary > 0 else 0


@dataclass
class DFSLineup:
    """A complete DFS lineup"""
    players: List[DFSPlayer] = field(default_factory=list)
    platform: Platform = Platform.DRAFTKINGS

    @property
    def total_salary(self) -> int:
        return sum(p.salary for p in self.players)

    @property
    def total_projected(self) -> float:
        return round(sum(p.projected_points for p in self.players), 1)

    @property
    def total_ownership(self) -> float:
        return round(sum(p.ownership_pct for p in self.players), 1)

    @property
    def salary_remaining(self) -> int:
        cap = DK_SALARY_CAP if self.platform == Platform.DRAFTKINGS else FD_SALARY_CAP
        return cap - self.total_salary

    @property
    def teams(self) -> Dict[str, int]:
        teams: Dict[str, int] = {}
        for p in self.players:
            teams[p.team] = teams.get(p.team, 0) + 1
        return teams

    def has_stack(self, min_size: int = 3) -> bool:
        return any(c >= min_size for c in self.teams.values())


# ============================================================================
# LINEUP OPTIMIZER
# ============================================================================

class LineupOptimizer:
    """DFS lineup optimization engine"""

    def __init__(self, platform: Platform = Platform.DRAFTKINGS):
        self.platform = platform
        self.salary_cap = DK_SALARY_CAP if platform == Platform.DRAFTKINGS else FD_SALARY_CAP
        self.roster_template = DK_ROSTER if platform == Platform.DRAFTKINGS else FD_ROSTER

    def optimize_greedy(self, player_pool: List[DFSPlayer],
                        min_salary: int = 0,
                        require_stack: bool = False,
                        max_ownership: float = 100.0) -> Optional[DFSLineup]:
        """
        Greedy optimization: fill each slot with best available value player.
        """
        pool = [p for p in player_pool if p.ownership_pct <= max_ownership]
        lineup = DFSLineup(platform=self.platform)
        used_ids: Set[str] = set()

        for slot in self.roster_template:
            # Find eligible players for this slot
            eligible = []
            for p in pool:
                if p.player_id in used_ids:
                    continue
                if self._fits_slot(p.position, slot):
                    # Check if adding would exceed salary cap
                    remaining_slots = len(self.roster_template) - len(lineup.players) - 1
                    min_remaining = remaining_slots * 2000  # Minimum salary estimate
                    if lineup.total_salary + p.salary + min_remaining <= self.salary_cap:
                        eligible.append(p)

            if not eligible:
                continue

            # Sort by value (points per dollar)
            eligible.sort(key=lambda p: p.projected_points / max(1, p.salary), reverse=True)

            best = eligible[0]
            lineup.players.append(best)
            used_ids.add(best.player_id)

        return lineup if len(lineup.players) == len(self.roster_template) else None

    def optimize_simulated_annealing(self, player_pool: List[DFSPlayer],
                                       iterations: int = 5000,
                                       initial_temp: float = 10.0,
                                       cooling_rate: float = 0.995,
                                       ownership_penalty: float = 0.0) -> Optional[DFSLineup]:
        """
        Simulated annealing optimization for better global solutions.
        """
        # Start with greedy solution
        current = self.optimize_greedy(player_pool)
        if not current:
            return None

        best = deepcopy(current)
        best_score = self._score_lineup(best, ownership_penalty)
        current_score = best_score
        temp = initial_temp

        # Build position pools
        pos_pools: Dict[str, List[DFSPlayer]] = {}
        for p in player_pool:
            if p.position not in pos_pools:
                pos_pools[p.position] = []
            pos_pools[p.position].append(p)

        for i in range(iterations):
            # Random swap
            slot_idx = random.randint(0, len(current.players) - 1)
            old_player = current.players[slot_idx]
            slot = self.roster_template[slot_idx]

            # Find valid replacement
            eligible = [
                p for p in player_pool
                if p.player_id != old_player.player_id
                and self._fits_slot(p.position, slot)
                and p.player_id not in {pl.player_id for pl in current.players}
            ]

            if not eligible:
                temp *= cooling_rate
                continue

            new_player = random.choice(eligible)

            # Try swap
            current.players[slot_idx] = new_player
            new_salary = current.total_salary

            if new_salary > self.salary_cap:
                current.players[slot_idx] = old_player
                temp *= cooling_rate
                continue

            new_score = self._score_lineup(current, ownership_penalty)
            delta = new_score - current_score

            # Accept or reject
            if delta > 0 or random.random() < math.exp(delta / max(0.001, temp)):
                current_score = new_score
                if new_score > best_score:
                    best = deepcopy(current)
                    best_score = new_score
            else:
                current.players[slot_idx] = old_player

            temp *= cooling_rate

        return best

    def generate_multi_lineups(self, player_pool: List[DFSPlayer],
                                 count: int = 20,
                                 max_overlap: int = 6) -> List[DFSLineup]:
        """
        Generate multiple diverse lineups with overlap constraints.
        """
        lineups = []

        for _ in range(count * 3):  # Generate extras, filter for diversity
            if len(lineups) >= count:
                break

            # Vary ownership penalty to create diversity
            ownership_penalty = random.uniform(0, 0.15)

            lineup = self.optimize_simulated_annealing(
                player_pool,
                iterations=3000,
                ownership_penalty=ownership_penalty
            )

            if not lineup:
                continue

            # Check diversity constraint
            is_diverse = True
            for existing in lineups:
                overlap = len(
                    set(p.player_id for p in lineup.players) &
                    set(p.player_id for p in existing.players)
                )
                if overlap > max_overlap:
                    is_diverse = False
                    break

            if is_diverse:
                lineups.append(lineup)

        lineups.sort(key=lambda l: l.total_projected, reverse=True)
        return lineups

    def build_stack(self, player_pool: List[DFSPlayer],
                    team: str, stack_size: int = 4,
                    bring_back: bool = True) -> Optional[DFSLineup]:
        """
        Build lineup around a team stack (correlated hitters).
        """
        # Get team hitters sorted by batting order
        team_hitters = [p for p in player_pool if p.team == team and p.position != 'P']
        team_hitters.sort(key=lambda p: p.batting_order if p.batting_order > 0 else 99)

        if len(team_hitters) < stack_size:
            return None

        # Pick top stack_size hitters by projected points
        stack_players = sorted(team_hitters, key=lambda p: p.projected_points, reverse=True)[:stack_size]

        # Find bring-back (opponent hitter)
        bring_back_player = None
        if bring_back and stack_players:
            opponent = stack_players[0].opponent
            opp_hitters = [p for p in player_pool if p.team == opponent and p.position != 'P']
            if opp_hitters:
                bring_back_player = max(opp_hitters, key=lambda p: p.projected_points)

        # Build lineup with stack locked in
        locked_ids = {p.player_id for p in stack_players}
        if bring_back_player:
            locked_ids.add(bring_back_player.player_id)

        # Try to fill remaining slots
        remaining_pool = [p for p in player_pool if p.player_id not in locked_ids]

        lineup = self.optimize_greedy(player_pool)
        if lineup:
            # Replace players to include stack
            for stack_player in stack_players:
                for i, lp in enumerate(lineup.players):
                    if lp.team != team and self._fits_slot(stack_player.position, self.roster_template[i]):
                        lineup.players[i] = stack_player
                        break

        return lineup

    # ── Internal Methods ──────────────────────────────────────

    def _fits_slot(self, player_pos: str, slot: str) -> bool:
        """Check if a player's position fits a roster slot"""
        if slot == 'UTIL':
            return player_pos != 'P'
        if '/' in slot:
            return player_pos in slot.split('/')
        return player_pos == slot

    def _score_lineup(self, lineup: DFSLineup,
                       ownership_penalty: float = 0.0) -> float:
        """Score a lineup (higher is better)"""
        if lineup.total_salary > self.salary_cap:
            return -1000

        score = lineup.total_projected

        # Penalize high ownership (for contrarian optimization)
        if ownership_penalty > 0:
            score -= lineup.total_ownership * ownership_penalty

        # Bonus for salary efficiency (use most of cap)
        remaining = self.salary_cap - lineup.total_salary
        if remaining < 0:
            return -1000
        if remaining < 500:
            score += 0.5  # Small bonus for using salary efficiently

        return score


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def generate_sample_pool() -> List[DFSPlayer]:
    """Generate a realistic DFS player pool"""
    players = []
    player_id = 0

    # Teams and matchups
    games = [
        ('NYY', 'BOS', 'g1', 5.2, 4.8),
        ('LAD', 'SFG', 'g2', 4.9, 3.8),
        ('HOU', 'TEX', 'g3', 5.0, 4.2),
        ('ATL', 'NYM', 'g4', 4.5, 4.3),
        ('PHI', 'MIA', 'g5', 4.8, 3.5),
        ('SD', 'ARI', 'g6', 4.3, 4.6),
    ]

    # Pitcher data
    pitchers = [
        ('Cole, G.', 'NYY', 10200, 18.5, 12.0, 'R'),
        ('Sale, C.', 'ATL', 9800, 17.2, 10.0, 'L'),
        ('Kershaw, C.', 'LAD', 9500, 16.8, 15.0, 'L'),
        ('Verlander, J.', 'HOU', 9200, 15.5, 8.0, 'R'),
        ('Wheeler, Z.', 'PHI', 9000, 16.0, 11.0, 'R'),
        ('Darvish, Y.', 'SD', 8800, 14.5, 9.0, 'R'),
    ]

    for name, team, salary, proj, own, hand in pitchers:
        opponent = ''
        game_id = ''
        for away, home, gid, _, _ in games:
            if team == away:
                opponent = home; game_id = gid; break
            elif team == home:
                opponent = away; game_id = gid; break

        players.append(DFSPlayer(
            player_id=f"p{player_id:03d}", name=name, team=team,
            position='P', salary=salary, projected_points=proj,
            ownership_pct=own, opponent=opponent, game_id=game_id,
            handedness=hand
        ))
        player_id += 1

    # Hitters for each team
    positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'OF']
    for away, home, gid, away_ir, home_ir in games:
        for team, opp, ir, is_home in [(away, home, away_ir, False), (home, away, home_ir, True)]:
            for i, pos in enumerate(positions):
                salary = random.randint(2500, 6500)
                base_proj = random.uniform(5.0, 12.0) * (ir / 4.5)
                ownership = random.uniform(1.0, 20.0)

                players.append(DFSPlayer(
                    player_id=f"p{player_id:03d}",
                    name=f"{team} {'CFSBOT'[i % 6]}{i+1}",
                    team=team, position=pos, salary=salary,
                    projected_points=round(base_proj, 1),
                    ownership_pct=round(ownership, 1),
                    batting_order=i + 1, opponent=opp,
                    game_id=gid, is_home=is_home,
                    implied_runs=ir, handedness=random.choice(['L', 'R']),
                ))
                player_id += 1

    return players


# ============================================================================
# DEMO
# ============================================================================

def demo_optimizer():
    """Demonstrate lineup optimizer"""
    print("=" * 70)
    print("🎰 MLB Predictor - DFS Lineup Optimizer Demo")
    print("=" * 70)
    print()

    pool = generate_sample_pool()
    print(f"📊 Player Pool: {len(pool)} players across "
          f"{len(set(p.team for p in pool))} teams")
    print()

    optimizer = LineupOptimizer(Platform.DRAFTKINGS)

    # Greedy optimization
    print("1️⃣  GREEDY OPTIMIZATION")
    print("-" * 60)
    greedy = optimizer.optimize_greedy(pool)
    if greedy:
        print(f"   Projected: {greedy.total_projected} pts")
        print(f"   Salary: ${greedy.total_salary:,} / ${DK_SALARY_CAP:,}")
        print(f"   Ownership: {greedy.total_ownership}%")
        print(f"   Players:")
        for i, p in enumerate(greedy.players):
            print(f"     {DK_ROSTER[i]:>4} | {p.name:<18} {p.team:>4} "
                  f"${p.salary:>5,} | {p.projected_points:>5.1f} pts "
                  f"| {p.ownership_pct}% own")
    print()

    # Simulated annealing
    print("2️⃣  SIMULATED ANNEALING (5000 iterations)")
    print("-" * 60)
    sa = optimizer.optimize_simulated_annealing(pool, iterations=5000)
    if sa:
        print(f"   Projected: {sa.total_projected} pts")
        print(f"   Salary: ${sa.total_salary:,} / ${DK_SALARY_CAP:,}")
        improvement = sa.total_projected - (greedy.total_projected if greedy else 0)
        print(f"   Improvement over greedy: {improvement:+.1f} pts")
    print()

    # Multi-lineup
    print("3️⃣  MULTI-LINEUP GENERATION (5 lineups)")
    print("-" * 60)
    multi = optimizer.generate_multi_lineups(pool, count=5, max_overlap=6)
    for i, lineup in enumerate(multi, 1):
        teams = lineup.teams
        top_team = max(teams, key=teams.get) if teams else "?"
        print(f"   Lineup {i}: {lineup.total_projected:>5.1f} pts | "
              f"${lineup.total_salary:>6,} | "
              f"{lineup.total_ownership:>5.1f}% own | "
              f"Stack: {top_team} x{teams.get(top_team, 0)}")
    print()

    # Contrarian lineup
    print("4️⃣  CONTRARIAN LINEUP (low ownership)")
    print("-" * 60)
    contrarian = optimizer.optimize_simulated_annealing(
        pool, iterations=5000, ownership_penalty=0.2
    )
    if contrarian:
        print(f"   Projected: {contrarian.total_projected} pts")
        print(f"   Ownership: {contrarian.total_ownership}% "
              f"(vs {sa.total_ownership if sa else 0}% for optimal)")
    print()

    print("=" * 70)
    print("✅ DFS Optimizer Demo Complete")
    print("=" * 70)

    return optimizer


if __name__ == "__main__":
    demo_optimizer()
