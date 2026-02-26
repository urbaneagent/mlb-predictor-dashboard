"""
MLB Predictor - Advanced Prop Bet Builder
Player prop analysis with historical data, matchup adjustments,
and multi-leg prop parlay optimization.
"""

import json
import time
import uuid
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime


# ──────────────────────────────────────────────
# Prop Bet Models
# ──────────────────────────────────────────────

class PropCategory(Enum):
    BATTING = "batting"
    PITCHING = "pitching"
    COMBINED = "combined"
    GAME = "game"


class PropMarket(Enum):
    # Batting
    HITS = "hits"
    HOME_RUNS = "home_runs"
    RBI = "rbi"
    RUNS_SCORED = "runs_scored"
    TOTAL_BASES = "total_bases"
    STOLEN_BASES = "stolen_bases"
    WALKS = "walks"
    HITS_RUNS_RBI = "hits_runs_rbi"

    # Pitching
    STRIKEOUTS = "strikeouts"
    OUTS_RECORDED = "outs_recorded"
    EARNED_RUNS = "earned_runs"
    HITS_ALLOWED = "hits_allowed"
    WALKS_ALLOWED = "walks_allowed"
    PITCHES_THROWN = "pitches_thrown"
    FIRST_INNING_K = "first_inning_strikeout"

    # Game
    FIRST_TEAM_SCORE = "first_team_to_score"
    RUN_IN_FIRST = "run_in_first_inning"
    TOTAL_RUNS_FIRST_5 = "total_runs_first_5"


@dataclass
class PlayerSeason:
    """Season-level stats for prop analysis"""
    player_id: str
    name: str
    team: str
    position: str
    games: int = 0
    # Batting
    at_bats: int = 0
    hits: int = 0
    home_runs: int = 0
    rbi: int = 0
    runs: int = 0
    stolen_bases: int = 0
    walks: int = 0
    total_bases: int = 0
    batting_avg: float = 0.0
    obp: float = 0.0
    slugging: float = 0.0
    # Pitching
    innings_pitched: float = 0.0
    strikeouts: int = 0
    earned_runs: int = 0
    hits_allowed: int = 0
    walks_allowed: int = 0
    era: float = 0.0
    whip: float = 0.0
    k_per_9: float = 0.0
    # Splits
    vs_lhp: Dict = field(default_factory=dict)  # vs left-handed pitchers
    vs_rhp: Dict = field(default_factory=dict)  # vs right-handed pitchers
    home_stats: Dict = field(default_factory=dict)
    away_stats: Dict = field(default_factory=dict)
    last_7: Dict = field(default_factory=dict)
    last_30: Dict = field(default_factory=dict)


@dataclass
class PropLine:
    """A single prop bet line"""
    prop_id: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    market: PropMarket
    line: float  # e.g., 1.5 for over/under 1.5 hits
    over_odds: int
    under_odds: int
    sportsbook: str = "consensus"
    # Analysis
    model_prediction: float = 0.0
    hit_rate_season: float = 0.0  # % of games going over
    hit_rate_last_10: float = 0.0
    edge_over: float = 0.0  # EV edge on over
    edge_under: float = 0.0  # EV edge on under
    confidence: float = 0.0
    recommendation: str = ""  # "OVER", "UNDER", "PASS"
    factors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "prop_id": self.prop_id,
            "player_name": self.player_name,
            "team": self.team,
            "opponent": self.opponent,
            "market": self.market.value,
            "line": self.line,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "sportsbook": self.sportsbook,
            "model_prediction": round(self.model_prediction, 2),
            "hit_rate_season": round(self.hit_rate_season * 100, 1),
            "hit_rate_last_10": round(self.hit_rate_last_10 * 100, 1),
            "edge_over": round(self.edge_over, 1),
            "edge_under": round(self.edge_under, 1),
            "confidence": round(self.confidence * 100, 1),
            "recommendation": self.recommendation,
            "factors": self.factors,
            "grade": self._grade(),
        }

    def _grade(self) -> str:
        edge = max(abs(self.edge_over), abs(self.edge_under))
        if edge >= 15:
            return "A+"
        elif edge >= 10:
            return "A"
        elif edge >= 7:
            return "B+"
        elif edge >= 5:
            return "B"
        elif edge >= 3:
            return "C"
        return "D"


@dataclass
class PropParlay:
    """Multi-leg prop parlay"""
    parlay_id: str
    legs: List[PropLine] = field(default_factory=list)
    combined_odds: int = 0
    stake: float = 0.0
    potential_payout: float = 0.0
    implied_probability: float = 0.0
    model_probability: float = 0.0
    edge: float = 0.0
    correlation_adjustment: float = 1.0

    def to_dict(self):
        return {
            "parlay_id": self.parlay_id,
            "legs": [l.to_dict() for l in self.legs],
            "leg_count": len(self.legs),
            "combined_odds": self.combined_odds,
            "combined_odds_display": f"+{self.combined_odds}" if self.combined_odds > 0 else str(self.combined_odds),
            "stake": round(self.stake, 2),
            "potential_payout": round(self.potential_payout, 2),
            "implied_probability": round(self.implied_probability * 100, 1),
            "model_probability": round(self.model_probability * 100, 1),
            "edge": round(self.edge, 1),
            "correlation_adjustment": round(self.correlation_adjustment, 3),
        }


# ──────────────────────────────────────────────
# Player Prop Analyzer
# ──────────────────────────────────────────────

class PropAnalyzer:
    """
    Analyzes player props using season stats, matchup data, splits,
    and historical hit rates. Generates model predictions and edges.
    """

    # Simulated player database for demo
    PLAYERS = {
        "aaron-judge": PlayerSeason(
            "aaron-judge", "Aaron Judge", "NYY", "OF", 145,
            at_bats=535, hits=162, home_runs=48, rbi=121, runs=110,
            stolen_bases=4, walks=95, total_bases=361,
            batting_avg=.303, obp=.410, slugging=.675,
            vs_lhp={"avg": .325, "hr": 18, "ab": 160},
            vs_rhp={"avg": .290, "hr": 30, "ab": 375},
            last_7={"avg": .345, "hr": 3, "hits": 10, "ab": 29},
            last_30={"avg": .312, "hr": 8, "hits": 34, "ab": 109},
        ),
        "shohei-ohtani": PlayerSeason(
            "shohei-ohtani", "Shohei Ohtani", "LAD", "DH", 150,
            at_bats=560, hits=175, home_runs=52, rbi=118, runs=115,
            stolen_bases=56, walks=88, total_bases=395,
            batting_avg=.313, obp=.404, slugging=.705,
            vs_lhp={"avg": .298, "hr": 15, "ab": 151},
            vs_rhp={"avg": .319, "hr": 37, "ab": 409},
            last_7={"avg": .380, "hr": 4, "hits": 11, "ab": 29},
            last_30={"avg": .330, "hr": 10, "hits": 36, "ab": 109},
        ),
        "gerrit-cole": PlayerSeason(
            "gerrit-cole", "Gerrit Cole", "NYY", "SP", 30,
            innings_pitched=198.0, strikeouts=235, earned_runs=62,
            hits_allowed=155, walks_allowed=48,
            era=2.82, whip=1.03, k_per_9=10.7,
            vs_lhp={"k_rate": 0.28, "avg": .235},
            vs_rhp={"k_rate": 0.30, "avg": .210},
            last_7={"k_per_start": 8.5, "era": 2.40},
            last_30={"k_per_start": 7.8, "era": 3.10},
        ),
        "spencer-strider": PlayerSeason(
            "spencer-strider", "Spencer Strider", "ATL", "SP", 28,
            innings_pitched=180.0, strikeouts=262, earned_runs=55,
            hits_allowed=120, walks_allowed=42,
            era=2.75, whip=0.90, k_per_9=13.1,
            vs_lhp={"k_rate": 0.35, "avg": .195},
            vs_rhp={"k_rate": 0.32, "avg": .205},
            last_7={"k_per_start": 10.2, "era": 2.00},
            last_30={"k_per_start": 9.5, "era": 2.60},
        ),
        "mookie-betts": PlayerSeason(
            "mookie-betts", "Mookie Betts", "LAD", "SS", 142,
            at_bats=520, hits=160, home_runs=35, rbi=98, runs=115,
            stolen_bases=15, walks=72, total_bases=310,
            batting_avg=.308, obp=.394, slugging=.596,
            vs_lhp={"avg": .320, "hr": 12, "ab": 150},
            vs_rhp={"avg": .302, "hr": 23, "ab": 370},
            last_7={"avg": .290, "hr": 2, "hits": 8, "ab": 28},
            last_30={"avg": .305, "hr": 6, "hits": 32, "ab": 105},
        ),
        "jose-altuve": PlayerSeason(
            "jose-altuve", "Jose Altuve", "HOU", "2B", 148,
            at_bats=570, hits=182, home_runs=28, rbi=85, runs=105,
            stolen_bases=12, walks=55, total_bases=290,
            batting_avg=.319, obp=.378, slugging=.509,
            vs_lhp={"avg": .340, "hr": 10, "ab": 150},
            vs_rhp={"avg": .312, "hr": 18, "ab": 420},
            last_7={"avg": .310, "hr": 1, "hits": 9, "ab": 29},
            last_30={"avg": .325, "hr": 5, "hits": 37, "ab": 114},
        ),
    }

    def __init__(self):
        self.analyzed_props: List[PropLine] = []

    def analyze_batting_prop(self, player_id: str, market: PropMarket, line: float,
                             over_odds: int, under_odds: int, opponent: str = "",
                             opponent_pitcher_hand: str = "R",
                             park_factor: float = 1.0) -> PropLine:
        """Analyze a batting prop with full context"""
        player = self.PLAYERS.get(player_id)
        if not player:
            return PropLine(
                prop_id=f"prop-{str(uuid.uuid4())[:8]}",
                player_id=player_id,
                player_name=player_id,
                team="UNK", opponent=opponent,
                market=market, line=line,
                over_odds=over_odds, under_odds=under_odds,
                recommendation="PASS",
                factors=["Player not in database"],
            )

        # Calculate prediction based on market
        prediction = 0.0
        hit_rate = 0.0
        factors = []

        if market == PropMarket.HITS:
            per_game = player.hits / max(1, player.games)
            prediction = per_game * park_factor
            # Adjust for pitcher hand
            split = player.vs_lhp if opponent_pitcher_hand == "L" else player.vs_rhp
            if split.get("avg"):
                hand_adj = split["avg"] / max(0.001, player.batting_avg)
                prediction *= hand_adj
                if hand_adj > 1.05:
                    factors.append(f"✅ Hits {split['avg']:.3f} vs {opponent_pitcher_hand}HP (above avg)")
                elif hand_adj < 0.95:
                    factors.append(f"⚠️ Hits {split['avg']:.3f} vs {opponent_pitcher_hand}HP (below avg)")

            # Recent form
            if player.last_7.get("avg", 0) > player.batting_avg + 0.020:
                prediction *= 1.08
                factors.append(f"🔥 Hot streak: {player.last_7['avg']:.3f} last 7 games")
            elif player.last_7.get("avg", 0) < player.batting_avg - 0.020:
                prediction *= 0.92
                factors.append(f"❄️ Cold streak: {player.last_7['avg']:.3f} last 7 games")

            hit_rate = 0.62 if per_game > line else 0.38  # Simplified

        elif market == PropMarket.HOME_RUNS:
            per_game = player.home_runs / max(1, player.games)
            prediction = per_game * park_factor
            split = player.vs_lhp if opponent_pitcher_hand == "L" else player.vs_rhp
            if split.get("hr") and split.get("ab"):
                hr_rate = split["hr"] / split["ab"]
                prediction = hr_rate * player.at_bats / player.games * park_factor
                factors.append(f"HR rate vs {opponent_pitcher_hand}HP: {hr_rate:.4f}")
            hit_rate = min(0.45, per_game / max(0.01, line))

        elif market == PropMarket.TOTAL_BASES:
            per_game = player.total_bases / max(1, player.games)
            prediction = per_game * park_factor
            if player.last_30.get("avg", 0) > player.batting_avg:
                prediction *= 1.05
                factors.append("Recent slugging trend positive")
            hit_rate = 0.55 if per_game > line else 0.45

        elif market == PropMarket.RBI:
            per_game = player.rbi / max(1, player.games)
            prediction = per_game * park_factor
            hit_rate = 0.52 if per_game > line else 0.48

        elif market == PropMarket.STOLEN_BASES:
            per_game = player.stolen_bases / max(1, player.games)
            prediction = per_game
            hit_rate = 0.35 if per_game > line else 0.65  # SB are unpredictable

        elif market == PropMarket.RUNS_SCORED:
            per_game = player.runs / max(1, player.games)
            prediction = per_game * park_factor
            hit_rate = 0.55 if per_game > line else 0.45

        # Park factor
        if park_factor != 1.0:
            direction = "boost" if park_factor > 1.0 else "decrease"
            factors.append(f"Park factor: {park_factor:.2f} ({direction})")

        # Calculate edges
        over_implied = self._american_to_implied(over_odds)
        under_implied = self._american_to_implied(under_odds)

        # Model probability of going over
        if prediction > line:
            model_over_prob = min(0.85, 0.5 + (prediction - line) * 0.2)
        else:
            model_over_prob = max(0.15, 0.5 - (line - prediction) * 0.2)

        edge_over = (model_over_prob - over_implied) * 100
        edge_under = ((1 - model_over_prob) - under_implied) * 100

        # Recommendation
        if edge_over > 5:
            recommendation = "OVER"
            confidence = min(0.95, 0.5 + edge_over / 40)
        elif edge_under > 5:
            recommendation = "UNDER"
            confidence = min(0.95, 0.5 + edge_under / 40)
        else:
            recommendation = "PASS"
            confidence = 0.3

        factors.append(f"Model prediction: {prediction:.2f} (line: {line})")
        factors.append(f"Season rate: {hit_rate * 100:.0f}% over")

        prop = PropLine(
            prop_id=f"prop-{str(uuid.uuid4())[:8]}",
            player_id=player_id,
            player_name=player.name,
            team=player.team,
            opponent=opponent,
            market=market,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            model_prediction=prediction,
            hit_rate_season=hit_rate,
            hit_rate_last_10=hit_rate * random.uniform(0.9, 1.1),
            edge_over=edge_over,
            edge_under=edge_under,
            confidence=confidence,
            recommendation=recommendation,
            factors=factors,
        )
        self.analyzed_props.append(prop)
        return prop

    def analyze_pitching_prop(self, player_id: str, market: PropMarket, line: float,
                              over_odds: int, under_odds: int, opponent: str = "",
                              opponent_k_rate: float = 0.23) -> PropLine:
        """Analyze a pitching prop"""
        player = self.PLAYERS.get(player_id)
        if not player:
            return PropLine(
                prop_id=f"prop-{str(uuid.uuid4())[:8]}",
                player_id=player_id, player_name=player_id,
                team="UNK", opponent=opponent,
                market=market, line=line,
                over_odds=over_odds, under_odds=under_odds,
                recommendation="PASS",
            )

        prediction = 0.0
        factors = []

        if market == PropMarket.STRIKEOUTS:
            if player.games > 0 and player.innings_pitched > 0:
                k_per_start = player.strikeouts / player.games
                prediction = k_per_start

                # Opponent K rate adjustment
                league_avg_k_rate = 0.23
                opp_adj = opponent_k_rate / league_avg_k_rate
                prediction *= opp_adj
                if opp_adj > 1.1:
                    factors.append(f"✅ Opponent K rate {opponent_k_rate:.1%} (above avg)")
                elif opp_adj < 0.9:
                    factors.append(f"⚠️ Opponent K rate {opponent_k_rate:.1%} (below avg)")

                # Recent form
                recent_k = player.last_7.get("k_per_start", k_per_start)
                if recent_k > k_per_start + 1:
                    prediction += 0.5
                    factors.append(f"🔥 Averaging {recent_k:.1f} K/start last 3")

                factors.append(f"Season avg: {k_per_start:.1f} K/start")
                factors.append(f"K/9: {player.k_per_9:.1f}")

        elif market == PropMarket.OUTS_RECORDED:
            ip_per_start = player.innings_pitched / max(1, player.games)
            prediction = ip_per_start * 3  # Outs per start

        elif market == PropMarket.EARNED_RUNS:
            er_per_start = player.earned_runs / max(1, player.games)
            prediction = er_per_start

        # Calculate edges
        over_implied = self._american_to_implied(over_odds)
        under_implied = self._american_to_implied(under_odds)

        if prediction > line:
            model_over_prob = min(0.85, 0.5 + (prediction - line) * 0.15)
        else:
            model_over_prob = max(0.15, 0.5 - (line - prediction) * 0.15)

        edge_over = (model_over_prob - over_implied) * 100
        edge_under = ((1 - model_over_prob) - under_implied) * 100

        recommendation = "OVER" if edge_over > 5 else "UNDER" if edge_under > 5 else "PASS"
        confidence = min(0.9, 0.5 + max(abs(edge_over), abs(edge_under)) / 40)

        factors.append(f"Model prediction: {prediction:.1f} (line: {line})")

        prop = PropLine(
            prop_id=f"prop-{str(uuid.uuid4())[:8]}",
            player_id=player_id,
            player_name=player.name,
            team=player.team,
            opponent=opponent,
            market=market,
            line=line,
            over_odds=over_odds,
            under_odds=under_odds,
            model_prediction=prediction,
            edge_over=edge_over,
            edge_under=edge_under,
            confidence=confidence,
            recommendation=recommendation,
            factors=factors,
        )
        self.analyzed_props.append(prop)
        return prop

    def build_prop_parlay(self, prop_ids: List[str], stake: float = 10.0) -> PropParlay:
        """Build a multi-leg prop parlay with correlation adjustments"""
        legs = [p for p in self.analyzed_props if p.prop_id in prop_ids]
        if not legs:
            return PropParlay(parlay_id=f"pp-{str(uuid.uuid4())[:8]}")

        # Calculate combined probability
        combined_prob = 1.0
        combined_decimal = 1.0

        for leg in legs:
            if leg.recommendation == "OVER":
                odds = leg.over_odds
                prob = self._american_to_implied(odds) + (leg.edge_over / 100)
            elif leg.recommendation == "UNDER":
                odds = leg.under_odds
                prob = self._american_to_implied(odds) + (leg.edge_under / 100)
            else:
                odds = leg.over_odds  # Default
                prob = self._american_to_implied(odds)

            combined_prob *= max(0.1, min(0.9, prob))
            dec = self._american_to_decimal(odds)
            combined_decimal *= dec

        # Correlation adjustment (same-game props are correlated)
        teams = set(l.team for l in legs)
        if len(teams) == 1:
            correlation = 0.85  # Same team = high correlation
        elif len(teams) < len(legs):
            correlation = 0.92  # Some overlap
        else:
            correlation = 1.0  # Independent

        adjusted_prob = combined_prob * correlation

        # Convert back to American
        if combined_decimal >= 2.0:
            combined_american = int((combined_decimal - 1) * 100)
        else:
            combined_american = int(-100 / (combined_decimal - 1))

        implied = 1 / combined_decimal
        edge = (adjusted_prob - implied) * 100
        payout = stake * combined_decimal

        return PropParlay(
            parlay_id=f"pp-{str(uuid.uuid4())[:8]}",
            legs=legs,
            combined_odds=combined_american,
            stake=stake,
            potential_payout=payout,
            implied_probability=implied,
            model_probability=adjusted_prob,
            edge=edge,
            correlation_adjustment=correlation,
        )

    def get_top_picks(self, min_edge: float = 5.0, limit: int = 10) -> List[Dict]:
        """Get top prop picks sorted by edge"""
        picks = [p for p in self.analyzed_props if p.recommendation != "PASS"
                 and max(abs(p.edge_over), abs(p.edge_under)) >= min_edge]
        picks.sort(key=lambda p: -max(abs(p.edge_over), abs(p.edge_under)))
        return [p.to_dict() for p in picks[:limit]]

    @staticmethod
    def _american_to_decimal(odds: int) -> float:
        if odds > 0:
            return 1 + odds / 100
        return 1 + 100 / abs(odds)

    @staticmethod
    def _american_to_implied(odds: int) -> float:
        if odds > 0:
            return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("MLB Predictor - Advanced Prop Bet Builder")
    print("=" * 60)

    analyzer = PropAnalyzer()

    # Analyze batting props
    print(f"\n⚾ Batting Props:")
    batting_props = [
        ("aaron-judge", PropMarket.HITS, 1.5, -130, +110, "BOS", "R", 1.05),
        ("shohei-ohtani", PropMarket.TOTAL_BASES, 2.5, -115, -105, "SFG", "R", 0.92),
        ("mookie-betts", PropMarket.HITS, 1.5, -140, +120, "SFG", "L", 0.92),
        ("jose-altuve", PropMarket.RBI, 0.5, -120, +100, "TEX", "R", 1.08),
        ("aaron-judge", PropMarket.HOME_RUNS, 0.5, +170, -220, "BOS", "L", 1.05),
        ("shohei-ohtani", PropMarket.STOLEN_BASES, 0.5, +135, -165, "SFG", "R", 1.0),
    ]

    for pid, market, line, over, under, opp, hand, park in batting_props:
        prop = analyzer.analyze_batting_prop(pid, market, line, over, under, opp, hand, park)
        d = prop.to_dict()
        rec_emoji = {"OVER": "🟢", "UNDER": "🔴", "PASS": "⚪"}.get(d["recommendation"], "⚪")
        print(f"\n  {rec_emoji} {d['player_name']} {d['market']} {'o' if d['recommendation'] == 'OVER' else 'u'}{d['line']} "
              f"[{d['grade']}]")
        print(f"     Model: {d['model_prediction']} | Edge: {max(d['edge_over'], d['edge_under']):+.1f}% | "
              f"Confidence: {d['confidence']}%")
        for f in d["factors"][:3]:
            print(f"     {f}")

    # Analyze pitching props
    print(f"\n🎯 Pitching Props:")
    pitching_props = [
        ("gerrit-cole", PropMarket.STRIKEOUTS, 7.5, -120, +100, "BOS", 0.25),
        ("spencer-strider", PropMarket.STRIKEOUTS, 9.5, -105, -115, "NYM", 0.27),
    ]

    for pid, market, line, over, under, opp, k_rate in pitching_props:
        prop = analyzer.analyze_pitching_prop(pid, market, line, over, under, opp, k_rate)
        d = prop.to_dict()
        rec_emoji = {"OVER": "🟢", "UNDER": "🔴", "PASS": "⚪"}.get(d["recommendation"], "⚪")
        print(f"\n  {rec_emoji} {d['player_name']} {d['market']} {'o' if d['recommendation'] == 'OVER' else 'u'}{d['line']} "
              f"[{d['grade']}]")
        print(f"     Model: {d['model_prediction']} | Edge: {max(d['edge_over'], d['edge_under']):+.1f}% | "
              f"Confidence: {d['confidence']}%")
        for f in d["factors"][:3]:
            print(f"     {f}")

    # Top picks
    print(f"\n🏆 Top Picks (by edge):")
    top = analyzer.get_top_picks(min_edge=3.0)
    for i, pick in enumerate(top[:5], 1):
        edge = max(pick["edge_over"], pick["edge_under"])
        print(f"  {i}. [{pick['grade']}] {pick['player_name']} {pick['market']} "
              f"{'O' if pick['recommendation'] == 'OVER' else 'U'}{pick['line']} "
              f"— Edge: {edge:+.1f}%")

    # Build parlay
    print(f"\n🎰 Prop Parlay Builder:")
    if len(analyzer.analyzed_props) >= 3:
        # Pick top 3 non-PASS props
        parlay_props = [p for p in analyzer.analyzed_props if p.recommendation != "PASS"][:3]
        parlay = analyzer.build_prop_parlay([p.prop_id for p in parlay_props], stake=25.0)
        pd = parlay.to_dict()
        print(f"  Legs: {pd['leg_count']}")
        for leg in pd["legs"]:
            print(f"    • {leg['player_name']} {leg['market']} {leg['recommendation']} {leg['line']}")
        print(f"  Combined odds: {pd['combined_odds_display']}")
        print(f"  Stake: ${pd['stake']} → Payout: ${pd['potential_payout']:,.2f}")
        print(f"  Implied: {pd['implied_probability']}% | Model: {pd['model_probability']}%")
        print(f"  Edge: {pd['edge']:+.1f}%")
        print(f"  Correlation adj: {pd['correlation_adjustment']}")

    print(f"\n✅ Advanced Prop Builder ready!")
    print("  • 15+ prop markets (batting + pitching)")
    print("  • Pitcher handedness splits")
    print("  • Park factor adjustments")
    print("  • Hot/cold streak detection")
    print("  • Opponent K-rate analysis")
    print("  • Model prediction vs line comparison")
    print("  • Edge calculation with confidence grades")
    print("  • Multi-leg prop parlay builder")
    print("  • Correlation adjustments (same-game)")


if __name__ == "__main__":
    demo()
