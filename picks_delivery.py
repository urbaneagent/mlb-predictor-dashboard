"""
MLB Predictor - Picks Delivery System
Formats and delivers daily picks via multiple channels:
- Telegram bot messages
- Email digest
- API endpoint (for web dashboard)
- CSV/PDF export for betting shops
"""
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Pick:
    """A single betting pick."""
    game_id: int
    away_team: str
    home_team: str
    game_time: str
    pick_type: str  # moneyline, spread, total, F5, prop
    pick_side: str  # Home, Away, Over, Under
    pick_team: str  # Team name if ML/spread
    pick_line: str  # e.g., "-1.5", "O 8.5"
    odds: int  # American odds, e.g., -110
    model_probability: float  # Our model's probability
    implied_probability: float  # Book's implied probability
    edge: float  # Model - Implied (value edge)
    confidence: str  # HIGH, MEDIUM, LOW
    kelly_fraction: float  # Recommended bet fraction
    recommended_units: float  # Bet size in units
    reasoning: str  # Short explanation
    away_pitcher: str = ""
    home_pitcher: str = ""
    weather: str = ""
    injury_note: str = ""
    model_score: float = 0.0  # Overall model confidence score


@dataclass
class DailyPicksCard:
    """Complete daily picks package."""
    date: str
    total_picks: int
    picks: list  # List[Pick]
    record_ytd: str  # "42-28 (+14.2u)"
    roi_ytd: float
    streak: str  # "W4"
    bankroll_status: str  # "Up 14.2 units"
    model_version: str
    generated_at: str
    notes: str = ""


class PicksFormatter:
    """
    Formats picks for different delivery channels.
    """

    # Confidence emojis
    CONFIDENCE_EMOJI = {
        "HIGH": "🔥", "MEDIUM": "✅", "LOW": "⚡"
    }

    EDGE_EMOJI = {
        "strong": "💎", "good": "⭐", "marginal": "📊"
    }

    def format_telegram(self, card: DailyPicksCard) -> str:
        """Format picks for Telegram delivery."""
        lines = [
            f"⚾ **MLB PICKS — {card.date}**",
            f"📊 Record: {card.record_ytd} | ROI: {card.roi_ytd:+.1f}%",
            f"🔥 Streak: {card.streak} | {card.bankroll_status}",
            "",
            f"━━━━━━━━━━━━━━━━━━",
        ]

        # Sort by confidence (HIGH first) then by edge
        sorted_picks = sorted(card.picks, key=lambda p: (
            {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}.get(p.get('confidence') if isinstance(p, dict) else p.confidence, 3),
            -(p.get('edge') if isinstance(p, dict) else p.edge)
        ))

        for p in sorted_picks:
            if isinstance(p, dict):
                pick = Pick(**p) if not isinstance(p, Pick) else p
            else:
                pick = p

            conf_emoji = self.CONFIDENCE_EMOJI.get(pick.confidence, "📊")
            edge_label = "💎" if pick.edge > 0.08 else "⭐" if pick.edge > 0.04 else "📊"

            # Pick line display
            if pick.pick_type == "moneyline":
                pick_display = f"**{pick.pick_team}** ML ({pick.odds:+d})"
            elif pick.pick_type == "spread":
                pick_display = f"**{pick.pick_team}** {pick.pick_line} ({pick.odds:+d})"
            elif pick.pick_type == "total":
                pick_display = f"**{pick.pick_side}** {pick.pick_line} ({pick.odds:+d})"
            elif pick.pick_type == "F5":
                pick_display = f"**{pick.pick_team}** F5 {pick.pick_line} ({pick.odds:+d})"
            else:
                pick_display = f"**{pick.pick_side}** {pick.pick_line} ({pick.odds:+d})"

            matchup_line = f"{pick.away_team} @ {pick.home_team}"
            if pick.game_time:
                matchup_line += f" — {pick.game_time}"

            lines.extend([
                "",
                f"{conf_emoji} {edge_label} **{pick.confidence}** ({pick.recommended_units:.1f}u)",
                f"🏟️ {matchup_line}",
                f"📌 {pick_display}",
                f"📈 Model: {pick.model_probability:.1%} vs Book: {pick.implied_probability:.1%} | Edge: {pick.edge:.1%}",
            ])

            if pick.reasoning:
                lines.append(f"💡 {pick.reasoning}")

            if pick.injury_note:
                lines.append(f"🏥 {pick.injury_note}")

        lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━",
            f"📊 {card.total_picks} picks today | Model v{card.model_version}",
            "⚠️ Entertainment only. Gamble responsibly.",
        ])

        if card.notes:
            lines.append(f"\n📝 {card.notes}")

        return '\n'.join(lines)

    def format_email_html(self, card: DailyPicksCard) -> str:
        """Format picks as HTML email."""
        picks_html = ""
        sorted_picks = sorted(card.picks, key=lambda p: (
            {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}.get(p.get('confidence', 'LOW') if isinstance(p, dict) else p.confidence, 3),
            -(p.get('edge', 0) if isinstance(p, dict) else p.edge)
        ))

        for p in sorted_picks:
            pick = p if isinstance(p, dict) else asdict(p)
            conf = pick.get('confidence', 'LOW')
            conf_color = {'HIGH': '#dc2626', 'MEDIUM': '#059669', 'LOW': '#6b7280'}[conf]

            if pick.get('pick_type') == 'moneyline':
                pick_str = f"{pick['pick_team']} ML ({pick['odds']:+d})"
            elif pick.get('pick_type') == 'total':
                pick_str = f"{pick['pick_side']} {pick['pick_line']} ({pick['odds']:+d})"
            else:
                pick_str = f"{pick['pick_team']} {pick.get('pick_line', '')} ({pick['odds']:+d})"

            picks_html += f"""
            <tr>
                <td style="padding:12px;border-bottom:1px solid #eee">
                    <strong>{pick['away_team']} @ {pick['home_team']}</strong><br>
                    <span style="font-size:12px;color:#888">{pick.get('game_time', '')}</span>
                </td>
                <td style="padding:12px;border-bottom:1px solid #eee;font-weight:bold">{pick_str}</td>
                <td style="padding:12px;border-bottom:1px solid #eee">
                    <span style="color:{conf_color};font-weight:bold">{conf}</span>
                </td>
                <td style="padding:12px;border-bottom:1px solid #eee">{pick['model_probability']:.1%}</td>
                <td style="padding:12px;border-bottom:1px solid #eee;color:#059669;font-weight:bold">
                    {pick['edge']:.1%}
                </td>
                <td style="padding:12px;border-bottom:1px solid #eee">{pick['recommended_units']:.1f}u</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html>
<head><style>
    body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #1e293b; color: white; padding: 10px; text-align: left; font-size: 12px; }}
</style></head>
<body>
    <h1 style="color:#1e293b">⚾ MLB Picks — {card.date}</h1>
    <p style="color:#888">Record: {card.record_ytd} | ROI: {card.roi_ytd:+.1f}% | {card.streak}</p>
    <table>
        <thead>
            <tr><th>Game</th><th>Pick</th><th>Conf</th><th>Model %</th><th>Edge</th><th>Size</th></tr>
        </thead>
        <tbody>{picks_html}</tbody>
    </table>
    <p style="color:#888;font-size:12px;margin-top:20px">
        {card.total_picks} picks | Model v{card.model_version} | ⚠️ Entertainment only
    </p>
</body></html>"""

    def format_csv(self, card: DailyPicksCard) -> str:
        """Format picks as CSV for export."""
        headers = [
            "Date", "Game", "Pick Type", "Pick", "Odds", "Model %",
            "Implied %", "Edge", "Confidence", "Units", "Reasoning"
        ]
        lines = [','.join(headers)]

        for p in card.picks:
            pick = p if isinstance(p, dict) else asdict(p)
            game = f"{pick['away_team']} @ {pick['home_team']}"

            if pick.get('pick_type') == 'moneyline':
                pick_str = f"{pick['pick_team']} ML"
            elif pick.get('pick_type') == 'total':
                pick_str = f"{pick['pick_side']} {pick.get('pick_line', '')}"
            else:
                pick_str = f"{pick['pick_team']} {pick.get('pick_line', '')}"

            row = [
                card.date, game, pick.get('pick_type', ''), pick_str,
                str(pick.get('odds', '')),
                f"{pick.get('model_probability', 0):.3f}",
                f"{pick.get('implied_probability', 0):.3f}",
                f"{pick.get('edge', 0):.3f}",
                pick.get('confidence', ''),
                f"{pick.get('recommended_units', 0):.1f}",
                f'"{pick.get("reasoning", "")}"'
            ]
            lines.append(','.join(row))

        return '\n'.join(lines)

    def format_dashboard_json(self, card: DailyPicksCard) -> dict:
        """Format picks for web dashboard API response."""
        picks_list = []
        for p in card.picks:
            pick = p if isinstance(p, dict) else asdict(p)
            picks_list.append(pick)

        return {
            "date": card.date,
            "summary": {
                "total_picks": card.total_picks,
                "high_confidence": len([p for p in picks_list if p.get('confidence') == 'HIGH']),
                "medium_confidence": len([p for p in picks_list if p.get('confidence') == 'MEDIUM']),
                "low_confidence": len([p for p in picks_list if p.get('confidence') == 'LOW']),
                "avg_edge": sum(p.get('edge', 0) for p in picks_list) / len(picks_list) if picks_list else 0,
                "total_units_risked": sum(p.get('recommended_units', 0) for p in picks_list),
            },
            "record": card.record_ytd,
            "roi": card.roi_ytd,
            "streak": card.streak,
            "bankroll": card.bankroll_status,
            "model_version": card.model_version,
            "picks": picks_list,
            "generated_at": card.generated_at,
            "notes": card.notes
        }


def generate_sample_picks() -> DailyPicksCard:
    """Generate sample picks for demo/testing."""
    picks = [
        Pick(
            game_id=746123, away_team="NYY", home_team="BOS",
            game_time="7:10 PM ET", pick_type="moneyline",
            pick_side="Away", pick_team="NYY", pick_line="",
            odds=-125, model_probability=0.58, implied_probability=0.556,
            edge=0.024, confidence="MEDIUM", kelly_fraction=0.035,
            recommended_units=1.5,
            reasoning="Yankees bats hot vs LHP (last 14d: .289/.358/.492). Cole elite at Fenway (2.45 ERA last 6 starts).",
            away_pitcher="Gerrit Cole", home_pitcher="Brayan Bello",
            weather="72°F, partly cloudy, wind 8mph out to CF"
        ),
        Pick(
            game_id=746124, away_team="LAD", home_team="SF",
            game_time="9:45 PM ET", pick_type="moneyline",
            pick_side="Away", pick_team="LAD", pick_line="",
            odds=-165, model_probability=0.68, implied_probability=0.623,
            edge=0.057, confidence="HIGH", kelly_fraction=0.062,
            recommended_units=2.5,
            reasoning="Ohtani splits vs RHP (.340/.400/.680). Giants pen 5.12 ERA last 7 days. Dodgers 8-2 L10 on road.",
            away_pitcher="Yoshinobu Yamamoto", home_pitcher="Logan Webb",
            weather="68°F, clear, Oracle Park wind from left field"
        ),
        Pick(
            game_id=746125, away_team="HOU", home_team="TEX",
            game_time="8:05 PM ET", pick_type="total",
            pick_side="Over", pick_team="", pick_line="O 8.5",
            odds=-105, model_probability=0.59, implied_probability=0.512,
            edge=0.078, confidence="HIGH", kelly_fraction=0.072,
            recommended_units=3.0,
            reasoning="Both pens depleted (HOU 3rd straight day, TEX used closer yesterday). Wind 15mph out to right. 12-3 Over in last 15 meetings at Globe Life.",
            away_pitcher="Framber Valdez", home_pitcher="Nathan Eovaldi",
            weather="95°F, wind 15mph out to RF, Globe Life roof OPEN",
            injury_note="TEX: Corey Seager DTD (hamstring), may sit"
        ),
        Pick(
            game_id=746126, away_team="ATL", home_team="PHI",
            game_time="7:05 PM ET", pick_type="F5",
            pick_side="Home", pick_team="PHI", pick_line="F5 -0.5",
            odds=-120, model_probability=0.61, implied_probability=0.545,
            edge=0.065, confidence="HIGH", kelly_fraction=0.058,
            recommended_units=2.0,
            reasoning="Wheeler dominates ATL (1.89 ERA in 8 career starts at Citizens Bank). F5 avoids ATL's elite late pen. PHI lineup .302 BA first 5 innings at home.",
            away_pitcher="Max Fried", home_pitcher="Zack Wheeler",
            weather="78°F, clear"
        ),
    ]

    return DailyPicksCard(
        date=datetime.now().strftime("%B %d, %Y"),
        total_picks=len(picks),
        picks=[asdict(p) for p in picks],
        record_ytd="42-28 (+14.2u)",
        roi_ytd=11.8,
        streak="W4",
        bankroll_status="Up 14.2 units",
        model_version="5.0",
        generated_at=datetime.now(timezone.utc).isoformat(),
        notes="Season starts in ~5 weeks. These are preseason simulation picks based on projected lineups."
    )


if __name__ == "__main__":
    card = generate_sample_picks()
    formatter = PicksFormatter()

    print("=" * 60)
    print("TELEGRAM FORMAT")
    print("=" * 60)
    print(formatter.format_telegram(card))

    print("\n" + "=" * 60)
    print("CSV FORMAT")
    print("=" * 60)
    print(formatter.format_csv(card))
