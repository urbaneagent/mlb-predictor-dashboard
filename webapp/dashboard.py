"""
HTML Dashboard Renderer for MLB Predictor.
Generates a professional single-page dashboard with:
  - Today's Top Hitters (hit probability %)
  - Today's Game Picks (win probability %)
  - Legacy matchup data
  - Model performance stats
"""
from datetime import datetime
from typing import List, Dict, Any, Optional

from .config import APP_VERSION, MODEL_VERSION


def render_dashboard(
    picks: List[Dict[str, Any]],
    leaderboard: Dict[str, Any],
    model_info: Dict[str, Any],
    hits_data: Optional[Dict[str, Any]] = None,
    wins_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the full HTML dashboard with live predictions."""

    today = datetime.now().strftime("%A, %B %d, %Y")
    updated = datetime.now().strftime("%I:%M %p ET")
    record = leaderboard.get("record", {})
    season = record.get("season_2025", {})
    last30 = record.get("last_30_days", {})
    locks = record.get("locks_only", {})
    model_stats = leaderboard.get("model_stats", {})

    # ── Live Predictions Mode ─────────────────────────────────
    is_demo = True
    mode_label = "PRESEASON PROJECTIONS"
    mode_badge_class = "badge-demo"

    top_hitters = []
    game_picks = []

    if hits_data:
        top_hitters = hits_data.get("top_hitters", [])
        if hits_data.get("mode") == "regular_season":
            is_demo = False
            mode_label = "LIVE"
            mode_badge_class = "badge-live"
        elif hits_data.get("mode") == "spring_training":
            mode_label = "SPRING TRAINING"
            mode_badge_class = "badge-spring"

    if wins_data:
        game_picks = wins_data.get("predictions", [])

    disclaimer_html = ""
    if is_demo:
        disclaimer_html = """
        <div class="demo-banner">
            ⚠️ <strong>PRESEASON PROJECTIONS</strong> — Based on 2025 stats.
            Live predictions begin Opening Day, March 27, 2026.
            Data shown is for demonstration purposes.
        </div>"""

    # ── Top Hitters Table ─────────────────────────────────────
    hitters_rows = ""
    for i, h in enumerate(top_hitters[:20], 1):
        hit_pct = f"{h['hit_probability']*100:.1f}%"
        avg = f".{int(h['batter_avg']*1000):03d}" if h['batter_avg'] > 0 else ".---"
        ops = f"{h['batter_ops']:.3f}" if h['batter_ops'] > 0 else "---"

        # Confidence coloring
        hp = h['hit_probability']
        if hp >= 0.82:
            row_class = "row-lock"
        elif hp >= 0.75:
            row_class = "row-strong"
        elif hp >= 0.68:
            row_class = "row-lean"
        else:
            row_class = ""

        platoon_icon = "✅" if h.get("platoon_advantage") else ""
        home_icon = "🏠" if h.get("is_home") else ""

        hitters_rows += f"""
        <tr class="{row_class}">
            <td class="rank">#{i}</td>
            <td class="player-name">
                {h['batter']}
                <span class="team-badge">{h['team']}</span>
            </td>
            <td class="stat">{h.get('position', '')}</td>
            <td class="stat hit-prob">{hit_pct}</td>
            <td class="stat">{avg}</td>
            <td class="stat">{ops}</td>
            <td class="stat pitcher-cell">
                {h['vs_pitcher']}
                <span class="pitcher-era">({h['pitcher_era']:.2f})</span>
            </td>
            <td class="stat">{platoon_icon}</td>
            <td class="stat">{home_icon}</td>
            <td class="confidence">{h['confidence']}</td>
        </tr>"""

    no_hitters_msg = ""
    if not top_hitters:
        no_hitters_msg = """
        <tr>
            <td colspan="10" class="no-data">
                ⚾ No games scheduled today. Check back on game day!
            </td>
        </tr>"""

    # ── Game Picks Table ──────────────────────────────────────
    picks_rows = ""
    for i, g in enumerate(game_picks, 1):
        away = g["away"]
        home = g["home"]
        pick_prob = f"{g['pick_probability']*100:.1f}%"
        away_wp = f"{away['win_probability']*100:.1f}%"
        home_wp = f"{home['win_probability']*100:.1f}%"

        # Highlight the picked team
        pick_abbr = g["pick"]
        away_class = "pick-highlight" if pick_abbr == away["abbr"] else ""
        home_class = "pick-highlight" if pick_abbr == home["abbr"] else ""

        picks_rows += f"""
        <tr>
            <td class="rank">#{i}</td>
            <td class="game-label">{g['game_label']}</td>
            <td class="stat {away_class}">
                {away['abbr']}
                <span class="wp">{away_wp}</span>
            </td>
            <td class="stat {home_class}">
                {home['abbr']}
                <span class="wp">{home_wp}</span>
            </td>
            <td class="stat pitcher-cell">
                {away['probable_pitcher']} ({away['pitcher_era']:.2f})
                <span class="vs-text">vs</span>
                {home['probable_pitcher']} ({home['pitcher_era']:.2f})
            </td>
            <td class="stat pick-cell">
                <span class="pick-badge">{g['pick']}</span>
                {pick_prob}
            </td>
            <td class="confidence">{g['confidence']}</td>
        </tr>"""

    no_picks_msg = ""
    if not game_picks:
        no_picks_msg = """
        <tr>
            <td colspan="7" class="no-data">
                ⚾ No games scheduled today. Check back on game day!
            </td>
        </tr>"""

    # ── Legacy Picks Table (from CSV) ─────────────────────────
    legacy_rows = ""
    for i, p in enumerate(picks[:10], 1):
        edge_pct = f"{p['edge']*100:.1f}%"
        win_pct = f"{p['win_prob']*100:.0f}%"
        hit_pct = f"{p['hit_prob']*100:.1f}%"
        conf = p.get("confidence", "")

        edge_val = p["edge"]
        if edge_val >= 0.08:
            edge_class = "edge-lock"
        elif edge_val >= 0.05:
            edge_class = "edge-strong"
        elif edge_val >= 0.03:
            edge_class = "edge-lean"
        else:
            edge_class = "edge-value"

        batter_info = p.get("batter", "")
        if p.get("batter_team"):
            batter_info += f' <span class="team-badge">{p["batter_team"]}</span>'
        pitcher_info = p.get("pitcher", "")
        if p.get("pitcher_team"):
            pitcher_info += f' <span class="team-badge">{p["pitcher_team"]}</span>'

        legacy_rows += f"""
        <tr>
            <td class="rank">#{i}</td>
            <td class="player-name">{batter_info}</td>
            <td class="player-name">{pitcher_info}</td>
            <td class="stat">{win_pct}</td>
            <td class="stat {edge_class}">{edge_pct}</td>
            <td class="stat">{hit_pct}</td>
            <td class="confidence">{conf}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Predictor — Live Daily Predictions</title>
    <meta name="description" content="AI-powered MLB predictions with live daily hit probabilities and team win projections. Updated every 2 hours on game days.">
    <style>
        :root {{
            --bg: #0a0e17;
            --surface: #131927;
            --surface2: #1a2235;
            --border: #2a3550;
            --text: #e8ecf4;
            --text-dim: #8892a8;
            --accent: #4f8cff;
            --green: #22c55e;
            --red: #ef4444;
            --orange: #f59e0b;
            --gold: #fbbf24;
            --purple: #a78bfa;
            --cyan: #06b6d4;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 0 24px; }}

        /* Header */
        header {{
            background: linear-gradient(135deg, #0f1629 0%, #1a2744 100%);
            border-bottom: 1px solid var(--border);
            padding: 28px 0;
        }}
        .header-inner {{
            display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 16px;
        }}
        .logo {{ display: flex; align-items: center; gap: 12px; }}
        .logo-icon {{ font-size: 36px; }}
        .logo h1 {{ font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }}
        .logo h1 span {{ color: var(--accent); }}
        .header-meta {{ text-align: right; color: var(--text-dim); font-size: 14px; }}
        .header-meta .date {{ font-size: 16px; color: var(--text); font-weight: 600; }}

        /* Demo Banner */
        .demo-banner {{
            background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
            border: 1px solid rgba(245,158,11,0.3);
            border-radius: 12px;
            padding: 16px 24px;
            margin: 20px 0;
            color: var(--orange);
            font-size: 14px;
            text-align: center;
        }}

        /* Stats Banner */
        .stats-banner {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; padding: 24px 0;
        }}
        .stat-card {{
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; padding: 20px; text-align: center;
        }}
        .stat-card .label {{
            font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
            color: var(--text-dim); margin-bottom: 4px;
        }}
        .stat-card .value {{ font-size: 32px; font-weight: 700; }}
        .stat-card .sub {{ font-size: 13px; color: var(--text-dim); margin-top: 4px; }}
        .value-green {{ color: var(--green); }}
        .value-gold {{ color: var(--gold); }}
        .value-accent {{ color: var(--accent); }}
        .value-purple {{ color: var(--purple); }}
        .value-cyan {{ color: var(--cyan); }}

        /* Section */
        section {{ margin: 32px 0; }}
        .section-header {{
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 16px;
        }}
        .section-header h2 {{ font-size: 22px; font-weight: 700; }}
        .section-header h2 .icon {{ margin-right: 8px; }}
        .badge {{
            font-size: 12px; padding: 4px 12px; border-radius: 20px; font-weight: 600;
        }}
        .badge-live {{
            background: rgba(34, 197, 94, 0.15); color: var(--green);
            border: 1px solid rgba(34, 197, 94, 0.3);
        }}
        .badge-demo {{
            background: rgba(245, 158, 11, 0.15); color: var(--orange);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        .badge-spring {{
            background: rgba(6, 182, 212, 0.15); color: var(--cyan);
            border: 1px solid rgba(6, 182, 212, 0.3);
        }}

        /* Table */
        .table-wrap {{
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; overflow-x: auto;
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            text-align: left; padding: 14px 16px; font-size: 11px;
            text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim);
            background: var(--surface2); border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }}
        td {{
            padding: 12px 16px; border-bottom: 1px solid rgba(42, 53, 80, 0.5);
            font-size: 14px;
        }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover {{ background: rgba(79, 140, 255, 0.04); }}
        .rank {{ font-weight: 700; color: var(--text-dim); width: 50px; }}
        .player-name {{ font-weight: 600; white-space: nowrap; }}
        .stat {{ font-variant-numeric: tabular-nums; text-align: center; }}
        .confidence {{ white-space: nowrap; font-weight: 600; }}
        .team-badge {{
            font-size: 11px; padding: 2px 6px; border-radius: 4px;
            background: rgba(79, 140, 255, 0.12); color: var(--accent);
            margin-left: 6px; font-weight: 600;
        }}
        .no-data {{
            text-align: center; padding: 40px 20px !important;
            color: var(--text-dim); font-size: 16px;
        }}

        /* Hit probability highlighting */
        .hit-prob {{ font-weight: 700; }}
        .row-lock td.hit-prob {{ color: var(--green); }}
        .row-strong td.hit-prob {{ color: var(--gold); }}
        .row-lean td.hit-prob {{ color: var(--orange); }}
        .row-lock {{ background: rgba(34, 197, 94, 0.04); }}
        .row-strong {{ background: rgba(251, 191, 36, 0.03); }}

        .pitcher-cell {{ white-space: nowrap; }}
        .pitcher-era {{ color: var(--text-dim); font-size: 12px; margin-left: 4px; }}
        .vs-text {{ color: var(--text-dim); margin: 0 6px; font-size: 12px; }}

        /* Game picks */
        .game-label {{ font-weight: 700; white-space: nowrap; font-size: 15px; }}
        .wp {{ font-size: 12px; color: var(--text-dim); margin-left: 6px; }}
        .pick-highlight {{
            background: rgba(34, 197, 94, 0.08) !important;
            font-weight: 700;
        }}
        .pick-badge {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            background: var(--green); color: var(--bg); font-weight: 700;
            font-size: 12px; margin-right: 6px;
        }}
        .pick-cell {{ white-space: nowrap; font-weight: 700; }}

        /* Edge colors */
        .edge-lock {{ color: var(--green); font-weight: 700; background: rgba(34, 197, 94, 0.08); border-radius: 4px; }}
        .edge-strong {{ color: var(--gold); font-weight: 700; }}
        .edge-lean {{ color: var(--orange); font-weight: 600; }}
        .edge-value {{ color: var(--text-dim); }}

        /* API Links */
        .api-links {{ display: flex; gap: 8px; flex-wrap: wrap; }}
        .api-link {{
            font-size: 12px; padding: 6px 14px; border-radius: 6px;
            background: var(--surface2); border: 1px solid var(--border);
            color: var(--accent); text-decoration: none; font-weight: 600;
            transition: all 0.2s;
        }}
        .api-link:hover {{ background: rgba(79, 140, 255, 0.12); border-color: var(--accent); }}

        /* Two-column */
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 768px) {{
            .two-col {{ grid-template-columns: 1fr; }}
            .stats-banner {{ grid-template-columns: repeat(2, 1fr); }}
        }}

        /* Pulse */
        @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .pulse {{ animation: pulse 2s infinite; }}

        /* Footer */
        footer {{
            border-top: 1px solid var(--border); padding: 24px 0;
            margin-top: 48px; color: var(--text-dim); font-size: 13px; text-align: center;
        }}
        footer a {{ color: var(--accent); text-decoration: none; }}

        /* Tab navigation */
        .tab-nav {{
            display: flex; gap: 4px; background: var(--surface); border: 1px solid var(--border);
            border-radius: 10px; padding: 4px; margin-bottom: 16px; width: fit-content;
        }}
        .tab-btn {{
            padding: 8px 20px; border-radius: 8px; border: none;
            background: transparent; color: var(--text-dim); font-weight: 600;
            cursor: pointer; font-size: 14px; transition: all 0.2s;
        }}
        .tab-btn.active {{ background: var(--accent); color: white; }}
        .tab-btn:hover:not(.active) {{ background: var(--surface2); color: var(--text); }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
    </style>
</head>
<body>

<header>
    <div class="container header-inner">
        <div class="logo">
            <span class="logo-icon">⚾</span>
            <h1>MLB <span>Predictor</span></h1>
        </div>
        <div class="header-meta">
            <div class="date">{today}</div>
            <div>Updated {updated} · v{APP_VERSION} · Model {model_info.get('version', MODEL_VERSION)}</div>
        </div>
    </div>
</header>

<main class="container">
    {disclaimer_html}

    <!-- Summary Stats -->
    <div class="stats-banner">
        <div class="stat-card">
            <div class="label">Today's Games</div>
            <div class="value value-accent">{len(game_picks)}</div>
            <div class="sub">{'Projected matchups' if is_demo else 'Scheduled games'}</div>
        </div>
        <div class="stat-card">
            <div class="label">Top Hitters Tracked</div>
            <div class="value value-cyan">{len(top_hitters)}</div>
            <div class="sub">Ranked by hit probability</div>
        </div>
        <div class="stat-card">
            <div class="label">Season Record</div>
            <div class="value value-green">{season.get('wins', 0)}-{season.get('losses', 0)}</div>
            <div class="sub">{season.get('win_rate', 0)*100:.1f}% Win Rate</div>
        </div>
        <div class="stat-card">
            <div class="label">Season ROI</div>
            <div class="value value-gold">+{season.get('roi', 0)}%</div>
            <div class="sub">+{season.get('units_profit', 0)} units profit</div>
        </div>
    </div>

    <!-- ════════════════════════════════════════════════════════ -->
    <!-- TODAY'S TOP HITTERS -->
    <!-- ════════════════════════════════════════════════════════ -->
    <section>
        <div class="section-header">
            <h2><span class="icon">🔥</span>Today's Top Hitters</h2>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="badge {mode_badge_class} {'pulse' if not is_demo else ''}">{('● ' if not is_demo else '')}{mode_label}</span>
                <a class="api-link" href="/api/predictions/today/hits">JSON API</a>
            </div>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Batter</th>
                        <th>Pos</th>
                        <th>Hit Prob</th>
                        <th>AVG</th>
                        <th>OPS</th>
                        <th>vs Pitcher (ERA)</th>
                        <th title="Platoon Advantage">PLT</th>
                        <th title="Home Game">Home</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {hitters_rows}
                    {no_hitters_msg}
                </tbody>
            </table>
        </div>
    </section>

    <!-- ════════════════════════════════════════════════════════ -->
    <!-- TODAY'S GAME PICKS -->
    <!-- ════════════════════════════════════════════════════════ -->
    <section>
        <div class="section-header">
            <h2><span class="icon">🏆</span>Today's Game Picks</h2>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span class="badge {mode_badge_class} {'pulse' if not is_demo else ''}">{('● ' if not is_demo else '')}{mode_label}</span>
                <a class="api-link" href="/api/predictions/today/wins">JSON API</a>
            </div>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Game</th>
                        <th>Away (Win %)</th>
                        <th>Home (Win %)</th>
                        <th>Pitchers (ERA)</th>
                        <th>Pick</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {picks_rows}
                    {no_picks_msg}
                </tbody>
            </table>
        </div>
    </section>

    <!-- ════════════════════════════════════════════════════════ -->
    <!-- LEGACY MATCHUP ANALYSIS -->
    <!-- ════════════════════════════════════════════════════════ -->
    <section>
        <div class="section-header">
            <h2><span class="icon">📊</span>Batter-Pitcher Matchup Analysis</h2>
            <div class="api-links">
                <a class="api-link" href="/api/predictions/today">Combined JSON</a>
                <a class="api-link" href="/docs">Swagger Docs</a>
            </div>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Batter</th>
                        <th>vs Pitcher</th>
                        <th>Win %</th>
                        <th>Edge</th>
                        <th>Hit %</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {legacy_rows}
                </tbody>
            </table>
        </div>
    </section>

    <!-- ════════════════════════════════════════════════════════ -->
    <!-- MODEL INFO & API -->
    <!-- ════════════════════════════════════════════════════════ -->
    <section>
        <div class="two-col">
            <div>
                <div class="section-header">
                    <h2><span class="icon">🧠</span>Model Details</h2>
                </div>
                <div class="stat-card" style="text-align: left;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div>
                            <div class="label">Version</div>
                            <div style="font-size: 16px; font-weight: 700; color: var(--accent);">{model_info.get('version', MODEL_VERSION)}</div>
                        </div>
                        <div>
                            <div class="label">Framework</div>
                            <div style="font-size: 16px; font-weight: 700;">{model_info.get('framework', 'XGBoost')}</div>
                        </div>
                    </div>
                    <div style="margin-top: 14px;">
                        <div class="label">Live Features (v2.0)</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">
                            <span style="font-size:12px;padding:3px 10px;border-radius:6px;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);color:var(--green);">Live MLB Schedule</span>
                            <span style="font-size:12px;padding:3px 10px;border-radius:6px;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);color:var(--green);">Hit Probability Model</span>
                            <span style="font-size:12px;padding:3px 10px;border-radius:6px;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);color:var(--green);">Win Probability Model</span>
                            <span style="font-size:12px;padding:3px 10px;border-radius:6px;background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.3);color:var(--green);">2hr Auto-Refresh</span>
                        </div>
                    </div>
                    <div style="margin-top: 14px;">
                        <div class="label">All Features</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">
                            {"".join(f'<span style="font-size:11px;padding:3px 8px;border-radius:6px;background:var(--surface2);border:1px solid var(--border);">{f}</span>' for f in model_info.get("features", []))}
                        </div>
                    </div>
                </div>
            </div>
            <div>
                <div class="section-header">
                    <h2><span class="icon">⚡</span>API Endpoints</h2>
                </div>
                <div class="stat-card" style="text-align: left;">
                    <div style="display: grid; gap: 10px; font-family: monospace; font-size: 13px;">
                        <div style="padding: 8px 14px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border);">
                            <span style="color: var(--green);">GET</span> <span style="color: var(--accent);">/api/predictions/today/hits</span>
                            <span style="color: var(--text-dim); margin-left: 8px; font-size: 11px;">— Top hitters by hit prob</span>
                        </div>
                        <div style="padding: 8px 14px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border);">
                            <span style="color: var(--green);">GET</span> <span style="color: var(--accent);">/api/predictions/today/wins</span>
                            <span style="color: var(--text-dim); margin-left: 8px; font-size: 11px;">— Game win predictions</span>
                        </div>
                        <div style="padding: 8px 14px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border);">
                            <span style="color: var(--green);">GET</span> <span style="color: var(--accent);">/api/predictions/today</span>
                            <span style="color: var(--text-dim); margin-left: 8px; font-size: 11px;">— Combined predictions</span>
                        </div>
                        <div style="padding: 8px 14px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border);">
                            <span style="color: var(--green);">GET</span> <span style="color: var(--accent);">/api/predictions/matchup/NYY/BOS</span>
                            <span style="color: var(--text-dim); margin-left: 8px; font-size: 11px;">— H2H matchup</span>
                        </div>
                        <div style="padding: 8px 14px; background: var(--bg); border-radius: 8px; border: 1px solid var(--border);">
                            <span style="color: var(--green);">GET</span> <span style="color: var(--accent);">/api/leaderboard</span>
                            <span style="color: var(--text-dim); margin-left: 8px; font-size: 11px;">— Historical accuracy</span>
                        </div>
                    </div>
                    <div style="margin-top: 14px;">
                        <a class="api-link" href="/docs" style="font-size: 14px; padding: 8px 20px;">📖 Swagger Docs</a>
                        <a class="api-link" href="/redoc" style="font-size: 14px; padding: 8px 20px; margin-left: 8px;">📚 ReDoc</a>
                    </div>
                </div>
            </div>
        </div>
    </section>

</main>

<footer>
    <div class="container">
        <p>MLB Predictor v{APP_VERSION} · {model_info.get('version', MODEL_VERSION)} ·
        Powered by XGBoost + MLB Stats API ·
        <a href="/docs">API Docs</a> ·
        <a href="/api/health">Health</a></p>
        <p style="margin-top: 8px;">
            🔄 Auto-refreshes every 5 min · Data updates every 2 hours on game days
        </p>
        <p style="margin-top: 4px;">⚠️ For entertainment purposes only. Please gamble responsibly.</p>
    </div>
</footer>

<script>
    // Auto-refresh every 5 minutes
    setTimeout(() => location.reload(), 300000);
</script>

</body>
</html>"""

    return html
