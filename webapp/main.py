"""
MLB Predictor — Production Web App & API
==========================================
FastAPI application serving LIVE daily predictions, matchup analysis,
leaderboard, and a professional HTML dashboard.

NEW Live Endpoints (v2.0):
    GET  /api/predictions/today/hits  → Today's top hitters by hit probability
    GET  /api/predictions/today/wins  → Today's game picks with win probability
    GET  /api/predictions/today       → Combined: hits + wins

Legacy Endpoints (preserved):
    GET  /                              → HTML Dashboard
    GET  /api/predictions/matchup/{a}/{h} → Specific matchup from CSV data
    GET  /api/leaderboard               → Historical accuracy
    GET  /api/model                     → Model info
    GET  /api/health                    → Health check
    GET  /docs                          → Swagger UI
"""

import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import APP_TITLE, APP_VERSION, MODEL_VERSION, HOST, PORT, DEBUG, TEAM_NAMES
from .data_loader import data_loader
from .dashboard import render_dashboard
from .live_predictions import live_predictions

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title=f"{APP_TITLE} API",
    description=(
        "AI-powered MLB game predictions with live daily data from MLB Stats API. "
        "Features hit probability rankings, team win predictions, "
        "confidence ratings, and historical accuracy tracking."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "MLB Predictor", "url": "https://mlb-predictor.up.railway.app"},
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── HTML Dashboard ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Landing page — professional dashboard with live predictions."""
    try:
        combined = live_predictions.get_combined()
        hits_data = combined.get("top_hitters", {})
        wins_data = combined.get("game_picks", {})
    except Exception as e:
        logger.error(f"Error generating live predictions: {e}")
        hits_data = {"top_hitters": [], "mode": "error"}
        wins_data = {"predictions": [], "mode": "error"}

    # Legacy data for leaderboard/model info
    leaderboard = data_loader.get_leaderboard()
    model_info = data_loader.get_model_info()

    # Legacy picks (fallback if live fails)
    legacy_picks = data_loader.get_todays_picks(top_n=15)

    return render_dashboard(
        picks=legacy_picks,
        leaderboard=leaderboard,
        model_info=model_info,
        hits_data=hits_data,
        wins_data=wins_data,
    )


# ══════════════════════════════════════════════════════════════
# NEW LIVE ENDPOINTS (v2.0)
# ══════════════════════════════════════════════════════════════

@app.get("/api/predictions/today/hits", tags=["Live Predictions"])
async def predictions_today_hits(
    top_n: int = Query(default=30, ge=1, le=90, description="Number of hitters to return"),
):
    """
    🔥 Today's Top Hitters — ranked by probability of recording at least 1 hit.
    
    Factors: batting average, platoon splits, pitcher quality, park factor, lineup position.
    During off-season, returns preseason projections based on 2025 stats.
    """
    try:
        result = live_predictions.get_todays_hits(top_n=top_n)
        return result
    except Exception as e:
        logger.error(f"Error in /today/hits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/today/wins", tags=["Live Predictions"])
async def predictions_today_wins():
    """
    🏆 Today's Game Picks — win probability for each game.
    
    Factors: starting pitcher quality (ERA/WHIP/K9), lineup OPS,
    bullpen strength, home field advantage, park factor.
    During off-season, returns preseason projections based on 2025 stats.
    """
    try:
        result = live_predictions.get_todays_wins()
        return result
    except Exception as e:
        logger.error(f"Error in /today/wins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/today", tags=["Live Predictions"])
async def predictions_today_combined(
    top_n: int = Query(default=15, ge=1, le=50, description="Number of picks to return"),
    min_edge: float = Query(default=0.0, ge=0, le=0.5, description="Minimum edge filter"),
    sort_by: str = Query(default="edge", description="Sort field: edge, win_prob, value_score"),
):
    """
    📊 Today's Combined Predictions — top hitters + game picks.
    
    Returns both hit probability rankings and team win predictions.
    Falls back to CSV-based predictions when no live games are available.
    During off-season, returns preseason projections based on 2025 stats.
    """
    try:
        combined = live_predictions.get_combined()

        # Also include legacy CSV picks as fallback/supplemental data
        legacy_picks = data_loader.get_todays_picks(top_n=top_n, sort_by=sort_by)
        if min_edge > 0:
            legacy_picks = [p for p in legacy_picks if p.get("edge", 0) >= min_edge]

        combined["legacy_picks"] = legacy_picks
        combined["model_version"] = MODEL_VERSION

        return combined

    except Exception as e:
        logger.error(f"Error in /today combined: {e}")
        # Fallback to legacy
        picks = data_loader.get_todays_picks(top_n=top_n, sort_by=sort_by)
        if min_edge > 0:
            picks = [p for p in picks if p.get("edge", 0) >= min_edge]
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model_version": MODEL_VERSION,
            "total_picks": len(picks),
            "picks": picks,
            "mode": "legacy_fallback",
        }


# ══════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS (preserved for backward compatibility)
# ══════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "operational",
        "version": APP_VERSION,
        "model": MODEL_VERSION,
        "mode": "live" if live_predictions.is_regular_season else "preseason",
        "season_starts": "2026-03-27",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/predictions/matchup/{away_team}/{home_team}", tags=["Legacy"])
async def predictions_matchup(away_team: str, home_team: str):
    """
    Detailed matchup analysis between two teams (from CSV data).
    Provide 2-3 letter team abbreviations (e.g., NYY, BOS, LAD).
    """
    away = away_team.upper()
    home = home_team.upper()

    if away not in TEAM_NAMES and away not in [t for t in TEAM_NAMES]:
        raise HTTPException(status_code=404, detail=f"Unknown team: {away_team}")
    if home not in TEAM_NAMES and home not in [t for t in TEAM_NAMES]:
        raise HTTPException(status_code=404, detail=f"Unknown team: {home_team}")

    result = data_loader.get_matchup(away, home)
    result["date"] = datetime.now().strftime("%Y-%m-%d")
    result["model_version"] = MODEL_VERSION
    return result


@app.get("/api/leaderboard", tags=["Performance"])
async def leaderboard():
    """Historical accuracy tracking and model performance."""
    return data_loader.get_leaderboard()


@app.get("/api/model", tags=["System"])
async def model_info():
    """Model metadata, version, features, and data sources."""
    info = data_loader.get_model_info()
    info["live_features"] = [
        "Live MLB schedule + lineups",
        "Hit probability model (platoon, park, pitcher quality)",
        "Team win probability (ERA, WHIP, K/9, OPS, bullpen, HFA)",
        "2-hour cache with auto-refresh",
        "Preseason demo mode with 2025 stats",
    ]
    return info


@app.get("/api/teams", tags=["Reference"])
async def teams():
    """List all MLB team abbreviations and full names."""
    return {"teams": [{"abbr": k, "name": v} for k, v in sorted(TEAM_NAMES.items())]}


# ── Run ──────────────────────────────────────────────────────

def start():
    """Entry point for running the server."""
    import uvicorn
    logger.info(f"Starting {APP_TITLE} API v{APP_VERSION} on {HOST}:{PORT}")
    uvicorn.run(
        "webapp.main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info",
    )
