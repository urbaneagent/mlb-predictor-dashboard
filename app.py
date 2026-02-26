"""
MLB Predictor - Main API Application
Unified FastAPI server that ties together all prediction modules.
"""
import os
import logging
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MLB Predictor API",
    description="AI-powered MLB game predictions with live odds, bankroll management, and performance tracking.",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Health & Info =============

@app.get("/")
async def root():
    return {
        "name": "MLB Predictor API",
        "version": "5.0.0",
        "status": "operational",
        "features": [
            "Live odds tracking (The Odds API)",
            "Game state engine (MLB StatsAPI - pitch by pitch)",
            "Injury & news intelligence",
            "Bankroll management (Kelly Criterion)",
            "Performance tracking (ROI, CLV, Sharpe)",
            "Picks delivery (Telegram, email, CSV)",
            "Stadium & weather factors",
            "Umpire bias analysis",
            "Parlay builder",
            "Prop bet analyzer",
            "Season simulator",
            "REST API + WebSocket"
        ],
        "docs": "/docs",
        "dashboard": "/dashboard"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ============= Dashboard =============

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the picks dashboard."""
    dashboard_path = Path(__file__).parent / "picks_dashboard.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)


# ============= Register Module Routes =============

def register_modules():
    """Register all feature modules with the FastAPI app."""

    # 1. Game State Engine (Live games)
    try:
        from game_state_engine import GameStateEngine, create_game_state_routes
        engine = GameStateEngine()
        create_game_state_routes(app, engine)
        logger.info("✅ Game State Engine registered")
    except ImportError as e:
        logger.warning(f"⚠️ Game State Engine not loaded: {e}")

    # 2. Live Odds Tracker
    try:
        from live_odds_tracker import LiveOddsTracker
        odds_api_key = os.environ.get("ODDS_API_KEY", "")
        odds_tracker = LiveOddsTracker(api_key=odds_api_key)
        logger.info("✅ Live Odds Tracker registered")
    except ImportError as e:
        logger.warning(f"⚠️ Live Odds Tracker not loaded: {e}")

    # 3. Injury & News Tracker
    try:
        from injury_news_tracker import InjuryNewsTracker, create_injury_routes
        injury_tracker = InjuryNewsTracker()
        create_injury_routes(app, injury_tracker)
        logger.info("✅ Injury & News Tracker registered")
    except ImportError as e:
        logger.warning(f"⚠️ Injury Tracker not loaded: {e}")

    # 4. Bankroll Manager
    try:
        from bankroll_manager import BankrollManager
        bankroll = BankrollManager(starting_bankroll=1000, unit_size=10)
        logger.info("✅ Bankroll Manager registered")
    except ImportError as e:
        logger.warning(f"⚠️ Bankroll Manager not loaded: {e}")

    # 5. Performance Tracker
    try:
        from performance_tracker import PerformanceTracker, create_performance_routes
        perf_tracker = PerformanceTracker()
        create_performance_routes(app, perf_tracker)
        logger.info("✅ Performance Tracker registered")
    except ImportError as e:
        logger.warning(f"⚠️ Performance Tracker not loaded: {e}")

    # 6. Picks Delivery
    try:
        from picks_delivery import PicksFormatter, generate_sample_picks
        formatter = PicksFormatter()

        @app.get("/api/v1/picks/today")
        async def get_todays_picks(format: str = "json"):
            card = generate_sample_picks()
            if format == "telegram":
                return {"text": formatter.format_telegram(card)}
            elif format == "csv":
                return {"csv": formatter.format_csv(card)}
            elif format == "html":
                return HTMLResponse(formatter.format_email_html(card))
            else:
                return formatter.format_dashboard_json(card)

        logger.info("✅ Picks Delivery registered")
    except ImportError as e:
        logger.warning(f"⚠️ Picks Delivery not loaded: {e}")

    # 7. Stadium Factors
    try:
        from stadium_factors import StadiumFactorEngine
        stadium_engine = StadiumFactorEngine()

        @app.get("/api/v1/stadiums/{stadium}")
        async def get_stadium_factors(stadium: str):
            return stadium_engine.get_factors(stadium)

        logger.info("✅ Stadium Factors registered")
    except ImportError as e:
        logger.warning(f"⚠️ Stadium Factors not loaded: {e}")

    # 8. Umpire Bias
    try:
        from umpire_bias_engine import UmpireBiasEngine
        umpire_engine = UmpireBiasEngine()

        @app.get("/api/v1/umpires/{umpire_name}")
        async def get_umpire_stats(umpire_name: str):
            return umpire_engine.get_umpire_profile(umpire_name)

        logger.info("✅ Umpire Bias Engine registered")
    except ImportError as e:
        logger.warning(f"⚠️ Umpire Bias Engine not loaded: {e}")

    # 9. Prediction API
    try:
        from prediction_api import PredictionEngine
        prediction_engine = PredictionEngine()

        @app.get("/api/v1/predict/{game_id}")
        async def predict_game(game_id: int):
            return prediction_engine.predict(game_id)

        logger.info("✅ Prediction Engine registered")
    except ImportError as e:
        logger.warning(f"⚠️ Prediction Engine not loaded: {e}")

    # 10. Parlay Builder
    try:
        from parlay_builder import ParlayBuilder
        parlay = ParlayBuilder()

        @app.post("/api/v1/parlays/build")
        async def build_parlay(picks: list):
            return parlay.build(picks)

        logger.info("✅ Parlay Builder registered")
    except ImportError as e:
        logger.warning(f"⚠️ Parlay Builder not loaded: {e}")


# Register all modules
register_modules()


# ============= Error Handling =============

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return {"error": str(exc), "path": str(request.url)}


# ============= Run =============

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
