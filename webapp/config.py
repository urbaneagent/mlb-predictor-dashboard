"""
Configuration for MLB Predictor Web App
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("MLB_DATA_DIR", str(PROJECT_ROOT / "data")))
DAILY_PREDICTIONS_CSV = DATA_DIR / "daily_predictions.csv"
FINAL_PREDICTIONS_CSV = DATA_DIR / "FINAL_predictions.csv"
MODELS_DIR = DATA_DIR / "models"

# App settings
APP_TITLE = "MLB Predictor"
APP_VERSION = "2.0.0"
MODEL_VERSION = "v125 (XGBoost Ensemble + Live Pipeline)"
DEBUG = os.getenv("MLB_DEBUG", "false").lower() == "true"
HOST = os.getenv("MLB_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("MLB_PORT", "8000")))

# Team full names
TEAM_NAMES = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles", "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies", "DET": "Detroit Tigers",
    "HOU": "Houston Astros", "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins", "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "ATH": "Athletics",
    "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres", "SF": "San Francisco Giants",
    "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
}

# Team alias normalization (CSV uses different abbrevs)
TEAM_ALIASES = {
    "SFG": "SF", "TBR": "TB", "WSN": "WSH", "KCR": "KC", "OAK": "ATH",
    "AZ": "ARI", "CHW": "CWS", "CWH": "CWS",
    # Reverse mappings too
    "SF": "SFG", "TB": "TBR", "WSH": "WSN", "KC": "KCR", "ATH": "OAK",
}

# Confidence tiers
def get_confidence(edge: float) -> str:
    """Convert edge percentage to confidence label."""
    if edge >= 0.08:
        return "🔒 LOCK"
    elif edge >= 0.05:
        return "💪 Strong"
    elif edge >= 0.03:
        return "👀 Lean"
    else:
        return "📊 Value"
