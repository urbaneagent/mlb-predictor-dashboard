#!/usr/bin/env python3
"""
Entry point for the MLB Predictor Web Application.
Run: python run_webapp.py
Or:  uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from webapp.main import start

if __name__ == "__main__":
    start()
