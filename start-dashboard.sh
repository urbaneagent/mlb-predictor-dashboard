#!/bin/bash
# MLB Predictor Dashboard - Start Script
# Starts the FastAPI server and Cloudflare tunnel

cd /Users/mikeross/.openclaw/workspace/projects/mlb-predictor
source venv/bin/activate

export MLB_DATA_DIR="/Users/mikeross/MLB_Predictions"
export MLB_HOST="0.0.0.0"
export MLB_PORT="8765"

# Start FastAPI server
uvicorn webapp.main:app --host 0.0.0.0 --port 8765 &
SERVER_PID=$!
echo "Server started (PID: $SERVER_PID)"

# Wait for server to be ready
sleep 3

# Start Cloudflare tunnel
cloudflared tunnel --url http://localhost:8765 &
TUNNEL_PID=$!
echo "Tunnel started (PID: $TUNNEL_PID)"

# Write PIDs for cleanup
echo "$SERVER_PID" > /tmp/mlb-server.pid
echo "$TUNNEL_PID" > /tmp/mlb-tunnel.pid

wait
