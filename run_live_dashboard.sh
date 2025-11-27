#!/bin/bash

# Kill any existing processes
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start Backend (Python FastAPI)
echo "Starting Python Backend (Mock Mode if Kafka missing)..."
python3 -m uvicorn src.dashboard_server:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start Frontend
echo "Starting Frontend Dashboard..."
cd dashboard
npm run dev &
FRONTEND_PID=$!

# Trap Ctrl+C to kill all
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

echo "Dashboard is live at http://localhost:5173"
echo "Press Ctrl+C to stop."

wait
