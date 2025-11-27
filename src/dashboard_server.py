"""FastAPI WebSocket Server for Hybrid ML Scheduler Dashboard.

This module implements the backend API server that:
- Manages WebSocket connections for real-time dashboard updates
- Hosts REST endpoints for historical data and simulation control
- Orchestrates the continuous simulation engine
- Broadcasts scheduler performance metrics to connected clients

The server runs on http://localhost:8000 by default and provides:
- WebSocket endpoint: ws://localhost:8000/ws
- REST API: http://localhost:8000/api/*

Typical Usage:
    Run with uvicorn:
    ```bash
    uvicorn src.dashboard_server:app --host 0.0.0.0 --port 8000
    ```
"""
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from loguru import logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    """Manages WebSocket connections for live dashboard updates.
    
    This class maintains a list of active WebSocket connections and provides
    methods to connect, disconnect, and broadcast messages to all clients.
    
    Attributes:
        active_connections (List[WebSocket]): List of currently connected WebSocket clients
    """
    
    def __init__(self):
        """Initialize the connection manager with an empty connection list."""
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection.
        
        Args:
            websocket (WebSocket): The WebSocket connection to accept
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from the active list.
        
        Args:
            websocket (WebSocket): The WebSocket connection to remove
        """
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send a JSON message to all connected WebSocket clients.
        
        Args:
            message (dict): The message dictionary to broadcast as JSON
        """
        logger.info(f"Broadcasting to {len(self.active_connections)} connections")
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

manager = ConnectionManager()

# Store history in memory (limit to last 1000 points)
history = []
MAX_HISTORY = 1000

@app.get("/")
async def root():
    return {"status": "online", "message": "Hybrid ML Scheduler Dashboard API is running"}

@app.get("/api/history")
async def get_history():
    return history

@app.get("/api/full_history")
async def get_full_history():
    """Read full history from CSV"""
    try:
        if simulation.history_file.exists():
            import pandas as pd
            df = pd.read_csv(simulation.history_file)
            return df.to_dict(orient='records')
        return []
    except Exception as e:
        logger.error(f"Error reading history: {e}")
        return []

@app.delete("/api/history")
async def clear_history():
    """Clear the history CSV"""
    try:
        if simulation.history_file.exists():
            # Keep header
            import pandas as pd
            df = pd.read_csv(simulation.history_file, nrows=0)
            df.to_csv(simulation.history_file, index=False)
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def broadcast_sync(message: dict):
    """Helper to broadcast from synchronous code via async loop"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # If loop is running (e.g. inside the server process), create task
        asyncio.create_task(manager.broadcast(message))
    else:
        # If loop is not running (unlikely if called from server), run until complete
        loop.run_until_complete(manager.broadcast(message))

async def process_message(message: dict):
    """Process and store message before broadcasting"""
    # Add to history
    history.append(message)
    if len(history) > MAX_HISTORY:
        history.pop(0)
    
    await manager.broadcast(message)

async def consume_kafka_messages():
    """Background task to consume messages from Kafka"""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            'scheduler_events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
        logger.info("Kafka Consumer started")
        
        # Non-blocking loop
        # We use a small timeout to yield control back to event loop
        while True:
            # Poll for messages (returns a dict of {TopicPartition: [messages]})
            # timeout_ms=100 makes it non-blocking enough
            message_batch = consumer.poll(timeout_ms=100)
            
            for tp, messages in message_batch.items():
                for message in messages:
                    await process_message(message.value)
            
            # Yield control
            await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Kafka Consumer Error: {e}")
        # Retry logic could go here
        await asyncio.sleep(5)
        # Restart
        asyncio.create_task(consume_kafka_messages())

from src.simulation_engine import ContinuousSimulation

# Global Simulation Instance
simulation = ContinuousSimulation(broadcast_callback=manager.broadcast)

@app.post("/api/pause")
async def pause_simulation():
    simulation.pause()
    return {"status": "paused"}

@app.post("/api/resume")
async def resume_simulation():
    simulation.resume()
    return {"status": "resumed"}

@app.on_event("startup")
async def startup_event():
    # Start the continuous simulation engine
    asyncio.create_task(simulation.start())
