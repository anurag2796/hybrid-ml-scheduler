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
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
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
