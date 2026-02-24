
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to python path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.api.routes import health, metrics, simulation, websocket, observability
from src.simulation_engine import ContinuousSimulation

# Initialize Simulation Engine
simulation_instance = ContinuousSimulation(websocket.manager.broadcast)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Hybrid ML Scheduler Backend...")
    await init_db()
    
    # Inject simulation engine into routes
    simulation.set_simulation_engine(simulation_instance)
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if simulation_instance.is_running:
        simulation_instance.stop()
    await close_db()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(simulation.router)
app.include_router(websocket.router)
app.include_router(observability.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.server:app", host="0.0.0.0", port=8000, reload=True)
