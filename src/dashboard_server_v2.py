"""
Modernized Dashboard Server with Modular API Architecture.

This is the new main entry point that uses the refactored API routes.
"""
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from backend.core.config import settings
from backend.core.database import init_db, close_db
from backend.core.redis import redis_client
from backend.api.routes import health, websocket, simulation, metrics, observability
from backend.__version__ import __version__
from backend.middleware.observability import ObservabilityMiddleware, ErrorTrackingMiddleware
from backend.middleware.rate_limit import RateLimitMiddleware
from backend.middleware.security import SecurityHeadersMiddleware
from backend.services.logging_service import setup_logging
from src.simulation_engine import ContinuousSimulation


# Global simulation engine
simulation = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Hybrid ML Scheduler API...")
    
    # Setup structured logging
    setup_logging(environment=settings.environment)
    logger.info(f"✅ Logging configured for {settings.environment} environment")
    
    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    # Connect to Redis
    try:
        await redis_client.connect()
        logger.info("✅ Redis connected")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed (degraded mode): {e}")
    
    # Initialize simulation engine
    global simulation
    sim_engine = ContinuousSimulation(broadcast_callback=websocket.manager.broadcast)
    simulation.set_simulation_engine(sim_engine)
    simulation = sim_engine
    logger.info("✅ Simulation engine initialized")
    
    # Start simulation by default
    asyncio.create_task(sim_engine.start())
    logger.info("✅ Simulation started")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    
    if simulation:
        simulation.stop()
        logger.info("✅ Simulation stopped")
    
    await close_db()
    await redis_client.disconnect()
    logger.info("✅ Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Hybrid ML Scheduler API",
    description="Real-time GPU scheduling simulation with ML optimization",
    version=__version__,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware (applied first)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# Observability middleware
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(ErrorTrackingMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(websocket.router)
app.include_router(simulation.router)
app.include_router(metrics.router)
app.include_router(observability.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Hybrid ML Scheduler API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard_server_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
