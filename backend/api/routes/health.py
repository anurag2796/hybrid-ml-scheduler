"""
Health check endpoints for monitoring system status.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from loguru import logger

from backend.core.database import engine
from backend.core.redis import redis_client
from backend.core.config import settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness check - verifies all dependencies are available.
    
    Returns 200 if ready, 503 if not ready.
    """
    checks = {
        "database": "unknown",
        "redis": "unknown"
    }
    
    # Check database
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
        logger.error(f"Database health check failed: {e}")
    
    # Check Redis
    try:
        await redis_client.connect()
        await redis_client.set("health_check", "ok", ttl=10)
        value = await redis_client.get("health_check")
        if value == "ok":
            checks["redis"] = "healthy"
        else:
            checks["redis"] = "unhealthy: test failed"
    except Exception as e:
        checks["redis"] = f"degraded: {str(e)}"
        logger.warning(f"Redis health check failed (degraded mode): {e}")
    
    # Determine overall status
    is_ready = checks["database"] == "healthy"
    status_code = 200 if is_ready else 503
    
    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/live")
async def liveness_check():
    """
    Liveness check - verifies the application is running.
    
    This is a simple endpoint that should always return 200
    unless the application is completely crashed.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/info")
async def system_info():
    """Get system information."""
    return {
        "app_name": "Hybrid ML Scheduler",
        "version": "1.0.0",
        "environment": settings.environment,
        "debug": settings.debug,
        "database": {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "database": settings.postgres_db,
        },
        "redis": {
            "host": settings.redis_host,
            "port": settings.redis_port,
            "enabled": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }
