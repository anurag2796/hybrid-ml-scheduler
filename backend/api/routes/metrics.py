"""
Metrics and monitoring endpoints.
"""
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from backend.services.cache_service import cache_service

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Exposes all collected metrics in Prometheus format.
    """
    metrics = generate_latest()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)


@router.get("/cache")
async def cache_stats():
    """
    Get cache performance statistics.
    
    Returns hit/miss rates and cache size.
    """
    stats = cache_service.get_stats()
    return {
        "cache_stats": stats,
        "status": "operational"
    }


@router.post("/cache/reset")
async def reset_cache_stats():
    """Reset cache statistics (not the cache itself)."""
    cache_service.reset_stats()
    return {"status": "stats_reset"}


@router.post("/cache/invalidate/{namespace}")
async def invalidate_cache_namespace(namespace: str):
    """
    Invalidate all cache entries in a namespace.
    
    Args:
        namespace: Cache namespace to invalidate (e.g., 'metrics', 'leaderboard')
    """
    count = await cache_service.invalidate_namespace(namespace)
    return {
        "status": "invalidated",
        "namespace": namespace,
        "keys_deleted": count
    }
