"""
Caching service for performance optimization.

Provides Redis-backed caching with automatic fallback to in-memory cache.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
from loguru import logger

from backend.core.redis import redis_client


class CacheService:
    """
    Centralized caching service with Redis backend and in-memory fallback.
    
    Features:
    - Automatic serialization/deserialization
    - TTL support
    - In-memory fallback when Redis unavailable
    - Namespace support for organized caching
    """
    
    def __init__(self):
        self._memory_cache: Dict[str, tuple[Any, Optional[datetime]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced cache key."""
        return f"{namespace}:{key}"
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            namespace: Cache namespace (e.g., 'metrics', 'leaderboard')
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._make_key(namespace, key)
        
        # Try Redis first
        try:
            value = await redis_client.get(cache_key)
            if value is not None:
                self._cache_hits += 1
                logger.debug(f"Cache HIT (Redis): {cache_key}")
                return json.loads(value) if isinstance(value, str) else value
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        
        # Fallback to memory cache
        if cache_key in self._memory_cache:
            value, expires_at = self._memory_cache[cache_key]
            if expires_at is None or expires_at >= datetime.utcnow():
                self._cache_hits += 1
                logger.debug(f"Cache HIT (Memory): {cache_key}")
                return value
            else:
                # Expired, remove it
                del self._memory_cache[cache_key]
        
        self._cache_misses += 1
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    async def set(
        self, 
        namespace: str, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (None = no expiry)
            
        Returns:
            True if successful
        """
        cache_key = self._make_key(namespace, key)
        
        # Try Redis first
        try:
            serialized = json.dumps(value)
            await redis_client.set(cache_key, serialized, ttl=ttl)
            logger.debug(f"Cache SET (Redis): {cache_key}, ttl={ttl}")
            return True
        except Exception as e:
            logger.warning(f"Redis set failed: {e}, using memory cache")
        
        # Fallback to memory cache
        expires_at = None
        if ttl is not None:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self._memory_cache[cache_key] = (value, expires_at)
        logger.debug(f"Cache SET (Memory): {cache_key}, ttl={ttl}")
        return True
    
    async def delete(self, namespace: str, key: str) -> bool:
        """Delete value from cache."""
        cache_key = self._make_key(namespace, key)
        
        # Delete from Redis
        try:
            await redis_client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")
        
        # Delete from memory
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        logger.debug(f"Cache DELETE: {cache_key}")
        return True
    
    async def invalidate_namespace(self, namespace: str) -> int:
        """
        Invalidate all keys in a namespace.
        
        Args:
            namespace: Namespace to invalidate
            
        Returns:
            Number of keys deleted
        """
        # For memory cache, filter and delete
        keys_to_delete = [
            k for k in self._memory_cache.keys() 
            if k.startswith(f"{namespace}:")
        ]
        
        for key in keys_to_delete:
            del self._memory_cache[key]
        
        logger.info(f"Invalidated {len(keys_to_delete)} keys in namespace '{namespace}'")
        return len(keys_to_delete)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            (self._cache_hits / total_requests * 100) 
            if total_requests > 0 
            else 0
        )
        
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self._memory_cache)
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0


# Global cache service instance
cache_service = CacheService()
