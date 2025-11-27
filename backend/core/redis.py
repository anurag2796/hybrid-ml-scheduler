"""
Redis connection and caching utilities.
"""
from redis import asyncio as aioredis
from typing import Optional, Any
import json
from loguru import logger

from backend.core.config import settings


class RedisClient:
    """Async Redis client wrapper with caching utilities."""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis = await aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
            )
            # Test connection
            await self.redis.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis = None
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis disconnected")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache with optional TTL."""
        if not self.redis:
            return
        
        try:
            ttl = ttl or settings.cache_ttl
            await self.redis.setex(
                key,
                ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
    
    async def delete(self, key: str):
        """Delete key from cache."""
        if not self.redis:
            return
        
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.redis:
            return False
        
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False
    
    async def lpush(self, key: str, *values):
        """Push values to list (left)."""
        if not self.redis:
            return
        
        try:
            await self.redis.lpush(key, *[json.dumps(v, default=str) for v in values])
        except Exception as e:
            logger.error(f"Redis LPUSH error: {e}")
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list:
        """Get range from list."""
        if not self.redis:
            return []
        
        try:
            values = await self.redis.lrange(key, start, end)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.error(f"Redis LRANGE error: {e}")
            return []
    
    async def ltrim(self, key: str, start: int, end: int):
        """Trim list to specified range."""
        if not self.redis:
            return
        
        try:
            await self.redis.ltrim(key, start, end)
        except Exception as e:
            logger.error(f"Redis LTRIM error: {e}")
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter."""
        if not self.redis:
            return 0
        
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCR error: {e}")
            return 0


# Global Redis client instance
redis_client = RedisClient()


async def get_redis() -> RedisClient:
    """Dependency for getting Redis client."""
    return redis_client
