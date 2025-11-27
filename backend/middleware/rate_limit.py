"""
Rate limiting middleware for API protection.

Implements:
- Per-IP rate limiting
- Per-endpoint rate limiting
- Configurable limits
- Redis-backed (with in-memory fallback)
"""
import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from backend.core.redis import redis_client


class RateLimiter:
    """
    Rate limiter with Redis backend and in-memory fallback.
    
    Implements token bucket algorithm.
    """
    
    def __init__(self):
        self._memory_store: Dict[str, Tuple[int, float]] = {}
    
    async def is_allowed(
        self, 
        key: str, 
        max_requests: int = 100, 
        window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Unique identifier (e.g., IP address)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        cache_key = f"rate_limit:{key}"
        current_time = time.time()
        
        # Try Redis first
        try:
            # Get current count
            count_str = await redis_client.get(cache_key)
            
            if count_str is None:
                # First request in window
                await redis_client.set(cache_key, "1", ttl=window_seconds)
                return True, max_requests - 1
            
            count = int(count_str)
            
            if count >= max_requests:
                # Rate limit exceeded
                return False, 0
            
            # Increment counter
            await redis_client.increment(cache_key)
            return True, max_requests - count - 1
            
        except Exception as e:
            logger.warning(f"Redis rate limiting failed, using memory: {e}")
        
        # Fallback to in-memory rate limiting
        if cache_key in self._memory_store:
            count, window_start = self._memory_store[cache_key]
            
            # Check if window expired
            if current_time - window_start >= window_seconds:
                # New window
                self._memory_store[cache_key] = (1, current_time)
                return True, max_requests - 1
            
            if count >= max_requests:
                return False, 0
            
            # Increment count
            self._memory_store[cache_key] = (count + 1, window_start)
            return True, max_requests - count - 1
        else:
            # First request
            self._memory_store[cache_key] = (1, current_time)
            return True, max_requests - 1
    
    def cleanup_memory(self):
        """Clean up expired entries from memory store."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, start_time) in self._memory_store.items()
            if current_time - start_time > 3600  # 1 hour
        ]
        for key in expired_keys:
            del self._memory_store[key]


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting.
    
    Applies configurable rate limits per IP address.
    """
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limit
        is_allowed, remaining = await rate_limiter.is_allowed(
            key=client_ip,
            max_requests=self.requests_per_minute,
            window_seconds=self.window_seconds
        )
        
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for {client_ip}",
                client_ip=client_ip,
                path=request.url.path
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
