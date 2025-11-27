"""
FastAPI middleware for observability.

Provides:
- Request correlation IDs
- Request/response logging
- Performance tracking
- Error tracking
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from loguru import logger

from backend.services.logging_service import set_correlation_id, clear_correlation_id, log_performance
from backend.services.performance_service import (
    http_requests_total,
    http_request_duration_seconds
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request observability.
    
    Tracks:
    - Correlation IDs
    - Request/response logging
    - Performance metrics
    - Error tracking
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        correlation_id = set_correlation_id(correlation_id)
        
        # Start timer
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"→ {request.method} {request.url.path}",
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        # Process request
        response = None
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as e:
            logger.error(
                f"Request failed: {type(e).__name__}: {str(e)}",
                error_type=type(e).__name__,
                error_message=str(e),
                path=request.url.path,
                method=request.method
            )
            raise
            
        finally:
            # Calculate duration
            duration = time.time() - start_time
            duration_ms = duration * 1000
            
            # Log response
            logger.info(
                f"← {request.method} {request.url.path} {status_code} ({duration_ms:.2f}ms)",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms
            )
            
            # Record metrics
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=str(status_code)
            ).inc()
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log slow requests
            if duration_ms > 1000:
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} took {duration_ms:.2f}ms",
                    duration_ms=duration_ms,
                    threshold_ms=1000
                )
            
            # Clear correlation ID
            clear_correlation_id()


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for error tracking and reporting.
    
    Catches unhandled exceptions and logs them with full context.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Log error with full context
            logger.exception(
                f"Unhandled exception: {type(e).__name__}",
                error_type=type(e).__name__,
                error_message=str(e),
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else "unknown"
            )
            
            # Re-raise to let FastAPI handle it
            raise
