"""
Performance monitoring service using Prometheus metrics.
"""
from prometheus_client import Counter, Histogram, Gauge, Summary
from functools import wraps
import time
from typing import Callable
from loguru import logger


# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query latency',
    ['operation', 'table']
)

cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']  # result: hit, miss, set, delete
)

simulation_tasks_processed = Counter(
    'simulation_tasks_processed_total',
    'Total simulation tasks processed'
)

simulation_tasks_in_flight = Gauge(
    'simulation_tasks_in_flight',
    'Current number of tasks being processed'
)

model_retraining_duration_seconds = Summary(
    'model_retraining_duration_seconds',
    'Model retraining duration'
)

websocket_connections = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections'
)


def track_time(metric: Histogram, **labels):
    """
    Decorator to track execution time of async functions.
    
    Usage:
        @track_time(database_query_duration_seconds, operation='select', table='tasks')
        async def get_tasks():
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        return wrapper
    return decorator


def track_time_sync(metric: Histogram, **labels):
    """
    Decorator to track execution time of sync functions.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metric.labels(**labels).observe(duration)
        return wrapper
    return decorator


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    @staticmethod
    def record_cache_hit(namespace: str):
        """Record a cache hit."""
        cache_operations_total.labels(operation='get', result='hit').inc()
        logger.debug(f"Cache hit: {namespace}")
    
    @staticmethod
    def record_cache_miss(namespace: str):
        """Record a cache miss."""
        cache_operations_total.labels(operation='get', result='miss').inc()
        logger.debug(f"Cache miss: {namespace}")
    
    @staticmethod
    def record_cache_set(namespace: str):
        """Record a cache set operation."""
        cache_operations_total.labels(operation='set', result='success').inc()
    
    @staticmethod
    def record_task_processed():
        """Record a simulation task processed."""
        simulation_tasks_processed.inc()
    
    @staticmethod
    def set_websocket_connections(count: int):
        """Update WebSocket connections gauge."""
        websocket_connections.set(count)
    
    @staticmethod
    async def track_db_query(operation: str, table: str, query_func: Callable):
        """Track database query performance."""
        start_time = time.time()
        try:
            result = await query_func()
            duration = time.time() - start_time
            database_query_duration_seconds.labels(
                operation=operation,
                table=table
            ).observe(duration)
            
            if duration > 1.0:
                logger.warning(f"Slow query: {operation} on {table} took {duration:.2f}s")
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Query failed after {duration:.2f}s: {e}")
            raise


# Global monitor instance
performance_monitor = PerformanceMonitor()
