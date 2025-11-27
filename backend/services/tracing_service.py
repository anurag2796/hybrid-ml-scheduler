"""
Distributed tracing with OpenTelemetry.

Provides request tracing across services and database calls.
"""
from typing import Callable, Optional
from functools import wraps
import time
from loguru import logger

# We'll use a lightweight tracing approach since OpenTelemetry full setup
# requires additional infrastructure. This provides the core tracing functionality.


class Span:
    """
    Lightweight span implementation for distributed tracing.
    
    In production, this would be replaced with OpenTelemetry spans.
    """
    
    def __init__(self, name: str, parent_id: Optional[str] = None):
        self.name = name
        self.span_id = str(time.time_ns())
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = None
        self.attributes = {}
        self.events = []
        self.status = "ok"
    
    def set_attribute(self, key: str, value):
        """Set a span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_status(self, status: str):
        """Set span status (ok, error)."""
        self.status = status
    
    def end(self):
        """End the span."""
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        logger.debug(
            f"Span completed: {self.name}",
            span_id=self.span_id,
            parent_id=self.parent_id,
            duration_ms=duration_ms,
            status=self.status,
            attributes=self.attributes
        )
        
        return duration_ms


class Tracer:
    """
    Lightweight tracer for distributed tracing.
    
    Provides basic tracing functionality with span management.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.current_span = None
    
    def start_span(self, name: str, parent_id: Optional[str] = None) -> Span:
        """Start a new span."""
        span = Span(name, parent_id)
        self.current_span = span
        logger.debug(f"Started span: {name}", span_id=span.span_id)
        return span
    
    def end_span(self, span: Span):
        """End a span."""
        duration = span.end()
        if span == self.current_span:
            self.current_span = span.parent_id
        return duration


# Global tracer instance
tracer = Tracer("hybrid-scheduler")


def trace_function(span_name: Optional[str] = None, **span_attributes):
    """
    Decorator to trace function execution.
    
    Usage:
        @trace_function("process_task", task_type="ml")
        async def process_task():
            ...
    """
    def decorator(func: Callable):
        nonlocal span_name
        if span_name is None:
            span_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span = tracer.start_span(span_name)
            
            # Add attributes
            for key, value in span_attributes.items():
                span.set_attribute(key, value)
            
            try:
                result = await func(*args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error")
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                tracer.end_span(span)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            span = tracer.start_span(span_name)
            
            # Add attributes
            for key, value in span_attributes.items():
                span.set_attribute(key, value)
            
            try:
                result = func(*args, **kwargs)
                span.set_status("ok")
                return result
            except Exception as e:
                span.set_status("error")
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                tracer.end_span(span)
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class TraceContext:
    """
    Context manager for manual span creation.
    
    Usage:
        async with TraceContext("operation_name") as span:
            span.set_attribute("key", "value")
            await do_work()
    """
    
    def __init__(self, span_name: str):
        self.span_name = span_name
        self.span = None
    
    def __enter__(self):
        self.span = tracer.start_span(self.span_name)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_status("error")
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))
        else:
            self.span.set_status("ok")
        
        tracer.end_span(self.span)
        return False
    
    async def __aenter__(self):
        self.span = tracer.start_span(self.span_name)
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_status("error")
            self.span.set_attribute("error.type", exc_type.__name__)
            self.span.set_attribute("error.message", str(exc_val))
        else:
            self.span.set_status("ok")
        
        tracer.end_span(self.span)
        return False
