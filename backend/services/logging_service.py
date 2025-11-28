"""
Structured logging configuration with correlation IDs and JSON formatting.

Provides production-ready logging with:
- Correlation IDs for request tracking
- JSON structured format
- Log levels by environment
- Performance logging
"""
import sys
import uuid
import json
from contextvars import ContextVar
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger
from pathlib import Path

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIdFilter:
    """Add correlation ID to log records."""
    
    def __call__(self, record):
        correlation_id = correlation_id_var.get()
        record["extra"]["correlation_id"] = correlation_id or "N/A"
        return True


def patch_json(record):
    """Patch record with JSON serialization."""
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "correlation_id": record["extra"].get("correlation_id", "N/A"),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra fields
    for key, value in record["extra"].items():
        if key not in ["correlation_id", "json"]:
            subset[key] = value
    
    # Add exception if present
    if record["exception"]:
        subset["exception"] = {
            "type": record["exception"].type.__name__,
            "value": str(record["exception"].value),
            "traceback": record["exception"].traceback
        }
    
    record["extra"]["json"] = json.dumps(subset)


def setup_logging(environment: str = "development", log_dir: str = "logs"):
    """
    Configure structured logging for the application.
    
    Args:
        environment: Environment name (development, staging, production)
        log_dir: Directory for log files
    """
    # Remove default logger
    logger.remove()
    
    # Configure patcher
    logger.configure(patcher=patch_json)
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Console logging (human-readable in dev, JSON in prod)
    if environment == "development":
        # Human-readable format for development
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level> | <yellow>corr_id={extra[correlation_id]}</yellow>",
            level="DEBUG",
            colorize=True,
            filter=CorrelationIdFilter()
        )
    else:
        # JSON format for production
        logger.add(
            sys.stderr,
            format="{extra[json]}",
            level="INFO",
            serialize=False,
            filter=CorrelationIdFilter()
        )
    
    # File logging - JSON format always
    logger.add(
        f"{log_dir}/app.log",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        format="{extra[json]}",
        level="INFO",
        serialize=False,
        filter=CorrelationIdFilter()
    )
    
    # Error log file
    logger.add(
        f"{log_dir}/error.log",
        rotation="50 MB",
        retention="60 days",
        compression="gz",
        format="{extra[json]}",
        level="ERROR",
        serialize=False,
        filter=CorrelationIdFilter()
    )
    
    logger.info(f"Logging configured for environment: {environment}")


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation ID for current context.
    
    Args:
        correlation_id: Optional correlation ID, generates UUID if not provided
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id():
    """Clear correlation ID from current context."""
    correlation_id_var.set(None)


def log_performance(operation: str, duration_ms: float, **extra):
    """
    Log performance metrics in structured format.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **extra: Additional context fields
    """
    logger.info(
        f"Performance: {operation} completed in {duration_ms:.2f}ms",
        operation=operation,
        duration_ms=duration_ms,
        **extra
    )


def log_error(error: Exception, context: Dict[str, Any] = None):
    """
    Log error with structured context.
    
    Args:
        error: The exception that occurred
        context: Additional context dictionary
    """
    logger.error(
        f"Error occurred: {type(error).__name__}: {str(error)}",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {}
    )


# Convenience logger instance
structured_logger = logger
