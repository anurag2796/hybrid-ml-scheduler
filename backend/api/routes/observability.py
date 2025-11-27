"""
Observability endpoints for logging and tracing.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pathlib import Path
import json
from datetime import datetime, timedelta

from backend.services.logging_service import get_correlation_id

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/correlation-id")
async def get_current_correlation_id():
    """Get the current request's correlation ID."""
    correlation_id = get_correlation_id()
    return {
        "correlation_id": correlation_id or "None",
        "message": "Correlation ID for current request"
    }


@router.get("/logs/tail")
async def tail_logs(
    lines: int = Query(100, ge=1, le=1000, description="Number of lines to return"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, ERROR, etc.)")
):
    """
    Get recent log entries.
    
    Args:
        lines: Number of log lines to return (max 1000)
        level: Optional log level filter
    """
    log_file = Path("logs/app.log")
    
    if not log_file.exists():
        return {
            "logs": [],
            "message": "Log file not found. Logs may not be initialized yet."
        }
    
    try:
        # Read last N lines
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
        
        # Get last N lines
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        # Parse JSON logs and filter by level if specified
        parsed_logs = []
        for line in recent_lines:
            try:
                log_entry = json.loads(line.strip())
                if level is None or log_entry.get('level', '').upper() == level.upper():
                    parsed_logs.append(log_entry)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
        
        return {
            "logs": parsed_logs,
            "count": len(parsed_logs),
            "total_lines": len(all_lines)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


@router.get("/logs/errors")
async def get_error_logs(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    limit: int = Query(100, ge=1, le=1000, description="Max errors to return")
):
    """
    Get recent error logs.
    
    Args:
        hours: Number of hours to look back (max 7 days)
        limit: Maximum number of errors to return
    """
    log_file = Path("logs/error.log")
    
    if not log_file.exists():
        return {
            "errors": [],
            "message": "Error log file not found."
        }
    
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
        
        # Parse and filter by time
        errors = []
        for line in reversed(all_lines):  # Start from most recent
            if len(errors) >= limit:
                break
            
            try:
                log_entry = json.loads(line.strip())
                log_time = datetime.fromisoformat(log_entry.get('timestamp', ''))
                
                if log_time >= cutoff_time:
                    errors.append(log_entry)
            except (json.JSONDecodeError, ValueError):
                continue
        
        return {
            "errors": errors,
            "count": len(errors),
            "hours": hours
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read error logs: {str(e)}")


@router.get("/logs/search")
async def search_logs(
    query: str = Query(..., min_length=1, description="Search term"),
    limit: int = Query(100, ge=1, le=1000, description="Max results")
):
    """
    Search logs for a specific term.
    
    Args:
        query: Search query string
        limit: Maximum results to return
    """
    log_file = Path("logs/app.log")
    
    if not log_file.exists():
        return {
            "results": [],
            "message": "Log file not found."
        }
    
    try:
        results = []
        
        with open(log_file, 'r') as f:
            for line in f:
                if len(results) >= limit:
                    break
                
                if query.lower() in line.lower():
                    try:
                        log_entry = json.loads(line.strip())
                        results.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        return {
            "results": results,
            "count": len(results),
            "query": query
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search logs: {str(e)}")


@router.get("/health-check")
async def observability_health():
    """Check observability system health."""
    log_dir = Path("logs")
    
    return {
        "status": "healthy",
        "logs_directory": str(log_dir.absolute()),
        "logs_directory_exists": log_dir.exists(),
        "app_log_exists": (log_dir / "app.log").exists(),
        "error_log_exists": (log_dir / "error.log").exists(),
        "correlation_id_support": True,
        "tracing_enabled": True
    }
