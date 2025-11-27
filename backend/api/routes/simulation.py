"""
Simulation control and data endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from loguru import logger

router = APIRouter(prefix="/api", tags=["simulation"])

# This will be set by the main application
simulation_engine = None


def set_simulation_engine(engine):
    """Set the simulation engine instance."""
    global simulation_engine
    simulation_engine = engine


@router.post("/start")
async def start_simulation():
    """Start the simulation."""
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    if simulation_engine.is_running:
        return {"status": "already_running", "message": "Simulation is already running"}
    
    # Start simulation in background
    import asyncio
    asyncio.create_task(simulation_engine.start())
    
    return {"status": "started", "message": "Simulation started successfully"}


@router.post("/stop")
async def stop_simulation():
    """Stop the simulation."""
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    if not simulation_engine.is_running:
        return {"status": "not_running", "message": "Simulation is not running"}
    
    simulation_engine.stop()
    return {"status": "stopped", "message": "Simulation stopped successfully"}


@router.post("/pause")
async def pause_simulation():
    """Pause the simulation."""
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    if not simulation_engine.is_running:
        raise HTTPException(status_code=400, detail="Simulation is not running")
    
    simulation_engine.pause()
    return {"status": "paused", "message": "Simulation paused"}


@router.post("/resume")
async def resume_simulation():
    """Resume a paused simulation."""
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    if not simulation_engine.is_paused:
        raise HTTPException(status_code=400, detail="Simulation is not paused")
    
    simulation_engine.resume()
    return {"status": "resumed", "message": "Simulation resumed"}


@router.get("/status")
async def get_simulation_status():
    """Get current simulation status."""
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    return {
        "is_running": simulation_engine.is_running,
        "is_paused": simulation_engine.is_paused,
        "tasks_processed": simulation_engine.tasks_processed,
        "metrics": simulation_engine.metrics
    }


@router.get("/full_history")
async def get_full_history():
    """
    Get full simulation history.
    
    Returns metrics and performance data for all schedulers.
    """
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    # Calculate comparison data
    comparison = []
    for name, metrics in simulation_engine.metrics.items():
        total_tasks = metrics.get('tasks', 0)
        total_time = metrics.get('total_time', 0.0)
        avg_time = total_time / max(1, total_tasks)
        
        comparison.append({
            'name': name,
            'avg_time': avg_time,
            'total_tasks': total_tasks,
            'total_time': total_time
        })
    
    return {
        "comparison": comparison,
        "tasks_processed": simulation_engine.tasks_processed
    }
