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
    Get full simulation history (training data).
    
    Returns:
        List[Dict]: List of historical task execution data.
    """
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    try:
        from backend.services.simulation_data_service import SimulationDataService
        # Get latest 1000 records
        data = await SimulationDataService.get_latest_training_data(limit=1000)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch full history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/comparative")
async def get_comparative_history(limit: int = 100):
    """
    Get comparative history of tasks and scheduler results.
    
    Args:
        limit: Maximum number of tasks to return
        
    Returns:
        List[Dict]: List of tasks with their scheduler results.
    """
    if simulation_engine is None:
        raise HTTPException(status_code=500, detail="Simulation engine not initialized")
    
    try:
        from backend.services.simulation_data_service import SimulationDataService
        return await SimulationDataService.get_comparative_history(limit)
    except Exception as e:
        logger.error(f"Failed to fetch comparative history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
