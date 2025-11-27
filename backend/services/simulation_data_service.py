"""
Service layer for simulation data persistence.

Provides high-level business logic wrapping the repository layer.
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from backend.core.database import get_db_context
from backend.repositories import TrainingDataRepository, SchedulerResultRepository, TaskRepository
from backend.models.schemas import TrainingDataCreate, SchedulerResultCreate, TaskCreate
from src.workload_generator import Task


class SimulationDataService:
    """Service for managing simulation data persistence."""
    
    @staticmethod
    async def save_training_data_batch(data_list: List[Dict]) -> int:
        """
        Save a batch of training data to the database.
        
        Args:
            data_list: List of dictionaries containing training data
            
        Returns:
            Number of records saved
        """
        if not data_list:
            return 0
            
        try:
            async with get_db_context() as db:
                repo = TrainingDataRepository(db)
                
                # Convert dicts to Pydantic models
                training_data = [
                    TrainingDataCreate(
                        size=d['size'],
                        compute_intensity=d['compute_intensity'],
                        memory_required=d['memory_required'],
                        memory_per_size=d['memory_required'] / (d['size'] + 1),
                        compute_to_memory=d['compute_intensity'] / (d['memory_required'] + 1),
                        optimal_gpu_fraction=d['optimal_gpu_fraction'],
                        optimal_time=d['optimal_time']
                    )
                    for d in data_list
                ]
                
                # Bulk insert
                await repo.create_many(training_data)
                await db.commit()
                
                logger.debug(f"Saved {len(training_data)} training records to database")
                return len(training_data)
                
        except Exception as e:
            logger.error(f"Failed to save training data batch: {e}")
            return 0
    
    @staticmethod
    async def get_latest_training_data(limit: int = 1000) -> List[Dict]:
        """
        Get the latest training data for model retraining.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of training data dictionaries
        """
        try:
            async with get_db_context() as db:
                repo = TrainingDataRepository(db)
                records = await repo.get_latest(limit=limit)
                
                # Convert to list of dicts for compatibility with existing code
                return [
                    {
                        'size': r.size,
                        'compute_intensity': r.compute_intensity,
                        'memory_required': r.memory_required,
                        'memory_per_size': r.memory_per_size,
                        'compute_to_memory': r.compute_to_memory,
                        'optimal_gpu_fraction': r.optimal_gpu_fraction,
                        'optimal_time': r.optimal_time
                    }
                    for r in records
                ]
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    @staticmethod
    async def save_scheduler_results(task: Task, results: Dict[str, Dict]) -> int:
        """
        Save scheduler results for a task.
        
        Args:
            task: The task that was scheduled
            results: Dictionary of scheduler results
            
        Returns:
            Number of results saved
        """
        try:
            async with get_db_context() as db:
                result_repo = SchedulerResultRepository(db)
                task_repo = TaskRepository(db)
                
                # First, save the task if it doesn't exist
                task_data = TaskCreate(
                    task_id=task.task_id,
                    size=task.size,
                    compute_intensity=task.compute_intensity,
                    memory_required=task.memory_required,
                    duration_estimate=task.duration_estimate,
                    arrival_time=task.arrival_time,
                    dependencies=task.dependencies if hasattr(task, 'dependencies') else []
                )
                
                # Check if task exists, if not create it
                existing_task = await task_repo.get_by_task_id(task.task_id)
                if not existing_task:
                    await task_repo.create(task_data)
                
                # Convert results to Pydantic models
                scheduler_results = []
                for scheduler_name, result in results.items():
                    scheduler_results.append(
                        SchedulerResultCreate(
                            task_id=task.task_id,
                            scheduler_name=scheduler_name,
                            gpu_fraction=result['gpu_fraction'],
                            cpu_fraction=1.0 - result['gpu_fraction'],
                            gpu_id=result.get('gpu_id'),
                            actual_time=result['actual_time'],
                            energy_consumption=result.get('energy'),
                            execution_cost=result.get('cost'),
                            extra_metadata={}
                        )
                    )
                
                # Bulk insert results
                await result_repo.create_many(scheduler_results)
                await db.commit()
                
                logger.debug(f"Saved {len(scheduler_results)} scheduler results for task {task.task_id}")
                return len(scheduler_results)
                
        except Exception as e:
            logger.error(f"Failed to save scheduler results: {e}")
            return 0
    
    @staticmethod
    async def get_scheduler_stats(scheduler_name: str) -> Dict:
        """
        Get aggregate statistics for a scheduler.
        
        Args:
            scheduler_name: Name of the scheduler
            
        Returns:
            Dictionary with aggregate stats
        """
        try:
            async with get_db_context() as db:
                repo = SchedulerResultRepository(db)
                return await repo.get_scheduler_stats(scheduler_name)
        except Exception as e:
            logger.error(f"Failed to get scheduler stats: {e}")
            return {}
    
    @staticmethod
    async def cleanup_old_data(keep_last_n: int = 10000):
        """
        Clean up old training data to prevent database bloat.
        
        Args:
            keep_last_n: Number of most recent records to keep
        """
        try:
            async with get_db_context() as db:
                repo = TrainingDataRepository(db)
                deleted = await repo.delete_old_records(keep_last_n)
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old training records")
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
