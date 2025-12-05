"""
Handles saving simulation data to the database.
Wraps the repository stuff so the main code doesn't have to deal with it.
"""
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger

from backend.core.database import get_db_context
from backend.repositories import TrainingDataRepository, SchedulerResultRepository, TaskRepository
from backend.models.schemas import TrainingDataCreate, SchedulerResultCreate, TaskCreate
from backend.models.domain import Task
from src.workload_generator import Task as TaskObj
from sqlalchemy import select


class SimulationDataService:
    # Service for managing simulation data persistence.
    
    @staticmethod
    async def save_training_data_batch(data_list: List[Dict]) -> int:
        """
        Saves a batch of training data to the DB.
        Returns the number of records saved.
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
        Grabs the latest training data so we can retrain the model.
        Uses a 30s cache so we don't hammer the DB.
        """
        from backend.services.cache_service import cache_service
        
        # Try cache first
        cache_key = f"training_data_latest_{limit}"
        cached_data = await cache_service.get("training_data", cache_key)
        if cached_data is not None:
            logger.debug(f"Returning cached training data ({len(cached_data)} records)")
            return cached_data
        
        # Cache miss - fetch from database
        try:
            async with get_db_context() as db:
                repo = TrainingDataRepository(db)
                records = await repo.get_latest(limit=limit)
                
                # Convert to list of dicts
                data = [
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
                
                # Cache for 30 seconds
                await cache_service.set("training_data", cache_key, data, ttl=30)
                logger.debug(f"Cached {len(data)} training records")
                
                return data
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    @staticmethod
    async def save_scheduler_results(task: TaskObj, results: Dict[str, Dict]) -> int:
        """
        Saves the results from the scheduler for a specific task.
        Returns how many results got saved.
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
    async def save_scheduler_results_batch(batch_data: List[tuple]) -> int:
        """
        Saves a batch of (task, results) tuples to the DB.
        """
        if not batch_data:
            return 0
            
        try:
            async with get_db_context() as db:
                result_repo = SchedulerResultRepository(db)
                task_repo = TaskRepository(db)
                
                tasks_to_create = []
                results_to_create = []
                
                # Process batch
                for task, results in batch_data:
                    # Prepare Task
                    # Check if we already added this task to our local list to avoid dupes in batch
                    if not any(t.task_id == task.task_id for t in tasks_to_create):
                        tasks_to_create.append(TaskCreate(
                            task_id=task.task_id,
                            size=task.size,
                            compute_intensity=task.compute_intensity,
                            memory_required=task.memory_required,
                            duration_estimate=task.duration_estimate,
                            arrival_time=task.arrival_time,
                            dependencies=task.dependencies if hasattr(task, 'dependencies') else []
                        ))
                    
                    # Prepare Results
                    for scheduler_name, result in results.items():
                        results_to_create.append(SchedulerResultCreate(
                            task_id=task.task_id,
                            scheduler_name=scheduler_name,
                            gpu_fraction=result['gpu_fraction'],
                            cpu_fraction=1.0 - result['gpu_fraction'],
                            gpu_id=result.get('gpu_id'),
                            actual_time=result['actual_time'],
                            energy_consumption=result.get('energy'),
                            execution_cost=result.get('cost'),
                            extra_metadata={}
                        ))
                
                # Bulk insert Tasks
                # First check which tasks already exist to avoid UniqueViolation
                if tasks_to_create:
                    existing_ids_res = await db.execute(
                        select(Task.task_id).where(Task.task_id.in_([t.task_id for t in tasks_to_create]))
                    )
                    existing_ids = set(existing_ids_res.scalars().all())
                    
                    # Filter out existing tasks
                    new_tasks = [t for t in tasks_to_create if t.task_id not in existing_ids]
                    
                    if new_tasks:
                        await task_repo.create_many(new_tasks)
                
                # Bulk insert Results
                if results_to_create:
                    await result_repo.create_many(results_to_create)
                
                await db.commit()
                
                logger.debug(f"Saved batch: {len(tasks_to_create)} tasks, {len(results_to_create)} results")
                return len(results_to_create)
                
        except Exception as e:
            logger.error(f"Failed to save scheduler results batch: {e}")
            return 0
        """
        Gets the stats for a scheduler (like avg time, energy, etc).
        Caches it for 10s.
        """
        from backend.services.cache_service import cache_service
        
        # Try cache first
        cache_key = f"scheduler_stats_{scheduler_name}"
        cached_stats = await cache_service.get("scheduler_stats", cache_key)
        if cached_stats is not None:
            logger.debug(f"Returning cached stats for {scheduler_name}")
            return cached_stats
        
        # Cache miss - fetch from database
        try:
            async with get_db_context() as db:
                repo = SchedulerResultRepository(db)
                stats = await repo.get_scheduler_stats(scheduler_name)
                
                # Cache for 10 seconds
                await cache_service.set("scheduler_stats", cache_key, stats, ttl=10)
                logger.debug(f"Cached stats for {scheduler_name}")
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get scheduler stats: {e}")
            return {}
    
    @staticmethod
    async def cleanup_old_data(keep_last_n: int = 10000):
        """
        Deletes old training data so the DB doesn't get too huge.
        """
        try:
            async with get_db_context() as db:
                repo = TrainingDataRepository(db)
                deleted = await repo.delete_old_records(keep_last_n)
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old training records")
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    @staticmethod
    async def get_comparative_history(limit: int = 100) -> List[Dict]:
        """
        Gets the comparative history of tasks and scheduler results.
        """
        try:
            async with get_db_context() as db:
                repo = SchedulerResultRepository(db)
                return await repo.get_comparative_history(limit)
        except Exception as e:
            logger.error(f"Failed to get comparative history: {e}")
            return []
