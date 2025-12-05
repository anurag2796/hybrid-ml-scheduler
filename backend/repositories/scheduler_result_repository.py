"""
Repository for scheduler results.
"""
from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession  
from sqlalchemy import select, and_, desc, func
from datetime import datetime

from backend.models.domain import SchedulerResult, Task
from backend.models.schemas import SchedulerResultCreate, SchedulerResultResponse


class SchedulerResultRepository:
    """Repository for scheduler result operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, result: SchedulerResultCreate) -> SchedulerResult:
        """Create a new scheduler result."""
        db_result = SchedulerResult(**result.model_dump())
        self.session.add(db_result)
        await self.session.flush()
        await self.session.refresh(db_result)
        return db_result
    
    async def create_many(self, results: List[SchedulerResultCreate]) -> List[SchedulerResult]:
        """Bulk create scheduler results."""
        db_objects = [SchedulerResult(**r.model_dump()) for r in results]
        self.session.add_all(db_objects)
        await self.session.flush()
        return db_objects
    
    async def get_by_task_id(self, task_id: int) -> List[SchedulerResult]:
        """Get all results for a specific task."""
        stmt = (
            select(SchedulerResult)
            .where(SchedulerResult.task_id == task_id)
            .order_by(SchedulerResult.executed_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_by_scheduler(
        self, 
        scheduler_name: str, 
        limit: int = 100
    ) -> List[SchedulerResult]:
        """Get recent results for a scheduler."""
        stmt = (
            select(SchedulerResult)
            .where(SchedulerResult.scheduler_name == scheduler_name)
            .order_by(desc(SchedulerResult.executed_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_latest(self, limit: int = 100) -> List[SchedulerResult]:
        """Get latest results across all schedulers."""
        stmt = (
            select(SchedulerResult)
            .order_by(desc(SchedulerResult.executed_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_scheduler_stats(self, scheduler_name: str) -> Dict:
        """Get aggregate stats for a scheduler."""
        stmt = select(
            func.count(SchedulerResult.id).label('count'),
            func.avg(SchedulerResult.actual_time).label('avg_time'),
            func.min(SchedulerResult.actual_time).label('min_time'),
            func.max(SchedulerResult.actual_time).label('max_time'),
            func.sum(SchedulerResult.energy_consumption).label('total_energy'),
            func.sum(SchedulerResult.execution_cost).label('total_cost')
        ).where(SchedulerResult.scheduler_name == scheduler_name)
        
        result = await self.session.execute(stmt)
        row = result.one()
        
        return {
            'count': row.count or 0,
            'avg_time': float(row.avg_time or 0),
            'min_time': float(row.min_time or 0),
            'max_time': float(row.max_time or 0),
            'total_energy': float(row.total_energy or 0),
            'total_cost': float(row.total_cost or 0)
        }

        
    async def get_comparative_history(self, limit: int = 100) -> List[Dict]:
        """
        Get tasks with all their scheduler results for comparison.
        Returns a list of dicts, one per task, with nested results.
        """
        # We want to fetch the last N tasks and all their results
        # This is a bit complex in pure ORM async, so we'll do it in two steps or a join
        
        # 1. Get last N tasks
        stmt_tasks = (
            select(Task)
            .order_by(desc(Task.task_id))
            .limit(limit)
        )
        tasks_res = await self.session.execute(stmt_tasks)
        tasks = tasks_res.scalars().all()
        
        if not tasks:
            return []
            
        task_ids = [t.task_id for t in tasks]
        
        # 2. Get results for these tasks
        stmt_results = (
            select(SchedulerResult)
            .where(SchedulerResult.task_id.in_(task_ids))
        )
        results_res = await self.session.execute(stmt_results)
        results = results_res.scalars().all()
        
        # 3. Assemble
        # Map task_id -> list of results
        results_map = {}
        for r in results:
            if r.task_id not in results_map:
                results_map[r.task_id] = {}
            results_map[r.task_id][r.scheduler_name] = {
                'time': r.actual_time,
                'gpu_fraction': r.gpu_fraction,
                'cost': r.execution_cost
            }
            
        # Create final list
        comparative_data = []
        for t in tasks:
            comparative_data.append({
                'task_id': t.task_id,
                'size': t.size,
                'intensity': t.compute_intensity,
                'results': results_map.get(t.task_id, {})
            })
            
        return comparative_data
