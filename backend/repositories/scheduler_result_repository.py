"""
Repository for scheduler results.
"""
from typing import List, Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession  
from sqlalchemy import select, and_, desc, func
from datetime import datetime

from backend.models.domain import SchedulerResult
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
