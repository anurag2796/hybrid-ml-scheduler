"""
Repository pattern for database access.

Provides clean abstraction over database operations.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from datetime import datetime, timedelta
from loguru import logger

from backend.models.domain import TrainingData
from backend.models.schemas import TrainingDataCreate, TrainingDataResponse


class TrainingDataRepository:
    """Repository for training data operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, data: TrainingDataCreate) -> TrainingData:
        """Create a new training data record."""
        db_data = TrainingData(**data.model_dump())
        self.session.add(db_data)
        await self.session.flush()
        await self.session.refresh(db_data)
        return db_data
    
    async def create_many(self, data_list: List[TrainingDataCreate]) -> List[TrainingData]:
        """Bulk create training data records (optimized)."""
        db_objects = [TrainingData(**data.model_dump()) for data in data_list]
        self.session.add_all(db_objects)
        await self.session.flush()
        return db_objects
    
    async def get_latest(self, limit: int = 1000) -> List[TrainingData]:
        """Get latest training data records."""
        stmt = (
            select(TrainingData)
            .order_by(desc(TrainingData.created_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[TrainingData]:
        """Get training data within date range."""
        stmt = (
            select(TrainingData)
            .where(
                and_(
                    TrainingData.created_at >= start_date,
                    TrainingData.created_at <= end_date
                )
            )
            .order_by(desc(TrainingData.created_at))
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def count_all(self) -> int:
        """Count total training data records."""
        stmt = select(func.count()).select_from(TrainingData)
        result = await self.session.execute(stmt)
        return result.scalar() or 0
    
    async def delete_old_records(self, keep_last_n: int = 10000):
        """Delete old records, keeping only the latest N."""
        count = await self.count_all()
        if count <= keep_last_n:
            return 0
        
        # Get the cutoff timestamp
        stmt = (
            select(TrainingData.created_at)
            .order_by(desc(TrainingData.created_at))
            .offset(keep_last_n)
            .limit(1)
        )
        result = await self.session.execute(stmt)
        cutoff_time = result.scalar()
        
        if cutoff_time:
            stmt = TrainingData.__table__.delete().where(
                TrainingData.created_at < cutoff_time
            )
            result = await self.session.execute(stmt)
            await self.session.commit()
            return result.rowcount
        return 0
