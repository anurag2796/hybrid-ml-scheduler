"""
Repository for tasks.
"""
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from backend.models.domain import Task as TaskModel
from backend.models.schemas import TaskCreate, TaskResponse


class TaskRepository:
    """Repository for task operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, task: TaskCreate) -> TaskModel:
        """Create a new task."""
        db_task = TaskModel(**task.model_dump())
        self.session.add(db_task)
        await self.session.flush()
        await self.session.refresh(db_task)
        return db_task
    
    async def create_many(self, tasks: List[TaskCreate]) -> List[TaskModel]:
        """Bulk create tasks."""
        db_objects = [TaskModel(**t.model_dump()) for t in tasks]
        self.session.add_all(db_objects)
        await self.session.flush()
        return db_objects
    
    async def get_by_task_id(self, task_id: int) -> Optional[TaskModel]:
        """Get task by task_id."""
        stmt = select(TaskModel).where(TaskModel.task_id == task_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_latest(self, limit: int = 100) -> List[TaskModel]:
        """Get latest tasks."""
        stmt = (
            select(TaskModel)
            .order_by(desc(TaskModel.created_at))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
