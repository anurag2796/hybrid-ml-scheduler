"""
Sets up the database connection.
Uses SQLAlchemy with asyncpg so it's non-blocking.
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from loguru import logger

from backend.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def init_db():
    """Creates all the tables if they don't exist."""
    try:
        async with engine.begin() as conn:
            # Import models to register them
            from backend.models.domain import Task, SchedulerResult, Metric
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get a DB session.
    Usage: async def my_endpoint(db: AsyncSession = Depends(get_db)):
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """Context manager for when we need a DB session outside of a request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_db():
    """Closes the DB connection pool."""
    await engine.dispose()
    logger.info("Database connections closed")
