"""
Database initialization and setup script.

Creates the database, tables, and initial data.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from loguru import logger

from backend.core.config import settings
from backend.core.database import init_db, engine, Base


async def create_database():
    """Create the database if it doesn't exist."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    try:
        # Connect to postgres database to create our database
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (settings.postgres_db,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f"CREATE DATABASE {settings.postgres_db}")
            logger.info(f"Database '{settings.postgres_db}' created successfully")
        else:
            logger.info(f"Database '{settings.postgres_db}' already exists")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


async def create_tables():
    """Create all database tables."""
    try:
        async with engine.begin() as conn:
            # Import models to register them
            from backend.models.domain import Task, SchedulerResult, Metric, TrainingData, SimulationState
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("All tables created successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


async def verify_setup():
    """Verify database setup."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
            tables = [row[0] for row in result]
            
            logger.info(f"Found {len(tables)} tables:")
            for table in tables:
                logger.info(f"  - {table}")
            
            return len(tables) > 0
    except Exception as e:
        logger.error(f"Error verifying setup: {e}")
        return False


async def main():
    """Main setup function."""
    logger.info("=" * 80)
    logger.info("DATABASE INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Database: {settings.postgres_db}")
    logger.info(f"Host: {settings.postgres_host}:{settings.postgres_port}")
    logger.info(f"User: {settings.postgres_user}")
    logger.info("=" * 80)
    
    # Step 1: Create database
    logger.info("\n[1/3] Creating database...")
    await create_database()
    
    # Step 2: Create tables
    logger.info("\n[2/3] Creating tables...")
    await create_tables()
    
    # Step 3: Verify setup
    logger.info("\n[3/3] Verifying setup...")
    success = await verify_setup()
    
    if success:
        logger.info("\n✅ Database initialization complete!")
    else:
        logger.error("\n❌ Database initialization failed!")
        sys.exit(1)
    
    # Close connections
    from backend.core.database import close_db
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
