
import sys
import os
import asyncio
from pathlib import Path
from sqlalchemy import text

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.database import get_db_context

async def main():
    print("⏳ Truncating database tables...")
    try:
        async with get_db_context() as db:
            # Postgres specific truncate
            await db.execute(text('TRUNCATE TABLE scheduler_results, training_data, tasks RESTART IDENTITY CASCADE'))
            await db.commit()
            print("✅ Database truncated successfully.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
