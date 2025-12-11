
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.config import settings
from sqlalchemy import create_engine, text

def main():
    print(f"Connecting to: {settings.database_url}")
    engine = create_engine(settings.database_url)
    
    try:
        with engine.connect() as conn:
            # Check count first
            result = conn.execute(text("SELECT COUNT(*) FROM scheduler_results WHERE execution_cost < 0"))
            count = result.scalar()
            print(f"Found {count} rows with negative cost.")
            
            if count > 0:
                print("Deleting...")
                result = conn.execute(text("DELETE FROM scheduler_results WHERE execution_cost < 0"))
                conn.commit()
                print(f"Deleted {result.rowcount} bad rows.")
            else:
                print("No bad rows found.")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
