
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.config import settings
from sqlalchemy import create_engine
import pandas as pd

def main():
    print(f"Connecting to: {settings.database_url}")
    engine = create_engine(settings.database_url)
    
    with engine.connect() as conn:
        print("\n--- Scheduler Results Summary ---")
        try:
            query = """
            SELECT 
                scheduler_name, 
                COUNT(*) as count,
                AVG(actual_time) as avg_time,
                SUM(execution_cost) as total_cost,
                AVG(execution_cost) as avg_cost
            FROM scheduler_results 
            GROUP BY scheduler_name
            """
            df = pd.read_sql(query, conn)
            print(df)
            
            # Check specifically for RL Agent 0s or NULLs
            rl_data = df[df['scheduler_name'] == 'rl_agent']
            if not rl_data.empty:
                cost = rl_data.iloc[0]['total_cost']
                print(f"\nRL Agent Total Cost: {cost}")
                if cost == 0 or pd.isna(cost):
                    print("⚠️ RL Agent cost is invalid!")
            else:
                print("⚠️ RL Agent not found in results!")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
