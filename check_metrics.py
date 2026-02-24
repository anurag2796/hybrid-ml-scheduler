
import asyncio
from sqlalchemy import create_engine, text
from backend.core.config import settings

def check_metrics():
    # Use sync driver for this script
    db_url = settings.database_url.replace("+asyncpg", "") # ensure sync if needed, but config.database_url is sync
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:

            schedulers = ['hybrid_ml', 'rl_agent', 'greedy', 'round_robin', 'random']
            
            print(f"{'Scheduler':<15} | {'Avg Time':<10} | {'Avg Energy':<10} | {'Avg Cost':<10} | {'Count':<10}")
            print("-" * 65)
            
            for scheduler in schedulers:
                query = text(f"""
                SELECT AVG(actual_time), AVG(energy_consumption), AVG(execution_cost), COUNT(*)
                FROM (
                    SELECT actual_time, energy_consumption, execution_cost
                    FROM scheduler_results
                    WHERE scheduler_name = '{scheduler}'
                    ORDER BY id DESC
                    LIMIT 20000
                ) as recent
                """)
                result = conn.execute(query).fetchone()
                print(f"{scheduler:<15} | {result[0]:<10.4f} | {result[1]:<10.4f} | {result[2]:<10.4f} | {result[3]:<10}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_metrics()
