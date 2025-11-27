"""Quick check of database contents."""
import asyncio
from backend.services import SimulationDataService

async def check_database():
    print("Checking database contents...")
    
    # Check training data
    data = await SimulationDataService.get_latest_training_data(limit=10)
    print(f"\n✅ Training data records: {len(data)}")
    if data:
        print(f"   Latest record: size={data[0]['size']:.1f}, compute={data[0]['compute_intensity']:.2f}")
    
    print("\n📊 Database is ready and accessible!")

if __name__ == "__main__":
    asyncio.run(check_database())
