"""
Integration test for database persistence in simulation engine.
"""
import pytest
import anyio
from backend.services import SimulationDataService
from backend.core.database import init_db


@pytest.mark.anyio
async def test_training_data_save_and_retrieve():
    """Test saving and retrieving training data."""
    
    # Sample training data
    test_data = [
        {
            'size': 100.0,
            'compute_intensity': 0.7,
            'memory_required': 1000.0,
            'optimal_gpu_fraction': 0.8,
            'optimal_time': 1.5
        },
        {
            'size': 200.0,
            'compute_intensity': 0.3,
            'memory_required': 500.0,
            'optimal_gpu_fraction': 0.3,
            'optimal_time': 2.1
        }
    ]
    
    # Save data
    count = await SimulationDataService.save_training_data_batch(test_data)
    assert count == 2, "Should save 2 records"
    
    # Retrieve data  
    retrieved = await SimulationDataService.get_latest_training_data(limit=10)
    assert len(retrieved) >= 2, "Should retrieve at least 2 records"
    
    # Verify data integrity
    last_two = retrieved[:2]
    assert last_two[0]['size'] in [100.0, 200.0]
    assert last_two[0]['compute_intensity'] in[0.3, 0.7]


@pytest.mark.anyio
async def test_empty_batch_handling():
    """Test handling of empty batches."""
    count = await SimulationDataService.save_training_data_batch([])
    assert count == 0, "Should handle empty batch gracefully"


if __name__ == "__main__":
    # Run test
    asyncio.run(test_training_data_save_and_retrieve())
    asyncio.run(test_empty_batch_handling())
    print("✅ All integration tests passed!")
