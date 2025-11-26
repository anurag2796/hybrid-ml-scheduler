
import unittest
from unittest.mock import MagicMock
from src.online_scheduler import OnlineScheduler, ResourceState
from src.workload_generator import Task

class TestEnergyOptimization(unittest.TestCase):
    def test_energy_cost_calculation(self):
        """Test that scheduler selects GPU based on energy-aware cost"""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8] # GPU-heavy task
        
        # Create scheduler with high energy weight (prefer efficiency)
        scheduler = OnlineScheduler(model=mock_model, num_gpus=2, energy_weight=0.9)
        
        # Manually set GPU states
        # GPU 0: Low load (0.1) -> Fast execution -> Lower Energy
        # GPU 1: High load (0.9) -> Slow execution -> Higher Energy
        scheduler.gpu_states[0].current_load = 0.1
        scheduler.gpu_states[1].current_load = 0.9
        
        task = Task(0, 1000, 0.8, 100, 0, 1.0)
        
        # With identical hardware, lower load = faster = less energy
        # So it should pick GPU 0
        best_gpu = scheduler._find_best_gpu(task, gpu_fraction=0.8)
        self.assertEqual(best_gpu, 0)
        
    def test_weight_impact(self):
        """Test that energy weight parameter is stored correctly"""
        mock_model = MagicMock()
        s1 = OnlineScheduler(mock_model, energy_weight=0.1)
        s2 = OnlineScheduler(mock_model, energy_weight=0.9)
        
        self.assertEqual(s1.energy_weight, 0.1)
        self.assertEqual(s2.energy_weight, 0.9)

if __name__ == '__main__':
    unittest.main()
