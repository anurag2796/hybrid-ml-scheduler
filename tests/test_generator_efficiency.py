
import unittest
import numpy as np
from src.workload_generator import WorkloadGenerator, Task

class TestWorkloadGenerator(unittest.TestCase):
    def test_stream_generation(self):
        """Test that the stream generator yields tasks"""
        wg = WorkloadGenerator(seed=42)
        stream = wg.generate_workload_stream(num_tasks=10)
        
        tasks = []
        for task in stream:
            self.assertIsInstance(task, Task)
            tasks.append(task)
            
        self.assertEqual(len(tasks), 10)
        
    def test_backward_compatibility(self):
        """Test that generate_workload still works and produces same results"""
        # Original implementation logic check (conceptually)
        # We can't easily check against the OLD code since we overwrote it, 
        # but we can check consistency.
        
        wg1 = WorkloadGenerator(seed=123)
        tasks1 = wg1.generate_workload(num_tasks=100)
        
        wg2 = WorkloadGenerator(seed=123)
        tasks2 = list(wg2.generate_workload_stream(num_tasks=100))
        
        self.assertEqual(len(tasks1), len(tasks2))
        for t1, t2 in zip(tasks1, tasks2):
            self.assertEqual(t1.task_id, t2.task_id)
            self.assertEqual(t1.size, t2.size)
            self.assertAlmostEqual(t1.arrival_time, t2.arrival_time)

    def test_large_scale_simulation(self):
        """Test that we can generate a large number of tasks without error"""
        wg = WorkloadGenerator(seed=42)
        # Generate 100,000 tasks but only consume a few to verify it starts
        stream = wg.generate_workload_stream(num_tasks=100000)
        
        count = 0
        for _ in stream:
            count += 1
            if count >= 100:
                break
        
        self.assertEqual(count, 100)

if __name__ == '__main__':
    unittest.main()
