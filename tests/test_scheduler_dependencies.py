
import unittest
from unittest.mock import MagicMock
from src.online_scheduler import OnlineScheduler
from src.workload_generator import Task

class TestSchedulerDependencies(unittest.TestCase):
    def test_topological_scheduling(self):
        """Test that tasks are scheduled in topological order"""
        # Create a mock model that always predicts 0.5
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        
        scheduler = OnlineScheduler(model=mock_model, num_gpus=1)
        
        # Create a diamond dependency graph:
        #   0
        #  / \
        # 1   2
        #  \ /
        #   3
        
        t0 = Task(0, 100, 0.5, 10, 0, 1.0, dependencies=[])
        t1 = Task(1, 100, 0.5, 10, 0, 1.0, dependencies=[0])
        t2 = Task(2, 100, 0.5, 10, 0, 1.0, dependencies=[0])
        t3 = Task(3, 100, 0.5, 10, 0, 1.0, dependencies=[1, 2])
        
        # Submit in reverse order to test queue handling
        scheduler.submit_task(t3)
        scheduler.submit_task(t2)
        scheduler.submit_task(t1)
        scheduler.submit_task(t0)
        
        decisions = scheduler.process_queue()
        
        scheduled_ids = [d['task_id'] for d in decisions]
        
        # Verify order
        self.assertEqual(len(scheduled_ids), 4)
        self.assertEqual(scheduled_ids[0], 0, "Task 0 must be first")
        self.assertIn(1, scheduled_ids[1:3], "Task 1 must be after 0 and before 3")
        self.assertIn(2, scheduled_ids[1:3], "Task 2 must be after 0 and before 3")
        self.assertEqual(scheduled_ids[3], 3, "Task 3 must be last")

if __name__ == '__main__':
    unittest.main()
