
import unittest
from src.rl_scheduler import QLearningScheduler
from src.workload_generator import Task

class TestRLScheduler(unittest.TestCase):
    def test_q_learning_update(self):
        """Test that Q-table updates after scheduling"""
        scheduler = QLearningScheduler(num_gpus=2, learning_rate=0.5)
        task = Task(0, 1000, 0.5, 100, 0, 1.0)
        
        # Initial state
        state = scheduler._get_state(task)
        initial_q = scheduler._get_q_value(state, 0) # Action 0 (CPU)
        self.assertEqual(initial_q, 0.0)
        
        # Force action 0 (CPU) by mocking choice? 
        # Easier: Just call schedule_task and check if ANY Q-value changed
        decision = scheduler.schedule_task(task)
        action = decision['action']
        
        # Check Q-value updated
        new_q = scheduler._get_q_value(state, action)
        self.assertNotEqual(new_q, 0.0, "Q-value should be updated")
        self.assertLess(new_q, 0.0, "Reward is negative cost, so Q should be negative")

    def test_epsilon_decay(self):
        """Test that epsilon decays over time"""
        scheduler = QLearningScheduler(epsilon=1.0, epsilon_decay=0.9)
        task = Task(0, 1000, 0.5, 100, 0, 1.0)
        
        scheduler.schedule_task(task)
        self.assertEqual(scheduler.epsilon, 0.9)
        
        scheduler.schedule_task(task)
        self.assertAlmostEqual(scheduler.epsilon, 0.81)

if __name__ == '__main__':
    unittest.main()
