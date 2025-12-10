import unittest
import numpy as np
import os
import sys
import shutil
import tempfile

# Add project root to path
sys.path.append(os.getcwd())

from src.workload_generator import WorkloadGenerator
from src.dqn_scheduler import DQNScheduler
from src.simulator import VirtualMultiGPU

class TestRLConvergence(unittest.TestCase):
    
    def setUp(self):
        self.wg = WorkloadGenerator(seed=42)
        # Fast learning for tests
        self.scheduler = DQNScheduler(
            num_gpus=4,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99, # Fast decay
            batch_size=32
        )
        self.simulator = VirtualMultiGPU(num_gpus=4)
        
    def test_convergence(self):
        """Test that the agent converges to near-zero regret within 600 tasks"""
        tasks = self.wg.generate_workload(num_tasks=600)
        regrets = []
        
        for task in tasks:
            # Oracle
            best_time = float('inf')
            for frac in np.linspace(0, 1, 11):
                res = self.simulator.simulate_task_execution(task, frac)
                if res['actual_time'] < best_time:
                    best_time = res['actual_time']
            
            # RL
            action_dict = self.scheduler.get_action(task)
            rl_res = self.simulator.simulate_task_execution(task, action_dict['gpu_fraction'])
            
            # Reward Calculation (Simple version)
            # Cost = Time (ignoring energy for simple test check, or assuming energy correlated)
            # To be precise, let's match the scheduler's internal logic 
            # But here we just check if it matches Oracle Time
            
            # Provide feedback
            metrics = {'time': rl_res['actual_time'], 'energy': 0.0} # Dummy energy, focus on time
            self.scheduler.observe(task, action_dict['action'], metrics)
            
            regret = rl_res['actual_time'] - best_time
            regrets.append(regret)
            
        # Check last 50 tasks
        final_regret = np.mean(regrets[-50:])
        print(f"\nFinal Average Regret (Last 50): {final_regret:.4f}")
        
        # It should be very close to 0
        self.assertLess(final_regret, 0.1, "RL Agent failed to converge (Regret > 0.1)")
        
    def test_replay_buffer_growth(self):
        """Test that replay buffer grows as we observe"""
        initial_size = len(self.scheduler.memory)
        tasks = self.wg.generate_workload(num_tasks=10)
        
        for task in tasks:
            action_dict = self.scheduler.get_action(task)
            self.scheduler.observe(task, action_dict['action'], {'time': 1.0, 'energy': 1.0})
            
        final_size = len(self.scheduler.memory)
        self.assertEqual(final_size, initial_size + 10)
        
    def test_epsilon_decay(self):
        """Test that epsilon decays after observations"""
        initial_eps = self.scheduler.epsilon
        task = self.wg.generate_workload(num_tasks=1)[0]
        
        action_dict = self.scheduler.get_action(task)
        self.scheduler.observe(task, action_dict['action'], {'time': 1.0, 'energy': 1.0})
        
        self.assertLess(self.scheduler.epsilon, initial_eps)

if __name__ == '__main__':
    unittest.main()
