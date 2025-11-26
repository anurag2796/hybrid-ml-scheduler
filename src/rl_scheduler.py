"""
Reinforcement Learning Scheduler
Uses Q-Learning to optimize task scheduling dynamically
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple
from loguru import logger
import random
from dataclasses import dataclass

from .workload_generator import Task
from .profiler import HardwareProfiler
from .online_scheduler import OnlineScheduler, ResourceState

class QLearningScheduler(OnlineScheduler):
    """
    Scheduler that uses Q-Learning to make decisions.
    Inherits from OnlineScheduler to reuse resource management logic.
    """
    
    def __init__(self, num_gpus: int = 4, energy_weight: float = 0.5, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.9995, min_epsilon: float = 0.01):
        
        # Initialize parent without model (we don't use the random forest here)
        self.num_gpus = num_gpus
        self.energy_weight = energy_weight
        self.gpu_states = [
            ResourceState(
                gpu_id=i,
                total_memory=8000,
                available_memory=8000,
                current_load=0.0,
                tasks_running=0
            )
            for i in range(num_gpus)
        ]
        self.task_queue = []
        self.scheduled_tasks = []
        
        # RL Parameters
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-Table: Maps state -> action values
        # State: (TaskSize, Intensity, SystemLoad)
        # Action: 0 (CPU), 1..N (GPU ID + 1)
        self.q_table = {}
        
        # Action space: 0 = CPU, 1..N = GPU 0..N-1
        self.actions = list(range(self.num_gpus + 1))
        
        logger.info(f"QLearningScheduler initialized (Epsilon: {self.epsilon})")

    def _get_state(self, task: Task) -> Tuple[int, int, int]:
        """
        Discretize state space
        Returns: (size_bucket, intensity_bucket, load_bucket)
        """
        # 1. Task Size Bucket (0: Small, 1: Medium, 2: Large)
        if task.size < 1000:
            s_bucket = 0
        elif task.size < 5000:
            s_bucket = 1
        else:
            s_bucket = 2
            
        # 2. Compute Intensity Bucket (0: Low, 1: High)
        i_bucket = 0 if task.compute_intensity < 0.5 else 1
        
        # 3. System Load Bucket (0: Low, 1: High)
        avg_load = np.mean([g.current_load for g in self.gpu_states])
        l_bucket = 0 if avg_load < 0.5 else 1
        
        return (s_bucket, i_bucket, l_bucket)

    def _get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for state-action pair, defaulting to 0.0"""
        return self.q_table.get((state, action), 0.0)

    def _choose_action(self, state: Tuple) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Exploit: Choose max Q-value
        q_values = [self._get_q_value(state, a) for a in self.actions]
        max_q = max(q_values)
        
        # Handle ties randomly
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def _calculate_reward(self, task: Task, action: int, execution_time: float) -> float:
        """
        Calculate reward based on Cost = (1-w)*Time + w*Energy
        Reward is negative cost (we want to maximize reward / minimize cost)
        """
        is_gpu = action > 0
        
        # Estimate energy
        energy = HardwareProfiler.estimate_energy(execution_time, is_gpu)
        
        # Normalize (approximate max values for scaling)
        norm_time = execution_time / 10.0
        norm_energy = energy / 500.0
        
        cost = (1.0 - self.energy_weight) * norm_time + self.energy_weight * norm_energy
        
        return -cost

    def schedule_task(self, task: Task) -> Dict:
        """Schedule a single task using Q-Learning"""
        state = self._get_state(task)
        action = self._choose_action(state)
        
        # Decode action
        if action == 0:
            # CPU
            gpu_id = -1 # Indicator for CPU
            gpu_fraction = 0.0
            cpu_fraction = 1.0
            # CPU is slower for intense tasks
            estimated_time = task.duration_estimate * (1.0 + task.compute_intensity * 2.0)
        else:
            # GPU
            gpu_id = action - 1
            gpu_fraction = 1.0
            cpu_fraction = 0.0
            # GPU is faster for intense tasks
            # Add load factor
            load_factor = 1.0 + self.gpu_states[gpu_id].current_load
            estimated_time = (task.duration_estimate / (1.0 + task.compute_intensity)) * load_factor
            
            # Update GPU state
            self.gpu_states[gpu_id].available_memory -= task.memory_required
            self.gpu_states[gpu_id].tasks_running += 1
            self.gpu_states[gpu_id].current_load = 1.0 - (self.gpu_states[gpu_id].available_memory / self.gpu_states[gpu_id].total_memory)

        # Calculate Reward (Immediate)
        reward = self._calculate_reward(task, action, estimated_time)
        
        # Update Q-Table (Q-Learning Update Rule)
        # Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s', a')) - Q(s,a))
        # Since we don't know next state s' immediately (it depends on next task arrival),
        # we simplify by assuming s' is the current state (or 0). 
        # For a true online learner, we'd update this when the NEXT task arrives.
        # Here, we'll do a simplified update assuming terminal state for this single decision step.
        
        current_q = self._get_q_value(state, action)
        new_q = current_q + self.lr * (reward - current_q)
        self.q_table[(state, action)] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        decision = {
            'task_id': task.task_id,
            'gpu_id': gpu_id, # -1 for CPU
            'gpu_fraction': gpu_fraction,
            'cpu_fraction': cpu_fraction,
            'estimated_time': estimated_time,
            'scheduled_time': 0, # Placeholder
            'action': action,
            'reward': reward
        }
        
        self.scheduled_tasks.append(decision)
        return decision

    def save_model(self, filepath: str):
        """Save Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        logger.info(f"RL model saved to {filepath}")

        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        logger.info(f"RL model loaded from {filepath}")

    def randomize_resources(self):
        """
        Randomize resource states for training.
        Simulates different system conditions (Empty, Busy, Full).
        """
        for gpu in self.gpu_states:
            # Random load between 0.0 and 1.0
            gpu.current_load = random.random()
            
            # Random available memory (correlated with load)
            # High load -> Low memory
            mem_fraction = 1.0 - gpu.current_load
            # Add some noise
            mem_fraction = max(0.0, min(1.0, mem_fraction + random.uniform(-0.1, 0.1)))
            
            gpu.available_memory = gpu.total_memory * mem_fraction
            gpu.tasks_running = int(gpu.current_load * 10)
