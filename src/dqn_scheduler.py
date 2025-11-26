"""
Deep Q-Network (DQN) Scheduler
Uses Deep Reinforcement Learning to optimize task scheduling
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Tuple
from loguru import logger
import pickle

from .workload_generator import Task
from .profiler import HardwareProfiler
from .online_scheduler import OnlineScheduler, ResourceState

# Define the Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Dueling DQN: Value and Advantage streams
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return val + (adv - adv.mean(dim=1, keepdim=True))

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQNScheduler(OnlineScheduler):
    """
    Scheduler that uses Deep Q-Learning (DQN) to make decisions.
    """
    
    def __init__(self, num_gpus: int = 4, energy_weight: float = 0.5, 
                 learning_rate: float = 0.0001, gamma: float = 0.99, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                 epsilon_decay: float = 0.9999, buffer_size: int = 50000,
                 batch_size: int = 128, target_update: int = 100,
                 monitor_callback=None):
        
        super().__init__(model=None, num_gpus=num_gpus, energy_weight=energy_weight, monitor_callback=monitor_callback)
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
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # State Dimension: 
        # Task Features (3): Size, Intensity, Memory
        # System Features (Num GPUs): REMOVED to match simulator physics
        # Total = 3
        self.state_dim = 3
        
        # Action Dimension: CPU + Num GPUs
        self.action_dim = 1 + num_gpus
        
        # Networks
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        
        # Track last state/action for learning
        self.last_state = None
        self.last_action = None
        
        logger.info(f"DQNScheduler initialized on {self.device}")

    def _get_state_vector(self, task: Task) -> np.ndarray:
        """
        Construct continuous state vector
        [TaskSize, Intensity, Memory]
        """
        # Normalize task features (approximate max values)
        task_features = [
            task.size / 10000.0,
            task.compute_intensity,
            task.memory_required / 5000.0
        ]
        
        # gpu_loads = [g.current_load for g in self.gpu_states]
        # REMOVED: Load is irrelevant for this benchmark
        
        return np.array(task_features, dtype=np.float32)

    def _choose_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def _calculate_reward(self, task: Task, action: int, execution_time: float) -> float:
        """Calculate reward (negative weighted cost)"""
        is_gpu = action > 0
        energy = HardwareProfiler.estimate_energy(execution_time, is_gpu)
        
        # Less aggressive normalization
        norm_time = execution_time  # Raw seconds (0.1 - 10.0 range usually)
        norm_energy = energy / 100.0 # Scale down energy (0-500 -> 0-5)
        
        cost = (1.0 - self.energy_weight) * norm_time + self.energy_weight * norm_energy
        
        # Return negative cost directly
        return -cost

    def train_step(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(state).gather(1, action)
        
        # Double DQN:
        # 1. Select action using policy_net
        with torch.no_grad():
            next_actions = self.policy_net(next_state).argmax(1).unsqueeze(1)
            
            # 2. Evaluate action using target_net
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            
            expected_q_values = reward + (self.gamma * next_q_values * (1 - done))
            
        # Loss (Huber Loss for stability)
        loss = nn.SmoothL1Loss()(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _get_optimal_action(self, task: Task) -> int:
        """
        Determine the optimal action (Oracle) for a task based on current state.
        Minimizes the weighted cost of Time and Energy.
        """
        best_action = -1
        min_cost = float('inf')
        
        # Check CPU (Action 0)
        cpu_time = task.duration_estimate
        cpu_energy = HardwareProfiler.estimate_energy(cpu_time, is_gpu=False)
        
        # Match reward function logic EXACTLY
        # norm_time = execution_time
        # norm_energy = energy / 100.0
        cpu_cost = (1.0 - self.energy_weight) * cpu_time + self.energy_weight * (cpu_energy / 100.0)
        
        if cpu_cost < min_cost:
            min_cost = cpu_cost
            best_action = 0
            
        # Check GPUs (Action 1..N)
        for i, gpu in enumerate(self.gpu_states):
            # NO LOAD FACTOR (Match Simulator Physics)
            
            speedup = 1.0 + (3.0 * task.compute_intensity)
            gpu_execution_time = task.duration_estimate / speedup
            transfer_time = task.memory_required / 16000.0
            
            gpu_time = (gpu_execution_time + transfer_time)
            gpu_energy = HardwareProfiler.estimate_energy(gpu_time, is_gpu=True)
            
            # Match reward function logic EXACTLY
            gpu_cost = (1.0 - self.energy_weight) * gpu_time + self.energy_weight * (gpu_energy / 100.0)
            
            if gpu_cost < min_cost:
                min_cost = gpu_cost
                best_action = i + 1
                
        return best_action

    def pretrain(self, tasks: List[Task], epochs: int = 5):
        """
        Supervised pre-training using optimal decisions (Oracle).
        """
        logger.info(f"Pre-training DQN on {len(tasks)} tasks for {epochs} epochs...")
        
        # Generate Dataset
        states = []
        actions = []
        
        for task in tasks:
            # self.randomize_resources() # REMOVED: State no longer depends on resources
            state = self._get_state_vector(task)
            optimal_action = self._get_optimal_action(task)
            
            states.append(state)
            actions.append(optimal_action)
            
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        
        # Train Loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001) # Higher LR for pre-training
        
        self.policy_net.train()
        
        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(len(states))
            states = states[indices]
            actions = actions[indices]
            
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.policy_net(batch_states)
                loss = criterion(outputs, batch_actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        # Update target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.info("Pre-training complete.")

    def schedule_task(self, task: Task) -> Dict:
        """Schedule a task and train"""
        current_state = self._get_state_vector(task)
        action = self._choose_action(current_state)
        
        # Execute Action (Simulated)
        if action == 0: # CPU
            gpu_id = -1
            gpu_fraction = 0.0
            cpu_fraction = 1.0
            estimated_time = task.duration_estimate
        else: # GPU
            gpu_id = action - 1
            gpu_fraction = 1.0
            cpu_fraction = 0.0
            
            # Match Simulator Physics
            speedup = 1.0 + (3.0 * task.compute_intensity)
            gpu_execution_time = task.duration_estimate / speedup
            transfer_time = task.memory_required / 16000.0
            
            base_time = gpu_execution_time + transfer_time
            
            # Contention
            # load_factor = 1.0 + self.gpu_states[gpu_id].current_load
            # IGNORE LOAD FACTOR to match Simulator Physics
            estimated_time = base_time # * load_factor
            
            # Update GPU state
            self.gpu_states[gpu_id].available_memory -= task.memory_required
            self.gpu_states[gpu_id].tasks_running += 1
            self.gpu_states[gpu_id].current_load = 1.0 - (self.gpu_states[gpu_id].available_memory / self.gpu_states[gpu_id].total_memory)

        reward = self._calculate_reward(task, action, estimated_time)
        
        # Store transition (s, a, r, s', done)
        # Note: We don't have the "true" next state (next task) yet.
        # But for independent task scheduling, we can treat the next state as the state AFTER resource update
        # OR we can just use the current state as next state (simplified).
        # Better: Use the state vector of the NEXT task when it arrives.
        # However, to keep it simple and contained:
        # We will store the transition from Last State -> Current State
        
        if self.last_state is not None:
            self.memory.push(self.last_state, self.last_action, reward, current_state, False)
            self.train_step()
            
        self.last_state = current_state
        self.last_action = action
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        decision = {
            'task_id': task.task_id,
            'gpu_id': gpu_id,
            'gpu_fraction': gpu_fraction,
            'cpu_fraction': cpu_fraction,
            'estimated_time': estimated_time,
            'scheduled_time': 0,
            'action': action,
            'reward': reward
        }
        
        self.scheduled_tasks.append(decision)
        
        if self.monitor_callback:
            self.monitor_callback({
                'type': 'decision',
                'data': decision,
                'utilization': self.get_utilization(),
                'rl_metrics': {
                    'epsilon': self.epsilon,
                    'reward': reward,
                    'loss': 0.0 # Placeholder, could track loss if returned from train_step
                }
            })
            
        return decision

    def randomize_resources(self):
        """Randomize resource states for training"""
        for gpu in self.gpu_states:
            gpu.current_load = random.random()
            mem_fraction = 1.0 - gpu.current_load
            mem_fraction = max(0.0, min(1.0, mem_fraction + random.uniform(-0.1, 0.1)))
            gpu.available_memory = gpu.total_memory * mem_fraction
            gpu.tasks_running = int(gpu.current_load * 10)

    def save_model(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)
        logger.info(f"DQN model saved to {filepath}")

    def load_model(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.policy_net.eval()
        logger.info(f"DQN model loaded from {filepath}")
    
    def reset_state(self):
        """Reset scheduler state"""
        for gpu_state in self.gpu_states:
            gpu_state.available_memory = gpu_state.total_memory
            gpu_state.current_load = 0.0
            gpu_state.tasks_running = 0
        self.task_queue.clear()
        self.scheduled_tasks.clear()
        self.last_state = None
        self.last_action = None
