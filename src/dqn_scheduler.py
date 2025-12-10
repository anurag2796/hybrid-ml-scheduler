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
from typing import Dict, List, Tuple, Any
from loguru import logger
import pickle

from .workload_generator import Task
from .profiler import HardwareProfiler
from .online_scheduler import OnlineScheduler, ResourceState

# Define the Neural Network
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        
        # Enhanced Architecture with Batch Norm and Dropout
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Dueling DQN: Value and Advantage streams
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, output_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Handle single sample (batch norm requires > 1 sample usually, but we can manage)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.relu(self.fc1(x))
        # x = self.bn1(x) # Skip BN for online single-sample inference stability or use eval mode
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
    Now supports interactive learning loop.
    """
    
    def __init__(self, num_gpus: int = 4, energy_weight: float = 0.5, 
                 learning_rate: float = 0.0001, gamma: float = 0.99, 
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, 
                 epsilon_decay: float = 0.9995, buffer_size: int = 50000,
                 batch_size: int = 64, target_update: int = 50,
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
        
        # RL Parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        self.energy_weight = energy_weight
        
        # State Dimension: 
        # Task Features (3): Size, Intensity, Memory
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
        
        logger.info(f"DQNScheduler initialized on {self.device} with Enhanced Architecture")

    def _get_state_vector(self, task: Task) -> np.ndarray:
        """
        Construct continuous state vector [Size, Intensity, Memory]
        """
        task_features = [
            task.size / 10000.0, # Normalize
            task.compute_intensity,
            task.memory_required / 5000.0 # Normalize
        ]
        return np.array(task_features, dtype=np.float32)

    def get_action(self, task: Task) -> Dict:
        """
        Select an action for the given task.
        Returns detailed action info for simulation.
        """
        state = self._get_state_vector(task)
        
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
                
        # Decode Action
        if action == 0: # CPU
            gpu_fraction = 0.0
            gpu_id = -1
        else: # GPU
            gpu_fraction = 1.0 # Simple discrete choice: Full GPU
            gpu_id = action - 1
            
        return {
            'action': action,
            'gpu_fraction': gpu_fraction,
            'gpu_id': gpu_id,
            'state_vector': state # Keep for observation
        }

    def observe(self, task: Task, action: int, reward_metrics: Dict):
        """
        Receive feedback from the environment (Simulator) and learn.
        """
        # 1. Calculate Reward
        # Minimize Cost => Maximize Negative Cost
        # Cost = (1-w)*Time + w*Energy
        
        time_taken = reward_metrics['time']
        energy_used = reward_metrics['energy']
        
        # Normalize for numerical stability
        norm_time = time_taken / 10.0 
        norm_energy = energy_used / 500.0
        
        cost = (1.0 - self.energy_weight) * norm_time + self.energy_weight * norm_energy
        reward = -cost
        
        # 2. Store in Memory
        current_state = self._get_state_vector(task)
        
        # Simplified Next State: Since tasks are independent IID in this generator,
        # the next state is just a new random task.
        # Ideally, we'd pass the *next* task's state here, but we can treat this as a terminal state for the task.
        # Or better: Standard Q-learning assumes s' is the state the agent ENDS UP in.
        # But here the agent stays in the "Scheduler" state and receives a NEW task.
        # So s' is effectively the state vector of the NEXT task.
        # For simplicity in this IID setting, we can set done=True effectively, 
        # or use the current state as a proxy if we want to learn generic value of state.
        
        # Let's treat each scheduling decision as an episode of length 1 for now (Bandit-like but with state),
        # or assume s' is 0 vector (terminal).
        # However, to use the Q-learning update rule properly with gamma, we usually need s'.
        # Since the next task is random, V(s') is just expected value of random tasks.
        
        next_state = np.zeros_like(current_state) # Dummy
        done = True # Treat as terminal for this specific task
        
        self.memory.push(current_state, action, reward, next_state, done)
        
        # 3. Train
        loss = self.train_step()
        
        # 4. Decay Epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 5. Log/Monitor (Optional)
        # if self.update_counter % 100 == 0:
        #    logger.debug(f"RL Train: Eps={self.epsilon:.4f}, Reward={reward:.4f}, Loss={loss}")

    def train_step(self) -> float:
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(state).gather(1, action)
        
        # Compute Target Q
        with torch.no_grad():
            # Double DQN
            next_actions = self.policy_net(next_state).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_state).gather(1, next_actions)
            expected_q_values = reward + (self.gamma * next_q_values * (1 - done))
            
        # Loss
        loss = nn.SmoothL1Loss()(q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def save_model(self, filepath: str):
        torch.save(self.policy_net.state_dict(), filepath)

    def load_model(self, filepath: str):
        self.policy_net.load_state_dict(torch.load(filepath))
        self.policy_net.eval()
        
    # Legacy method wrapper if needed, but we use get_action now
    def schedule_task(self, task: Task) -> Dict:
        # This is for the OnlineScheduler interface if used directly
        action_dict = self.get_action(task)
        # We can't observe() here because we don't have the result yet.
        # This compatibility method might need adjustment if used elsewhere.
        return {
            'task_id': task.task_id,
            'gpu_id': action_dict['gpu_id'],
            'gpu_fraction': action_dict['gpu_fraction'],
            'estimated_time': task.duration_estimate, # Placeholder
            'scheduled_time': 0,
            'action': action_dict['action']
        }
