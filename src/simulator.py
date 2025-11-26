"""
Virtual Multi-GPU Simulator for baseline comparisons
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger
import time

from .workload_generator import Task, WorkloadGenerator


class VirtualMultiGPU:
    """Simulates multiple GPUs"""
    
    def __init__(self, num_gpus: int = 4, memory_per_gpu: int = 8000):
        self.num_gpus = num_gpus
        self.memory_per_gpu = memory_per_gpu
        self.task_timeline = []
        self.execution_log = []
        
        logger.info(f"VirtualMultiGPU initialized with {num_gpus} GPUs")
    
    def simulate_task_execution(self, 
                               task: Task,
                               gpu_fraction: float,
                               duration_multiplier: float = 1.0) -> Dict:
        """Simulate task execution"""
        # Simulate GPU speedup based on compute intensity
        cpu_time = task.duration_estimate
        # Higher intensity = better GPU speedup (up to 4x)
        speedup = 1.0 + (3.0 * task.compute_intensity)
        gpu_execution_time = cpu_time / speedup
        
        # Data Transfer Overhead (PCIe Bandwidth ~16GB/s)
        # Only applies if using GPU
        transfer_time = 0.0
        if gpu_fraction > 0:
            transfer_time = task.memory_required / 16000.0
            
        # Combined time
        # GPU part includes transfer time
        gpu_part = (gpu_execution_time + transfer_time) * gpu_fraction
        cpu_part = cpu_time * (1.0 - gpu_fraction)
        
        total_time = (gpu_part + cpu_part) * duration_multiplier
        
        result = {
            'task_id': task.task_id,
            'gpu_fraction': gpu_fraction,
            'predicted_time': task.duration_estimate,
            'actual_time': total_time,
            'speedup': task.duration_estimate / total_time if total_time > 0 else 0,
        }
        
        self.execution_log.append(result)
        return result
    
    def evaluate_baseline_schedulers(self,
                                     workload: List[Task]) -> Dict:
        """Evaluate baseline scheduling strategies"""
        baselines = {}
        
        # Round-Robin: Alternate GPU allocation
        rr_times = []
        for i, task in enumerate(workload):
            gpu_frac = 0.5 if i % 2 == 0 else 0.3
            result = self.simulate_task_execution(task, gpu_frac)
            rr_times.append(result['actual_time'])
        baselines['round_robin'] = {
            'makespan': np.sum(rr_times),
            'avg_time': np.mean(rr_times),
            'max_time': np.max(rr_times),
        }
        
        # Random: Random allocation
        random_times = []
        for task in workload:
            gpu_frac = np.random.uniform(0.2, 0.8)
            result = self.simulate_task_execution(task, gpu_frac)
            random_times.append(result['actual_time'])
        baselines['random'] = {
            'makespan': np.sum(random_times),
            'avg_time': np.mean(random_times),
            'max_time': np.max(random_times),
        }
        
        # Greedy: Assign based on compute intensity
        greedy_times = []
        for task in workload:
            gpu_frac = task.compute_intensity
            result = self.simulate_task_execution(task, gpu_frac)
            greedy_times.append(result['actual_time'])
        baselines['greedy'] = {
            'makespan': np.sum(greedy_times),
            'avg_time': np.mean(greedy_times),
            'max_time': np.max(greedy_times),
        }

        # Offline Optimal: Theoretically best possible split
        # We find the gpu_fraction that minimizes time for EACH task individually
        optimal_times = []
        for task in workload:
            # Simple search for optimal fraction
            best_time = float('inf')
            for frac in np.linspace(0, 1, 21): # Check 0.0, 0.05, ..., 1.0
                result = self.simulate_task_execution(task, frac)
                if result['actual_time'] < best_time:
                    best_time = result['actual_time']
            optimal_times.append(best_time)
            
        baselines['offline_optimal'] = {
            'makespan': np.sum(optimal_times),
            'avg_time': np.mean(optimal_times),
            'max_time': np.max(optimal_times),
        }
        
        logger.info(f"Baseline evaluation complete")
        return baselines
