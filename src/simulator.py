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
        """
        Simulate the execution of a single task on the virtual hardware.
        
        This method models:
        1. Speedup provided by GPU based on task compute intensity.
        2. Data transfer overhead (PCIE bandwidth) for GPU usage.
        3. Energy consumption based on power profiles of CPU vs GPU.
        4. Financial cost based on cloud pricing models (e.g., AWS spot rates).
        
        Args:
            task: The task object to execute.
            gpu_fraction: Fraction of the task offloaded to GPU (0.0=CPU only, 1.0=GPU only).
            duration_multiplier: Optional scaler for execution time (e.g., to simulate interference).
            
        Returns:
            Dict containing execution metrics (time, energy, cost).
        """
        # 1. Simulate GPU speedup based on compute intensity
        # Tasks with higher compute intensity benefit more from GPU parallelism.
        cpu_time = task.duration_estimate
        
        # Max speedup is 4x for perfectly parallel tasks (intensity=1.0)
        speedup = 1.0 + (3.0 * task.compute_intensity)
        gpu_execution_time = cpu_time / speedup
        
        # 2. Data Transfer Overhead 
        # Simulating PCIe Bandwidth ~16GB/s.
        # This penalizes small, memory-heavy tasks on GPU.
        transfer_time = 0.0
        if gpu_fraction > 0:
            transfer_time = task.memory_required / 16000.0
            
        # 3. Calculate Combined Execution Time
        # The task is split: `gpu_fraction` runs on GPU, rest on CPU.
        gpu_part = (gpu_execution_time + transfer_time) * gpu_fraction
        cpu_part = cpu_time * (1.0 - gpu_fraction)
        
        total_time = (gpu_part + cpu_part) * duration_multiplier
        
        # 4. Calculate Energy (Joules)
        # Power Model: GPU ~200W, CPU ~100W under load.
        # This linear interpolation estimates system power.
        power_rate = gpu_fraction * 200.0 + (1.0 - gpu_fraction) * 100.0
        energy = power_rate * total_time
        
        # 5. Calculate Financial Cost ($)
        # Pricing Model (approx AWS): GPU instance ~$0.36/hr ($0.0001/s), CPU ~$0.07/hr ($0.00002/s).
        # Note: GPU is ~5x more expensive than CPU per second.
        cost_rate = gpu_fraction * 0.0001 + (1.0 - gpu_fraction) * 0.00002
        cost = cost_rate * total_time
        
        result = {
            'task_id': task.task_id,
            'gpu_fraction': gpu_fraction,
            'predicted_time': task.duration_estimate,
            'actual_time': total_time,
            'speedup': task.duration_estimate / (total_time + 1e-9),
            'energy': energy,
            'cost': cost
        }
        
        self.execution_log.append(result)
        return result
    
    def simulate_workload(self, workload: List[Task], strategy: str) -> List[Dict]:
        """Simulate execution of a full workload with a specific strategy"""
        results = []
        for i, task in enumerate(workload):
            gpu_frac = 0.0
            
            if strategy == "round_robin":
                # Strict Alternation: 50% GPU, 30% GPU (Just a simple heuristic pattern)
                gpu_frac = 0.5 if i % 2 == 0 else 0.3
                    
            elif strategy == "random":
                # Uniform Random: Assign anywhere between 20% and 80% to GPU
                gpu_frac = np.random.uniform(0.2, 0.8)
                
            elif strategy == "greedy":
                # Heuristic: Assign to GPU proportional to compute intensity.
                # Compute intensive tasks get more GPU.
                gpu_frac = task.compute_intensity
                
            elif strategy == "cpu_only":
                # Baseline 1: Pure CPU Execution
                gpu_frac = 0.0
                
            elif strategy == "gpu_only":
                # Baseline 2: Pure GPU Execution
                gpu_frac = 1.0
                
            elif strategy == "oracle":
                 # Theoretical Best: Search for optimal fraction by "simulating" all options
                best_time = float('inf')
                best_frac = 0.0
                for frac in np.linspace(0, 1, 11): 
                    t_res = self.simulate_task_execution(task, frac)
                    if t_res['actual_time'] < best_time:
                        best_time = t_res['actual_time']
                        best_frac = frac
                gpu_frac = best_frac
            
            # Execute
            res = self.simulate_task_execution(task, gpu_frac)
            results.append(res)
            
        return results

    def evaluate_baseline_schedulers(self,
                                     workload: List[Task]) -> Dict:
        """Evaluate baseline scheduling strategies (Legacy wrapper)"""
        baselines = {}
        
        for strategy in ['round_robin', 'random', 'greedy']:
            results = self.simulate_workload(workload, strategy)
            times = [r['actual_time'] for r in results]
            baselines[strategy] = {
                'makespan': np.sum(times),
                'avg_time': np.mean(times),
                'max_time': np.max(times),
            }
        
        logger.info(f"Baseline evaluation complete")
        return baselines
