"""
Online Scheduler - Real-time task scheduling using ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from loguru import logger
from dataclasses import dataclass
import time

from .ml_models import PerformancePredictor
from .workload_generator import Task
from .profiler import HardwareProfiler


@dataclass
class ResourceState:
    """Current state of a virtual GPU"""
    gpu_id: int
    total_memory: int
    available_memory: int
    current_load: float
    tasks_running: int


class OnlineScheduler:
    """Uses trained ML models to dynamically schedule tasks at runtime"""
    
    def __init__(self, model: PerformancePredictor, num_gpus: int = 4, energy_weight: float = 0.5, monitor_callback=None):
        self.model = model
        self.num_gpus = num_gpus
        self.energy_weight = energy_weight
        self.monitor_callback = monitor_callback
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
        self.task_timeline = []
        
        logger.info(f"OnlineScheduler initialized with {num_gpus} virtual GPUs")
    
    def submit_task(self, task: Task):
        """Submit a task for scheduling"""
        self.task_queue.append(task)
        logger.debug(f"Task {task.task_id} submitted")
    
    def _predict_placement(self, task: Task) -> Dict:
        """Use ML model to predict best placement"""
        # Prepare features for model
        features = pd.DataFrame([{
            'task_size': task.size,
            'compute_intensity': task.compute_intensity,
            'memory_required': task.memory_required,
            'memory_per_size': task.memory_required / (task.size + 1),
            'compute_to_memory': task.compute_intensity / (task.memory_required + 1),
        }])
        
        # Get model prediction
        gpu_fraction_pred = self.model.predict(features)[0]
        
        # Clamp prediction
        gpu_fraction_pred = np.clip(gpu_fraction_pred, 0.0, 1.0)
        
        return {
            'gpu_fraction': gpu_fraction_pred,
            'cpu_fraction': 1.0 - gpu_fraction_pred,
        }
    
    def _find_best_gpu(self, task: Task, gpu_fraction: float) -> int:
        """Find best GPU to assign task to using Energy-Aware Cost"""
        best_gpu = -1
        min_cost = float('inf')
        
        for i in range(self.num_gpus):
            gpu_state = self.gpu_states[i]
            
            # Estimate execution time (simple model: load increases time)
            base_time = task.duration_estimate
            load_factor = 1.0 + gpu_state.current_load
            estimated_time = base_time * load_factor
            
            # Estimate energy
            # If gpu_fraction > 0.5, we treat it as GPU-heavy
            is_gpu_task = gpu_fraction > 0.5
            estimated_energy = HardwareProfiler.estimate_energy(estimated_time, is_gpu_task)
            
            # Weighted Cost
            # Normalize time and energy to roughly 0-1 range for fair weighting
            # (Assuming max time ~10s, max energy ~500J)
            norm_time = estimated_time / 10.0
            norm_energy = estimated_energy / 500.0
            
            cost = (1.0 - self.energy_weight) * norm_time + self.energy_weight * norm_energy
            
            if cost < min_cost:
                min_cost = cost
                best_gpu = i
                
        return best_gpu
    
    def schedule_task(self, task: Task) -> Dict:
        """Schedule a single task"""
        # Get placement prediction
        placement = self._predict_placement(task)
        
        # Find best GPU
        gpu_id = self._find_best_gpu(task, placement['gpu_fraction'])
        
        # Allocate resources
        gpu_state = self.gpu_states[gpu_id]
        
        gpu_state.available_memory -= task.memory_required
        gpu_state.tasks_running += 1
        gpu_state.current_load = 1.0 - (gpu_state.available_memory / gpu_state.total_memory)
        
        decision = {
            'task_id': task.task_id,
            'gpu_id': gpu_id,
            'gpu_fraction': placement['gpu_fraction'],
            'cpu_fraction': placement['cpu_fraction'],
            'estimated_time': task.duration_estimate,
            'scheduled_time': time.time(),
        }
        
        self.scheduled_tasks.append(decision)
        logger.debug(f"Task {task.task_id} scheduled on GPU {gpu_id}")
        
        if self.monitor_callback:
            self.monitor_callback({
                'type': 'decision',
                'data': decision,
                'utilization': self.get_utilization()
            })
        
        return decision
    
    def process_queue(self) -> List[Dict]:
        """Process all tasks in queue respecting dependencies"""
        logger.info(f"Processing queue with {len(self.task_queue)} tasks")
        
        # Separate tasks into ready (no deps) and waiting
        ready_queue = []
        waiting_queue = []
        completed_tasks = set()
        
        # Initial sort
        for task in self.task_queue:
            if not task.dependencies:
                ready_queue.append(task)
            else:
                waiting_queue.append(task)
        
        decisions = []
        
        # Process loop
        while ready_queue:
            # Get next ready task
            task = ready_queue.pop(0)
            
            # Schedule it
            decision = self.schedule_task(task)
            decisions.append(decision)
            completed_tasks.add(task.task_id)
            
            # Check waiting queue for newly ready tasks
            # Iterate backwards to allow safe removal
            for i in range(len(waiting_queue) - 1, -1, -1):
                waiting_task = waiting_queue[i]
                # Check if all dependencies are met
                if all(dep_id in completed_tasks for dep_id in waiting_task.dependencies):
                    ready_queue.append(waiting_task)
                    waiting_queue.pop(i)
        
        if waiting_queue:
            logger.warning(f"Could not schedule {len(waiting_queue)} tasks due to missing dependencies/cycles")
            
        self.task_queue.clear()
        logger.info(f"Processed {len(decisions)} tasks")
        
        return decisions
    
    def get_utilization(self) -> Dict:
        """Get current resource utilization"""
        utilizations = {}
        for gpu_state in self.gpu_states:
            utilizations[f'gpu_{gpu_state.gpu_id}'] = {
                'utilization': float(gpu_state.current_load),
                'tasks_running': gpu_state.tasks_running,
                'available_memory': gpu_state.available_memory,
            }
        
        avg_util = np.mean([s.current_load for s in self.gpu_states])
        utilizations['average_utilization'] = float(avg_util)
        
        return utilizations
    
    def reset_state(self):
        """Reset scheduler state"""
        for gpu_state in self.gpu_states:
            gpu_state.available_memory = gpu_state.total_memory
            gpu_state.current_load = 0.0
            gpu_state.tasks_running = 0
        
        self.task_queue.clear()
        self.scheduled_tasks.clear()
        
        logger.info("Scheduler state reset")
