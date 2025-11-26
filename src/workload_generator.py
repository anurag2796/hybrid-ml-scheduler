"""
Workload Generator
Creates synthetic parallel workloads for training and evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class Task:
    """Represents a parallel task"""
    task_id: int
    size: int  # Problem size
    compute_intensity: float  # 0-1, higher = more GPU-friendly
    memory_required: int  # MB
    arrival_time: float
    duration_estimate: float = None
    dependencies: List[int] = None  # List of parent task IDs
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'size': self.size,
            'compute_intensity': self.compute_intensity,
            'memory_required': self.memory_required,
            'arrival_time': self.arrival_time,
            'duration_estimate': self.duration_estimate,
            'dependencies': str(self.dependencies) # Store as string for CSV simplicity
        }


class WorkloadGenerator:
    """Generates synthetic but realistic workloads"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.tasks = []
        logger.info(f"WorkloadGenerator initialized with seed {seed}")
    
    def generate_workload_stream(
        self,
        num_tasks: int = 1000,
        task_size_range: Tuple[int, int] = (100, 5000),
        compute_intensity_range: Tuple[float, float] = (0.1, 1.0),
        memory_range: Tuple[int, int] = (10, 500),
        arrival_rate: float = 100.0,
        duration_model: str = "size_based",
        dependency_prob: float = 0.2,
        max_dependencies: int = 3
    ):
        """
        Generator that yields tasks one by one for memory efficiency.
        
        This method uses Python's `yield` keyword to create a generator. This is crucial for 
        large-scale simulations where creating millions of Task objects at once would consume 
        too much RAM. Instead, tasks are created on-the-fly as they are requested.
        
        Args:
            num_tasks: Number of tasks to generate
            task_size_range: (min_size, max_size) for the problem size
            compute_intensity_range: (min_intensity, max_intensity) where higher means more GPU-bound
            memory_range: (min_memory_MB, max_memory_MB) required by the task
            arrival_rate: Average number of tasks arriving per second (Poisson process parameter)
            duration_model: Model to estimate task duration ('size_based' or 'random')
            dependency_prob: Probability that a task depends on previous tasks
            max_dependencies: Maximum number of dependencies per task
            
            
        Yields:
            Task object: A single generated task
        """
        current_time = 0.0
        
        for i in range(num_tasks):
            # Generate inter-arrival time using Exponential distribution
            # This models a Poisson arrival process, which is standard for queuing systems
            inter_arrival = np.random.exponential(1.0 / arrival_rate)
            current_time += inter_arrival
            
            # Randomly sample task characteristics from Uniform distributions
            size = np.random.randint(task_size_range[0], task_size_range[1])
            compute_intensity = np.random.uniform(
                compute_intensity_range[0], 
                compute_intensity_range[1]
            )
            memory = np.random.randint(memory_range[0], memory_range[1])
            
            # Generate dependencies
            dependencies = []
            if i > 0 and np.random.random() < dependency_prob:
                # Can only depend on previous tasks
                # Look back window to avoid long chains
                window_size = min(i, 50) 
                potential_parents = list(range(max(0, i - window_size), i))
                num_deps = np.random.randint(1, min(len(potential_parents), max_dependencies) + 1)
                dependencies = sorted(np.random.choice(potential_parents, num_deps, replace=False).tolist())
            
            
            # Estimate duration based on size and intensity
            if duration_model == "size_based":
                # Heuristic: Duration grows super-linearly with size (O(N^1.5))
                # and is inversely proportional to compute intensity (more intense = harder)
                # Note: This is a synthetic model for demonstration
                base_duration = (size / 1000) ** 1.5
                duration = base_duration / (compute_intensity + 0.5)
            else:
                # Fallback to simple random duration
                duration = np.random.exponential(0.1)
            
            # Create the Task object
            task = Task(
                task_id=i,
                size=size,
                compute_intensity=compute_intensity,
                memory_required=memory,
                arrival_time=current_time,
                duration_estimate=duration,
                dependencies=dependencies
            )
            
            # Yield the task to the consumer immediately
            # This pauses execution here until the next task is requested
            yield task

    def generate_workload(
        self,
        num_tasks: int = 1000,
        task_size_range: Tuple[int, int] = (100, 5000),
        compute_intensity_range: Tuple[float, float] = (0.1, 1.0),
        memory_range: Tuple[int, int] = (10, 500),
        arrival_rate: float = 100.0,
        duration_model: str = "size_based",
        dependency_prob: float = 0.2,
        max_dependencies: int = 3
    ) -> List[Task]:
        """
        Generate synthetic workload (returns full list).
        
        This method is a wrapper around `generate_workload_stream` for backward compatibility.
        It consumes the entire stream and returns a list of all tasks.
        
        Args:
            num_tasks: Number of tasks to generate
            task_size_range: (min_size, max_size)
            compute_intensity_range: (min_intensity, max_intensity)
            memory_range: (min_memory_MB, max_memory_MB)
            arrival_rate: Tasks per second
            duration_model: How to estimate task duration
            
        Returns:
            List of Task objects
        """
        logger.info(f"Generating {num_tasks} tasks...")
        
        # Consume the generator to create a list
        # WARNING: This may use a lot of memory for very large num_tasks
        self.tasks = list(self.generate_workload_stream(
            num_tasks=num_tasks,
            task_size_range=task_size_range,
            compute_intensity_range=compute_intensity_range,
            memory_range=memory_range,
            arrival_rate=arrival_rate,
            duration_model=duration_model,
            dependency_prob=dependency_prob,
            max_dependencies=max_dependencies
        ))
        
        logger.info(f"Generated {len(self.tasks)} tasks")
        return self.tasks
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert tasks to pandas DataFrame"""
        data = [task.to_dict() for task in self.tasks]
        df = pd.DataFrame(data)
        logger.debug(f"Converted {len(df)} tasks to DataFrame")
        return df
    
    def save_workload(self, filepath: str):
        """Save workload to CSV"""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Workload saved to {filepath}")
    
    @staticmethod
    def load_workload(filepath: str) -> 'WorkloadGenerator':
        """Load workload from CSV"""
        df = pd.read_csv(filepath)
        generator = WorkloadGenerator()
        
        for _, row in df.iterrows():
            task = Task(
                task_id=int(row['task_id']),
                size=int(row['size']),
                compute_intensity=float(row['compute_intensity']),
                memory_required=int(row['memory_required']),
                arrival_time=float(row['arrival_time']),
                duration_estimate=float(row['duration_estimate']),
                dependencies=eval(str(row.get('dependencies', '[]'))) # Safe eval for list string
            )
            generator.tasks.append(task)
        
        logger.info(f"Loaded {len(generator.tasks)} tasks from {filepath}")
        return generator
    
    def get_statistics(self) -> Dict:
        """Get workload statistics"""
        if not self.tasks:
            return {}
        
        sizes = [t.size for t in self.tasks]
        intensities = [t.compute_intensity for t in self.tasks]
        
        stats = {
            'num_tasks': len(self.tasks),
            'total_duration': self.tasks[-1].arrival_time,
            'avg_size': np.mean(sizes),
            'avg_intensity': np.mean(intensities),
            'size_std': np.std(sizes),
            'intensity_std': np.std(intensities),
        }
        
        logger.debug(f"Workload stats: {stats}")
        return stats
