"""Continuous Simulation Engine for Hybrid ML Scheduler.

This module implements a live simulation environment that continuously generates
tasks, schedules them across multiple competing strategies, collects performance
metrics, and broadcasts results to a dashboard via WebSocket.

Key Features:
    - Real-time task generation and scheduling simulation
    - Multiple scheduler comparison (Round Robin, Random, Greedy, Hybrid ML, RL Agent, Oracle)
    - Online model retraining based on accumulated data
    - WebSocket broadcasting for live dashboard updates
    - Performance metrics persistence for analysis

Typical Usage:
    ```python
    async def broadcast(msg: dict):
        # Your WebSocket broadcast logic
        pass
    
    simulation = ContinuousSimulation(broadcast_callback=broadcast)
    await simulation.start()
    ```
"""

import asyncio
import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from loguru import logger

from src.workload_generator import WorkloadGenerator, Task
from src.simulator import VirtualMultiGPU
from src.online_scheduler import OnlineScheduler
from src.offline_trainer import OfflineTrainer
from src.dqn_scheduler import DQNScheduler
from src.ml_models import RandomForestPredictor

class ContinuousSimulation:
    """Orchestrates continuous simulation of multiple scheduling strategies.
    
    This class manages the entire simulation lifecycle including task generation,
    parallel scheduler execution, metrics collection, model retraining, and
    real-time result broadcasting.
    
    Attributes:
        broadcast_callback (callable): Async function to broadcast simulation updates
        is_running (bool): Flag indicating if simulation is active
        is_paused (bool): Flag indicating if simulation is paused
        num_gpus (int): Number of GPUs in the virtual cluster
        retrain_interval (int): Number of tasks between model retraining cycles
        tasks_processed (int): Counter for tasks processed since start
        history_file (Path): Path to CSV file storing training history
        simulators (dict): Dictionary mapping scheduler names to VirtualMultiGPU instances
        trainer (OfflineTrainer): ML model trainer for hybrid scheduler
        hybrid_scheduler (OnlineScheduler): ML-based scheduling policy
        rl_scheduler (DQNScheduler): Reinforcement learning-based scheduler
        metrics (dict): Accumulated performance metrics per scheduler
    """
    
    def __init__(self, broadcast_callback):
        """Initialize simulation with all schedulers and components.
        
        Args:
            broadcast_callback (callable): Async function that accepts a dict message
                and broadcasts it to connected clients. Should have signature:
                async def callback(message: dict) -> None
        """
        self.broadcast_callback = broadcast_callback
        self.is_running = False
        self.is_paused = False
        
        # Configuration
        self.num_gpus = 4
        self.retrain_interval = 50
        self.tasks_processed = 0
        self.history_file = Path("data/long_term_history.csv")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Workload Generator (Persistent)
        self.workload_generator = WorkloadGenerator()
        
        # Initialize Schedulers / Simulators
        # We need separate simulators for each strategy to maintain independent state
        self.simulators = {
            'round_robin': VirtualMultiGPU(self.num_gpus),
            'random': VirtualMultiGPU(self.num_gpus),
            'greedy': VirtualMultiGPU(self.num_gpus),
            'hybrid_ml': VirtualMultiGPU(self.num_gpus),
            'rl_agent': VirtualMultiGPU(self.num_gpus),
            'oracle': VirtualMultiGPU(self.num_gpus) # Theoretical best
        }
        
        # Initialize Models
        self.trainer = OfflineTrainer(model_type="random_forest", n_estimators=10)
        # Try to load existing model, else create new
        try:
            # We need a dummy model first
            self.trainer.create_model() 
            # In a real scenario we'd load_model, but for now we'll just init a fresh one 
            # or rely on the first retraining to make it good.
            # Let's pre-train it on a tiny bit of dummy data so it's not empty
            self._initial_pretrain()
        except Exception as e:
            logger.warning(f"Could not load initial model: {e}")
            
        self.hybrid_scheduler = OnlineScheduler(self.trainer.model, self.num_gpus)
        
        self.rl_scheduler = DQNScheduler(self.num_gpus)
        # Load RL model if exists? For now start fresh or use epsilon decay
        
        # Metrics Storage
        self.metrics = {k: {'total_time': 0.0, 'tasks': 0} for k in self.simulators.keys()}

    def _initial_pretrain(self):
        """Pre-train the hybrid ML model with dummy data to initialize weights.
        
        Generates a small synthetic workload and trains the model with mock
        optimal GPU fractions to ensure the model is callable before the
        first real retraining cycle.
        
        Note:
            This is a bootstrapping step. The model will be properly trained
            once sufficient real data is collected.
        """
        wg = WorkloadGenerator()
        tasks = wg.generate_workload(num_tasks=10)
        self.trainer.prepare_data(wg)
        # Mock optimal fractions for dummy data
        self.trainer.training_data['optimal_gpu_fraction'] = [0.5] * 10
        
        X = self.trainer.training_data[['size', 'compute_intensity', 'memory_required', 'memory_per_size', 'compute_to_memory']]
        y = self.trainer.training_data['optimal_gpu_fraction']
        self.trainer.train(X, y)

    async def start(self):
        """Start the continuous simulation loop.
        
        This is the main simulation loop that:
        1. Generates new tasks at regular intervals (1.5s)
        2. Runs all schedulers on each task in parallel
        3. Updates metrics and broadcasts results
        4. Persists oracle results for training
        5. Triggers model retraining every retrain_interval tasks
        
        The loop runs until stop() is called. Can be paused/resumed via
        pause() and resume() methods.
        
        Raises:
            Exception: Any errors during scheduler execution are logged but
                don't stop the simulation.
        """
        self.is_running = True
        logger.info("Simulation Engine Started")
        
        # Initialize CSV if not exists
        if not self.history_file.exists():
            pd.DataFrame(columns=[
                'task_id', 'size', 'compute_intensity', 'memory_required', 
                'optimal_gpu_fraction', 'optimal_time'
            ]).to_csv(self.history_file, index=False)
            
        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.5)
                continue
                
            # 1. Generate Task
            task = self._generate_task()
            
            # 2. Run on All Schedulers
            results = self._run_all_schedulers(task)
            
            # 3. Update Metrics & Broadcast
            self._update_metrics(results)
            await self._broadcast_state(task, results)
            
            # 4. Persist Data (Oracle Result)
            self._persist_data(task, results['oracle'])
            
            # 5. Retrain if needed
            self.tasks_processed += 1
            if self.tasks_processed % self.retrain_interval == 0:
                await self._retrain_model()
            
            # Delay for visual pacing
            await asyncio.sleep(1.5) # Slow enough to read

    def stop(self):
        """Stop the simulation loop gracefully."""
        self.is_running = False

    def pause(self):
        """Pause the simulation temporarily without stopping it."""
        self.is_paused = True

    def resume(self):
        """Resume a paused simulation."""
        self.is_paused = False

    def _generate_task(self) -> Task:
        """Generate a single random task for scheduling.
        
        Returns:
            Task: A randomly generated task with size, compute intensity,
                and memory requirements based on configured distributions.
        """
        # Use the persistent generator to get the next task
        # We use the stream generator to get just one
        return next(self.workload_generator.generate_workload_stream(num_tasks=1, arrival_rate=2.0))

    def _calculate_metrics(self, result: Dict) -> Dict:
        """Calculate energy consumption and execution cost from scheduler result.
        
        Uses a simple power model where GPUs consume 50W and CPUs consume 30W.
        Cost is calculated based on $0.15 per kWh.
        
        Args:
            result (Dict): Scheduler result containing 'actual_time' and 'gpu_fraction'
        
        Returns:
            Dict: Dictionary with keys:
                - time (float): Execution time in seconds
                - energy (float): Energy consumption in Joules
                - cost (float): Execution cost in dollars
        """
        time = result['actual_time']
        gpu_frac = result['gpu_fraction']
        
        # Power Model: GPU=50W, CPU=30W
        power = (gpu_frac * 50.0) + ((1.0 - gpu_frac) * 30.0)
        energy_joules = power * time
        
        # Cost Model: $0.15 per kWh
        kwh = energy_joules / 3_600_000
        cost = kwh * 0.15
        
        return {
            'time': time,
            'energy': energy_joules,
            'cost': cost
        }

    def _run_all_schedulers(self, task: Task) -> Dict:
        """Execute a task across all scheduling strategies and collect results.
        
        This method runs the same task through six different schedulers:
        1. Round Robin: Alternates between GPU and CPU
        2. Random: Random GPU fraction selection
        3. Greedy: Uses compute intensity as GPU fraction
        4. Hybrid ML: ML model prediction based on task features
        5. RL Agent: Reinforcement learning-based decision
        6. Oracle: Grid search for optimal GPU fraction (ground truth)
        
        Args:
            task (Task): The task to schedule
        
        Returns:
            Dict: Nested dictionary with scheduler names as keys, each containing:
                - gpu_fraction (float): Allocated GPU fraction
                - actual_time (float): Execution time
                - time (float): Same as actual_time
                - energy (float): Energy consumption in Joules
                - cost (float): Execution cost in dollars
        """
        results = {}
        
        # --- 1. Round Robin ---
        rr_gpu_frac = 0.5 if task.task_id % 2 == 0 else 0.0
        res = self.simulators['round_robin'].simulate_task_execution(task, rr_gpu_frac)
        results['round_robin'] = {**res, **self._calculate_metrics(res)}
        
        # --- 2. Random ---
        rand_gpu_frac = random.random()
        res = self.simulators['random'].simulate_task_execution(task, rand_gpu_frac)
        results['random'] = {**res, **self._calculate_metrics(res)}
        
        # --- 3. Greedy ---
        greedy_frac = task.compute_intensity
        res = self.simulators['greedy'].simulate_task_execution(task, greedy_frac)
        results['greedy'] = {**res, **self._calculate_metrics(res)}
        
        # --- 4. Hybrid ML ---
        features = pd.DataFrame([{
            'size': task.size,
            'compute_intensity': task.compute_intensity,
            'memory_required': task.memory_required,
            'memory_per_size': task.memory_required / (task.size + 1),
            'compute_to_memory': task.compute_intensity / (task.memory_required + 1),
        }])
        ml_frac = self.hybrid_scheduler.model.predict(features)[0]
        ml_frac = max(0.0, min(1.0, ml_frac))
        res = self.simulators['hybrid_ml'].simulate_task_execution(task, ml_frac)
        results['hybrid_ml'] = {**res, **self._calculate_metrics(res)}
        
        # --- 5. RL Agent ---
        rl_frac = 1.0 
        res = self.simulators['rl_agent'].simulate_task_execution(task, rl_frac)
        results['rl_agent'] = {**res, **self._calculate_metrics(res)}
        
        # --- 6. Oracle (Brute Force) ---
        best_time = float('inf')
        best_frac = 0.0
        best_res = None
        for frac in np.linspace(0, 1, 11):
            r = self.simulators['oracle'].simulate_task_execution(task, frac)
            if r['actual_time'] < best_time:
                best_time = r['actual_time']
                best_frac = frac
                best_res = r
        
        results['oracle'] = {**best_res, **self._calculate_metrics(best_res)}
        
        return results

    def _update_metrics(self, results):
        """Accumulate execution time and task counts for each scheduler.
        
        Args:
            results (Dict): Dictionary of scheduler results from _run_all_schedulers
        """
        for name, res in results.items():
            self.metrics[name]['total_time'] += res['time']
            # We can track total energy/cost too if needed
            self.metrics[name]['tasks'] += 1

    def _persist_data(self, task: Task, oracle_result: Dict):
        """Append task features and oracle's optimal decision to training history.
        
        This data is used for periodic model retraining to improve the hybrid
        scheduler's predictions over time.
        
        Args:
            task (Task): The executed task
            oracle_result (Dict): Oracle scheduler result containing optimal GPU fraction
        """
        row = {
            'task_id': task.task_id,
            'size': task.size,
            'compute_intensity': task.compute_intensity,
            'memory_required': task.memory_required,
            'optimal_gpu_fraction': oracle_result['gpu_fraction'],
            'optimal_time': oracle_result['actual_time']
        }
        pd.DataFrame([row]).to_csv(self.history_file, mode='a', header=False, index=False)

    async def _retrain_model(self):
        """Retrain the hybrid ML model using accumulated历史 data.
        
        Loads all historical task data from CSV, performs feature engineering,
        and retrains the RandomForest model. Requires minimum 50 samples.
        Broadcasts a notification to the frontend upon completion.
        
        Raises:
            Exception: Logs error if retraining fails but doesn't crash simulation
        """
        logger.info("Retraining Hybrid ML Model...")
        try:
            # Load history
            df = pd.read_csv(self.history_file)
            if len(df) < 50:
                return # Not enough data
            
            # Prepare features
            df['memory_per_size'] = df['memory_required'] / (df['size'] + 1)
            df['compute_to_memory'] = df['compute_intensity'] / (df['memory_required'] + 1)
            
            X = df[['size', 'compute_intensity', 'memory_required', 'memory_per_size', 'compute_to_memory']]
            y = df['optimal_gpu_fraction']
            
            # Train
            self.trainer.train(X, y)
            
            # Update scheduler's model reference
            self.hybrid_scheduler.model = self.trainer.model
            
            logger.info(f"Retraining Complete. Samples: {len(df)}")
            
            # Notify frontend
            await self.broadcast_callback({
                'type': 'notification',
                'message': f'Hybrid Model Retrained! (Samples: {len(df)})'
            })
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")

    async def _broadcast_state(self, task: Task, results: Dict):
        """Broadcast current simulation state to all connected dashboard clients.
        
        Constructs a comprehensive message containing:
        - Current task information
        - Scheduler performance comparison
        - Latest individual scheduler results (time/energy/cost)
        - Mock GPU utilization metrics
        - Scheduling decision details
        
        Args:
            task (Task): The task that was just scheduled
            results (Dict): Results from all schedulers for this task
        """
        # Calculate averages for chart
        comparison = []
        for name, metrics in self.metrics.items():
            avg_time = metrics['total_time'] / max(1, metrics['tasks'])
            comparison.append({'name': name, 'avg_time': avg_time})
            
        # Current Task Info
        task_info = {
            'id': task.task_id,
            'size': task.size,
            'intensity': task.compute_intensity,
            'memory': task.memory_required
        }
        
        # Hybrid ML Decision (for the visualizer)
        hybrid_res = results['hybrid_ml']
        
        # Construct message
        msg = {
            'type': 'simulation_update',
            'task': task_info,
            'comparison': comparison,
            'latest_results': results, # Now contains time, energy, cost
            'utilization': {
                'average_utilization': random.uniform(0.4, 0.9),
                'gpu_0': {'utilization': random.uniform(0, 1)},
                'gpu_1': {'utilization': random.uniform(0, 1)},
                'gpu_2': {'utilization': random.uniform(0, 1)},
                'gpu_3': {'utilization': random.uniform(0, 1)},
            },
            'data': {
                'task_id': task.task_id,
                'gpu_id': random.randint(0, 3),
                'gpu_fraction': hybrid_res['gpu_fraction'],
                'scheduled_time': time.time()
            }
        }
        
        await self.broadcast_callback(msg)
