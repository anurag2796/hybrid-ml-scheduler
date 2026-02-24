"""
This is the main engine that runs the simulation.
It generates tasks, runs them through all the schedulers, and sends the results to the dashboard.
"""

import asyncio
import time
import random
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from src.workload_generator import WorkloadGenerator, Task
from src.simulator import VirtualMultiGPU
from src.online_scheduler import OnlineScheduler
from src.offline_trainer import OfflineTrainer
from src.dqn_scheduler import DQNScheduler

class ContinuousSimulation:
    # Controls the whole simulation loop.
    # Generates tasks, runs schedulers, saves data, and retrains the model.
    
    def __init__(self, broadcast_callback):
        """Sets up the simulation with all the schedulers and stuff."""
        self.broadcast_callback = broadcast_callback
        self.is_running = False
        self.is_paused = False
        
        # Load Configuration
        self.config = self._load_config()
        
        # Configuration
        self.num_gpus = self.config['hardware'].get('num_virtual_gpus', 4)
        self.retrain_interval = 50
        self.tasks_processed = 0
        self.history_file = Path("data/long_term_history.csv")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance Optimizations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.batch_buffer = [] # For training data
        self.results_buffer = [] # For full scheduler results
        self.batch_size = 1 # Write to disk every 1 task (Immediate updates)
        
        # Initialize Workload Generator (Persistent)
        self.workload_generator = WorkloadGenerator(seed=self.config['workload_generation'].get('seed', 42))
        
        # Create a persistent task stream so task_ids increment correctly
        # We generate a large number of tasks to keep the simulation running
        self.task_stream = self.workload_generator.generate_workload_stream(
            num_tasks=1000000, 
            arrival_rate=2.0
        )
        
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
        
        # --- FIX: Pre-train RL Agent to avoid high initial latency ---
        try:
            logger.info("Pre-training RL Agent to fix initial performance...")
            # Generate 1000 tasks for "Heuristic Pre-training"
            # This teaches the agent basic rules (Big Math -> GPU, Tiny script -> CPU)
            pretrain_tasks = self.workload_generator.generate_workload(num_tasks=1000)
            self.rl_scheduler.pretrain(pretrain_tasks, epochs=5)
            logger.info("RL Agent pre-training complete.")
        except Exception as e:
            logger.error(f"Failed to pre-train RL agent: {e}")
        
        # Metrics Storage
        self.metrics = {k: {'total_time': 0.0, 'tasks': 0} for k in self.simulators.keys()}

    def _load_config(self) -> Dict:
        """Loads the config from the yaml file."""
        try:
            with open("config.yaml", "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.yaml, using defaults: {e}")
            return {'hardware': {}, 'workload_generation': {}}

    def _initial_pretrain(self):
        """
        Trains the model on some dummy data just so it's not empty when we start.
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
        """
        Starts the main loop.
        Generates tasks, runs them, saves data, and updates the dashboard.
        """
        self.is_running = True
        logger.info("Simulation Engine Started")
        
        # Initialize CSV if not exists
        if not self.history_file.exists():
            pd.DataFrame(columns=[
                'task_id', 'size', 'compute_intensity', 'memory_required', 
                'optimal_gpu_fraction', 'optimal_time'
            ]).to_csv(self.history_file, index=False)
            
        loop = asyncio.get_running_loop()

        while self.is_running:
            if self.is_paused:
                await asyncio.sleep(0.5)
                continue
                
            # 1. Generate Task
            task = self._generate_task()
            
            # 2. Run on All Schedulers (Offload to ThreadPool)
            # This prevents blocking the async loop during heavy computation
            results = await loop.run_in_executor(self.executor, self._run_all_schedulers, task)
            
            # 3. Update Metrics & Broadcast
            self._update_metrics(results)
            await self._broadcast_state(task, results)
            
            # 4. Persist Data (Oracle Result) - Buffered
            self._persist_data(task, results['oracle'], results)
            
            # 5. Retrain if needed
            self.tasks_processed += 1
            if self.tasks_processed % self.retrain_interval == 0:
                # Flush buffer before retraining to ensure latest data is available
                self._flush_batch()
                await self._retrain_model()
            
            # Delay for visual pacing
            await asyncio.sleep(1.5) # Slow enough to read

    def stop(self):
        """Stop the simulation loop gracefully."""
        self.is_running = False
        self._flush_batch() # Ensure pending data is saved
        self.executor.shutdown(wait=False)

    def pause(self):
        """Pause the simulation temporarily without stopping it."""
        self.is_paused = True

    def resume(self):
        """Resume a paused simulation."""
        self.is_paused = False

    def _generate_task(self) -> Task:
        """Gets the next task from our generator."""
        try:
            # Get next task from the persistent stream
            return next(self.task_stream)
        except StopIteration:
            # If we run out of tasks, restart the stream
            logger.info("Task stream exhausted, restarting...")
            self.task_stream = self.workload_generator.generate_workload_stream(
                num_tasks=1000000, 
                arrival_rate=2.0
            )
            return next(self.task_stream)

    def _calculate_metrics(self, result: Dict) -> Dict:
        """
        Calculates energy and cost based on how long the task took and how much GPU it used.
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
        """
        Runs the task on all the different schedulers (Round Robin, Random, ML, etc.)
        so we can compare them.
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
        # Get action from RL Agent (Exploration vs Exploitation handled inside)
        rl_action_dict = self.rl_scheduler.get_action(task)
        rl_frac = rl_action_dict['gpu_fraction']
        
        res = self.simulators['rl_agent'].simulate_task_execution(task, rl_frac)
        
        # Calculate Reward based on actual execution
        # We need the metrics to calculate reward (negative cost)
        metrics = self._calculate_metrics(res)
        
        # Feedback to RL Agent (Observe and Learn)
        # We pass the reward back 
        # Reward = -(w*Time + (1-w)*Energy) (or similar logic inside scheduler)
        self.rl_scheduler.observe(
            task=task,
            action=rl_action_dict['action'],
            reward_metrics=metrics # Scheduler will calculate scalar reward
        )
        
        results['rl_agent'] = {**res, **metrics}
        
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
        """Updates the total time and task count for each scheduler."""
        for name, res in results.items():
            self.metrics[name]['total_time'] += res['time']
            # We can track total energy/cost too if needed
            self.metrics[name]['tasks'] += 1

    def _persist_data(self, task: Task, oracle_result: Dict, all_results: Dict = None):
        """
        Saves the task data and the best result (Oracle) to our buffer.
        We write to disk/DB in batches.
        """
        row = {
            'task_id': task.task_id,
            'size': task.size,
            'compute_intensity': task.compute_intensity,
            'memory_required': task.memory_required,
            'optimal_gpu_fraction': oracle_result['gpu_fraction'],
            'optimal_time': oracle_result['actual_time']
        }
        self.batch_buffer.append(row)
        
        if all_results:
            self.results_buffer.append((task, all_results))
        
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        """Writes the buffered data to the DB and CSV."""
        if not self.batch_buffer:
            return
            
        buffer_copy = self.batch_buffer.copy()
        self.batch_buffer = []
        
        # 1. Save to database (async, non-blocking)
        try:
            from backend.services import SimulationDataService
            # Create async task to save to database
            asyncio.create_task(
                SimulationDataService.save_training_data_batch(buffer_copy)
            )
            
            # Also save full scheduler results if we have them
            if self.results_buffer:
                results_copy = self.results_buffer.copy()
                self.results_buffer = []
                asyncio.create_task(
                    SimulationDataService.save_scheduler_results_batch(results_copy)
                )
                
            logger.debug(f"Queued {len(buffer_copy)} records for database save")
        except Exception as e:
            logger.error(f"Failed to queue database save: {e}")
        
        # 2. Save to CSV as backup
        try:
            df = pd.DataFrame(buffer_copy)
            df.to_csv(self.history_file, mode='a', header=False, index=False)
            logger.debug(f"Flushed {len(buffer_copy)} records to CSV backup")
        except Exception as e:
            logger.error(f"Failed to flush CSV batch: {e}")


    async def _retrain_model(self):
        """
        Retrains the ML model using the latest data.
        Uses a sliding window so it doesn't get too slow.
        """
        logger.info("Retraining Hybrid ML Model...")
        try:
            df = None
            
            # Try database first
            try:
                from backend.services import SimulationDataService
                data = await SimulationDataService.get_latest_training_data(limit=1000)
                if data:
                    df = pd.DataFrame(data)
                    logger.debug(f"Loaded {len(df)} records from database for retraining")
            except Exception as e:
                logger.warning(f"Failed to load from database: {e}, falling back to CSV")
            
            # Fallback to CSV if database fails or is empty  
            if df is None or len(df) == 0:
                if self.history_file.exists():
                    df = pd.read_csv(self.history_file)
                    logger.debug(f"Loaded {len(df)} records from CSV for retraining")
                    
                    # Keep only last 1000 samples
                    if len(df) > 1000:
                        df = df.tail(1000)
                else:
                    logger.warning("No training data available (neither database nor CSV)")
                    return
            
            if len(df) < 50:
                logger.info(f"Not enough data for retraining ({len(df)} samples, need 50)")
                return
            
            # Prepare features (if not already computed from database)
            if 'memory_per_size' not in df.columns:
                df['memory_per_size'] = df['memory_required'] / (df['size'] + 1)
            if 'compute_to_memory' not in df.columns:
                df['compute_to_memory'] = df['compute_intensity'] / (df['memory_required'] + 1)
            
            X = df[['size', 'compute_intensity', 'memory_required', 'memory_per_size', 'compute_to_memory']]
            y = df['optimal_gpu_fraction']
            
            # Train (in executor to avoid blocking)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, self.trainer.train, X, y)
            
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
        """Sends the current state to the frontend via WebSocket."""
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
