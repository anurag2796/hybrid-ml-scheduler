
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
import yaml
import pandas as pd
import asyncio
import numpy as np

from src.pipeline import (
    run_profiling_phase,
    run_data_generation_phase,
    run_offline_training_phase,
    run_online_scheduling_phase,
    run_rl_training_phase
)
from src.simulator import VirtualMultiGPU
from src.workload_generator import WorkloadGenerator
from src.visualization import plot_comparison, plot_cost_analysis, plot_latency_distribution
from src.reporting import generate_enhanced_report
from backend.services.simulation_data_service import SimulationDataService
from backend.core.database import init_db, close_db

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

async def main():
    logger.add("logs/heavy_simulation.log", rotation="500 MB")
    
    logger.info("="*80)
    logger.info("HEAVY WORKLOAD SIMULATION (10,000 TASKS)")
    logger.info("="*80)
    
    # 0. Initialize Database
    logger.info("Initializing Database...")
    await init_db()
    
    config_path = "config_heavy.yaml"
    if not Path(config_path).exists():
        logger.error(f"{config_path} not found!")
        return
        
    config = load_config(config_path)
    
    # Phase 1: Profiling (Standard)
    run_profiling_phase(config)
    
    # Phase 2: Data Gen (Standard)
    wg, _ = run_data_generation_phase(config)
    
    # Phase 3: Offline Training (Standard)
    trainer, _ = run_offline_training_phase(config, wg)
    
    # Phase 4.5: RL Training (Standard)
    rl_scheduler = run_rl_training_phase(config, trainer)
    
    # Phase 5: Heavy Evaluation & DB Population
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: HEAVY WORKLOAD EVALUATION & DB POPULATION")
    logger.info("="*80)
    
    num_eval_tasks = config['workload_generation']['evaluation_tasks']
    logger.info(f"Generating {num_eval_tasks} evaluation tasks...")
    
    test_wg = WorkloadGenerator(seed=999)
    test_tasks = test_wg.generate_workload(
        num_tasks=num_eval_tasks,
        task_size_range=tuple(config['workload_generation']['task_size_range']),
        compute_intensity_range=tuple(config['workload_generation']['compute_intensity_range']),
        memory_range=tuple(config['workload_generation']['memory_range'])
    )
    
    # Save training data to DB (Simulating that we saw this 'historical' data)
    # We'll batch save the task definitions first
    # Actually, SimulationDataService.save_scheduler_results handles task creation if missing.
    
    simulator = VirtualMultiGPU(num_gpus=config['hardware']['num_virtual_gpus'])
    
    baselines = {}
    latencies = {} # For CDF/Boxplot
    
    # 1. Run Baselines
    strategies = config['evaluation']['baseline_strategies']
    for strategy in strategies:
        logger.info(f"Running baseline: {strategy}...")
        results = simulator.simulate_workload(test_tasks, strategy)
        
        # Calculate Metrics
        times = [r['actual_time'] for r in results]
        costs = [r['cost'] for r in results]
        total_time = sum(times)
        total_cost = sum(costs)
        
        baselines[strategy] = {
            'makespan': total_time,
            'avg_time': np.mean(times),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99),
            'throughput': len(test_tasks) / total_time * config['hardware']['num_virtual_gpus'], # approx
            'total_cost': total_cost,
            'cost_efficiency': len(test_tasks) / (total_cost + 1e-6)
        }
        latencies[strategy] = times
        
        # Save to DB (Background or Batch)
        # To avoid slowing down too much, we'll save in chunks
        logger.info(f"Saving {len(results)} results for {strategy} to DB...")
        batch_data = []
        for i, task in enumerate(test_tasks):
            batch_data.append((task, {strategy: results[i]}))
            
            if len(batch_data) >= 500:
                await SimulationDataService.save_scheduler_results_batch(batch_data)
                batch_data = []
        
        if batch_data:
            await SimulationDataService.save_scheduler_results_batch(batch_data)

    # 2. Run Hybrid ML
    logger.info("Running Hybrid ML Scheduler...")
    our_results = []
    our_times = []
    our_costs = []
    
    for task in test_tasks:
         # Predict
        features = pd.DataFrame([{
            'size': task.size,
            'compute_intensity': task.compute_intensity,
            'memory_required': task.memory_required,
            'memory_per_size': task.memory_required / (task.size + 1),
            'compute_to_memory': task.compute_intensity / (task.memory_required + 1),
        }])
        gpu_fraction = trainer.model.predict(features)[0]
        gpu_fraction = max(0.0, min(1.0, gpu_fraction))
        
        # Simulate
        result = simulator.simulate_task_execution(task, gpu_fraction)
        our_results.append(result)
        our_times.append(result['actual_time'])
        our_costs.append(result['cost'])

    baselines['hybrid_ml'] = {
        'makespan': sum(our_times),
        'avg_time': np.mean(our_times),
        'p95_time': np.percentile(our_times, 95),
        'p99_time': np.percentile(our_times, 99),
        'throughput': len(test_tasks) / sum(our_times) * config['hardware']['num_virtual_gpus'],
        'total_cost': sum(our_costs),
        'cost_efficiency': len(test_tasks) / (sum(our_costs) + 1e-6)
    }
    latencies['hybrid_ml'] = our_times
    
    # Save Hybrid ML Results to DB
    logger.info("Saving Hybrid ML results to DB...")
    batch_data = []
    for i, task in enumerate(test_tasks):
        batch_data.append((task, {'hybrid_ml': our_results[i]}))
        if len(batch_data) >= 500:
            await SimulationDataService.save_scheduler_results_batch(batch_data)
            batch_data = []
    if batch_data:
        await SimulationDataService.save_scheduler_results_batch(batch_data)

    # 3. Run RL Agent
    logger.info("Running RL Agent (Evaluated)...")
    rl_scheduler.epsilon = 0.0 # Greedy evaluation
    rl_results = []
    rl_times = []
    rl_costs = []
    
    for task in test_tasks:
        rl_scheduler.reset_state()
        # Note: In real sim we'd loop, but here getting decision is enough for 'simulated' execution
        decision = rl_scheduler.schedule_task(task)
        
        # The RL scheduler returns a decision, we need to run it in the simulator to get actuals
        # The decision dict contains 'action' (0=CPU, 1=GPU, etc if discrete, or fraction)
        # RL uses discrete actions usually mapped to gpu_fraction
        
        # Map action to fraction
        # Action 0: CPU (0.0), Action 1: GPU (1.0)
        action = decision['action']
        gpu_fraction = decision['gpu_fraction'] 
        
        result = simulator.simulate_task_execution(task, gpu_fraction)
        rl_results.append(result)
        rl_times.append(result['actual_time'])
        rl_costs.append(result['cost'])
        
    baselines['rl_agent'] = {
        'makespan': sum(rl_times),
        'avg_time': np.mean(rl_times),
        'p95_time': np.percentile(rl_times, 95),
        'p99_time': np.percentile(rl_times, 99),
        'throughput': len(test_tasks) / sum(rl_times) * config['hardware']['num_virtual_gpus'],
        'total_cost': sum(rl_costs),
        'cost_efficiency': len(test_tasks) / (sum(rl_costs) + 1e-6)
    }
    latencies['rl_agent'] = rl_times

    # Save RL Results to DB
    logger.info("Saving RL Agent results to DB...")
    batch_data = []
    for i, task in enumerate(test_tasks):
        batch_data.append((task, {'rl_agent': rl_results[i]}))
        if len(batch_data) >= 500:
            await SimulationDataService.save_scheduler_results_batch(batch_data)
            batch_data = []
    if batch_data:
        await SimulationDataService.save_scheduler_results_batch(batch_data)

    # Generate Plots
    logger.info("Generating plots...")
    plot_comparison(baselines, output_dir=config['output']['plots_dir'])
    plot_cost_analysis(baselines, output_dir=config['output']['plots_dir'])
    plot_latency_distribution(latencies, output_dir=config['output']['plots_dir'])
    
    # Generate Report
    logger.info("Generating PDF Report...")
    report_path = "data/results/Heavy_Workload_Report.pdf"
    generate_enhanced_report(report_path, baselines, config['output']['plots_dir'])
    
    logger.info(f"SUCCESS! Report generated at {report_path}")
    
    await close_db()

if __name__ == "__main__":
    asyncio.run(main())
