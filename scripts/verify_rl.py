import asyncio
import numpy as np
import pandas as pd
from loguru import logger
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.workload_generator import WorkloadGenerator
from src.dqn_scheduler import DQNScheduler
from src.simulator import VirtualMultiGPU
from src.profiler import HardwareProfiler

def calculate_reward_metrics(result):
    # Match simulation_engine logic
    time = result['actual_time']
    gpu_frac = result['gpu_fraction']
    power = (gpu_frac * 50.0) + ((1.0 - gpu_frac) * 30.0)
    energy_joules = power * time
    return {'time': time, 'energy': energy_joules}

async def run_verification():
    logger.info("Starting RL Verification Simulation...")
    
    # 1. Setup
    wg = WorkloadGenerator(seed=42)
    # Increase epsilon decay to ensure it doesn't stop exploring too fast in this short run
    # default was 0.9995. Let's make it 0.99 for faster convergence in this short test, 
    # OR slower (0.9999) if we want to be safe. 
    # Actually, for 1000 tasks, 0.999^1000 = 0.36. 0.99^1000 is tiny.
    # Let's use 0.995 to decays reasonably fast.
    scheduler = DQNScheduler(
        num_gpus=4, 
        epsilon_start=1.0, 
        epsilon_end=0.05, 
        epsilon_decay=0.995,
        batch_size=32 # Smaller batch for faster updates
    )
    simulator = VirtualMultiGPU(num_gpus=4)
    
    tasks = wg.generate_workload(num_tasks=1000)
    
    regrets = []
    rewards = []
    
    print(f"{'Task':<6} | {'Oracle':<8} | {'RL Agent':<8} | {'Regret':<8} | {'Epsilon':<8} | {'Action':<6}")
    print("-" * 65)

    window_size = 50
    
    for i, task in enumerate(tasks):
        # 2. Oracle (Best Possible)
        best_time = float('inf')
        for frac in np.linspace(0, 1, 11):
            res = simulator.simulate_task_execution(task, frac)
            if res['actual_time'] < best_time:
                best_time = res['actual_time']
                
        # 3. RL Agent
        action_dict = scheduler.get_action(task)
        rl_res = simulator.simulate_task_execution(task, action_dict['gpu_fraction'])
        
        # 4. Feedback
        metrics = calculate_reward_metrics(rl_res)
        scheduler.observe(task, action_dict['action'], metrics)
        
        # 5. Track stats
        regret = rl_res['actual_time'] - best_time
        regrets.append(regret)
        
        # Print every 50
        if (i + 1) % 50 == 0:
            avg_regret = np.mean(regrets[-50:])
            print(f"{i+1:<6} | {best_time:<8.4f} | {rl_res['actual_time']:<8.4f} | {avg_regret:<8.4f} | {scheduler.epsilon:<8.4f} | {action_dict['action']:<6}")
            
    avg_total_regret = np.mean(regrets)
    logger.info(f"Final Average Regret: {avg_total_regret:.4f}")
    
    # Check if it learned (compare first 100 vs last 100)
    first_100 = np.mean(regrets[:100])
    last_100 = np.mean(regrets[-100:])
    logger.info(f"Regret Improvement: {first_100:.4f} -> {last_100:.4f}")
    
    if last_100 < first_100:
        logger.success("✅ RL Agent is learning! Regret decreased.")
    else:
        logger.warning("⚠️ RL Agent did not improve. Tuning might be needed.")

if __name__ == "__main__":
    asyncio.run(run_verification())
