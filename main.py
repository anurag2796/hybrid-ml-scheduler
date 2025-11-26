"""
Main Entry Point - Complete project execution pipeline
"""

import yaml
from loguru import logger

from src.pipeline import (
    run_profiling_phase,
    run_data_generation_phase,
    run_offline_training_phase,
    run_online_scheduling_phase,
    run_rl_training_phase,
    run_evaluation_phase
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Main execution pipeline"""
    # Setup logging
    logger.add("logs/scheduler.log", rotation="500 MB")
    
    logger.info("="*80)
    logger.info("HYBRID OFFLINE-ONLINE ML SCHEDULER FOR PARALLEL COMPUTING")
    logger.info("="*80)
    
    # Load configuration
    config = load_config()
    logger.info(f"Configuration loaded")
    
    # Phase 1: Profiling
    profiler, profile = run_profiling_phase(config)
    
    # Phase 2: Data Generation
    wg, tasks = run_data_generation_phase(config)
    
    # Phase 3: Offline Training
    trainer, train_results = run_offline_training_phase(config, wg)
    
    # Phase 4: Online Scheduling
    scheduler, decisions = run_online_scheduling_phase(config, trainer)
    
    # Phase 4.5: RL Training
    rl_scheduler = run_rl_training_phase(config)
    
    # Phase 5: Evaluation
    baselines, results_df = run_evaluation_phase(config, scheduler, decisions, rl_scheduler)
    
    logger.info("\n" + "="*80)
    logger.info("PROJECT COMPLETE")
    logger.info("="*80)
    logger.info("Check data/results/ for all output files")


if __name__ == "__main__":
    main()
