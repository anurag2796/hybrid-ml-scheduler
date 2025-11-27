"""
Pipeline Execution Phases
Extracted from main.py for modularity.
"""

from pathlib import Path
from loguru import logger
import pandas as pd
import time
import json

from src.profiler import HardwareProfiler
from src.workload_generator import WorkloadGenerator
from src.offline_trainer import OfflineTrainer
from src.online_scheduler import OnlineScheduler
from src.dqn_scheduler import DQNScheduler
from src.simulator import VirtualMultiGPU
from src.visualization import plot_comparison, plot_workload_characteristics


def run_profiling_phase(config: dict):
    """Phase 1: Hardware Profiling"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: HARDWARE PROFILING")
    logger.info("="*80)
    
    profiler = HardwareProfiler(device_type=config['hardware']['device'])
    profile = profiler.profile_range(sizes=config['profiling']['matrix_sizes'])
    
    if config['profiling']['save_profiles']:
        profiler.save_profile(config['profiling']['profile_output'])
    
    return profiler, profile


def run_data_generation_phase(config: dict):
    """Phase 2: Generate Workload Data"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: WORKLOAD DATA GENERATION")
    logger.info("="*80)
    
    wg = WorkloadGenerator(seed=config['workload_generation']['seed'])
    tasks = wg.generate_workload(
        num_tasks=config['workload_generation']['num_tasks'],
        task_size_range=tuple(config['workload_generation']['task_size_range']),
        compute_intensity_range=tuple(config['workload_generation']['compute_intensity_range']),
        memory_range=tuple(config['workload_generation']['memory_range']),
        arrival_rate=config['workload_generation']['arrival_rate']
    )
    
    # Save workload
    output_dir = Path("data/workload_traces")
    output_dir.mkdir(parents=True, exist_ok=True)
    wg.save_workload(str(output_dir / "workload_train.csv"))
    
    stats = wg.get_statistics()
    logger.info(f"Workload stats: {stats}")
    
    # Visualize workload
    plot_workload_characteristics(tasks, output_dir=config['output']['plots_dir'])
    
    return wg, tasks


def run_offline_training_phase(config: dict, wg: WorkloadGenerator):
    """Phase 3: Offline ML Model Training"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: OFFLINE ML MODEL TRAINING")
    logger.info("="*80)
    
    trainer = OfflineTrainer(
        model_type=config['ml_models']['model_type'],
        **config['ml_models'][config['ml_models']['model_type']]
    )
    
    # Run training pipeline
    model_output = Path("models")
    model_output.mkdir(parents=True, exist_ok=True)
    
    results = trainer.run_full_pipeline(
        wg,
        model_output_path=str(model_output / "scheduler_model.pkl")
    )
    
    logger.info(f"Training complete. Model saved.")
    logger.info(f"Feature importances: {results['feature_importances']}")
    
    return trainer, results


def run_online_scheduling_phase(config: dict, trainer: OfflineTrainer):
    """Phase 4: Online Scheduling Simulation"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: ONLINE SCHEDULING SIMULATION")
    logger.info("="*80)
    
    # Kafka Setup
    producer = None
    try:
        from kafka import KafkaProducer
        
        producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        logger.info("Kafka Producer initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Kafka Producer: {e}")

    def kafka_callback(data):
        if producer:
            try:
                # Add timestamp if missing
                if 'timestamp' not in data:
                    data['timestamp'] = time.time()
                
                producer.send('scheduler_events', value=data)
                # Small delay to simulate real-time execution for visualization
                time.sleep(0.1) 
            except Exception as e:
                logger.error(f"Failed to send to Kafka: {e}")

    # Create scheduler with trained model
    scheduler = OnlineScheduler(
        model=trainer.model,
        num_gpus=config['hardware']['num_virtual_gpus'],
        monitor_callback=kafka_callback
    )
    
    # Generate test workload
    test_wg = WorkloadGenerator(seed=42)
    test_tasks = test_wg.generate_workload(
        num_tasks=config['workload_generation']['simulation_tasks'],
        arrival_rate=config['workload_generation']['arrival_rate']
    )
    
    # Submit and schedule tasks
    for task in test_tasks:
        scheduler.submit_task(task)
    
    decisions = scheduler.process_queue()
    
    if producer:
        producer.flush()
        producer.close()
    
    logger.info(f"Scheduled {len(decisions)} tasks")
    logger.info(f"Utilization: {scheduler.get_utilization()}")
    
    return scheduler, decisions


def run_rl_training_phase(config: dict):
    """Phase 4.5: RL Scheduler Training"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 4.5: RL SCHEDULER TRAINING")
    logger.info("="*80)
    
    # Initialize RL Scheduler
    rl_scheduler = DQNScheduler(
        num_gpus=config['hardware']['num_virtual_gpus'],
        energy_weight=0.0 # Pure Performance
    )
    
    # Generate training workload for RL
    # Needs many tasks to learn
    train_wg = WorkloadGenerator(seed=101)
    train_tasks = train_wg.generate_workload(
        num_tasks=config['workload_generation']['rl_training_tasks'],
        arrival_rate=config['workload_generation']['arrival_rate'],
        task_size_range=tuple(config['workload_generation']['task_size_range']),
        compute_intensity_range=tuple(config['workload_generation']['compute_intensity_range']),
        memory_range=tuple(config['workload_generation']['memory_range'])
    )
    
    # Pre-train (Supervised Learning)
    logger.info("Generating pre-training data...")
    pretrain_wg = WorkloadGenerator(seed=42)
    pretrain_tasks = pretrain_wg.generate_workload(
        num_tasks=5000,
        arrival_rate=config['workload_generation']['arrival_rate'],
        task_size_range=tuple(config['workload_generation']['task_size_range']),
        compute_intensity_range=tuple(config['workload_generation']['compute_intensity_range']),
        memory_range=tuple(config['workload_generation']['memory_range'])
    )
    rl_scheduler.pretrain(pretrain_tasks, epochs=50)
    
    # Lower epsilon to preserve pre-trained knowledge
    rl_scheduler.epsilon = 0.1
    logger.info(f"Epsilon set to {rl_scheduler.epsilon} for fine-tuning")

    # Train (Online Learning)
    logger.info("Training RL agent (Online)...")
    for task in train_tasks:
        rl_scheduler.randomize_resources()
        rl_scheduler.schedule_task(task)
        
    logger.info(f"RL Training complete. Final Epsilon: {rl_scheduler.epsilon:.4f}")
    
    # Save model
    model_output = Path(config['output']['model_dir'])
    rl_scheduler.save_model(str(model_output / "rl_dqn_model.pth"))
    
    return rl_scheduler


def run_evaluation_phase(config: dict, scheduler, decisions, rl_scheduler=None):
    """Phase 5: Evaluation & Analysis"""
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: EVALUATION & ANALYSIS")
    logger.info("="*80)
    
    # Create simulator for baseline comparison
    simulator = VirtualMultiGPU(num_gpus=config['hardware']['num_virtual_gpus'])
    
    # Generate test workload for baselines
    # Generate test workload for baselines (Distribution Shift: CPU favored)
    test_wg = WorkloadGenerator(seed=99)
    test_tasks = test_wg.generate_workload(
        num_tasks=config['workload_generation']['evaluation_tasks'],
        task_size_range=(10, 100), # Small tasks (CPU favored due to transfer overhead)
        compute_intensity_range=(0.0, 0.1), # Low intensity
        memory_range=(2000, 5000) # High memory
    )
    
    # Evaluate baselines
    baselines = simulator.evaluate_baseline_schedulers(test_tasks)
    
    # Evaluate our scheduler
    our_times = []
    for i, task in enumerate(test_tasks):
        # Re-predict using our model for fair comparison on same workload
        
        # 1. Predict placement
        features = pd.DataFrame([{
            'size': task.size,
            'compute_intensity': task.compute_intensity,
            'memory_required': task.memory_required,
            'memory_per_size': task.memory_required / (task.size + 1),
            'compute_to_memory': task.compute_intensity / (task.memory_required + 1),
        }])
        gpu_fraction = scheduler.model.predict(features)[0]
        gpu_fraction = max(0.0, min(1.0, gpu_fraction))
        
        # 2. Simulate execution
        result = simulator.simulate_task_execution(task, gpu_fraction)
        our_times.append(result['actual_time'])
        
    baselines['hybrid_ml'] = {
        'makespan': sum(our_times),
        'avg_time': sum(our_times) / len(our_times),
        'max_time': max(our_times),
    }

    # Evaluate RL Scheduler
    if rl_scheduler:
        rl_times = []
        # Disable exploration for evaluation (Greedy Policy)
        rl_scheduler.epsilon = 0.0
        
        for task in test_tasks:
            pass 
            
        # Re-run properly
        for task in test_tasks:
            rl_scheduler.reset_state()
            decision = rl_scheduler.schedule_task(task)
            rl_times.append(decision['estimated_time'])
            
        baselines['rl_agent'] = {
            'makespan': sum(rl_times),
            'avg_time': sum(rl_times) / len(rl_times),
            'max_time': max(rl_times),
        }

    logger.info("Baseline Performance:")
    for strategy, metrics in baselines.items():
        logger.info(f"  {strategy}: {metrics}")
    
    # Generate Plots
    plot_comparison(baselines, output_dir=config['output']['plots_dir'])
    
    # Save results
    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(decisions)
    results_df.to_csv(str(results_dir / "scheduling_results.csv"), index=False)
    
    logger.info(f"Results saved to {results_dir}")
    
    return baselines, results_df
