import json
import time
import threading
import random
from loguru import logger
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from src.workload_generator import WorkloadGenerator
from src.dqn_scheduler import DQNScheduler
from main import load_config

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPIC = 'scheduler-events'

def get_kafka_producer():
    """Retry logic for connecting to Kafka"""
    producer = None
    for i in range(10):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            logger.info("Connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            logger.warning(f"Kafka not available, retrying in 5s... ({i+1}/10)")
            time.sleep(5)
    return None

def run_simulation_loop():
    """Runs the simulation loop"""
    logger.info("Starting simulation loop...")
    
    # Connect to Kafka
    producer = get_kafka_producer()
    if not producer:
        logger.error("Could not connect to Kafka. Exiting.")
        return

    def kafka_callback(data):
        """Callback to publish to Kafka"""
        try:
            producer.send(KAFKA_TOPIC, value=data)
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")

    config = load_config()
    
    logger.info("Initializing RL Scheduler...")
    rl_scheduler = DQNScheduler(
        num_gpus=config['hardware']['num_virtual_gpus'],
        monitor_callback=kafka_callback
    )
    
    # Generate infinite stream of tasks
    wg = WorkloadGenerator(seed=int(time.time()))
    
    logger.info("Starting Live Scheduling...")
    
    task_id = 0
    while True:
        # Generate a few tasks
        new_tasks = wg.generate_workload(num_tasks=1, arrival_rate=0.5)
        for task in new_tasks:
            task.task_id = task_id
            task_id += 1
            
            # Simulate arrival delay
            time.sleep(random.uniform(0.5, 2.0)) 
            
            # Schedule
            rl_scheduler.randomize_resources() # Simulate dynamic environment
            rl_scheduler.schedule_task(task)
            
            # Also send resource update
            kafka_callback({
                'type': 'resources',
                'data': rl_scheduler.get_utilization()
            })

if __name__ == "__main__":
    run_simulation_loop()
