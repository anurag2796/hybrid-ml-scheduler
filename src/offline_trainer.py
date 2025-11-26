"""
Offline Training Pipeline
Processes workload data and trains ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from loguru import logger
from .simulator import VirtualMultiGPU


from .ml_models import RandomForestPredictor, XGBoostPredictor
from .workload_generator import WorkloadGenerator


class OfflineTrainer:
    """Orchestrates offline training pipeline"""
    
    def __init__(self, model_type: str = "random_forest", **model_kwargs):
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.model = None
        self.training_data = None
        self.features = None
        self.target = None
        
        logger.info(f"OfflineTrainer initialized with {model_type}")
    
    def prepare_data(self, workload_generator: WorkloadGenerator) -> pd.DataFrame:
        """
        Prepare training data from workload
        
        Args:
            workload_generator: WorkloadGenerator instance with tasks
            
        Returns:
            DataFrame with features for training
        """
        logger.info("Preparing training data...")
        
        tasks = workload_generator.tasks
        
        # Extract features from tasks
        features_list = []
        for task in tasks:
            features_list.append({
                'task_size': task.size,
                'compute_intensity': task.compute_intensity,
                'memory_required': task.memory_required,
                'duration_estimate': task.duration_estimate,
                # Derived features
                'memory_per_size': task.memory_required / (task.size + 1),
                'compute_to_memory': task.compute_intensity / (task.memory_required + 1),
            })
        
        df = pd.DataFrame(features_list)
        
        # Create target: optimal resource allocation
        # Simple heuristic: GPU good for high compute_intensity tasks
        simulator = VirtualMultiGPU(num_gpus=1)
        optimal_fractions = []
        optimal_times = []

        for i, task in enumerate(tasks):
            best_frac = 0.0
            best_time = float('inf')
            # Try gpu_fraction from 0.0 up to 1.0 (inclusive, fine steps)
            for frac in np.linspace(0, 1, 11):
                sim_result = simulator.simulate_task_execution(task, gpu_fraction=frac)
                total_time = sim_result['actual_time']
                if total_time < best_time:
                    best_time = total_time
                    best_frac = frac
            optimal_fractions.append(best_frac)
            optimal_times.append(best_time)

        df['optimal_gpu_fraction'] = optimal_fractions
        df['optimal_total_time'] = optimal_times

        self.training_data = df
        logger.info(f"Prepared {len(df)} training samples")
        
        return df
    
    def create_model(self):
        """Create model instance based on type"""
        if self.model_type == "random_forest":
            self.model = RandomForestPredictor(**self.model_kwargs)
        elif self.model_type == "xgboost":
            self.model = XGBoostPredictor(**self.model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Created {self.model_type} model")
        return self.model
    
    def train(self, 
              X: pd.DataFrame, 
              y: pd.Series,
              test_size: float = 0.2) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target values
            test_size: Test set fraction
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.create_model()
        
        logger.info(f"Training model on {len(X)} samples...")
        
        results = self.model.fit(X, y, test_size=test_size)
        
        # Get feature importances
        importances = self.model.feature_importance()
        logger.info(f"Top 3 important features: {dict(list(importances.items())[:3])}")
        
        return results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        metrics = self.model.evaluate(X_test, y_test)
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def run_full_pipeline(self, 
                         workload_generator: WorkloadGenerator,
                         model_output_path: str = "models/scheduler_model.pkl") -> Dict:
        """
        Run complete offline training pipeline
        
        Args:
            workload_generator: WorkloadGenerator with tasks
            model_output_path: Where to save the model
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("="*60)
        logger.info("Starting Offline Training Pipeline")
        logger.info("="*60)
        
        # Step 1: Prepare data
        df = self.prepare_data(workload_generator)
        
        # Step 2: Extract features and target
        feature_cols = ['task_size', 'compute_intensity', 'memory_required', 
                       'memory_per_size', 'compute_to_memory']
        X = df[feature_cols]
        y = df['optimal_gpu_fraction']
        
        # Step 3: Create and train model
        self.create_model()
        train_results = self.train(X, y, test_size=0.2)
        
        # Step 4: Save model
        self.save_model(model_output_path)
        
        # Step 5: Generate feature importances
        importances = self.model.feature_importance()
        
        pipeline_results = {
            'training_results': train_results,
            'feature_importances': importances,
            'model_path': model_output_path,
            'data_stats': {
                'num_samples': len(X),
                'num_features': len(feature_cols),
            }
        }
        
        logger.info("="*60)
        logger.info("Offline Training Pipeline Complete")
        logger.info("="*60)
        
        return pipeline_results

    def empirical_best_split(task, simulator, split_steps=11):
        """
        Find the gpu_fraction in [0, 1] that minimizes simulated runtime for the given task.
        Args:
            task: Task object (from WorkloadGenerator)
            simulator: VirtualMultiGPU or equivalent (should expose simulate_task_execution)
            split_steps: Granularity (default: test 0.0, 0.1, ..., 1.0)
        Returns:
            (best_fraction, best_time)
        """
        best_fraction = 0.0
        best_time = float('inf')
        # Try gpu_fraction from 0.0 up to 1.0 (inclusive)
        for frac in np.linspace(0, 1, split_steps):
            sim_result = simulator.simulate_task_execution(task, gpu_fraction=frac)
            total_time = sim_result['actual_time']
            if total_time < best_time:
                best_time = total_time
                best_fraction = frac
        return best_fraction, best_time

