"""
Hardware Performance Profiler for M4 Max
Measures CPU vs GPU performance characteristics
"""

import time
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger


class HardwareProfiler:
    """Profiles CPU and GPU performance"""
    
    def __init__(self, device_type: str = "mps"):
        self.device_type = device_type
        self.device = torch.device(device_type if torch.cuda.is_available() or torch.backends.mps.is_available() else "cpu")
        self.cpu_times = []
        self.gpu_times = []
        self.problem_sizes = []
        self.cpu_gpu_ratio = None
        
        logger.info(f"Hardware Profiler initialized on device: {self.device}")
    
    def benchmark_cpu(self, matrix_size: int, iterations: int = 3) -> float:
        """
        Benchmark CPU matrix multiplication
        
        Args:
            matrix_size: Size of square matrix (matrix_size x matrix_size)
            iterations: Number of iterations for averaging
            
        Returns:
            Average execution time in seconds
        """
        times = []
        
        for _ in range(iterations):
            A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
            
            start = time.perf_counter()
            C = np.matmul(A, B)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        logger.debug(f"CPU benchmark {matrix_size}x{matrix_size}: {avg_time:.4f}s")
        return avg_time
    
    def benchmark_gpu(self, matrix_size: int, iterations: int = 3) -> float:
        """
        Benchmark GPU matrix multiplication using MPS
        
        Args:
            matrix_size: Size of square matrix
            iterations: Number of iterations
            
        Returns:
            Average execution time or None if GPU unavailable
        """
        if self.device.type == "cpu":
            logger.warning("GPU not available, returning None")
            return None
        
        times = []
        
        try:
            for _ in range(iterations):
                A = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float32)
                B = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float32)
                
                # Synchronize before timing
                if self.device.type == "mps":
                    torch.mps.synchronize()
                else:
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                C = torch.matmul(A, B)
                
                # Synchronize after computation
                if self.device.type == "mps":
                    torch.mps.synchronize()
                else:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
        
        except Exception as e:
            logger.error(f"GPU benchmarking failed: {e}")
            return None
        
        avg_time = np.mean(times)
        logger.debug(f"GPU benchmark {matrix_size}x{matrix_size}: {avg_time:.4f}s")
        return avg_time
    
    def profile_range(self, sizes: List[int] = None) -> Dict:
        """
        Profile hardware across multiple problem sizes
        
        Args:
            sizes: List of matrix sizes to test
            
        Returns:
            Performance profile dictionary
        """
        if sizes is None:
            sizes = [256, 512, 1024, 2048]
        
        logger.info(f"Starting hardware profiling with sizes: {sizes}")
        
        for size in sizes:
            logger.info(f"Profiling size {size}x{size}...")
            
            cpu_time = self.benchmark_cpu(size)
            gpu_time = self.benchmark_gpu(size)
            
            self.problem_sizes.append(size)
            self.cpu_times.append(cpu_time)
            self.gpu_times.append(gpu_time if gpu_time else cpu_time)
        
        return self.get_performance_model()
    
    def get_performance_model(self) -> Dict:
        """
        Generate performance model from profiling data
        
        Returns:
            Performance model with key metrics
        """
        if not self.cpu_times or not self.gpu_times:
            logger.warning("No profiling data available")
            return {'cpu_only': True}
        
        # Filter out None GPU times
        valid_ratios = [
            c/g for c, g in zip(self.cpu_times, self.gpu_times) 
            if g is not None and g > 0
        ]
        
        if not valid_ratios:
            logger.warning("No valid GPU benchmarks")
            return {'cpu_only': True}
        
        # Calculate average ratio
        self.cpu_gpu_ratio = np.mean(valid_ratios)
        
        model = {
            'cpu_only': False,
            'cpu_gpu_ratio': float(self.cpu_gpu_ratio),
            'gpu_fraction': float(self.cpu_gpu_ratio / (self.cpu_gpu_ratio + 1)),
            'cpu_fraction': float(1 / (self.cpu_gpu_ratio + 1)),
            'profiled_sizes': self.problem_sizes,
            'cpu_times': [float(t) for t in self.cpu_times],
            'gpu_times': [float(t) if t else None for t in self.gpu_times]
        }
        
        logger.info(f"Performance Model: CPU/GPU Ratio = {self.cpu_gpu_ratio:.2f}")
        logger.info(f"GPU should receive {model['gpu_fraction']:.1%} of work")
        
        return model
    
    def save_profile(self, filepath: str):
        """Save performance profile to JSON"""
        profile = self.get_performance_model()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=2)
        
        logger.info(f"Profile saved to {filepath}")
    
    @staticmethod
    def load_profile(filepath: str) -> Dict:
        """Load performance profile from JSON"""
        with open(filepath, 'r') as f:
            profile = json.load(f)
        
        logger.info(f"Profile loaded from {filepath}")
    @staticmethod
    def estimate_energy(time_seconds: float, is_gpu: bool) -> float:
        """
        Estimate energy consumption in Joules
        
        Simple Power Model for M4 Max:
        - GPU Power: ~50W under load
        - CPU Power: ~30W under load
        - Idle/Overhead: ~5W
        """
        power_watts = 50.0 if is_gpu else 30.0
        return power_watts * time_seconds
