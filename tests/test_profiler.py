
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.profiler import HardwareProfiler

class TestHardwareProfiler:
    
    @pytest.fixture
    def profiler(self):
        # Use CPU to avoid needing actual GPU for tests
        return HardwareProfiler(device_type="cpu")

    def test_initialization(self, profiler):
        assert profiler.device_type == "cpu"
        assert profiler.cpu_times == []
        assert profiler.gpu_times == []

    @patch('src.profiler.time.perf_counter')
    @patch('src.profiler.np.matmul')
    def test_benchmark_cpu(self, mock_matmul, mock_perf_counter, profiler):
        # Mock time to return 0 then 0.1 (diff 0.1)
        mock_perf_counter.side_effect = [0.0, 0.1, 0.0, 0.1, 0.0, 0.1]
        
        avg_time = profiler.benchmark_cpu(matrix_size=100, iterations=3)
        
        assert avg_time == pytest.approx(0.1)
        assert mock_matmul.call_count == 3

    def test_benchmark_gpu_no_gpu(self, profiler):
        # Initialized with CPU, so should return None
        result = profiler.benchmark_gpu(matrix_size=100)
        assert result is None

    @patch('src.profiler.HardwareProfiler.benchmark_cpu')
    @patch('src.profiler.HardwareProfiler.benchmark_gpu')
    def test_profile_range(self, mock_gpu, mock_cpu, profiler):
        mock_cpu.return_value = 1.0
        mock_gpu.return_value = 0.5
        
        sizes = [100, 200]
        model = profiler.profile_range(sizes)
        
        assert len(profiler.problem_sizes) == 2
        assert len(profiler.cpu_times) == 2
        assert len(profiler.gpu_times) == 2
        assert model['cpu_gpu_ratio'] == pytest.approx(2.0) # 1.0 / 0.5 = 2.0

    def test_get_performance_model_empty(self, profiler):
        model = profiler.get_performance_model()
        assert model['cpu_only'] is True

    def test_estimate_energy(self):
        # GPU: 50W * 2s = 100J
        assert HardwareProfiler.estimate_energy(2.0, is_gpu=True) == 100.0
        # CPU: 30W * 2s = 60J
        assert HardwareProfiler.estimate_energy(2.0, is_gpu=False) == 60.0
