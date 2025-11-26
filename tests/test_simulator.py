
import pytest
from src.simulator import VirtualMultiGPU
from src.workload_generator import Task

class TestVirtualMultiGPU:
    
    @pytest.fixture
    def simulator(self):
        return VirtualMultiGPU(num_gpus=2)
        
    @pytest.fixture
    def sample_task(self):
        return Task(
            task_id=1,
            size=1000,
            compute_intensity=0.5,
            memory_required=100,
            arrival_time=0.0,
            duration_estimate=1.0
        )

    def test_initialization(self, simulator):
        assert simulator.num_gpus == 2
        assert simulator.memory_per_gpu == 8000
        assert simulator.task_timeline == []
        assert simulator.execution_log == []

    def test_simulate_task_execution_cpu_only(self, simulator, sample_task):
        # 0% GPU -> 100% CPU
        result = simulator.simulate_task_execution(sample_task, gpu_fraction=0.0)
        
        assert result['task_id'] == 1
        assert result['gpu_fraction'] == 0.0
        assert result['actual_time'] == sample_task.duration_estimate
        assert result['speedup'] == 1.0

    def test_simulate_task_execution_gpu_only(self, simulator, sample_task):
        # 100% GPU
        # Speedup = 1 + 3 * 0.5 = 2.5
        # GPU time = 1.0 / 2.5 = 0.4
        # Transfer time = 100 / 16000 = 0.00625
        # Total = 0.4 + 0.00625 = 0.40625
        
        result = simulator.simulate_task_execution(sample_task, gpu_fraction=1.0)
        
        expected_speedup = 1.0 + (3.0 * sample_task.compute_intensity)
        expected_gpu_time = sample_task.duration_estimate / expected_speedup
        expected_transfer = sample_task.memory_required / 16000.0
        expected_total = expected_gpu_time + expected_transfer
        
        assert result['gpu_fraction'] == 1.0
        assert result['actual_time'] == pytest.approx(expected_total)

    def test_simulate_task_execution_hybrid(self, simulator, sample_task):
        # 50% GPU, 50% CPU
        gpu_frac = 0.5
        result = simulator.simulate_task_execution(sample_task, gpu_fraction=gpu_frac)
        
        expected_speedup = 1.0 + (3.0 * sample_task.compute_intensity)
        expected_gpu_time = sample_task.duration_estimate / expected_speedup
        expected_transfer = sample_task.memory_required / 16000.0
        
        gpu_part = (expected_gpu_time + expected_transfer) * gpu_frac
        cpu_part = sample_task.duration_estimate * (1.0 - gpu_frac)
        expected_total = gpu_part + cpu_part
        
        assert result['actual_time'] == pytest.approx(expected_total)

    def test_evaluate_baseline_schedulers(self, simulator, sample_task):
        workload = [sample_task, sample_task] # 2 tasks
        baselines = simulator.evaluate_baseline_schedulers(workload)
        
        assert 'round_robin' in baselines
        assert 'random' in baselines
        assert 'greedy' in baselines
        assert 'offline_optimal' in baselines
        
        for strategy in baselines:
            assert 'makespan' in baselines[strategy]
            assert 'avg_time' in baselines[strategy]
            assert 'max_time' in baselines[strategy]
