
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.online_scheduler import OnlineScheduler
from src.workload_generator import Task

class TestOnlineScheduler:
    
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.predict.return_value = [0.5] # Always predict 0.5 split
        return model
        
    @pytest.fixture
    def scheduler(self, mock_model):
        return OnlineScheduler(model=mock_model, num_gpus=2)
        
    @pytest.fixture
    def sample_task(self):
        return Task(task_id=1, size=100, compute_intensity=0.5, memory_required=10, arrival_time=0.0, duration_estimate=1.0)

    def test_initialization(self, scheduler):
        assert scheduler.num_gpus == 2
        assert len(scheduler.gpu_states) == 2
        assert scheduler.task_queue == []

    def test_submit_task(self, scheduler, sample_task):
        scheduler.submit_task(sample_task)
        assert len(scheduler.task_queue) == 1
        assert scheduler.task_queue[0] == sample_task

    def test_predict_placement(self, scheduler, sample_task):
        placement = scheduler._predict_placement(sample_task)
        assert placement['gpu_fraction'] == 0.5
        assert placement['cpu_fraction'] == 0.5
        scheduler.model.predict.assert_called_once()

    @patch('src.online_scheduler.HardwareProfiler.estimate_energy')
    def test_find_best_gpu(self, mock_energy, scheduler, sample_task):
        mock_energy.return_value = 10.0
        
        # Make GPU 0 busy, GPU 1 free
        scheduler.gpu_states[0].current_load = 0.9
        scheduler.gpu_states[1].current_load = 0.0
        
        best_gpu = scheduler._find_best_gpu(sample_task, gpu_fraction=0.5)
        
        # Should pick GPU 1 because it has lower load -> lower time -> lower cost
        assert best_gpu == 1

    @patch('src.online_scheduler.OnlineScheduler._find_best_gpu')
    @patch('src.online_scheduler.OnlineScheduler._predict_placement')
    def test_schedule_task(self, mock_predict, mock_find, scheduler, sample_task):
        mock_predict.return_value = {'gpu_fraction': 0.5, 'cpu_fraction': 0.5}
        mock_find.return_value = 0
        
        decision = scheduler.schedule_task(sample_task)
        
        assert decision['task_id'] == 1
        assert decision['gpu_id'] == 0
        assert decision['gpu_fraction'] == 0.5
        
        # Check resource update
        assert scheduler.gpu_states[0].tasks_running == 1
        assert scheduler.gpu_states[0].available_memory < 8000

    def test_process_queue_no_deps(self, scheduler):
        t1 = Task(task_id=1, size=10, compute_intensity=0.1, memory_required=1, arrival_time=0.0, duration_estimate=1.0)
        t2 = Task(task_id=2, size=10, compute_intensity=0.1, memory_required=1, arrival_time=0.0, duration_estimate=1.0)
        
        scheduler.submit_task(t1)
        scheduler.submit_task(t2)
        
        decisions = scheduler.process_queue()
        
        assert len(decisions) == 2
        assert len(scheduler.task_queue) == 0

    def test_process_queue_with_deps(self, scheduler):
        t1 = Task(task_id=1, size=10, compute_intensity=0.1, memory_required=1, arrival_time=0.0, duration_estimate=1.0)
        t2 = Task(task_id=2, size=10, compute_intensity=0.1, memory_required=1, arrival_time=0.0, duration_estimate=1.0)
        t2.dependencies = [1] # t2 depends on t1
        
        scheduler.submit_task(t2) # Submit t2 first to test ordering logic
        scheduler.submit_task(t1)
        
        decisions = scheduler.process_queue()
        
        assert len(decisions) == 2
        # t1 should be scheduled before t2
        assert decisions[0]['task_id'] == 1
        assert decisions[1]['task_id'] == 2

    def test_get_utilization(self, scheduler):
        util = scheduler.get_utilization()
        assert 'gpu_0' in util
        assert 'gpu_1' in util
        assert 'average_utilization' in util

    def test_reset_state(self, scheduler, sample_task):
        scheduler.schedule_task(sample_task)
        assert len(scheduler.scheduled_tasks) == 1
        
        scheduler.reset_state()
        
        assert len(scheduler.scheduled_tasks) == 0
        assert scheduler.gpu_states[0].tasks_running == 0
        assert scheduler.gpu_states[0].current_load == 0.0
