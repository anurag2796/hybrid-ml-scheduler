
import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import (
    run_profiling_phase,
    run_data_generation_phase,
    run_offline_training_phase,
    run_online_scheduling_phase,
    run_rl_training_phase,
    run_evaluation_phase
)

class TestPipeline:
    
    @pytest.fixture
    def mock_config(self):
        return {
            'hardware': {'device': 'cpu', 'num_virtual_gpus': 2},
            'profiling': {'matrix_sizes': [10], 'save_profiles': False, 'profile_output': 'dummy'},
            'workload_generation': {
                'seed': 42, 'num_tasks': 10, 'arrival_rate': 1.0,
                'task_size_range': [10, 20], 'compute_intensity_range': [0.1, 0.2], 'memory_range': [10, 20],
                'simulation_tasks': 5, 'rl_training_tasks': 5, 'evaluation_tasks': 5
            },
            'output': {'plots_dir': 'dummy_plots', 'results_dir': 'dummy_results', 'model_dir': 'dummy_models'},
            'ml_models': {'model_type': 'random_forest', 'random_forest': {'n_estimators': 2}}
        }

    @patch('src.pipeline.HardwareProfiler')
    def test_run_profiling_phase(self, mock_profiler_cls, mock_config):
        mock_profiler = mock_profiler_cls.return_value
        mock_profiler.profile_range.return_value = {}
        
        profiler, profile = run_profiling_phase(mock_config)
        
        assert profiler == mock_profiler
        assert profile == {}
        mock_profiler.profile_range.assert_called_once()

    @patch('src.pipeline.WorkloadGenerator')
    @patch('src.pipeline.plot_workload_characteristics')
    def test_run_data_generation_phase(self, mock_plot, mock_wg_cls, mock_config):
        mock_wg = mock_wg_cls.return_value
        mock_wg.generate_workload.return_value = []
        mock_wg.get_statistics.return_value = {}
        
        wg, tasks = run_data_generation_phase(mock_config)
        
        assert wg == mock_wg
        assert tasks == []
        mock_wg.generate_workload.assert_called_once()

    @patch('src.pipeline.OfflineTrainer')
    def test_run_offline_training_phase(self, mock_trainer_cls, mock_config):
        mock_trainer = mock_trainer_cls.return_value
        mock_trainer.run_full_pipeline.return_value = {'feature_importances': {}}
        mock_wg = MagicMock()
        
        trainer, results = run_offline_training_phase(mock_config, mock_wg)
        
        assert trainer == mock_trainer
        mock_trainer.run_full_pipeline.assert_called_once()

    @patch('src.pipeline.OnlineScheduler')
    @patch('src.pipeline.WorkloadGenerator')
    def test_run_online_scheduling_phase(self, mock_wg_cls, mock_scheduler_cls, mock_config):
        mock_trainer = MagicMock()
        mock_scheduler = mock_scheduler_cls.return_value
        mock_scheduler.process_queue.return_value = []
        mock_scheduler.get_utilization.return_value = {}
        
        scheduler, decisions = run_online_scheduling_phase(mock_config, mock_trainer)
        
        assert scheduler == mock_scheduler
        assert decisions == []
        mock_scheduler.process_queue.assert_called_once()

    @patch('src.pipeline.DQNScheduler')
    @patch('src.pipeline.WorkloadGenerator')
    def test_run_rl_training_phase(self, mock_wg_cls, mock_rl_cls, mock_config):
        mock_rl = mock_rl_cls.return_value
        mock_rl.epsilon = 0.1
        
        rl_scheduler = run_rl_training_phase(mock_config)
        
        assert rl_scheduler == mock_rl
        mock_rl.pretrain.assert_called_once()
        mock_rl.save_model.assert_called_once()

    @patch('src.pipeline.VirtualMultiGPU')
    @patch('src.pipeline.plot_comparison')
    def test_run_evaluation_phase(self, mock_plot, mock_sim_cls, mock_config):
        mock_sim = mock_sim_cls.return_value
        mock_sim.evaluate_baseline_schedulers.return_value = {}
        mock_sim.simulate_task_execution.return_value = {'actual_time': 1.0}
        
        mock_scheduler = MagicMock()
        mock_scheduler.model.predict.return_value = [0.5]
        
        mock_rl = MagicMock()
        mock_rl.schedule_task.return_value = {'estimated_time': 1.0}
        
        decisions = []
        
        baselines, results_df = run_evaluation_phase(mock_config, mock_scheduler, decisions, mock_rl)
        
        assert isinstance(baselines, dict)
        mock_sim.evaluate_baseline_schedulers.assert_called_once()
