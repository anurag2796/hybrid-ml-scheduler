
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.offline_trainer import OfflineTrainer
from src.workload_generator import Task, WorkloadGenerator

class TestOfflineTrainer:
    
    @pytest.fixture
    def trainer(self):
        return OfflineTrainer(model_type="random_forest", n_estimators=10)
        
    @pytest.fixture
    def mock_workload_generator(self):
        wg = MagicMock(spec=WorkloadGenerator)
        tasks = [
            Task(task_id=1, size=100, compute_intensity=0.1, memory_required=10, arrival_time=0.0, duration_estimate=1.0),
            Task(task_id=2, size=200, compute_intensity=0.9, memory_required=20, arrival_time=0.0, duration_estimate=2.0)
        ]
        wg.tasks = tasks
        return wg

    def test_initialization(self, trainer):
        assert trainer.model_type == "random_forest"
        assert trainer.model_kwargs == {'n_estimators': 10}
        assert trainer.model is None

    @patch('src.offline_trainer.VirtualMultiGPU')
    def test_prepare_data(self, mock_vmgpu_cls, trainer, mock_workload_generator):
        # Mock simulator to return a fixed time
        mock_sim = mock_vmgpu_cls.return_value
        mock_sim.simulate_task_execution.return_value = {'actual_time': 1.0}
        
        df = trainer.prepare_data(mock_workload_generator)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'task_size' in df.columns
        assert 'optimal_gpu_fraction' in df.columns
        assert 'optimal_total_time' in df.columns

    @patch('src.offline_trainer.RandomForestPredictor')
    def test_create_model_rf(self, mock_rf, trainer):
        model = trainer.create_model()
        assert model == mock_rf.return_value
        assert trainer.model is not None

    def test_create_model_invalid(self):
        trainer = OfflineTrainer(model_type="invalid")
        with pytest.raises(ValueError):
            trainer.create_model()

    def test_train(self, trainer):
        # Mock model
        trainer.model = MagicMock()
        trainer.model.fit.return_value = {'score': 0.9}
        trainer.model.feature_importance.return_value = {'feature1': 0.5}
        
        X = pd.DataFrame({'f1': [1, 2], 'f2': [3, 4]})
        y = pd.Series([0, 1])
        
        results = trainer.train(X, y)
        
        assert results['score'] == 0.9
        trainer.model.fit.assert_called_once()

    @patch('src.offline_trainer.OfflineTrainer.prepare_data')
    @patch('src.offline_trainer.OfflineTrainer.train')
    @patch('src.offline_trainer.OfflineTrainer.save_model')
    @patch('src.offline_trainer.OfflineTrainer.create_model')
    def test_run_full_pipeline(self, mock_create, mock_save, mock_train, mock_prep, trainer, mock_workload_generator):
        # Setup mocks
        mock_prep.return_value = pd.DataFrame({
            'task_size': [100], 'compute_intensity': [0.5], 'memory_required': [10],
            'memory_per_size': [0.1], 'compute_to_memory': [0.05],
            'optimal_gpu_fraction': [0.5]
        })
        mock_train.return_value = {'score': 0.9}
        trainer.model = MagicMock()
        trainer.model.feature_importance.return_value = {}
        
        results = trainer.run_full_pipeline(mock_workload_generator, "dummy_path")
        
        assert results['training_results']['score'] == 0.9
        mock_prep.assert_called_once()
        mock_create.assert_called_once()
        mock_train.assert_called_once()
        mock_save.assert_called_once_with("dummy_path")
