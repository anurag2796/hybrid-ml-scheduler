"""
Machine Learning Models for Performance Prediction
Uses Random Forest and XGBoost for predicting execution times
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
from typing import Tuple, Dict
from loguru import logger


class PerformancePredictor:
    """Base class for performance prediction models"""

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

        logger.info(f"PerformancePredictor initialized with {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the model"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2)
        }

        logger.info(f"Model Evaluation - MAE: {mae:.4f}, R2: {r2:.4f}")
        return metrics

    def save(self, filepath: str):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Load model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


class RandomForestPredictor(PerformancePredictor):
    """Random Forest based performance predictor"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, **kwargs):
        super().__init__(model_type="random_forest")
        # FIXED: Don't pass n_jobs=-1 here, let it come from kwargs
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Fit Random Forest model

        Args:
            X: Feature matrix
            y: Target values
            test_size: Fraction for test split
        """
        self.feature_names = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # Train model
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        logger.info(f"Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")

        self.is_fitted = True

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        logger.info(f"CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        return {'train_r2': train_score, 'test_r2': test_score}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def feature_importance(self) -> Dict:
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)

        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class XGBoostPredictor(PerformancePredictor):
    """XGBoost based performance predictor"""

    def __init__(self, n_estimators: int = 100, max_depth: int = 7,
                 learning_rate: float = 0.1, **kwargs):
        super().__init__(model_type="xgboost")
        # FIXED: Remove any conflicting kwargs before passing
        kwargs.pop('n_jobs', None)  # Remove n_jobs if present
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            tree_method='hist',
            **kwargs
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Fit XGBoost model
        
        Args:
            X: Feature matrix
            y: Target values
            test_size: Fraction for test split
        """
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Train model
        logger.info("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        logger.info(f"Train R2: {train_score:.4f}, Test R2: {test_score:.4f}")
        
        self.is_fitted = True
        
        return {'train_r2': train_score, 'test_r2': test_score}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def feature_importance(self) -> Dict:
        """Get feature importances"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
