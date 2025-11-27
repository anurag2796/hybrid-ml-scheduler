"""
SQLAlchemy domain models for database tables.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Text, Index, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from datetime import datetime

from backend.core.database import Base


class Task(Base):
    """Task model representing workload tasks."""
    
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, unique=True, index=True, nullable=False)
    size = Column(Float, nullable=False)
    compute_intensity = Column(Float, nullable=False)
    memory_required = Column(Float, nullable=False)
    duration_estimate = Column(Float, nullable=False)
    arrival_time = Column(Float, nullable=False)
    dependencies = Column(JSONB, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_tasks_compute_intensity', 'compute_intensity'),
        Index('ix_tasks_size', 'size'),
        Index('ix_tasks_created_at', 'created_at'),
    )


class SchedulerResult(Base):
    """Results from scheduler executions."""
    
    __tablename__ = "scheduler_results"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey('tasks.task_id'), nullable=False, index=True)
    scheduler_name = Column(String(50), nullable=False, index=True)
    
    # Resource allocation
    gpu_fraction = Column(Float, nullable=False)
    cpu_fraction = Column(Float, nullable=False)
    gpu_id = Column(Integer)
    
    # Performance metrics
    actual_time = Column(Float, nullable=False)
    energy_consumption = Column(Float)
    execution_cost = Column(Float)
    
    # Additional metadata
    extra_metadata = Column(JSONB, default=dict)
    
    # Timestamps
    executed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('ix_scheduler_results_scheduler_time', 'scheduler_name', 'executed_at'),
        Index('ix_scheduler_results_task_scheduler', 'task_id', 'scheduler_name'),
    )


class Metric(Base):
    """Aggregate metrics for schedulers over time."""
    
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    scheduler_name = Column(String(50), nullable=False, index=True)
    
    # Aggregate stats
    total_tasks = Column(Integer, default=0)
    total_time = Column(Float, default=0.0)
    avg_time = Column(Float, default=0.0)
    min_time = Column(Float)
    max_time = Column(Float)
    
    # Energy and cost
    total_energy = Column(Float, default=0.0)
    total_cost = Column(Float, default=0.0)
    
    # Win/loss tracking
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    
    # Time window
    window_start = Column(DateTime(timezone=True), nullable=False)
    window_end = Column(DateTime(timezone=True), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_metrics_scheduler_window', 'scheduler_name', 'window_start', 'window_end'),
    )


class TrainingData(Base):
    """Historical training data for ML models."""
    
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Task features
    size = Column(Float, nullable=False)
    compute_intensity = Column(Float, nullable=False)
    memory_required = Column(Float, nullable=False)
    memory_per_size = Column(Float, nullable=False)
    compute_to_memory = Column(Float, nullable=False)
    
    # Optimal solution (from Oracle)
    optimal_gpu_fraction = Column(Float, nullable=False)
    optimal_time = Column(Float, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)



class SimulationState(Base):
    """Current simulation state for persistence."""
    
    __tablename__ = "simulation_state"
    
    id = Column(Integer, primary_key=True)
    is_running = Column(Boolean, default=False)
    is_paused = Column(Boolean, default=False)
    tasks_processed = Column(Integer, default=0)
    last_retrain_at = Column(DateTime(timezone=True))
    
    # Configuration snapshot
    config = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
