"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any


# Task Schemas
class TaskBase(BaseModel):
    """Base task schema with common fields."""
    task_id: int
    size: float
    compute_intensity: float
    memory_required: float
    duration_estimate: float
    arrival_time: float
    dependencies: List[int] = Field(default_factory=list)


class TaskCreate(TaskBase):
    """Schema for creating a new task."""
    pass


class TaskResponse(TaskBase):
    """Schema for task response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


# Scheduler Result Schemas
class SchedulerResultBase(BaseModel):
    """Base scheduler result schema."""
    task_id: int
    scheduler_name: str
    gpu_fraction: float
    cpu_fraction: float
    gpu_id: Optional[int] = None
    actual_time: float
    energy_consumption: Optional[float] = None
    execution_cost: Optional[float] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class SchedulerResultCreate(SchedulerResultBase):
    """Schema for creating scheduler result."""
    pass


class SchedulerResultResponse(SchedulerResultBase):
    """Schema for scheduler result response."""
    id: int
    executed_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Metric Schemas
class MetricBase(BaseModel):
    """Base metric schema."""
    scheduler_name: str
    total_tasks: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    total_energy: float = 0.0
    total_cost: float = 0.0
    wins: int = 0
    losses: int = 0
    window_start: datetime
    window_end: datetime


class MetricCreate(MetricBase):
    """Schema for creating metric."""
    pass


class MetricResponse(MetricBase):
    """Schema for metric response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


# Training Data Schemas
class TrainingDataBase(BaseModel):
    """Base training data schema."""
    size: float
    compute_intensity: float
    memory_required: float
    memory_per_size: float
    compute_to_memory: float
    optimal_gpu_fraction: float
    optimal_time: float


class TrainingDataCreate(TrainingDataBase):
    """Schema for creating training data."""
    pass


class TrainingDataResponse(TrainingDataBase):
    """Schema for training data response."""
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Simulation State Schemas
class SimulationStateBase(BaseModel):
    """Base simulation state schema."""
    is_running: bool = False
    is_paused: bool = False
    tasks_processed: int = 0
    last_retrain_at: Optional[datetime] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class SimulationStateResponse(SimulationStateBase):
    """Schema for simulation state response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


# WebSocket Message Schemas
class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages."""
    type: str  # 'task', 'result', 'metric', 'state'
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Dashboard Data Schemas
class SchedulerComparison(BaseModel):
    """Schema for scheduler comparison data."""
    scheduler_name: str
    avg_time: float
    total_tasks: int
    wins: int
    energy_consumption: float
    cost: float


class DashboardState(BaseModel):
    """Schema for complete dashboard state."""
    task: Optional[TaskResponse] = None
    results: List[SchedulerResultResponse]
    metrics: List[MetricResponse]
    schedulers: List[SchedulerComparison]
    simulation_state: SimulationStateResponse


# Health Check Schemas
class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    version: str
    database: str
    redis: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
