# 04. Configuration

This document outlines the system configuration files, deployment settings, and customization mechanisms.

## `config.yaml` Settings

The root-level `config.yaml` file drives the `ContinuousSimulation` and all offline batch processes.

```yaml
hardware:
  device: "mps"  # Backend hardware profiling, Metal Performance Shaders
  num_virtual_gpus: 4 # Number of simulated GPUs
  
simulation:
  retrain_interval: 50 # Retrain ML models every 50 tasks

workload_generation:
  seed: 42
  arrival_rate: 100 # Tasks per second
  task_size_range: [100, 10000]
  compute_intensity_range: [0.0, 1.0]

ml_models:
  model_type: "random_forest"
  random_forest:
    n_estimators: 100
```

## Dashboard Env Variables (`dashboard/.env`)
The React dashboard can be configured using standard Vite environment variables, controlling variables like the absolute backend URL, websocket retries, and UI feature toggles.

## Docker & Server Deployment
The system can be containerized using `docker-compose.yml`. A typical setup runs the FastAPI `uvicorn` backend on port `8000`, the PostgreSQL DB on `5432`, and the React frontend on `5173`. 
The `run_live_dashboard.sh` initiates this local sequence directly for development.
