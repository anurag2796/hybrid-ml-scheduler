# Hybrid ML Scheduler: Documentation Brain

Welcome to the Documentation Brain for the **Hybrid ML Scheduler** project. This wiki serves as the definitive guide to the architecture, codebase, APIs, and workflows of our system. 

## Project Overview

The Hybrid ML Scheduler is a simulation framework for comparing different task scheduling strategies in heterogeneous computing environments (e.g., CPU + GPUs). It aims to solve the "Heterogeneous Scheduling Bottleneck" by testing Reinforcement Learning (RL) and Machine Learning (Random Forest) models against standard heuristic schedulers (Round Robin, Greedy) and a theoretical Oracle. A real-time React dashboard visualizes the cluster load, scheduler decisions, performance, energy, and costs in real-time.

## Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd hybrid_ml_scheduler

# Install dependencies
pip install -r requirements.txt
cd dashboard && npm install && cd ..

# Run Live Dashboard Mode
./run_live_dashboard.sh

# Run Heavy Scientific Offline Mode
python scripts/run_heavy_simulation.py
```

## System State
Currently, the pipeline has a fully functional continuous simulation engine (`src/simulation_engine.py`) and a real-time Vite/React dashboard. We are stabilizing the RL agent's pre-training logic to minimize early exploration penalties and building out our full database logging via PostgreSQL.

## Table of Contents

- [01 Architecture](01_architecture.md): System design, core components, and diagrams.
- [02 Code Reference](02_code_reference.md): Walkthrough of classes, physics models, and schemas.
- [03 API Reference](03_api_reference.md): WebSockets, FastAPI endpoints, and data payloads.
- [04 Configuration](04_configuration.md): `config.yaml`, scaling the simulation, and environment settings.
- [05 Data Workflow](05_data_workflow.md): How a task goes from generation to execution and metric logging.
- [06 Current Challenges](06_current_challenges.md): Technical debt, TODOs, and known bugs.
- [07 Troubleshooting](07_troubleshooting.md): Common errors and the 100-Question FAQ.
- [08 Project Journey](08_project_journey.md): Evolution of the project architecture and legacy phases.
