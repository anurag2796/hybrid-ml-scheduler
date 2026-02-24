# 07. Troubleshooting & FAQ

This document addresses common errors and provides the 100-Question project FAQ base.

## Common Errors and Fixes

1. **`WebSocketConnectionError: ws://localhost:8000/ws`**
   * **Problem**: The Vite dashboard shows a flashing generic API Error Modal.
   * **Fix**: Ensure the FastAPI backend is running via `run_live_dashboard.sh`. Ensure no other processes are squatting on `:8000`.

2. **`ModuleNotFoundError: No module named 'torch'`**
   * **Problem**: Missing Apple Silicon/MPS compatibility or PyTorch native.
   * **Fix**: Follow up via the specific Python MPS wheel (`pip install -r requirements.txt`). Ensure python 3.10+ is standard.

3. **`Exception: RL pre-train collision during batch fetch`**
   * **Problem**: During Docker deploy, the RL DQN Experience Replay buffer triggers an under-fill logic crash.
   * **Fix**: Lower the `simulate_task_execution` batch queue buffer length dynamically in `config.yaml`.

## Selected FAQ

* **What is the state space?** Task Size, Intensity, Memory.
* **What is the action space?** CPU, GPU 0, GPU 1, GPU 2, GPU 3.
* **What is the reward function?** Negative combination of Time and Energy. Why? Because RL maximizes score, and we want to minimize execution time constraints.
* **Does it model Network Latency?** Implicitly via "Transfer Time" (`Time = Size / Bandwidth`).
* **Why use SQLite/CSV for logging?** Zero configuration setup for local developers. The long term strategy relies on Postgres.
* **What is "Arrival Rate"?** How many tasks enter the queue per second from the generator.
