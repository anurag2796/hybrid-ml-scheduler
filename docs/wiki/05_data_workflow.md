# 05. Data Workflow

This document illustrates how a single Task moves through the Hybrid ML Scheduler ecosystem.

1. **Task Ingestion**
   * The `WorkloadGenerator` generates a unique `Task` object.
   * `Task` is parameterized by `size` (MB), `compute_intensity` (FLOPs/Bytes), and `memory_required`.

2. **Parallel Scheduling**
   * The `ContinuousSimulation` loops the `Task` through all 6 Virtual Schedulers (Round Robin, Random, Greedy, Hybrid ML, RL Agent, Oracle).
   * **Hybrid ML Inference**: The Task is passed to `sklearn.RandomForestRegressor.predict([Size, Intensity, Memory])` to fetch an optimal GPU fraction `[0.0, 1.0]`.
   * **RL Agent Inference**: The Task state tensor `[Size, Intensity, Memory]` passes through a PyTorch DQN. An Epsilon-Greedy calculation picks a discrete action (GPU0, GPU1...) and outputs an action fraction.

3. **Physics Verification & Execution**
   * The `VirtualMultiGPU.simulate_task_execution()` determines the real simulated time execution latency.
   * Total actual Time latency = `Task Size / (Transfer Bandwidth)` + simulated mathematical computation bounds.

4. **Metrics Generation**
   * Energy ($J$) = Time $\times$ (Power W).
   * Cost ($\$$) = Energy $/ (3.6 \times 10^6) \times \$0.15 $.

5. **Persistence**
   * The Oracle brute forces the actual best outcome. 
   * The Task parameters, optimal fraction, and metric logs are persisted into `scheduler_results` PostgreSQL table (`data/long_term_history.csv` async backup layer).

6. **Feedback & Visualization**
   * The RL Agent `observe()`s the Cost scalar metric for experience replay back-propagation.
   * The FastAPI WebSocket broadcasts the updated aggregated metric leaderboard state to the React dashboard.
