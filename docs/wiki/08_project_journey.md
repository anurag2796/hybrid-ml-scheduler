# 08. Project Journey

This document captures the evolution and architectural phases of the Hybrid ML Scheduler across its lifecycle.

## Phase 1: Heuristic "Rules-Based" Approach (Legacy)
Our earliest iterations focused entirely on hardcoded Round Robin, Greedy, and Random simulations to validate the underlying `VirtualGPU` physics math for task distributions. We quickly realized the NP-Hard nature of Amdahl's Law in heterogeneous clusters required complex runtime adjustments based on Memory intensity vs Hardware availability constraints.

## Phase 2: Supervised Training (Hybrid ML)
We introduced Scikit-Learn `RandomForestRegressor`. By running a hypothetical perfect "Oracle" offline batch job representing $10,000$ Task variations, we generated an optimal training dataset representing the "perfect GPU fractional split for any known size". The system operated in an Offline configuration.

## Phase 3: Unsupervised DQN Evolution (The RL Agent)
Because Offline dataset inference lacks real-time insight into the cluster's active "live load" variation, we engineered the PyTorch Deep Q-Network Agent. We replaced Static Inference passing with dynamic Epsilon-Greedy inference logic, calculating latency penalty Rewards against Energy Consumption. The RL agent successfully generated dynamic adaptation.

## Phase 4: Full Stack Orchestration (The Live Dashboard)
We merged the independent Python batch scripts into the FastAPI asynchronous backend core. Using WebSockets, the simulated payload distributions and live metric tracking were cast onto the React/Vite visualization dashboard, creating a real-time cluster "brain monitor".

## Phase 5 (Current): Database Logging & Architectural Refactoring
We are actively transitioning the internal data logs into structured PostgreSQL repositories, separating ML model serialization, fixing Author/Dependency module mismatch bugs, and standardizing error handling codebases to guarantee end-to-end framework resilience.
