# Hybrid ML Scheduler - Project Documentation

**Version:** 2.0.0
**Date:** November 27, 2025

---

## 1. Executive Summary

The **Hybrid ML Scheduler** is an advanced simulation and scheduling system designed to optimize task allocation in heterogeneous computing environments (CPU + GPU). It leverages **Machine Learning (Random Forest)** and **Reinforcement Learning (DQN)** to make intelligent scheduling decisions that balance execution time, energy consumption, and cost.

The system features a continuous simulation engine, a robust backend API, persistent storage, and a real-time interactive dashboard for visualization and analysis.

---

## 2. Installation & Setup

### Prerequisites
*   **Python 3.9+**
*   **Node.js 16+**
*   **PostgreSQL** (Optional, for full persistence)
*   **Redis** (Optional, for caching)

### Backend Setup
1.  **Create Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Server**:
    ```bash
    uvicorn src.dashboard_server:app --reload
    ```
    The API will be available at `http://localhost:8000`.

### Frontend Setup
1.  **Navigate to Dashboard**:
    ```bash
    cd dashboard
    ```
2.  **Install Dependencies**:
    ```bash
    npm install
    ```
3.  **Start Development Server**:
    ```bash
    npm run dev
    ```
    The dashboard will be available at `http://localhost:5173`.

---

## 3. System Architecture (High-Level Design)

The system follows a modern, modular architecture composed of four main layers:

1.  **Presentation Layer (Frontend):** A React-based dashboard for real-time monitoring and control.
2.  **Application Layer (Backend):** A FastAPI server handling API requests, WebSocket streaming, and business logic.
3.  **Simulation Layer (Engine):** A Python-based engine that generates workloads and executes scheduling strategies.
4.  **Data Layer (Persistence):** PostgreSQL for long-term storage and Redis for high-speed caching.

```mermaid
graph TD
    subgraph "Frontend (React)"
        Dash[Dashboard UI]
    end

    subgraph "Backend (FastAPI)"
        API[API Endpoints]
        WS[WebSocket Manager]
        Sim[Simulation Engine]
        Gen[Workload Generator]
        Schedulers[Schedulers]
    end

    subgraph "Data Layer"
        DB[(PostgreSQL)]
        File[History CSV]
    end

    Gen -->|Tasks| Sim
    Sim -->|State| Schedulers
    Schedulers -->|Decisions| Sim
    Sim -->|Real-time Updates| WS
    WS -->|JSON Stream| Dash
    Sim -->|Persist Results| File
    Sim -->|Persist Results| DB
    Dash -->|Control (Start/Stop)| API
    API -->|Commands| Sim
```

### Data Flow Overview
1.  **Workload Generation:** The Simulation Engine generates synthetic tasks with varying characteristics (size, compute intensity, memory).
2.  **Scheduling:** Tasks are processed by multiple schedulers in parallel (Round Robin, Random, Greedy, Hybrid ML, RL Agent, Oracle).
3.  **Execution Simulation:** A Virtual Cluster model estimates execution time and energy based on the scheduling decision.
4.  **Data Persistence:** Results are saved to PostgreSQL via the Backend API. Training data is buffered and stored.
5.  **Model Retraining:** The Offline Trainer periodically retrains the ML model using the latest historical data from the database.
6.  **Visualization:** The Backend broadcasts real-time updates via WebSockets to the Frontend Dashboard.

---

## 4. Deep Dive: Core Logic & Algorithms

This section explains the internal mechanics of the system, allowing you to understand the "how" and "why" without reading the code.

### 4.1. Workload Generation (The "Tasks")
The system generates a continuous stream of synthetic tasks that mimic real-world parallel computing jobs.

*   **Generation Process:** Tasks arrive according to a **Poisson Process** (exponentially distributed inter-arrival times), creating a realistic, bursty workload.
*   **Task Attributes:**
    *   **Size ($N$):** The magnitude of the problem (100 - 5000 units).
    *   **Compute Intensity ($I$):** A value between 0.0 and 1.0 indicating how much the task benefits from parallelization.
        *   $I \approx 1.0$: Highly parallelizable (Matrix Multiplication, Deep Learning).
        *   $I \approx 0.0$: Serial (I/O bound, recursive logic).
    *   **Memory Required ($M$):** RAM usage (10 - 500 MB).
*   **Duration Model:** The estimated base duration is calculated as:
    $$ T_{base} \propto \frac{N^{1.5}}{I + 0.5} $$
    *This means larger tasks take super-linearly longer, but high compute intensity reduces time (assuming parallel hardware).*

### 4.2. The Schedulers (The "Competitors")
The system runs six scheduling strategies in parallel for every task to compare their performance.

#### 1. Round Robin (Baseline)
*   **Logic:** Alternates blindly between resources.
*   **Behavior:** Task $i$ goes to GPU, Task $i+1$ goes to CPU.
*   **Pros/Cons:** Simple but inefficient; sends GPU-hostile tasks to GPU and vice versa.

#### 2. Random (Baseline)
*   **Logic:** Assigns a random fraction of the task to the GPU ($0.0$ to $1.0$).
*   **Pros/Cons:** Acts as a stochastic baseline to prove that other methods are learning.

#### 3. Greedy (Heuristic)
*   **Logic:** Uses the task's **Compute Intensity** directly as the GPU fraction.
*   **Formula:** $Fraction_{GPU} = Intensity$
*   **Rationale:** High intensity tasks *should* go to GPU. This is a strong heuristic baseline.

#### 4. Hybrid ML (The "Brain")
*   **Type:** Supervised Learning (Random Forest Regressor via Scikit-Learn).
    *   **Model:** `RandomForestRegressor(n_estimators=100, max_depth=15)`.
    *   **Preprocessing:** Features are scaled using `StandardScaler`.
*   **Input Features:**
    *   `size`: Raw task size.
    *   `compute_intensity`: Parallelizability factor (0.0 - 1.0).
    *   `memory_required`: RAM usage in MB.
    *   `memory_per_size`: Density metric ($Memory / (Size + 1)$).
    *   `compute_to_memory`: Ratio metric ($Intensity / (Memory + 1)$).
*   **Decision Logic:**
    1.  **Predict:** The model predicts the optimal **GPU Fraction** ($0.0 - 1.0$).
    2.  **Select GPU:** The scheduler calculates a "Cost" for each available GPU to find the best placement:
        $$ Cost = (1 - w_E) \times \frac{T_{est}}{10.0} + w_E \times \frac{E_{est}}{500.0} $$
        *Where $w_E$ is the Energy Weight (default 0.5), normalizing time to ~10s and energy to ~500J.*
*   **Training Loop:**
    *   Retrains every 50 tasks (sliding window).
    *   Uses "Oracle" decisions (retrospective optimal choices) as the ground truth labels.

#### 5. RL Agent (Deep Q-Network)
*   **Architecture:** Dueling DQN (Deep Q-Network).
    *   **Value Stream:** Estimates state value $V(s)$.
    *   **Advantage Stream:** Estimates action advantage $A(s, a)$.
    *   **Aggregation:** $Q(s, a) = V(s) + (A(s, a) - \text{mean}(A))$.
    *   **Hidden Layers:** 2x Fully Connected (256 units, ReLU activation).
*   **State Space (Normalized):**
    *   $\text{Size} / 10000.0$
    *   $\text{Compute Intensity}$ (Raw 0-1)
    *   $\text{Memory} / 5000.0$
*   **Action Space:** Discrete options $[ \text{CPU}, \text{GPU}_0, \text{GPU}_1, \dots, \text{GPU}_N ]$.
*   **Reward Function:**
    *   The agent aims to maximize specific rewards defined as negative cost:
    *   $$ R = - \left[ (1 - w) \times T_{exec} + w \times \frac{E_{joules}}{100.0} \right] $$
*   **Hyperparameters:**
    *   `Gamma` (Discount Factor): 0.99
    *   `Epsilon` (Exploration): Starts at 1.0, decays to 0.01 (Factor: 0.9999).
    *   `Replay Buffer`: 50,000 transitions.
    *   `Batch Size`: 128.
    *   `Target Update`: Every 100 steps.

#### 6. Oracle (The "Ground Truth")
*   **Logic:** A theoretical solver that "cheats" by trying every possible split (0% to 100% in 5% steps).
*   **Purpose:** It finds the absolute mathematical minimum execution time for a task.
*   **Usage:**
    *   Acts as the **Label** for the Hybrid ML model (Supervised Learning).
    *   Serves as the **Performance Ceiling** (100% Efficiency) for comparison.

### 4.3. Simulation Physics
How do we calculate "Time" and "Energy"?

*   **Execution Time:**
    *   **CPU Time:** Base duration.
    *   **GPU Time:** $\frac{Base Duration}{Speedup} + TransferTime$
    *   **Speedup:** $1.0 + (3.0 \times Intensity)$ (Max 4x speedup for high intensity).
    *   **Transfer Time:** $\frac{Memory}{Bandwidth}$ (Simulating PCIe bottlenecks).
*   **Energy Consumption:**
    *   **GPU Power:** 50W (Active).
    *   **CPU Power:** 30W (Active).
    *   $Energy (Joules) = Power \times Time$.

---

## 5. System Flow & Architecture

### 5.1. The "Loop"
1.  **Generate:** A new task is born (`WorkloadGenerator`).
2.  **Broadcast:** The task is sent to all 6 schedulers simultaneously.
3.  **Decide:** Each scheduler makes its move (Predict, Randomize, or Calculate).
4.  **Simulate:** The `VirtualMultiGPU` calculates the *result* (Time, Energy) for each decision.
5.  **Persist:** Results are saved to PostgreSQL.
6.  **Learn:**
    *   **Hybrid ML:** If 50 tasks have passed, fetch history -> Retrain Random Forest.
    *   **RL Agent:** Store transition -> Update Q-Network weights.
7.  **Visualize:** Send JSON packet via WebSocket to the Dashboard.

### 5.2. Component Details

#### Frontend Dashboard (`/dashboard`)
*   **Tech:** React, Vite, TailwindCSS, Recharts.
*   **Key Views:**
    *   **Performance Race:** A live bar chart where shorter bars = faster schedulers.
    *   **Enhanced Analytics:**
        *   **Heatmap:** Shows which tasks (Size vs. Intensity) perform best on GPU.
        *   **Win/Loss Matrix:** How often Scheduler A beats Scheduler B.
        *   **State Management**: Uses React `useState` and `useEffect` for real-time data updates.

![Dashboard Mockup](docs/images/dashboard_mockup.png)

#### Backend Server (`/backend`)
*   **Tech:** FastAPI, Uvicorn, SQLAlchemy (Async), Pydantic.
*   **Role:** The central nervous system. It orchestrates the simulation, manages the DB connection, and serves the API.
*   **Security:** Implements Rate Limiting (Redis) and Input Validation to protect the system.

#### Data Layer
*   **PostgreSQL:** Stores the "Truth". Every single task execution is logged here.
*   **Redis:** The "Short-term Memory". Caches high-speed data like current stats to prevent DB overload.

## 6. Technical Reference

### 6.1. Database Schema (PostgreSQL)
The system uses a relational schema optimized for time-series performance.

#### `tasks` Table
Stores metadata for every generated task.
| Column | Type | Description |
|--------|------|-------------|
| `task_id` | Integer (PK) | Unique identifier for the task. |
| `size` | Float | Problem size ($N$). |
| `compute_intensity` | Float | Parallelizability factor ($0.0 - 1.0$). |
| `memory_required` | Float | RAM required in MB. |
| `arrival_time` | Float | Simulation timestamp of arrival. |
| `dependencies` | JSONB | List of parent task IDs. |

#### `scheduler_results` Table
Logs the outcome of every scheduling decision.
| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Unique result ID. |
| `task_id` | Integer (FK) | Reference to the task. |
| `scheduler_name` | String | Name of the strategy (e.g., 'hybrid_ml'). |
| `gpu_fraction` | Float | Allocated GPU portion ($0.0 - 1.0$). |
| `actual_time` | Float | Execution time in seconds. |
| `energy_consumption` | Float | Energy used in Joules. |
| `execution_cost` | Float | Cost in USD. |

#### `training_data` Table
Historical data used to train the Hybrid ML model.
| Column | Type | Description |
|--------|------|-------------|
| `size` | Float | Task size. |
| `compute_intensity` | Float | Task intensity. |
| `optimal_gpu_fraction` | Float | **Label:** The best fraction found by Oracle. |
| `optimal_time` | Float | The execution time achieved by Oracle. |

### 6.2. API Specification (FastAPI)

#### Simulation Control
*   `POST /api/simulation/start`: Begin the continuous simulation.
*   `POST /api/simulation/stop`: Gracefully stop the engine.
*   `POST /api/simulation/pause`: Temporarily halt task generation.
*   `GET /api/simulation/status`: Get current stats (tasks processed, running state).

#### Data & Metrics
*   `GET /api/full_history`: Retrieve the latest 1000 training records (used for charts).
*   `GET /api/metrics`: Expose Prometheus-formatted metrics for scraping.
*   `GET /api/health`: Check connectivity to PostgreSQL and Redis.

#### WebSocket (`/ws`)
*   **Protocol:** JSON over WebSocket.
*   **Events:**
    *   `simulation_update`: Real-time packet with current task, scheduler results, and cluster utilization.
    *   `notification`: System alerts (e.g., "Model Retrained").

### 6.3. Configuration Management
The system is configured via environment variables (using `pydantic-settings`).

#### Key Variables (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | App environment (dev/prod). |
| `POSTGRES_HOST` | localhost | Database host. |
| `POSTGRES_DB` | hybrid_scheduler_db | Database name. |
| `REDIS_HOST` | localhost | Redis cache host. |
| `NUM_GPUS` | 4 | Number of virtual GPUs to simulate. |
| `RETRAIN_INTERVAL` | 50 | Tasks between model updates. |

---

## 7. Directory Structure
```
hybrid_ml_scheduler/
├── backend/                # FastAPI Application
│   ├── api/                # Routes and Controllers
│   ├── core/               # Config and DB setup
│   ├── middleware/         # Rate Limit, Security
│   ├── models/             # SQLAlchemy & Pydantic models
│   └── services/           # Business Logic (Data, Cache)
├── dashboard/              # React Frontend
│   ├── src/
│   │   ├── components/     # Reusable UI widgets (Charts)
│   │   └── App.jsx         # Main Dashboard View
├── src/                    # Simulation Engine
│   ├── simulation_engine.py # Main Loop
│   ├── online_scheduler.py  # Hybrid ML Logic
│   ├── dqn_scheduler.py     # RL Logic
│   └── workload_generator.py # Task Factory
├── scripts/                # Utility Scripts (Init DB, Verify)
└── tests/                  # Unit and Integration Tests
```

---

