# Hybrid ML Scheduler

> A real-time scheduler comparison system that optimizes heterogeneous GPU/CPU task scheduling using machine learning and reinforcement learning techniques.

---

### 📚 **Documentation Center**
**For deep technical details, theory, and the "Textbook" guide:**
*   [**📖 Project Wiki (Modular Documentation)**](docs/wiki/README.md) - *Comprehensive guides covering Architecture, Code, Data Workflow, and FAQs.*
*   [**📄 Legacy Project Wiki (v10.0 Ultimate Edition)**](PROJECT_WIKI_DOCUMENTATION.md) - *Legacy 1000+ lines monolithic file.*
*   [**📄 PDF Manual**](PROJECT_WIKI_DOCUMENTATION.pdf) - *The printable version of the legacy wiki.*

---

## 🚀 Overview

The Hybrid ML Scheduler is a simulation framework for comparing different task scheduling strategies in heterogeneous computing environments. It features:

- **6 Scheduling Strategies**: Round Robin, Random, Greedy, Hybrid ML, RL Agent, and Oracle (optimal baseline)
- **Live Dashboard**: Real-time WebSocket-based visualization of scheduler performance
- **Online Learning**: Continuous model retraining based on accumulated execution data
- **Comprehensive Metrics**: Time, energy consumption, and cost tracking for each scheduler

### 🍳 The Concept: A "Kitchen" Metaphor
To understand the complexity, imagine a restaurant kitchen:
*   **4 Master Chefs (GPUs):** Fast but expensive ($$$).
*   **1 Prep Cook (CPU):** Slow but cheap ($).
*   **The Challenge:** Moving ingredients to a Chef takes time. Simple tasks (chopping an onion) are faster with the Prep Cook because you save the walk. Complex tasks (Soufflé) need the Chef.
*   **Our Solution:** An **RL Agent** that learns to be the perfect Kitchen Manager, assigning tasks based on "Cooking Time" vs "Walking Time".

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Dashboard (React)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Global View │  │ Scheduler    │  │  Historical  │      │
│  │  Comparison  │  │ Details      │  │  Analysis    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │ WebSocket (ws://localhost:8000/ws)
┌───────────────────────────▼─────────────────────────────────┐
│             FastAPI Backend (dashboard_server.py)            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Continuous Simulation Engine                   │   │
│  │  • Task Generation (WorkloadGenerator)                │   │
│  │  • Parallel Scheduler Execution                       │   │
│  │  • Metrics Collection & Broadcasting                  │   │
│  │  • Model Retraining (every 50 tasks)                  │   │
│  └──────────────────────────────────────────────────────┘   │
34 └─────────────────────────────────────────────────────────────┘
           │                    │                    │
    ┌──────▼──────┐     ┌──────▼──────┐    ┌───────▼────────┐
    │   Hybrid ML  │     │  RL Agent   │    │  Simple Rules  │
    │  (RF Model)  │     │  (DQN)      │    │  (RR/Random)   │
    └──────────────┘     └─────────────┘    └────────────────┘
```

### Data Flow

1. **Task Generation**: `WorkloadGenerator` creates tasks with random size, compute intensity, and memory requirements
2. **Parallel Execution**: Each task runs through all 6 schedulers simultaneously
3. **Oracle Computation**: Brute-force grid search finds optimal GPU fraction (ground truth)
4. **Metrics Calculation**: Energy (50W GPU, 30W CPU) and cost ($0.15/kWh) computed
5. **Persistence**: Oracle results saved to `data/long_term_history.csv`
6. **Retraining**: Every 50 tasks, the Hybrid ML model retrains on accumulated data
7. **Broadcasting**: Results sent to all WebSocket clients in real-time

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **Node.js 16+** (for dashboard)
- **pip** and **npm**

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd hybrid_ml_scheduler
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd dashboard
npm install
cd ..
```

## 🏃 Usage

### Quick Start: Live Dashboard

The easiest way to see the system in action is to launch the live dashboard:

```bash
./run_live_dashboard.sh
```

This script:
1. Starts the FastAPI backend on `http://localhost:8000`
2. Launches the React frontend on `http://localhost:5173`
3. Begins the continuous simulation

Open **http://localhost:5173** in your browser to view the dashboard.

### Dashboard Views

#### 1. **Global Comparison**
- Performance race bar chart comparing all schedulers
- Workload distribution scatter plot
- Average metrics summary

#### 2. **Scheduler Details** (click any scheduler button)
- Virtual cluster load over time
- Radar chart comparing against Oracle baseline
- GPU/CPU resource split

#### 3. **Historical Analysis**
- Long-term performance trends
- Model retraining history
- Data collection statistics

### Running Experiments Manually

For offline experiments without the dashboard (Scientific Mode):

```bash
python scripts/run_heavy_simulation.py
```

This runs a batch simulation of 10,000 tasks and outputs:
- **CDF Plots:** `data/results/plots/latency_cdf.png`
- **Cost Analysis:** `data/results/plots/cost_comparison.png`
- **Detailed Logs:** `data/results/heavy_simulation_report.csv`

## 📝 API Documentation

### WebSocket Protocol

**Endpoint**: `ws://localhost:8000/ws`

**Message Format**:
```json
{
  "type": "simulation_update",
  "task": {
    "id": 123,
    "size": 960,
    "intensity": 0.75,
    "memory": 81
  },
  "latest_results": {
    "hybrid_ml": {"time": 0.45, "energy": 22.5, "cost": 0.0009},
    "oracle": {"time": 0.42, "energy": 21.0, "cost": 0.0008},
    ...
  },
  "comparison": [
    {"name": "hybrid_ml", "avg_time": 0.48},
    ...
  ],
  "utilization": {
    "average_utilization": 0.65,
    "gpu_0": {"utilization": 0.72},
    ...
  }
}
```

### REST Endpoints

#### `GET /`
Health check endpoint
```json
{"status": "online", "message": "Hybrid ML Scheduler Dashboard API is running"}
```

#### `GET /api/full_history`
Retrieve complete historical training data
```json
[
  {"task_id": 0, "size": 960, "compute_intensity": 0.26, ...},
  ...
]
```

#### `DELETE /api/history`
Clear all historical data

#### `POST /api/pause`
Pause the simulation

#### `POST /api/resume`
Resume a paused simulation

## 🧪 Testing

Run the test suite:

```bash
# Backend tests
pytest tests/
```

## 📂 Project Structure

```
hybrid_ml_scheduler/
├── src/
│   ├── simulation_engine.py    # Core simulation loop
│   ├── dashboard_server.py      # FastAPI WebSocket server
│   ├── workload_generator.py    # Task generation
│   ├── simulator.py             # GPU/CPU execution model
│   ├── offline_trainer.py       # ML model training
│   ├── online_scheduler.py      # Hybrid ML scheduler
│   ├── rl_scheduler.py          # RL agent
│   ├── dqn_scheduler.py         # Deep Q-Network implementation
│   ├── ml_models.py             # RandomForest predictor
│   └── profiler.py              # Performance profiling
├── dashboard/
│   └── src/
│       ├── App.jsx              # Main React component
│       └── index.css            # Cyberpunk theme styles
├── data/
│   └── long_term_history.csv    # Training data persistence
├── tests/                       # Unit tests
├── main.py                      # Entry point for offline experiments
├── run_live_dashboard.sh        # Dashboard launcher
├── config.yaml                  # System configuration
└── requirements.txt             # Python dependencies
```

## 🎯 Schedulers Explained

| Scheduler | Strategy | GPU Fraction Selection |
|-----------|----------|------------------------|
| **Round Robin** | Alternating | 0.5 if task_id % 2 == 0, else 0.0 |
| **Random** | Stochastic | Uniform random [0, 1] |
| **Greedy** | Heuristic | Uses task.compute_intensity |
| **Hybrid ML** | ML-Based (RandomForest) | Predicts speedup using Size + Intensity |
| **RL Agent** | Reinforcement Learning (DQN) | Learns Q-Values for discrete GPU allocations |
| **Oracle** | Optimal | Grid search over 11 fractions [0, 0.1, ..., 1.0] |

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
simulation:
  num_gpus: 4
  retrain_interval: 50
  
workload:
  size_range: [500, 1500]
  intensity_range: [0.1, 0.9]
  memory_range: [50, 150]
```

## 📈 Performance Metrics

The system tracks three key metrics:

1. **Execution Time** (seconds): Wall-clock time to complete the task
2. **Energy** (Joules): Power model: GPU=50W, CPU=30W
3. **Cost** (dollars): Based on $0.15 per kWh electricity rate

## 🚧 Development

### Adding a New Scheduler

1. Create a new scheduler class in `src/`
2. Implement the scheduling logic
3. Add to `simulation_engine.py`:
```python
self.simulators['my_scheduler'] = VirtualMultiGPU(self.num_gpus)
```
4. Update the `_run_all_schedulers` method
5. Update frontend `SCHEDULERS` constant in `App.jsx`

### Modifying the Power Model

Edit `_calculate_metrics` in `simulation_engine.py`:
```python
# Power Model: GPU=50W, CPU=30W
power = (gpu_frac * GPU_POWER) + ((1.0 - gpu_frac) * CPU_POWER)
```

## 📜 License

MIT License

---
**See [`docs/wiki/README.md`](docs/wiki/README.md) for the full modular technical encyclopedia.**
