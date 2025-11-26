# Hybrid Offline-Online ML Scheduler for Parallel Computing on M4 Max

## Complete Project Setup & Execution Guide

### Project Overview

This project implements a **Hybrid Offline-Online ML Scheduler** for dynamic resource allocation in parallel computing systems. It's specifically optimized for MacBook M4 Max with 36GB RAM.

**Key Components:**
- Phase 1: Hardware Profiling (benchmark CPU/GPU performance)
- Phase 2: Workload Data Generation (create synthetic training data)
- Phase 3: Offline ML Training (train performance prediction models)
- Phase 4: Online Scheduling (real-time task scheduling)
- Phase 5: Evaluation & Analysis (compare against baselines)

---

## Installation & Setup (15 minutes)

### Step 1: Create Project Structure

```bash
# Create project directory
mkdir scheduler-project
cd scheduler-project

# Create subdirectories
mkdir -p src notebooks data/{workload_traces,profiles,results} models logs
```

### Step 2: Create Python Virtual Environment

```bash
# Using pyenv (you have Python 3.12.7)
pyenv local 3.12.7

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Dependencies

```bash
# Copy the requirements.txt content and install
pip install -r requirements.txt
```

**Note for M4 Max:** PyTorch will automatically use MPS (Metal Performance Shaders) backend. Verify with:

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Step 4: Copy Configuration Files

Create `config.yaml` in project root with the provided configuration.

---

## Week-by-Week Implementation

### WEEK 1: Profiling & Data Generation

#### Day 1-2: Hardware Profiling

**File: `src/profiler.py`** (100+ lines)

```python
from src.profiler import HardwareProfiler

# Initialize profiler
profiler = HardwareProfiler(device_type="mps")

# Run benchmarks on different matrix sizes
profile = profiler.profile_range(sizes=[256, 512, 1024, 2048])

# Save profile for later use
profiler.save_profile("data/profiles/hardware_profile.json")

print(f"CPU/GPU Ratio: {profile['cpu_gpu_ratio']:.2f}")
print(f"GPU should handle: {profile['gpu_fraction']:.1%} of work")
```

**Expected Output:**
```
Hardware Profiler initialized on device: mps
Starting hardware profiling with sizes: [256, 512, 1024, 2048]
Profiling size 256x256...
CPU benchmark 256x256: 0.0023s
GPU benchmark 256x256: 0.0051s
...
Performance Model: CPU/GPU Ratio = 0.45
GPU should receive 31.0% of work
```

**Key Metrics:**
- CPU vs GPU execution time for different problem sizes
- CPU/GPU performance ratio (determines optimal work splitting)
- Hardware profile saved for reproducibility

---

#### Day 3-5: Workload Generation

**File: `src/workload_generator.py`** (150+ lines)

```python
from src.workload_generator import WorkloadGenerator

# Generate synthetic workload
wg = WorkloadGenerator(seed=42)

tasks = wg.generate_workload(
    num_tasks=10000,
    task_size_range=(100, 5000),
    compute_intensity_range=(0.1, 1.0),
    memory_range=(10, 500),
    arrival_rate=100.0
)

# Save for training
wg.save_workload("data/workload_traces/training_workload.csv")

# View statistics
stats = wg.get_statistics()
print(f"Generated {stats['num_tasks']} tasks")
print(f"Avg task size: {stats['avg_size']:.0f}")
print(f"Avg compute intensity: {stats['avg_intensity']:.2f}")
```

**Generated Features:**
- `task_id`: Unique identifier
- `size`: Problem size (compute amount)
- `compute_intensity`: 0-1, higher = GPU-friendly
- `memory_required`: MB needed
- `arrival_time`: When task arrives
- `duration_estimate`: Expected execution time

---

### WEEK 2: Offline ML Training

#### Day 6-8: ML Model Training

**File: `src/ml_models.py`** (220+ lines)

```python
from src.ml_models import RandomForestPredictor, XGBoostPredictor
from src.workload_generator import WorkloadGenerator

# Load training workload
wg = WorkloadGenerator.load("data/workload_traces/training_workload.csv")

# Option 1: Use Random Forest (recommended for M4 Max)
model = RandomForestPredictor(n_estimators=100, max_depth=15)

# Option 2: Use XGBoost (more accurate but slower)
# model = XGBoostPredictor(n_estimators=100, max_depth=7)

# Prepare features
df = wg.to_dataframe()
features = ['task_size', 'compute_intensity', 'memory_required']
X = df[features]
y = df['duration_estimate']

# Train model
results = model.fit(X, y, test_size=0.2)
print(f"Train R²: {results['train_r2']:.4f}")
print(f"Test R²: {results['test_r2']:.4f}")

# Get feature importances
importances = model.feature_importance()
print(f"Feature importances: {importances}")

# Save model for online use
model.save("models/scheduler_model.pkl")
```

**Expected Output:**
```
Training Random Forest model...
Train R2: 0.8234, Test R2: 0.8120
CV R2: 0.8100 (+/- 0.0150)
Feature importances:
  - task_size: 0.4521
  - compute_intensity: 0.3842
  - memory_required: 0.1637
Model saved to models/scheduler_model.pkl
```

#### Day 9-10: Offline Trainer Pipeline

**File: `src/offline_trainer.py`** (180+ lines)

```python
from src.offline_trainer import OfflineTrainer
from src.workload_generator import WorkloadGenerator

# Initialize trainer
trainer = OfflineTrainer(
    model_type="random_forest",
    n_estimators=100,
    max_depth=15
)

# Load workload
wg = WorkloadGenerator.load("data/workload_traces/training_workload.csv")

# Run complete pipeline
results = trainer.run_full_pipeline(
    wg,
    model_output_path="models/scheduler_model.pkl"
)

print(f"Training complete!")
print(f"Feature importances: {results['feature_importances']}")
```

---

### WEEK 3: Online Scheduling & Simulation

#### Day 11-13: Online Scheduler Implementation

**File: `src/online_scheduler.py`** (200+ lines)

```python
from src.online_scheduler import OnlineScheduler
from src.ml_models import RandomForestPredictor

# Load trained model
model = RandomForestPredictor.load("models/scheduler_model.pkl")

# Create online scheduler with 4 virtual GPUs
scheduler = OnlineScheduler(model=model, num_gpus=4)

# Generate test tasks
from src.workload_generator import WorkloadGenerator
wg = WorkloadGenerator(seed=99)
test_tasks = wg.generate_workload(num_tasks=1000)

# Submit tasks
for task in test_tasks:
    scheduler.submit_task(task)

# Schedule all tasks
decisions = scheduler.process_queue()

print(f"Scheduled {len(decisions)} tasks")

# Check utilization
util = scheduler.get_utilization()
print(f"Average GPU utilization: {util['average_utilization']:.1%}")

for gpu_util in util.items():
    if gpu_util[0] != 'average_utilization':
        print(f"  {gpu_util[0]}: {gpu_util[1]['utilization']:.1%}")
```

#### Day 14: Multi-GPU Simulation

**File: `src/simulator.py`** (120+ lines)

```python
from src.simulator import VirtualMultiGPU
from src.workload_generator import WorkloadGenerator

# Create simulator
simulator = VirtualMultiGPU(num_gpus=4, memory_per_gpu=8000)

# Generate test workload
wg = WorkloadGenerator(seed=123)
test_tasks = wg.generate_workload(num_tasks=500)

# Evaluate baseline schedulers
baselines = simulator.evaluate_baseline_schedulers(test_tasks)

print("Baseline Performance Comparison:")
for strategy, metrics in baselines.items():
    print(f"\n{strategy}:")
    print(f"  Makespan: {metrics['makespan']:.2f}s")
    print(f"  Avg time: {metrics['avg_time']:.4f}s")
    print(f"  Max time: {metrics['max_time']:.4f}s")
```

**Expected Baselines:**
- `round_robin`: Fixed 50-50 split
- `random`: Random allocation
- `greedy`: Based on compute intensity

---

### WEEK 4: Evaluation & Analysis

#### Day 15-18: Complete Pipeline Execution

**File: `main.py`** (200+ lines)

```bash
# Run complete project
python main.py
```

This executes all 5 phases:

1. **Phase 1**: Hardware profiling → `data/profiles/hardware_profile.json`
2. **Phase 2**: Workload generation → `data/workload_traces/workload_train.csv`
3. **Phase 3**: ML training → `models/scheduler_model.pkl` + feature importances
4. **Phase 4**: Online scheduling → 1000+ task schedules
5. **Phase 5**: Evaluation → comparison with baselines

#### Day 19: Performance Analysis & Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load scheduling results
results = pd.read_csv("data/results/scheduling_results.csv")

# Analyze predictions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# GPU fraction distribution
axes[0, 0].hist(results['gpu_fraction'], bins=50)
axes[0, 0].set_title('GPU Fraction Distribution')
axes[0, 0].set_xlabel('GPU Fraction')

# Task size vs GPU allocation
axes[0, 1].scatter(results['task_id'], results['gpu_fraction'], alpha=0.5)
axes[0, 1].set_title('GPU Allocation Over Time')
axes[0, 1].set_xlabel('Task ID')

# Comparison metrics
strategies = ['round_robin', 'random', 'greedy', 'our_scheduler']
makespans = [5.2, 4.8, 4.5, 4.1]  # Example values
axes[1, 0].bar(strategies, makespans)
axes[1, 0].set_title('Makespan Comparison')
axes[1, 0].set_ylabel('Total Time (s)')

# Speedup analysis
speedups = [1.0, 1.08, 1.15, 1.27]  # vs CPU baseline
axes[1, 1].bar(strategies, speedups)
axes[1, 1].set_title('Speedup vs CPU Baseline')
axes[1, 1].set_ylabel('Speedup Factor')

plt.tight_layout()
plt.savefig('data/results/performance_analysis.png', dpi=300)
plt.show()
```

---

## Expected Performance Metrics

### Phase 1: Hardware Profiling
```
CPU/GPU Ratio: 0.4-0.6 (GPU is 1.7-2.5x faster)
GPU work fraction: 30-45% (CPU-bound overall on M4)
Memory bandwidth: 400+ GB/s
```

### Phase 3: ML Model Accuracy
```
Random Forest:
  - Train R²: 0.80-0.85
  - Test R²: 0.78-0.82
  - MAE: 0.05-0.15 (for 0-1 GPU fraction predictions)
```

### Phase 5: Scheduling Performance
```
Speedup vs Round-Robin: 1.2-1.4x
Speedup vs Greedy: 1.05-1.15x
Load Balance: 0.90-0.95
```

---

## Troubleshooting on M4 Max

### Issue: MPS Device Not Available
```python
# Check MPS support
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True

# If False, reinstall PyTorch:
pip install --upgrade torch torchvision torchaudio
```

### Issue: Slow GPU Performance
- Expected on M4 Max: GPU is integrated, not discrete
- MPS has ~60% speedup vs CPU on matrix operations
- This is sufficient for the project!

### Issue: Memory Issues
- Your 36GB RAM is more than enough
- If OOM: reduce `num_tasks` in config or batch size

---

## Project Deliverables

### Code Files
- `src/profiler.py`: Hardware profiling
- `src/workload_generator.py`: Synthetic workload generation
- `src/ml_models.py`: ML prediction models
- `src/offline_trainer.py`: Training pipeline
- `src/online_scheduler.py`: Online scheduling
- `src/simulator.py`: Baseline simulation
- `main.py`: Complete pipeline
- `config.yaml`: Configuration

### Data Files
- `data/profiles/hardware_profile.json`: Performance characteristics
- `data/workload_traces/training_workload.csv`: Training data
- `data/results/scheduling_results.csv`: Scheduling decisions
- `models/scheduler_model.pkl`: Trained model

### Results & Analysis
- Performance comparison charts
- Model accuracy plots
- Speedup analysis
- Load balance metrics
- Feature importance ranking

### Documentation
- `README.md`: Project overview
- `METHODOLOGY.md`: Technical approach
- `RESULTS.md`: Experimental findings
- Jupyter notebooks with analysis

---

## Next Steps & Extensions

### Easy Extensions (1-2 weeks)
1. Add energy consumption optimization
2. Test with different ML models (SVM, Neural Networks)
3. Add multiple workload types (convolution, sorting, sparse operations)
4. Implement online model updating

### Medium Extensions (2-4 weeks)
1. Add real GPU profiling (if you get GPU access)
2. Implement hierarchical scheduling
3. Add task dependencies and DAG scheduling
4. Distributed scheduler across machines

### Advanced Extensions (4-8 weeks)
1. Implement reinforcement learning scheduler (Q-learning)
2. Add dynamic feature extraction
3. Multi-objective optimization (time + energy)
4. Publish paper on approach!

---

## Estimated Timeline

| Phase | Difficulty | Time | Status |
|-------|-----------|------|--------|
| Setup | Easy | 1 day | ✓ |
| Profiling | Easy | 2 days | |
| Workload Gen | Easy | 2 days | |
| ML Training | Medium | 3 days | |
| Online Scheduler | Medium | 3 days | |
| Evaluation | Medium | 3 days | |
| Analysis & Report | Medium | 3 days | |
| **Total** | **Medium** | **~4 weeks** | |

---

## Questions & Support

Refer to:
- Comments in code for detailed explanations
- Config file for tuning parameters
- Research papers in `docs/` folder
- GitHub issues for similar projects

Good luck with your project!
