
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from pathlib import Path
import datetime

def create_text_page(pdf, title, content, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, weight='bold')
    
    # Content
    y_pos = 0.85
    for line in content.split('\n'):
        # Simple word wrap
        words = line.split()
        current_line = ""
        for word in words:
            if len(current_line + " " + word) > 80:
                plt.text(0.1, y_pos, current_line, ha='left', va='top', fontsize=fontsize, fontfamily='monospace')
                y_pos -= 0.02
                current_line = word
            else:
                current_line += " " + word if current_line else word
        
        if current_line:
            plt.text(0.1, y_pos, current_line, ha='left', va='top', fontsize=fontsize, fontfamily='monospace')
            y_pos -= 0.02
        
        y_pos -= 0.01 # Paragraph spacing
        
        if y_pos < 0.05:
            pdf.savefig(fig)
            plt.close()
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            y_pos = 0.95

    pdf.savefig(fig)
    plt.close()

def add_image_page(pdf, image_path, title):
    if not Path(image_path).exists():
        print(f"Warning: {image_path} not found")
        return

    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, weight='bold')
    
    img = mpimg.imread(image_path)
    # Maintain aspect ratio, fit to page
    plt.imshow(img)
    
    # Adjust layout to fit image
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])
    
    pdf.savefig(fig)
    plt.close()

def main():
    output_path = "data/results/Project_Report.pdf"
    plots_dir = Path("data/results/plots")
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title & Summary
        title = "Hybrid ML Scheduler Project Report"
        summary = """
Date: {}

1. EXECUTIVE SUMMARY
--------------------
This project implements a Hybrid Offline-Online Scheduler for parallel computing tasks.
It compares a static Offline Trainer (Random Forest) against an adaptive Reinforcement Learning (DQN) agent.

Experiment: "Distribution Shift & Realistic Physics"
- We introduced realistic data transfer costs and intensity-based speedups.
- We trained models on GPU-favorable tasks (High Intensity).
- We evaluated on CPU-favorable tasks (Low Intensity, High Transfer Cost).

Results:
- Offline Trainer: FAILED (Makespan: 122.92s). It failed to adapt and blindly used GPU.
- RL Agent: SUCCESS (Makespan: 21.12s). It adapted online and switched to CPU.
- Improvement: RL Agent was ~6x faster than the Offline Trainer.

2. CODE FLOW
------------
The execution pipeline follows these steps (in main.py):

1. Initialization:
   - Load configuration from config.yaml.
   - Setup logging.

2. Phase 1: Hardware Profiling (src/profiler.py)
   - Benchmarks CPU vs GPU matrix multiplication.
   - Establishes baseline performance ratios.

3. Phase 2: Data Generation (src/workload_generator.py)
   - Generates synthetic training data.
   - In this experiment: Biased towards High Compute Intensity (GPU optimal).

4. Phase 3: Offline Training (src/offline_trainer.py)
   - Trains a Random Forest model on the generated data.
   - Learns to predict "GPU" for everything (due to bias).

5. Phase 4.5: RL Training (src/dqn_scheduler.py)
   - Pre-trains DQN agent on the same biased data.
   - Agent starts with "Always GPU" policy.

6. Phase 5: Evaluation (src/simulator.py)
   - Generates TEST data: Biased towards Low Intensity + High Memory (CPU optimal).
   - Runs the Virtual Simulator with Transfer Costs.
   - Compares strategies.
   - RL Agent enables online learning (epsilon=0.1) to adapt during this phase.

""".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        create_text_page(pdf, title, summary)
        
        # Page 2: Results Table
        results = """
3. EXPERIMENTAL RESULTS
-----------------------
Workload: 500 Tasks (Low Intensity, High Memory)

Strategy          | Makespan (s) | Avg Time (s) | Status
------------------|--------------|--------------|-------
Offline Optimal   | 12.97        | 0.03         | Benchmark
Round Robin       | 56.87        | 0.11         | Baseline
Random            | 67.14        | 0.13         | Baseline
Greedy            | 18.30        | 0.04         | Baseline
Hybrid ML (RF)    | 122.92       | 0.25         | FAILED
RL Agent (DQN)    | 21.12        | 0.04         | SUCCESS

Key Insight:
The RL Agent successfully unlearned its training bias and adapted to the new environment, whereas the Offline Trainer remained static and performed poorly.
"""
        create_text_page(pdf, "Detailed Results", results)
        
        # Page 3+: Plots
        add_image_page(pdf, plots_dir / "makespan_comparison.png", "Makespan Comparison")
        add_image_page(pdf, plots_dir / "avg_duration_comparison.png", "Average Task Duration")
        add_image_page(pdf, plots_dir / "speedup_comparison.png", "Speedup vs Baseline")
        add_image_page(pdf, plots_dir / "compute_vs_memory.png", "Workload Characteristics")
        
    print(f"Report generated at {output_path}")

if __name__ == "__main__":
    main()
