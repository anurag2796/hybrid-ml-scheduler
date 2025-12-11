
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.workload_generator import WorkloadGenerator

def main():
    print("Generating verification workload...")
    gen = WorkloadGenerator(seed=123)
    tasks = gen.generate_workload(num_tasks=1000)
    
    sizes = [t.size for t in tasks]
    intensities = [t.compute_intensity for t in tasks]
    memories = [t.memory_required for t in tasks]

    print("\n--- Task Size Stats (Pareto expected) ---")
    print(f"Mean: {np.mean(sizes):.2f}")
    print(f"Median: {np.median(sizes):.2f}")
    print(f"Max: {np.max(sizes)}")
    print(f"Min: {np.min(sizes)}")
    print(f"Skewness check: Mean > Median? {np.mean(sizes) > np.median(sizes)}")

    print("\n--- Compute Intensity Stats (Bimodal expected) ---")
    print(f"Mean: {np.mean(intensities):.2f}")
    print(f"Std Dev: {np.std(intensities):.2f}")
    # Simple check for peaks (very rough)
    low_intensity = [x for x in intensities if x < 0.5]
    high_intensity = [x for x in intensities if x >= 0.5]
    print(f"Count < 0.5: {len(low_intensity)}")
    print(f"Count >= 0.5: {len(high_intensity)}")

    print("\n--- Memory Stats (Correlated expected) ---")
    correlation = np.corrcoef(sizes, memories)[0, 1]
    print(f"Correlation with Size: {correlation:.4f}")

if __name__ == "__main__":
    main()
