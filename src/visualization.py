"""
Visualization Module
Generates plots and charts for analyzing scheduling performance
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from loguru import logger

def plot_comparison(results: Dict[str, Dict[str, float]], 
                   output_dir: str = "data/results/plots"):
    """
    Plot comparison of different scheduling strategies
    
    Args:
        results: Dictionary mapping strategy names to metrics
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    strategies = list(results.keys())
    metrics = list(results[strategies[0]].keys())
    
    # 1. Makespan Comparison
    plt.figure(figsize=(10, 6))
    makespans = [results[s]['makespan'] for s in strategies]
    bars = plt.bar(strategies, makespans, color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6'])
    
    plt.title('Total Execution Time (Makespan) by Strategy', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(output_path / 'makespan_comparison.png', dpi=300)
    plt.close()
    
    # 2. Average Task Duration Comparison
    plt.figure(figsize=(10, 6))
    avg_times = [results[s]['avg_time'] for s in strategies]
    bars = plt.bar(strategies, avg_times, color=['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6'])
    
    plt.title('Average Task Duration by Strategy', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(output_path / 'avg_duration_comparison.png', dpi=300)
    plt.close()
    
    # 3. Speedup relative to Baseline (First strategy)
    baseline_makespan = makespans[0]
    speedups = [baseline_makespan / m for m in makespans]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, speedups, color=['#95a5a6', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6'])
    
    plt.title(f'Speedup Factor (relative to {strategies[0]})', fontsize=14)
    plt.ylabel('Speedup Factor (Higher is Better)', fontsize=12)
    plt.axhline(y=1.0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(output_path / 'speedup_comparison.png', dpi=300)
    plt.close()
    
    logger.info(f"Plots saved to {output_path}")

def plot_workload_characteristics(tasks: List[Dict], output_dir: str = "data/results/plots"):
    """
    Plot characteristics of the generated workload
    
    Args:
        tasks: List of task dictionaries or Task objects
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame if needed
    if not isinstance(tasks, pd.DataFrame):
        if hasattr(tasks[0], 'to_dict'):
            data = [t.to_dict() for t in tasks]
        else:
            data = tasks
        df = pd.DataFrame(data)
    else:
        df = tasks
        
    # 1. Task Size Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['size'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    plt.title('Task Size Distribution', fontsize=14)
    plt.xlabel('Task Size', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(output_path / 'task_size_dist.png', dpi=300)
    plt.close()
    
    # 2. Compute Intensity vs Memory
    plt.figure(figsize=(10, 6))
    plt.scatter(df['memory_required'], df['compute_intensity'], 
               c=df['size'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Task Size')
    plt.title('Compute Intensity vs Memory Requirement', fontsize=14)
    plt.xlabel('Memory Required (MB)', fontsize=12)
    plt.ylabel('Compute Intensity (0-1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path / 'compute_vs_memory.png', dpi=300)
    plt.close()
    
    # 3. Arrival Rate (Tasks per second)
    plt.figure(figsize=(12, 6))
    # Bin arrival times into 1-second intervals
    max_time = int(df['arrival_time'].max()) + 1
    bins = range(0, max_time + 1)
    plt.hist(df['arrival_time'], bins=bins, color='#2ecc71', alpha=0.7)
    plt.title('Task Arrival Rate Over Time', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Tasks per Second', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(output_path / 'arrival_rate.png', dpi=300)
    plt.close()
    
    # 4. Dependency Graph (First 50 tasks)
    # Only if dependencies exist
    if 'dependencies' in df.columns:
        try:
            import networkx as nx
            
            # Parse dependencies if they are strings
            if isinstance(df['dependencies'].iloc[0], str):
                df['dependencies'] = df['dependencies'].apply(eval)
                
            subset = df.head(50)
            G = nx.DiGraph()
            
            for _, task in subset.iterrows():
                G.add_node(task['task_id'], size=task['size'])
                for dep in task['dependencies']:
                    if dep in subset['task_id'].values:
                        G.add_edge(dep, task['task_id'])
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='#9b59b6', 
                   node_size=500, font_color='white', font_size=8,
                   edge_color='gray', arrows=True)
            plt.title('Task Dependency Graph (First 50 Tasks)', fontsize=14)
            plt.savefig(output_path / 'dependency_graph.png', dpi=300)
            plt.close()
        except ImportError:
            logger.warning("NetworkX not installed, skipping dependency graph")
        except Exception as e:
            logger.warning(f"Failed to plot dependency graph: {e}")

    logger.info(f"Workload plots saved to {output_path}")
