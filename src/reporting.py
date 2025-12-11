import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
from loguru import logger

def create_text_page(pdf, title, content, fontsize=10):
    """Creates a text-only page in the PDF."""
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
            if len(current_line + " " + word) > 90:
                plt.text(0.1, y_pos, current_line, ha='left', va='top', fontsize=fontsize, fontfamily='monospace')
                y_pos -= 0.015
                current_line = word
            else:
                current_line += " " + word if current_line else word
        
        if current_line:
            plt.text(0.1, y_pos, current_line, ha='left', va='top', fontsize=fontsize, fontfamily='monospace')
            y_pos -= 0.015
        
        y_pos -= 0.005 # Paragraph spacing
        
        if y_pos < 0.05:
            pdf.savefig(fig)
            plt.close()
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            y_pos = 0.95

    pdf.savefig(fig)
    plt.close()

def add_image_page(pdf, image_path, title):
    """Adds an image to a page in the PDF."""
    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Warning: {image_path} not found")
        return

    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, weight='bold')
    
    try:
        img = mpimg.imread(str(path))
        # Maintain aspect ratio, fit to page
        plt.imshow(img)
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.90])
        pdf.savefig(fig)
    except Exception as e:
        logger.error(f"Failed to add image {image_path}: {e}")
    finally:
        plt.close()

def generate_enhanced_report(output_path, baselines, plots_dir):
    """
    Generates a PDF report with detailed metrics and plots.
    
    Args:
        output_path: Path to save the PDF.
        baselines: Dictionary of results (e.g., {'strategy': {'makespan': ..., 'latencies': ...}}).
        plots_dir: Directory containing generated plots.
    """
    output_path = Path(output_path)
    plots_dir = Path(plots_dir)
    
    logger.info(f"Generating report at {output_path}")
    
    with PdfPages(output_path) as pdf:
        # Page 1: Executive Summary
        title = "Heavy Workload Simulation Report"
        
        # Calculate summary metrics
        strategies = list(baselines.keys())
        best_strategy = min(baselines, key=lambda k: baselines[k].get('makespan', float('inf')))
        worst_strategy = max(baselines, key=lambda k: baselines[k].get('makespan', float('-inf')))
        
        summary_text = f"""
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

1. EXECUTIVE SUMMARY
--------------------
This report analyzes the performance of scheduling strategies under a HEAVY workload regime.
Workload Characteristics:
- Total Tasks: ~10,000 per strategy
- Compute Bias: High (0.6 - 1.0 intensity)
- Size Range: 100 - 5000 units
- Memory Range: 100 - 10000 MB

Key Findings:
- Best Performing Strategy: {best_strategy.upper()}
- Worst Performing Strategy: {worst_strategy.upper()}
- Speedup: {best_strategy.upper()} was {baselines[worst_strategy]['makespan'] / baselines[best_strategy]['makespan']:.2f}x faster than {worst_strategy.upper()}.

2. DETAILED METRICS
-------------------
"""
        
        # Add detailed table
        headers = ["Strategy", "Makespan(s)", "Avg Latency(s)", "P95 Latency(s)", "P99 Latency(s)", "Throughput(T/s)"]
        summary_text += f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<14} | {headers[3]:<14} | {headers[4]:<14} | {headers[5]:<14}\n"
        summary_text += "-"*100 + "\n"
        
        for name, metrics in baselines.items():
            makespan = metrics.get('makespan', 0)
            avg_lat = metrics.get('avg_time', 0)
            p95 = metrics.get('p95_time', 0)
            p99 = metrics.get('p99_time', 0)
            throughput = metrics.get('throughput', 0)
            
            summary_text += f"{name:<20} | {makespan:<12.2f} | {avg_lat:<14.4f} | {p95:<14.4f} | {p99:<14.4f} | {throughput:<14.2f}\n"

        summary_text += """
        
3. COST ANALYSIS
----------------
(Abstract Cost units based on compute time and resource usage)
"""
        summary_text += f"{'Strategy':<20} | {'Total Cost':<12} | {'Cost Efficiency':<15}\n"
        summary_text += "-"*60 + "\n"
        
        for name, metrics in baselines.items():
            cost = metrics.get('total_cost', 0)
            efficiency = metrics.get('cost_efficiency', 0)
            summary_text += f"{name:<20} | {cost:<12.2f} | {efficiency:<15.2f}\n"

        create_text_page(pdf, title, summary_text)
        
        # Plots
        add_image_page(pdf, plots_dir / "makespan_comparison.png", "Makespan Comparison")
        add_image_page(pdf, plots_dir / "latency_cdf.png", "Latency CDF (Cumulative Distribution)")
        add_image_page(pdf, plots_dir / "latency_boxplot.png", "Latency Distribution (Box Plot)")
        add_image_page(pdf, plots_dir / "cost_comparison.png", "Total Operational Cost")
        add_image_page(pdf, plots_dir / "speedup_comparison.png", "Speedup vs Baseline")
        add_image_page(pdf, plots_dir / "compute_vs_memory.png", "Workload Characteristics")
