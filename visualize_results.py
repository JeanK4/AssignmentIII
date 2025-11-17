"""
Data Visualization for Disk Scheduling Algorithms
Assignment III - Operating Systems 2025-2

This script creates visualizations to compare the performance of
FCFS, SCAN, and C-SCAN disk scheduling algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import os


def load_sequences(filename: str = "sequences.txt") -> Tuple[int, List[int], List[int], List[int], List[int]]:
    """
    Load sequence data from file.
    
    Args:
        filename: Path to the sequences file
        
    Returns:
        Tuple of (initial_position, requests, fcfs_seq, scan_seq, cscan_seq)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    initial_position = int(lines[0].strip())
    requests = list(map(int, lines[1].strip().split(',')))
    fcfs_sequence = list(map(int, lines[2].strip().split(',')))
    scan_sequence = list(map(int, lines[3].strip().split(',')))
    cscan_sequence = list(map(int, lines[4].strip().split(',')))
    
    return initial_position, requests, fcfs_sequence, scan_sequence, cscan_sequence


def calculate_movements(sequence: List[int]) -> List[int]:
    """
    Calculate cumulative head movements from a sequence.
    
    Args:
        sequence: List of cylinder positions
        
    Returns:
        List of cumulative movements
    """
    movements = [0]
    total = 0
    for i in range(1, len(sequence)):
        movement = abs(sequence[i] - sequence[i-1])
        total += movement
        movements.append(total)
    return movements


def plot_head_movement_over_time(fcfs_seq: List[int], scan_seq: List[int], 
                                  cscan_seq: List[int], output_dir: str = "."):
    """
    Plot head position over time for all three algorithms.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Disk Head Movement Over Time', fontsize=16, fontweight='bold')
    
    algorithms = [
        ("FCFS (First-Come, First-Served)", fcfs_seq, 'blue'),
        ("SCAN (Elevator Algorithm)", scan_seq, 'green'),
        ("C-SCAN (Circular SCAN)", cscan_seq, 'red')
    ]
    
    for idx, (title, sequence, color) in enumerate(algorithms):
        ax = axes[idx]
        
        # Plot every 10th point to avoid overcrowding
        step = max(1, len(sequence) // 200)
        x = list(range(0, len(sequence), step))
        y = [sequence[i] for i in x]
        
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.7)
        ax.scatter(x[::5], y[::5], color=color, s=10, alpha=0.5)
        
        ax.set_xlabel('Request Number', fontsize=10)
        ax.set_ylabel('Cylinder Position', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-100, 5100)
        
        # Add statistics
        total_movement = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
        ax.text(0.02, 0.98, f'Total Movement: {total_movement:,} cylinders', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'head_movement_over_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_cumulative_movement(fcfs_seq: List[int], scan_seq: List[int], 
                             cscan_seq: List[int], output_dir: str = "."):
    """
    Plot cumulative head movement comparison.
    """
    fcfs_movements = calculate_movements(fcfs_seq)
    scan_movements = calculate_movements(scan_seq)
    cscan_movements = calculate_movements(cscan_seq)
    
    plt.figure(figsize=(12, 7))
    
    # Plot every 5th point for clarity
    step = max(1, len(fcfs_movements) // 200)
    
    x_fcfs = list(range(0, len(fcfs_movements), step))
    x_scan = list(range(0, len(scan_movements), step))
    x_cscan = list(range(0, len(cscan_movements), step))
    
    plt.plot(x_fcfs, [fcfs_movements[i] for i in x_fcfs], 
             label='FCFS', color='blue', linewidth=2, alpha=0.7)
    plt.plot(x_scan, [scan_movements[i] for i in x_scan], 
             label='SCAN', color='green', linewidth=2, alpha=0.7)
    plt.plot(x_cscan, [cscan_movements[i] for i in x_cscan], 
             label='C-SCAN', color='red', linewidth=2, alpha=0.7)
    
    plt.xlabel('Request Number', fontsize=12)
    plt.ylabel('Cumulative Head Movement (cylinders)', fontsize=12)
    plt.title('Cumulative Head Movement Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add final values
    plt.text(0.98, 0.02, 
             f'Final Movements:\nFCFS: {fcfs_movements[-1]:,}\nSCAN: {scan_movements[-1]:,}\nC-SCAN: {cscan_movements[-1]:,}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cumulative_movement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_comparison(fcfs_seq: List[int], scan_seq: List[int], 
                                cscan_seq: List[int], num_requests: int, 
                                output_dir: str = "."):
    """
    Create bar charts comparing algorithm performance.
    """
    # Calculate total movements
    fcfs_total = sum(abs(fcfs_seq[i] - fcfs_seq[i-1]) for i in range(1, len(fcfs_seq)))
    scan_total = sum(abs(scan_seq[i] - scan_seq[i-1]) for i in range(1, len(scan_seq)))
    cscan_total = sum(abs(cscan_seq[i] - cscan_seq[i-1]) for i in range(1, len(cscan_seq)))
    
    # Calculate averages
    fcfs_avg = fcfs_total / num_requests
    scan_avg = scan_total / num_requests
    cscan_avg = cscan_total / num_requests
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Comparison', fontsize=16, fontweight='bold')
    
    algorithms = ['FCFS', 'SCAN', 'C-SCAN']
    colors = ['blue', 'green', 'red']
    
    # Total movement comparison
    ax1 = axes[0]
    totals = [fcfs_total, scan_total, cscan_total]
    bars1 = ax1.bar(algorithms, totals, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Head Movement (cylinders)', fontsize=12)
    ax1.set_title('Total Head Movement', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, total in zip(bars1, totals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(total):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Average movement comparison
    ax2 = axes[1]
    averages = [fcfs_avg, scan_avg, cscan_avg]
    bars2 = ax2.bar(algorithms, averages, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Average Movement per Request (cylinders)', fontsize=12)
    ax2.set_title('Average Movement per Request', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, avg in zip(bars2, averages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_efficiency_metrics(fcfs_seq: List[int], scan_seq: List[int], 
                            cscan_seq: List[int], output_dir: str = "."):
    """
    Plot efficiency improvement percentages.
    """
    fcfs_total = sum(abs(fcfs_seq[i] - fcfs_seq[i-1]) for i in range(1, len(fcfs_seq)))
    scan_total = sum(abs(scan_seq[i] - scan_seq[i-1]) for i in range(1, len(scan_seq)))
    cscan_total = sum(abs(cscan_seq[i] - cscan_seq[i-1]) for i in range(1, len(cscan_seq)))
    
    scan_improvement = ((fcfs_total - scan_total) / fcfs_total) * 100
    cscan_improvement = ((fcfs_total - cscan_total) / fcfs_total) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['SCAN vs FCFS', 'C-SCAN vs FCFS']
    improvements = [scan_improvement, cscan_improvement]
    colors = ['green', 'red']
    
    bars = ax.barh(algorithms, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Efficiency Improvement (%)', fontsize=12)
    ax.set_title('Efficiency Improvement Over FCFS', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, improvement in zip(bars, improvements):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{improvement:.2f}%',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'efficiency_improvement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_request_distribution(requests: List[int], output_dir: str = "."):
    """
    Plot the distribution of cylinder requests.
    """
    plt.figure(figsize=(12, 6))
    
    plt.hist(requests, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('Cylinder Number', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Cylinder Requests', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add statistics
    mean = np.mean(requests)
    median = np.median(requests)
    std = np.std(requests)
    
    stats_text = f'Mean: {mean:.1f}\nMedian: {median:.1f}\nStd Dev: {std:.1f}'
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'request_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_seek_distance_distribution(fcfs_seq: List[int], scan_seq: List[int], 
                                    cscan_seq: List[int], output_dir: str = "."):
    """
    Plot the distribution of seek distances for each algorithm.
    """
    # Calculate seek distances
    fcfs_seeks = [abs(fcfs_seq[i] - fcfs_seq[i-1]) for i in range(1, len(fcfs_seq))]
    scan_seeks = [abs(scan_seq[i] - scan_seq[i-1]) for i in range(1, len(scan_seq))]
    cscan_seeks = [abs(cscan_seq[i] - cscan_seq[i-1]) for i in range(1, len(cscan_seq))]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Seek Distance Distribution', fontsize=16, fontweight='bold')
    
    data = [
        ("FCFS", fcfs_seeks, 'blue'),
        ("SCAN", scan_seeks, 'green'),
        ("C-SCAN", cscan_seeks, 'red')
    ]
    
    for idx, (title, seeks, color) in enumerate(data):
        ax = axes[idx]
        
        ax.hist(seeks, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Seek Distance (cylinders)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add statistics
        mean = np.mean(seeks)
        median = np.median(seeks)
        max_seek = max(seeks)
        
        stats_text = f'Mean: {mean:.1f}\nMedian: {median:.1f}\nMax: {max_seek}'
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'seek_distance_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_report(initial_pos: int, requests: List[int], 
                         fcfs_seq: List[int], scan_seq: List[int], 
                         cscan_seq: List[int], output_dir: str = "."):
    """
    Create a comprehensive summary report.
    """
    fcfs_total = sum(abs(fcfs_seq[i] - fcfs_seq[i-1]) for i in range(1, len(fcfs_seq)))
    scan_total = sum(abs(scan_seq[i] - scan_seq[i-1]) for i in range(1, len(scan_seq)))
    cscan_total = sum(abs(cscan_seq[i] - cscan_seq[i-1]) for i in range(1, len(cscan_seq)))
    
    num_requests = len(requests)
    
    report = f"""
{'='*80}
DISK SCHEDULING ALGORITHMS - VISUALIZATION SUMMARY
{'='*80}

Configuration:
  - Total Cylinders: 5,000 (0 to 4,999)
  - Number of Requests: {num_requests:,}
  - Initial Head Position: {initial_pos}

Request Statistics:
  - Mean: {np.mean(requests):.2f}
  - Median: {np.median(requests):.2f}
  - Std Dev: {np.std(requests):.2f}
  - Min: {min(requests)}
  - Max: {max(requests)}

{'='*80}
ALGORITHM PERFORMANCE COMPARISON
{'='*80}

1. FCFS (First-Come, First-Served):
   - Total Head Movement: {fcfs_total:,} cylinders
   - Average per Request: {fcfs_total/num_requests:.2f} cylinders
   - Baseline Algorithm

2. SCAN (Elevator Algorithm):
   - Total Head Movement: {scan_total:,} cylinders
   - Average per Request: {scan_total/num_requests:.2f} cylinders
   - Improvement over FCFS: {((fcfs_total-scan_total)/fcfs_total*100):.2f}%
   - Reduction: {fcfs_total-scan_total:,} cylinders

3. C-SCAN (Circular SCAN):
   - Total Head Movement: {cscan_total:,} cylinders
   - Average per Request: {cscan_total/num_requests:.2f} cylinders
   - Improvement over FCFS: {((fcfs_total-cscan_total)/fcfs_total*100):.2f}%
   - Reduction: {fcfs_total-cscan_total:,} cylinders

{'='*80}
RANKING (Best to Worst by Total Movement)
{'='*80}
"""
    
    results = [
        ("SCAN", scan_total),
        ("C-SCAN", cscan_total),
        ("FCFS", fcfs_total)
    ]
    results.sort(key=lambda x: x[1])
    
    for rank, (name, total) in enumerate(results, 1):
        report += f"{rank}. {name:<10} - {total:>10,} cylinders\n"
    
    report += "\n" + "="*80 + "\n"
    report += "Generated Visualizations:\n"
    report += "  1. head_movement_over_time.png - Shows head position changes\n"
    report += "  2. cumulative_movement.png - Cumulative movement comparison\n"
    report += "  3. performance_comparison.png - Bar charts of performance\n"
    report += "  4. efficiency_improvement.png - Efficiency gains over FCFS\n"
    report += "  5. request_distribution.png - Distribution of requests\n"
    report += "  6. seek_distance_distribution.png - Seek distance patterns\n"
    report += "="*80 + "\n"
    
    output_path = os.path.join(output_dir, 'visualization_summary.txt')
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nSummary report saved to: {output_path}")


def main():
    """Main function to generate all visualizations."""
    
    print("="*80)
    print("DISK SCHEDULING ALGORITHMS - DATA VISUALIZATION")
    print("="*80)
    print()
    
    # Check if sequences file exists
    if not os.path.exists("sequences.txt"):
        print("Error: sequences.txt not found!")
        print("Please run disk_scheduler.py first to generate the data.")
        return
    
    # Load data
    print("Loading sequence data...")
    initial_pos, requests, fcfs_seq, scan_seq, cscan_seq = load_sequences()
    print(f"Loaded {len(requests)} requests")
    print()
    
    # Create output directory for graphs
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}/")
    print()
    
    # Generate all visualizations
    print("Generating visualizations...")
    print("-" * 80)
    
    plot_head_movement_over_time(fcfs_seq, scan_seq, cscan_seq, output_dir)
    plot_cumulative_movement(fcfs_seq, scan_seq, cscan_seq, output_dir)
    plot_performance_comparison(fcfs_seq, scan_seq, cscan_seq, len(requests), output_dir)
    plot_efficiency_metrics(fcfs_seq, scan_seq, cscan_seq, output_dir)
    plot_request_distribution(requests, output_dir)
    plot_seek_distance_distribution(fcfs_seq, scan_seq, cscan_seq, output_dir)
    
    print("-" * 80)
    print()
    
    # Create summary report
    create_summary_report(initial_pos, requests, fcfs_seq, scan_seq, cscan_seq, output_dir)
    
    print()
    print("="*80)
    print("All visualizations completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
