#!/usr/bin/env python
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from lsm import TraditionalLSMTree, Constants
from lsm.learned_bloom import LearnedBloomTree, LEARNED_BLOOM_AVAILABLE
import pickle
import struct
from typing import List, Dict, Tuple, Optional

# Set up nice visualization styling
plt.style.use('ggplot')
sns.set_palette("colorblind")

def create_workload_patterns(num_records=1000000):
    """Create various workload patterns for comprehensive benchmarking."""
    print(f"Generating diverse workload patterns from {num_records:,} records...")
    
    # Generate base key-value pairs
    keys = np.linspace(0, 1000, num_records)
    values = [f"value-{i}" for i in range(num_records)]
    
    # Shuffle to simulate random write pattern
    indices = list(range(num_records))
    random.shuffle(indices)
    shuffled_keys = [keys[i] for i in indices]
    shuffled_values = [values[i] for i in indices]
    
    # Create key-value pairs
    data = list(zip(shuffled_keys, shuffled_values))
    
    # Split into write and different read patterns
    write_data = data[:int(num_records * 0.8)]  # 80% for writes
    
    # Create various read workloads
    workloads = {}
    
    # 1. Random reads (existing keys)
    random_reads = data[int(num_records * 0.8):]  # 20% for reads
    workloads["random"] = random_reads
    
    # 2. Sequential reads
    seq_data = sorted(data[int(num_records * 0.8):], key=lambda x: x[0])
    workloads["sequential"] = seq_data
    
    # 3. Level-specific reads (we'll create small, medium, large keys)
    # Small keys likely in higher levels (more recent)
    small_keys = [(key, val) for key, val in data if key < 300]
    if small_keys:
        workloads["level1_heavy"] = random.sample(small_keys, min(len(small_keys), len(random_reads)))
    
    # Medium keys
    medium_keys = [(key, val) for key, val in data if 300 <= key < 700]
    if medium_keys:
        workloads["level2_heavy"] = random.sample(medium_keys, min(len(medium_keys), len(random_reads)))
    
    # Large keys likely in lower levels (older data)
    large_keys = [(key, val) for key, val in data if key >= 700]
    if large_keys:
        workloads["level3_heavy"] = random.sample(large_keys, min(len(large_keys), len(random_reads)))
    
    # 4. Mix of existing and missing keys
    missing_keys = np.linspace(1001, 2000, len(random_reads) // 5)  # 20% missing keys
    missing_data = [(key, None) for key in missing_keys]
    mixed_data = random_reads[:int(len(random_reads) * 0.8)] + missing_data
    random.shuffle(mixed_data)
    workloads["mixed"] = mixed_data
    
    # 5. Hotspot access pattern - 20% of keys are accessed 80% of the time
    hotspot_base = random.sample(data[int(num_records * 0.8):], min(len(data[int(num_records * 0.8):]), 100))
    hotspot_reads = []
    for _ in range(len(random_reads) * 4 // 5):  # 80% of reads
        hotspot_reads.append(random.choice(hotspot_base))
    for _ in range(len(random_reads) // 5):  # 20% of reads
        hotspot_reads.append(random.choice(random_reads))
    workloads["hotspot"] = hotspot_reads[:len(random_reads)]
    
    print(f"Created {len(workloads)} different workload patterns")
    for name, workload in workloads.items():
        print(f"  - {name}: {len(workload)} operations")
    
    return write_data, workloads

def benchmark_traditional_tree(write_data, read_workloads, data_dir="benchmark_data/traditional"):
    """Benchmark traditional LSM tree with standard bloom filters."""
    print("\nBenchmarking traditional LSM tree...")
    
    # Clean up data directory
    if os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create traditional LSM tree
    tree = TraditionalLSMTree(
        data_dir=data_dir,
        buffer_size=1 * 1024 * 1024,  # 1MB
        size_ratio=10,
        base_fpr=0.01  # 1% base FPR
    )
    
    # Benchmark writes
    print("Writing data...")
    write_start = time.perf_counter()
    for key, value in write_data:
        tree.put(key, value)
    write_end = time.perf_counter()
    write_time = write_end - write_start
    
    # Force flush and compaction
    flush_start = time.perf_counter()
    tree.flush_data()  # Use flush_data() instead of flush()
    flush_end = time.perf_counter()
    flush_time = flush_end - flush_start
    
    # Benchmark different read workloads
    workload_results = {}
    
    for workload_name, read_data in read_workloads.items():
        print(f"Reading data with {workload_name} pattern...")
        hits = 0
        misses = 0
        read_start = time.perf_counter()
        results = []  # Store results for correctness verification
        
        for key, expected_value in read_data:
            value = tree.get(key)
            results.append((key, value))
            
            if expected_value is None:
                if value is None:
                    misses += 1
                else:
                    # Found a value when we shouldn't
                    pass
            else:
                if value == expected_value:
                    hits += 1
                else:
                    # Got wrong value
                    pass
        
        read_end = time.perf_counter()
        read_time = read_end - read_start
        
        # Calculate statistics
        num_reads = len(read_data)
        read_ops_per_sec = num_reads / read_time if read_time > 0 else 0
        
        workload_results[workload_name] = {
            'read_time': read_time,
            'read_ops_per_sec': read_ops_per_sec,
            'num_reads': num_reads,
            'hits': hits,
            'misses': misses,
            'results': results,  # Store for correctness verification
        }
        
        print(f"  {workload_name} - Read: {read_time:.2f}s, Reads/sec: {read_ops_per_sec:.2f}, Hit rate: {hits/num_reads:.2%}")
    
    # Get tree stats
    stats = tree.get_stats()
    
    results = {
        'write_time': write_time,
        'flush_time': flush_time,
        'write_ops_per_sec': len(write_data) / write_time if write_time > 0 else 0,
        'num_writes': len(write_data),
        'workloads': workload_results,
        'tree_stats': stats
    }
    
    print(f"Traditional tree - Write: {write_time:.2f}s")
    print(f"Writes/sec: {results['write_ops_per_sec']:.2f}")
    
    return results

def benchmark_learned_tree(write_data, read_workloads, data_dir="benchmark_data/learned", bloom_threshold=0.5):
    """Benchmark LSM tree with learned bloom filters."""
    if not LEARNED_BLOOM_AVAILABLE:
        print("Learned bloom filters not available, skipping benchmark")
        return None
        
    print("\nBenchmarking learned bloom filter LSM tree...")
    
    # Clean up data directory
    if os.path.exists(data_dir):
        import shutil
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create learned bloom filter LSM tree
    tree = LearnedBloomTree(
        data_dir=data_dir,
        buffer_size=1 * 1024 * 1024,  # 1MB
        size_ratio=10,
        base_fpr=0.01,  # 1% base FPR
        bloom_threshold=bloom_threshold
    )
    
    # Benchmark writes
    print("Writing data...")
    write_start = time.perf_counter()
    for key, value in write_data:
        tree.put(key, value)
    write_end = time.perf_counter()
    write_time = write_end - write_start
    
    # Force flush and compaction
    flush_start = time.perf_counter()
    tree.flush_data()  # Use flush_data() instead of flush()
    flush_end = time.perf_counter()
    flush_time = flush_end - flush_start
    
    # Wait for ML models to train
    print("Training bloom filter models...")
    train_start = time.perf_counter()
    tree.train_bloom_filters()
    train_end = time.perf_counter()
    train_time = train_end - train_start
    
    # Benchmark different read workloads
    workload_results = {}
    
    for workload_name, read_data in read_workloads.items():
        print(f"Reading data with {workload_name} pattern...")
        hits = 0
        misses = 0
        read_start = time.perf_counter()
        results = []  # Store results for correctness verification
        
        for key, expected_value in read_data:
            value = tree.get(key)
            results.append((key, value))
            
            if expected_value is None:
                if value is None:
                    misses += 1
                else:
                    # Found a value when we shouldn't
                    pass
            else:
                if value == expected_value:
                    hits += 1
                else:
                    # Got wrong value
                    pass
        
        read_end = time.perf_counter()
        read_time = read_end - read_start
        
        # Calculate statistics
        num_reads = len(read_data)
        read_ops_per_sec = num_reads / read_time if read_time > 0 else 0
        
        workload_results[workload_name] = {
            'read_time': read_time,
            'read_ops_per_sec': read_ops_per_sec,
            'num_reads': num_reads,
            'hits': hits,
            'misses': misses,
            'results': results,  # Store for correctness verification
        }
        
        print(f"  {workload_name} - Read: {read_time:.2f}s, Reads/sec: {read_ops_per_sec:.2f}, Hit rate: {hits/num_reads:.2%}")
    
    # Get tree stats
    stats = tree.get_stats()
    
    results = {
        'write_time': write_time,
        'flush_time': flush_time,
        'train_time': train_time,
        'write_ops_per_sec': len(write_data) / write_time if write_time > 0 else 0,
        'num_writes': len(write_data),
        'workloads': workload_results,
        'tree_stats': stats,
        'bloom_threshold': bloom_threshold
    }
    
    print(f"Learned tree - Write: {write_time:.2f}s, Train: {train_time:.2f}s")
    print(f"Writes/sec: {results['write_ops_per_sec']:.2f}")
    
    # Print learned bloom filter stats
    if 'learned_bloom' in stats:
        learned_bloom = stats['learned_bloom']
        print("\nLearned Bloom Filter Statistics:")
        print(f"Training events: {learned_bloom['training_events']}")
        print(f"Average training time: {learned_bloom['avg_training_time']:.4f}s")
        print(f"False negative rate: {learned_bloom['false_negative_rate']:.4%}")
        print(f"Learned bypasses: {learned_bloom['learned_bypasses']} of {learned_bloom['learned_checks']} checks ({learned_bloom['learned_bypass_rate']:.2%})")
        print(f"Traditional bypasses: {learned_bloom['traditional_bypasses']} of {learned_bloom['traditional_checks']} checks ({learned_bloom['traditional_bypass_rate']:.2%})")
    
    return results

def verify_correctness(traditional_results, learned_results):
    """Verify that both implementations return the same results."""
    if learned_results is None:
        print("No learned bloom filter results to verify")
        return True
    
    all_correct = True
    print("\nVerifying correctness across workloads...")
    
    for workload_name in traditional_results['workloads'].keys():
        trad_results = dict(traditional_results['workloads'][workload_name]['results'])
        learned_results_dict = dict(learned_results['workloads'][workload_name]['results'])
        
        mismatches = 0
        for key, trad_value in trad_results.items():
            if key in learned_results_dict:
                if trad_value != learned_results_dict[key]:
                    mismatches += 1
                    all_correct = False
        
        if mismatches > 0:
            print(f"❌ {workload_name}: {mismatches} result mismatches detected!")
        else:
            print(f"✅ {workload_name}: All results match")
    
    if all_correct:
        print("All results match between traditional and learned implementations!")
    else:
        print("Warning: Some results differ between implementations")
    
    return all_correct

def plot_comparison(traditional_results, learned_results, output_dir="benchmark_plots"):
    """Plot comprehensive comparison between traditional and learned bloom filters."""
    if learned_results is None:
        print("No learned bloom filter results to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Read throughput comparison across workloads
    workloads = list(traditional_results['workloads'].keys())
    trad_read_ops = [traditional_results['workloads'][w]['read_ops_per_sec'] for w in workloads]
    learned_read_ops = [learned_results['workloads'][w]['read_ops_per_sec'] for w in workloads]
    
    plt.figure(figsize=(12, 7))
    bar_width = 0.35
    x = np.arange(len(workloads))
    
    plt.bar(x - bar_width/2, trad_read_ops, bar_width, label='Traditional Bloom Filter')
    plt.bar(x + bar_width/2, learned_read_ops, bar_width, label='Learned Bloom Filter')
    
    plt.xlabel('Workload Pattern', fontsize=12)
    plt.ylabel('Operations per Second', fontsize=12)
    plt.title('Read Throughput Comparison Across Workloads', fontsize=14)
    plt.xticks(x, workloads, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'read_throughput_comparison.png'), dpi=300)
    
    # 2. Speedup ratio across workloads
    speedups = [l/t if t > 0 else 0 for l, t in zip(learned_read_ops, trad_read_ops)]
    
    plt.figure(figsize=(12, 7))
    plt.bar(workloads, speedups, color=sns.color_palette("YlOrRd", len(speedups)))
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
    plt.xlabel('Workload Pattern', fontsize=12)
    plt.ylabel('Speedup Ratio (Learned/Traditional)', fontsize=12)
    plt.title('Read Performance Improvement with Learned Bloom Filters', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_ratio.png'), dpi=300)
    
    # 3. Combined latency comparison line chart
    plt.figure(figsize=(12, 7))
    
    # Sort workloads by traditional latency for better visualization
    workload_latencies = [(w, traditional_results['workloads'][w]['read_time']) for w in workloads]
    workload_latencies.sort(key=lambda x: x[1])
    sorted_workloads = [w[0] for w in workload_latencies]
    
    trad_times = [traditional_results['workloads'][w]['read_time'] for w in sorted_workloads]
    learned_times = [learned_results['workloads'][w]['read_time'] for w in sorted_workloads]
    
    plt.plot(sorted_workloads, trad_times, marker='o', linewidth=2, label='Traditional Bloom Filter')
    plt.plot(sorted_workloads, learned_times, marker='s', linewidth=2, label='Learned Bloom Filter')
    
    plt.xlabel('Workload Pattern', fontsize=12)
    plt.ylabel('Total Read Time (s)', fontsize=12)
    plt.title('Read Latency Comparison', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300)
    
    # 4. Memory usage comparison
    if 'tree_stats' in traditional_results and 'tree_stats' in learned_results:
        # Extract bloom filter sizes from stats
        trad_sizes = []
        learned_sizes = []
        level_labels = []
        
        trad_stats = traditional_results['tree_stats']
        learned_stats = learned_results['tree_stats']
        
        for level in range(1, 4):  # Levels 1-3
            level_key = f'level_{level}'
            if level_key in trad_stats and level_key in learned_stats:
                trad_level = trad_stats[level_key]
                learned_level = learned_stats[level_key]
                
                if 'bloom_size_bytes' in trad_level and 'bloom_size_bytes' in learned_level:
                    trad_sizes.append(trad_level['bloom_size_bytes'] / 1024)  # KB
                    learned_sizes.append(learned_level['bloom_size_bytes'] / 1024)  # KB
                    level_labels.append(f'Level {level}')
        
        if level_labels:
            plt.figure(figsize=(10, 6))
            x = np.arange(len(level_labels))
            width = 0.35
            
            plt.bar(x - width/2, trad_sizes, width, label='Traditional Bloom Filter')
            plt.bar(x + width/2, learned_sizes, width, label='Learned Bloom Filter')
            
            plt.xlabel('LSM Tree Level', fontsize=12)
            plt.ylabel('Bloom Filter Size (KB)', fontsize=12)
            plt.title('Memory Usage Comparison by Level', fontsize=14)
            plt.xticks(x, level_labels)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), dpi=300)
    
    # 5. Multi-metric radar chart
    if workloads:
        metrics = ['Read Throughput', 'Memory Usage', 'Write Throughput']
        
        # Normalize metrics between 0 and 1 for radar chart
        trad_write_ops = traditional_results['write_ops_per_sec']
        learned_write_ops = learned_results['write_ops_per_sec']
        
        # Average read throughput across workloads
        trad_avg_read = sum(trad_read_ops) / len(trad_read_ops)
        learned_avg_read = sum(learned_read_ops) / len(learned_read_ops)
        
        # Memory usage (if available)
        trad_memory = sum(trad_sizes) if 'trad_sizes' in locals() else 0
        learned_memory = sum(learned_sizes) if 'learned_sizes' in locals() else 0
        
        # Normalize: higher is better for our display
        max_read = max(trad_avg_read, learned_avg_read)
        max_write = max(trad_write_ops, learned_write_ops)
        max_memory = max(trad_memory, learned_memory) if trad_memory > 0 and learned_memory > 0 else 1
        
        trad_metrics = [
            trad_avg_read / max_read if max_read > 0 else 0,
            (max_memory - trad_memory) / max_memory if max_memory > 0 else 0,  # Invert so less memory is better
            trad_write_ops / max_write if max_write > 0 else 0
        ]
        
        learned_metrics = [
            learned_avg_read / max_read if max_read > 0 else 0,
            (max_memory - learned_memory) / max_memory if max_memory > 0 else 0,  # Invert so less memory is better
            learned_write_ops / max_write if max_write > 0 else 0
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        trad_metrics += trad_metrics[:1]  # Close the polygon
        learned_metrics += learned_metrics[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, trad_metrics, 'b-', linewidth=2, label='Traditional')
        ax.fill(angles, trad_metrics, 'b', alpha=0.1)
        
        ax.plot(angles, learned_metrics, 'r-', linewidth=2, label='Learned')
        ax.fill(angles, learned_metrics, 'r', alpha=0.1)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Performance Metrics Comparison (Higher is Better)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300)
    
    # 6. Bloom filter metrics by level
    if 'learned_bloom' in learned_results.get('tree_stats', {}):
        learned_bloom_stats = learned_results['tree_stats']['learned_bloom']
        
        # Check for level-specific metrics
        level_metrics = {}
        for key, value in learned_bloom_stats.items():
            if key.startswith('level_') and isinstance(value, dict):
                level = key.split('_')[1]
                level_metrics[level] = value
        
        if level_metrics:
            # Extract metrics by level
            levels = sorted(level_metrics.keys())
            accuracies = []
            fprs = []
            fnrs = []
            
            for level in levels:
                metrics = level_metrics[level]
                if 'accuracy' in metrics:
                    accuracies.append(metrics['accuracy'])
                else:
                    accuracies.append(0)
                
                if 'fpr' in metrics:
                    fprs.append(metrics['fpr'])
                else:
                    fprs.append(0)
                
                if 'fnr' in metrics:
                    fnrs.append(metrics['fnr'])
                else:
                    fnrs.append(0)
            
            # Create grouped bar chart for bloom filter metrics
            plt.figure(figsize=(12, 7))
            x = np.arange(len(levels))
            width = 0.25
            
            plt.bar(x - width, accuracies, width, label='Accuracy')
            plt.bar(x, fprs, width, label='False Positive Rate')
            plt.bar(x + width, fnrs, width, label='False Negative Rate')
            
            plt.xlabel('Tree Level', fontsize=12)
            plt.ylabel('Rate', fontsize=12)
            plt.title('Learned Bloom Filter Metrics by Level', fontsize=14)
            plt.xticks(x, [f'Level {level}' for level in levels])
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bloom_metrics_by_level.png'), dpi=300)
    
    print(f"Plots saved to {output_dir}/")

def main():
    """Main benchmark function."""
    print("Starting LSM Tree Bloom Filter benchmark...")
    
    # Create test data with different workload patterns
    write_data, read_workloads = create_workload_patterns(num_records=100000)
    
    # Benchmark traditional tree
    traditional_results = benchmark_traditional_tree(write_data, read_workloads)
    
    # Benchmark learned bloom filter tree with different thresholds
    learned_results = benchmark_learned_tree(write_data, read_workloads, bloom_threshold=0.5)
    
    # Verify correctness
    if learned_results:
        verify_correctness(traditional_results, learned_results)
    
    # Plot comparison
    plot_comparison(traditional_results, learned_results)
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main() 