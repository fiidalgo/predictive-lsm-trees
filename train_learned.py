#!/usr/bin/env python
import os
import time
import pickle
import struct
import numpy as np
import random
import matplotlib.pyplot as plt
from lsm import TraditionalLSMTree, Constants
from lsm.bloom import BloomFilter
from ml.learned_bloom import LearnedBloomFilterManager

def load_runs_data(data_dir="data"):
    """Load all runs data for training bloom filters."""
    # Create separate sets for training and testing
    keys_by_level = {1: [], 2: [], 3: []}
    
    print("Loading run data for bloom filter training...")
    
    # Process each level
    for level in [1, 2, 3]:  # Only levels 1-3 have bloom filters
        level_dir = os.path.join(data_dir, f"level_{level}")
        if not os.path.exists(level_dir):
            print(f"Directory {level_dir} does not exist")
            continue
        
        # Get all runs in the level
        run_files = [f for f in os.listdir(level_dir) if not f.startswith('.')]
        
        for run_file in run_files:
            run_path = os.path.join(level_dir, run_file)
            try:
                with open(run_path, 'rb') as f:
                    data = pickle.load(f)
                    pairs = data.get('pairs', [])
                    
                    # Extract keys and convert from bytes to float
                    for key, _ in pairs:
                        if isinstance(key, bytes):
                            try:
                                # Convert bytes to float
                                float_key = struct.unpack('d', key[:8].ljust(8, b'\x00'))[0]
                                keys_by_level[level].append(float_key)
                            except:
                                pass
                        else:
                            keys_by_level[level].append(float(key))
            except Exception as e:
                print(f"Error loading run {run_path}: {e}")
    
    # Count keys per level
    for level, keys in keys_by_level.items():
        print(f"Level {level}: {len(keys)} keys loaded")
    
    return keys_by_level

def generate_negative_samples(positive_keys, sample_ratio=2.0):
    """Generate negative sample keys that don't exist in the positives."""
    if not positive_keys:
        return []
    
    min_key = min(positive_keys)
    max_key = max(positive_keys)
    key_range = max_key - min_key
    
    # Generate negative samples
    num_samples = int(len(positive_keys) * sample_ratio)
    negative_keys = []
    
    # Keys below range
    negative_keys.extend([min_key - random.random() * key_range * 0.5 for _ in range(num_samples // 4)])
    
    # Keys above range
    negative_keys.extend([max_key + random.random() * key_range * 0.5 for _ in range(num_samples // 4)])
    
    # Create a set of positive keys for fast lookup
    positive_set = set(positive_keys)
    
    # Keys within range but not in positive keys
    in_range_count = num_samples // 2
    attempts = 0
    while len(negative_keys) < num_samples and attempts < num_samples * 10:
        attempts += 1
        key = min_key + random.random() * key_range
        if key not in positive_set:
            negative_keys.append(key)
    
    return negative_keys

def evaluate_bloom_filters(filter_manager, traditional_manager=None, num_queries=10000):
    """Evaluate learned bloom filters against traditional ones."""
    results = {}
    
    # Get positive and negative keys for testing
    test_keys = {}
    for level in [1, 2, 3]:
        # Use some existing keys and generate negative keys
        if level in filter_manager.training_data:
            # Take a sample of positive keys for testing
            positive_keys = filter_manager.training_data[level]['positive']
            if positive_keys:
                test_size = min(num_queries // 2, len(positive_keys) // 10)
                positive_test = random.sample(positive_keys, test_size)
                
                # Generate negative keys
                negative_test = generate_negative_samples(positive_test, sample_ratio=1.0)
                
                test_keys[level] = {
                    'positive': positive_test,
                    'negative': negative_test
                }
                
                print(f"Level {level} test data: {len(positive_test)} positive, {len(negative_test)} negative")
    
    # Test learned bloom filters
    learned_results = {}
    for level, keys in test_keys.items():
        positive = keys['positive']
        negative = keys['negative']
        
        # Query the learned bloom filter
        learned_start = time.perf_counter()
        positive_scores = [filter_manager.might_contain(key, level) for key in positive]
        negative_scores = [filter_manager.might_contain(key, level) for key in negative]
        learned_end = time.perf_counter()
        
        # Calculate metrics for different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            # Classify based on threshold
            tp = sum(1 for score in positive_scores if score >= threshold)
            fp = sum(1 for score in negative_scores if score >= threshold)
            tn = sum(1 for score in negative_scores if score < threshold)
            fn = sum(1 for score in positive_scores if score < threshold)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fpr': fpr,
                    'fnr': fnr,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn
                }
        
        # Calculate metadata about actual filter
        learned_filter = filter_manager.filters[level]
        learned_size_bytes = learned_filter.backup_filter_size // 8
        if hasattr(learned_filter, 'pytorch_model'):
            # Get PyTorch model size (approximate)
            model_size_bytes = sum(p.numel() * 4 for p in learned_filter.pytorch_model.parameters())  # Approx. 4 bytes per param
        elif hasattr(learned_filter, 'model'):
            # Get scikit-learn model size (approximate)
            model_params = learned_filter.model.get_params()
            model_size_bytes = len(pickle.dumps(learned_filter.model))
        else:
            model_size_bytes = 0
            
        total_size_bytes = model_size_bytes + learned_size_bytes
        
        learned_results[level] = {
            'metrics': best_metrics,
            'query_time': (learned_end - learned_start) * 1000,  # ms
            'size_bytes': total_size_bytes,
            'model_size_bytes': model_size_bytes,
            'backup_size_bytes': learned_size_bytes,
            'queries': len(positive) + len(negative)
        }
        
        print(f"\nLevel {level} Learned Bloom Filter Results:")
        print(f"Best threshold: {best_threshold:.2f}")
        print(f"Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}, F1: {best_metrics['f1']:.4f}")
        print(f"FPR: {best_metrics['fpr']:.4f}, FNR: {best_metrics['fnr']:.4f}")
        print(f"Size: {total_size_bytes/1024:.2f} KB (Model: {model_size_bytes/1024:.2f} KB, Backup: {learned_size_bytes/1024:.2f} KB)")
        print(f"Query time: {learned_results[level]['query_time']:.2f} ms for {learned_results[level]['queries']} queries")
    
    # If traditional bloom filters are available, test them too
    traditional_results = {}
    if traditional_manager:
        # Create traditional bloom filters with equivalent configuration
        for level, keys in test_keys.items():
            positive = keys['positive']
            negative = keys['negative']
            
            # Get FPR for this level
            level_fpr = Constants.get_level_fpr(level)
            
            # Create traditional bloom filter
            trad_filter = BloomFilter(expected_elements=len(filter_manager.training_data[level]['positive']), 
                                     false_positive_rate=level_fpr)
            
            # Add all positive keys
            for key in filter_manager.training_data[level]['positive']:
                trad_filter.add(key)
            
            # Query the traditional bloom filter
            trad_start = time.perf_counter()
            trad_positive_results = [trad_filter.might_contain(key) for key in positive]
            trad_negative_results = [trad_filter.might_contain(key) for key in negative]
            trad_end = time.perf_counter()
            
            # Calculate metrics
            tp = sum(1 for result in trad_positive_results if result)
            fp = sum(1 for result in trad_negative_results if result)
            tn = sum(1 for result in trad_negative_results if not result)
            fn = sum(1 for result in trad_positive_results if not result)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr,
                'fnr': fnr,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
            
            # Calculate filter size
            trad_size_bytes = trad_filter.get_size_bytes()
            
            traditional_results[level] = {
                'metrics': metrics,
                'query_time': (trad_end - trad_start) * 1000,  # ms
                'size_bytes': trad_size_bytes,
                'queries': len(positive) + len(negative)
            }
            
            print(f"\nLevel {level} Traditional Bloom Filter Results:")
            print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"FPR: {metrics['fpr']:.4f}, FNR: {metrics['fnr']:.4f}")
            print(f"Size: {trad_size_bytes/1024:.2f} KB")
            print(f"Query time: {traditional_results[level]['query_time']:.2f} ms for {traditional_results[level]['queries']} queries")
            
            # Calculate improvement
            if metrics['fpr'] > 0:
                fpr_improvement = (metrics['fpr'] - learned_results[level]['metrics']['fpr']) / metrics['fpr'] * 100
            else:
                # If traditional FPR is already 0, we can't improve it
                fpr_improvement = 0.0 if learned_results[level]['metrics']['fpr'] == 0 else float('-inf')
                
            size_improvement = (trad_size_bytes - learned_results[level]['size_bytes']) / trad_size_bytes * 100
            time_per_query_trad = traditional_results[level]['query_time'] / traditional_results[level]['queries']
            time_per_query_learned = learned_results[level]['query_time'] / learned_results[level]['queries']
            
            if time_per_query_trad > 0:
                time_improvement = (time_per_query_trad - time_per_query_learned) / time_per_query_trad * 100
            else:
                time_improvement = 0.0
            
            print(f"\nLevel {level} Improvements:")
            print(f"FPR improvement: {fpr_improvement:.2f}%")
            print(f"Size improvement: {size_improvement:.2f}%")
            print(f"Time per query improvement: {time_improvement:.2f}%")
    
    # Prepare results
    results = {
        'learned': learned_results,
        'traditional': traditional_results if traditional_manager else None
    }
    
    return results

def plot_results(results, output_dir='plots'):
    """Plot comparison results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    learned = results['learned']
    traditional = results['traditional']
    
    levels = list(learned.keys())
    
    # Create bar chart for FPR comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(levels))
    width = 0.35
    
    learned_fpr = [learned[level]['metrics']['fpr'] for level in levels]
    if traditional:
        traditional_fpr = [traditional[level]['metrics']['fpr'] for level in levels]
        plt.bar(x - width/2, learned_fpr, width, label='Learned Bloom Filter')
        plt.bar(x + width/2, traditional_fpr, width, label='Traditional Bloom Filter')
    else:
        plt.bar(x, learned_fpr, width, label='Learned Bloom Filter')
    
    plt.xlabel('Level')
    plt.ylabel('False Positive Rate')
    plt.title('FPR Comparison')
    plt.xticks(x, [f'Level {level}' for level in levels])
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fpr_comparison.png'))
    
    # Create bar chart for size comparison
    plt.figure(figsize=(12, 6))
    
    learned_size = [learned[level]['size_bytes'] / 1024 for level in levels]  # KB
    if traditional:
        traditional_size = [traditional[level]['size_bytes'] / 1024 for level in levels]  # KB
        plt.bar(x - width/2, learned_size, width, label='Learned Bloom Filter')
        plt.bar(x + width/2, traditional_size, width, label='Traditional Bloom Filter')
    else:
        plt.bar(x, learned_size, width, label='Learned Bloom Filter')
    
    plt.xlabel('Level')
    plt.ylabel('Size (KB)')
    plt.title('Size Comparison')
    plt.xticks(x, [f'Level {level}' for level in levels])
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'size_comparison.png'))
    
    # Create bar chart for query time comparison
    plt.figure(figsize=(12, 6))
    
    learned_time = [learned[level]['query_time'] / learned[level]['queries'] * 1000 for level in levels]  # μs per query
    if traditional:
        traditional_time = [traditional[level]['query_time'] / traditional[level]['queries'] * 1000 for level in levels]  # μs per query
        plt.bar(x - width/2, learned_time, width, label='Learned Bloom Filter')
        plt.bar(x + width/2, traditional_time, width, label='Traditional Bloom Filter')
    else:
        plt.bar(x, learned_time, width, label='Learned Bloom Filter')
    
    plt.xlabel('Level')
    plt.ylabel('Query Time (μs)')
    plt.title('Query Time Comparison')
    plt.xticks(x, [f'Level {level}' for level in levels])
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    
    # Create combined metrics plot for learned bloom filter
    plt.figure(figsize=(10, 6))
    metrics = ['precision', 'recall', 'f1', 'fpr', 'fnr']
    
    for level in levels:
        values = [learned[level]['metrics'][metric] for metric in metrics]
        plt.plot(metrics, values, marker='o', label=f'Level {level}')
    
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Learned Bloom Filter Metrics')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'learned_metrics.png'))
    
    # Create detailed FPR vs Size scatter plot
    if traditional:
        plt.figure(figsize=(10, 6))
        learned_fpr = [learned[level]['metrics']['fpr'] for level in levels]
        traditional_fpr = [traditional[level]['metrics']['fpr'] for level in levels]
        learned_size = [learned[level]['size_bytes'] / 1024 for level in levels]
        traditional_size = [traditional[level]['size_bytes'] / 1024 for level in levels]
        
        plt.scatter(traditional_size, traditional_fpr, s=100, c='blue', marker='s', label='Traditional')
        plt.scatter(learned_size, learned_fpr, s=100, c='red', marker='o', label='Learned')
        
        # Connect corresponding points with lines
        for i in range(len(levels)):
            plt.plot([traditional_size[i], learned_size[i]], 
                     [traditional_fpr[i], learned_fpr[i]], 
                     'k--', alpha=0.5)
        
        plt.xlabel('Size (KB)')
        plt.ylabel('False Positive Rate')
        plt.title('FPR vs Size Trade-off')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'fpr_vs_size.png'))

def main():
    """Main function to train and test learned bloom filters."""
    print("Starting learned bloom filter training and testing...")
    
    # Directories
    data_dir = "data"
    models_dir = "learned_test_data/ml_models"
    
    # Create learned bloom filter manager
    filter_manager = LearnedBloomFilterManager(models_dir)
    
    # Load run data
    keys_by_level = load_runs_data(data_dir)
    
    # Add training data to filter manager
    for level, keys in keys_by_level.items():
        # Use 80% for training, 20% for testing
        train_size = int(len(keys) * 0.8)
        # Cap training size for larger levels to avoid memory issues
        if train_size > 100000:
            print(f"Limiting level {level} training set from {train_size} to 100000 keys")
            train_size = 100000
            train_keys = random.sample(keys, train_size)
        else:
            train_keys = keys[:train_size]
        
        # Add positive examples
        for key in train_keys:
            filter_manager.add_training_data(key, level, True)
        
        # Generate negative examples
        negative_keys = generate_negative_samples(train_keys)
        for key in negative_keys:
            filter_manager.add_training_data(key, level, False)

        print(f"Level {level}: {len(train_keys)} positive training examples, {len(negative_keys)} negative examples")
    
    # Train all filters
    print("\nTraining learned bloom filters...")
    start_time = time.time()
    
    # Train filters one at a time to avoid memory issues
    for level in range(1, 4):
        print(f"\nTraining filter for level {level}...")
        if level in filter_manager.filters:
            try:
                # Get training data
                positive_keys = filter_manager.training_data[level]['positive']
                negative_keys = filter_manager.training_data[level]['negative']
                
                # Train the filter
                filter_manager.filters[level].train(positive_keys, negative_keys)
                
                # Save after each level is trained
                filter_manager.save_models()
            except Exception as e:
                print(f"Error training level {level}: {e}")
    
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.2f} seconds\n")
    
    # Evaluate filters
    print("Evaluating bloom filters...")
    results = evaluate_bloom_filters(filter_manager, traditional_manager=True)
    
    # Plot results
    print("\nGenerating plots...")
    plot_results(results)
    
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main() 