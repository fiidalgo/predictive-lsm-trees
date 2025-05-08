import os
import time
import random
import statistics
import logging
import numpy as np
import pickle
import struct
import matplotlib.pyplot as plt
from lsm import TraditionalLSMTree, LearnedLSMTree, Constants

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)

# Global counters for metrics tracking
class MetricsCounters:
    # Bloom filter metrics
    bloom_checks = 0  # Number of bloom filters queried
    bloom_bypasses = 0  # Number of bloom filters not queried (bypassed)
    
    # Disk metrics (actual)
    disk_reads = 0 
    disk_writes = 0
    
    # Page access metrics
    learned_pages_checked = 0
    traditional_pages_checked = 0
    
    # Lookup tracking
    lookups_attempted = 0
    false_negatives = 0  # When learned model misses a key that exists
    
    @classmethod
    def reset(cls):
        """Reset all counters to zero."""
        cls.bloom_checks = 0
        cls.bloom_bypasses = 0
        cls.disk_reads = 0
        cls.disk_writes = 0
        cls.learned_pages_checked = 0
        cls.traditional_pages_checked = 0
        cls.lookups_attempted = 0
        cls.false_negatives = 0
        
        # Re-register with learned module to ensure it has the reset counters
        try:
            from lsm.learned import register_metrics
            register_metrics(cls)
        except ImportError:
            # Skip if we can't import yet (during initialization)
            pass

class PerformanceTest:
    def __init__(self):
        print("Initializing test environment...")
        
        # Common data directory
        self.data_dir = "data"
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print("Error: Data directory doesn't exist or is empty.")
            print("Please run load_data.py first to create the test dataset.")
            exit(1)
        
        # Constants for key/value sizes
        self.KEY_SIZE_BYTES = 16
        self.VALUE_SIZE_BYTES = 100
        self.ENTRY_SIZE_BYTES = self.KEY_SIZE_BYTES + self.VALUE_SIZE_BYTES
        
        # Buffer and level sizes
        self.BUFFER_SIZE_BYTES = 1 * 1024 * 1024  # 1MB
        self.LEVEL_SIZE_RATIO = 10  # Each level is 10x larger than previous
        
        # Calculate approx entries per level based on entry size
        self.entries_per_buffer = self.BUFFER_SIZE_BYTES // self.ENTRY_SIZE_BYTES
        
        print("Loading existing data from data directory...")
        
        # Initialize traditional tree to use the existing data
        self.traditional_tree = TraditionalLSMTree(
            data_dir=self.data_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01
        )
        
        # Create learned tree in memory
        self.learned_dir = "learned_test_data"
        os.makedirs(self.learned_dir, exist_ok=True)
        
        # Copy data structure from traditional to learned if needed
        print("Copying data structure to learned tree...")
        self._copy_tree_structure(self.data_dir, self.learned_dir)
        
        # Initialize the learned tree with the copied data
        self.learned_tree = LearnedLSMTree(
            data_dir=self.learned_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01,
            train_every=4,
            validation_rate=0.05
        )
        
        # Load pre-trained ML models
        print("Loading ML models...")
        self._load_ml_models()
        
        # Reset global counters
        MetricsCounters.reset()
        
        # Register the metrics counter with the learned module
        from lsm.learned import register_metrics
        register_metrics(MetricsCounters)
        
        # For collecting results
        self.metrics = {
            'put': {'traditional': [], 'learned': []},
            'get': {'traditional': [], 'learned': []},
            'range': {'traditional': [], 'learned': []},
            'remove': {'traditional': [], 'learned': []},
            'bloom_checks': {'total': 0, 'bypassed': 0},
            'disk_io': {'reads': 0, 'writes': 0},
            'accuracy': {
                'false_negatives': 0,
                'total_lookups': 0,
                'false_negative_rate': 0.0
            }
        }
        
        # For tracking model accuracy
        self.fnr_per_level = {}  # Track FNR by level
        
        print("Test environment initialization complete")
        
    def _copy_tree_structure(self, source_dir, dest_dir):
        """Copy the tree structure from source to destination directory."""
        import shutil
        
        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} does not exist.")
            return False
            
        # Create destination if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            
        print(f"Copying data from {source_dir} to {dest_dir}")
            
        # Copy all files and directories except ml_models
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(dest_dir, item)
            
            if item == 'ml_models':
                print(f"Skipping ml_models directory")
                continue
                
            if os.path.isdir(s):
                print(f"Copying directory: {s}")
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)
                
                # Copy files in the subdirectory
                for file in os.listdir(s):
                    src_file = os.path.join(s, file)
                    dst_file = os.path.join(d, file)
                    if os.path.isfile(src_file):
                        print(f"  Copying file: {src_file}")
                        shutil.copy2(src_file, dst_file)
            else:
                print(f"Copying file: {s}")
                shutil.copy2(s, d)
                
        print(f"Directory copy complete")
        return True
    
    def _load_ml_models(self):
        """Load pre-trained ML models from the designated path."""
        start_time = time.time()
        
        # Check if the ML models exist in learned_test_data/ml_models
        ml_models_dir = os.path.join(self.learned_dir, "ml_models")
        
        if not os.path.exists(ml_models_dir):
            os.makedirs(ml_models_dir, exist_ok=True)
        
        # Load models for the learned tree
        try:
            print(f"Loading models from {ml_models_dir}")
            self.learned_tree.ml_models.load_models()
            
            # Get accuracy metrics
            bloom_acc = self.learned_tree.ml_models.get_bloom_accuracy()
            fence_acc = self.learned_tree.ml_models.get_fence_accuracy()
            
            print(f"Models loaded successfully")
            print(f"Bloom Filter Accuracy: {bloom_acc:.2%}")
            print(f"Fence Pointer Accuracy: {fence_acc:.2%}")
            
            # Update statistics
            self.learned_tree.training_stats['bloom']['last_accuracy'] = bloom_acc
            self.learned_tree.training_stats['fence']['last_accuracy'] = fence_acc
            
        except Exception as e:
            print(f"Error loading models, will create new ones: {e}")
            self._train_ml_models()
        
        elapsed = time.time() - start_time
        print(f"Model loading complete in {elapsed:.2f} seconds")
        
    def _train_ml_models(self):
        """Train ML models on the existing data if they don't exist."""
        from train_models import ModelTrainer
        
        print("Training ML models...")
        trainer = ModelTrainer(data_dir=self.data_dir, model_dir=self.learned_dir)
        trainer.train_models()
        
        # Reload the trained models
        self.learned_tree.ml_models.load_models()
        print("ML models training complete")
    
    def _time_operation(self, tree, func_name, *args, **kwargs):
        """Time an operation with high precision."""
        try:
            # Get the function from the tree
            func = getattr(tree, func_name)
            
            # Make sure we got a callable function
            if not callable(func):
                print(f"Error: {func_name} is not callable")
                return None, -1
                
            # Time the operation
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            # Return result and elapsed time in microseconds
            return result, (end_time - start_time) * 1000000
            
        except Exception as e:
            print(f"Error in operation {func_name}: {e}")
            return None, -1
            
    def _generate_random_key(self):
        """Generate a random key as a float."""
        return random.uniform(0, 1000000)
        
    def _generate_random_value(self, key):
        """Generate a random value string based on the key."""
        return f"value-{key}-{random.randint(0, 100000)}"
        
    def print_level_statistics(self):
        """Print statistics about both trees."""
        print("\n====== Traditional LSM Tree Statistics ======")
        trad_stats = self.traditional_tree.get_stats()
        total_runs = 0
        total_entries = 0
        
        # Check if buffer stats exist
        if 'buffer' in trad_stats and 'entries' in trad_stats['buffer']:
            print(f"Buffer: {trad_stats['buffer']['entries']} entries " + 
                  f"({trad_stats['buffer']['size_bytes']/1024:.1f} KB)")
        else:
            print("Buffer: Stats not available")
        
        for i, level in enumerate(trad_stats.get('levels', [])):
            total_runs += level.get('run_count', 0)
            total_entries += level.get('entries', 0)
            
            print(f"Level {i}: {level.get('run_count', 0)} runs, " + 
                  f"{level.get('entries', 0)} entries " + 
                  f"({level.get('size_bytes', 0)/1024/1024:.2f} MB)")
            
        print(f"Total: {total_runs} runs, {total_entries} entries")
        
        print("\n====== Learned LSM Tree Statistics ======")
        learned_stats = self.learned_tree.get_stats()
        total_runs = 0
        total_entries = 0
        
        # Check if buffer stats exist
        if 'buffer' in learned_stats and 'entries' in learned_stats['buffer']:
            print(f"Buffer: {learned_stats['buffer']['entries']} entries " + 
                  f"({learned_stats['buffer']['size_bytes']/1024:.1f} KB)")
        else:
            print("Buffer: Stats not available")
        
        for i, level in enumerate(learned_stats.get('levels', [])):
            total_runs += level.get('run_count', 0)
            total_entries += level.get('entries', 0)
            
            print(f"Level {i}: {level.get('run_count', 0)} runs, " + 
                  f"{level.get('entries', 0)} entries " + 
                  f"({level.get('size_bytes', 0)/1024/1024:.2f} MB)")
            
        print(f"Total: {total_runs} runs, {total_entries} entries")
        
        # Print ML model training stats
        print("\n====== ML Model Stats ======")
        bloom_acc = self.learned_tree.ml_models.get_bloom_accuracy()
        fence_acc = self.learned_tree.ml_models.get_fence_accuracy()
        
        print(f"Bloom Filter Accuracy: {bloom_acc:.2%}")
        print(f"Fence Pointer Accuracy: {fence_acc:.2%}")

    def test_random_lookups(self, num_lookups: int = 100):
        """Test random key lookups to measure performance and accuracy."""
        print(f"\n===== Testing {num_lookups} Random Lookups =====")
        
        # Reset counters
        MetricsCounters.reset()
        
        # Get all keys present in the traditional tree (ground truth)
        trad_stats = self.traditional_tree.get_stats()
        existing_keys = []
        
        # Extract a sample of existing keys from each level
        for i, level_stats in enumerate(trad_stats.get('levels', [])):
            level = self.traditional_tree.levels[i]
            
            for run in level.runs:
                try:
                    with open(run.file_path, 'rb') as f:
                        data = pickle.load(f)
                        pairs = data.get('pairs', [])
                        
                        # Sample keys from this run
                        max_samples = min(50, len(pairs))
                        if max_samples > 0:
                            sampled_pairs = random.sample(pairs, max_samples)
                            for key, _ in sampled_pairs:
                                if isinstance(key, bytes):
                                    # Convert bytes to float
                                    key = struct.unpack('!d', key[:8])[0]
                                existing_keys.append((key, i))  # Store key and level
                except Exception as e:
                    print(f"Error loading keys from run: {e}")
        
        # Sample 90% of tests from existing keys, 10% non-existing
        num_existing = int(num_lookups * 0.9)  # 90% existing keys
        num_nonexisting = num_lookups - num_existing  # 10% non-existing keys
        
        # Make sure we don't request more existing keys than we have
        num_existing = min(num_existing, len(existing_keys))
        
        if num_existing > 0:
            existing_samples = random.sample(existing_keys, num_existing)
        else:
            existing_samples = []
            
        # Generate non-existing keys
        non_existing_keys = []
        for _ in range(num_nonexisting):
            key = self._generate_random_key()
            # Make sure it doesn't exist (simplified check)
            if not any(abs(key - ex_key) < 0.0001 for ex_key, _ in existing_keys):
                non_existing_keys.append(key)
        
        # Combine existing and non-existing keys
        all_keys = [(key, level) for key, level in existing_samples] + [(key, -1) for key in non_existing_keys]
        random.shuffle(all_keys)  # Shuffle the keys
        
        print(f"Testing with {len(existing_samples)} existing and {len(non_existing_keys)} non-existing keys")
        
        # Initialize metrics
        traditional_times = []
        learned_times = []
        false_negatives = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        
        # Track FNR by level
        level_metrics = {}
        
        # Perform lookups
        for key, actual_level in all_keys:
            # Traditional lookup (ground truth)
            trad_result, trad_time = self._time_operation(self.traditional_tree, "get", key)
            traditional_times.append(trad_time)
            
            # Learned lookup
            learned_result, learned_time = self._time_operation(self.learned_tree, "get", key)
            learned_times.append(learned_time)
            
            # Check accuracy
            if trad_result is not None:  # Key exists in traditional
                if learned_result is not None:  # Correctly found in learned
                    true_positives += 1
                else:  # Key exists but learned model missed it - FALSE NEGATIVE
                    false_negatives += 1
                    # Record which level had the miss for detailed analysis
                    if actual_level not in level_metrics:
                        level_metrics[actual_level] = {"fn": 0, "total": 0}
                    level_metrics[actual_level]["fn"] += 1
            else:  # Key doesn't exist in traditional
                if learned_result is None:  # Correctly not found in learned
                    true_negatives += 1
                else:  # Key doesn't exist but learned model found something - FALSE POSITIVE
                    false_positives += 1
            
            # Track level statistics
            if actual_level >= 0:  # Only for real keys in a level
                if actual_level not in level_metrics:
                    level_metrics[actual_level] = {"fn": 0, "total": 0}
                level_metrics[actual_level]["total"] += 1
        
        # Calculate overall metrics
        total = len(all_keys)
        positives = true_positives + false_negatives
        negatives = true_negatives + false_positives
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Most important: FALSE NEGATIVE RATE (keys we missed)
        false_negative_rate = false_negatives / positives if positives > 0 else 0
        
        # Calculate FNR per level
        for level, metrics in level_metrics.items():
            if metrics["total"] > 0:
                metrics["fnr"] = metrics["fn"] / metrics["total"]
            else:
                metrics["fnr"] = 0
        
        # Store results
        self.metrics['get']['traditional'] = traditional_times
        self.metrics['get']['learned'] = learned_times
        self.metrics['accuracy']['false_negatives'] = false_negatives
        self.metrics['accuracy']['total_lookups'] = total
        self.metrics['accuracy']['false_negative_rate'] = false_negative_rate
        self.fnr_per_level = level_metrics
        
        # Print results summary
        print("\n===== Random Lookup Results =====")
        print(f"Traditional avg lookup time: {statistics.mean(traditional_times):.2f} µs")
        print(f"Learned avg lookup time: {statistics.mean(learned_times):.2f} µs")
        print(f"Speedup: {statistics.mean(traditional_times) / statistics.mean(learned_times):.2f}x")
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"False Negative Rate: {false_negative_rate:.4f}")
        
        # Print FNR by level
        print("\nFalse Negative Rate by Level:")
        for level in sorted(level_metrics.keys()):
            metrics = level_metrics[level]
            print(f"Level {level}: {metrics['fnr']:.4f} ({metrics['fn']}/{metrics['total']})")
        
        # Update global metrics
        print(f"\nBloom Filter Checks: {MetricsCounters.bloom_checks}")
        print(f"Bloom Filter Bypasses: {MetricsCounters.bloom_bypasses}")
        print(f"Bypass Rate: {MetricsCounters.bloom_bypasses / MetricsCounters.bloom_checks:.2%}" if MetricsCounters.bloom_checks > 0 else "Bypass Rate: N/A")
        
        return {
            'traditional_time': statistics.mean(traditional_times),
            'learned_time': statistics.mean(learned_times), 
            'speedup': statistics.mean(traditional_times) / statistics.mean(learned_times),
            'false_negative_rate': false_negative_rate
        }

    def test_range_queries(self, num_queries: int = 24):
        """Test range queries to measure performance and accuracy."""
        print(f"\n===== Testing {num_queries} Range Queries =====")
        
        # Reset counters
        MetricsCounters.reset()
        
        # Parameters for ranges - smaller range sizes for faster testing
        range_sizes = [5, 20, 50]  # Different range sizes to test
        
        # Track results for each range size
        results_by_size = {}
        
        for range_size in range_sizes:
            print(f"\nTesting ranges of size ~{range_size} keys")
            
            traditional_times = []
            learned_times = []
            trad_result_counts = []
            learned_result_counts = []
            
            # Generate random ranges - fewer queries per size
            queries_per_size = num_queries // len(range_sizes)
            for _ in range(queries_per_size):
                # Generate random start key
                start_key = self._generate_random_key()
                # Calculate end key based on range size (approximate)
                end_key = start_key + random.uniform(range_size * 0.1, range_size * 0.2)
                
                # Traditional range query
                trad_results, trad_time = self._time_operation(
                    self.traditional_tree, "range", start_key, end_key
                )
                if trad_results is not None:
                    traditional_times.append(trad_time)
                    trad_result_counts.append(len(trad_results))
                
                # Learned range query
                learned_results, learned_time = self._time_operation(
                    self.learned_tree, "range", start_key, end_key
                )
                if learned_results is not None:
                    learned_times.append(learned_time)
                    learned_result_counts.append(len(learned_results))
                
                # Check if the learned tree missed any keys
                if trad_results is not None and learned_results is not None:
                    # Convert to sets for comparison (just the keys)
                    trad_keys = set(k for k, _ in trad_results)
                    learned_keys = set(k for k, _ in learned_results)
                    
                    # Keys missed by the learned tree
                    missed_keys = trad_keys - learned_keys
                    if missed_keys:
                        # Record false negatives
                        MetricsCounters.false_negatives += len(missed_keys)
            
            # Calculate statistics for this range size
            if traditional_times and learned_times:
                avg_trad_time = statistics.mean(traditional_times)
                avg_learned_time = statistics.mean(learned_times)
                speedup = avg_trad_time / avg_learned_time if avg_learned_time > 0 else 0
                
                avg_trad_count = statistics.mean(trad_result_counts) if trad_result_counts else 0
                avg_learned_count = statistics.mean(learned_result_counts) if learned_result_counts else 0
                
                completeness = avg_learned_count / avg_trad_count if avg_trad_count > 0 else 1.0
                
                results_by_size[range_size] = {
                    'traditional_time': avg_trad_time,
                    'learned_time': avg_learned_time,
                    'speedup': speedup,
                    'trad_result_count': avg_trad_count,
                    'learned_result_count': avg_learned_count,
                    'completeness': completeness
                }
                
                print(f"Range ~{range_size} results:")
                print(f"  Traditional avg time: {avg_trad_time:.2f} µs, Avg results: {avg_trad_count:.1f}")
                print(f"  Learned avg time: {avg_learned_time:.2f} µs, Avg results: {avg_learned_count:.1f}")
                print(f"  Speedup: {speedup:.2f}x, Completeness: {completeness:.2%}")
        
        # Calculate overall statistics
        all_trad_times = []
        all_learned_times = []
        
        for size_results in results_by_size.values():
            all_trad_times.append(size_results['traditional_time'])
            all_learned_times.append(size_results['learned_time'])
        
        if all_trad_times and all_learned_times:
            overall_speedup = statistics.mean(all_trad_times) / statistics.mean(all_learned_times)
            
            print("\nOverall Range Query Results:")
            print(f"Average Traditional Time: {statistics.mean(all_trad_times):.2f} µs")
            print(f"Average Learned Time: {statistics.mean(all_learned_times):.2f} µs")
            print(f"Overall Speedup: {overall_speedup:.2f}x")
            
            # Store overall metrics
            self.metrics['range']['traditional'] = all_trad_times
            self.metrics['range']['learned'] = all_learned_times
        
        return results_by_size

    def test_deletions(self, num_deletions: int = 1_000):
        """Test deletions with different patterns."""
        print("\n== TESTING DELETIONS ==")
        print(f"Running {num_deletions} deletions per pattern")
        
        # Reset counters for deletion tests
        MetricsCounters.reset()
        
        # Different deletion patterns
        patterns = {
            'random': lambda: random.uniform(0, 1_000_000_000),  # Delete randomly
            'recent': lambda: random.uniform(800_000_000, 1_000_000_000)  # Delete recent keys
        }
        
        for pattern_name, key_gen in patterns.items():
            print(f"\nPattern: {pattern_name}")
            
            # Generate keys to delete
            delete_keys = [key_gen() for _ in range(num_deletions)]
            
            # Traditional deletions
            print("  Traditional deletions: ", end="", flush=True)
            trad_start_time = time.perf_counter()
            
            for i, key in enumerate(delete_keys):
                if i % 250 == 0 and i > 0:
                    print(f"{i}/{num_deletions}...", end="", flush=True)
                
                # Time traditional deletion
                _, duration = self._time_operation(
                    self.traditional_tree, "remove", key
                )
                self.metrics['remove']['traditional'].append(duration)
            
            trad_end_time = time.perf_counter()
            trad_total_time = (trad_end_time - trad_start_time) * 1000000  # μs
            print(" Done.")
            
            # Learned deletions
            print("  Learned deletions: ", end="", flush=True)
            learned_start_time = time.perf_counter()
            
            for i, key in enumerate(delete_keys):
                if i % 250 == 0 and i > 0:
                    print(f"{i}/{num_deletions}...", end="", flush=True)
                
                # Time learned deletion
                _, duration = self._time_operation(
                    self.learned_tree, "remove", key
                )
                self.metrics['remove']['learned'].append(duration)
            
            learned_end_time = time.perf_counter()
            learned_total_time = (learned_end_time - learned_start_time) * 1000000  # μs
            print(" Done.")
            
            # Print deletion statistics
            speedup = trad_total_time / learned_total_time if learned_total_time > 0 else float('inf')
            print(f"  Results: Speedup: {speedup:.2f}x")

    def visualize_results(self):
        """Generate visualizations for the performance results."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Ensure the plots directory exists
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # 1. Get operation latencies
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Operation Latency Comparison', fontsize=16)
            
            operations = [
                ('get', 'Get Operation', 0, 0),
                ('range', 'Range Operation', 0, 1),
                ('put', 'Put Operation', 1, 0),
                ('remove', 'Remove Operation', 1, 1)
            ]
            
            for op, title, row, col in operations:
                if op in self.metrics and 'traditional' in self.metrics[op] and 'learned' in self.metrics[op]:
                    trad_times = self.metrics[op]['traditional']
                    learned_times = self.metrics[op]['learned']
                    
                    if trad_times and learned_times:
                        avg_trad = statistics.mean(trad_times)
                        avg_learned = statistics.mean(learned_times)
                        
                        axs[row, col].bar(['Traditional', 'Learned'], [avg_trad, avg_learned])
                        axs[row, col].set_title(title)
                        axs[row, col].set_ylabel('Latency (µs)')
                        
                        # Show speedup on plot
                        speedup = avg_trad / avg_learned if avg_learned > 0 else 0
                        axs[row, col].text(1, avg_learned, f'{speedup:.2f}x', ha='center', va='bottom')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(plots_dir, 'operation_latency.png'))
            
            # 2. Bloom filter effectiveness
            if MetricsCounters.bloom_checks > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = ['Checked', 'Bypassed']
                values = [MetricsCounters.bloom_checks - MetricsCounters.bloom_bypasses, 
                          MetricsCounters.bloom_bypasses]
                
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title('Bloom Filter Effectiveness')
                plt.savefig(os.path.join(plots_dir, 'bloom_effectiveness.png'))
            
            # 3. False Negative Rate by Level
            if self.fnr_per_level:
                levels = sorted(self.fnr_per_level.keys())
                fnrs = [self.fnr_per_level[lvl]['fnr'] for lvl in levels]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar([f'Level {lvl}' for lvl in levels], fnrs)
                ax.set_title('False Negative Rate by Level')
                ax.set_ylabel('False Negative Rate')
                ax.set_ylim(0, max(max(fnrs) * 1.1, 0.01))  # Set y-axis to start at 0
                
                for i, fnr in enumerate(fnrs):
                    ax.text(i, fnr, f'{fnr:.3f}', ha='center', va='bottom')
                    
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'fnr_by_level.png'))
                
            print(f"Visualizations saved to {plots_dir} directory")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    def print_results(self):
        """Print detailed results of all tests."""
        print("\n====== Final Performance Results ======")
        
        # Print operation latencies
        operations = ['get', 'range', 'put', 'remove']
        for op in operations:
            if op in self.metrics and 'traditional' in self.metrics[op] and 'learned' in self.metrics[op]:
                trad_times = self.metrics[op]['traditional']
                learned_times = self.metrics[op]['learned']
                
                if trad_times and learned_times:
                    avg_trad = statistics.mean(trad_times)
                    avg_learned = statistics.mean(learned_times)
                    speedup = avg_trad / avg_learned if avg_learned > 0 else 0
                    
                    print(f"\n{op.capitalize()} Operation:")
                    print(f"  Traditional: {avg_trad:.2f} µs")
                    print(f"  Learned:     {avg_learned:.2f} µs")
                    print(f"  Speedup:     {speedup:.2f}x")
        
        # Print bloom filter effectiveness
        print("\nBloom Filter Effectiveness:")
        if MetricsCounters.bloom_checks > 0:
            bypass_rate = MetricsCounters.bloom_bypasses / MetricsCounters.bloom_checks
            print(f"  Total Checks:   {MetricsCounters.bloom_checks}")
            print(f"  Bypassed:       {MetricsCounters.bloom_bypasses}")
            print(f"  Bypass Rate:    {bypass_rate:.2%}")
        else:
            print("  No bloom filter checks recorded")
        
        # Print accuracy metrics
        print("\nAccuracy Metrics:")
        fnr = self.metrics['accuracy']['false_negative_rate']
        total = self.metrics['accuracy']['total_lookups']
        fn_count = self.metrics['accuracy']['false_negatives']
        
        print(f"  False Negative Rate: {fnr:.4f}")
        print(f"  False Negatives: {fn_count} out of {total} lookups")
        
        # Print FNR by level
        if self.fnr_per_level:
            print("\nFalse Negative Rate by Level:")
            for level in sorted(self.fnr_per_level.keys()):
                metrics = self.fnr_per_level[level]
                print(f"  Level {level}: {metrics['fnr']:.4f} ({metrics['fn']}/{metrics['total']})")
        
        # Print ML model metrics from the learned tree
        print("\nML Model Metrics:")
        bloom_acc = self.learned_tree.ml_models.get_bloom_accuracy()
        fence_acc = self.learned_tree.ml_models.get_fence_accuracy()
        prediction_stats = self.learned_tree.ml_models.get_prediction_stats()
        
        print(f"  Bloom Filter Model Accuracy: {bloom_acc:.2%}")
        print(f"  Fence Pointer Model Accuracy: {fence_acc:.2%}")
        
        if prediction_stats:
            bloom_time = prediction_stats.get('avg_bloom_prediction_time', 0) * 1000000  # Convert to µs
            fence_time = prediction_stats.get('avg_fence_prediction_time', 0) * 1000000  # Convert to µs
            
            print(f"  Avg Bloom Prediction Time: {bloom_time:.2f} µs")
            print(f"  Avg Fence Prediction Time: {fence_time:.2f} µs")
            
            bloom_cache_size = prediction_stats.get('bloom_cache_size', 0)
            fence_cache_size = prediction_stats.get('fence_cache_size', 0)
            
            print(f"  Bloom Cache Size: {bloom_cache_size}")
            print(f"  Fence Cache Size: {fence_cache_size}")
            
            fence_hits = prediction_stats.get('fence_direct_cache_hits', 0)
            fence_misses = prediction_stats.get('fence_direct_cache_misses', 0)
            
            if fence_hits + fence_misses > 0:
                fence_hit_rate = fence_hits / (fence_hits + fence_misses)
                print(f"  Fence Cache Hit Rate: {fence_hit_rate:.2%}")

def main():
    """Main function to run performance tests."""
    print("Starting LSM tree performance testing...")
    
    # Initialize the test
    test = PerformanceTest()
    
    # Print initial tree statistics
    test.print_level_statistics()
    
    # Run lookup test to measure accuracy and performance
    test.test_random_lookups(num_lookups=100)
    
    # Run range query test
    test.test_range_queries(num_queries=24)
    
    # Print final results
    test.print_results()
    
    # Generate visualizations
    test.visualize_results()
    
    print("\nPerformance testing complete!")

if __name__ == "__main__":
    main() 