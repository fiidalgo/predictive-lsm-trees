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

# Create plots directory if it doesn't exist
os.makedirs("python-implementation/plots", exist_ok=True)

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
        
        # Initialize traditional tree (ground-truth)
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
        
        # Initialize the "full-ML" LSM (we'll only steal its bloom model)
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
        
        # Now build a "hybrid" copy of the traditional tree
        # that uses our learned bloom filter but the same fence pointer.
        # (We deep-copy via a fresh TraditionalLSMTree on a separate directory.)
        self.hybrid_dir = "hybrid_test_data"
        os.makedirs(self.hybrid_dir, exist_ok=True)
        self._copy_tree_structure(self.data_dir, self.hybrid_dir)
        self.hybrid_tree = TraditionalLSMTree(
            data_dir=self.hybrid_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01
        )

        # Add learned bloom filter at the level while keeping run-level filters
        from ml.models import FastBloomFilter  # For type hinting
        import struct as _struct
        
        for lvl in self.hybrid_tree.levels:
            bf = self.learned_tree.ml_models.bloom_models.get(lvl.level_num)
            if not bf:
                continue
                
            # Keep each run's original bloom filter so run.might_contain still prunes
            orig_level_get = lvl.get
            
            def level_get(key, 
                         orig_get=orig_level_get, 
                         bf=bf):
                # Process binary key to float if needed
                if isinstance(key, bytes):
                    try:
                        # Try to get first 8 bytes as a double
                        key_float = _struct.unpack('!d', key[:8])[0]
                    except (_struct.error, ValueError, TypeError):
                        # On error, skip learned bloom and go with original method
                        return orig_get(key)
                else:
                    key_float = key
                
                # Now ask the bloom filter using the numeric key
                try:
                    # FastBloomFilter.predict handles feature creation and returns True/False
                    exists = bf.predict(key_float)
                    MetricsCounters.bloom_checks += 1
                    if not exists:  # False means key definitely doesn't exist
                        MetricsCounters.bloom_bypasses += 1
                        return None
                except Exception:
                    # If anything goes wrong with the bloom filter, fall back to original
                    pass
                
                # If we get here, either bloom says key may exist or there was an error
                return orig_get(key)
                
            lvl.get = level_get
        
        # Reset global counters
        MetricsCounters.reset()
        
        # Register the metrics counter with the learned module
        from lsm.learned import register_metrics
        register_metrics(MetricsCounters)
        
        # For collecting results
        self.metrics = {
            'put': {'traditional': [], 'learned': [], 'hybrid': []},
            'get': {'traditional': [], 'learned': [], 'hybrid': []},
            'remove': {'traditional': [], 'learned': [], 'hybrid': []},
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
            
            # Propagate the trained bloom threshold to all levels
            self.learned_tree._load_ml_models()
            
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
        from train_classifier import ModelTrainer
        
        print("Training ML models...")
        trainer = ModelTrainer(data_dir=self.data_dir, model_dir=self.learned_dir)
        trainer.train_models()
        
        # Reload the trained models
        self.learned_tree.ml_models.load_models()
        
        # Propagate the trained bloom threshold to all levels
        self.learned_tree._load_ml_models()
        
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
        hybrid_times = []
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
            
            # Hybrid lookup (only bloom learned, fence pointer default)
            hybrid_result, hybrid_time = self._time_operation(self.hybrid_tree, "get", key)
            hybrid_times.append(hybrid_time)
            
            # Check accuracy
            if trad_result is not None:  # Key exists in traditional
                if hybrid_result is not None:  # Correctly found in hybrid
                    true_positives += 1
                else:  # Key exists but hybrid model missed it - FALSE NEGATIVE
                    false_negatives += 1
                    # Record which level had the miss for detailed analysis
                    if actual_level not in level_metrics:
                        level_metrics[actual_level] = {"fn": 0, "total": 0}
                    level_metrics[actual_level]["fn"] += 1
            else:  # Key doesn't exist in traditional
                if hybrid_result is None:  # Correctly not found in hybrid
                    true_negatives += 1
                else:  # Key doesn't exist but hybrid model found something - FALSE POSITIVE
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
        self.metrics['get']['hybrid'] = hybrid_times
        self.metrics['accuracy']['false_negatives'] = false_negatives
        self.metrics['accuracy']['total_lookups'] = total
        self.metrics['accuracy']['false_negative_rate'] = false_negative_rate
        self.fnr_per_level = level_metrics
        
        # Print results summary
        print("\n===== Random Lookup Results =====")
        print(f"Traditional avg lookup time: {statistics.mean(traditional_times):.2f} µs")
        print(f"Hybrid avg lookup time: {statistics.mean(hybrid_times):.2f} µs")
        print(f"Speedup: {statistics.mean(traditional_times) / statistics.mean(hybrid_times):.2f}x")
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
            'hybrid_time': statistics.mean(hybrid_times), 
            'speedup': statistics.mean(traditional_times) / statistics.mean(hybrid_times),
            'false_negative_rate': false_negative_rate
        }

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
            
            # Hybrid deletions
            print("  Hybrid deletions: ", end="", flush=True)
            hybrid_start_time = time.perf_counter()
            
            for i, key in enumerate(delete_keys):
                if i % 250 == 0 and i > 0:
                    print(f"{i}/{num_deletions}...", end="", flush=True)
                
                # Time hybrid deletion
                _, duration = self._time_operation(
                    self.hybrid_tree, "remove", key
                )
                self.metrics['remove']['hybrid'].append(duration)
            
            hybrid_end_time = time.perf_counter()
            hybrid_total_time = (hybrid_end_time - hybrid_start_time) * 1000000  # μs
            print(" Done.")
            
            # Print deletion statistics
            speedup = trad_total_time / hybrid_total_time if hybrid_total_time > 0 else float('inf')
            print(f"  Results: Speedup: {speedup:.2f}x")

    def visualize_results(self):
        """Visualize the performance results."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create plots directory if it doesn't exist
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Prepare data for visualization
            operation_types = ['get', 'remove']
            
            for op_type in operation_types:
                if not self.metrics[op_type]['traditional'] or not self.metrics[op_type]['hybrid']:
                    continue  # Skip if no data
                    
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Calculate means
                trad_mean = np.mean(self.metrics[op_type]['traditional'])
                hybrid_mean = np.mean(self.metrics[op_type]['hybrid'])
                
                # Create labels
                labels = ['Traditional', 'Hybrid (Learned Bloom)']
                means = [trad_mean, hybrid_mean]
                
                # Create bar chart
                plt.bar(labels, means, color=['blue', 'green'])
                
                # Add value labels on bars
                for i, v in enumerate(means):
                    plt.text(i, v + 5, f"{v:.2f}", ha='center')
                
                # Add title and labels
                plt.title(f'{op_type.capitalize()} Operation Performance')
                plt.ylabel('Time (µs)')
                plt.ylim(0, max(means) * 1.2)  # Set y-limit with headroom
                
                # Save figure
                plt.savefig(f"{plots_dir}/{op_type}_performance.png")
                print(f"Saved {plots_dir}/{op_type}_performance.png")
                
                # Close the figure to free memory
                plt.close()
            
            # Create a detailed GET operations plot
            if 'get' in self.metrics and self.metrics['get']['traditional'] and self.metrics['get']['hybrid']:
                plt.figure(figsize=(12, 6))
                
                # Get data for individual operations
                trad_times = self.metrics['get']['traditional']
                hybrid_times = self.metrics['get']['hybrid']
                
                # Make sure both arrays have the same length (use the smaller length)
                min_length = min(len(trad_times), len(hybrid_times))
                trad_times = trad_times[:min_length]
                hybrid_times = hybrid_times[:min_length]
                
                # Create x-axis values (operation indices)
                x = np.arange(min_length)
                
                # Plot individual GET operations
                plt.plot(x, trad_times, 'b-', label='Traditional', alpha=0.7)
                plt.plot(x, hybrid_times, 'g-', label='Hybrid (Learned Bloom)', alpha=0.7)
                
                # Add labels and title
                plt.title('GET Operation Time for Each Data Point')
                plt.xlabel('Operation Index')
                plt.ylabel('Time (µs)')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Save figure
                plt.savefig(f"{plots_dir}/detailed_get_operations.png")
                print(f"Saved {plots_dir}/detailed_get_operations.png")
                
                # Close the figure
                plt.close()
                
            # Create a speedup comparison figure
            plt.figure(figsize=(10, 6))
            
            # Prepare speedup data
            speedups = []
            labels = []
            
            for op_type in operation_types:
                if not self.metrics[op_type]['traditional'] or not self.metrics[op_type]['hybrid']:
                    continue  # Skip if no data
                    
                trad_mean = np.mean(self.metrics[op_type]['traditional'])
                hybrid_mean = np.mean(self.metrics[op_type]['hybrid'])
                
                if hybrid_mean > 0:
                    speedup = trad_mean / hybrid_mean
                    speedups.append(speedup)
                    labels.append(op_type.capitalize())
            
            # Create bar chart for speedups
            plt.bar(labels, speedups, color='orange')
            
            # Add value labels
            for i, v in enumerate(speedups):
                plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
                
            # Add title and labels
            plt.title('Speedup Factors (Traditional / Hybrid)')
            plt.ylabel('Speedup Factor')
            plt.ylim(0, max(speedups) * 1.2)  # Set y-limit with headroom
            
            # Save figure
            plt.savefig(f"{plots_dir}/speedup_comparison.png")
            print(f"Saved {plots_dir}/speedup_comparison.png")
            
            # Close the figure
            plt.close()
            
            # FNR comparison
            if self.fnr_per_level:
                plt.figure(figsize=(10, 6))
                
                # Prepare FNR data
                levels = sorted(self.fnr_per_level.keys())
                fnrs = [self.fnr_per_level[lvl]['fnr'] for lvl in levels]
                
                # Create bar chart for FNRs
                plt.bar([f"Level {lvl}" for lvl in levels], fnrs, color='red')
                
                # Add value labels
                for i, v in enumerate(fnrs):
                    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
                    
                # Add title and labels
                plt.title('False Negative Rate by Level (Hybrid Bloom Filter)')
                plt.ylabel('False Negative Rate')
                plt.ylim(0, max(max(fnrs) * 1.2, 0.05))  # Set y-limit with headroom
                
                # Save figure
                plt.savefig(f"{plots_dir}/fnr_by_level.png")
                print(f"Saved {plots_dir}/fnr_by_level.png")
                
                # Close the figure
                plt.close()
                
        except ImportError:
            print("Matplotlib not available for visualization. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error during visualization: {e}")

    def print_results(self):
        """Print the performance results."""
        print("\n===== PERFORMANCE TEST RESULTS =====")
        
        # Print all operation results
        for op_type in ['get', 'remove']:
            if not self.metrics[op_type]['traditional'] or not self.metrics[op_type]['hybrid']:
                continue  # Skip if no data for this operation
                
            # Calculate statistics
            trad_times = self.metrics[op_type]['traditional']
            hybrid_times = self.metrics[op_type]['hybrid']
            
            trad_mean = statistics.mean(trad_times) if trad_times else 0
            hybrid_mean = statistics.mean(hybrid_times) if hybrid_times else 0
            
            trad_median = statistics.median(trad_times) if trad_times else 0
            hybrid_median = statistics.median(hybrid_times) if hybrid_times else 0
            
            try:
                trad_std_dev = statistics.stdev(trad_times) if len(trad_times) > 1 else 0
                hybrid_std_dev = statistics.stdev(hybrid_times) if len(hybrid_times) > 1 else 0
            except statistics.StatisticsError:
                trad_std_dev = 0
                hybrid_std_dev = 0
                
            speedup = trad_mean / hybrid_mean if hybrid_mean > 0 else 0
            
            # Print results for this operation
            print(f"\n----- {op_type.upper()} OPERATION -----")
            print(f"Traditional:")
            print(f"  Mean: {trad_mean:.2f} µs")
            print(f"  Median: {trad_median:.2f} µs")
            print(f"  Std Dev: {trad_std_dev:.2f} µs")
            print(f"  Min: {min(trad_times):.2f} µs, Max: {max(trad_times):.2f} µs")
            
            print(f"Hybrid (Learned Bloom):")
            print(f"  Mean: {hybrid_mean:.2f} µs")
            print(f"  Median: {hybrid_median:.2f} µs")
            print(f"  Std Dev: {hybrid_std_dev:.2f} µs")
            print(f"  Min: {min(hybrid_times):.2f} µs, Max: {max(hybrid_times):.2f} µs")
            
            print(f"Speedup (Traditional/Hybrid): {speedup:.2f}x")
        
        # Print accuracy metrics
        print("\n----- ACCURACY METRICS -----")
        fnr = self.metrics['accuracy']['false_negative_rate']
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"False Negatives: {self.metrics['accuracy']['false_negatives']}")
        print(f"Total Lookups: {self.metrics['accuracy']['total_lookups']}")
        
        # Print bloom filter metrics
        print("\n----- BLOOM FILTER METRICS -----")
        try:
            bloom_checks = MetricsCounters.bloom_checks
            bloom_bypasses = MetricsCounters.bloom_bypasses
            bypass_rate = bloom_bypasses / bloom_checks if bloom_checks > 0 else 0
            
            print(f"Bloom Checks: {bloom_checks}")
            print(f"Bloom Bypasses: {bloom_bypasses}")
            print(f"Bypass Rate: {bypass_rate:.2%}")
        except:
            print("Bloom filter metrics not available")
            
        # Print I/O metrics
        print("\n----- I/O METRICS -----")
        try:
            print(f"Disk Reads: {MetricsCounters.disk_reads}")
            print(f"Disk Writes: {MetricsCounters.disk_writes}")
        except:
            print("I/O metrics not available")
            
        # Print page access metrics
        print("\n----- PAGE ACCESS METRICS -----")
        try:
            print(f"Traditional Pages Checked: {MetricsCounters.traditional_pages_checked}")
            print(f"Hybrid Pages Checked: {MetricsCounters.learned_pages_checked}")
            
            # Calculate page access ratio
            if MetricsCounters.traditional_pages_checked > 0:
                page_ratio = MetricsCounters.learned_pages_checked / MetricsCounters.traditional_pages_checked
                print(f"Page Check Ratio (Hybrid/Traditional): {page_ratio:.2f}x")
        except:
            print("Page access metrics not available")
            
        # Print FNR by level
        if self.fnr_per_level:
            print("\n----- FALSE NEGATIVE RATE BY LEVEL -----")
            for level in sorted(self.fnr_per_level.keys()):
                metrics = self.fnr_per_level[level]
                print(f"Level {level}: {metrics['fnr']:.4f} ({metrics['fn']}/{metrics['total']})")
        
        print("\n===== END OF RESULTS =====")

        # Return a summary of the results
        return {
            'get_speedup': trad_mean / hybrid_mean if 'get' in self.metrics and hybrid_mean > 0 else 0,
            'false_negative_rate': fnr
        }

def main():
    """Main function to run performance tests."""
    print("Starting LSM tree performance testing...")
    
    # Initialize the test
    test = PerformanceTest()
    
    # Print initial tree statistics
    test.print_level_statistics()
    
    # Run lookup test to measure accuracy and performance
    print("\nComparing Traditional LSM Tree vs Hybrid LSM Tree with Learned Bloom Filter")
    print("The hybrid approach uses traditional fence pointers but learned bloom filters")
    
    test.test_random_lookups(num_lookups=100)
    
    # Print final results
    test.print_results()
    
    # Generate visualizations
    test.visualize_results()
    
    print("\nPerformance testing complete! Detailed plots have been saved to the 'python-implementation/plots' directory.")

if __name__ == "__main__":
    main() 