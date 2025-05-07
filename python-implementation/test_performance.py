import os
import time
import random
import statistics
import logging
import numpy as np
import pickle
import struct
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
        if os.path.exists(self.learned_dir):
            import shutil
            shutil.rmtree(self.learned_dir)
        os.makedirs(self.learned_dir, exist_ok=True)
        
        # Copy data structure from traditional to learned
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
        
        # Make sure the learned tree loads the copied data
        print("Loading data into learned tree...")
        try:
            # Force the learned tree to discover the copied data
            self.learned_tree.load_existing_data()
        except Exception as e:
            print(f"Warning: Error loading existing data into learned tree: {e}")
            import traceback
            traceback.print_exc()
        
        # Train ML models
        print("Training ML models...")
        self._train_ml_models()
        
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
            'false_negatives': {
                'count': 0,
                'total_lookups': 0,
                'by_pattern': {}  # Track false negatives per pattern
            }
        }
        
        # Specialized batch size for loading data
        self.LOAD_BATCH_SIZE = 10000
        
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
    
    def _train_ml_models(self):
        """Train ML models on the loaded data."""
        start_time = time.time()
        
        # Get all runs from the traditional tree
        trad_stats = self.traditional_tree.get_stats()
        
        # Get total number of runs across all levels
        run_count = 0
        for level_stats in trad_stats.get('levels', []):
            run_count += level_stats.get('run_count', 0)
        
        print(f"Training on data from {run_count} runs across all levels...")
        
        # Ensure the modules are loaded properly
        import random
        import struct
        
        # IMPORTANT: Use the ORIGINAL data directory for training, not the learned directory!
        total_keys_added = 0
        total_negative_samples = 0
        
        # Directly access traditional tree runs for training
        for level_idx, level in enumerate(self.traditional_tree.levels):
            if not level.runs:
                continue
                
            print(f"Training models for Level {level_idx} with {len(level.runs)} runs...")
            
            # Train models for each run in the ORIGINAL tree
            for run in level.runs:
                try:
                    # Load run data
                    with open(run.file_path, 'rb') as f:
                        data = pickle.load(f)
                        pairs = data.get('pairs', [])
                        
                        if not pairs:
                            print(f"  Run {run.file_path} has no data, skipping")
                            continue
                            
                        print(f"  Processing run with {len(pairs)} entries")
                        
                        # Sample data if needed to speed up training
                        if len(pairs) > 10000:
                            print(f"  Sampling {10000} records from {len(pairs)} for faster training")
                            pairs = random.sample(pairs, 10000)
                        
                        # Get key set for generating negative samples
                        key_set = set()
                        
                        # Calculate page boundaries
                        records_per_page = run.page_size // Constants.ENTRY_SIZE_BYTES
                        
                        # Get min/max key range
                        min_key = run.min_key
                        max_key = run.max_key
                        
                        if isinstance(min_key, bytes):
                            min_key = struct.unpack('!d', min_key[:8])[0]
                        if isinstance(max_key, bytes):
                            max_key = struct.unpack('!d', max_key[:8])[0]
                        
                        # Process each key-value pair
                        keys_added = 0
                        for i, (key, _) in enumerate(pairs):
                            # Convert key from bytes to float if needed
                            if isinstance(key, bytes):
                                key = struct.unpack('!d', key[:8])[0]
                                
                            # Add to key set
                            key_set.add(key)
                            
                            # Calculate page id
                            page_id = i // records_per_page
                            
                            # Add to training data
                            self.learned_tree.ml_models.add_bloom_training_data(key, True)  # Key exists
                            self.learned_tree.ml_models.add_fence_training_data(key, level_idx, page_id)
                            keys_added += 1
                            
                        total_keys_added += keys_added
                        print(f"  Added {keys_added} positive training samples")
                        
                        # Generate negative samples for bloom filter training
                        negative_samples = 0
                        if min_key is not None and max_key is not None and min_key < max_key:
                            key_range = max_key - min_key
                            # Generate as many negative samples as positive samples for better class balance
                            num_negative_samples = min(len(pairs), 10000)  # Use same sample size as positive samples
                            
                            print(f"  Adding {num_negative_samples} negative samples for class balance")
                            
                            for _ in range(num_negative_samples):
                                # Generate a key outside the range
                                fake_key = random.uniform(min_key - key_range * 0.1, max_key + key_range * 0.1)
                                
                                # Ensure it's not an actual key
                                if fake_key not in key_set:
                                    self.learned_tree.ml_models.add_bloom_training_data(fake_key, False)
                                    negative_samples += 1
                            
                            total_negative_samples += negative_samples
                            print(f"  Successfully added {negative_samples} negative samples")
                    
                except Exception as e:
                    print(f"  Error processing run: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Total training data collected: {total_keys_added} positive samples, {total_negative_samples} negative samples")
        if total_keys_added == 0:
            print("ERROR: No training data collected, models cannot be trained!")
            return
            
        # Train the models with all the collected data
        print("Training ML models with collected data...")
        
        # Check if there's any training data
        bloom_data_size = len(self.learned_tree.ml_models.bloom_data) if hasattr(self.learned_tree.ml_models, 'bloom_data') else 0
        fence_data_size = len(self.learned_tree.ml_models.fence_data) if hasattr(self.learned_tree.ml_models, 'fence_data') else 0
        print(f"Bloom filter training data: {bloom_data_size} samples")
        print(f"Fence pointer training data: {fence_data_size} samples")
        
        try:
            # Get initial accuracy values for comparison
            init_bloom_acc = self.learned_tree.ml_models.get_bloom_accuracy()
            init_fence_acc = self.learned_tree.ml_models.get_fence_accuracy()
            
            # Display proper initialization message instead of misleading accuracy
            print(f"Starting model training from initial state. No accuracy metrics available yet.")
            
            # Time the bloom filter training
            bloom_start = time.time()
            self.learned_tree.ml_models.train_bloom_model()
            bloom_time = time.time() - bloom_start
            print(f"Bloom filter model training completed in {bloom_time:.2f} seconds")
            
            # Time the fence pointer training
            fence_start = time.time()
            self.learned_tree.ml_models.train_fence_model()
            fence_time = time.time() - fence_start
            print(f"Fence pointer model training completed in {fence_time:.2f} seconds")
            
            # Update statistics
            new_bloom_acc = self.learned_tree.ml_models.get_bloom_accuracy()
            new_fence_acc = self.learned_tree.ml_models.get_fence_accuracy()
            
            # Check if accuracy values actually changed
            if abs(new_bloom_acc - init_bloom_acc) < 0.0001:
                print("WARNING: Bloom filter accuracy didn't change after training! Training might not be working.")
            
            if abs(new_fence_acc - init_fence_acc) < 0.0001:
                print("WARNING: Fence pointer accuracy didn't change after training! Training might not be working.")
                
            self.learned_tree.training_stats['bloom']['last_accuracy'] = new_bloom_acc
            self.learned_tree.training_stats['fence']['last_accuracy'] = new_fence_acc
            
            print(f"New accuracy values: Bloom Filter: {new_bloom_acc:.2%}, Fence Pointer: {new_fence_acc:.2%}")
            
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        print(f"ML model training complete in {elapsed:.2f} seconds")
        
        # Get ML model stats
        stats = self.learned_tree.get_stats()
        if 'ml_models' in stats:
            ml_stats = stats['ml_models']
            print(f"Bloom Filter Accuracy: {ml_stats.get('bloom_accuracy', 0):.2%}")
            print(f"Fence Pointer Accuracy: {ml_stats.get('fence_accuracy', 0):.2%}")
        
    def _time_operation(self, tree, func_name, *args, **kwargs):
        """Time an operation with high precision."""
        try:
            # Get the function from the tree
            func = getattr(tree, func_name)
            
            # Make sure we got a callable function
            if not callable(func):
                raise TypeError(f"Retrieved attribute '{func_name}' is not callable: {type(func)}")
            
            start_time = time.perf_counter_ns()  # Use nanoseconds for maximum precision
            result = func(*args, **kwargs)
            end_time = time.perf_counter_ns()
            
            # Convert to microseconds (1 ns = 0.001 μs)
            duration_us = (end_time - start_time) / 1000
            return result, duration_us
        except Exception as e:
            print(f"Error in _time_operation: {e}")
            print(f"Tree type: {type(tree)}, Function name: {func_name}")
            print(f"Args: {args}, Kwargs: {kwargs}")
            raise
    
    def _generate_random_key(self):
        """Generate a random key as a float (will be converted to fixed-size bytes)."""
        return random.uniform(0, 1_000_000_000)
        
    def _generate_random_value(self, key):
        """Generate a deterministic value from key."""
        # Use a simple transformation to ensure the value is deterministic for the key
        # This helps in validation later
        return f"v{key:.6f}"
        
    def print_level_statistics(self):
        """Print statistics about the current level structure of both trees."""
        print("\n== LEVEL STATISTICS ==")
        
        try:
            # Traditional Tree Stats
            trad_stats = self.traditional_tree.get_stats()
            print("\nTraditional Tree:")
            
            if isinstance(trad_stats.get('levels'), list):
                total_entries = 0
                total_size_mb = 0
                
                print(f"{'Level':<6}{'Entries':<12}{'Size (MB)':<12}{'Runs':<6}")
                print("-" * 36)
                
                for i, level in enumerate(trad_stats['levels']):
                    entries = level.get('entries', 0)
                    size_mb = entries * self.ENTRY_SIZE_BYTES / (1024 * 1024)
                    run_count = len(level.get('runs', []))
                    total_entries += entries
                    total_size_mb += size_mb
                    print(f"{i:<6}{entries:,d} {'':<5}{size_mb:.2f} {'':<5}{run_count}")
                
                print("-" * 36)
                print(f"{'Total':<6}{total_entries:,d} {'':<5}{total_size_mb:.2f}")
            
            # Learned Tree Stats
            learned_stats = self.learned_tree.get_stats()
            print("\nLearned Tree:")
            
            if isinstance(learned_stats.get('levels'), list):
                total_entries = 0
                total_size_mb = 0
                
                print(f"{'Level':<6}{'Entries':<12}{'Size (MB)':<12}{'Runs':<6}")
                print("-" * 36)
                
                for i, level in enumerate(learned_stats['levels']):
                    entries = level.get('entries', 0)
                    size_mb = entries * self.ENTRY_SIZE_BYTES / (1024 * 1024)
                    run_count = len(level.get('runs', []))
                    total_entries += entries
                    total_size_mb += size_mb
                    print(f"{i:<6}{entries:,d} {'':<5}{size_mb:.2f} {'':<5}{run_count}")
                
                print("-" * 36)
                print(f"{'Total':<6}{total_entries:,d} {'':<5}{total_size_mb:.2f}")
            
            # ML model stats if available (concise version)
            if 'ml_models' in learned_stats:
                print("\nML Models: ", end="")
                ml_stats = learned_stats['ml_models']
                print(f"Bloom Filter Accuracy: {ml_stats.get('bloom_accuracy', 0):.2%}, ", end="")
                print(f"Fence Pointer Accuracy: {ml_stats.get('fence_accuracy', 0):.2%}")
        except Exception as e:
            print(f"Error getting tree statistics: {e}")
        
    def test_random_lookups(self, num_lookups: int = 5_000):
        """Test random lookups with different access patterns."""
        print("\n== TESTING RANDOM LOOKUPS ==")
        print(f"Running {num_lookups} lookups per pattern")
        
        # Reset counters for fresh measurement
        MetricsCounters.reset()
        
        # Initialize false negative tracking
        self.metrics['false_negatives'] = {
            'count': 0,
            'total_lookups': 0,
            'by_pattern': {}  # Track false negatives per pattern
        }
        
        # Different access patterns
        patterns = {
            'uniform': lambda: random.uniform(0, 1_000_000_000),  # Entire range
            'recent': lambda: random.uniform(800_000_000, 1_000_000_000),  # 20% newest
            'old': lambda: random.uniform(0, 200_000_000),  # 20% oldest
            'hot': lambda: random.choice([
                random.uniform(0, 200_000_000),      # 33% in oldest
                random.uniform(400_000_000, 600_000_000),  # 33% in middle
                random.uniform(800_000_000, 1_000_000_000)  # 33% in newest
            ])
        }
        
        for pattern_name, key_gen in patterns.items():
            print(f"\nPattern: {pattern_name}")
            self.metrics['false_negatives']['by_pattern'][pattern_name] = {
                'count': 0,
                'total': 0
            }
            
            # Reset bloom filter stats for this pattern
            bloom_checks_before = MetricsCounters.bloom_checks
            bloom_bypasses_before = MetricsCounters.bloom_bypasses
            
            # Reset page access counters for this pattern
            trad_pages_before = MetricsCounters.traditional_pages_checked
            learned_pages_before = MetricsCounters.learned_pages_checked
            
            # Reset disk reads counter
            disk_reads_before = MetricsCounters.disk_reads
            
            # Generate keys for the pattern
            pattern_keys = [key_gen() for _ in range(num_lookups)]
            
            # Test traditional lookups
            print("  Traditional lookups: ", end="", flush=True)
            trad_start_time = time.perf_counter()
            trad_values = []
            
            for i, key in enumerate(pattern_keys):
                if i % 1000 == 0 and i > 0:
                    print(f"{i}/{num_lookups}...", end="", flush=True)
                
                # Time traditional lookup
                value, duration = self._time_operation(
                    self.traditional_tree, "get", key
                )
                trad_values.append(value)
                self.metrics['get']['traditional'].append(duration)
            
            trad_end_time = time.perf_counter()
            trad_total_time = (trad_end_time - trad_start_time) * 1000000  # μs
            print(" Done.")
            
            # Calculate page checks and disk reads for traditional
            trad_pages_checked = MetricsCounters.traditional_pages_checked - trad_pages_before
            trad_disk_reads = MetricsCounters.disk_reads - disk_reads_before
            
            # Reset disk reads counter for learned implementation
            disk_reads_before = MetricsCounters.disk_reads
            
            # Test learned lookups
            print("  Learned lookups: ", end="", flush=True)
            learned_start_time = time.perf_counter()
            learned_values = []
            
            for i, key in enumerate(pattern_keys):
                if i % 1000 == 0 and i > 0:
                    print(f"{i}/{num_lookups}...", end="", flush=True)
                
                # Track lookup attempt
                MetricsCounters.lookups_attempted += 1
                
                # Time learned lookup (one at a time, no batching)
                value, duration = self._time_operation(
                    self.learned_tree, "get", key
                )
                learned_values.append(value)
                self.metrics['get']['learned'].append(duration)
            
            learned_end_time = time.perf_counter()
            learned_total_time = (learned_end_time - learned_start_time) * 1000000  # μs
            print(" Done.")
            
            # Calculate page checks and disk reads for learned
            learned_pages_checked = MetricsCounters.learned_pages_checked - learned_pages_before
            learned_disk_reads = MetricsCounters.disk_reads - disk_reads_before
            
            # Calculate bloom filter stats for this pattern
            pattern_bloom_checks = MetricsCounters.bloom_checks - bloom_checks_before
            pattern_bloom_bypasses = MetricsCounters.bloom_bypasses - bloom_checks_before
            
            # Validate correctness
            mismatches = 0
            for i, (trad_val, learned_val) in enumerate(zip(trad_values, learned_values)):
                if trad_val != learned_val:
                    mismatches += 1
            
            # Check for false negatives (when ML model missed finding a key that exists)
            for i, (trad_val, learned_val) in enumerate(zip(trad_values, learned_values)):
                self.metrics['false_negatives']['total_lookups'] += 1
                self.metrics['false_negatives']['by_pattern'][pattern_name]['total'] += 1
                
                if trad_val is not None and learned_val is None:
                    self.metrics['false_negatives']['count'] += 1
                    self.metrics['false_negatives']['by_pattern'][pattern_name]['count'] += 1
                    MetricsCounters.false_negatives += 1
            
            # Print pattern statistics (concise)
            bypass_rate = pattern_bloom_bypasses / pattern_bloom_checks if pattern_bloom_checks > 0 else 0
            speedup = trad_total_time / learned_total_time if learned_total_time > 0 else float('inf')
            page_ratio = learned_pages_checked / trad_pages_checked if trad_pages_checked > 0 else float('inf')
            
            print(f"  Results: Speedup: {speedup:.2f}x, Bloom bypasses: {bypass_rate:.2%}, " +
                  f"Page access ratio: {page_ratio:.2f}x, Mismatches: {mismatches}")

    def test_range_queries(self, num_queries: int = 100):
        """Test range queries with different sizes."""
        print("\n== TESTING RANGE QUERIES ==")
        print(f"Running {num_queries} queries per range size")
        
        # Reset counters for range tests
        MetricsCounters.reset()
        
        range_sizes = [10, 100, 1000]  # Different range sizes to test
        
        for size in range_sizes:
            print(f"\nRange size: {size}")
            
            # Reset page access counters for this range size
            trad_pages_before = MetricsCounters.traditional_pages_checked
            learned_pages_before = MetricsCounters.learned_pages_checked
            disk_reads_before = MetricsCounters.disk_reads
            
            # Generate range queries to ensure consistency
            range_queries = []
            for _ in range(num_queries):
                start_key = random.uniform(0, 1_000_000_000 - size * 1000)
                end_key = start_key + size * 1000  # Scale range based on size
                range_queries.append((start_key, end_key))
            
            # Traditional range queries
            print("  Traditional queries: ", end="", flush=True)
            trad_start_time = time.perf_counter()
            trad_result_counts = []
            
            for i, (start_key, end_key) in enumerate(range_queries):
                if i % 25 == 0 and i > 0:
                    print(f"{i}/{num_queries}...", end="", flush=True)
                
                # Time traditional range query
                results, duration = self._time_operation(
                    self.traditional_tree, "range", start_key, end_key
                )
                trad_result_counts.append(len(results))
                self.metrics['range']['traditional'].append(duration)
            
            trad_end_time = time.perf_counter()
            trad_total_time = (trad_end_time - trad_start_time) * 1000000  # μs
            print(" Done.")
            
            # Calculate page checks and disk reads for traditional
            trad_pages_checked = MetricsCounters.traditional_pages_checked - trad_pages_before
            trad_disk_reads = MetricsCounters.disk_reads - disk_reads_before
            
            # Reset disk reads counter for learned implementation
            disk_reads_before = MetricsCounters.disk_reads
            
            # Learned range queries
            print("  Learned queries: ", end="", flush=True)
            learned_start_time = time.perf_counter()
            learned_result_counts = []
            
            for i, (start_key, end_key) in enumerate(range_queries):
                if i % 25 == 0 and i > 0:
                    print(f"{i}/{num_queries}...", end="", flush=True)
                
                # Time learned range query
                results, duration = self._time_operation(
                    self.learned_tree, "range", start_key, end_key
                )
                learned_result_counts.append(len(results))
                self.metrics['range']['learned'].append(duration)
            
            learned_end_time = time.perf_counter()
            learned_total_time = (learned_end_time - learned_start_time) * 1000000  # μs
            print(" Done.")
            
            # Calculate page checks and disk reads for learned
            learned_pages_checked = MetricsCounters.learned_pages_checked - learned_pages_before
            learned_disk_reads = MetricsCounters.disk_reads - disk_reads_before
            
            # Validate results
            avg_trad_results = sum(trad_result_counts) / len(trad_result_counts) if trad_result_counts else 0
            avg_learned_results = sum(learned_result_counts) / len(learned_result_counts) if learned_result_counts else 0
            
            # Print statistics (concise)
            speedup = trad_total_time / learned_total_time if learned_total_time > 0 else float('inf')
            page_ratio = learned_pages_checked / trad_pages_checked if trad_pages_checked > 0 else float('inf')
            
            print(f"  Results: Speedup: {speedup:.2f}x, Page ratio: {page_ratio:.2f}x, " +
                  f"Avg results - Trad: {avg_trad_results:.1f}, Learned: {avg_learned_results:.1f}")

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

    def print_results(self):
        """Print performance comparison results."""
        print("\n=============================================")
        print("         PERFORMANCE COMPARISON RESULTS      ")
        print("=============================================")
        
        # Process results
        operation_types = ['get', 'put', 'range', 'remove']
        
        for op_type in operation_types:
            trad = self.metrics[op_type]['traditional']
            learned = self.metrics[op_type]['learned']
            
            if not trad or not learned:
                continue
                
            print(f"\n{op_type.upper()} OPERATION:")
            
            # Calculate statistics
            trad_avg = statistics.mean(trad) if trad else 0
            learned_avg = statistics.mean(learned) if learned else 0
            
            if trad_avg > 0 and learned_avg > 0:
                speedup = trad_avg / learned_avg
                # Check for unrealistically high speedups that would indicate a measurement problem
                if speedup > 20:
                    print(f"  CAUTION: Measured speedup ({speedup:.2f}x) exceeds realistic expectations")
                
                print(f"  Avg time: Trad={trad_avg:.2f}μs, Learned={learned_avg:.2f}μs, Speedup={speedup:.2f}x")
            else:
                print(f"  Avg time: Trad={trad_avg:.2f}μs, Learned={learned_avg:.2f}μs, Speedup=N/A")
                    
        # Print false negative statistics
        print("\nFALSE NEGATIVE RATE:")
        false_negative_rate = MetricsCounters.false_negatives / MetricsCounters.lookups_attempted if MetricsCounters.lookups_attempted > 0 else 0
        print(f"  Overall: {false_negative_rate:.2%} ({MetricsCounters.false_negatives}/{MetricsCounters.lookups_attempted})")
        
        # Print bloom filter statistics
        print("\nBLOOM FILTER USAGE:")
        bypass_rate = MetricsCounters.bloom_bypasses / MetricsCounters.bloom_checks if MetricsCounters.bloom_checks > 0 else 0
        print(f"  Bypass rate: {bypass_rate:.2%} ({MetricsCounters.bloom_bypasses}/{MetricsCounters.bloom_checks})")
        
        # Print disk I/O statistics
        print(f"\nDISK ACCESS:")
        print(f"  Reads: {MetricsCounters.disk_reads:,}")
        print(f"  Writes: {MetricsCounters.disk_writes:,}")
        
        # Print page access statistics
        print(f"\nPAGE ACCESS:")
        page_ratio = MetricsCounters.learned_pages_checked / MetricsCounters.traditional_pages_checked if MetricsCounters.traditional_pages_checked > 0 else float('inf')
        print(f"  Traditional: {MetricsCounters.traditional_pages_checked:,}")
        print(f"  Learned: {MetricsCounters.learned_pages_checked:,}")
        print(f"  Ratio (Learned/Traditional): {page_ratio:.2f}x")
        
        # Print ML model performance for learned implementation
        try:
            stats = self.learned_tree.get_stats()
            print("\nML MODEL PERFORMANCE:")
            print(f"  Bloom Filter Accuracy: {stats['ml_models']['bloom_accuracy']:.2%}")
            print(f"  Fence Pointer Accuracy: {stats['ml_models']['fence_accuracy']:.2%}")
        except Exception as e:
            pass

    def load_existing_data(self):
        """Force the tree to reload existing runs from disk."""
        # This will reload all the run files from the data directory
        self._load_existing_runs()

def main():
    """Main function to run performance tests."""
    print("Starting LSM tree performance tests...")
    
    # Create test instance
    test = PerformanceTest()
    
    # Print level statistics
    test.print_level_statistics()
    
    # Run lookup test
    test.test_random_lookups(num_lookups=5_000)
    
    # Reset counters between tests
    MetricsCounters.reset()
    
    # Run range query test
    test.test_range_queries(num_queries=100)
    
    # Reset counters between tests
    MetricsCounters.reset()
    
    # Run deletion test
    test.test_deletions(num_deletions=1_000)
    
    # Print overall performance comparison
    test.print_results()
    
if __name__ == "__main__":
    main() 