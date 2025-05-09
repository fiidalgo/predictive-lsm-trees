import os
import time
import random
import statistics
import logging
import numpy as np
import pickle
import struct
import matplotlib.pyplot as plt
from lsm import TraditionalLSMTree, Constants

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

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
        
        # Initialize traditional tree
        self.traditional_tree = TraditionalLSMTree(
            data_dir=self.data_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01
        )
        
        # Set up directories for other implementations
        self.classifier_dir = "classifier_test_data"
        self.learned_dir = "learned_test_data"
        
        os.makedirs(self.classifier_dir, exist_ok=True)
        os.makedirs(self.learned_dir, exist_ok=True)
        
        # Copy data structure to other directories
        print("Copying data structure to test trees...")
        self._copy_tree_structure(self.data_dir, self.classifier_dir)
        self._copy_tree_structure(self.data_dir, self.learned_dir)
        
        # Initialize classifier-based tree
        self.classifier_tree = TraditionalLSMTree(
            data_dir=self.classifier_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01
        )
        
        # Initialize learned bloom filter tree
        self.learned_tree = TraditionalLSMTree(
            data_dir=self.learned_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01
        )
        
        # Load the classifier models
        self._load_classifier_models()
        
        # Load the learned bloom filter models
        self._load_learned_bloom_models()
        
        # Store workload keys for testing
        self.workload_keys = {
            'random': [],
            'sequential': [],
            'level1': [],
            'level2': [],
            'level3': []
        }
        
        # For storing results
        self.results = {
            'random': {'traditional': [], 'classifier': [], 'learned': []},
            'sequential': {'traditional': [], 'classifier': [], 'learned': []},
            'level1': {'traditional': [], 'classifier': [], 'learned': []},
            'level2': {'traditional': [], 'classifier': [], 'learned': []},
            'level3': {'traditional': [], 'classifier': [], 'learned': []}
        }
        
        # Reset counters
        MetricsCounters.reset()
        
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
                        shutil.copy2(src_file, dst_file)
            else:
                shutil.copy2(s, d)
                
        print(f"Directory copy complete")
        return True
    
    def _load_classifier_models(self):
        """Load the classifier models for level prediction."""
        print("Loading classifier models...")
        try:
            # Load classifier models
            models = {}
            for level in range(1, 4):  # For levels 1, 2, 3
                model_path = f"models/classifier_bloom_lvl_{level}.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        models[level] = pickle.load(f)
                    print(f"Loaded classifier model for level {level}")
                else:
                    print(f"Warning: Classifier model not found for level {level}")
            
            # Apply the classifier models to the classifier tree
            import struct as _struct
            
            for lvl in self.classifier_tree.levels:
                level_num = lvl.level_num
                if level_num in models:
                    # Modify the level's get method to use the classifier
                    orig_level_get = lvl.get
                    model = models[level_num]
                    
                    def level_get_with_classifier(key, 
                                                 orig_get=orig_level_get, 
                                                 model=model):
                        # Process key if needed
                        if isinstance(key, bytes):
                            try:
                                key_float = _struct.unpack('!d', key[:8])[0]
                            except:
                                return orig_get(key)
                        else:
                            key_float = key
                        
                        # Use the classifier to predict if this level contains the key
                        try:
                            # Prepare features (simple feature is just the key)
                            features = np.array([[key_float]])
                            
                            # Predict if key is in this level
                            prediction = model.predict(features)[0]
                            
                            MetricsCounters.bloom_checks += 1
                            
                            if prediction == 0:  # Model predicts key is not in this level
                                MetricsCounters.bloom_bypasses += 1
                                return None
                        except Exception as e:
                            # If prediction fails, fall back to traditional method
                            pass
                        
                        # If we get here, either model predicts the key might be here,
                        # or we had an error and need to fall back
                        return orig_get(key)
                    
                    # Replace the level's get method
                    lvl.get = level_get_with_classifier
            
            print("Classifier models loaded and applied")
        except Exception as e:
            print(f"Error loading classifier models: {e}")
    
    def _load_learned_bloom_models(self):
        """Load the learned bloom filter models."""
        print("Loading learned bloom filter models...")
        try:
            # Load learned bloom filter models
            models = {}
            for level in range(1, 4):  # For levels 1, 2, 3
                model_path = f"models/learned_bloom_lvl_{level}.pkl"
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        models[level] = pickle.load(f)
                    print(f"Loaded learned bloom filter for level {level}")
                else:
                    print(f"Warning: Learned bloom filter not found for level {level}")
            
            # Apply the learned bloom filter models to the learned tree
            import struct as _struct
            
            for lvl in self.learned_tree.levels:
                level_num = lvl.level_num
                if level_num in models:
                    # Modify the level's get method to use the learned bloom filter
                    orig_level_get = lvl.get
                    model = models[level_num]
                    
                    def level_get_with_learned_bloom(key, 
                                                   orig_get=orig_level_get, 
                                                   model=model):
                        # Process key if needed
                        if isinstance(key, bytes):
                            try:
                                key_float = _struct.unpack('!d', key[:8])[0]
                            except:
                                return orig_get(key)
                        else:
                            key_float = key
                        
                        # Use the learned bloom filter to check key membership
                        try:
                            # Prepare features
                            features = np.array([[key_float]])
                            
                            # Predict membership
                            exists = model.predict(features)[0]
                            
                            MetricsCounters.bloom_checks += 1
                            
                            if exists == 0:  # Model predicts key doesn't exist
                                MetricsCounters.bloom_bypasses += 1
                                return None
                        except Exception as e:
                            # If prediction fails, fall back to traditional method
                            pass
                        
                        # If we get here, either model predicts the key might exist,
                        # or we had an error and need to fall back
                        return orig_get(key)
                    
                    # Replace the level's get method
                    lvl.get = level_get_with_learned_bloom
            
            print("Learned bloom filter models loaded and applied")
        except Exception as e:
            print(f"Error loading learned bloom filter models: {e}")
    
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
    
    def _generate_workload_keys(self, num_keys=100):
        """Generate keys for each workload type."""
        print("Generating workload keys...")
        
        # Get tree statistics to extract real keys
        trad_stats = self.traditional_tree.get_stats()
        
        # Dict to store all keys from each level
        level_keys = {1: [], 2: [], 3: []}
        all_keys = []
        
        # Extract keys from each level
        for i, level_stats in enumerate(trad_stats.get('levels', [])):
            if i >= len(self.traditional_tree.levels):
                continue
                
            level = i + 1  # Level numbers are 1-based
            if level > 3:  # We only care about levels 1-3
                continue
                
            level_obj = self.traditional_tree.levels[i]
            
            for run in level_obj.runs:
                try:
                    with open(run.file_path, 'rb') as f:
                        data = pickle.load(f)
                        pairs = data.get('pairs', [])
                        
                        for key, _ in pairs:
                            if isinstance(key, bytes):
                                # Convert bytes to float
                                key = struct.unpack('!d', key[:8])[0]
                            
                            level_keys[level].append(key)
                            all_keys.append((key, level))
                except Exception as e:
                    print(f"Error loading keys from run in level {level}: {e}")
        
        # Create random workload (sample from all levels)
        if all_keys:
            random_sample = random.sample(all_keys, min(num_keys, len(all_keys)))
            self.workload_keys['random'] = [key for key, _ in random_sample]
        else:
            self.workload_keys['random'] = [self._generate_random_key() for _ in range(num_keys)]
        
        # Create sequential workload (take keys in order)
        if all_keys:
            sorted_keys = sorted([key for key, _ in all_keys])
            start_idx = random.randint(0, max(0, len(sorted_keys) - num_keys))
            self.workload_keys['sequential'] = sorted_keys[start_idx:start_idx + min(num_keys, len(sorted_keys))]
        else:
            self.workload_keys['sequential'] = [i for i in range(num_keys)]
        
        # Create level-specific workloads
        for level in range(1, 4):
            if level_keys[level]:
                self.workload_keys[f'level{level}'] = random.sample(
                    level_keys[level], 
                    min(num_keys, len(level_keys[level]))
                )
            else:
                print(f"Warning: No keys found for level {level}, using random keys")
                self.workload_keys[f'level{level}'] = [self._generate_random_key() for _ in range(num_keys)]
        
        # Print summary of workload keys
        for workload, keys in self.workload_keys.items():
            print(f"  {workload}: {len(keys)} keys")
    
    def _generate_random_key(self):
        """Generate a random key as a float."""
        return random.uniform(0, 1000000)
    
    def run_workload_tests(self, num_keys=100):
        """Run performance tests for all workloads."""
        print("\n===== RUNNING WORKLOAD TESTS =====")
        
        # Generate workload keys if not already done
        if not self.workload_keys['random']:
            self._generate_workload_keys(num_keys)
        
        # Workload types
        workloads = ['random', 'sequential', 'level1', 'level2', 'level3']
        
        # Run tests for each workload
        for workload in workloads:
            print(f"\nRunning {workload} workload tests...")
            
            # Reset counters for each workload
            MetricsCounters.reset()
            
            # Get keys for this workload
            keys = self.workload_keys[workload]
            
            if not keys:
                print(f"  No keys available for {workload} workload, skipping")
                continue
            
            # Test traditional implementation
            print("  Testing traditional implementation...")
            for i, key in enumerate(keys):
                _, time_us = self._time_operation(self.traditional_tree, "get", key)
                self.results[workload]['traditional'].append(time_us)
                if i % 20 == 0:
                    print(f"    Completed {i+1}/{len(keys)} operations")
                
            # Test classifier implementation
            print("  Testing classifier implementation...")
            for i, key in enumerate(keys):
                _, time_us = self._time_operation(self.classifier_tree, "get", key)
                self.results[workload]['classifier'].append(time_us)
                if i % 20 == 0:
                    print(f"    Completed {i+1}/{len(keys)} operations")
                
            # Test learned bloom filter implementation
            print("  Testing learned bloom filter implementation...")
            for i, key in enumerate(keys):
                _, time_us = self._time_operation(self.learned_tree, "get", key)
                self.results[workload]['learned'].append(time_us)
                if i % 20 == 0:
                    print(f"    Completed {i+1}/{len(keys)} operations")
                
            # Print summary for this workload
            trad_avg = statistics.mean(self.results[workload]['traditional'])
            class_avg = statistics.mean(self.results[workload]['classifier'])
            learn_avg = statistics.mean(self.results[workload]['learned'])
            
            print(f"  Results for {workload} workload:")
            print(f"    Traditional: {trad_avg:.2f} µs")
            print(f"    Classifier: {class_avg:.2f} µs")
            print(f"    Learned Bloom: {learn_avg:.2f} µs")
            print(f"    Speedup (Classifier vs Traditional): {trad_avg/class_avg:.2f}x")
            print(f"    Speedup (Learned Bloom vs Traditional): {trad_avg/learn_avg:.2f}x")
        
        print("\nAll workload tests completed")
    
    def generate_plots(self):
        """Generate the requested plots."""
        print("\nGenerating plots...")
        
        # Make sure plots directory exists
        os.makedirs("plots", exist_ok=True)
        
        # Plot 1: Summary bar chart
        self._generate_summary_bar_chart()
        
        # Plots 2-6: Line charts for each workload
        workloads = ['random', 'sequential', 'level1', 'level2', 'level3']
        for workload in workloads:
            self._generate_workload_line_chart(workload)
            
        print("All plots generated")
    
    def _generate_summary_bar_chart(self):
        """Generate a summary bar chart comparing all workloads and implementations."""
        plt.figure(figsize=(12, 8))
        
        # Workload types
        workloads = ['random', 'sequential', 'level1', 'level2', 'level3']
        
        # Calculate mean times for each workload and implementation
        means = {
            'traditional': [],
            'classifier': [],
            'learned': []
        }
        
        for workload in workloads:
            for impl in ['traditional', 'classifier', 'learned']:
                if self.results[workload][impl]:
                    means[impl].append(statistics.mean(self.results[workload][impl]))
                else:
                    means[impl].append(0)
        
        # Set up the plot
        x = np.arange(len(workloads))  # X-axis locations
        width = 0.25  # Width of the bars
        
        # Create bars
        plt.bar(x - width, means['traditional'], width, label='Traditional', color='blue')
        plt.bar(x, means['classifier'], width, label='Classifier', color='green')
        plt.bar(x + width, means['learned'], width, label='Learned Bloom', color='red')
        
        # Add labels, title, etc.
        plt.xlabel('Workload Type')
        plt.ylabel('Average Runtime (µs)')
        plt.title('Performance Comparison by Workload and Implementation')
        plt.xticks(x, workloads)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(means['traditional']):
            plt.text(i - width, v + 5, f"{v:.1f}", ha='center', va='bottom')
        for i, v in enumerate(means['classifier']):
            plt.text(i, v + 5, f"{v:.1f}", ha='center', va='bottom')
        for i, v in enumerate(means['learned']):
            plt.text(i + width, v + 5, f"{v:.1f}", ha='center', va='bottom')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig("plots/workload_comparison.png")
        print("  Saved plots/workload_comparison.png")
        plt.close()
    
    def _generate_workload_line_chart(self, workload):
        """Generate a line chart for a specific workload."""
        plt.figure(figsize=(10, 6))
        
        # Get data for this workload
        trad_times = self.results[workload]['traditional']
        class_times = self.results[workload]['classifier']
        learn_times = self.results[workload]['learned']
        
        # Ensure all arrays have data
        if not trad_times or not class_times or not learn_times:
            print(f"  Warning: Missing data for {workload} workload, skipping plot")
            plt.close()
            return
        
        # Create x-axis values (operation indices)
        min_len = min(len(trad_times), len(class_times), len(learn_times))
        x = np.arange(min_len)
        
        # Trim arrays to same length
        trad_times = trad_times[:min_len]
        class_times = class_times[:min_len]
        learn_times = learn_times[:min_len]
        
        # Plot the lines
        plt.plot(x, trad_times, 'b-', label='Traditional', alpha=0.7)
        plt.plot(x, class_times, 'g-', label='Classifier', alpha=0.7)
        plt.plot(x, learn_times, 'r-', label='Learned Bloom', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Operation Index')
        plt.ylabel('Runtime (µs)')
        plt.title(f'{workload.capitalize()} Workload Performance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"plots/{workload}_performance.png")
        print(f"  Saved plots/{workload}_performance.png")
        plt.close()

    def print_summary(self):
        """Print a summary of test results."""
        print("\n===== PERFORMANCE TEST SUMMARY =====")
        
        workloads = ['random', 'sequential', 'level1', 'level2', 'level3']
        implementations = ['traditional', 'classifier', 'learned']
        
        # Print table header
        print(f"{'Workload':<15} {'Traditional':<15} {'Classifier':<15} {'Learned Bloom':<15} {'Classifier Speedup':<20} {'Learned Speedup':<20}")
        print("-" * 100)
        
        # Print results for each workload
        for workload in workloads:
            trad = statistics.mean(self.results[workload]['traditional']) if self.results[workload]['traditional'] else 0
            clas = statistics.mean(self.results[workload]['classifier']) if self.results[workload]['classifier'] else 0
            learn = statistics.mean(self.results[workload]['learned']) if self.results[workload]['learned'] else 0
            
            clas_speedup = trad / clas if clas > 0 else 0
            learn_speedup = trad / learn if learn > 0 else 0
            
            print(f"{workload:<15} {trad:<15.2f} {clas:<15.2f} {learn:<15.2f} {clas_speedup:<20.2f} {learn_speedup:<20.2f}")
        
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

def main():
    """Main function to run performance tests."""
    print("Starting LSM tree performance testing...")
    
    # Initialize the test environment
    test = PerformanceTest()
    
    # Run workload tests with 100 keys each
    test.run_workload_tests(num_keys=100)
    
    # Generate plots
    test.generate_plots()
    
    # Print summary
    test.print_summary()
    
    print("\nPerformance testing complete! Detailed plots have been saved to the 'plots' directory.")

if __name__ == "__main__":
    main() 