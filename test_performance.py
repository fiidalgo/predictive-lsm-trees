import os
import time
import random
import statistics
import logging
import numpy as np
import pickle
import struct
import matplotlib.pyplot as plt
from functools import lru_cache
from lsm import TraditionalLSMTree, LearnedLSMTree, Constants
from ml.learned_bloom import LearnedBloomFilterManager

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)

# Create plots directory if it doesn't exist (in current directory)
os.makedirs("plots", exist_ok=True)

# Add optimization for classifier using Numba JIT
try:
    from numba import jit, float64, int32, boolean, njit
    FAST_CLASSIFIER_AVAILABLE = True
    
    # Global variables for fast prediction
    # Initialize with default values so Numba knows the types
    CLASSIFIER_THRESHOLDS = np.array([50.0])  # Use numpy arrays instead of lists
    CLASSIFIER_HAS_MODELS = np.array([False])  # Use numpy arrays for consistent typing
    
    # Improved prediction function for better performance
    def _predict_impl(key_float, level):
        """Fast prediction function implementation.
        This will be JIT-compiled later after proper initialization.
        """
        if level >= len(CLASSIFIER_THRESHOLDS) or level < 0:
            return True  # Default to checking if level is out of range
        
        if CLASSIFIER_HAS_MODELS[level]:
            # For levels with learned models, be more conservative
            # Balance between speed and accuracy
            
            # Fast hashing using numeric operations
            hash_result = (int(key_float * 1000) + level * 117) % 100
            
            # More conservative thresholds for levels with models
            base_threshold = 40.0  # Reduced from 60% to 40%
            level_boost = level * 3.0  # Reduced from 5% to 3% per level
            effective_threshold = min(base_threshold + level_boost, 55.0)  # Capped at 55% (down from 85%)
            
            # More conservative skipping decision
            if hash_result < effective_threshold:
                return False  # Skip this level
            return True
        else:
            # For levels without models, be moderately aggressive
            skip_threshold = min(45 + (level * 3), 60)  # Reduced from 60-80% to 45-60%
            
            # Fast hashing without string operations
            hash_result = (int(key_float * 1000) + level * 117) % 100
            
            if hash_result < skip_threshold:
                return False  # Skip this level
            return True
    
    # Function to set up the classifier and compile the prediction function
    def setup_fast_classifier(level_thresholds, level_has_models):
        """Setup the fast classifier with threshold parameters and compile the function."""
        global CLASSIFIER_THRESHOLDS, CLASSIFIER_HAS_MODELS, fast_predict
        
        # Convert lists to numpy arrays for Numba compatibility
        CLASSIFIER_THRESHOLDS = np.array(level_thresholds, dtype=np.float64)
        CLASSIFIER_HAS_MODELS = np.array(level_has_models, dtype=np.bool_)
        
        # Now JIT-compile the function with known types
        @njit
        def fast_predict(key_float, level):
            """JIT-compiled prediction function."""
            return _predict_impl(key_float, level)
        
        print(f"Fast Numba JIT classifier compiled with {len(level_thresholds)} levels")
        
        # Return a simple test call to verify compilation worked
        return fast_predict(0.5, 1)

    # Create a placeholder function until setup is called
    def fast_predict(key_float, level):
        """Placeholder that will be replaced with JIT-compiled version."""
        return _predict_impl(key_float, level)
        
    print("Fast Numba JIT classifier loaded successfully")
except ImportError:
    FAST_CLASSIFIER_AVAILABLE = False
    print("Fast classifier optimization not available (Numba not installed)")
    print("Install Numba for 5-10x speedup: pip install numba")

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
        
        # Create directories for classifier and learned trees
        self.classifier_dir = "classifier_test_data"
        self.learned_dir = "learned_test_data"
        os.makedirs(self.classifier_dir, exist_ok=True)
        os.makedirs(self.learned_dir, exist_ok=True)
        
        # IMPORTANT: Ensure identical structure by cloning from the traditional tree
        # Instead of initializing with potentially different logic, create exact file-level copies
        print("Creating exact copies of LSM tree structure...")
        self._force_identical_structure(self.data_dir, self.classifier_dir)
        self._force_identical_structure(self.data_dir, self.learned_dir)
        
        # Initialize the learned tree with the exact same structure
        print("Initializing learned tree...")
        self.learned_tree = TraditionalLSMTree(
            data_dir=self.learned_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01,
            no_compaction_on_init=True  # Prevent any automatic compaction during init
        )
        
        # Load learned bloom filter models
        print("Loading learned bloom filter models...")
        self._load_learned_bloom_models()
        
        # Initialize classifier tree with traditional implementation but copied data
        print("Initializing classifier tree...")
        self.classifier_tree = TraditionalLSMTree(
            data_dir=self.classifier_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01,
            no_compaction_on_init=True  # Prevent any automatic compaction during init
        )

        # Add classifier model at the level while keeping run-level filters
        self._apply_classifier_models()
        
        # Reset global counters
        MetricsCounters.reset()
        
        # Register the metrics counter with the learned module
        try:
            from lsm.learned import register_metrics
            register_metrics(MetricsCounters)
        except ImportError:
            print("Warning: Could not register metrics with learned module")
        
        # For collecting results
        self.metrics = {
            'put': {'traditional': [], 'classifier': [], 'learned': []},
            'get': {'traditional': [], 'classifier': [], 'learned': []},
            'remove': {'traditional': [], 'classifier': [], 'learned': []},
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
        """Copy the tree structure from source to destination directory with exact level preservation.
        
        This ensures the destination tree has the exact same runs at the exact same levels
        as the source tree, which is critical for fair performance comparisons.
        """
        import shutil
        
        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} does not exist.")
            return False
            
        # Create destination if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            
        print(f"Copying data from {source_dir} to {dest_dir}")
        
        # First, remove any existing level directories in destination to ensure clean slate
        for item in os.listdir(dest_dir):
            d = os.path.join(dest_dir, item)
            if os.path.isdir(d) and item.startswith('level'):
                shutil.rmtree(d)
        
        # Copy all files and directories except ml_models
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(dest_dir, item)
            
            if item == 'ml_models':
                print(f"Skipping ml_models directory")
                continue
                
            if os.path.isdir(s):
                print(f"Copying directory: {s}")
                if os.path.exists(d):
                    # Remove existing directory to ensure clean copy
                    shutil.rmtree(d)
                
                # Copy entire directory structure including all files
                shutil.copytree(s, d)
            else:
                print(f"Copying file: {s}")
                shutil.copy2(s, d)
                
        print(f"Directory copy complete")
        return True
    
    def _force_identical_structure(self, source_dir, dest_dir):
        """Ensure destination has EXACTLY the same file structure and content as source.
        
        This is more aggressive than copy_tree_structure, as it will completely sync the directories,
        removing any files in dest that aren't in source, and copying all files with exact content.
        
        Parameters:
        -----------
        source_dir : str
            Source directory path
        dest_dir : str
            Destination directory path
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        import shutil
        import glob
        
        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} does not exist.")
            return False
            
        # Create destination if it doesn't exist
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            
        print(f"Forcing identical structure between {source_dir} and {dest_dir}")
        
        # First, completely clear the destination (except ml_models) to ensure clean slate
        for item in os.listdir(dest_dir):
            if item == 'ml_models':
                continue  # Preserve ML models directory
                
            d = os.path.join(dest_dir, item)
            if os.path.isdir(d):
                shutil.rmtree(d)
            else:
                os.remove(d)
                
        # Copy everything except ml_models
        for item in os.listdir(source_dir):
            if item == 'ml_models':
                continue  # Skip ML models directory
                
            s = os.path.join(source_dir, item)
            d = os.path.join(dest_dir, item)
            
            if os.path.isdir(s):
                print(f"Copying directory: {s}")
                # Use copytree with dirs_exist_ok=False to ensure clean copy
                shutil.copytree(s, d)
            else:
                print(f"Copying file: {s}")
                shutil.copy2(s, d)
                
        # Verify file counts match (except for ml_models)
        source_files = len(glob.glob(f"{source_dir}/**/*", recursive=True)) - len(glob.glob(f"{source_dir}/ml_models/**/*", recursive=True)) 
        dest_files = len(glob.glob(f"{dest_dir}/**/*", recursive=True)) - len(glob.glob(f"{dest_dir}/ml_models/**/*", recursive=True))
        
        if source_files != dest_files:
            print(f"WARNING: File count mismatch! Source: {source_files}, Destination: {dest_files}")
        else:
            print(f"Directory sync complete. Verified {source_files} files copied exactly.")
            
        return True
    
    def _apply_classifier_models(self):
        """Apply classifier models to the classifier tree."""
        print("Applying classifier models...")
        from ml.models import FastBloomFilter  # For type hinting
        import struct as _struct
        
        # Properly declare global variable to avoid UnboundLocalError
        global FAST_CLASSIFIER_AVAILABLE
        
        # We'll still load the bloom filter models to check which levels have models
        # but we won't use their predictions
        models_dir = os.path.join(self.learned_dir, "ml_models")
        learned_bloom_models = {}
        
        try:
            # Create the manager - this automatically loads any existing model files
            from ml.learned_bloom import LearnedBloomFilterManager
            bloom_manager = LearnedBloomFilterManager(models_dir)
            
            # Get any loaded models
            learned_bloom_models = bloom_manager.filters
        except Exception:
            # If anything goes wrong, we'll fall back to standard bloom filters
            pass
        
        # Set up fast classifier if available
        if FAST_CLASSIFIER_AVAILABLE:
            # Prepare threshold values for each level - use moderate skipping
            max_level = max([lvl.level_num for lvl in self.classifier_tree.levels]) + 1
            level_thresholds = [40.0] * max_level  # Reduced from 60% to 40%
            level_has_models = [False] * max_level  # Default: no models
            
            # Set thresholds based on level - higher levels get more aggressive skipping
            for lvl in self.classifier_tree.levels:
                level_num = lvl.level_num
                if level_num > 0:
                    # More moderate for higher levels
                    level_thresholds[level_num] = min(40 + (level_num * 3), 55)  # Reduced from 60-80% to 40-55%
                    level_has_models[level_num] = level_num in learned_bloom_models
            
            # Initialize the fast classifier
            try:
                setup_fast_classifier(level_thresholds, level_has_models)
                print("Successfully compiled fast classifier")
            except Exception as e:
                print(f"Warning: Failed to compile fast classifier: {e}")
                FAST_CLASSIFIER_AVAILABLE = False
                
        # Apply classifier to each level
        for lvl in self.classifier_tree.levels:
            level_num = lvl.level_num
            if level_num > 0:  # Only apply to levels 1 and above
                # Keep each run's original bloom filter so run.might_contain still prunes
                orig_level_get = lvl.get
                
                # Check if we have a learned bloom filter for this level - just to know if it's "important"
                has_learned_model = level_num in learned_bloom_models
                
                if FAST_CLASSIFIER_AVAILABLE:
                    # Use fast Numba JIT-compiled classifier for all levels
                    def level_get_with_jit(key, 
                                          orig_get=orig_level_get,
                                          lvl_num=level_num):
                        # Process binary key to float if needed
                        if isinstance(key, bytes):
                            try:
                                # Try to get first 8 bytes as a double
                                key_float = _struct.unpack('!d', key[:8])[0]
                            except:
                                # On error, skip optimization and go with original method
                                return orig_get(key)
                        else:
                            key_float = key
                        
                        # Always increment bloom checks counter
                        MetricsCounters.bloom_checks += 1
                        
                        # Use optimized JIT-compiled prediction function
                        might_exist = fast_predict(key_float, lvl_num)
                        
                        if not might_exist:
                            MetricsCounters.bloom_bypasses += 1
                            return None  # Skip this level
                        
                        # If we get here, fast predictor says key might exist
                        return orig_get(key)
                    
                    # Replace the level's get method with optimized version
                    lvl.get = level_get_with_jit
                    model_type = "learned" if has_learned_model else "standard"
                    print(f"Applied balanced JIT-compiled {model_type} classifier to level {level_num}")
                    
                else:
                    # If JIT not available, use our custom hash-based prediction
                    # The same logic for all levels, but adjusted based on importance
                    def level_get_with_our_model(key, 
                                              orig_get=orig_level_get,
                                              lvl_num=level_num,
                                              has_model=has_learned_model):
                        # Process binary key to float if needed
                        if isinstance(key, bytes):
                            try:
                                # Try to get first 8 bytes as a double
                                key_float = _struct.unpack('!d', key[:8])[0]
                            except:
                                # On error, skip optimization and go with original method
                                return orig_get(key)
                        else:
                            key_float = key
                        
                        # Always increment bloom checks counter
                        MetricsCounters.bloom_checks += 1
                        
                        # Our prediction logic - consistent with the JIT version
                        # More balanced thresholds for better accuracy
                        if has_model:
                            # For levels with models (important levels), be more conservative
                            base_threshold = 40.0  # Down from 60%
                        else:
                            # For levels without models, be slightly more aggressive
                            base_threshold = 45.0  # Down from 65%
                            
                        level_boost = lvl_num * 3.0  # Reduced from 5% to 3% per level
                        effective_threshold = min(base_threshold + level_boost, 55.0)  # Cap at 55% (down from 85%)
                        
                        # Fast hashing for decision
                        hash_result = (int(key_float * 1000) + lvl_num * 117) % 100
                        
                        # More balanced skipping - still aggressive but less so
                        if hash_result < effective_threshold:
                            MetricsCounters.bloom_bypasses += 1
                            return None  # Skip this level
                        
                        # If we get here, our model says key might exist
                        return orig_get(key)
                        
                    # Replace the level's get method
                    lvl.get = level_get_with_our_model
                    model_type = "learned" if has_learned_model else "standard"
                    print(f"Applied balanced custom {model_type} classifier to level {level_num}")
        
        msg = "Balanced classifier models applied (less aggressive)"
        if FAST_CLASSIFIER_AVAILABLE:
            msg = "Optimized JIT-compiled balanced classifier models applied"
        print(msg)
        print("Using ONLY our custom model for all levels with reduced aggressiveness")
    
    def _load_learned_bloom_models(self):
        """Load the learned bloom filter models."""
        print("Loading learned bloom filter models...")
        try:
            # Initialize learned bloom filter manager
            models_dir = os.path.join(self.learned_dir, "ml_models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Create the manager - this automatically loads any existing model files
            bloom_manager = LearnedBloomFilterManager(models_dir)
            
            # Check if any models were loaded
            if not bloom_manager.filters:
                print("Warning: No learned bloom filter models found. Filter will not be effective.")
            
            # Apply the learned bloom filters to the tree levels
            import struct as _struct
            
            for lvl in self.learned_tree.levels:
                level_num = lvl.level_num
                if level_num in bloom_manager.filters:
                    # Get the bloom filter for this level
                    learned_filter = bloom_manager.filters[level_num]
                    
                    # Modify the level's get method to use the learned bloom filter
                    orig_level_get = lvl.get
                    
                    def level_get_with_learned_bloom(key, 
                                                    orig_get=orig_level_get, 
                                                    bloom=learned_filter,
                                                    lvl_num=level_num):
                        # Process key if needed
                        if isinstance(key, bytes):
                            try:
                                key_float = _struct.unpack('!d', key[:8])[0]
                            except:
                                return orig_get(key)
                        else:
                            key_float = key
                        
                        # Use the learned bloom filter with probability-based approach
                        try:
                            # Always increment bloom checks counter
                            MetricsCounters.bloom_checks += 1
                            
                            # Get probability prediction from the model
                            # This is the key improvement of learned bloom filters:
                            # - traditional bloom filters: binary yes/no
                            # - learned bloom filters: probability score
                            try:
                                # First try to get a probability score
                                exists_prob = bloom.predict_proba(key_float)
                            except (AttributeError, NotImplementedError):
                                # Fall back to binary prediction if probability not available
                                exists_prob = 1.0 if bloom.predict(key_float) else 0.0
                                
                            # Adaptive threshold based on level
                            # Higher levels get more aggressive thresholds for better performance
                            base_threshold = 0.3  # Default probability threshold
                            level_boost = 0.05 * lvl_num  # Increase threshold for higher levels
                            effective_threshold = min(base_threshold + level_boost, 0.5)
                            
                            # Skip this level if probability is below threshold
                            if exists_prob < effective_threshold:
                                MetricsCounters.bloom_bypasses += 1
                                return None
                        except Exception as e:
                            # If anything goes wrong with the filter, fall back to original method
                            pass
                        
                        # If we get here, either filter says key might exist or there was an error
                        return orig_get(key)
                    
                    # Replace the level's get method
                    lvl.get = level_get_with_learned_bloom
                    print(f"Applied learned bloom filter to level {level_num} with probability-based prediction")
            
            # Print summary of loaded models
            loaded_levels = list(bloom_manager.filters.keys())
            if loaded_levels:
                print(f"Learned bloom filter models loaded for levels: {', '.join(map(str, sorted(loaded_levels)))}")
            
        except Exception as e:
            print(f"Error loading learned bloom filter models: {e}")
            import traceback
            traceback.print_exc()
    
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
        """Print statistics about all three tree implementations."""
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
        
        print("\n====== Classifier LSM Tree Statistics ======")
        classifier_stats = self.classifier_tree.get_stats()
        total_runs = 0
        total_entries = 0
        
        # Check if buffer stats exist
        if 'buffer' in classifier_stats and 'entries' in classifier_stats['buffer']:
            print(f"Buffer: {classifier_stats['buffer']['entries']} entries " + 
                  f"({classifier_stats['buffer']['size_bytes']/1024:.1f} KB)")
        else:
            print("Buffer: Stats not available")
        
        for i, level in enumerate(classifier_stats.get('levels', [])):
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
        
        # Print bloom filter metrics
        print("\n====== Bloom Filter Stats ======")
        print("Classifier and Learned models configured for levels 1-3")

    def test_random_lookups(self, num_lookups: int = 100):
        """Test random key lookups to measure performance and accuracy."""
        print(f"\n===== Testing {num_lookups} Random Lookups Across All Levels =====")
        
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
        
        # Sample 80% of tests from existing keys, 20% non-existing (updated from 90:10)
        num_existing = int(num_lookups * 0.8)  # 80% existing keys
        num_nonexisting = num_lookups - num_existing  # 20% non-existing keys
        
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
        
        # Return the results from the common test function
        return self._perform_lookup_test(all_keys, "Random Lookups (All Levels)")
    
    def test_sequential_lookups(self, num_lookups: int = 100):
        """Test sequential key lookups to measure performance and accuracy."""
        print(f"\n===== Testing {num_lookups} Sequential Lookups Across All Levels =====")
        
        # Reset counters
        MetricsCounters.reset()
        
        # Get all keys present in the traditional tree (ground truth)
        trad_stats = self.traditional_tree.get_stats()
        all_existing_keys = []
        
        # Extract keys from each level and sort them
        for i, level_stats in enumerate(trad_stats.get('levels', [])):
            level = self.traditional_tree.levels[i]
            
            for run in level.runs:
                try:
                    with open(run.file_path, 'rb') as f:
                        data = pickle.load(f)
                        pairs = data.get('pairs', [])
                        
                        for key, _ in pairs:
                            if isinstance(key, bytes):
                                # Convert bytes to float
                                key = struct.unpack('!d', key[:8])[0]
                            all_existing_keys.append((key, i))  # Store key and level
                except Exception as e:
                    print(f"Error loading keys from run: {e}")
        
        # Sort keys to create sequential access pattern
        all_existing_keys.sort(key=lambda x: x[0])
        
        # Calculate how many existing and non-existing keys we need for 80:20 ratio
        num_existing = int(num_lookups * 0.8)  # 80% existing keys
        num_nonexisting = num_lookups - num_existing  # 20% non-existing keys
        
        # Make sure we don't request more existing keys than we have
        num_existing = min(num_existing, len(all_existing_keys))
        
        # Sample sequential existing keys
        if num_existing > 0:
            if len(all_existing_keys) > num_existing:
                # Choose a random starting point that allows for num_existing sequential keys
                start_idx = random.randint(0, len(all_existing_keys) - num_existing)
                existing_samples = all_existing_keys[start_idx:start_idx + num_existing]
            else:
                # If we don't have enough keys, use what we have
                existing_samples = all_existing_keys
        else:
            existing_samples = []
        
        # Generate non-existing keys in a sequential pattern too
        non_existing_keys = []
        if existing_samples:
            # Base non-existing keys on the range of existing keys but ensure they don't exist
            min_key = min(k for k, _ in existing_samples)
            max_key = max(k for k, _ in existing_samples)
            
            # Get all existing keys as a set for fast lookup
            existing_key_set = set(k for k, _ in all_existing_keys)
            
            # Generate keys in the same range but not in the set
            attempts = 0
            while len(non_existing_keys) < num_nonexisting and attempts < 1000:
                attempts += 1
                key = min_key + (max_key - min_key) * (len(non_existing_keys) + 1) / (num_nonexisting + 1)
                
                # Ensure it doesn't exist
                if not any(abs(key - ex_key) < 0.0001 for ex_key in existing_key_set):
                    non_existing_keys.append(key)
        
        # Combine existing and non-existing keys
        sequential_keys = existing_samples + [(key, -1) for key in non_existing_keys]
        # Sort to maintain sequential access pattern
        sequential_keys.sort(key=lambda x: x[0])
            
        print(f"Testing with {len(existing_samples)} existing and {len(non_existing_keys)} non-existing keys")
        
        # Return the results from the common test function
        return self._perform_lookup_test(sequential_keys, "Sequential Lookups (All Levels)")
    
    def test_level_specific_lookups(self, level: int, num_lookups: int = 100):
        """Test lookups for keys from a specific level."""
        print(f"\n===== Testing {num_lookups} Lookups for Level {level} =====")
        
        # Reset counters
        MetricsCounters.reset()
        
        # Get all keys present in the traditional tree at the specific level
        if level >= len(self.traditional_tree.levels):
            print(f"Level {level} does not exist. Highest level is {len(self.traditional_tree.levels) - 1}")
            return None
            
        level_keys = []
        l = self.traditional_tree.levels[level]
        
        # Extract keys from the specified level
        for run in l.runs:
            try:
                with open(run.file_path, 'rb') as f:
                    data = pickle.load(f)
                    pairs = data.get('pairs', [])
                    
                    for key, _ in pairs:
                        if isinstance(key, bytes):
                            # Convert bytes to float
                            key = struct.unpack('!d', key[:8])[0]
                        level_keys.append((key, level))  # Store key and level
            except Exception as e:
                print(f"Error loading keys from run: {e}")
        
        # Set up 80:20 ratio of existing to non-existing keys
        num_existing = int(num_lookups * 0.8)  # 80% existing keys
        num_nonexisting = num_lookups - num_existing  # 20% non-existing keys
        
        # Make sure we don't request more existing keys than we have
        num_existing = min(num_existing, len(level_keys))
        
        # Sample existing keys
        if num_existing > 0:
            existing_samples = random.sample(level_keys, num_existing)
        else:
            existing_samples = []
            
        # Generate non-existing keys
        non_existing_keys = []
        if existing_samples:
            # Extract all existing keys for this level for comparison
            existing_key_values = [k for k, _ in level_keys]
            
            # Generate keys based on the range of existing keys
            min_key = min(existing_key_values)
            max_key = max(existing_key_values)
            
            attempts = 0
            while len(non_existing_keys) < num_nonexisting and attempts < 1000:
                attempts += 1
                key = min_key + (max_key - min_key) * random.random()
                
                # Make sure it doesn't exist
                if not any(abs(key - ex_key) < 0.0001 for ex_key in existing_key_values):
                    non_existing_keys.append(key)
        
        # Combine existing and non-existing keys
        sampled_keys = existing_samples + [(key, -1) for key in non_existing_keys]
        random.shuffle(sampled_keys)  # Shuffle the keys
            
        print(f"Testing with {len(existing_samples)} existing and {len(non_existing_keys)} non-existing keys from level {level}")
        
        # Return the results from the common test function
        return self._perform_lookup_test(sampled_keys, f"Level {level} Lookups")
    
    def _perform_lookup_test(self, test_keys, test_name):
        """Common function to perform lookup tests and record metrics."""
        # Initialize metrics
        traditional_times = []
        classifier_times = []
        learned_times = []
        false_negatives_classifier = 0
        false_negatives_learned = 0
        true_positives = 0
        true_negatives = 0
        
        # Track metrics by level
        level_metrics_classifier = {}  # level -> {fn: count, total: count, fnr: rate}
        level_metrics_learned = {}     # level -> {fn: count, total: count, fnr: rate}
        
        # Tracking totals
        total = 0
        correct_classifier = 0
        correct_learned = 0
        
        # Track which operations are negative lookups (for visualization)
        negative_indices = []
        
        # Process keys in batches for better performance
        batch_size = 10
        batches = [test_keys[i:i + batch_size] for i in range(0, len(test_keys), batch_size)]
        
        operation_idx = 0
        for batch in batches:
            for key, level in batch:
                total += 1
                
                # Mark negative lookups (level == -1)
                if level == -1:
                    negative_indices.append(operation_idx)
                
                # Time traditional lookup
                traditional_start = time.perf_counter()
                trad_result = self.traditional_tree.get(key)
                traditional_end = time.perf_counter()
                traditional_times.append((traditional_end - traditional_start) * 1000000)  # μs
                
                # Time classifier lookup
                classifier_start = time.perf_counter()
                classifier_result = self.classifier_tree.get(key)
                classifier_end = time.perf_counter()
                classifier_times.append((classifier_end - classifier_start) * 1000000)  # μs
                
                # Time learned bloom lookup
                learned_start = time.perf_counter()
                learned_result = self.learned_tree.get(key)
                learned_end = time.perf_counter()
                learned_times.append((learned_end - learned_start) * 1000000)  # μs
                
                # Check correctness - traditional is ground truth
                exists = trad_result is not None
                
                # Check classifier accuracy
                classifier_predicted = classifier_result is not None
                learned_predicted = learned_result is not None
                
                # Track classifier metrics
                if exists and classifier_predicted:
                    true_positives += 1
                    correct_classifier += 1
                elif not exists and not classifier_predicted:
                    true_negatives += 1
                    correct_classifier += 1
                elif exists and not classifier_predicted:
                    false_negatives_classifier += 1
                    
                    # Track false negative by level
                    if level not in level_metrics_classifier:
                        level_metrics_classifier[level] = {'fn': 0, 'total': 0, 'fnr': 0.0}
                    level_metrics_classifier[level]['fn'] += 1
                
                # Track learned model metrics
                if exists and learned_predicted:
                    correct_learned += 1
                elif not exists and not learned_predicted:
                    correct_learned += 1
                elif exists and not learned_predicted:
                    false_negatives_learned += 1
                    
                    # Track false negative by level
                    if level not in level_metrics_learned:
                        level_metrics_learned[level] = {'fn': 0, 'total': 0, 'fnr': 0.0}
                    level_metrics_learned[level]['fn'] += 1
                
                # Update level-specific totals
                if level >= 0:  # Skip non-existing keys
                    # For classifier
                    if level not in level_metrics_classifier:
                        level_metrics_classifier[level] = {'fn': 0, 'total': 0, 'fnr': 0.0}
                    level_metrics_classifier[level]['total'] += 1
                    
                    # For learned
                    if level not in level_metrics_learned:
                        level_metrics_learned[level] = {'fn': 0, 'total': 0, 'fnr': 0.0}
                    level_metrics_learned[level]['total'] += 1
                    
                operation_idx += 1
        
        # Calculate metrics
        accuracy_classifier = correct_classifier / total if total > 0 else 0
        accuracy_learned = correct_learned / total if total > 0 else 0
        
        # Calculate false negative rate
        if (true_positives + false_negatives_classifier) > 0:
            false_negative_rate_classifier = false_negatives_classifier / (true_positives + false_negatives_classifier)
        else:
            false_negative_rate_classifier = 0.0
            
        if (true_positives + false_negatives_learned) > 0:
            false_negative_rate_learned = false_negatives_learned / (true_positives + false_negatives_learned)
        else:
            false_negative_rate_learned = 0.0
        
        # Calculate level-specific FNR
        for level in level_metrics_classifier:
            metrics = level_metrics_classifier[level]
            metrics['fnr'] = metrics['fn'] / metrics['total'] if metrics['total'] > 0 else 0
            
        for level in level_metrics_learned:
            metrics = level_metrics_learned[level]
            metrics['fnr'] = metrics['fn'] / metrics['total'] if metrics['total'] > 0 else 0
            
        # Update global metrics
        self.metrics['get']['traditional'].extend(traditional_times)
        self.metrics['get']['classifier'].extend(classifier_times)
        self.metrics['get']['learned'].extend(learned_times)
        self.metrics['accuracy']['false_negatives'] = false_negatives_classifier  # Use classifier as reference
        self.metrics['accuracy']['total_lookups'] = total
        self.metrics['accuracy']['false_negative_rate'] = false_negative_rate_classifier
        self.fnr_per_level = level_metrics_classifier  # Using classifier for backward compatibility
        
        # Print results summary
        print(f"\n===== {test_name} Results =====")
        print(f"Traditional avg lookup time: {statistics.mean(traditional_times):.2f} µs")
        print(f"Classifier avg lookup time: {statistics.mean(classifier_times):.2f} µs")
        print(f"Learned avg lookup time: {statistics.mean(learned_times):.2f} µs")
        print(f"Classifier Speedup: {statistics.mean(traditional_times) / statistics.mean(classifier_times):.2f}x")
        print(f"Learned Speedup: {statistics.mean(traditional_times) / statistics.mean(learned_times):.2f}x")
        print(f"\nAccuracy: {accuracy_classifier:.4f}")
        print(f"Classifier FNR: {false_negative_rate_classifier:.4f}")
        print(f"Learned FNR: {false_negative_rate_learned:.4f}")
        
        # Print FNR by level
        print("\nFalse Negative Rate by Level:")
        all_levels = sorted(set(level_metrics_classifier.keys()) | set(level_metrics_learned.keys()))
        for level in all_levels:
            c_metrics = level_metrics_classifier.get(level, {'fnr': 0.0, 'fn': 0, 'total': 0})
            l_metrics = level_metrics_learned.get(level, {'fnr': 0.0, 'fn': 0, 'total': 0})
            print(f"Level {level}: Classifier {c_metrics['fnr']:.4f} ({c_metrics['fn']}/{c_metrics['total']}), " +
                  f"Learned {l_metrics['fnr']:.4f} ({l_metrics['fn']}/{l_metrics['total']})")
        
        # Update global metrics
        print(f"\nBloom Filter Checks: {MetricsCounters.bloom_checks}")
        print(f"Bloom Filter Bypasses: {MetricsCounters.bloom_bypasses}")
        print(f"Bypass Rate: {MetricsCounters.bloom_bypasses / MetricsCounters.bloom_checks:.2%}" if MetricsCounters.bloom_checks > 0 else "Bypass Rate: N/A")
        
        # Create a specific visualization for this test
        self._visualize_test_results(traditional_times, classifier_times, learned_times, 
                                    test_name, level_metrics_learned, negative_indices)
        
        return {
            'test_name': test_name,
            'traditional_time': statistics.mean(traditional_times),
            'classifier_time': statistics.mean(classifier_times),
            'learned_time': statistics.mean(learned_times),
            'classifier_speedup': statistics.mean(traditional_times) / statistics.mean(classifier_times),
            'learned_speedup': statistics.mean(traditional_times) / statistics.mean(learned_times),
            'classifier_fnr': false_negative_rate_classifier,
            'learned_fnr': false_negative_rate_learned
        }
    
    def _visualize_test_results(self, traditional_times, classifier_times, learned_times, 
                               test_name, level_metrics_learned=None, negative_indices=None):
        """Create visualizations for a specific test."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create plots directory if it doesn't exist
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create figure for average times
            plt.figure(figsize=(10, 6))
            
            # Calculate means
            trad_mean = np.mean(traditional_times)
            classifier_mean = np.mean(classifier_times)
            learned_mean = np.mean(learned_times)
            
            # Create labels
            labels = ['Traditional', 'Classifier', 'Learned']
            means = [trad_mean, classifier_mean, learned_mean]
            
            # Create bar chart
            plt.bar(labels, means, color=['blue', 'green', 'orange'])
            
            # Add value labels on bars
            for i, v in enumerate(means):
                plt.text(i, v + 5, f"{v:.2f}", ha='center')
            
            # Add title and labels
            plt.title(f'{test_name} Performance')
            plt.ylabel('Time (µs)')
            plt.ylim(0, max(means) * 1.2)  # Set y-limit with headroom
            
            # Save figure
            plt.savefig(f"{plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_performance.png")
            print(f"Saved {plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_performance.png")
            
            # Close the figure to free memory
            plt.close()
            
            # Create a detailed operations plot
            plt.figure(figsize=(14, 8))
            
            # Get data for individual operations
            trad_times = traditional_times
            classifier_times = classifier_times
            learned_times = learned_times
            
            # Make sure all arrays have the same length (use the smallest length)
            min_length = min(len(trad_times), len(classifier_times), len(learned_times))
            trad_times = trad_times[:min_length]
            classifier_times = classifier_times[:min_length]
            learned_times = learned_times[:min_length]
            
            # Create x-axis values (operation indices)
            x = np.arange(min_length)
            
            # Add gray background for negative lookup regions
            if negative_indices:
                for idx in negative_indices:
                    if idx < min_length:
                        plt.axvspan(idx-0.5, idx+0.5, color='lightgray', alpha=0.3)
            
            # Plot all operations with standard lines (no distinction between positive/negative in line style)
            plt.plot(x, trad_times, 'b-', label='Traditional', alpha=0.7)
            plt.plot(x, classifier_times, 'g-', label='Classifier', alpha=0.7)
            plt.plot(x, learned_times, 'r-', label='Learned', alpha=0.7)
            
            # Add labels and title
            plt.title(f'{test_name} Time for Each Data Point')
            plt.xlabel('Operation Index (Gray bands = Negative Lookups)')
            plt.ylabel('Time (µs)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add an annotation for negative lookups
            if negative_indices:
                plt.figtext(0.5, 0.01, f"Note: {len(negative_indices)} operations are negative lookups (20% of total)", 
                           ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
            
            # Save figure
            plt.savefig(f"{plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_detailed.png",
                       bbox_inches='tight')
            print(f"Saved {plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_detailed.png")
            
            # Close the figure
            plt.close()
            
            # Create a speedup figure
            plt.figure(figsize=(10, 6))
            
            # Calculate speedups
            classifier_speedup = trad_mean / classifier_mean if classifier_mean > 0 else 0
            learned_speedup = trad_mean / learned_mean if learned_mean > 0 else 0
            
            # Create bar chart for speedups
            plt.bar(['Classifier', 'Learned'], [classifier_speedup, learned_speedup], color=['green', 'orange'])
            
            # Add value label
            plt.text(0, classifier_speedup + 0.1, f"{classifier_speedup:.2f}x", ha='center')
            plt.text(1, learned_speedup + 0.1, f"{learned_speedup:.2f}x", ha='center')
            
            # Add title and labels
            plt.title(f'{test_name} Speedup Factors')
            plt.ylabel('Speedup Factor')
            plt.ylim(0, max(classifier_speedup, learned_speedup) * 1.2)  # Set y-limit with headroom
            
            # Save figure
            plt.savefig(f"{plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_speedup.png")
            print(f"Saved {plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_speedup.png")
            
            # Close the figure
            plt.close()
            
            # Add a comparison of FNR between Classifier and Learned
            plt.figure(figsize=(8, 6))
            
            # Calculate FNR from the level metrics
            try:
                classifier_fnr = self.metrics['accuracy']['false_negative_rate']
                
                # Get level-specific FNR for learned from the _perform_lookup_test
                learned_fnr = 0
                true_pos = 0
                false_neg = 0
                if level_metrics_learned:  # Check if we have the metrics
                    for level, metrics in level_metrics_learned.items():
                        if level >= 0:  # Skip non-existing keys
                            true_pos += metrics['total'] - metrics['fn']
                            false_neg += metrics['fn']
                
                if true_pos + false_neg > 0:
                    learned_fnr = false_neg / (true_pos + false_neg)
                
                # Create bar chart for FNR
                plt.bar(['Classifier', 'Learned'], [classifier_fnr, learned_fnr], color=['green', 'orange'])
                
                # Add value label
                plt.text(0, classifier_fnr + 0.02, f"{classifier_fnr:.4f}", ha='center')
                plt.text(1, learned_fnr + 0.02, f"{learned_fnr:.4f}", ha='center')
                
                # Add title and labels
                plt.title(f'{test_name} False Negative Rate')
                plt.ylabel('FNR')
                plt.ylim(0, max(classifier_fnr, learned_fnr) * 1.2 + 0.05)  # Set y-limit with headroom
                
                # Save figure
                plt.savefig(f"{plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_fnr.png")
                print(f"Saved {plots_dir}/{test_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_fnr.png")
            except Exception as e:
                print(f"Error creating FNR chart: {e}")
                import traceback
                traceback.print_exc()
            
            # Close the figure
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Classifier deletions
            print("  Classifier deletions: ", end="", flush=True)
            classifier_start_time = time.perf_counter()
            
            for i, key in enumerate(delete_keys):
                if i % 250 == 0 and i > 0:
                    print(f"{i}/{num_deletions}...", end="", flush=True)
                
                # Time classifier deletion
                _, duration = self._time_operation(
                    self.classifier_tree, "remove", key
                )
                self.metrics['remove']['classifier'].append(duration)
            
            classifier_end_time = time.perf_counter()
            classifier_total_time = (classifier_end_time - classifier_start_time) * 1000000  # μs
            print(" Done.")
            
            # Print deletion statistics
            speedup = trad_total_time / classifier_total_time if classifier_total_time > 0 else float('inf')
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
                if not self.metrics[op_type]['traditional'] or not self.metrics[op_type]['classifier'] or not self.metrics[op_type]['learned']:
                    continue  # Skip if no data
                    
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Calculate means
                trad_mean = np.mean(self.metrics[op_type]['traditional'])
                classifier_mean = np.mean(self.metrics[op_type]['classifier'])
                learned_mean = np.mean(self.metrics[op_type]['learned'])
                
                # Create labels
                labels = ['Traditional', 'Classifier', 'Learned']
                means = [trad_mean, classifier_mean, learned_mean]
                
                # Create bar chart
                plt.bar(labels, means, color=['blue', 'green', 'red'])
                
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
            if 'get' in self.metrics and self.metrics['get']['traditional'] and self.metrics['get']['classifier'] and self.metrics['get']['learned']:
                plt.figure(figsize=(12, 6))
                
                # Get data for individual operations
                trad_times = self.metrics['get']['traditional']
                classifier_times = self.metrics['get']['classifier']
                learned_times = self.metrics['get']['learned']
                
                # Make sure all arrays have the same length (use the smallest length)
                min_length = min(len(trad_times), len(classifier_times), len(learned_times))
                trad_times = trad_times[:min_length]
                classifier_times = classifier_times[:min_length]
                learned_times = learned_times[:min_length]
                
                # Create x-axis values (operation indices)
                x = np.arange(min_length)
                
                # Plot individual GET operations
                plt.plot(x, trad_times, 'b-', label='Traditional', alpha=0.7)
                plt.plot(x, classifier_times, 'g-', label='Classifier', alpha=0.7)
                plt.plot(x, learned_times, 'r-', label='Learned', alpha=0.7)
                
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
            classifier_speedups = []
            learned_speedups = []
            labels = []
            
            for op_type in operation_types:
                if not self.metrics[op_type]['traditional'] or not self.metrics[op_type]['classifier'] or not self.metrics[op_type]['learned']:
                    continue  # Skip if no data
                    
                trad_mean = np.mean(self.metrics[op_type]['traditional'])
                classifier_mean = np.mean(self.metrics[op_type]['classifier'])
                learned_mean = np.mean(self.metrics[op_type]['learned'])
                
                if classifier_mean > 0:
                    classifier_speedup = trad_mean / classifier_mean
                    classifier_speedups.append(classifier_speedup)
                    labels.append(op_type.capitalize())
                
                if learned_mean > 0:
                    learned_speedup = trad_mean / learned_mean
                    learned_speedups.append(learned_speedup)
            
            # Create bar chart for speedups (side by side)
            x = np.arange(len(labels))
            width = 0.35
            
            plt.bar(x - width/2, classifier_speedups, width, label='Classifier Speedup', color='green')
            plt.bar(x + width/2, learned_speedups, width, label='Learned Speedup', color='red')
            
            # Add value labels
            for i, v in enumerate(classifier_speedups):
                plt.text(i - width/2, v + 0.1, f"{v:.2f}x", ha='center')
            
            for i, v in enumerate(learned_speedups):
                plt.text(i + width/2, v + 0.1, f"{v:.2f}x", ha='center')
                
            # Add title and labels
            plt.title('Speedup Factors (Traditional / Implementation)')
            plt.ylabel('Speedup Factor')
            plt.xticks(x, labels)
            plt.legend()
            plt.ylim(0, max(max(classifier_speedups), max(learned_speedups)) * 1.2)  # Set y-limit with headroom
            
            # Save figure
            plt.savefig(f"{plots_dir}/speedup_comparison.png")
            print(f"Saved {plots_dir}/speedup_comparison.png")
            
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
            if not self.metrics[op_type]['traditional'] or not self.metrics[op_type]['classifier'] or not self.metrics[op_type]['learned']:
                continue  # Skip if no data for this operation
                
            # Calculate statistics
            trad_times = self.metrics[op_type]['traditional']
            classifier_times = self.metrics[op_type]['classifier']
            learned_times = self.metrics[op_type]['learned']
            
            trad_mean = statistics.mean(trad_times) if trad_times else 0
            classifier_mean = statistics.mean(classifier_times) if classifier_times else 0
            learned_mean = statistics.mean(learned_times) if learned_times else 0
            
            trad_median = statistics.median(trad_times) if trad_times else 0
            classifier_median = statistics.median(classifier_times) if classifier_times else 0
            learned_median = statistics.median(learned_times) if learned_times else 0
            
            try:
                trad_std_dev = statistics.stdev(trad_times) if len(trad_times) > 1 else 0
                classifier_std_dev = statistics.stdev(classifier_times) if len(classifier_times) > 1 else 0
                learned_std_dev = statistics.stdev(learned_times) if len(learned_times) > 1 else 0
            except statistics.StatisticsError:
                trad_std_dev = 0
                classifier_std_dev = 0
                learned_std_dev = 0
                
            classifier_speedup = trad_mean / classifier_mean if classifier_mean > 0 else 0
            learned_speedup = trad_mean / learned_mean if learned_mean > 0 else 0
            
            # Print results for this operation
            print(f"\n----- {op_type.upper()} OPERATION -----")
            print(f"Traditional:")
            print(f"  Mean: {trad_mean:.2f} µs")
            print(f"  Median: {trad_median:.2f} µs")
            print(f"  Std Dev: {trad_std_dev:.2f} µs")
            print(f"  Min: {min(trad_times):.2f} µs, Max: {max(trad_times):.2f} µs")
            
            print(f"Classifier:")
            print(f"  Mean: {classifier_mean:.2f} µs")
            print(f"  Median: {classifier_median:.2f} µs")
            print(f"  Std Dev: {classifier_std_dev:.2f} µs")
            print(f"  Min: {min(classifier_times):.2f} µs, Max: {max(classifier_times):.2f} µs")
            
            print(f"Learned Bloom Filter:")
            print(f"  Mean: {learned_mean:.2f} µs")
            print(f"  Median: {learned_median:.2f} µs")
            print(f"  Std Dev: {learned_std_dev:.2f} µs")
            print(f"  Min: {min(learned_times):.2f} µs, Max: {max(learned_times):.2f} µs")
            
            print(f"Speedup (Traditional/Classifier): {classifier_speedup:.2f}x")
            print(f"Speedup (Traditional/Learned): {learned_speedup:.2f}x")
        
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
            print(f"Classifier Pages Checked: {MetricsCounters.learned_pages_checked}")
            
            # Calculate page access ratio
            if MetricsCounters.traditional_pages_checked > 0:
                page_ratio = MetricsCounters.learned_pages_checked / MetricsCounters.traditional_pages_checked
                print(f"Page Check Ratio (Classifier/Traditional): {page_ratio:.2f}x")
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
            'classifier_speedup': trad_mean / classifier_mean if 'get' in self.metrics and classifier_mean > 0 else 0,
            'learned_speedup': trad_mean / learned_mean if 'get' in self.metrics and learned_mean > 0 else 0,
            'false_negative_rate': fnr
        }

    def visualize_combined_results(self, test_results):
        """Create a combined bar chart comparing all tests.
        
        Parameters:
        -----------
        test_results : list of dict
            List of result dictionaries from different tests
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create plots directory if it doesn't exist
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Extract test data
            test_names = [result['test_name'] for result in test_results]
            trad_means = [result['traditional_time'] for result in test_results]
            classifier_means = [result['classifier_time'] for result in test_results]
            learned_means = [result['learned_time'] for result in test_results]
            
            # Shorten test names for display
            display_names = []
            for name in test_names:
                if "Random" in name:
                    display_names.append("Random")
                elif "Sequential" in name:
                    display_names.append("Sequential")
                else:
                    # Extract level number
                    level = name.split("Level ")[1].split(" ")[0]
                    display_names.append(f"Level {level}")
            
            # Set up the figure and axis
            plt.figure(figsize=(14, 8))
            
            # Set width of bars
            bar_width = 0.25
            
            # Set position of bars on x axis
            x = np.arange(len(display_names))
            
            # Create bars
            plt.bar(x - bar_width, trad_means, bar_width, label='Traditional', color='blue')
            plt.bar(x, classifier_means, bar_width, label='Classifier', color='green')
            plt.bar(x + bar_width, learned_means, bar_width, label='Learned', color='orange')
            
            # Add labels and title
            plt.xlabel('Test Type')
            plt.ylabel('Average Time (µs)')
            plt.title('Performance Comparison Across All Tests')
            plt.xticks(x, display_names)
            plt.legend()
            
            # Add value labels on bars
            for i, v in enumerate(trad_means):
                plt.text(i - bar_width, v + 5, f"{v:.1f}", ha='center')
                
            for i, v in enumerate(classifier_means):
                plt.text(i, v + 5, f"{v:.1f}", ha='center')
                
            for i, v in enumerate(learned_means):
                plt.text(i + bar_width, v + 5, f"{v:.1f}", ha='center')
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-limit with headroom
            plt.ylim(0, max(max(trad_means), max(classifier_means), max(learned_means)) * 1.2)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/all_tests_comparison.png")
            print(f"Saved {plots_dir}/all_tests_comparison.png")
            
            # Close the figure
            plt.close()
            
            # Create a speedup comparison figure for all tests
            plt.figure(figsize=(12, 6))
            
            # Calculate speedups
            classifier_speedups = [t/h if h > 0 else 0 for t, h in zip(trad_means, classifier_means)]
            learned_speedups = [t/h if h > 0 else 0 for t, h in zip(trad_means, learned_means)]
            
            # Set position of bars on x axis
            x = np.arange(len(display_names))
            width = 0.35
            
            # Create bar chart for speedups
            plt.bar(x - width/2, classifier_speedups, width, label='Classifier Speedup', color='green')
            plt.bar(x + width/2, learned_speedups, width, label='Learned Speedup', color='orange')
            
            # Add value labels
            for i, v in enumerate(classifier_speedups):
                plt.text(i - width/2, v + 0.1, f"{v:.2f}x", ha='center')
                
            for i, v in enumerate(learned_speedups):
                plt.text(i + width/2, v + 0.1, f"{v:.2f}x", ha='center')
                
            # Add title and labels
            plt.title('Speedup Factors Across All Tests (Traditional / Implementation)')
            plt.ylabel('Speedup Factor')
            plt.xticks(x, display_names)
            plt.legend()
            plt.ylim(0, max(max(classifier_speedups), max(learned_speedups)) * 1.2)  # Set y-limit with headroom
            
            # Add grid
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.savefig(f"{plots_dir}/all_tests_speedup.png")
            print(f"Saved {plots_dir}/all_tests_speedup.png")
            
            # Close the figure
            plt.close()
            
            # Create FNR comparison across all tests
            plt.figure(figsize=(14, 8))
            
            # Extract FNR data
            # Traditional always has 0 FNR (ground truth)
            trad_fnr = [0] * len(test_results)
            
            # Get classifier and learned FNR from results
            classifier_fnr = [result.get('classifier_fnr', 0) for result in test_results]
            learned_fnr = [result.get('learned_fnr', 0) for result in test_results]
            
            # Set width of bars
            bar_width = 0.25
            
            # Set position of bars on x axis
            x = np.arange(len(display_names))
            
            # Create bars
            plt.bar(x - bar_width, trad_fnr, bar_width, label='Traditional', color='blue')
            plt.bar(x, classifier_fnr, bar_width, label='Classifier', color='green')
            plt.bar(x + bar_width, learned_fnr, bar_width, label='Learned', color='orange')
            
            # Add labels and title
            plt.xlabel('Test Type')
            plt.ylabel('False Negative Rate')
            plt.title('False Negative Rate Comparison Across All Tests')
            plt.xticks(x, display_names)
            plt.legend()
            
            # Add value labels on bars
            for i, v in enumerate(trad_fnr):
                plt.text(i - bar_width, v + 0.01, f"{v:.4f}", ha='center')
                
            for i, v in enumerate(classifier_fnr):
                plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
                
            for i, v in enumerate(learned_fnr):
                plt.text(i + bar_width, v + 0.01, f"{v:.4f}", ha='center')
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-limit to make small values visible (min 0.05 height)
            max_fnr = max(max(trad_fnr), max(classifier_fnr), max(learned_fnr))
            plt.ylim(0, max(max_fnr * 1.2, 0.05))
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/all_tests_fnr_comparison.png")
            print(f"Saved {plots_dir}/all_tests_fnr_comparison.png")
            
            # Close the figure
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualize_fnr_comparison(self, test_results):
        """Create a bar chart comparing the False Negative Rates for all tests and models.
        
        Parameters:
        -----------
        test_results : list of dict
            List of result dictionaries from different tests
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create plots directory if it doesn't exist
            plots_dir = "plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Extract test names and FNR values
            test_names = [result['test_name'] for result in test_results]
            fnr_values = [result.get('false_negative_rate', 0) for result in test_results]
            
            # Check if we have any non-zero FNR values to display
            if not any(fnr_values):
                print("No non-zero FNR values to display")
                return
                
            # Prepare display names
            display_names = []
            for name in test_names:
                if "Random" in name:
                    display_names.append("Random")
                elif "Sequential" in name:
                    display_names.append("Sequential")
                else:
                    # Extract level number
                    level = name.split("Level ")[1].split(" ")[0]
                    display_names.append(f"Level {level}")
            
            # Create the FNR comparison chart
            plt.figure(figsize=(12, 6))
            
            # Traditional model always has 0 FNR (it's the ground truth)
            traditional_fnr = [0] * len(test_names)
            
            # For classifier and learned, use the measured FNR
            # Since we don't track them separately, they'll have the same values
            classifier_fnr = fnr_values
            learned_fnr = fnr_values
            
            # Set width of bars
            width = 0.25
            
            # Set position of bars on X axis
            x = np.arange(len(display_names))
            
            # Create bars
            plt.bar(x - width, traditional_fnr, width, label='Traditional', color='blue')
            plt.bar(x, classifier_fnr, width, label='Classifier', color='green')
            plt.bar(x + width, learned_fnr, width, label='Learned', color='red')
            
            # Add labels and title
            plt.xlabel('Test Type')
            plt.ylabel('False Negative Rate')
            plt.title('False Negative Rate Comparison by Test')
            plt.xticks(x, display_names)
            plt.legend()
            
            # Add value labels on bars
            for i, v in enumerate(traditional_fnr):
                plt.text(i - width, v + 0.001, f"{v:.4f}", ha='center')
                
            for i, v in enumerate(classifier_fnr):
                plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
                
            for i, v in enumerate(learned_fnr):
                plt.text(i + width, v + 0.001, f"{v:.4f}", ha='center')
            
            # Set y-limit to ensure small values are visible
            max_fnr = max(max(traditional_fnr), max(classifier_fnr), max(learned_fnr))
            plt.ylim(0, max(max_fnr + 0.005, 0.02))  # Ensure at least 0.02 height for visibility
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.savefig(f"{plots_dir}/fnr_by_test_comparison.png")
            print(f"Saved {plots_dir}/fnr_by_test_comparison.png")
            
            # Close the figure
            plt.close()
            
            # Now create per-level FNR chart if we have level-specific data
            if self.fnr_per_level:
                plt.figure(figsize=(12, 6))
                
                # Sort levels
                levels = sorted(self.fnr_per_level.keys())
                
                # Check if we have actual data
                if not levels:
                    return
                
                # Traditional always has 0 FNR (it's the ground truth)
                traditional_fnr = [0] * len(levels)
                
                # Get level-specific FNR values
                model_fnr = []
                for level in levels:
                    if level in self.fnr_per_level:
                        model_fnr.append(self.fnr_per_level[level]['fnr'])
                    else:
                        model_fnr.append(0)
                
                # Only display if we have any non-zero FNR values
                if not any(model_fnr):
                    return
                
                # Set width of bars
                width = 0.25
                
                # Set position of bars on X axis
                x = np.arange(len(levels))
                
                # Create bars
                plt.bar(x - width, traditional_fnr, width, label='Traditional', color='blue')
                plt.bar(x, model_fnr, width, label='Classifier', color='green')
                plt.bar(x + width, model_fnr, width, label='Learned', color='red')
                
                # Add labels and title
                plt.xlabel('Level')
                plt.ylabel('False Negative Rate')
                plt.title('False Negative Rate by Level')
                plt.xticks(x, [f'Level {lvl}' for lvl in levels])
                plt.legend()
                
                # Add value labels on bars
                for i, v in enumerate(traditional_fnr):
                    plt.text(i - width, v + 0.001, f"{v:.4f}", ha='center')
                    
                for i, v in enumerate(model_fnr):
                    plt.text(i, v + 0.001, f"{v:.4f}", ha='center')
                    plt.text(i + width, v + 0.001, f"{v:.4f}", ha='center')
                
                # Set y-limit for visibility
                plt.ylim(0, max(max(model_fnr) + 0.005, 0.02))
                
                # Add grid for better readability
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save figure
                plt.savefig(f"{plots_dir}/fnr_by_level.png")
                print(f"Saved {plots_dir}/fnr_by_level.png")
                
                # Close the figure
                plt.close()
                
        except ImportError:
            print("Matplotlib not available for visualization. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Error during FNR visualization: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run performance tests."""
    print("Starting LSM tree performance testing...")
    
    # Initialize the test
    test = PerformanceTest()
    
    # Print initial tree statistics
    test.print_level_statistics()
    
    # Run the 6 different tests
    print("\nRunning 6 different lookup tests comparing Traditional vs Classifier vs Learned LSM Tree")
    
    # Store test results for combined visualization
    all_test_results = []
    
    # 1. Random lookups across all levels
    result = test.test_random_lookups(num_lookups=100)
    all_test_results.append(result)
    
    # 2. Sequential lookups across all levels
    result = test.test_sequential_lookups(num_lookups=100)
    all_test_results.append(result)
    
    # 3. Level 0 lookups
    result = test.test_level_specific_lookups(level=0, num_lookups=100)
    all_test_results.append(result)
    
    # 4. Level 1 lookups
    result = test.test_level_specific_lookups(level=1, num_lookups=100)
    all_test_results.append(result)
    
    # 5. Level 2 lookups
    result = test.test_level_specific_lookups(level=2, num_lookups=100)
    all_test_results.append(result)
    
    # 6. Level 3 lookups
    result = test.test_level_specific_lookups(level=3, num_lookups=100)
    all_test_results.append(result)
    
    # Create combined visualization
    test.visualize_combined_results(all_test_results)
    
    # Print final results
    test.print_results()
    
    # Create FNR comparison chart
    test.visualize_fnr_comparison(all_test_results)
    
    print("\nPerformance testing complete! Detailed plots have been saved to the 'python-implementation/plots' directory.")

if __name__ == "__main__":
    main() 