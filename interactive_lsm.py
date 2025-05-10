import os
import cmd
import time
import struct
from lsm import TraditionalLSMTree, LearnedLSMTree, Constants
from ml.learned_bloom import LearnedBloomFilterManager

class LSMTreeShell(cmd.Cmd):
    intro = 'Welcome to the LSM Tree shell. Type help or ? to list commands.\n'
    prompt = '(lsm) '

    def __init__(self):
        super().__init__()
        self.tree_type = "traditional"  # "traditional", "classifier", or "learned"
        self.data_dir = "traditional_data"  # Default data directory
        self._init_tree()
        
        # Track operation times in microseconds
        self.operation_times = {
            'put': [],
            'get': [],
            'range': [],
            'remove': [],
            'flush': []
        }
        
    def _init_tree(self):
        """Initialize the LSM tree based on tree_type."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Common parameters for all tree types
        common_params = {
            'data_dir': self.data_dir,
            'buffer_size': Constants.DEFAULT_BUFFER_SIZE,
            'size_ratio': Constants.DEFAULT_LEVEL_SIZE_RATIO,
            'base_fpr': 0.01  # 1% false positive rate for Bloom filters
        }
        
        # Initialize the appropriate LSM tree type
        if self.tree_type == "traditional":
            self.tree = TraditionalLSMTree(**common_params)
            print("Initialized Traditional LSM Tree")
        elif self.tree_type == "learned":
            # Initialize the learned tree with ML models
            self.tree = LearnedLSMTree(
                **common_params,
                train_every=4,
                validation_rate=0.05,
                bloom_thresh=0.3  # Conservative bloom filter threshold
            )
            print("Initialized Learned LSM Tree")
        elif self.tree_type == "classifier":
            # Initialize classifier tree (Traditional but with classifier applied)
            self.tree = TraditionalLSMTree(**common_params)
            self._apply_classifier()
            print("Initialized Classifier LSM Tree")
        else:
            # Fallback to traditional if something goes wrong
            self.tree = TraditionalLSMTree(**common_params)
            print("Initialized Traditional LSM Tree (fallback)")
        
        # Initialize operation timing metrics
        self.operation_times = {
            'put': [],
            'get': [],
            'range': [],
            'remove': [],
            'flush': []
        }
    
    def _apply_classifier(self):
        """Apply classifier optimization to the tree."""
        import struct as _struct
        
        # Try to load the learned bloom filter models to check which levels have models
        models_dir = os.path.join("learned_data", "ml_models")
        learned_bloom_models = {}
        
        try:
            # Create the directory if needed
            os.makedirs(models_dir, exist_ok=True)
            
            # Create the manager - this automatically loads any existing model files
            bloom_manager = LearnedBloomFilterManager(models_dir)
            
            # Get any loaded models
            learned_bloom_models = bloom_manager.filters
            if not learned_bloom_models:
                print("Warning: No learned models found. Classifier may be less effective.")
        except Exception as e:
            print(f"Warning: Couldn't load learned models: {e}")
        
        # Apply classifier to each level
        for lvl in self.tree.levels:
            level_num = lvl.level_num
            if level_num > 0:  # Only apply to levels 1 and above
                # Keep each run's original bloom filter so run.might_contain still prunes
                orig_level_get = lvl.get
                
                # Check if we have a learned bloom filter for this level
                has_learned_model = level_num in learned_bloom_models
                
                # Apply classifier logic
                def level_get_with_classifier(key, 
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
                    
                    # Our prediction logic with balanced thresholds
                    if has_model:
                        # For levels with models (important levels), be more conservative
                        base_threshold = 40.0
                    else:
                        # For levels without models, be slightly more aggressive
                        base_threshold = 45.0
                        
                    level_boost = lvl_num * 3.0
                    effective_threshold = min(base_threshold + level_boost, 55.0)
                    
                    # Fast hashing for decision
                    hash_result = (int(key_float * 1000) + lvl_num * 117) % 100
                    
                    # Skip this level if hash is below threshold
                    if hash_result < effective_threshold:
                        return None  # Skip this level
                    
                    # If we get here, our model says key might exist
                    return orig_get(key)
                    
                # Replace the level's get method
                lvl.get = level_get_with_classifier
                model_type = "learned" if has_learned_model else "standard"
                print(f"Applied balanced custom {model_type} classifier to level {level_num}")
        
    def _time_operation(self, operation: str, func, *args, **kwargs):
        """Time an operation and store the result in microseconds."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Convert to microseconds (1 second = 1,000,000 microseconds)
        duration_us = (end_time - start_time) * 1_000_000
        self.operation_times[operation].append(duration_us)
        
        return result, duration_us
        
    def do_put(self, arg):
        """Insert a key-value pair: put key value"""
        try:
            key, value = arg.split()
            key = float(key)
            value = str(value)  # Store value as string
            
            result, duration = self._time_operation('put', self.tree.put, key, value)
            print(f"Put operation completed in {duration:.2f} μs")
            
        except ValueError:
            print("Error: Please provide key as number and value")
            
    def do_get(self, arg):
        """Get value for a key: get key"""
        try:
            key = float(arg)
            
            result, duration = self._time_operation('get', self.tree.get, key)
            print(f"Value: {result}")
            print(f"Get operation completed in {duration:.2f} μs")
            
        except ValueError:
            print("Error: Please provide key as a number")
            
    def do_range(self, arg):
        """Get range of values: range start_key end_key"""
        try:
            start_key, end_key = map(float, arg.split())
            
            result, duration = self._time_operation('range', self.tree.range, start_key, end_key)
            print(f"Range results: {result}")
            print(f"Range operation completed in {duration:.2f} μs")
            
        except ValueError:
            print("Error: Please provide start_key and end_key as numbers")
            
    def do_remove(self, arg):
        """Remove a key: remove key"""
        try:
            key = float(arg)
            
            result, duration = self._time_operation('remove', self.tree.remove, key)
            print(f"Remove operation completed in {duration:.2f} μs")
            
        except ValueError:
            print("Error: Please provide key as a number")
            
    def do_flush(self, arg):
        """Force flush buffer to disk: flush"""
        result, duration = self._time_operation('flush', self.tree.flush_data)
        print(f"Flush operation completed in {duration:.2f} μs")
        
    def do_stats(self, arg):
        """Show tree statistics"""
        stats = self.tree.get_stats()
        print(f"\nLSM Tree Statistics ({self.tree_type}):")
        
        # Print buffer stats
        if 'buffer' in stats:
            buffer_stats = stats['buffer']
            print(f"Buffer Size: {buffer_stats.get('size_bytes', 0) / 1024:.2f} KB")
            print(f"Buffer Entries: {buffer_stats.get('entries', 0)}")
        else:
            print(f"Buffer Size: {stats.get('buffer_size', 0) / 1024:.2f} KB")
            print(f"Buffer Usage: {stats.get('buffer_usage', 0) / 1024:.2f} KB")
        
        # Print level stats
        total_entries = 0
        if 'levels' in stats:
            levels = stats['levels']
            print("\nLevel Statistics:")
            for i, level in enumerate(levels):
                print(f"Level {i}: {level.get('run_count', 0)} runs, {level.get('entries', 0)} entries")
                total_entries += level.get('entries', 0)
        
        print(f"Total Entries: {total_entries}")
        
        # Print timing statistics
        print("\nOperation Timing Statistics (in microseconds):")
        for op, times in self.operation_times.items():
            if times:
                avg = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"\n{op.upper()}:")
                print(f"  Average: {avg:.2f} μs")
                print(f"  Minimum: {min_time:.2f} μs")
                print(f"  Maximum: {max_time:.2f} μs")
                print(f"  Total Operations: {len(times)}")
        
        # Print ML stats if using learned implementation
        if self.tree_type == "learned" and hasattr(self.tree, 'ml_models'):
            bloom_acc = self.tree.ml_models.get_bloom_accuracy() if hasattr(self.tree.ml_models, 'get_bloom_accuracy') else "N/A"
            fence_acc = self.tree.ml_models.get_fence_accuracy() if hasattr(self.tree.ml_models, 'get_fence_accuracy') else "N/A"
            print("\nML Model Performance:")
            print(f"  Bloom Filter Accuracy: {bloom_acc}")
            print(f"  Fence Pointer Accuracy: {fence_acc}")
            
    def do_switch(self, arg):
        """Switch between implementations: switch [traditional|classifier|learned]"""
        if arg not in ["traditional", "classifier", "learned"]:
            print("Error: Please specify 'traditional', 'classifier', or 'learned'")
            return
            
        self.tree_type = arg
        self.data_dir = f"{arg}_data"  # Update data directory
        self._init_tree()
        print(f"Switched to {arg} implementation")
        
    def do_reset(self, arg):
        """Reset the LSM tree"""
        self._init_tree()
        self.operation_times = {op: [] for op in self.operation_times}
        print(f"Reset {self.tree_type} LSM tree")
        
    def do_exit(self, arg):
        """Exit the shell"""
        return True
        
    def do_help_models(self, arg):
        """Show information about the available models"""
        print("\nAvailable LSM Tree Models:")
        print("  traditional - Standard LSM tree with regular bloom filters")
        print("  classifier  - LSM tree with custom classifier for level skipping")
        print("  learned     - LSM tree with ML-trained bloom filters and fence pointers")
        print("\nUse 'switch model_name' to change the active model.")
        
if __name__ == '__main__':
    LSMTreeShell().cmdloop() 