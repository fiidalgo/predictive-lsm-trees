import os
import cmd
import time
from lsm import TraditionalLSMTree, LearnedLSMTree, Constants

class LSMTreeShell(cmd.Cmd):
    intro = 'Welcome to the LSM Tree shell. Type help or ? to list commands.\n'
    prompt = '(lsm) '

    def __init__(self):
        super().__init__()
        self.tree_type = "traditional"  # or "learned"
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
        """Initialize the LSM tree."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize the LSM tree
        self.tree = TraditionalLSMTree(
            data_dir=self.data_dir,
            buffer_size=Constants.DEFAULT_BUFFER_SIZE,
            size_ratio=Constants.DEFAULT_LEVEL_SIZE_RATIO,
            base_fpr=0.01  # 1% false positive rate for Bloom filters
        )
        
        # Initialize operation timing metrics
        self.operation_times = {
            'put': [],
            'get': [],
            'range': [],
            'remove': [],
            'flush': []
        }
        
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
            value = str(float(value))  # Convert value to string
            
            result, duration = self._time_operation('put', self.tree.put, key, value)
            print(f"Put operation completed in {duration:.2f} μs")
            
        except ValueError:
            print("Error: Please provide key and value as numbers")
            
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
        print(f"Buffer Size: {stats['buffer_size'] / 1024:.2f} KB")
        print(f"Buffer Usage: {stats['buffer_usage'] / 1024:.2f} KB")
        print(f"Total Entries: {stats['total_entries']}")
        print(f"Levels: {stats['levels']}")
        
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
        if self.tree_type == "learned":
            print("\nML Model Performance:")
            print(f"  Bloom Filter Accuracy: {stats['bloom_accuracy']:.2%}")
            print(f"  Fence Pointer Accuracy: {stats['fence_accuracy']:.2%}")
            
    def do_switch(self, arg):
        """Switch between traditional and learned implementations: switch [traditional|learned]"""
        if arg not in ["traditional", "learned"]:
            print("Error: Please specify 'traditional' or 'learned'")
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
        
if __name__ == '__main__':
    LSMTreeShell().cmdloop() 