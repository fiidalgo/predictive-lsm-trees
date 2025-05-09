import os
import time
from lsm import TraditionalLSMTree, Constants

def main():
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize LSM tree with default parameters
    tree = TraditionalLSMTree(
        data_dir=data_dir,
        buffer_size=Constants.DEFAULT_BUFFER_SIZE,
        size_ratio=Constants.DEFAULT_LEVEL_SIZE_RATIO,
        base_fpr=0.01
    )
    
    print("Inserting key-value pairs...")
    # Insert some key-value pairs
    for i in range(1000):
        key = float(i)
        value = f"value_{i}"
        tree.put(key, value)
        
    # Force flush to disk
    print("Flushing to disk...")
    tree.flush_data()
    
    # Test point lookups
    print("\nTesting point lookups:")
    test_keys = [0, 500, 999, 1000]  # Last one shouldn't exist
    for key in test_keys:
        value = tree.get(float(key))
        if value is not None:
            print(f"Found key {key}: {value}")
        else:
            print(f"Key {key} not found")
            
    # Test range queries
    print("\nTesting range queries:")
    ranges = [(0, 5), (495, 505), (995, 1005)]
    for start, end in ranges:
        print(f"\nRange [{start}, {end}]:")
        results = tree.range(float(start), float(end))
        for key, value in results:
            print(f"  {key}: {value}")
            
    # Test deletions
    print("\nTesting deletions:")
    delete_key = 500
    print(f"Deleting key {delete_key}")
    tree.remove(float(delete_key))
    
    # Verify deletion
    value = tree.get(float(delete_key))
    if value is None:
        print(f"Key {delete_key} successfully deleted")
    else:
        print(f"Key {delete_key} still exists: {value}")
        
    # Get tree statistics
    print("\nTree statistics:")
    stats = tree.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 