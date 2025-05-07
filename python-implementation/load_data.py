import os
import time
import random
import shutil
import logging
from lsm import TraditionalLSMTree, Constants

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)

def generate_random_key():
    """Generate a random key as a float (will be converted to fixed-size bytes)."""
    return random.uniform(0, 1_000_000_000)
    
def generate_random_value(key):
    """Generate a deterministic value from key."""
    # Use a simple transformation to ensure the value is deterministic for the key
    # This helps in validation later
    return f"v{key:.6f}"

def cleanup_data_directory(data_dir):
    """Clean up old test data."""
    print(f"Cleaning up old test data directory: {data_dir}")
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
            print(f"Removed old test directory: {data_dir}")
        except Exception as e:
            print(f"Warning: Failed to remove {data_dir}: {e}")
    
    # Create the data directory
    os.makedirs(data_dir, exist_ok=True)

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)} minutes {int(seconds)} seconds"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hours {int(minutes)} minutes"

def load_data():
    """Load data into a single LSM tree structure for testing."""
    print("Starting data loading...")
    
    # Define constants
    DATA_DIR = "data"
    KEY_SIZE_BYTES = 16
    VALUE_SIZE_BYTES = 100
    ENTRY_SIZE_BYTES = KEY_SIZE_BYTES + VALUE_SIZE_BYTES
    BUFFER_SIZE_BYTES = 1 * 1024 * 1024  # 1MB
    LEVEL_SIZE_RATIO = 10  # Each level is 10x larger than previous
    LOAD_BATCH_SIZE = 100000  # Increased from 10000 to 100000 for faster loading
    
    # Clean up and create data directory
    cleanup_data_directory(DATA_DIR)
    
    # Calculate entries per level based on entry size
    entries_per_buffer = BUFFER_SIZE_BYTES // ENTRY_SIZE_BYTES
    
    # Set total entries for approximately 300MB of data
    total_entries = int(2.6 * 1_000_000)  # 2.6 million entries ≈ 301.6MB
    
    print(f"Will load {total_entries:,} entries ({total_entries * ENTRY_SIZE_BYTES / (1024*1024):.1f} MB)")
    
    # Initialize LSM tree for data loading
    tree = TraditionalLSMTree(
        data_dir=DATA_DIR,
        buffer_size=BUFFER_SIZE_BYTES,
        size_ratio=LEVEL_SIZE_RATIO,
        base_fpr=0.01  # 1% false positive rate for Bloom filters
    )
    
    # Calculate expected entries per level
    level_entries = [entries_per_buffer]
    for i in range(1, 7):  # Calculate for levels 1-6
        level_entries.append(level_entries[0] * (LEVEL_SIZE_RATIO ** i))
        
    print(f"Expected entries per level: Buffer={entries_per_buffer:,}, " + 
          ", ".join([f"L{i}={entries:,}" for i, entries in enumerate(level_entries[1:], 1)]))
    
    # Generate all keys and values
    print("\nGenerating random key-value pairs...")
    gen_start_time = time.time()
    keys = []
    values = []
    
    for i in range(total_entries):
        if i % 500000 == 0 and i > 0:
            elapsed = time.time() - gen_start_time
            rate = i / elapsed
            print(f"  Generated {i:,}/{total_entries:,} key-value pairs ({i/total_entries*100:.1f}%)" +
                  f" - Rate: {rate:.0f} keys/sec")
        
        key = generate_random_key()
        value = generate_random_value(key)
        
        keys.append(key)
        values.append(value)
    
    gen_elapsed = time.time() - gen_start_time
    print(f"Key generation completed in {format_time(gen_elapsed)} ({total_entries/gen_elapsed:.0f} keys/sec)")
    
    # Track level filling
    entries_loaded = 0
    buffer_fills = 0
    level_fills = [0] * 7  # Track fills for levels 0-6
    total_mb_loaded = 0
    
    # Insert into tree
    print("\nInserting data into LSM tree...")
    load_start_time = time.time()
    last_progress_time = load_start_time
    
    # Process in larger batches for faster loading
    for i in range(0, len(keys), LOAD_BATCH_SIZE):
        batch_end = min(i + LOAD_BATCH_SIZE, len(keys))
        batch_keys = keys[i:batch_end]
        batch_values = values[i:batch_end]
        
        batch_entries = len(batch_keys)
        entries_loaded += batch_entries
        total_mb_loaded = entries_loaded * ENTRY_SIZE_BYTES / (1024*1024)
        
        # Create batch of key-value pairs
        batch_pairs = list(zip(batch_keys, batch_values))
        
        # Build key-value dictionary for bulk insertion
        batch_dict = {}
        for key, value in batch_pairs:
            batch_dict[key] = value
            
        # Bulk insert instead of individual puts
        for key, value in batch_dict.items():
            tree.put(key, value)
        
        # Show progress every 5 seconds or when a batch completes
        current_time = time.time()
        if current_time - last_progress_time > 5 or batch_end == len(keys):
            last_progress_time = current_time
            elapsed = current_time - load_start_time
            insertion_rate = entries_loaded / elapsed if elapsed > 0 else 0
            
            # Calculate and display progress
            percent_complete = entries_loaded / total_entries * 100
            print(f"  Loaded {entries_loaded:,}/{total_entries:,} entries ({percent_complete:.1f}%) - " +
                  f"{total_mb_loaded:.1f} MB - Rate: {insertion_rate:.0f} keys/sec")
        
        # Check for level filling events
        tree_stats = tree.get_stats()
        if isinstance(tree_stats.get('levels'), list):
            for level_idx, level in enumerate(tree_stats['levels']):
                entries = level.get('entries', 0)
                if entries > 0:
                    # Check if this level has new runs since last check
                    run_count = len(level.get('runs', []))
                    if level_fills[level_idx] < run_count:
                        level_fills[level_idx] = run_count
                        if level_idx == 0:
                            buffer_fills += 1
                            print(f"  Buffer full #{buffer_fills}: {entries_loaded:,}/{total_entries:,} entries loaded ({total_mb_loaded:.1f}/{total_entries * ENTRY_SIZE_BYTES / (1024*1024):.1f} MB)")
                        else:
                            print(f"  Level {level_idx} now has {run_count} runs: {entries_loaded:,}/{total_entries:,} entries loaded")
    
    # Ensure final flush
    tree.flush_data()
    load_elapsed = time.time() - load_start_time
    print(f"Data loading complete in {format_time(load_elapsed)} ({total_entries/load_elapsed:.0f} keys/sec average)")
    
    # Print level statistics
    print("\n== LEVEL STATISTICS AFTER LOADING ==")
    
    try:
        tree_stats = tree.get_stats()
        
        if isinstance(tree_stats.get('levels'), list):
            total_entries = 0
            total_size_mb = 0
            
            print(f"{'Level':<6}{'Entries':<12}{'Size (MB)':<12}{'Capacity (MB)':<15}{'Runs':<6}{'Full':<6}")
            print("-" * 60)
            
            for level_stats in tree_stats['levels']:
                level = level_stats['level']
                entries = level_stats['entries']
                size_mb = level_stats['size_mb']
                run_count = level_stats['run_count']
                is_full = level_stats['is_full']
                
                # Format capacity differently for level 0 (runs) vs others (MB)
                if level == 0:
                    capacity = f"{level_stats.get('capacity_bytes', 'N/A')} runs"
                else:
                    capacity = f"{level_stats.get('capacity_mb', 0):.2f}"
                
                total_entries += entries
                total_size_mb += size_mb
                
                print(f"{level:<6}{entries:,d} {'':<5}{size_mb:.2f} {'':<5}{capacity} {'':<6}{run_count}{'':<5}{'Yes' if is_full else 'No'}")
            
            print("-" * 60)
            print(f"{'Total':<6}{total_entries:,d} {'':<5}{total_size_mb:.2f}")
            
            # Print run details for debugging
            print("\n== DETAILED RUN INFORMATION ==")
            for level_stats in tree_stats['levels']:
                level = level_stats['level']
                if level_stats['runs']:
                    print(f"\nLevel {level} runs:")
                    for i, run in enumerate(level_stats['runs']):
                        print(f"  Run {i+1}: {run['entries']:,} entries, {run['size_mb']:.2f} MB")
    
    except Exception as e:
        print(f"Error getting tree statistics: {e}")
    
    print("\nData directory is ready for testing!")
    print(f"Path: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    load_data() 