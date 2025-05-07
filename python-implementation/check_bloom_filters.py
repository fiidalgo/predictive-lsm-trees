import os
import pickle
import logging
from lsm import TraditionalLSMTree, Constants

# Configure logging to show info
logging.basicConfig(level=logging.INFO)

def check_bloom_filters():
    """Check Bloom filter settings across all levels."""
    print("Checking Bloom filter settings for Monkey FPR scaling...")
    
    # Load the existing tree from the data directory
    tree = TraditionalLSMTree(data_dir="data")
    
    # Print the expected FPR values for each level
    print("\n== EXPECTED FPR VALUES (Monkey Paper) ==")
    for level in range(7):
        fpr = Constants.get_level_fpr(level)
        print(f"Level {level}: {fpr:.8f} ({fpr*100:.6f}%)")
    
    # Check the actual Bloom filters in the data directory
    print("\n== ACTUAL BLOOM FILTER SETTINGS ==")
    
    for level_num in range(7):
        level_dir = os.path.join("data", f"level_{level_num}")
        if not os.path.exists(level_dir):
            print(f"Level {level_num}: No runs found")
            continue
            
        run_files = [f for f in os.listdir(level_dir) if f.startswith('run_')]
        if not run_files:
            print(f"Level {level_num}: No runs found")
            continue
            
        print(f"Level {level_num}:")
        for i, run_file in enumerate(run_files):
            run_path = os.path.join(level_dir, run_file)
            
            # Load the run metadata
            try:
                with open(run_path, 'rb') as f:
                    data = pickle.load(f)
                    bloom_filter = data.get('bloom_filter')
                    bloom_fpr = data.get('bloom_fpr')
                    
                    if bloom_filter is None:
                        print(f"  Run {i+1}: No Bloom filter")
                    else:
                        actual_fpr = bloom_filter.get_false_positive_rate()
                        expected_fpr = Constants.get_level_fpr(level_num)
                        filter_size = bloom_filter.get_size_bytes()
                        hash_count = bloom_filter.hash_count
                        
                        print(f"  Run {i+1}: FPR: {actual_fpr:.8f} ({actual_fpr*100:.6f}%)")
                        print(f"    Expected FPR: {expected_fpr:.8f} ({expected_fpr*100:.6f}%)")
                        print(f"    Size: {filter_size} bytes, Hash functions: {hash_count}")
                        
                        # Check if configured correctly
                        is_correct = abs(actual_fpr - expected_fpr) < 0.00001
                        print(f"    Configured correctly: {'✓' if is_correct else '✗'}")
                        
            except Exception as e:
                print(f"  Run {i+1}: Error loading run: {e}")
                
    # Check fence pointers as well
    print("\n== FENCE POINTER STATISTICS ==")
    for level_num in range(7):
        level_dir = os.path.join("data", f"level_{level_num}")
        if not os.path.exists(level_dir):
            print(f"Level {level_num}: No runs found")
            continue
            
        run_files = [f for f in os.listdir(level_dir) if f.startswith('run_')]
        if not run_files:
            print(f"Level {level_num}: No runs found")
            continue
            
        print(f"Level {level_num}:")
        for i, run_file in enumerate(run_files):
            run_path = os.path.join(level_dir, run_file)
            
            # Load the run metadata
            try:
                with open(run_path, 'rb') as f:
                    data = pickle.load(f)
                    num_pages = data.get('num_pages', 0)
                    page_min_keys = data.get('page_min_keys', [])
                    page_max_keys = data.get('page_max_keys', [])
                    
                    if not page_min_keys or not page_max_keys:
                        print(f"  Run {i+1}: No fence pointers found")
                    else:
                        print(f"  Run {i+1}: {num_pages} pages with fence pointers")
                        print(f"    Min/Max fence pointers: {len(page_min_keys)}/{len(page_max_keys)}")
                        
            except Exception as e:
                print(f"  Run {i+1}: Error loading run: {e}")

if __name__ == "__main__":
    check_bloom_filters() 