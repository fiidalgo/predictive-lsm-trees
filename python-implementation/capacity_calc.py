"""
LSM Tree Capacity Calculator
============================
Calculates the capacity of each level in the LSM tree based on fixed key and value sizes.
"""

from lsm.utils import Constants

def calculate_level_capacities():
    """Calculate entries per level and FPR per level."""
    # Constants
    KEY_SIZE_BYTES = Constants.KEY_SIZE_BYTES  # 16 bytes
    VALUE_SIZE_BYTES = Constants.VALUE_SIZE_BYTES  # 100 bytes
    ENTRY_SIZE_BYTES = KEY_SIZE_BYTES + VALUE_SIZE_BYTES  # 116 bytes
    
    # Level sizes
    BUFFER_SIZE = Constants.DEFAULT_BUFFER_SIZE  # 1MB
    SIZE_RATIO = Constants.DEFAULT_LEVEL_SIZE_RATIO  # 10x
    
    # Calculate capacity in entries for each level
    level_capacities = []
    level_fprs = []
    
    # Level 0 (Buffer) capacity
    buffer_entries = BUFFER_SIZE // ENTRY_SIZE_BYTES
    level_capacities.append(buffer_entries)
    level_fprs.append("N/A - In memory")
    
    # Calculate for other levels
    for i in range(1, 4):  # Calculate for levels 1, 2, 3
        level_size_bytes = BUFFER_SIZE * (SIZE_RATIO ** i)
        level_capacity = level_size_bytes // ENTRY_SIZE_BYTES
        level_capacities.append(level_capacity)
        
        # Calculate FPR for this level
        fpr = Constants.get_level_fpr(i)
        level_fprs.append(f"{fpr:.4f} ({fpr*100:.2f}%)")
    
    return level_capacities, level_fprs

def print_level_information():
    """Print information about each level."""
    capacities, fprs = calculate_level_capacities()
    
    # Bloom filter bits per key by level
    # Formula: -log(FPR) / (ln(2)^2) ≈ -1.44 * log(FPR)
    import math
    bits_per_key = []
    for i in range(1, 4):  # Calculate for levels 1, 2, 3
        fpr = Constants.get_level_fpr(i)
        bpk = -1.44 * math.log2(fpr)
        bits_per_key.append(f"{bpk:.2f}")
    
    # Add a placeholder for Level 0
    bits_per_key.insert(0, "N/A")
    
    # Optimal number of hash functions
    # Formula: (m/n) * ln(2) ≈ bits_per_key * 0.693
    hash_funcs = []
    for i in range(4):
        if i == 0:
            hash_funcs.append("N/A")
        else:
            bpk = float(bits_per_key[i].replace("N/A", "0"))
            hf = bpk * 0.693
            hash_funcs.append(f"{hf:.2f}")
    
    # Print table
    print(f"\nLSM Tree Level Capacity Information")
    print(f"Entry size: {Constants.ENTRY_SIZE_BYTES} bytes ({Constants.KEY_SIZE_BYTES} bytes key + {Constants.VALUE_SIZE_BYTES} bytes value)")
    print(f"Buffer size: {Constants.DEFAULT_BUFFER_SIZE / (1024*1024):.2f} MB")
    print(f"Size ratio: {Constants.DEFAULT_LEVEL_SIZE_RATIO}x\n")
    
    print(f"{'Level':<10} {'Size (MB)':<15} {'Entries':<15} {'FPR':<20} {'Bits/Key':<15} {'Hash Funcs':<15}")
    print(f"{'-'*75}")
    
    for i in range(4):  # For levels 0, 1, 2, 3
        level_size_mb = (Constants.DEFAULT_BUFFER_SIZE * (Constants.DEFAULT_LEVEL_SIZE_RATIO ** i)) / (1024*1024)
        print(f"{i:<10} {level_size_mb:<15.2f} {capacities[i]:<15,} {fprs[i]:<20} {bits_per_key[i]:<15} {hash_funcs[i]:<15}")

if __name__ == "__main__":
    print_level_information() 