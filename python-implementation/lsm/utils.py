import os
from typing import Tuple, List
import time

class Constants:
    """Constants used throughout the LSM tree implementation."""
    DEFAULT_BUFFER_SIZE = 1 * 1024 * 1024  # 1MB buffer
    DEFAULT_LEVEL_SIZE_RATIO = 10
    DEFAULT_BLOOM_FPR = 0.01  # Base FPR for level 1
    DEFAULT_PAGE_SIZE = 4096  # 4KB
    TOMBSTONE = "__DELETED__"  # Marker for deleted keys
    
    # RocksDB sizing parameters
    KEY_SIZE_BYTES = 16  # Fixed key size (16 bytes)
    VALUE_SIZE_BYTES = 100  # Fixed value size (100 bytes)
    ENTRY_SIZE_BYTES = KEY_SIZE_BYTES + VALUE_SIZE_BYTES  # Total entry size
    
    # Monkey paper FPR scaling
    @staticmethod
    def get_level_fpr(level, base_fpr=DEFAULT_BLOOM_FPR):
        """Get the FPR for a specific level following Monkey paper guidelines.
        FPR increases by factor of 10 as we go deeper, but capped at 0.5."""
        if level == 0:  # Buffer doesn't have a bloom filter
            return 1.0
        scaled_fpr = base_fpr * (10 ** (level - 1))
        return min(scaled_fpr, 0.5)  # Cap at 0.5 (50%)

def create_run_id() -> str:
    """Create a unique run ID based on timestamp."""
    return f"run_{int(time.time() * 1000)}"

def merge_sorted_pairs(pairs1: List[Tuple[float, str]], 
                      pairs2: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """Merge two sorted lists of key-value pairs."""
    result = []
    i, j = 0, 0
    
    while i < len(pairs1) and j < len(pairs2):
        if pairs1[i][0] < pairs2[j][0]:
            result.append(pairs1[i])
            i += 1
        elif pairs1[i][0] > pairs2[j][0]:
            result.append(pairs2[j])
            j += 1
        else:
            # For same key, take the more recent value (pairs2)
            result.append(pairs2[j])
            i += 1
            j += 1
            
    # Add remaining pairs
    result.extend(pairs1[i:])
    result.extend(pairs2[j:])
    
    return result

def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def calculate_level_capacity(base_capacity: int, level: int, size_ratio: int) -> int:
    """Calculate the capacity of a level based on size ratio."""
    return base_capacity * (size_ratio ** level)

def is_tombstone(value: str) -> bool:
    """Check if a value is a tombstone marker."""
    return value == Constants.TOMBSTONE 