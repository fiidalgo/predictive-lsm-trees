import os
import time
import logging
import threading
from typing import List, Tuple, Optional, Dict
from .buffer import MemoryBuffer
from .run import Run
from .utils import Constants, create_run_id, merge_sorted_pairs, ensure_directory, calculate_level_capacity, is_tombstone

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class Level:
    """Level in the LSM tree."""
    
    def __init__(self, level_num: int, capacity: int):
        self.level_num = level_num
        self.capacity = capacity  # Capacity in bytes (not number of runs)
        self.runs: List[Run] = []
        
    def is_empty(self) -> bool:
        """Check if level is empty."""
        return len(self.runs) == 0
        
    def is_full(self) -> bool:
        """Check if level is full."""
        # For leveling compaction, level 0 can have multiple runs
        # Other levels should have at most one run that can grow to capacity
        if self.level_num == 0:
            return len(self.runs) >= self.capacity
        else:
            # For levels > 0, check total size in bytes against capacity
            total_size = sum(run.get_size_bytes() for run in self.runs)
            return total_size >= self.capacity
        
    def get_capacity(self) -> int:
        """Get level capacity."""
        return self.capacity
        
    def get_level_number(self) -> int:
        """Get level number."""
        return self.level_num
        
    def add_run(self, run: Run) -> None:
        """Add a run to this level."""
        self.runs.append(run)
        
    def get(self, key: bytes) -> Optional[bytes]:
        """Get value for key from this level."""
        # Search runs from newest to oldest
        for run in reversed(self.runs):
            # Use bloom filter to avoid unnecessary disk reads
            if run.might_contain(key):
                value = run.get(key)
                if value is not None:
                    return None if is_tombstone(value) else value
        return None
        
    def range(self, start_key: bytes, end_key: bytes) -> List[Tuple[bytes, bytes]]:
        """Get range of key-value pairs from this level."""
        result = []
        seen_keys = set()
        
        # Search runs from newest to oldest
        for run in reversed(self.runs):
            pairs = run.range(start_key, end_key)
            for key, value in pairs:
                if key not in seen_keys and not is_tombstone(value):
                    result.append((key, value))
                    seen_keys.add(key)
                    
        return sorted(result)

class TraditionalLSMTree:
    """Traditional LSM tree implementation."""
    
    def __init__(self, 
                 data_dir: str,
                 buffer_size: int = Constants.DEFAULT_BUFFER_SIZE,
                 size_ratio: int = Constants.DEFAULT_LEVEL_SIZE_RATIO,
                 base_fpr: float = 0.01,
                 no_compaction_on_init: bool = False):
        logger.info("Initializing TraditionalLSMTree...")
        
        logger.info("Setting up data directory...")
        self.data_dir = data_dir
        ensure_directory(data_dir)
        
        logger.info("Initializing buffer...")
        self.buffer = MemoryBuffer(buffer_size)
        
        logger.info("Setting up tree parameters...")
        self.size_ratio = size_ratio
        self.base_fpr = base_fpr
        self.no_compaction_on_init = no_compaction_on_init
        self.tree_mutex = threading.Lock()
        
        logger.info("Initializing levels...")
        self.levels: List[Level] = []
        self._init_levels()
        
        logger.info("Loading existing runs...")
        self._load_existing_runs()
        
        logger.info("TraditionalLSMTree initialization complete")
        
    def _init_levels(self) -> None:
        """Initialize LSM tree levels."""
        logger.info("Starting level initialization...")
        max_level = 7  # Can be adjusted based on needs
        
        # Calculate capacities in BYTES for each level based on size ratio
        # Level 0 capacity is in number of runs (tiering)
        level0_capacity = 4  # Allow 4 runs in level 0 before compaction
        
        # For other levels, capacity is in bytes (leveling - one large run per level)
        base_size_bytes = self.buffer.size_bytes  # Buffer size (e.g., 1MB)
        
        for i in range(max_level):
            logger.info(f"Initializing level {i}...")
            if i == 0:
                # Level 0 uses number of runs as capacity
                capacity = level0_capacity
            else:
                # Other levels use bytes as capacity with size ratio
                capacity = base_size_bytes * (self.size_ratio ** i)
            
            self.levels.append(Level(i, capacity))
            logger.info(f"Level {i} created with capacity: {capacity} {'runs' if i == 0 else 'bytes'}")
            
        logger.info("Level initialization complete")
        
    def _load_existing_runs(self) -> None:
        """Load existing runs from disk."""
        logger.info("Starting to load existing runs...")
        if not os.path.exists(self.data_dir):
            logger.info("Data directory does not exist, skipping run loading")
            return
        
        # Temporarily store the original compaction setting
        original_compaction_setting = self.no_compaction_on_init
        # Force no compaction during run loading
        self.no_compaction_on_init = True
        
        try:
            for level_dir in os.listdir(self.data_dir):
                if not level_dir.startswith('level_'):
                    continue
                    
                level_num = int(level_dir.split('_')[1])
                level_path = os.path.join(self.data_dir, level_dir)
                
                if level_num >= len(self.levels):
                    continue
                    
                logger.info(f"Loading runs from level {level_num}...")
                
                # Get all run files and sort by timestamp (newest first)
                run_files = [f for f in os.listdir(level_path) if f.startswith('run_')]
                run_files.sort(reverse=True)  # Newest first
                
                # Only keep the most recent runs up to level capacity
                max_runs = self.levels[level_num].capacity
                run_files = run_files[:max_runs]
                
                # Delete old runs
                for old_file in os.listdir(level_path):
                    if old_file.startswith('run_') and old_file not in run_files:
                        try:
                            os.remove(os.path.join(level_path, old_file))
                            logger.info(f"Deleted old run: {old_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete old run {old_file}: {e}")
                
                # Load the kept runs
                for run_file in run_files:
                    run_path = os.path.join(level_path, run_file)
                    logger.info(f"Loading run: {run_path}")
                    # Pass level-specific FPR based on Monkey paper
                    run = Run(run_path, level_num)
                    run.load()
                    self.levels[level_num].add_run(run)
        finally:
            # Restore original compaction setting
            self.no_compaction_on_init = original_compaction_setting
                    
        logger.info("Finished loading existing runs")
        
    def _string_to_fixed_bytes(self, string_value: str) -> bytes:
        """Convert string value to fixed-size bytes."""
        bytes_value = string_value.encode('utf-8')
        # Truncate or pad to expected size
        if len(bytes_value) > Constants.VALUE_SIZE_BYTES:
            return bytes_value[:Constants.VALUE_SIZE_BYTES]
        else:
            return bytes_value.ljust(Constants.VALUE_SIZE_BYTES, b'\x00')
    
    def _float_to_fixed_bytes(self, float_key: float) -> bytes:
        """Convert float key to fixed-size bytes."""
        import struct
        # Convert float to 8-byte representation first
        bytes_key = struct.pack('!d', float_key)
        # Pad to 16 bytes for fixed-size key
        return bytes_key.ljust(Constants.KEY_SIZE_BYTES, b'\x00')
        
    def put(self, key: float, value: str) -> bool:
        """Insert or update a key-value pair."""
        with self.tree_mutex:
            logger.debug(f"Putting key: {key}, value: {value}")
            
            # Convert to fixed-size byte representation
            bytes_key = self._float_to_fixed_bytes(key)
            bytes_value = self._string_to_fixed_bytes(value)
            
            if not self.buffer.put(bytes_key, bytes_value):
                self._flush_buffer()
                return self.buffer.put(bytes_key, bytes_value)
            return True
        
    def get(self, key: float) -> Optional[str]:
        """Get value for a key."""
        logger.debug(f"Looking up key: {key}")
        
        # Convert to fixed-size byte representation
        bytes_key = self._float_to_fixed_bytes(key)
        
        # Check memory buffer first
        bytes_value = self.buffer.get(bytes_key)
        if bytes_value is not None:
            # Convert back to string
            value = bytes_value.rstrip(b'\x00').decode('utf-8')
            logger.debug(f"Key found in memory buffer with value: {value}")
            return None if is_tombstone(value) else value
            
        logger.debug("Key not found in memory buffer, checking disk levels")
        
        # Check each level
        for i, level in enumerate(self.levels):
            logger.debug(f"Checking level {i}")
            bytes_value = level.get(bytes_key)
            if bytes_value is not None:
                # Convert back to string
                value = bytes_value.rstrip(b'\x00').decode('utf-8')
                logger.debug(f"Key found in level {i} with value: {value}")
                return value
                
        logger.debug("Key not found in any level")
        return None
        
    def range(self, start_key: float, end_key: float) -> List[Tuple[float, str]]:
        """Get range of key-value pairs."""
        # Convert to fixed-size byte representation
        bytes_start_key = self._float_to_fixed_bytes(start_key)
        bytes_end_key = self._float_to_fixed_bytes(end_key)
        
        result = []
        seen_keys = set()
        
        # Get from buffer first
        buffer_pairs = self.buffer.range(bytes_start_key, bytes_end_key)
        for key, value in buffer_pairs:
            string_value = value.rstrip(b'\x00').decode('utf-8')
            if not is_tombstone(string_value):
                import struct
                float_key = struct.unpack('!d', key[:8])[0]
                result.append((float_key, string_value))
                seen_keys.add(key)
                
        # Get from each level
        for level in self.levels:
            level_pairs = level.range(bytes_start_key, bytes_end_key)
            for key, value in level_pairs:
                if key not in seen_keys:
                    string_value = value.rstrip(b'\x00').decode('utf-8')
                    import struct
                    float_key = struct.unpack('!d', key[:8])[0]
                    result.append((float_key, string_value))
                    seen_keys.add(key)
                    
        return sorted(result)
        
    def remove(self, key: float) -> bool:
        """Remove a key-value pair."""
        return self.put(key, Constants.TOMBSTONE)
        
    def load_from_file(self, file_path: str) -> bool:
        """Load key-value pairs from a binary file."""
        with self.tree_mutex:
            try:
                with open(file_path, 'rb') as f:
                    # Read pairs in batches
                    batch_size = 1000
                    while True:
                        batch = []
                        for _ in range(batch_size):
                            try:
                                key = float.from_bytes(f.read(8), 'little')
                                value_len = int.from_bytes(f.read(4), 'little')
                                value = f.read(value_len).decode('utf-8')
                                batch.append((key, value))
                            except EOFError:
                                break
                                
                        if not batch:
                            break
                            
                        # Insert batch
                        for key, value in batch:
                            if not self.put(key, value):
                                logger.error("Failed to insert key-value pair during file loading")
                                return False
                                
                        # Force flush if buffer is getting full
                        if self.buffer.is_full():
                            self._flush_buffer()
                            
                return True
            except Exception as e:
                logger.error(f"Error loading from file: {e}")
                return False
                
    def flush_if_necessary(self) -> None:
        """Flush buffer if it's full."""
        if self.buffer.is_full():
            self._flush_buffer()
            
    def flush_data(self) -> bool:
        """Force flush data to disk."""
        try:
            self._flush_buffer()
            return True
        except Exception as e:
            logger.error(f"Error flushing data: {e}")
            return False
            
    def _flush_buffer(self) -> None:
        """Flush buffer to disk."""
        if not self.buffer.get_sorted_pairs():
            return
            
        # Create run directory if it doesn't exist
        level_dir = os.path.join(self.data_dir, f"level_0")
        ensure_directory(level_dir)
        
        # Create new run
        run_path = os.path.join(level_dir, create_run_id())
        run = Run(run_path, 0)
        run.create_from_pairs(self.buffer.get_sorted_pairs())
        
        # Add run to level 0
        if self.levels[0].is_full() and not self.no_compaction_on_init:
            # Skip compaction during initialization if flag is set
            self._compact(0)
        self.levels[0].add_run(run)
        
        # Clear buffer
        self.buffer.clear()
        
    def _compact(self, level: int) -> None:
        """Compact a level by merging its runs with the next level."""
        # Skip compaction during initialization if flag is set
        if self.no_compaction_on_init:
            logger.info(f"Skipping compaction of level {level} due to no_compaction_on_init flag")
            return
            
        if level >= len(self.levels) - 1:
            logger.info(f"Cannot compact level {level} as it is the last level")
            return
            
        logger.info(f"Starting compaction of level {level} into level {level + 1}")
        current_level = self.levels[level]
        next_level = self.levels[level + 1]
        
        # For level 0, merge all runs into a single run
        # For other levels, merge with the existing run in the next level
        
        # Collect all pairs from current level's runs
        merged_pairs = []
        for run in current_level.runs:
            logger.info(f"Merging run with {run.entry_count} entries from level {level}")
            pairs = run.range(b'\x00' * Constants.KEY_SIZE_BYTES, b'\xff' * Constants.KEY_SIZE_BYTES)
            merged_pairs = merge_sorted_pairs(merged_pairs, pairs)
            # Delete the run file after merging
            logger.info(f"Deleting run file: {run.file_path}")
            os.remove(run.file_path)
            
        # If next level has runs and we're using leveling strategy (level > 0),
        # merge with them
        if next_level.runs:
            logger.info(f"Level {level + 1} has {len(next_level.runs)} runs to merge with")
            for run in next_level.runs:
                logger.info(f"Merging with existing run of {run.entry_count} entries in level {level + 1}")
                pairs = run.range(b'\x00' * Constants.KEY_SIZE_BYTES, b'\xff' * Constants.KEY_SIZE_BYTES)
                merged_pairs = merge_sorted_pairs(merged_pairs, pairs)
                # Delete the run file after merging
                logger.info(f"Deleting run file: {run.file_path}")
                os.remove(run.file_path)
                
        # Clear both levels
        current_level.runs.clear()
        next_level.runs.clear()
        
        # Create new run in next level
        if merged_pairs:
            level_dir = os.path.join(self.data_dir, f"level_{level + 1}")
            ensure_directory(level_dir)
            run_path = os.path.join(level_dir, create_run_id())
            run = Run(run_path, level + 1)
            run.create_from_pairs(merged_pairs)
            logger.info(f"Created new run with {run.entry_count} entries in level {level + 1}")
            
            # Check if the new run in the next level exceeds capacity
            next_level.add_run(run)
            if next_level.is_full():
                logger.info(f"Level {level + 1} is full after adding new run, cascading compaction")
                self._compact(level + 1)
            else:
                logger.info(f"Level {level + 1} has capacity remaining, compaction complete")
            
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the tree."""
        stats = {
            'buffer_size': self.buffer.size_bytes,
            'buffer_usage': self.buffer.current_size,
            'buffer_entries': self.buffer.get_size(),
            'levels': [],
            'total_runs': sum(len(level.runs) for level in self.levels),
            'runs_per_level': [len(level.runs) for level in self.levels],
            'level_capacities': [level.capacity for level in self.levels],
        }
        
        # Calculate total entries
        total_entries = self.buffer.get_size()
        
        # Collect level statistics
        for i, level in enumerate(self.levels):
            level_runs = []
            level_entries = 0
            level_size_bytes = 0
            
            for run in level.runs:
                level_entries += run.entry_count
                size_bytes = run.get_size_bytes()
                level_size_bytes += size_bytes
                
                level_runs.append({
                    'entries': run.entry_count,
                    'size_bytes': size_bytes,
                    'size_mb': size_bytes / (1024 * 1024),
                    'file_path': run.file_path
                })
                
            total_entries += level_entries
            
            level_stats = {
                'level': i,
                'entries': level_entries,
                'size_bytes': level_size_bytes,
                'size_mb': level_size_bytes / (1024 * 1024),
                'capacity_bytes': level.capacity if i > 0 else None,  # Level 0 capacity is in runs, not bytes
                'capacity_mb': level.capacity / (1024 * 1024) if i > 0 else None,
                'runs': level_runs,
                'run_count': len(level.runs),
                'is_full': level.is_full()
            }
            
            stats['levels'].append(level_stats)
            
        stats['total_entries'] = total_entries
            
        # Add Bloom filter statistics
        bloom_stats = []
        for i, level in enumerate(self.levels):
            if i == 0:  # Skip buffer level
                continue
                
            level_fpr = Constants.get_level_fpr(i)
            level_bloom_stats = {
                'level': i,
                'fpr': level_fpr,
                'runs': len(level.runs)
            }
            bloom_stats.append(level_bloom_stats)
            
        stats['bloom_filters'] = bloom_stats
        
        return stats

    def get_from_level(self, key: float, level: int) -> Optional[str]:
        """Get value for a key from a specific level only.
        
        Parameters:
        -----------
        key : float
            The key to look up
        level : int
            The specific level to check
        
        Returns:
        --------
        Optional[str]
            The value if found in the specified level, None otherwise
        """
        # Convert to fixed-size byte representation
        bytes_key = self._float_to_fixed_bytes(key)
        
        # Check if level is valid
        if level < 0 or level >= len(self.levels):
            return None
        
        # Get from the specified level only
        bytes_value = self.levels[level].get(bytes_key)
        if bytes_value is not None:
            # Convert back to string
            value = bytes_value.rstrip(b'\x00').decode('utf-8')
            return None if is_tombstone(value) else value
        
        return None

    def get_level_for_key(self, key: float) -> Optional[int]:
        """Determine which level contains the specified key.
        
        Parameters:
        -----------
        key : float
            The key to search for
        
        Returns:
        --------
        Optional[int]
            The level number where the key is found, or None if not found in any level
        """
        # Convert to fixed-size byte representation
        bytes_key = self._float_to_fixed_bytes(key)
        
        # Check memory buffer first
        bytes_value = self.buffer.get(bytes_key)
        if bytes_value is not None:
            # Key is in the buffer
            return -1  # Use -1 to represent buffer
        
        # Check each level
        for i, level in enumerate(self.levels):
            bytes_value = level.get(bytes_key)
            if bytes_value is not None:
                return i
            
        # Key not found in any level
        return None 