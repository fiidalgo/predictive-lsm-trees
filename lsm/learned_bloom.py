import os
import pickle
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
import struct
import numpy as np
from .run import Run
from .bloom import BloomFilter
from .traditional import TraditionalLSMTree, Level
from .utils import ensure_directory, is_tombstone, Constants, merge_sorted_pairs

# Import learned bloom filter
try:
    from ml.learned_bloom import LearnedBloomFilterManager
    LEARNED_BLOOM_AVAILABLE = True
except ImportError:
    LEARNED_BLOOM_AVAILABLE = False
    print("Learned bloom filter not available, falling back to traditional bloom filters")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LearnedBloomRun(Run):
    """Run implementation that uses learned bloom filters."""
    
    def __init__(self, file_path: str, level: int, bloom_fpr: float = None, bloom_manager=None):
        super().__init__(file_path, level, bloom_fpr)
        self.bloom_manager = bloom_manager
        self.bloom_threshold = 0.5  # Default threshold
        self.bloom_lookup_cache = {}  # Cache for bloom filter lookups
        self.cache_hits = 0
        self.cache_misses = 0
        self.might_contain_calls = 0
        self.might_contain_true = 0
        
    def might_contain(self, key, bypass_ml=False) -> bool:
        """Check if the run might contain the key using learned bloom filter.
        
        Parameters:
        -----------
        key : float or bytes
            The key to check
        bypass_ml : bool
            If True, use the traditional bloom filter even if learned filter is available
            
        Returns:
        --------
        bool
            True if the run might contain the key, False if definitely does not
        """
        self.might_contain_calls += 1
        
        # First, do a basic range check
        if self.min_key is None or self.max_key is None:
            # If we don't have min/max keys, assume it might contain the key
            return True
        
        # Convert key to float for the ML model if needed
        float_key = key
        if isinstance(key, bytes):
            try:
                float_key = struct.unpack('d', key[:8].ljust(8, b'\x00'))[0]
            except:
                float_key = 0.0  # Default on error
                
        # Check cache for this key
        cache_key = (os.path.basename(self.file_path), float_key)
        if cache_key in self.bloom_lookup_cache:
            self.cache_hits += 1
            result = self.bloom_lookup_cache[cache_key]
            if result:
                self.might_contain_true += 1
            return result
            
        self.cache_misses += 1
            
        # Basic range check
        try:
            # Handle bytes keys for min/max check
            min_key = self.min_key
            max_key = self.max_key
            
            if isinstance(min_key, bytes) and isinstance(float_key, float):
                min_key = struct.unpack('d', min_key[:8].ljust(8, b'\x00'))[0]
                
            if isinstance(max_key, bytes) and isinstance(float_key, float):
                max_key = struct.unpack('d', max_key[:8].ljust(8, b'\x00'))[0]
                
            if float_key < min_key or float_key > max_key:
                self.bloom_lookup_cache[cache_key] = False
                return False
        except:
            # On any error, be conservative
            pass
            
        # Use learned bloom filter if available
        if self.bloom_manager is not None and not bypass_ml and LEARNED_BLOOM_AVAILABLE:
            try:
                # Get prediction from learned bloom filter
                prediction = self.bloom_manager.might_contain(float_key, self.level)
                
                # Convert to boolean based on threshold
                result = prediction >= self.bloom_threshold
                
                # Cache the result
                self.bloom_lookup_cache[cache_key] = result
                
                if result:
                    self.might_contain_true += 1
                    
                return result
            except Exception as e:
                logger.warning(f"Learned bloom filter prediction failed: {e}")
                # Fall through to traditional bloom filter
                
        # Otherwise, use traditional bloom filter if available
        if self.bloom_filter is not None:
            result = self.bloom_filter.might_contain(key)
            # Cache the result
            self.bloom_lookup_cache[cache_key] = result
            if result:
                self.might_contain_true += 1
            return result
            
        # If no bloom filter available, assume it might contain the key
        self.might_contain_true += 1
        return True
    
class LearnedBloomLevel(Level):
    """Level implementation that uses learned bloom filters."""
    
    def __init__(self, level_id: int, capacity: int, bloom_manager=None):
        super().__init__(level_id, capacity)
        self.bloom_manager = bloom_manager
        self.use_learned_bloom = True
        
    def add_run(self, run: Run) -> None:
        """Add a run to the level, override to use LearnedBloomRun."""
        if isinstance(run, LearnedBloomRun):
            self.runs.append(run)
        else:
            # Convert to LearnedBloomRun if needed
            learned_run = LearnedBloomRun(run.file_path, run.level, run.bloom_fpr, self.bloom_manager)
            learned_run.min_key = run.min_key
            learned_run.max_key = run.max_key
            learned_run.bloom_filter = run.bloom_filter
            learned_run.entry_count = run.entry_count
            learned_run.page_min_keys = run.page_min_keys
            learned_run.page_max_keys = run.page_max_keys
            if hasattr(run, 'num_pages'):
                learned_run.num_pages = run.num_pages
            self.runs.append(learned_run)
            
    def get(self, key: bytes) -> Optional[bytes]:
        """Get value for key from this level using learned bloom filters."""
        # Search runs from newest to oldest
        for run in reversed(self.runs):
            # Use bloom filter to avoid unnecessary disk reads
            if run.might_contain(key, bypass_ml=not self.use_learned_bloom):
                value = run.get(key)
                if value is not None:
                    return None if is_tombstone(value) else value
        return None

class LearnedBloomTree(TraditionalLSMTree):
    """LSM tree implementation with learned bloom filters."""
    
    def __init__(self, 
                 data_dir: str,
                 buffer_size: int = Constants.DEFAULT_BUFFER_SIZE,
                 size_ratio: int = Constants.DEFAULT_LEVEL_SIZE_RATIO,
                 base_fpr: float = 0.01,
                 bloom_threshold: float = 0.5):
        # Initialize learned bloom filter manager
        self.models_dir = os.path.join(data_dir, "bloom_models")
        ensure_directory(self.models_dir)
        self.bloom_manager = LearnedBloomFilterManager(self.models_dir) if LEARNED_BLOOM_AVAILABLE else None
        self.bloom_threshold = bloom_threshold
        
        # Store buffer_size for later use
        self.buffer_size = buffer_size
        
        # Tracking metrics
        self.training_events = 0
        self.training_time = 0.0
        self.train_on_flush_count = 0
        
        # Stats for comparison
        self.bloom_stats = {
            'traditional_checks': 0,
            'learned_checks': 0,
            'traditional_bypasses': 0,
            'learned_bypasses': 0,
            'false_negatives': 0,
            'total_lookups': 0
        }
        
        # Call parent constructor
        super().__init__(data_dir, buffer_size, size_ratio, base_fpr)
        
    def _init_levels(self) -> None:
        """Initialize levels with learned bloom capabilities."""
        max_level = 7  # Same as traditional
        for i in range(max_level):
            capacity = self._calculate_level_capacity(i)
            # Use LearnedBloomLevel instead of Level
            self.levels.append(LearnedBloomLevel(i, capacity, self.bloom_manager))
    
    def _calculate_level_capacity(self, level: int) -> int:
        """Calculate capacity for a level."""
        if level == 0:
            return 4  # For level 0, capacity is number of runs before compaction
        else:
            return self.buffer_size * (self.size_ratio ** level)
        
    def _flush_buffer(self) -> None:
        """Flush in-memory buffer to disk, and collect training data."""
        # If buffer is empty, nothing to flush
        if not self.buffer.pairs:
            return
            
        # Start timing
        start_time = time.perf_counter()
            
        # Sort pairs by key
        pairs = sorted(self.buffer.pairs, key=lambda x: x[0])
        
        # Get next available run ID
        run_id = int(time.time() * 1000)
        level_0_dir = os.path.join(self.data_dir, "level_0")
        ensure_directory(level_0_dir)
        
        run_path = os.path.join(level_0_dir, f"run_{run_id}")
        level_0 = self.levels[0]
        
        # Create LearnedBloomRun instead of regular Run
        run = LearnedBloomRun(run_path, 0, None, self.bloom_manager)
        
        # Create from pairs
        run.create_from_pairs(pairs)
        
        # Add run to level 0
        level_0.add_run(run)
        
        # Clear the buffer
        self.buffer.clear()
        
        # Collect training data for bloom filters if available
        if self.bloom_manager is not None and LEARNED_BLOOM_AVAILABLE:
            self._collect_training_data(run, 0)
            
            # Periodically train bloom filters
            self.train_on_flush_count += 1
            if self.train_on_flush_count >= 5:  # Train after every 5 flushes
                self.train_bloom_filters()
                self.train_on_flush_count = 0
        
        # Check if level 0 is full and trigger compaction if needed
        if level_0.size() > level_0.capacity:
            self._compact(0)
            
        # Record metrics
        end_time = time.perf_counter()
        self.metrics['flush_time'] += (end_time - start_time)
        self.metrics['flush_count'] += 1
        
    def _compact(self, level: int) -> None:
        """Compact a level by merging runs, and collect training data."""
        # Start timing
        start_time = time.perf_counter()
        
        source_level = self.levels[level]
        target_level = self.levels[level + 1]
        
        # Skip if source level is empty
        if not source_level.runs:
            return
            
        # Get all pairs from the source level
        pairs = []
        for run in source_level.runs:
            with open(run.file_path, 'rb') as f:
                data = pickle.load(f)
                if 'pairs' in data:
                    pairs.extend(data['pairs'])
        
        # Sort and remove duplicates, keeping only the newest value for each key
        pairs = merge_sorted_pairs(pairs)
        
        # Clear the source level
        for run in source_level.runs:
            if os.path.exists(run.file_path):
                os.remove(run.file_path)
        source_level.runs = []
        
        # Check if target level has runs
        if target_level.runs:
            # Get all pairs from the target level
            target_pairs = []
            for run in target_level.runs:
                with open(run.file_path, 'rb') as f:
                    data = pickle.load(f)
                    if 'pairs' in data:
                        target_pairs.extend(data['pairs'])
            
            # Merge with pairs from source level
            pairs = merge_sorted_pairs(pairs + target_pairs)
            
            # Clear the target level
            for run in target_level.runs:
                if os.path.exists(run.file_path):
                    os.remove(run.file_path)
            target_level.runs = []
        
        # Create new run in the target level
        target_level_dir = os.path.join(self.data_dir, f"level_{level+1}")
        ensure_directory(target_level_dir)
        
        run_id = int(time.time() * 1000)
        run_path = os.path.join(target_level_dir, f"run_{run_id}")
        
        # Create LearnedBloomRun
        run = LearnedBloomRun(run_path, level + 1, None, self.bloom_manager)
        run.bloom_threshold = self.bloom_threshold
        
        # Create from pairs
        run.create_from_pairs(pairs)
        
        # Add run to target level
        target_level.add_run(run)
        
        # Collect training data for bloom filters if available
        if self.bloom_manager is not None and LEARNED_BLOOM_AVAILABLE:
            self._collect_training_data(run, level + 1)
            
            # Train bloom filters after compaction for higher levels
            if level >= 1:
                self.train_bloom_filters()
        
        # Check if target level is full and trigger compaction if needed
        if target_level.size() > target_level.capacity and level + 1 < len(self.levels) - 1:
            self._compact(level + 1)
            
        # Record metrics
        end_time = time.perf_counter()
        self.metrics['compact_time'] += (end_time - start_time)
        self.metrics['compact_count'] += 1
        
    def _collect_training_data(self, run, level: int) -> None:
        """Collect training data for learned bloom filters from a run."""
        if level == 0 or not LEARNED_BLOOM_AVAILABLE:
            return  # Level 0 doesn't use bloom filters
            
        # Load run data
        try:
            with open(run.file_path, 'rb') as f:
                data = pickle.load(f)
                pairs = data.get('pairs', [])
                
                # Sample a subset of keys if there are too many to avoid training slowdowns
                if len(pairs) > 20000:
                    pairs = random.sample(pairs, 20000)
                
                # Extract keys and convert from bytes to float
                positive_keys = []
                for key, _ in pairs:
                    if isinstance(key, bytes):
                        try:
                            # Convert bytes to float
                            float_key = struct.unpack('d', key[:8].ljust(8, b'\x00'))[0]
                            positive_keys.append(float_key)
                        except:
                            pass
                    else:
                        positive_keys.append(float(key))
                
                # Add positive examples to the bloom manager
                for key in positive_keys:
                    self.bloom_manager.add_training_data(key, level, True)
                    
                # Generate some negative examples
                min_key = min(positive_keys) if positive_keys else 0
                max_key = max(positive_keys) if positive_keys else 100
                key_range = max_key - min_key
                
                # Create a set of positive keys for fast negative checking
                positive_set = set(positive_keys)
                
                # Generate diverse negative samples
                negative_keys = []
                
                # Keys below range
                negative_keys.extend([min_key - random.random() * key_range * 0.5 for _ in range(min(1000, len(positive_keys) // 4))])
                
                # Keys above range
                negative_keys.extend([max_key + random.random() * key_range * 0.5 for _ in range(min(1000, len(positive_keys) // 4))])
                
                # Keys within range but not in positive keys
                num_in_range = min(2000, len(positive_keys) // 2)
                attempts = 0
                while len(negative_keys) < len(negative_keys) + num_in_range and attempts < num_in_range * 10:
                    attempts += 1
                    key = min_key + random.random() * key_range
                    if key not in positive_set:
                        negative_keys.append(key)
                        
                # Add negative examples to the bloom manager
                for key in negative_keys:
                    self.bloom_manager.add_training_data(key, level, False)
                    
                print(f"Collected {len(positive_keys)} positive and {len(negative_keys)} negative examples for level {level}")
                
        except Exception as e:
            logger.warning(f"Error collecting training data from run {run.file_path}: {e}")
            
    def train_bloom_filters(self) -> None:
        """Train all learned bloom filters."""
        if not LEARNED_BLOOM_AVAILABLE or self.bloom_manager is None:
            return
            
        # Start timing
        start_time = time.perf_counter()
        
        # Train filters
        self.bloom_manager.train_filters()
        
        # Update run thresholds
        for level in self.levels:
            for run in level.runs:
                if isinstance(run, LearnedBloomRun):
                    run.bloom_threshold = self.bloom_threshold
        
        # Record metrics
        end_time = time.perf_counter()
        training_time = end_time - start_time
        self.training_time += training_time
        self.training_events += 1
        
        print(f"Trained learned bloom filters in {training_time:.2f} seconds")
    
    def get(self, key: float) -> Optional[str]:
        """Get value for a key using learned bloom filters."""
        self.bloom_stats['total_lookups'] += 1
        
        # Convert to bytes for traditional lookups
        if isinstance(key, float):
            bytes_key = struct.pack('d', key)
        else:
            bytes_key = key
        
        # Check buffer first (most recent writes)
        buffer_value = self.buffer.get(bytes_key)
        if buffer_value is not None:
            return None if is_tombstone(buffer_value) else buffer_value.rstrip(b'\x00').decode('utf-8')
        
        # Check all levels - validate learned bloom filter against traditional
        # First pass with learned bloom filter
        learned_result = None
        for level in self.levels:
            # Skip empty levels
            if not level.runs:
                continue
                
            for run in reversed(level.runs):
                # Basic range check
                if bytes_key < run.min_key or bytes_key > run.max_key:
                    continue
                    
                # Use learned bloom filter if available
                if isinstance(run, LearnedBloomRun) and run.bloom_manager is not None:
                    self.bloom_stats['learned_checks'] += 1
                    
                    # Check if key might exist using learned filter
                    if run.might_contain(bytes_key, bypass_ml=False):
                        # Bloom filter says key might exist, check actual data
                        value = run.get(bytes_key)
                        if value is not None:
                            learned_result = None if is_tombstone(value) else value.rstrip(b'\x00').decode('utf-8')
                            return learned_result
                    else:
                        self.bloom_stats['learned_bypasses'] += 1
                else:
                    # Fallback to traditional bloom filter
                    self.bloom_stats['traditional_checks'] += 1
                    if run.might_contain(bytes_key):
                        value = run.get(bytes_key)
                        if value is not None:
                            learned_result = None if is_tombstone(value) else value.rstrip(b'\x00').decode('utf-8')
                            return learned_result
                    else:
                        self.bloom_stats['traditional_bypasses'] += 1
        
        # Every 100 operations, validate against traditional bloom filters to catch false negatives
        if self.bloom_stats['total_lookups'] % 100 == 0:
            # Second pass with traditional bloom filter
            traditional_result = None
            for level in self.levels:
                # Skip empty levels
                if not level.runs:
                    continue
                    
                for run in reversed(level.runs):
                    # Skip range check
                    if bytes_key < run.min_key or bytes_key > run.max_key:
                        continue
                        
                    # Always use traditional bloom filter
                    if isinstance(run, LearnedBloomRun):
                        if run.might_contain(bytes_key, bypass_ml=True):
                            value = run.get(bytes_key)
                            if value is not None:
                                traditional_result = None if is_tombstone(value) else value.rstrip(b'\x00').decode('utf-8')
                                break
                    else:
                        if run.might_contain(bytes_key):
                            value = run.get(bytes_key)
                            if value is not None:
                                traditional_result = None if is_tombstone(value) else value.rstrip(b'\x00').decode('utf-8')
                                break
                
                if traditional_result is not None:
                    break
                    
            # Check for false negatives
            if traditional_result is not None and learned_result is None:
                self.bloom_stats['false_negatives'] += 1
                return traditional_result
        
        return None
        
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the tree."""
        stats = super().get_stats()
        
        # Add learned bloom filter statistics
        if self.bloom_manager is not None and LEARNED_BLOOM_AVAILABLE:
            bloom_manager_stats = self.bloom_manager.get_stats()
            stats['learned_bloom'] = {
                'training_events': self.training_events,
                'training_time': self.training_time,
                'avg_training_time': self.training_time / self.training_events if self.training_events > 0 else 0,
                'learned_checks': self.bloom_stats['learned_checks'],
                'traditional_checks': self.bloom_stats['traditional_checks'],
                'learned_bypasses': self.bloom_stats['learned_bypasses'],
                'traditional_bypasses': self.bloom_stats['traditional_bypasses'],
                'false_negatives': self.bloom_stats['false_negatives'],
                'total_lookups': self.bloom_stats['total_lookups'],
                'false_negative_rate': self.bloom_stats['false_negatives'] / self.bloom_stats['total_lookups'] if self.bloom_stats['total_lookups'] > 0 else 0,
                'learned_bypass_rate': self.bloom_stats['learned_bypasses'] / self.bloom_stats['learned_checks'] if self.bloom_stats['learned_checks'] > 0 else 0,
                'traditional_bypass_rate': self.bloom_stats['traditional_bypasses'] / self.bloom_stats['traditional_checks'] if self.bloom_stats['traditional_checks'] > 0 else 0,
                'manager_stats': bloom_manager_stats
            }
            
            # Per-level run statistics
            run_stats = {}
            for level_id, level in enumerate(self.levels):
                level_stats = []
                for run in level.runs:
                    if isinstance(run, LearnedBloomRun):
                        level_stats.append({
                            'file': os.path.basename(run.file_path),
                            'might_contain_calls': run.might_contain_calls,
                            'might_contain_true': run.might_contain_true,
                            'rejection_rate': 1 - (run.might_contain_true / run.might_contain_calls if run.might_contain_calls > 0 else 0),
                            'cache_hits': run.cache_hits,
                            'cache_misses': run.cache_misses,
                            'cache_hit_rate': run.cache_hits / (run.cache_hits + run.cache_misses) if (run.cache_hits + run.cache_misses) > 0 else 0
                        })
                if level_stats:
                    run_stats[f'level_{level_id}'] = level_stats
                    
            stats['learned_bloom']['run_stats'] = run_stats
        
        return stats 