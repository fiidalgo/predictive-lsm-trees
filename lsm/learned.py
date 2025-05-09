import os
import time
import logging
import threading
import pickle
import random
import math
import struct
from typing import List, Tuple, Optional, Dict
from .buffer import MemoryBuffer
from .run import Run
from .utils import Constants, create_run_id, merge_sorted_pairs, ensure_directory, calculate_level_capacity, is_tombstone
from .traditional import TraditionalLSMTree, Level
from ml.models import LSMMLModels

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Change from INFO to WARNING
logger = logging.getLogger(__name__)

# Instead of relative import, use absolute import
# We'll define a variable to store global metrics
# This avoids circular imports
MetricsCounters = None  # This will be set by the test framework

# Function to register the metrics counters from outside
def register_metrics(metrics_counters):
    global MetricsCounters
    MetricsCounters = metrics_counters

# Create a learned version of Level that uses ML predictions
class LearnedLevel(Level):
    """Level with ML-only operations."""
    
    def __init__(self, level_id: int, max_runs: int, ml_models=None, bloom_thresh=0.3, fence_thresh=0.5):
        super().__init__(level_id, max_runs)
        self.ml_models = ml_models
        self.level_id = level_id
        self.bloom_thresh = bloom_thresh  # More aggressive threshold to bypass more runs
        self.fence_thresh = fence_thresh
        # Cache predictions to avoid repeated ML calls
        self.bloom_cache = {}  # key -> bool
        self.fence_cache = {}  # (run_id, key) -> page_id
        self.cache_hits = 0
        self.cache_misses = 0
        # Track false negatives (errors)
        self.false_negatives = 0
        self.bloom_false_negatives = 0
        self.fence_false_negatives = 0
        self.total_lookups = 0
        # Add bloom filter statistics
        self.bloom_checks = 0
        self.bloom_bypasses = 0
        self.fence_checks = 0
        # ML timing stats
        self.ml_timing = {
            'bloom_time': 0.0,
            'fence_time': 0.0
        }
    
    def _check_run_for_key(self, run, key):
        """Check if a run might contain a key using ML models.
        
        Parameters:
        -----------
        run : Run
            The run to check
        key : float
            The key to check
            
        Returns:
        --------
        bool
            True if the run might contain the key, False if definitely does not
        """
        # Convert key to proper format if needed
        byte_key = key
        if hasattr(self, '_float_to_fixed_bytes') and isinstance(key, float):
            byte_key = self._float_to_fixed_bytes(key)
        
        # Quick range check
        if byte_key < run.min_key or byte_key > run.max_key:
            return False
            
        # Use ML-based bloom filter
        if self.ml_models is not None:
            run_id = os.path.basename(run.file_path)
            cache_key = (run_id, key)
            
            # Check cache first
            if cache_key in self.bloom_cache:
                self.cache_hits += 1
                return self.bloom_cache[cache_key]
            
            self.cache_misses += 1
            # Use ML to predict if key exists
            bloom_prediction = self.ml_models.predict_bloom(key)
            might_contain = bloom_prediction >= self.bloom_thresh
            self.bloom_cache[cache_key] = might_contain
            
            # Update class-level counters
            self.bloom_checks += 1
            
            # Update global counters
            if MetricsCounters is not None:
                MetricsCounters.bloom_checks += 1
            
            # Check real bloom filter
            if not might_contain:
                self.bloom_bypasses += 1
                if MetricsCounters is not None:
                    MetricsCounters.bloom_bypasses += 1
                return False  # skip run
                
            # Double-check with actual bloom filter when available
            if run.bloom_filter is not None and not run.might_contain(byte_key):
                if MetricsCounters is not None:
                    MetricsCounters.bloom_bypasses += 1
                return False
            
            return True  # we *will* search this run
        
        # No ML models, use traditional bloom filter
        if MetricsCounters is not None:
            MetricsCounters.bloom_checks += 1
        result = run.might_contain(byte_key)
        if not result and MetricsCounters is not None:
            MetricsCounters.bloom_bypasses += 1
        return result
    
    def _check_run_for_key_batch(self, run, keys):
        """Check if a run might contain any of the given keys using ML predictions.
        
        Parameters:
        -----------
        keys : List[float] or numpy.ndarray
            The keys to check
            
        Returns:
        --------
        numpy.ndarray
            Boolean mask where True indicates the run might contain the key
        """
        import numpy as np
        import struct
        
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys, dtype=np.float64)
        elif keys.dtype != np.float64:
            keys = keys.astype(np.float64)
        
        # Create byte_keys for actual bloom filter checks if needed
        byte_keys = None
        if hasattr(self, '_float_to_fixed_bytes'):
            byte_keys = [self._float_to_fixed_bytes(key) for key in keys]
            
        # Convert run's min/max keys from bytes to float if needed
        min_key = run.min_key
        max_key = run.max_key
        
        # Handle different types of min/max keys
        if isinstance(min_key, bytes) and not isinstance(keys[0], bytes):
            # Ensure we have exactly 8 bytes for float64
            if len(min_key) != 8:
                # If we have more than 8 bytes, take the first 8
                # If we have less than 8 bytes, pad with zeros
                min_key = min_key[:8].ljust(8, b'\x00')
            min_key = struct.unpack('d', min_key)[0]
        elif isinstance(min_key, str):
            min_key = float(min_key)
            
        if isinstance(max_key, bytes) and not isinstance(keys[0], bytes):
            # Ensure we have exactly 8 bytes for float64
            if len(max_key) != 8:
                # If we have more than 8 bytes, take the first 8
                # If we have less than 8 bytes, pad with zeros
                max_key = max_key[:8].ljust(8, b'\x00')
            max_key = struct.unpack('d', max_key)[0]
        elif isinstance(max_key, str):
            max_key = float(max_key)
            
        # Quick range checks first (this is very fast)
        if isinstance(keys[0], bytes) and isinstance(min_key, bytes):
            # Use direct byte comparison for range check
            mask = np.array([(key >= min_key and key <= max_key) for key in keys])
        else:
            # Use numpy for faster range check with float keys
            mask = (keys >= min_key) & (keys <= max_key)
        
        # If no keys pass the range check, return empty mask
        if not mask.any():
            return mask
            
        # Only process keys that passed the range check
        remaining_keys = keys[mask]
        if byte_keys:
            remaining_byte_keys = [byte_keys[i] for i, m in enumerate(mask) if m]
        
        # Use ML-based bloom filter for remaining keys
        if self.ml_models is not None:
            run_id = os.path.basename(run.file_path)
            
            # Get batch predictions from ML model
            bloom_predictions = self.ml_models.predict_bloom_batch(remaining_keys)
            might_contain_mask = bloom_predictions >= self.bloom_thresh
            
            # Update bloom filter statistics
            self.bloom_checks += len(remaining_keys)
            self.bloom_bypasses += len(remaining_keys) - np.sum(might_contain_mask)
            
            # Update bloom filter cache for future single-key lookups
            for i, key in enumerate(remaining_keys):
                cache_key = (run_id, key)
                self.bloom_cache[cache_key] = might_contain_mask[i]
            
            # Create mask aligned with original keys
            result_mask = np.zeros_like(mask)
            result_mask[mask] = might_contain_mask
            
            # Update global counters
            if MetricsCounters is not None:
                MetricsCounters.bloom_checks += len(remaining_keys)
                MetricsCounters.bloom_bypasses += (len(remaining_keys) - np.sum(might_contain_mask))
            
            # Double-check with actual bloom filter when available
            if run.bloom_filter is not None and byte_keys:
                # Track bypasses from actual bloom filter checks
                actual_bypasses = 0
                
                # Create new mask that combines ML prediction with real bloom filter
                final_result_mask = np.zeros_like(mask)
                
                for i, check in enumerate(result_mask):
                    if check:
                        # Only check bloom filter for keys the ML model said might exist
                        byte_key = byte_keys[i]
                        if run.might_contain(byte_key):
                            final_result_mask[i] = True
                        else:
                            actual_bypasses += 1
                
                # Update global counters for actual bloom filter bypasses
                if MetricsCounters is not None:
                    MetricsCounters.bloom_bypasses += actual_bypasses
                
                # Return the combined mask
                return final_result_mask
            
            return result_mask
        else:
            # No ML models, use traditional bloom filter - less efficient
            # but we're not in this path anyway in our learned implementation
            result_mask = np.zeros_like(mask)
            
            # For each key, check with traditional bloom filter
            bloom_checks = 0
            bloom_bypasses = 0
            
            for i, key in enumerate(keys):
                if mask[i]:
                    bloom_checks += 1
                    byte_key = byte_keys[i] if byte_keys else key
                    contains = run.might_contain(byte_key)
                    result_mask[i] = contains
                    if not contains:
                        bloom_bypasses += 1
            
            # Update global counters
            if MetricsCounters is not None:
                MetricsCounters.bloom_checks += bloom_checks
                MetricsCounters.bloom_bypasses += bloom_bypasses
            
            return result_mask
    
    def get(self, key: float) -> Optional[str]:
        """Get using ONLY ML predictions, no fallbacks."""
        self.total_lookups += 1
        
        # Search runs from newest to oldest
        for run in reversed(self.runs):
            # Quick range check
            if key < run.min_key or key > run.max_key:
                continue
            
            # Use ONLY ML-based bloom filter, no fallback
            if self.ml_models is not None:
                self.bloom_checks += 1
                run_id = os.path.basename(run.file_path)
                cache_key = (run_id, key)
                
                # Check cache first
                if cache_key in self.bloom_cache:
                    self.cache_hits += 1
                    might_contain = self.bloom_cache[cache_key]
                else:
                    self.cache_misses += 1
                    # Use ML to predict if key exists
                    bloom_prediction = self.ml_models.predict_bloom(key)
                    might_contain = bloom_prediction >= self.bloom_thresh
                    self.bloom_cache[cache_key] = might_contain
                
                # Skip run if ML says key doesn't exist - THIS IS CRITICAL!
                if not might_contain:
                    self.bloom_bypasses += 1  # Count bloom filter bypasses
                    # **NO FALLBACK** - Skip this run entirely
                    continue
            else:
                # No ML models, skip ML-only logic
                continue
            
            # For small runs or runs without pages, we have to use traditional get
            # since there's no page prediction possible
            if not hasattr(run, 'num_pages') or run.num_pages <= 1:
                value = run.get(key)
                if value is not None:
                    return None if is_tombstone(value) else value
                continue
            
            # Use ONLY ML fence pointer prediction, no fallback
            self.fence_checks += 1
            # Reuse same run_id from above
            if cache_key in self.fence_cache:
                self.cache_hits += 1
                predicted_page = self.fence_cache[cache_key]
            else:
                self.cache_misses += 1
                # Use ML to predict page
                predicted_page = self.ml_models.predict_fence(key, self.level_id)
                predicted_page = min(max(0, predicted_page), run.num_pages - 1)  # Ensure valid page
                self.fence_cache[cache_key] = predicted_page
            
            # Check ONLY the predicted page, NO FALLBACK
            value = run.get_from_page(key, predicted_page)
            if value is not None:
                return None if is_tombstone(value) else value
            
            # **NO FULL-RUN FALLBACK** - treat as miss and continue
        
        return None
    
    def batch_get(self, keys):
        """Get multiple keys at once using ML predictions, with batching.
        
        Parameters:
        -----------
        keys : List[float]
            The keys to look up
            
        Returns:
        --------
        List[Optional[str]]
            List of values for each key, or None if key not found
        """
        import numpy as np
        keys_array = np.array(keys)
        results = [None] * len(keys)
        
        # Convert keys to byte representation for actual lookups
        byte_keys = [self._float_to_fixed_bytes(key) if hasattr(self, '_float_to_fixed_bytes') else key 
                    for key in keys_array]
        
        # For each run in each level, check for keys
        for run in reversed(self.runs):
            # Use ML to predict which keys might be in this run
            run_mask = self._check_run_for_key_batch(run, keys_array)
            
            # Skip this run if ML says no keys might be in it
            if not run_mask.any():
                continue
                
            # Double-check with real bloom filter when available
            if run.bloom_filter is not None:
                # Update the mask based on real bloom filter checks
                for i, (key, byte_key) in enumerate(zip(keys_array, byte_keys)):
                    if run_mask[i] and not run.might_contain(byte_key):
                        run_mask[i] = False
                        if MetricsCounters is not None:
                            MetricsCounters.bloom_bypasses += 1
                
                # If all keys were filtered out, skip this run
                if not run_mask.any():
                    continue
            
            # Get keys that might be in this run
            run_keys = keys_array[run_mask]
            run_indices = np.where(run_mask)[0]
            run_byte_keys = [byte_keys[i] for i in run_indices]
            
            # For small runs or runs without pages, we have to use traditional get
            if not hasattr(run, 'num_pages') or run.num_pages <= 1:
                for i, (key_idx, byte_key) in enumerate(zip(run_indices, run_byte_keys)):
                    value = run.get(byte_key)
                    if value is not None:
                        value_str = value.rstrip(b'\x00').decode('utf-8')
                        results[key_idx] = None if is_tombstone(value_str) else value_str
                continue
            
            # Use batched fence pointer prediction
            predicted_pages = self.ml_models.predict_fence_batch(run_keys, self.level_id)
            
            # Ensure valid page numbers
            predicted_pages = np.clip(predicted_pages, 0, run.num_pages - 1)
            
            # For each key, check only the predicted page - NO FALLBACK
            for i, (key_idx, byte_key) in enumerate(zip(run_indices, run_byte_keys)):
                value = run.get_from_page(byte_key, predicted_pages[i])
                if value is not None:
                    value_str = value.rstrip(b'\x00').decode('utf-8')
                    results[key_idx] = None if is_tombstone(value_str) else value_str
            
            # Remove found keys from the search
            mask = np.ones(len(keys), dtype=bool)
            for i, result in enumerate(results):
                if result is not None:
                    mask[i] = False
            
            # If all keys found, stop searching
            if not mask.any():
                break
                
            # Update keys_array to only include keys not found yet
            keys_array = keys_array[mask]
            byte_keys = [byte_keys[i] for i, m in enumerate(mask) if m]
            
            # If no keys left, stop searching
            if len(keys_array) == 0:
                break
        
        return results
    
    def clear_cache(self):
        """Clear prediction caches periodically to save memory."""
        if len(self.bloom_cache) > 10000 or len(self.fence_cache) > 10000:
            self.bloom_cache = {}
            self.fence_cache = {}

class LearnedLSMTree(TraditionalLSMTree):
    """LSM tree implementation with ML-only operations."""
    
    def __init__(self, 
                 data_dir: str,
                 buffer_size: int = Constants.DEFAULT_BUFFER_SIZE,
                 size_ratio: int = Constants.DEFAULT_LEVEL_SIZE_RATIO,
                 base_fpr: float = 0.01,
                 validation_rate: float = 0.05,
                 train_every: int = 4):  # How often to train models (every N flushes/compactions)
        # Initialize ML models
        models_dir = os.path.join(data_dir, "ml_models")
        ensure_directory(models_dir)
        self.ml_models = LSMMLModels(models_dir)
        
        # Training frequency settings
        self.train_every = train_every
        self._flush_count = 0
        self._compact_count = {}
        
        # Validation settings
        self.validation_rate = validation_rate
        self.validation_counter = 0
        self.ground_truth_misses = 0
        self.ground_truth_total = 0
        
        # Training statistics
        self.training_stats = {
            'bloom': {
                'samples_seen': 0,
                'last_accuracy': 0.0,
                'training_events': 0
            },
            'fence': {
                'samples_seen': 0,
                'last_accuracy': 0.0,
                'training_events': 0
            }
        }
        
        # ML timing stats
        self.ml_timing = {
            'bloom_time': 0.0,
            'fence_time': 0.0
        }
        
        # Call parent constructor
        super().__init__(data_dir, buffer_size, size_ratio, base_fpr)
    
    def _init_levels(self) -> None:
        """Initialize levels with ML capabilities."""
        max_level = 7  # Same as traditional
        for i in range(max_level):
            capacity = calculate_level_capacity(1, i, self.size_ratio)
            # Use LearnedLevel instead of Level
            self.levels.append(LearnedLevel(i, capacity, self.ml_models))
    
    def get(self, key: float) -> Optional[str]:
        """Get with ML-only search, with periodic validation."""
        # Convert to fixed-size byte representation
        bytes_key = self._float_to_fixed_bytes(key)
        
        # Check memory buffer first (same as traditional)
        bytes_value = self.buffer.get(bytes_key)
        if bytes_value is not None:
            # Convert back to string
            value = bytes_value.rstrip(b'\x00').decode('utf-8')
            return None if is_tombstone(value) else value
        
        # Clear caches periodically
        if hasattr(self, 'operation_count'):
            self.operation_count += 1
            if self.operation_count % 1000 == 0:
                for level in self.levels:
                    if hasattr(level, 'clear_cache'):
                        level.clear_cache()
        else:
            self.operation_count = 1
        
        # Occasionally do a ground truth validation
        do_validation = random.random() < self.validation_rate
        
        # Get result using ML
        ml_result = None
        for level in self.levels:
            # Skip empty levels
            if not level.runs:
                continue
                
            # Track ML timing
            start_time = time.time()
            
            # Use ML predictions
            if hasattr(level, 'ml_models') and level.ml_models is not None:
                # For each run in the level, not just the last one
                for run in reversed(level.runs):
                    # Check run range first
                    if bytes_key < run.min_key or bytes_key > run.max_key:
                        continue
                    
                    # Use ML-based bloom filter
                    level.bloom_checks += 1
                    run_id = os.path.basename(run.file_path)
                    cache_key = (run_id, key)
                    
                    # Check cache first
                    if cache_key in level.bloom_cache:
                        level.cache_hits += 1
                        might_contain = level.bloom_cache[cache_key]
                    else:
                        level.cache_misses += 1
                        # Use ML to predict if key exists
                        bloom_prediction = level.ml_models.predict_bloom(key)
                        might_contain = bloom_prediction >= level.bloom_thresh
                        level.bloom_cache[cache_key] = might_contain
                    
                    # Skip run if ML says key doesn't exist
                    if not might_contain:
                        level.bloom_bypasses += 1
                        continue
                    
                    # Use real bloom filter as verification when available
                    # This is crucial for accurate performance testing
                    if run.bloom_filter is not None and not run.might_contain(bytes_key):
                        # Real bloom filter says the key is not here
                        if MetricsCounters is not None:
                            MetricsCounters.bloom_bypasses += 1
                        continue
                        
                    # If we get here, the bloom filter thinks the key might exist
                    # For small runs or runs without pages, use traditional get
                    if not hasattr(run, 'num_pages') or run.num_pages <= 1:
                        bytes_value = run.get(bytes_key)
                        if bytes_value is not None:
                            value = bytes_value.rstrip(b'\x00').decode('utf-8')
                            ml_result = None if is_tombstone(value) else value
                            break
                        continue
                    
                    # Use ML fence pointer prediction
                    level.fence_checks += 1
                    if cache_key in level.fence_cache:
                        level.cache_hits += 1
                        predicted_page = level.fence_cache[cache_key]
                    else:
                        level.cache_misses += 1
                        predicted_page = level.ml_models.predict_fence(key, level.level_id)
                        if hasattr(run, 'num_pages') and run.num_pages > 0:
                            predicted_page = min(max(0, predicted_page), run.num_pages - 1)
                            level.fence_cache[cache_key] = predicted_page
                    
                    # Update ML timing
                    level.ml_timing['bloom_time'] += time.time() - start_time
                    
                    # Check predicted page
                    bytes_value = run.get_from_page(bytes_key, predicted_page)
                    if bytes_value is not None:
                        ml_result = bytes_value.rstrip(b'\x00').decode('utf-8')
                        break
                        
                if ml_result is not None:
                    break
            else:
                # Fallback to traditional search if no ML models
                bytes_result = level.get(bytes_key)
                if bytes_result is not None:
                    ml_result = bytes_result.rstrip(b'\x00').decode('utf-8')
                    break
                
        # Periodic validation to accurately measure false negatives
        if do_validation:
            self.ground_truth_total += 1
            # Get ground truth using traditional method
            traditional_result = None
            
            # Search all runs in all levels
            for level in self.levels:
                for run in reversed(level.runs):
                    if bytes_key < run.min_key or bytes_key > run.max_key:
                        continue
                    
                    if run.might_contain(bytes_key):
                        bytes_value = run.get(bytes_key)
                        if bytes_value is not None:
                            value = bytes_value.rstrip(b'\x00').decode('utf-8')
                            traditional_result = None if is_tombstone(value) else value
                            break
                
                if traditional_result is not None:
                    break
            
            # Check for false negatives
            if traditional_result is not None and ml_result is None:
                self.ground_truth_misses += 1
                
        # Return ML-based result
        return ml_result
    
    def _flush_buffer(self) -> None:
        """Flush buffer to disk."""
        # Call parent implementation first - identical behavior
        super()._flush_buffer()
        
        # After parent finishes, train models on the newly created run, but only every N flushes
        self._flush_count += 1
        if hasattr(self, 'levels') and self.levels and self.levels[0].runs and self._flush_count % self.train_every == 0:
            print(f"Training models after flush #{self._flush_count}")
            self._train_models(0, self.levels[0].runs[-1])
    
    def _compact(self, level: int) -> None:
        """Compact a level."""
        # Store runs for training
        next_level = level + 1
        
        # Call parent implementation - identical behavior
        super()._compact(level)
        
        # Track per-level compaction counts
        if level not in self._compact_count:
            self._compact_count[level] = 0
        self._compact_count[level] += 1
        
        # Train models on the newly created run, but only every N compactions per level
        if next_level < len(self.levels) and self.levels[next_level].runs and self._compact_count[level] % self.train_every == 0:
            print(f"Training models after level {level} compaction #{self._compact_count[level]}")
            self._train_models(next_level, self.levels[next_level].runs[-1])
    
    def _train_models(self, level: int, run: Run) -> None:
        """Train ML models with data from a run."""
        if not run or not run.entry_count:
            return
            
        # Collect training data
        key_set = set()
        
        try:
            with open(run.file_path, 'rb') as f:
                data = pickle.load(f)
                pairs = data.get('pairs', [])
                
                # Skip empty runs
                if not pairs:
                    return
                    
                # Sample training data if there are too many pairs (to reduce training time)
                if len(pairs) > 10000:
                    print(f"Sampling {10000} records from {len(pairs)} for faster training")
                    pairs = random.sample(pairs, 10000)
                
                # Calculate page boundaries
                records_per_page = run.page_size // (8 + 32)
                
                # Get key range and convert from bytes to float if needed
                min_key = run.min_key
                max_key = run.max_key
                if isinstance(min_key, bytes):
                    min_key = struct.unpack('d', min_key[:8].ljust(8, b'\x00'))[0]
                if isinstance(max_key, bytes):
                    max_key = struct.unpack('d', max_key[:8].ljust(8, b'\x00'))[0]
                
                # Add training data
                for i, (key, _) in enumerate(pairs):
                    # Convert key from bytes to float if needed
                    if isinstance(key, bytes):
                        key = struct.unpack('d', key[:8].ljust(8, b'\x00'))[0]
                    key_set.add(key)
                    page_id = i // records_per_page
                    
                    # Add to training data
                    self.ml_models.add_bloom_training_data(key, True)  # Key exists
                    self.ml_models.add_fence_training_data(key, level, page_id)
                
                # Add some negative samples for bloom filter
                if min_key is not None and max_key is not None:
                    key_range = max_key - min_key
                    num_negative_samples = min(len(pairs) // 3, 1000)
                    
                    for _ in range(num_negative_samples):
                        fake_key = random.uniform(min_key - key_range * 0.1, max_key + key_range * 0.1)
                        if fake_key not in key_set:
                            self.ml_models.add_bloom_training_data(fake_key, False)
                
                # Train models
                self.ml_models.train_bloom_model()
                self.ml_models.train_fence_model()
                
                # Update statistics
                self.training_stats['bloom']['samples_seen'] += len(pairs)
                self.training_stats['fence']['samples_seen'] += len(pairs)
                self.training_stats['bloom']['training_events'] += 1
                self.training_stats['fence']['training_events'] += 1
                self.training_stats['bloom']['last_accuracy'] = self.ml_models.get_bloom_accuracy()
                self.training_stats['fence']['last_accuracy'] = self.ml_models.get_fence_accuracy()
        except Exception as e:
            logger.warning(f"Error training models: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the LSM tree and ML models."""
        stats = super().get_stats()
        
        # Add ML model statistics
        stats['ml_models'] = {
            'bloom_accuracy': self.training_stats['bloom']['last_accuracy'],
            'fence_accuracy': self.training_stats['fence']['last_accuracy'],
            'bloom_samples': self.training_stats['bloom']['samples_seen'],
            'fence_samples': self.training_stats['fence']['samples_seen'],
            'bloom_training_events': self.training_stats['bloom']['training_events'],
            'fence_training_events': self.training_stats['fence']['training_events']
        }
        
        # Add ML usage statistics
        total_lookups = 0
        false_negatives = 0
        bloom_false_negatives = 0
        fence_false_negatives = 0
        cache_hits = 0
        cache_misses = 0
        bloom_checks = 0
        bloom_bypasses = 0
        fence_checks = 0
        
        for level in self.levels:
            if hasattr(level, 'total_lookups'):
                total_lookups += level.total_lookups
            if hasattr(level, 'false_negatives'):
                false_negatives += level.false_negatives
            if hasattr(level, 'bloom_false_negatives'):
                bloom_false_negatives += level.bloom_false_negatives
            if hasattr(level, 'fence_false_negatives'):
                fence_false_negatives += level.fence_false_negatives
            if hasattr(level, 'cache_hits'):
                cache_hits += level.cache_hits
            if hasattr(level, 'cache_misses'):
                cache_misses += level.cache_misses
            if hasattr(level, 'bloom_checks'):
                bloom_checks += level.bloom_checks
            if hasattr(level, 'bloom_bypasses'):
                bloom_bypasses += level.bloom_bypasses
            if hasattr(level, 'fence_checks'):
                fence_checks += level.fence_checks
        
        # Use ground truth validation if we have enough samples
        if hasattr(self, 'ground_truth_total') and self.ground_truth_total > 0:
            fn_rate = (self.ground_truth_misses / self.ground_truth_total * 100)
        else:
            fn_rate = (false_negatives / total_lookups * 100) if total_lookups > 0 else 0
                
        stats['ml_usage'] = {
            'total_lookups': total_lookups,
            'false_negatives': false_negatives,
            'bloom_false_negatives': bloom_false_negatives,
            'fence_false_negatives': fence_false_negatives, 
            'false_negative_rate': fn_rate,
            'validation_samples': getattr(self, 'ground_truth_total', 0),
            'validation_misses': getattr(self, 'ground_truth_misses', 0),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0,
            'bloom_checks': bloom_checks,
            'bloom_bypasses': bloom_bypasses,
            'bloom_bypass_rate': (bloom_bypasses / bloom_checks * 100) if bloom_checks > 0 else 0,
            'fence_checks': fence_checks
        }
        
        return stats 