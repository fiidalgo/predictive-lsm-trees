import numpy as np
import time
import random
import joblib
import os
from typing import List, Tuple, Optional, Dict, Set
import pickle

# Try to import Numba for optimization, but make it optional
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    # Create a dummy decorator if Numba is not available
    def njit(func):
        return func
    NUMBA_AVAILABLE = False

# Numba-optimized prediction functions
@njit
def predict_bloom_numba(keys, min_key, max_key, example_keys_array, positive_ranges, negative_ranges):
    """Ultra-fast Numba-optimized bloom filter prediction.
    
    This function is a simplified version of the predict_bloom_batch method,
    optimized for Numba compilation.
    
    Parameters:
    -----------
    keys : numpy.ndarray
        Array of keys to predict
    min_key : float
        Minimum key in the bloom filter
    max_key : float
        Maximum key in the bloom filter
    example_keys_array : numpy.ndarray
        Array of example keys known to exist
    positive_ranges : numpy.ndarray
        Array of positive key ranges, shape (n, 2)
    negative_ranges : numpy.ndarray
        Array of negative key ranges, shape (n, 2)
        
    Returns:
    --------
    numpy.ndarray
        Array of prediction scores for each key
    """
    results = np.empty(len(keys), dtype=np.float32)
    
    for i in range(len(keys)):
        key = keys[i]
        
        # Quick min/max check - use aggressive filtering
        if key < min_key - 0.5 or key > max_key + 0.5:
            results[i] = 0.05  # Far outside range - very unlikely
            continue
        
        # Check if key is in example keys (exact match)
        in_examples = False
        for ek in example_keys_array:
            if abs(key - ek) < 1e-10:  # approximate float equality
                results[i] = 0.98  # Almost certainly exists
                in_examples = True
                break
                
        if in_examples:
            continue
                
        # Check positive ranges
        in_positive_range = False
        for j in range(len(positive_ranges)):
            min_k = positive_ranges[j, 0]
            max_k = positive_ranges[j, 1]
            if min_k <= key <= max_k:
                results[i] = 0.9  # Likely exists
                in_positive_range = True
                break
                
        if in_positive_range:
            continue
                
        # Check negative ranges
        in_negative_range = False
        for j in range(len(negative_ranges)):
            min_k = negative_ranges[j, 0]
            max_k = negative_ranges[j, 1]
            if min_k <= key <= max_k:
                results[i] = 0.05  # Very unlikely to exist
                in_negative_range = True
                break
                
        if in_negative_range:
            continue
                
        # Default case - scale by distance
        if min_key <= key <= max_key:
            results[i] = 0.25  # Below threshold
        else:
            distance = min(abs(key - min_key), abs(key - max_key))
            range_size = max_key - min_key
            if range_size > 0:
                normalized_distance = min(distance / range_size, 1.0)
                results[i] = 0.25 - normalized_distance * 0.2
            else:
                results[i] = 0.15
                
    return results

@njit
def predict_fence_numba(keys, level_min_key, level_max_key, page_count, example_keys, example_pages, key_ranges):
    """Ultra-fast Numba-optimized fence pointer prediction.
    
    Parameters:
    -----------
    keys : numpy.ndarray
        Array of keys to predict
    level_min_key : float
        Minimum key in the level
    level_max_key : float
        Maximum key in the level
    page_count : int
        Number of pages in the level
    example_keys : numpy.ndarray
        Array of example keys
    example_pages : numpy.ndarray
        Array of page numbers for example keys
    key_ranges : numpy.ndarray
        Array of key ranges and pages, shape (n, 3)
        
    Returns:
    --------
    numpy.ndarray
        Array of predicted page numbers for each key
    """
    results = np.empty(len(keys), dtype=np.int32)
    
    for i in range(len(keys)):
        key = keys[i]
        
        # Check bounds
        if key < level_min_key:
            results[i] = 0  # First page
            continue
            
        if key > level_max_key:
            results[i] = page_count - 1  # Last page
            continue
            
        # Check for exact match in examples
        match_found = False
        closest_dist = np.inf
        closest_page = 0
        
        for j in range(len(example_keys)):
            if abs(key - example_keys[j]) < 1e-10:  # approximate float equality
                results[i] = example_pages[j]
                match_found = True
                break
            # Keep track of closest key
            dist = abs(key - example_keys[j])
            if dist < closest_dist:
                closest_dist = dist
                closest_page = example_pages[j]
                
        if match_found:
            continue
            
        # Use closest example if very close
        if closest_dist < 0.1:
            results[i] = closest_page
            continue
            
        # Check key ranges
        range_found = False
        for j in range(len(key_ranges)):
            start_key = key_ranges[j, 0]
            end_key = key_ranges[j, 1]
            page = int(key_ranges[j, 2])
            
            if start_key <= key <= end_key:
                results[i] = page
                range_found = True
                break
                
        if range_found:
            continue
            
        # Linear interpolation
        key_range = level_max_key - level_min_key
        if key_range > 0:
            normalized_pos = (key - level_min_key) / key_range
            predicted_page = int(normalized_pos * page_count)
            results[i] = max(0, min(page_count - 1, predicted_page))
        else:
            results[i] = 0
            
    return results

class FastBloomFilter:
    """Ultra-fast bloom filter approximation"""
    def __init__(self, error_rate=0.01, capacity=10000):
        self.min_key = float('inf')
        self.max_key = float('-inf')
        self.positive_ranges = []  # Ranges of keys that exist
        self.negative_ranges = []  # Ranges of keys that definitely don't exist
        self.example_keys = set()  # Set of example keys for exact matches
        self.hash_ranges = {}  # Hash buckets for approximate membership
        self.buckets = 64  # Number of hash buckets
        # For accuracy calculation
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        # Flag to track if we've been trained
        self.is_trained = False
        
    def train(self, keys, exists):
        """Train the fast bloom filter"""
        # Set trained flag
        self.is_trained = True
        
        # Reset accuracy counters
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # Find positive and negative keys
        positive_keys = [k for k, e in zip(keys, exists) if e]
        negative_keys = [k for k, e in zip(keys, exists) if not e]
        
        if not positive_keys:
            return
            
        self.min_key = min(positive_keys)
        self.max_key = max(positive_keys)
        
        # Clear existing data
        self.positive_ranges = []
        self.negative_ranges = []
        self.example_keys = set()
        self.hash_ranges = {}
        
        # Store some exact keys for fast lookups (all positive samples)
        if len(positive_keys) > 0:
            self.example_keys = set(random.sample(positive_keys, min(1000, len(positive_keys))))
        
        # Create hash buckets - only for positive keys
        for k in positive_keys:
            bucket = hash(k) % self.buckets
            if bucket not in self.hash_ranges:
                self.hash_ranges[bucket] = []
            self.hash_ranges[bucket].append(k)
            
        # Find empty ranges - an important optimization for filtering
        if len(positive_keys) > 1:
            positive_keys.sort()
            
            # Add negative ranges between positive key clusters
            for i in range(len(positive_keys)-1):
                gap = positive_keys[i+1] - positive_keys[i]
                if gap > 1.0:  # Gap larger than 1.0 - a significant gap
                    # Create a negative range in the middle of the gap
                    middle = (positive_keys[i] + positive_keys[i+1]) / 2
                    radius = gap * 0.35  # Use 35% of gap size for the negative range
                    self.negative_ranges.append((middle - radius, middle + radius))
                    
        # Add additional negative ranges from known negative examples
        # This improves filter performance tremendously
        if negative_keys:
            negative_keys.sort()
            
            # Find clusters of negative keys
            cluster_start = 0
            for i in range(1, len(negative_keys)):
                if negative_keys[i] - negative_keys[i-1] > 1.0:
                    # Process this cluster if it has enough keys
                    if i - cluster_start >= 3:
                        cluster_min = negative_keys[cluster_start]
                        cluster_max = negative_keys[i-1]
                        # Avoid overlapping with positive key ranges
                        if not any(p_min <= cluster_min <= p_max or p_min <= cluster_max <= p_max 
                                  for p_min, p_max in self.positive_ranges):
                            self.negative_ranges.append((cluster_min, cluster_max))
                    cluster_start = i
            
            # Process the last cluster
            if len(negative_keys) - cluster_start >= 3:
                cluster_min = negative_keys[cluster_start]
                cluster_max = negative_keys[-1]
                # Avoid overlapping with positive key ranges
                if not any(p_min <= cluster_min <= p_max or p_min <= cluster_max <= p_max 
                          for p_min, p_max in self.positive_ranges):
                    self.negative_ranges.append((cluster_min, cluster_max))
                    
        # Find positive ranges (clusters of keys)
        if len(positive_keys) > 10:
            positive_keys.sort()
            start_idx = 0
            for i in range(1, len(positive_keys)):
                if positive_keys[i] - positive_keys[i-1] > 0.2:  # Reasonable threshold
                    if i - start_idx > 10:  # Only use large clusters
                        cluster_min = positive_keys[start_idx] - 0.1
                        cluster_max = positive_keys[i-1] + 0.1
                        self.positive_ranges.append((cluster_min, cluster_max))
                    start_idx = i
                    
            # Last cluster
            if len(positive_keys) - start_idx > 10:
                cluster_min = positive_keys[start_idx] - 0.1
                cluster_max = positive_keys[-1] + 0.1
                self.positive_ranges.append((cluster_min, cluster_max))
        
        # Test accuracy on training data - for metrics only
        for key, exists in zip(keys, exists):
            pred = self.predict(key) >= 0.5  # Use normal threshold
            if exists and pred:
                self.true_positives += 1
            elif exists and not pred:
                self.false_negatives += 1
            elif not exists and pred:
                self.false_positives += 1
            else:  # not exists and not pred
                self.true_negatives += 1
    
    def predict(self, key):
        """Predict if key exists - balancing false positives vs filtering"""
        # If not trained, return midpoint probability
        if not self.is_trained:
            return 0.5
            
        # Quick min/max check - use more aggressive filtering
        if key < self.min_key - 0.5 or key > self.max_key + 0.5:
            return 0.05  # Far outside range - very unlikely (more aggressive)
            
        # Exact match check
        if key in self.example_keys:
            return 0.98  # Almost certainly exists
            
        # Check positive ranges
        for min_k, max_k in self.positive_ranges:
            if min_k <= key <= max_k:
                return 0.9  # Likely exists
                
        # Check negative ranges - most important for filtering!
        for min_k, max_k in self.negative_ranges:
            if min_k <= key <= max_k:
                return 0.05  # Very unlikely to exist (more aggressive)
                
        # Hash bucket check
        bucket = hash(key) % self.buckets
        if bucket in self.hash_ranges:
            closest = min(self.hash_ranges[bucket], key=lambda x: abs(x - key), default=None)
            if closest is not None:
                dist = abs(closest - key)
                if dist < 0.1:
                    return 0.8  # Close to known key
                elif dist < 1.0:
                    return 0.6  # Somewhat close
        
        # Important: Use a scaled probability based on distance to known keys
        # This helps the filter actually skip runs for keys that are far from any known key
        if self.min_key <= key <= self.max_key:
            # For keys within range but not in specific buckets, use lower probability
            return 0.25  # Well below the 0.3 threshold (more aggressive)
        else:
            # For keys outside the range, scale based on distance
            distance = min(abs(key - self.min_key), abs(key - self.max_key))
            range_size = self.max_key - self.min_key
            if range_size > 0:
                normalized_distance = min(distance / range_size, 1.0)
                return 0.25 - normalized_distance * 0.2  # Scale from 0.25 down to 0.05
            else:
                return 0.15  # Default for degenerate case

    def predict_bloom_batch(self, keys):
        """Batch predict for multiple keys at once.
        
        Parameters:
        -----------
        keys : numpy.ndarray
            Array of keys to predict
            
        Returns:
        --------
        numpy.ndarray
            Array of prediction scores for each key
        """
        import numpy as np
        results = np.empty(len(keys), dtype=np.float32)
        
        # If not trained, return midpoint probability for all keys
        if not self.is_trained:
            results.fill(0.5)
            return results
        
        # Use Numba-optimized prediction if available
        if NUMBA_AVAILABLE:
            # Convert example keys to numpy array
            example_keys_array = np.array(list(self.example_keys), dtype=np.float64)
            
            # Convert positive ranges to numpy array
            positive_ranges = np.array(self.positive_ranges, dtype=np.float64) if self.positive_ranges else np.zeros((0, 2), dtype=np.float64)
            
            # Convert negative ranges to numpy array
            negative_ranges = np.array(self.negative_ranges, dtype=np.float64) if self.negative_ranges else np.zeros((0, 2), dtype=np.float64)
            
            return predict_bloom_numba(keys, self.min_key, self.max_key, example_keys_array, positive_ranges, negative_ranges)
        
        # Fall back to pure Python implementation if Numba is not available
        for i, key in enumerate(keys):
            # Quick min/max check - use more aggressive filtering
            if key < self.min_key - 0.5 or key > self.max_key + 0.5:
                results[i] = 0.05  # Far outside range - very unlikely (more aggressive)
                continue
                
            # Exact match check
            if key in self.example_keys:
                results[i] = 0.98  # Almost certainly exists
                continue
                
            # Check positive ranges
            in_positive_range = False
            for min_k, max_k in self.positive_ranges:
                if min_k <= key <= max_k:
                    results[i] = 0.9  # Likely exists
                    in_positive_range = True
                    break
            
            if in_positive_range:
                continue
                
            # Check negative ranges - most important for filtering!
            in_negative_range = False
            for min_k, max_k in self.negative_ranges:
                if min_k <= key <= max_k:
                    results[i] = 0.05  # Very unlikely to exist (more aggressive)
                    in_negative_range = True
                    break
            
            if in_negative_range:
                continue
                
            # Hash bucket check
            bucket = hash(key) % self.buckets
            if bucket in self.hash_ranges:
                closest = min(self.hash_ranges[bucket], key=lambda x: abs(x - key), default=None)
                if closest is not None:
                    dist = abs(closest - key)
                    if dist < 0.1:
                        results[i] = 0.8  # Close to known key
                        continue
                    elif dist < 1.0:
                        results[i] = 0.6  # Somewhat close
                        continue
            
            # Important: Use a scaled probability based on distance to known keys
            if self.min_key <= key <= self.max_key:
                # For keys within range but not in specific buckets, use lower probability
                results[i] = 0.25  # Well below the 0.3 threshold (more aggressive)
            else:
                # For keys outside the range, scale based on distance
                distance = min(abs(key - self.min_key), abs(key - self.max_key))
                range_size = self.max_key - self.min_key
                if range_size > 0:
                    normalized_distance = min(distance / range_size, 1.0)
                    results[i] = 0.25 - normalized_distance * 0.2  # Scale from 0.25 down to 0.05
                else:
                    results[i] = 0.15  # Default for degenerate case
        
        return results

class FastFencePointer:
    """Ultra-fast fence pointer approximation"""
    def __init__(self):
        self.key_to_page = {}  # Cache of exact key->page mappings
        self.level_data = {}   # Per-level data
        self.cache_hits = 0
        self.cache_misses = 0
        
    def train(self, keys, levels, pages):
        """Train the fast fence pointer"""
        # Store mappings by level
        for key, level, page in zip(keys, levels, pages):
            if level not in self.level_data:
                self.level_data[level] = {
                    'min_key': float('inf'),
                    'max_key': float('-inf'),
                    'key_ranges': [],  # (start_key, end_key, page) tuples
                    'examples': {},    # Sample key->page mappings
                    'page_count': 0    # Maximum page count seen
                }
                
            # Update level stats
            level_info = self.level_data[level]
            level_info['min_key'] = min(level_info['min_key'], key)
            level_info['max_key'] = max(level_info['max_key'], key)
            level_info['page_count'] = max(level_info['page_count'], page + 1)
            
            # Store example mappings (limited to avoid memory issues)
            if len(level_info['examples']) < 1000:
                level_info['examples'][key] = page
                
            # Store in global cache too (limited)
            if len(self.key_to_page) < 10000:
                self.key_to_page[(key, level)] = page
                
        # Build key ranges for each level
        for level, level_info in self.level_data.items():
            # Get keys and pages for this level
            level_keys = []
            level_pages = []
            for (k, l), p in self.key_to_page.items():
                if l == level:
                    level_keys.append(k)
                    level_pages.append(p)
                    
            # Sort by key
            key_page_pairs = sorted(zip(level_keys, level_pages))
            
            # Build ranges
            if len(key_page_pairs) > 0:
                current_page = key_page_pairs[0][1]
                range_start = key_page_pairs[0][0]
                
                for key, page in key_page_pairs[1:]:
                    if page != current_page:
                        # End of range
                        level_info['key_ranges'].append((range_start, key, current_page))
                        range_start = key
                        current_page = page
                        
                # Add last range
                level_info['key_ranges'].append((range_start, level_info['max_key'], current_page))
    
    def predict(self, key, level):
        """Predict page for key - extremely fast approximation"""
        # Check direct cache first
        if (key, level) in self.key_to_page:
            self.cache_hits += 1
            return self.key_to_page[(key, level)]
            
        self.cache_misses += 1
        
        # Check if level exists
        if level not in self.level_data:
            return 0  # Default to first page
            
        level_info = self.level_data[level]
        
        # Check bounds
        if key < level_info['min_key']:
            return 0  # Before first page
        if key > level_info['max_key']:
            return level_info['page_count'] - 1  # After last page
            
        # Check examples for exact match
        if key in level_info['examples']:
            # Add to cache
            self.key_to_page[(key, level)] = level_info['examples'][key]
            return level_info['examples'][key]
            
        # Check example for nearest match
        if level_info['examples']:
            nearest = min(level_info['examples'].keys(), key=lambda k: abs(k - key))
            if abs(nearest - key) < 0.1:  # Very close key
                page = level_info['examples'][nearest]
                # Add to cache
                self.key_to_page[(key, level)] = page
                return page
                
        # Check ranges
        for start_key, end_key, page in level_info['key_ranges']:
            if start_key <= key <= end_key:
                # Add to cache
                self.key_to_page[(key, level)] = page
                return page
                
        # Approximate with linear interpolation
        key_range = level_info['max_key'] - level_info['min_key']
        if key_range > 0:
            normalized_pos = (key - level_info['min_key']) / key_range
            predicted_page = int(normalized_pos * level_info['page_count'])
            result = max(0, min(level_info['page_count'] - 1, predicted_page))
            # Add to cache
            self.key_to_page[(key, level)] = result
            return result
        
        return 0  # Default to first page

    def predict_fence_batch(self, keys, level):
        """Batch predict pages for multiple keys at once.
        
        Parameters:
        -----------
        keys : numpy.ndarray
            Array of keys to predict
        level : int
            Level number
            
        Returns:
        --------
        numpy.ndarray
            Array of predicted page numbers for each key
        """
        import numpy as np
        results = np.empty(len(keys), dtype=np.int32)
        
        # Check if level exists
        if level not in self.level_data:
            results.fill(0)  # Default to first page
            return results
            
        level_info = self.level_data[level]
        
        # Use Numba-optimized prediction if available
        if NUMBA_AVAILABLE and len(keys) > 10:  # Only use Numba for larger batches
            # Convert example keys and pages to numpy arrays
            example_keys = np.array(list(level_info['examples'].keys()), dtype=np.float64)
            example_pages = np.array([level_info['examples'][k] for k in example_keys], dtype=np.int32)
            
            # Convert key ranges to numpy array (start_key, end_key, page)
            key_ranges = np.zeros((len(level_info['key_ranges']), 3), dtype=np.float64)
            for i, (start_key, end_key, page) in enumerate(level_info['key_ranges']):
                key_ranges[i, 0] = start_key
                key_ranges[i, 1] = end_key
                key_ranges[i, 2] = page
                
            return predict_fence_numba(
                keys, 
                level_info['min_key'], 
                level_info['max_key'],
                level_info['page_count'],
                example_keys,
                example_pages,
                key_ranges
            )
            
        # Fall back to pure Python implementation
        for i, key in enumerate(keys):
            # Check direct cache first
            if (key, level) in self.key_to_page:
                self.cache_hits += 1
                results[i] = self.key_to_page[(key, level)]
                continue
                
            self.cache_misses += 1
            
            # Check bounds
            if key < level_info['min_key']:
                results[i] = 0  # First page
                continue
                
            if key > level_info['max_key']:
                results[i] = level_info['page_count'] - 1  # Last page
                continue
                
            # Check examples for exact match
            if key in level_info['examples']:
                # Add to cache
                self.key_to_page[(key, level)] = level_info['examples'][key]
                results[i] = level_info['examples'][key]
                continue
                
            # Check example for nearest match
            if level_info['examples']:
                nearest = min(level_info['examples'].keys(), key=lambda k: abs(k - key))
                if abs(nearest - key) < 0.1:  # Very close key
                    page = level_info['examples'][nearest]
                    # Add to cache
                    self.key_to_page[(key, level)] = page
                    results[i] = page
                    continue
                    
            # Check ranges
            range_found = False
            for start_key, end_key, page in level_info['key_ranges']:
                if start_key <= key <= end_key:
                    # Add to cache
                    self.key_to_page[(key, level)] = page
                    results[i] = page
                    range_found = True
                    break
                    
            if range_found:
                continue
                
            # Approximate with linear interpolation
            key_range = level_info['max_key'] - level_info['min_key']
            if key_range > 0:
                normalized_pos = (key - level_info['min_key']) / key_range
                predicted_page = int(normalized_pos * level_info['page_count'])
                result = max(0, min(level_info['page_count'] - 1, predicted_page))
                # Add to cache
                self.key_to_page[(key, level)] = result
                results[i] = result
            else:
                results[i] = 0  # Default to first page
        
        return results

class LSMMLModels:
    """Manager class for LSM tree ML models."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Use custom ultra-fast models
        self.bloom_model = FastBloomFilter()
        self.fence_model = FastFencePointer()
        
        # Prediction cache - avoid repeated calculations
        self.bloom_cache = {}  # key -> prediction
        self.fence_cache = {}  # (key, level) -> prediction
        self.max_cache_size = 100000  # Increased cache size
        
        # Training data
        self.bloom_data = []
        self.fence_data = []
        
        # Accuracy tracking
        self.bloom_accuracy = 0.9  # Initialize with optimistic value
        self.fence_accuracy = 0.9  # Initialize with optimistic value
        
        # Timing metrics
        self.bloom_prediction_time = 0.0
        self.fence_prediction_time = 0.0
        self.bloom_prediction_count = 0
        self.fence_prediction_count = 0
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load existing models if available."""
        bloom_path = os.path.join(self.models_dir, "fast_bloom.pkl")
        fence_path = os.path.join(self.models_dir, "fast_fence.pkl")
        
        if os.path.exists(bloom_path):
            try:
                with open(bloom_path, 'rb') as f:
                    self.bloom_model = pickle.load(f)
            except:
                pass
                
        if os.path.exists(fence_path):
            try:
                with open(fence_path, 'rb') as f:
                    self.fence_model = pickle.load(f)
            except:
                pass
            
    def _save_models(self):
        """Save models to disk."""
        bloom_path = os.path.join(self.models_dir, "fast_bloom.pkl")
        fence_path = os.path.join(self.models_dir, "fast_fence.pkl")
        
        try:
            with open(bloom_path, 'wb') as f:
                pickle.dump(self.bloom_model, f)
                
            with open(fence_path, 'wb') as f:
                pickle.dump(self.fence_model, f)
        except:
            pass
        
    def add_bloom_training_data(self, key: float, exists: bool):
        """Add training data for Bloom filter model."""
        self.bloom_data.append((key, exists))
        # Clear cache when new data is added
        self.bloom_cache = {}
        
    def add_fence_training_data(self, key: float, level: int, page: int):
        """Add training data for fence pointer model."""
        self.fence_data.append((key, level, page))
        # Clear cache when new data is added
        self.fence_cache = {}
        
    def train_bloom_model(self):
        """Train Bloom filter model."""
        if not self.bloom_data:
            return
            
        start_time = time.perf_counter()
            
        # Extract keys and labels
        keys = [key for key, _ in self.bloom_data]
        exists = [e for _, e in self.bloom_data]
        
        # Train the fast model
        self.bloom_model.train(keys, exists)
        
        # Calculate accuracy - focus on sensitivity (avoiding false negatives)
        # Formula: sensitivity = TP / (TP + FN)
        true_positives = self.bloom_model.true_positives
        false_negatives = self.bloom_model.false_negatives
        false_positives = self.bloom_model.false_positives
        true_negatives = self.bloom_model.true_negatives
        
        # Calculate sensitivity (recall) - avoiding false negatives is crucial
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
        
        # Calculate general accuracy
        total = true_positives + false_negatives + false_positives + true_negatives
        general_accuracy = (true_positives + true_negatives) / total if total > 0 else 1.0
        
        # For bloom filters, we care more about sensitivity than general accuracy
        self.bloom_accuracy = sensitivity
        
        end_time = time.perf_counter()
        print(f"Fast bloom model trained in {(end_time - start_time)*1000:.2f}ms")
        print(f"Bloom filter sensitivity: {sensitivity:.2%} (avoiding false negatives)")
        print(f"Bloom filter general accuracy: {general_accuracy:.2%}")
        print(f"Bloom stats - TP: {true_positives}, FN: {false_negatives}, FP: {false_positives}, TN: {true_negatives}")
        
        # Clear cache
        self.bloom_cache = {}
        
        self._save_models()
        
    def train_fence_model(self):
        """Train fence pointer model."""
        if not self.fence_data:
            return
            
        start_time = time.perf_counter()
        
        # Extract data
        keys = [key for key, _, _ in self.fence_data]
        levels = [level for _, level, _ in self.fence_data]
        pages = [page for _, _, page in self.fence_data]
        
        # Train the fast model
        self.fence_model.train(keys, levels, pages)
        
        # Calculate accuracy 
        correct = 0
        total = 0
        errors = []
        for key, level, page in self.fence_data[-1000:]:  # Check last 1000 samples for speed
            pred = self.fence_model.predict(key, level)
            errors.append(abs(pred - page))
            if pred == page:  # Exact match
                correct += 1
            total += 1
            
        # Exact accuracy
        exact_accuracy = correct / total if total > 0 else 1.0
        
        # Mean absolute error based accuracy
        if total > 0:
            mae = sum(errors) / total
            max_page = max(pages) if pages else 1
            self.fence_accuracy = 1.0 - (mae / max_page) if max_page > 0 else 1.0
        else:
            self.fence_accuracy = 1.0
            
        end_time = time.perf_counter()
        print(f"Fast fence model trained in {(end_time - start_time)*1000:.2f}ms with {exact_accuracy:.2%} exact accuracy, {self.fence_accuracy:.2%} overall accuracy")
        
        # Clear cache
        self.fence_cache = {}
        
        self._save_models()
        
    def predict_bloom(self, key: float) -> float:
        """Predict if a key exists using Bloom filter model."""
        # Ultra-fast path: Check cache first
        if key in self.bloom_cache:
            return self.bloom_cache[key]
        
        # Time the prediction
        start_time = time.perf_counter()
        
        # OPTIMIZATION: For extreme speed, use faster range checks before model
        # Outside range checks
        if key < self.bloom_model.min_key - 0.5 or key > self.bloom_model.max_key + 0.5:
            result = 0.1  # Very unlikely to exist
            self.bloom_cache[key] = result
            
            # Record timing
            end_time = time.perf_counter()
            self.bloom_prediction_time += (end_time - start_time)
            self.bloom_prediction_count += 1
            return result
        
        # Exact match in example keys - extremely fast
        if key in self.bloom_model.example_keys:
            result = 0.95  # Almost certainly exists
            self.bloom_cache[key] = result
            
            # Record timing
            end_time = time.perf_counter()
            self.bloom_prediction_time += (end_time - start_time)
            self.bloom_prediction_count += 1
            return result
        
        # Full bloom model prediction
        result = self.bloom_model.predict(key)
        
        # Store in cache
        if len(self.bloom_cache) < self.max_cache_size:
            self.bloom_cache[key] = result
        
        # Record timing
        end_time = time.perf_counter()
        self.bloom_prediction_time += (end_time - start_time)
        self.bloom_prediction_count += 1
        
        return result
    
    def predict_bloom_batch(self, keys) -> List[float]:
        """Batch predict if keys exist using Bloom filter model.
        
        Parameters:
        -----------
        keys : List[float] or numpy.ndarray
            List of keys to predict
            
        Returns:
        --------
        numpy.ndarray
            Array of prediction scores for each key
        """
        import numpy as np
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys)
        
        # Time the batch prediction
        start_time = time.perf_counter()
        
        # Use the fast bloom model's batch prediction
        results = self.bloom_model.predict_bloom_batch(keys)
        
        # Record timing
        end_time = time.perf_counter()
        prediction_time = (end_time - start_time)
        self.bloom_prediction_time += prediction_time
        self.bloom_prediction_count += len(keys)
        
        # Update cache for future single-key lookups
        for i, key in enumerate(keys):
            if len(self.bloom_cache) < self.max_cache_size:
                self.bloom_cache[key] = results[i]
        
        return results
    
    def predict_fence(self, key: float, level: int) -> int:
        """Predict page number for a key using fence pointer model."""
        # Ultra-fast path: Check cache first
        cache_key = (key, level)
        if cache_key in self.fence_cache:
            return self.fence_cache[cache_key]
        
        # Time the prediction
        start_time = time.perf_counter()
        
        # OPTIMIZATION: Fast path for boundary keys
        if level in self.fence_model.level_data:
            level_info = self.fence_model.level_data[level]
            
            # Check bounds for extremely fast prediction
            if key <= level_info['min_key']:
                result = 0  # First page
                self.fence_cache[cache_key] = result
                
                # Record timing
                end_time = time.perf_counter()
                self.fence_prediction_time += (end_time - start_time)
                self.fence_prediction_count += 1
                return result
            
            if key >= level_info['max_key']:
                result = level_info['page_count'] - 1  # Last page
                self.fence_cache[cache_key] = result
                
                # Record timing
                end_time = time.perf_counter()
                self.fence_prediction_time += (end_time - start_time)
                self.fence_prediction_count += 1
                return result
            
            # Check exact match in example keys - extremely fast
            if key in level_info['examples']:
                result = level_info['examples'][key]
                self.fence_cache[cache_key] = result
                
                # Record timing
                end_time = time.perf_counter()
                self.fence_prediction_time += (end_time - start_time)
                self.fence_prediction_count += 1
                return result
        
        # Use fast predict
        result = self.fence_model.predict(key, level)
        
        # Store in cache
        if len(self.fence_cache) < self.max_cache_size:
            self.fence_cache[cache_key] = result
        
        # Record timing
        end_time = time.perf_counter()
        self.fence_prediction_time += (end_time - start_time)
        self.fence_prediction_count += 1
        
        return result
    
    def predict_fence_batch(self, keys, level: int) -> List[int]:
        """Batch predict page numbers for keys using fence pointer model.
        
        Parameters:
        -----------
        keys : List[float] or numpy.ndarray
            List of keys to predict
        level : int
            Level number
            
        Returns:
        --------
        numpy.ndarray
            Array of predicted page numbers for each key
        """
        import numpy as np
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys)
        
        # Time the batch prediction
        start_time = time.perf_counter()
        
        # Use the fast fence model's batch prediction
        results = self.fence_model.predict_fence_batch(keys, level)
        
        # Record timing
        end_time = time.perf_counter()
        prediction_time = (end_time - start_time)
        self.fence_prediction_time += prediction_time
        self.fence_prediction_count += len(keys)
        
        # Update cache for future single-key lookups
        for i, key in enumerate(keys):
            cache_key = (key, level)
            if len(self.fence_cache) < self.max_cache_size:
                self.fence_cache[cache_key] = results[i]
        
        return results
        
    def get_bloom_accuracy(self) -> float:
        """Get current Bloom filter model accuracy."""
        return self.bloom_accuracy
        
    def get_fence_accuracy(self) -> float:
        """Get current fence pointer model accuracy."""
        return self.fence_accuracy
        
    def get_prediction_stats(self) -> dict:
        """Get prediction timing statistics."""
        stats = {
            'bloom_prediction_time': self.bloom_prediction_time,
            'fence_prediction_time': self.fence_prediction_time,
            'bloom_prediction_count': self.bloom_prediction_count,
            'fence_prediction_count': self.fence_prediction_count,
            'avg_bloom_prediction_time': (self.bloom_prediction_time / self.bloom_prediction_count) 
                                        if self.bloom_prediction_count > 0 else 0,
            'avg_fence_prediction_time': (self.fence_prediction_time / self.fence_prediction_count)
                                        if self.fence_prediction_count > 0 else 0,
            'total_model_overhead': self.bloom_prediction_time + self.fence_prediction_time,
            'bloom_cache_size': len(self.bloom_cache),
            'fence_cache_size': len(self.fence_cache),
            'fence_direct_cache_hits': getattr(self.fence_model, 'cache_hits', 0),
            'fence_direct_cache_misses': getattr(self.fence_model, 'cache_misses', 0)
        }
        return stats 