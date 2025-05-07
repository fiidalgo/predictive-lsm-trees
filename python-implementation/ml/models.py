import numpy as np
import time
import random
import joblib
import os
from typing import List, Tuple, Optional, Dict, Set
import pickle
from collections import Counter

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
    """Fast ML-based bloom filter predictor.
    
    This model predicts EXACTLY ONE LEVEL where a key is most likely to be found,
    allowing us to skip bloom filter checks for all other levels.
    """
    
    def __init__(self):
        # Classification model that predicts which specific level contains the key
        self.model = None
        self.trained = False
        self.accuracy = 0.0
        self.stats = {'correct': 0, 'total': 0}
        self.level_distribution = {}  # Track predictions per level
        self.min_max_keys = {}  # Store min/max keys per level for feature engineering
        
    def train(self, X, y_levels):
        """Train the bloom filter model to predict which single level contains a key.
        
        X: List of keys as features
        y_levels: List of level numbers where each key exists (or -1 if key doesn't exist)
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        import numpy as np
        import time
        
        if len(X) == 0 or len(y_levels) == 0:
            print("Error: No training data provided")
            return False
        
        # Compute min/max keys per level for better feature engineering
        keys_by_level = {}
        for i, level in enumerate(y_levels):
            if level != -1:  # Skip non-existent keys
                if level not in keys_by_level:
                    keys_by_level[level] = []
                keys_by_level[level].append(X[i])
        
        # Store min/max for each level
        self.min_max_keys = {}
        for level, keys in keys_by_level.items():
            self.min_max_keys[level] = (min(keys), max(keys))
            
        # Create engineered features
        X_features = self._create_features(X)
            
        # Format input data for model
        X_formatted = np.array(X_features)
        y_levels = np.array(y_levels)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X_formatted, y_levels, test_size=0.2, random_state=42, stratify=y_levels)
        
        print(f"Training level predictor with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        # Count the number of examples per level for reporting
        unique_levels, counts = np.unique(y_train, return_counts=True)
        level_counts = {int(level): int(count) for level, count in zip(unique_levels, counts)}
        print(f"Level distribution in training data: {level_counts}")
        
        start_time = time.time()
        
        # Train a more powerful classifier to predict which level contains the key
        self.model = GradientBoostingClassifier(
            n_estimators=200,        # More trees for better accuracy
            max_depth=6,             # Deeper trees to capture more patterns
            learning_rate=0.1,       # Good default learning rate
            random_state=42,
            subsample=0.8            # Use 80% of samples per tree for robustness
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        self.trained = True
        y_pred = self.model.predict(X_val)
        
        # Calculate accuracy metrics
        correct = np.sum(y_pred == y_val)
        total = len(y_val)
        accuracy = correct / total if total > 0 else 0
        
        # Track level distribution in predictions
        pred_levels, pred_counts = np.unique(y_pred, return_counts=True)
        self.level_distribution = {int(level): int(count) for level, count in zip(pred_levels, pred_counts)}
        
        # Store stats
        self.stats = {'correct': int(correct), 'total': int(total)}
        self.accuracy = accuracy
        
        end_time = time.time()
        training_time_ms = (end_time - start_time) * 1000
        
        print(f"Level predictor trained in {training_time_ms:.2f}ms")
        print(f"Level prediction accuracy: {accuracy:.2%}")
        print(f"Predicted level distribution: {self.level_distribution}")
        
        # Print feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            sorted_indices = np.argsort(importances)[::-1]
            print("Feature importances:")
            for i in sorted_indices[:5]:  # Top 5 features
                print(f"  {feature_names[i]}: {importances[i]:.4f}")
                
        return True
    
    def _create_features(self, keys):
        """Create rich features for the model from raw keys.
        
        Better features help the model make more accurate predictions.
        """
        import numpy as np
        features = []
        
        for key in keys:
            key_float = float(key)
            
            # Basic features
            feature_vector = [key_float]
            
            # Add distance to min/max of each level features
            for level, (min_key, max_key) in self.min_max_keys.items():
                # Normalized position within level range
                if max_key > min_key:
                    position_in_level = (key_float - min_key) / (max_key - min_key)
                    feature_vector.append(position_in_level)
                    
                    # Add distance to min and max
                    dist_to_min = abs(key_float - min_key)
                    dist_to_max = abs(key_float - max_key)
                    feature_vector.append(dist_to_min)
                    feature_vector.append(dist_to_max)
                    
                    # Is key within level range?
                    in_range = 1.0 if min_key <= key_float <= max_key else 0.0
                    feature_vector.append(in_range)
            
            features.append(feature_vector)
            
        return features
    
    def predict(self, key):
        """Predict which single level most likely contains the key.
        
        Returns:
        --------
        int: The predicted level number, or -1 if key doesn't exist in any level
        """
        if not self.trained or not self.model:
            # Default conservative behavior - check level 0
            return 0
        
        # Create features for prediction
        X = self._create_features([key])
        
        try:
            # Predict exactly one level
            predicted_level = int(self.model.predict(X)[0])
            return predicted_level
        except Exception as e:
            # On error, default to conservative behavior
            print(f"Error in level prediction: {e}")
            return 0  # Default to checking level 0

class FastFencePointer:
    """Fast ML-based fence pointer predictor.
    
    This model predicts which page in a run likely contains a key,
    allowing us to directly check the most probable page first.
    """
    
    def __init__(self):
        # Regression model for predicting page numbers
        self.model = None
        self.trained = False
        self.accuracy = 0.0
        self.page_counts = {}  # Track page counts per level
        self.exact_hits = 0
        self.total_predictions = 0
        self.near_hits = 0  # Within ±1 page
        
    def train(self, X, y):
        """Train the fence pointer model on X features and y labels.
        
        X should include [key, level]
        y should be the page_id
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        import numpy as np
        import time
        
        if len(X) == 0 or len(y) == 0:
            print("Error: No training data provided")
            return False
            
        # Convert inputs to numpy arrays if they aren't already
        X = np.array(X)
        y = np.array(y)
            
        # Extract levels and page counts
        levels = np.array([x[1] for x in X])
        unique_levels = np.unique(levels)
        
        # Calculate page counts per level
        for level in unique_levels:
            level_indices = np.where(levels == level)[0]
            level_pages = np.array([y[i] for i in level_indices])
            self.page_counts[int(level)] = int(np.max(level_pages) + 1)  # +1 because 0-indexed
            
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training fence pointer with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        start_time = time.time()
        
        # Train a more powerful random forest model for fence pointers
        self.model = RandomForestRegressor(
            n_estimators=100,     # Number of trees
            max_depth=10,         # Deeper trees
            random_state=42,
            n_jobs=-1             # Use all cores
        )
        self.model.fit(X_train, y_train)
        
        end_time = time.time()
        training_time_ms = (end_time - start_time) * 1000
        
        # Evaluate on validation set
        self.trained = True
        y_pred = self.model.predict(X_val)
        
        # Calculate accuracy metrics - fence pointers need to predict exact page
        exact_matches = 0
        near_matches = 0  # Within ±1 page
        
        for i in range(len(y_val)):
            true_page = int(y_val[i])
            pred_page = int(round(y_pred[i]))
            
            if true_page == pred_page:
                exact_matches += 1
                
            if abs(true_page - pred_page) <= 1:
                near_matches += 1
                
        # Calculate accuracies
        self.exact_hits = exact_matches
        self.total_predictions = len(y_val)
        self.near_hits = near_matches
        
        exact_accuracy = exact_matches / len(y_val) if len(y_val) > 0 else 0
        near_accuracy = near_matches / len(y_val) if len(y_val) > 0 else 0
        
        self.accuracy = near_accuracy  # We consider near matches to be accurate enough
        
        print(f"Fence pointer trained in {training_time_ms:.2f}ms")
        print(f"Fence pointer accuracy: exact={exact_accuracy:.2%}, near={near_accuracy:.2%}")
        
        return True
    
    def predict(self, key, level):
        """Predict which page in a run likely contains the key."""
        if not self.trained or not self.model:
            # Default to page 0 if not trained
            return 0
            
        # Format key and level for model input
        X = self._format_input(key, level)
        
        # Get prediction
        try:
            # Predict and round to nearest integer
            page = int(round(self.model.predict([X])[0]))
            
            # Ensure prediction is within bounds
            if level in self.page_counts:
                max_page = self.page_counts[level] - 1
                page = max(0, min(page, max_page))
                
            return page
            
        except Exception as e:
            # On error, default to page 0
            print(f"Error in fence prediction: {e}")
            return 0
            
    def _format_input(self, key, level):
        """Format input for model prediction."""
        if isinstance(key, bytes):
            import struct
            try:
                # Convert bytes to float
                key = struct.unpack('!d', key[:8])[0]
            except:
                # On error, return a default value
                return [0.0, level]
                
        # Return feature vector [key, level]
        return [float(key), int(level)]

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
        bloom_path = os.path.join(self.models_dir, "bloom_model.pkl")
        fence_path = os.path.join(self.models_dir, "fence_model.pkl")
        
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
        bloom_path = os.path.join(self.models_dir, "bloom_model.pkl")
        fence_path = os.path.join(self.models_dir, "fence_model.pkl")
        
        try:
            with open(bloom_path, 'wb') as f:
                pickle.dump(self.bloom_model, f)
                
            with open(fence_path, 'wb') as f:
                pickle.dump(self.fence_model, f)
        except:
            pass
        
    def add_bloom_training_data(self, key: float, level: int):
        """Add training data for Bloom filter model.
        
        Instead of True/False, we now use the level number:
        - Level 0, 1, 2, 3, etc: The key exists in this specific level
        - Level -1: The key doesn't exist in any level
        """
        self.bloom_data.append((key, level))
        # Clear cache when new data is added
        self.bloom_cache = {}
        
    def add_fence_training_data(self, key: float, level: int, page: int):
        """Add training data for fence pointer model."""
        self.fence_data.append((key, level, page))
        # Clear cache when new data is added
        self.fence_cache = {}
        
    def train_bloom_model(self):
        """Train Bloom filter model to predict which level contains a key."""
        if not self.bloom_data:
            return
            
        start_time = time.perf_counter()
            
        # Extract keys and levels
        keys = [key for key, _ in self.bloom_data]
        levels = [level for _, level in self.bloom_data]
        
        # Balance the dataset - get counts per level
        level_counts = Counter(levels)
        print(f"Original level distribution: {dict(level_counts)}")
        
        # Calculate class weights for training
        class_weights = {}
        total_samples = len(levels)
        num_classes = len(level_counts)
        
        for level, count in level_counts.items():
            # Inverse frequency weighting
            class_weights[level] = total_samples / (num_classes * count)
        
        print(f"Using class weights: {class_weights}")
        
        # Create balanced sample for improved training
        balanced_keys = []
        balanced_levels = []
        
        # Find the minimum count for balancing (use at least 20000 samples per class)
        target_count = max(20000, min(level_counts.values()))
        
        # Create undersampled training data for majority classes, 
        # and oversampled data for minority classes
        for level in level_counts:
            # Get indices for this level
            level_indices = [i for i, l in enumerate(levels) if l == level]
            
            if len(level_indices) > target_count:
                # Undersample for majority classes
                sampled_indices = random.sample(level_indices, target_count)
            else:
                # Oversample for minority classes
                sampled_indices = random.choices(level_indices, k=target_count)
                
            # Add sampled data
            for idx in sampled_indices:
                balanced_keys.append(keys[idx])
                balanced_levels.append(levels[idx])
                
        print(f"Rebalanced dataset: {len(balanced_keys)} samples")
        
        # Train the model with balanced data
        self.bloom_model.train(balanced_keys, balanced_levels)
        
        # Get accuracy from the model
        self.bloom_accuracy = self.bloom_model.accuracy
        
        end_time = time.perf_counter()
        print(f"Level predictor trained in {(end_time - start_time)*1000:.2f}ms")
        print(f"Level prediction accuracy: {self.bloom_accuracy:.2%}")
        
        # Clear cache
        self.bloom_cache = {}
        
        self._save_models()
        
    def train_fence_model(self):
        """Train fence pointer model."""
        if not self.fence_data:
            return
            
        start_time = time.perf_counter()
        
        # Extract X (key, level) and y (page)
        X = []
        y = []
        
        for key, level, page in self.fence_data:
            X.append([float(key), int(level)])  # Input features: key and level
            y.append(int(page))                # Target: page number
        
        # Train the model with the correct function signature
        self.fence_model.train(X, y)
        
        # Get accuracy
        self.fence_accuracy = self.fence_model.accuracy
        
        end_time = time.perf_counter()
        training_time = (end_time - start_time) * 1000
        
        print(f"Fence pointer model training completed in {training_time/1000:.2f} seconds")
        
        # Clear cache
        self.fence_cache = {}
        
        self._save_models()
        
    def predict_bloom(self, key: float) -> int:
        """Predict which level most likely contains the key.
        
        Returns:
        --------
        int: The predicted level number, or -1 if the key likely doesn't exist
        """
        # Check cache first
        if key in self.bloom_cache:
            return self.bloom_cache[key]
        
        # Time the prediction
        start_time = time.perf_counter()
        
        # Get prediction from the model
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

    def save_models(self):
        """Save trained models to disk."""
        import pickle
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            
        # Save bloom filter model
        bloom_path = os.path.join(self.models_dir, "bloom_model.pkl")
        with open(bloom_path, 'wb') as f:
            pickle.dump(self.bloom_model, f)
            
        # Save fence pointer model
        fence_path = os.path.join(self.models_dir, "fence_model.pkl") 
        with open(fence_path, 'wb') as f:
            pickle.dump(self.fence_model, f)
            
        # Save training data for potential retraining
        data_path = os.path.join(self.models_dir, "training_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'bloom_data': self.bloom_data,
                'fence_data': self.fence_data,
                'bloom_accuracy': self.bloom_accuracy,
                'fence_accuracy': self.fence_accuracy
            }, f)
            
        print(f"Models and training data saved to {self.models_dir}")
        
    def load_models(self):
        """Load trained models from disk."""
        import pickle
        import os
        
        # Load bloom filter model
        bloom_path = os.path.join(self.models_dir, "bloom_model.pkl")
        if os.path.exists(bloom_path):
            with open(bloom_path, 'rb') as f:
                self.bloom_model = pickle.load(f)
                
        # Load fence pointer model
        fence_path = os.path.join(self.models_dir, "fence_model.pkl")
        if os.path.exists(fence_path):
            with open(fence_path, 'rb') as f:
                self.fence_model = pickle.load(f)
                
        # Load training statistics if available
        data_path = os.path.join(self.models_dir, "training_data.pkl")
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.bloom_accuracy = data.get('bloom_accuracy', 0)
                self.fence_accuracy = data.get('fence_accuracy', 0)
                
        print(f"Models loaded from {self.models_dir}") 