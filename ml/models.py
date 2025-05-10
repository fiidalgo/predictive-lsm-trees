import numpy as np
import time
import random
import joblib
import os
from typing import List, Tuple, Optional, Dict, Set
import pickle
from collections import Counter
import inspect
import logging
import math

# Set LightGBM logger to ERROR level to suppress warnings
logging.getLogger('lightgbm').setLevel(logging.ERROR)

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
    
    This model predicts if a key exists (binary classification) rather than
    trying to predict which level contains the key. This creates a simpler
    decision boundary and improves accuracy.
    """
    
    def __init__(self):
        # Classification model that predicts if key exists (binary)
        self.model = None
        self.models = []  # Ensemble of models
        self.use_ensemble = True  # Enable ensemble prediction by default
        self.trained = False
        self.accuracy = 0.0
        self.recall = 0.0  # Focus on recall - we want to minimize false negatives
        self.stats = {'correct': 0, 'total': 0, 'true_positives': 0, 'false_negatives': 0}
        self.min_max_keys = {}  # Store min/max keys for feature engineering
        self.quantiles = {}     # Store quantiles of key distribution for feature engineering
        self.decision_threshold = 0.5  # Will be optimized during training
        
        # For hash-based features
        try:
            import mmh3
            self.hash_function = mmh3.hash
            self.has_mmh3 = True
        except ImportError:
            import hashlib
            self.hash_function = lambda x: int.from_bytes(hashlib.md5(str(x).encode('utf-8')).digest()[:4], 'little')
            self.has_mmh3 = False
        
    def train(self, X, y_exists):
        """Train the bloom filter model to predict existence (binary).
        
        X: List of keys as features
        y_exists: List of booleans (True if key exists, False if not)
        """
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        import time
        
        # Try to import tqdm for progress bars
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            print("Note: Install tqdm package for progress bars (pip install tqdm)")
            
        # Try to import optional models - XGBoost and LightGBM support callbacks
        has_xgboost = False
        has_lightgbm = False
        try:
            import xgboost as xgb
            has_xgboost = True
        except ImportError:
            pass
            
        try:
            import lightgbm as lgb
            has_lightgbm = True
        except ImportError:
            pass
        
        if len(X) == 0 or len(y_exists) == 0:
            print("Error: No training data provided")
            return False
            
        # Limit data size for faster training
        MAX_SAMPLES = 100000
        if len(X) > MAX_SAMPLES:
            print(f"WARNING: Dataset is large ({len(X)} samples). Limiting to {MAX_SAMPLES} random samples for faster training.")
            # Take stratified sample for balanced classes
            pos_indices = [i for i, y in enumerate(y_exists) if y]
            neg_indices = [i for i, y in enumerate(y_exists) if not y]
            
            # Calculate how many of each class to take
            pos_sample_size = min(len(pos_indices), MAX_SAMPLES // 2)
            neg_sample_size = min(len(neg_indices), MAX_SAMPLES // 2)
            
            # Sample indices
            sampled_pos_indices = random.sample(pos_indices, pos_sample_size)
            sampled_neg_indices = random.sample(neg_indices, neg_sample_size)
            sampled_indices = sampled_pos_indices + sampled_neg_indices
            
            # Create new dataset
            X = [X[i] for i in sampled_indices]
            y_exists = [y_exists[i] for i in sampled_indices]
            print(f"Reduced to {len(X)} training samples")
        
        print("Starting training process...")
        
        # Compute min/max keys for feature engineering
        self.min_max_keys = {'global': (min(X), max(X))}
        
        # Compute quantiles for positive samples (existing keys)
        positive_keys = [X[i] for i, exists in enumerate(y_exists) if exists]
        if positive_keys:
            self.quantiles = {
                'p10': np.percentile(positive_keys, 10),
                'p25': np.percentile(positive_keys, 25),
                'p50': np.percentile(positive_keys, 50),
                'p75': np.percentile(positive_keys, 75),
                'p90': np.percentile(positive_keys, 90)
            }
        
        # Create engineered features
        print(f"Creating enriched features for {len(X)} samples...")
        X_features = []
        
        # Batch process features to show progress but still be efficient
        batch_size = 5000
        num_batches = (len(X) + batch_size - 1) // batch_size
        
        if has_tqdm:
            pbar = tqdm(total=len(X), desc="Creating features")
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(len(X), (i + 1) * batch_size)
                batch = X[start_idx:end_idx]
                
                # Process the batch
                batch_features = self._create_features(batch)
                X_features.extend(batch_features)
                
                # Update progress
                pbar.update(len(batch))
            pbar.close()
        else:
            # Process without progress bar
            X_features = self._create_features(X)
        
        # Format input data for model - ensure y is binary (0 or 1)
        X_formatted = np.array(X_features)
        # Convert to explicitly boolean then to int to ensure only 0,1 values
        y_formatted = np.array([bool(y) for y in y_exists], dtype=int)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_formatted, y_formatted, test_size=0.2, random_state=42, stratify=y_formatted
        )
        
        print(f"Training binary classifier with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        # Count positive vs negative examples - ensure we're only using non-negative values
        # Explicitly check and fix y_train values
        y_train_fixed = np.clip(y_train, 0, 1)  # Clip to 0-1 range
        class_counts = np.bincount(y_train_fixed, minlength=2)  # Ensure minlength=2 for binary classification
        print(f"Class distribution in training data: {class_counts[0]} negatives, {class_counts[1]} positives")
        
        start_time = time.time()
        
        # Create custom sample weights for GradientBoostingClassifier since it doesn't support class_weight
        sample_weights = np.ones(len(y_train_fixed))
        if class_counts[1] > 0:  # If we have positive samples
            # Assign higher weight to positive samples (similar to class_weight)
            weight_ratio = class_counts[0] / class_counts[1]  # Ratio of negatives to positives
            # Make positives 5x more important (aggressive weighting)
            for i in range(len(y_train_fixed)):
                if y_train_fixed[i] == 1:  # Positive sample
                    sample_weights[i] = 5.0 * weight_ratio
        
        if self.use_ensemble:
            # Create an ensemble of diverse models for better robustness
            self.models = []
            
            # Create appropriate models based on available libraries
            if has_xgboost:
                print("Using XGBoost for enhanced performance")
                # XGBoost with callback support
                gb_model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                gb_model2 = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.7,
                    random_state=44,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            elif has_lightgbm:
                print("Using LightGBM for enhanced performance")
                # LightGBM with callback support
                gb_model = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=-1,  # Silence warnings
                    verbose=-1     # Turn off LightGBM stdout
                )
                gb_model2 = lgb.LGBMClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.7,
                    random_state=44,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=-1,  # Silence warnings
                    verbose=-1     # Turn off LightGBM stdout
                )
            else:
                # Standard scikit-learn models
                gb_model = GradientBoostingClassifier(
                    n_estimators=100,  # Reduced from 300
                    max_depth=5,       # Reduced from 7
                    learning_rate=0.1,
                    subsample=0.7,     # Increased subsampling to make training faster
                    random_state=42
                    # GradientBoostingClassifier doesn't support class_weight
                )
                gb_model2 = GradientBoostingClassifier(
                    n_estimators=100,  # Reduced from 500
                    max_depth=4,       # Reduced from 5
                    learning_rate=0.1, # Increased from 0.05 to make training faster
                    subsample=0.6,     # Reduced from 0.7
                    random_state=44
                    # GradientBoostingClassifier doesn't support class_weight
                )
            
            # Add RandomForest for diversity
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=43,
                class_weight='balanced'  # Use 'balanced' instead of undefined class_weight variable
            )
            
            # Add models to ensemble
            self.models = [gb_model, rf_model, gb_model2]
            
            # Train each model
            for i, model in enumerate(self.models):
                print(f"Training ensemble model {i+1}/{len(self.models)}...")
                if has_tqdm:
                    with tqdm(total=100, desc=f"Model {i+1} training") as pbar:
                        # Create a callback to update the progress bar
                        class ProgressCallback:
                            def __init__(self, pbar):
                                self.pbar = pbar
                                self.iter_count = 0
                                self.total_iters = 100  # Estimate
                                
                            def __call__(self, est):
                                self.iter_count += 1
                                progress = min(100, int(100 * self.iter_count / self.total_iters))
                                self.pbar.update(progress - self.pbar.n)
                        
                        # Train the model - GradientBoostingClassifier doesn't support callbacks
                        if i == 1:  # RandomForest - use class_weight
                            model.fit(X_train, y_train_fixed)
                        else:  # GradientBoosting - use sample_weight but no callback
                            # Check if the model has monitor parameter (like XGBoost would)
                            if hasattr(model, 'fit') and 'callback' in model.fit.__code__.co_varnames:
                                # XGBoost models likely use callback instead of monitor
                                model.fit(X_train, y_train_fixed, sample_weight=sample_weights, callback=ProgressCallback(pbar))
                            elif hasattr(model, 'fit') and 'callbacks' in model.fit.__code__.co_varnames:
                                # LightGBM models use callbacks (plural)
                                model.fit(X_train, y_train_fixed, sample_weight=sample_weights, callbacks=[ProgressCallback(pbar)])
                            elif hasattr(model, 'fit') and 'monitor' in model.fit.__code__.co_varnames:
                                # Models that support monitor parameter
                                model.fit(X_train, y_train_fixed, sample_weight=sample_weights, monitor=ProgressCallback(pbar))
                            else:
                                # Standard scikit-learn doesn't support callbacks
                                model.fit(X_train, y_train_fixed, sample_weight=sample_weights)
                            
                        # Update progress manually since callback might not work
                        pbar.update(100 - pbar.n)  # Ensure we reach 100%
                else:
                    if i == 1:  # RandomForest - use class_weight
                        model.fit(X_train, y_train_fixed)
                    else:  # GradientBoosting - use sample_weight
                        model.fit(X_train, y_train_fixed, sample_weight=sample_weights)
            
            # Create a primary model for fallback and feature importance
            self.model = self.models[0]
            
            print(f"Ensemble of {len(self.models)} models trained")
        else:
            # Hyperparameter optimization using RandomizedSearchCV
            param_dist = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
            
            # Create a more powerful classifier optimized for recall
            base_model = GradientBoostingClassifier(random_state=42)
            
            # Use RandomizedSearchCV to find best parameters
            # Focus on recall as scoring metric to minimize false negatives
            print("Performing hyperparameter search...")
            try:
                search = RandomizedSearchCV(
                    base_model, param_distributions=param_dist,
                    n_iter=10, cv=3, scoring='recall', n_jobs=-1,
                    random_state=42
                )
                
                # Fit the search on training data
                search.fit(X_train, y_train_fixed)
                self.model = search.best_estimator_
                print(f"Best parameters: {search.best_params_}")
                print(f"Best cross-validation recall: {search.best_score_:.4f}")
            except Exception as e:
                print(f"Error in hyperparameter search: {e}")
                # Fall back to default model
                self.model = GradientBoostingClassifier(
                    n_estimators=100,  # Reduced from 300
                    max_depth=5,       # Reduced from 7
                    learning_rate=0.1,
                    subsample=0.7,     # Increased subsampling to make training faster
                    random_state=42
                    # GradientBoostingClassifier doesn't support class_weight
                )
                self.model.fit(X_train, y_train_fixed, sample_weight=sample_weights)
            
        # Hard negative mining for a second training round
        if len(X_val) > 0:
            print("Performing hard negative mining...")
            # Get predictions with probabilities
            if self.use_ensemble:
                y_proba = self._ensemble_predict_proba(X_val)
            else:
                y_proba = self.model.predict_proba(X_val)[:,1]
            
            # Ensure y_val is also fixed
            y_val_fixed = np.clip(y_val, 0, 1)
            
            # Find hard negatives (false positives with high scores)
            hard_negative_indices = np.where((y_val_fixed == 0) & (y_proba > 0.3))[0]
            
            if len(hard_negative_indices) > 0:
                print(f"Found {len(hard_negative_indices)} hard negatives")
                
                # Add hard negatives to training data
                hard_negative_X = X_val[hard_negative_indices]
                hard_negative_y = y_val_fixed[hard_negative_indices]
                
                # Combine with original training data
                X_train_enhanced = np.vstack((X_train, hard_negative_X))
                y_train_enhanced = np.concatenate((y_train_fixed, hard_negative_y))
                
                # Retrain models with hard negatives
                if self.use_ensemble:
                    for i, model in enumerate(self.models):
                        print(f"Retraining ensemble model {i+1}/{len(self.models)} with hard negatives...")
                        
                        # Create weights for enhanced dataset
                        enhanced_weights = np.ones(len(y_train_enhanced))
                        for j in range(len(y_train_enhanced)):
                            if y_train_enhanced[j] == 1:  # Positive sample
                                enhanced_weights[j] = 5.0  # Higher weight for positives
                        
                        if has_tqdm:
                            with tqdm(total=100, desc=f"Retraining model {i+1}") as pbar:
                                if i == 1:  # RandomForest - use class_weight
                                    model.fit(X_train_enhanced, y_train_enhanced)
                                else:  # GradientBoosting - use sample_weight
                                    model.fit(X_train_enhanced, y_train_enhanced, sample_weight=enhanced_weights)
                                pbar.update(100)  # Complete the progress bar
                        else:
                            if i == 1:  # RandomForest - use class_weight
                                model.fit(X_train_enhanced, y_train_enhanced)
                            else:  # GradientBoosting - use sample_weight
                                model.fit(X_train_enhanced, y_train_enhanced, sample_weight=enhanced_weights)
                else:
                    # Create weights for enhanced dataset for non-ensemble case
                    enhanced_weights = np.ones(len(y_train_enhanced))
                    for j in range(len(y_train_enhanced)):
                        if y_train_enhanced[j] == 1:  # Positive sample
                            enhanced_weights[j] = 5.0  # Higher weight for positives
                    
                    # Retrain model with hard negatives
                    if has_tqdm:
                        with tqdm(total=100, desc="Retraining with hard negatives") as pbar:
                            self.model.fit(X_train_enhanced, y_train_enhanced, sample_weight=enhanced_weights)
                            pbar.update(100)  # Complete the progress bar
                    else:
                        self.model.fit(X_train_enhanced, y_train_enhanced, sample_weight=enhanced_weights)
        
        # Evaluate on validation set
        self.trained = True
        
        # Get validation predictions with probabilities
        if self.use_ensemble:
            y_proba = self._ensemble_predict_proba(X_val)
        else:
            y_proba = self.model.predict_proba(X_val)[:,1]
        
        # Find optimal threshold for 98% recall (more conservative threshold)
        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0.5
        best_recall = 0.0
        target_recall = 0.98  # Increased from 0.95 to 0.98
        
        # Ensure y_val is fixed
        y_val_fixed = np.clip(y_val, 0, 1)
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate recall
            true_positives = np.sum((y_val_fixed == 1) & (y_pred == 1))
            actual_positives = np.sum(y_val_fixed == 1)
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            
            # Find threshold that gives at least target recall if possible
            if recall >= target_recall and threshold > best_threshold:
                best_threshold = threshold
                best_recall = recall
                
        # If we couldn't achieve target recall, use the threshold with the highest recall
        if best_recall < target_recall:
            best_threshold = thresholds[np.argmax([
                np.sum((y_val_fixed == 1) & (y_proba >= t)) / np.sum(y_val_fixed == 1) 
                if np.sum(y_val_fixed == 1) > 0 else 0 
                for t in thresholds
            ])]
            best_recall = np.sum((y_val_fixed == 1) & (y_proba >= best_threshold)) / np.sum(y_val_fixed == 1) if np.sum(y_val_fixed == 1) > 0 else 0
        
        self.decision_threshold = best_threshold
        print(f"Optimal decision threshold: {best_threshold:.4f} for recall: {best_recall:.4f}")
        
        # Final evaluation with optimal threshold
        y_pred = (y_proba >= self.decision_threshold).astype(int)
        
        # Calculate accuracy metrics
        correct = np.sum(y_pred == y_val_fixed)
        total = len(y_val_fixed)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate recall (critical for Bloom filter)
        true_positives = np.sum((y_val_fixed == 1) & (y_pred == 1))
        actual_positives = np.sum(y_val_fixed == 1)
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        # Calculate false negatives (missed keys - most critical metric)
        false_negatives = np.sum((y_val_fixed == 1) & (y_pred == 0))
        false_negative_rate = false_negatives / actual_positives if actual_positives > 0 else 0
        
        # Store stats
        self.stats = {
            'correct': int(correct), 
            'total': int(total),
            'true_positives': int(true_positives),
            'false_negatives': int(false_negatives)
        }
        self.accuracy = accuracy
        self.recall = recall
        
        end_time = time.time()
        training_time_ms = (end_time - start_time) * 1000
        
        print(f"Bloom filter trained in {training_time_ms:.2f}ms")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"False Negative Rate: {false_negative_rate:.2%}")
        
        # Print feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            sorted_indices = np.argsort(importances)[::-1]
            print("Feature importances:")
            for i in sorted_indices[:10]:  # Top 10 features
                print(f"  {feature_names[i]}: {importances[i]:.4f}")
                
        return True
    
    def _create_features(self, keys):
        """Create rich features for the model from raw keys.
        
        Better features help the model make more accurate predictions.
        """
        import numpy as np
        features = []
        
        # Ensure has_mmh3 attribute exists (in case it wasn't properly pickled)
        if not hasattr(self, 'has_mmh3'):
            try:
                import mmh3
                self.has_mmh3 = True
                self.hash_function = mmh3.hash
            except ImportError:
                self.has_mmh3 = False
                import hashlib
                self.hash_function = lambda x: int.from_bytes(hashlib.md5(str(x).encode('utf-8')).digest()[:4], 'little')
        
        # Get global min/max if available
        global_min = self.min_max_keys.get('global', (float('-inf'), float('inf')))[0]
        global_max = self.min_max_keys.get('global', (float('-inf'), float('inf')))[1]
        global_range = global_max - global_min
        
        # Get quantiles if available
        p10 = self.quantiles.get('p10', global_min)
        p25 = self.quantiles.get('p25', global_min)
        p50 = self.quantiles.get('p50', (global_min + global_max) / 2)
        p75 = self.quantiles.get('p75', global_max)
        p90 = self.quantiles.get('p90', global_max)
        
        for key in keys:
            feature_vector = self._create_features_for_key(key, global_min, global_max, global_range, p10, p25, p50, p75, p90)
            features.append(feature_vector)
            
        return features
    
    def _create_features_for_key(self, key, global_min=None, global_max=None, global_range=None, p10=None, p25=None, p50=None, p75=None, p90=None):
        """Create features for a single key.
        
        This method is extracted from _create_features to allow for progress bar tracking.
        """
        import numpy as np
        import random
        
        # Check if we need to get global values (if not provided)
        if global_min is None or global_max is None:
            global_min = self.min_max_keys.get('global', (float('-inf'), float('inf')))[0]
            global_max = self.min_max_keys.get('global', (float('-inf'), float('inf')))[1]
            global_range = global_max - global_min
            
            # Get quantiles if available
            p10 = self.quantiles.get('p10', global_min)
            p25 = self.quantiles.get('p25', global_min)
            p50 = self.quantiles.get('p50', (global_min + global_max) / 2)
            p75 = self.quantiles.get('p75', global_max)
            p90 = self.quantiles.get('p90', global_max)
        
        key_float = float(key)
        feature_vector = []
        
        # Basic feature - raw key value
        feature_vector.append(key_float)
        
        # Normalized position in global range (0 to 1)
        if global_range > 0:
            norm_pos = (key_float - global_min) / global_range
            feature_vector.append(norm_pos)
        else:
            feature_vector.append(0.5)  # Default to middle
            
        # Log-scaled features - helps with large ranges
        if key_float > 0:
            feature_vector.append(np.log(key_float))
        else:
            feature_vector.append(0)
            
        if global_max > 0:
            feature_vector.append(np.log1p(global_max - key_float) if key_float < global_max else 0)
        else:
            feature_vector.append(0)
            
        # Distance to quantiles - helps cluster similar keys
        feature_vector.append(abs(key_float - p10) if p10 is not None else 0)
        feature_vector.append(abs(key_float - p25) if p25 is not None else 0)
        feature_vector.append(abs(key_float - p50) if p50 is not None else 0) 
        feature_vector.append(abs(key_float - p75) if p75 is not None else 0)
        feature_vector.append(abs(key_float - p90) if p90 is not None else 0)
        
        # Relative position to quantiles (categorical representation)
        feature_vector.append(1 if key_float < p10 else 0)
        feature_vector.append(1 if p10 <= key_float < p25 else 0)
        feature_vector.append(1 if p25 <= key_float < p50 else 0)
        feature_vector.append(1 if p50 <= key_float < p75 else 0)
        feature_vector.append(1 if p75 <= key_float < p90 else 0)
        feature_vector.append(1 if key_float >= p90 else 0)
        
        # Hash-based features
        try:
            # Convert to string and compute various hashes
            key_str = str(key_float).encode('utf-8')
            
            # Use multiple hash seeds for better distribution
            for seed in range(6):
                try:
                    if self.has_mmh3:
                        import mmh3
                        hash_val = mmh3.hash(key_str, seed) % 1000
                        hash_val = hash_val / 1000.0  # Normalize to 0-1
                        feature_vector.append(hash_val)
                    else:
                        import hashlib
                        h = hashlib.md5(f"{key_str}_{seed}".encode('utf-8')).digest()
                        hash_val = int.from_bytes(h[:4], 'little') % 1000
                        hash_val = hash_val / 1000.0  # Normalize to 0-1
                        feature_vector.append(hash_val)
                except Exception as e:
                    # Fallback on individual hash error
                    feature_vector.append(hash(f"{key_float}_{seed}") % 1000 / 1000.0)
                    
            # Add binary features (individual bits from hash)
            for i in range(6):
                try:
                    if self.has_mmh3:
                        import mmh3
                        hash_val = mmh3.hash(key_str, i)
                        # Extract 4 bits from different positions
                        for shift in [0, 8, 16, 24]:
                            bit = (hash_val >> shift) & 1
                            feature_vector.append(bit)
                    else:
                        import hashlib
                        h = hashlib.md5(f"{key_str}_{i}".encode('utf-8')).digest()
                        val = int.from_bytes(h[:4], 'little')
                        for shift in [0, 8, 16, 24]:
                            bit = (val >> shift) & 1
                            feature_vector.append(bit)
                except Exception as e:
                    # Fallback on individual hash error
                    h = hash(f"{key_float}_{i}")
                    for _ in range(4):  # Add 4 random bits as fallback
                        feature_vector.append(random.randint(0, 1))
                        
        except Exception as e:
            # Add zeros as fallback if hash features fail
            print(f"Warning: Hash features generation failed: {e}")
            for i in range(48):  # Add placeholders for all hash features
                feature_vector.append(0.0)
                
        return feature_vector
    
    def _ensemble_predict_proba(self, X):
        """Combine predictions from all models in the ensemble.
        
        For Bloom filters, we want to be conservative and predict a key exists
        if ANY model thinks it might exist (maximum probability).
        """
        import numpy as np
        
        # Handle edge cases
        if not hasattr(self, 'models') or not self.models:
            if hasattr(self, 'model') and self.model:
                try:
                    return self.model.predict_proba(X)[:,1]
                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    # Return conservative predictions (all 1.0)
                    return np.ones(len(X), dtype=float)
            else:
                # No models at all, return conservative predictions
                return np.ones(len(X), dtype=float)
            
        # Get predictions from each model
        probas = []
        for model in self.models:
            try:
                proba = model.predict_proba(X)[:,1]
                probas.append(proba)
            except Exception as e:
                print(f"Error in model prediction: {e}")
                
        if not probas:
            # No successful predictions, return conservative predictions
            return np.ones(len(X), dtype=float)
            
        # Take maximum probability across all models
        # This is conservative - if any model thinks key exists, we predict it exists
        return np.max(np.vstack(probas), axis=0)
    
    def predict(self, key):
        """Predict if the key exists (binary prediction).
        
        Returns:
        --------
        bool: True if key likely exists, False otherwise
        """
        # Handle edge cases with missing attributes
        if not hasattr(self, 'trained') or not self.trained:
            # Default conservative behavior - assume exists
            return True
            
        if not hasattr(self, 'model') or not self.model:
            if not hasattr(self, 'models') or not self.models:
                # No models trained
                return True
            
        # Ensure has_mmh3 is set
        if not hasattr(self, 'has_mmh3'):
            try:
                import mmh3
                self.has_mmh3 = True
                self.hash_function = mmh3.hash
            except ImportError:
                self.has_mmh3 = False
                import hashlib
                self.hash_function = lambda x: int.from_bytes(hashlib.md5(str(x).encode('utf-8')).digest()[:4], 'little')
        
        # Create features for prediction
        try:
            X = self._create_features([key])
            
            # Use ensemble prediction if available
            if hasattr(self, 'use_ensemble') and self.use_ensemble and hasattr(self, 'models') and self.models:
                probability = self._ensemble_predict_proba(X)[0]
            else:
                # Get probability of existence
                probability = self.model.predict_proba(X)[0][1]
                
            # Use optimized threshold
            if hasattr(self, 'decision_threshold'):
                threshold = self.decision_threshold
            else:
                threshold = 0.5  # Default threshold
                
            return probability >= threshold
        except Exception as e:
            # On error, default to conservative behavior
            print(f"Error in existence prediction: {e}")
            return True  # Assume key exists
            
    def predict_batch(self, keys):
        """Batch predict if keys exist.
        
        Parameters:
        -----------
        keys : List[float] or numpy.ndarray
            List of keys to predict
            
        Returns:
        --------
        numpy.ndarray
            Array of probabilities (0.0 to 1.0) that each key exists
        """
        import numpy as np
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys)
        
        # Create results array
        results = np.ones(len(keys), dtype=float)
        
        # Early return if no model
        if not self.bloom_models[0].trained:
            return results
            
        # Time the batch prediction
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_hits = 0
            for i, key in enumerate(keys):
                if key in self.bloom_cache:
                    results[i] = self.bloom_cache[key]
                    cache_hits += 1
                    
            # Only predict for keys not in cache
            if cache_hits < len(keys):
                # Get indices of keys not in cache
                predict_indices = [i for i, key in enumerate(keys) if key not in self.bloom_cache]
                predict_keys = keys[predict_indices]
                
                # Create features for these keys
                X = self.bloom_models[0]._create_features(predict_keys)
                
                # Get raw probabilities
                if self.bloom_models[0].use_ensemble and self.bloom_models[0].models:
                    probabilities = self.bloom_models[0]._ensemble_predict_proba(X)
                else:
                    # Fall back to single model
                    probabilities = self.bloom_models[0].model.predict_proba(X)[:, 1]
                
                # Update results and cache
                for i, idx in enumerate(predict_indices):
                    results[idx] = probabilities[i]
                    if len(self.bloom_cache) < self.max_cache_size:
                        self.bloom_cache[keys[idx]] = probabilities[i]
                
            # Record timing
            end_time = time.perf_counter()
            self.bloom_prediction_time += (end_time - start_time)
            self.bloom_prediction_count += len(keys)
            
            return results
            
        except Exception as e:
            # On error, return all 1.0 (assume all keys exist)
            print(f"Error in batch bloom prediction: {e}")
            end_time = time.perf_counter()
            self.bloom_prediction_time += (end_time - start_time)
            self.bloom_prediction_count += len(keys)
            return np.ones(len(keys), dtype=float)
    
    def predict_fence(self, key: float, level: int) -> int:
        """Predict page number for a key using the fence pointer model for the specified level.
        
        Parameters:
        -----------
        key : float
            The key to predict page for
        level : int
            Level number (0-6)
            
        Returns:
        --------
        int: Predicted page number
        """
        # Ensure level is valid
        if level < 0 or level >= self.MAX_LEVEL:
            # Default to level 0 if invalid
            level = 0
            
        # Ultra-fast path: Check cache first
        cache_key = (key, level)
        if cache_key in self.fence_cache:
            return self.fence_cache[cache_key]
        
        # Time the prediction
        start_time = time.perf_counter()
        
        # Use the specific fence model for this level
        result = self.fence_models[level].predict(key, level)
        
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
        
        # Ensure level is valid
        if level < 0 or level >= self.MAX_LEVEL:
            # Default to level 0 if invalid
            level = 0
            
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys)
        
        # Time the batch prediction
        start_time = time.perf_counter()
        
        # Create results array
        results = np.empty(len(keys), dtype=np.int32)
        
        # Simply predict each key individually using the level-specific model
        for i, key in enumerate(keys):
            results[i] = self.fence_models[level].predict(key, level)
            
            # Update cache for future single-key lookups
            cache_key = (key, level)
            if len(self.fence_cache) < self.max_cache_size:
                self.fence_cache[cache_key] = results[i]
        
        # Record timing
        end_time = time.perf_counter()
        prediction_time = (end_time - start_time)
        self.fence_prediction_time += prediction_time
        self.fence_prediction_count += len(keys)
        
        return results
        
    def get_bloom_accuracy(self) -> float:
        """Get average Bloom filter model accuracy across all levels."""
        # Return average accuracy across all trained levels
        trained_levels = [lvl for lvl, acc in self.bloom_accuracy.items() if acc > 0]
        if trained_levels:
            return sum(self.bloom_accuracy[lvl] for lvl in trained_levels) / len(trained_levels)
        return 0.0
        
    def get_fence_accuracy(self) -> float:
        """Get average fence pointer model accuracy across all levels."""
        # Return average accuracy across all trained levels
        trained_levels = [lvl for lvl, acc in self.fence_accuracy.items() if acc > 0]
        if trained_levels:
            return sum(self.fence_accuracy[lvl] for lvl in trained_levels) / len(trained_levels)
        return 0.0
        
    def get_bloom_accuracy_by_level(self) -> dict:
        """Get Bloom filter accuracy for each level."""
        return self.bloom_accuracy
        
    def get_fence_accuracy_by_level(self) -> dict:
        """Get fence pointer accuracy for each level."""
        return self.fence_accuracy
        
    def get_prediction_stats(self) -> dict:
        """Get prediction timing statistics."""
        # Collect cache hit statistics across all fence models
        fence_cache_hits = 0
        fence_cache_misses = 0
        for lvl, model in self.fence_models.items():
            fence_cache_hits += getattr(model, 'cache_hits', 0)
            fence_cache_misses += getattr(model, 'cache_misses', 0)
            
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
            'fence_direct_cache_hits': fence_cache_hits,
            'fence_direct_cache_misses': fence_cache_misses,
            'bloom_accuracy_by_level': self.bloom_accuracy,
            'fence_accuracy_by_level': self.fence_accuracy
        }
        return stats

    def save_models(self):
        """Save trained models to disk."""
        import pickle
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            
        # Save all bloom filter models
        for lvl, model in self.bloom_models.items():
            bloom_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
            try:
                with open(bloom_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Error saving bloom model for level {lvl}: {e}")
                
        # Save all fence pointer models
        for lvl, model in self.fence_models.items():
            fence_path = os.path.join(self.models_dir, f"fence_model_lvl_{lvl}.pkl")
            try:
                with open(fence_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Error saving fence model for level {lvl}: {e}")
            
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
        
        # Load all bloom filter models
        for lvl in range(self.MAX_LEVEL):
            bloom_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
            
            if os.path.exists(bloom_path):
                try:
                    with open(bloom_path, 'rb') as f:
                        self.bloom_models[lvl] = pickle.load(f)
                    # Ensure ensemble prediction is enabled
                    self.bloom_models[lvl].use_ensemble = self.ensemble_prediction
                    
                    # Ensure hash attribute is set correctly
                    if not hasattr(self.bloom_models[lvl], 'has_mmh3'):
                        try:
                            import mmh3
                            self.bloom_models[lvl].has_mmh3 = True
                            self.bloom_models[lvl].hash_function = mmh3.hash
                        except ImportError:
                            self.bloom_models[lvl].has_mmh3 = False
                            import hashlib
                            self.bloom_models[lvl].hash_function = lambda x: int.from_bytes(hashlib.md5(str(x).encode('utf-8')).digest()[:4], 'little')
                except Exception as e:
                    print(f"Error loading bloom model for level {lvl}: {e}")
        
        # Load all fence pointer models
        for lvl in range(self.MAX_LEVEL):
            fence_path = os.path.join(self.models_dir, f"fence_model_lvl_{lvl}.pkl")
            
            if os.path.exists(fence_path):
                try:
                    with open(fence_path, 'rb') as f:
                        self.fence_models[lvl] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading fence model for level {lvl}: {e}")
                
        # Load training statistics if available
        data_path = os.path.join(self.models_dir, "training_data.pkl")
        if os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data.get('bloom_accuracy'), dict):
                        self.bloom_accuracy = data.get('bloom_accuracy', self.bloom_accuracy)
                    if isinstance(data.get('fence_accuracy'), dict):
                        self.fence_accuracy = data.get('fence_accuracy', self.fence_accuracy)
                    
                    # Load training data if available
                    if isinstance(data.get('bloom_data'), dict):
                        self.bloom_data = data.get('bloom_data', self.bloom_data)
                    if isinstance(data.get('fence_data'), dict):
                        self.fence_data = data.get('fence_data', self.fence_data)
            except Exception as e:
                print(f"Error loading training data: {e}")
                
        print(f"Models loaded from {self.models_dir}")
        
    def predict_bloom_level(self, key):
        """Predict which level most likely contains the key.
        
        Parameters:
        -----------
        key : float
            The key to check
        
        Returns:
        --------
        tuple: (best_level, {lvl: prob})
            best_level: level number with highest prediction score
            Dictionary mapping each level to its probability score
        """
        # Check cache first
        if key in self.bloom_cache:
            return self.bloom_cache[key]
            
        # Time the prediction
        start_time = time.perf_counter()
        
        # Get predictions for each level
        probs = {}
        try:
            for lvl, model in self.bloom_models.items():
                # Skip levels with no trained model
                if not hasattr(model, 'trained') or not model.trained:
                    probs[lvl] = 0.1  # Low probability for untrained levels
                    continue
                    
                # Create features for this level
                X = model._create_features([key])
                
                # Get probability from model
                if model.use_ensemble and hasattr(model, 'models') and model.models:
                    probability = model._ensemble_predict_proba(X)[0]
                else:
                    # Fall back to single model if ensemble not available
                    if hasattr(model, 'model') and model.model is not None:
                        probability = model.model.predict_proba(X)[0][1]
                    else:
                        probability = 0.1  # Default if no model available
                
                probs[lvl] = probability
                
            # Find level with highest probability
            if probs:
                best_level = max(probs, key=probs.get)
            else:
                best_level = 0  # Default to level 0 if no predictions
                
            # Store in cache
            if len(self.bloom_cache) < self.max_cache_size:
                self.bloom_cache[key] = (best_level, probs)
                
            # Record timing
            end_time = time.perf_counter()
            self.bloom_prediction_time += (end_time - start_time)
            self.bloom_prediction_count += 1
            
            return best_level, probs
            
        except Exception as e:
            # On error, return level 0 with high probability for safety
            print(f"Error in bloom level prediction: {e}")
            end_time = time.perf_counter()
            self.bloom_prediction_time += (end_time - start_time)
            self.bloom_prediction_count += 1
            return 0, {0: 1.0}
            
    def predict_bloom(self, key: float) -> float:
        """Predict if a key exists.
        
        Returns:
        --------
        float: Probability that the key exists (0.0 to 1.0)
        """
        # Use predict_bloom_level to get the best level and its probability
        best_level, probs = self.predict_bloom_level(key)
        
        # Return the highest probability across all levels
        if probs:
            return probs[best_level]
        else:
            # Default to 1.0 if no predictions (assume key exists for safety)
            return 1.0

class FastFencePointer:
    """Enhanced ML-based fence pointer predictor with hierarchical approach.
    
    This model predicts which page in a run contains a key using regression
    rather than classification, with rich page-aware features and a hierarchical
    approach for better accuracy with large page counts.
    """
    
    def __init__(self, min_samples_per_page=50, max_samples_per_page=1000, max_total_samples=30000):
        # Main model for predicting page numbers
        self.model = None
        # Coarse-grained model for bucket prediction
        self.coarse_model = None
        # Dictionary mapping bucket IDs to fine-grained models
        self.fine_models = {}
        
        self.trained = False
        self.accuracy = 0.0
        self.page_counts = {}  # Track page counts per level
        self.exact_hits = 0
        self.total_predictions = 0
        self.near_hits = 0  # Within ±1 page
        
        # Page boundaries for feature engineering
        self.page_boundaries = {}  # level -> [(min_key, max_key, page_id), ...]
        self.bucket_size = 10  # Reduced from 20 to 10 for finer-grained buckets
        self.use_hierarchical = True  # Whether to use hierarchical approach
        
        # Classification threshold - use classification for levels with few pages
        self.classification_threshold = 100  # Increased from 20 to 100
        
        # Sampling parameters
        self.min_samples_per_page = min_samples_per_page
        self.max_samples_per_page = max_samples_per_page
        self.max_total_samples = max_total_samples
        
        # Cache for predictions
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _create_features_for_key(self, key, level):
        """Create rich features for a single key to be used for prediction."""
        # Ensure required attributes exist for feature creation
        if not hasattr(self, 'bucket_size'):
            self.bucket_size = 20
        if not hasattr(self, 'page_boundaries'):
            self.page_boundaries = {}
        if not hasattr(self, 'page_counts'):
            self.page_counts = {}
            
        # Basic features
        features = [
            float(key),               # Raw key
            float(level),             # Level
            np.log1p(abs(float(key))) # Log-transformed key 
        ]
        
        # Add page-aware features if we have page boundary information
        if level in self.page_boundaries and self.page_boundaries[level]:
            boundaries = self.page_boundaries[level]
            
            # Find distances to each page boundary
            min_distance = float('inf')
            closest_page = 0
            page_position = 0.0  # Estimated position along pages (0.0 to page_count)
            
            # Global min and max for normalized position
            global_min_key = boundaries[0][0] if boundaries else 0
            global_max_key = boundaries[-1][1] if boundaries else 1
            
            for min_key, max_key, page_id in boundaries:
                # Check if key is inside this page range
                if min_key <= key <= max_key:
                    # Key is inside this page range
                    # Compute fractional position within page
                    if max_key > min_key:
                        within_page_pos = (key - min_key) / (max_key - min_key)
                    else:
                        within_page_pos = 0.5
                    
                    # Set exact page position
                    page_position = float(page_id) + within_page_pos
                    closest_page = page_id
                    min_distance = 0
                    break
                else:
                    # Key is outside this page range
                    # Calculate distance to page boundaries
                    dist_to_min = abs(key - min_key)
                    dist_to_max = abs(key - max_key)
                    dist = min(dist_to_min, dist_to_max)
                    
                    if dist < min_distance:
                        min_distance = dist
                        closest_page = page_id
                        if key < min_key:
                            # Key is before this page
                            page_position = float(page_id) - (dist_to_min / (min_key + 1))
                        else:
                            # Key is after this page
                            page_position = float(page_id) + 1.0 + (dist_to_max / (max_key + 1))
            
            # Calculate distance to global min/max (normalized)
            dist_to_global_min = abs(key - global_min_key) / (global_max_key - global_min_key + 1e-10)
            dist_to_global_max = abs(key - global_max_key) / (global_max_key - global_min_key + 1e-10)
            
            # Add page-aware features
            features.extend([
                closest_page,          # Closest page based on boundaries
                page_position,         # Estimated fractional page position
                min_distance,          # Distance to closest page boundary
                page_position / max(1, len(boundaries)),  # Normalized position (0-1)
                dist_to_global_min,    # Distance to global min (normalized)
                dist_to_global_max,    # Distance to global max (normalized)
                page_position / max(1, self.page_counts.get(level, 1))  # True normalized position
            ])
            
            # Add bucket information for hierarchical approach
            if len(boundaries) > self.bucket_size:
                bucket_count = (len(boundaries) + self.bucket_size - 1) // self.bucket_size
                bucket = min(int(page_position / self.bucket_size), bucket_count - 1)
                relative_pos = page_position - (bucket * self.bucket_size)
                
                features.extend([
                    bucket,             # Coarse bucket ID
                    relative_pos        # Position within bucket
                ])
        
        return features
    
    def train(self, X, y, progress_callback=None, cv_folds=3, n_iter=10, early_stopping=True):
        """Train the fence pointer model using regression and hierarchical approach.
        
        Parameters:
        -----------
        X : List[List[float]]
            Feature vectors for each key
        y : List[int]
            Target page numbers for each key
        progress_callback : callable, optional
            Callback for progress tracking
        cv_folds : int
            Number of cross-validation folds (unused in this implementation)
        n_iter : int
            Number of iterations (used for n_estimators)
        early_stopping : bool
            Whether to use early stopping (default: True)
            
        Returns:
        --------
        bool
            True if training succeeded
        """
        try:
            # Ensure required attributes exist
            if not hasattr(self, 'classification_threshold'):
                self.classification_threshold = 20  # Default threshold for small page counts
                
            # Ensure sampling parameters exist
            if not hasattr(self, 'min_samples_per_page'):
                self.min_samples_per_page = 50
            if not hasattr(self, 'max_samples_per_page'):
                self.max_samples_per_page = 500
            if not hasattr(self, 'max_total_samples'):
                self.max_total_samples = 10000
                
            # Ensure hierarchical parameters exist
            if not hasattr(self, 'bucket_size'):
                self.bucket_size = 20  # Default bucket size
            if not hasattr(self, 'use_hierarchical'):
                self.use_hierarchical = True  # Enable hierarchical by default
                
            # Ensure page structures exist
            if not hasattr(self, 'page_boundaries'):
                self.page_boundaries = {}
            if not hasattr(self, 'page_counts'):
                self.page_counts = {}
            
            # Import libraries - try to use LightGBM for better performance if available
            try:
                import lightgbm as lgb
                use_lightgbm = True
                print("Using LightGBM for faster training")
            except ImportError:
                use_lightgbm = False
                from sklearn.ensemble import GradientBoostingRegressor
                print("LightGBM not available, using GradientBoostingRegressor")
                
            from sklearn.model_selection import train_test_split
            from sklearn.utils import resample
            from sklearn.metrics import accuracy_score
            import numpy as np
            
            if len(X) == 0 or len(y) == 0:
                print("No training data provided")
                self.trained = True
                self.accuracy = 0.25  # Default when no data
                return True
                
            # Count unique pages and examine distribution for stratified sampling
            unique_pages = sorted(set(y))
            page_counts = {page: y.count(page) for page in unique_pages}
            
            print(f"Training with {len(unique_pages)} unique pages")
            
            # Determine if we should use classification instead of regression
            use_classification = len(unique_pages) <= self.classification_threshold
            if use_classification:
                print(f"Using classification approach for small page count ({len(unique_pages)} pages)")
            
            # Extract raw keys from feature vectors
            raw_keys = []
            for i, item in enumerate(X):
                # Extract key (first element) if X contains feature vectors
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    key = item[0]  # First element is the raw key
                else:
                    key = item  # X contains raw keys already
                raw_keys.append(key)
            
            # Create balanced dataset with stratified sampling and oversampling of rare pages
            balanced_X = []
            balanced_y = []
            balanced_raw = []
            
            # For very small pages, increase min_samples to ensure enough data
            min_samples = max(self.min_samples_per_page, 
                              min(300, self.max_samples_per_page // 2))
            
            # Determine which pages need oversampling (less than min_samples)
            pages_to_oversample = {page for page, count in page_counts.items() 
                                  if count < min_samples}
            
            print(f"Oversampling {len(pages_to_oversample)} under-represented pages " 
                  f"(out of {len(unique_pages)} total pages)")
            
            # Process each page's samples
            for page in unique_pages:
                # Get indices for samples of this page
                page_indices = [i for i, label in enumerate(y) if label == page]
                
                if not page_indices:
                    continue  # Skip empty pages
                
                # Get data for this page
                page_X = [X[i] for i in page_indices]
                page_y = [y[i] for i in page_indices]
                page_raw = [raw_keys[i] for i in page_indices]
                
                if len(page_indices) > self.max_samples_per_page:
                    # Downsample if too many samples
                    sampled_indices = np.random.choice(
                        len(page_indices), 
                        self.max_samples_per_page, 
                        replace=False
                    )
                    page_X = [page_X[i] for i in sampled_indices]
                    page_y = [page_y[i] for i in sampled_indices]
                    page_raw = [page_raw[i] for i in sampled_indices]
                elif page in pages_to_oversample:
                    # Oversample if not enough samples
                    target_samples = min(min_samples, self.max_samples_per_page)
                    if len(page_indices) < target_samples:
                        # Oversample with replacement to reach at least min_samples
                        resampled_indices = np.random.choice(
                            len(page_indices), 
                            target_samples - len(page_indices), 
                            replace=True
                        )
                        page_X.extend([page_X[i] for i in resampled_indices])
                        page_y.extend([page_y[i] for i in resampled_indices])
                        page_raw.extend([page_raw[i] for i in resampled_indices])
                
                # Add to balanced dataset
                balanced_X.extend(page_X)
                balanced_y.extend(page_y)
                balanced_raw.extend(page_raw)
            
            # Limit total samples if needed
            MAX_TOTAL_SAMPLES = self.max_total_samples
            if len(balanced_X) > MAX_TOTAL_SAMPLES:
                print(f"Limiting from {len(balanced_X)} to {MAX_TOTAL_SAMPLES} samples for faster training")
                # Ensure stratified sampling is maintained
                combined = list(zip(balanced_X, balanced_y, balanced_raw))
                np.random.seed(42)
                np.random.shuffle(combined)
                balanced_X, balanced_y, balanced_raw = zip(*combined[:MAX_TOTAL_SAMPLES])
                balanced_X, balanced_y, balanced_raw = list(balanced_X), list(balanced_y), list(balanced_raw)
            
            print(f"Using {len(balanced_X)} balanced samples for training")
            
            # Create enhanced features for each key
            X_enhanced = []
            for i, key in enumerate(balanced_raw):
                features = self._create_features_for_key(key, 0)
                X_enhanced.append(features)
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val, raw_train, raw_val = train_test_split(
                X_enhanced, balanced_y, balanced_raw, test_size=0.2, random_state=42, stratify=balanced_y
            )
            
            # Determine if we should use hierarchical approach
            use_hierarchy = (self.use_hierarchical and 
                         len(unique_pages) > self.bucket_size and
                         not use_classification)  # Don't use hierarchical with classification
            
            # Train main model based on approach (classification or regression)
            if use_classification:
                # Use LightGBM classifier if available
                if use_lightgbm:
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder().fit(y_train)
                    y_train_encoded = label_encoder.transform(y_train)
                    y_val_encoded = label_encoder.transform(y_val)
                    
                    self.model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        verbosity=-1,  # Silence warnings
                        verbose=-1     # Turn off LightGBM stdout
                    )
                    
                    # Train with early stopping if enabled
                    eval_set = [(np.array(X_val), np.array(y_val_encoded))] if early_stopping else None
                    self.model.fit(np.array(X_train), np.array(y_train_encoded), 
                                  eval_set=eval_set)
                    
                    # Store label encoder for prediction
                    self.label_encoder = label_encoder
                    
                    # Calculate accuracy
                    y_pred = self.model.predict(X_val)
                    self.accuracy = accuracy_score(y_val_encoded, y_pred)
                else:
                    # Fall back to GradientBoostingClassifier if LightGBM not available
                    from sklearn.ensemble import GradientBoostingClassifier
                    self.model = GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42
                    )
                    self.model.fit(X_train, y_train)
                    
                    # Calculate accuracy
                    y_pred = self.model.predict(X_val)
                    self.accuracy = accuracy_score(y_val, y_pred)
                
                # For classification, we don't need hierarchical approach
                self.use_hierarchical = False
            else:
                # Standard regression approach
                if use_lightgbm:
                    self.model = lgb.LGBMRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42,
                        early_stopping_rounds=5 if early_stopping else None,  # Early stopping
                        verbosity=-1,  # Silence warnings
                        verbose=-1     # Turn off LightGBM stdout
                    )
                    
                    # Set up validation data for early stopping
                    eval_set = [(np.array(X_val), np.array(y_val))] if early_stopping else None
                    self.model.fit(np.array(X_train), np.array(y_train), 
                                  eval_set=eval_set)
                else:
                    self.model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=42
                    )
                    self.model.fit(X_train, y_train)
                
                print(f"Training regression model with {len(X_train)} samples...")
                
                # Hierarchical approach for models with many pages
                if use_hierarchy:
                    print(f"Using hierarchical approach with bucket size {self.bucket_size}")
                    
                    # Calculate bucket count
                    bucket_count = (len(unique_pages) + self.bucket_size - 1) // self.bucket_size
                    
                    # Map each page to its bucket
                    page_to_bucket = {page: min(page // self.bucket_size, bucket_count - 1) 
                                     for page in unique_pages}
                    
                    # Assign samples to buckets
                    bucket_data = {}
                    for i, page in enumerate(balanced_y):
                        bucket = page_to_bucket[page]
                        
                        if bucket not in bucket_data:
                            bucket_data[bucket] = {"X": [], "y": [], "pages": set()}
                        
                        bucket_data[bucket]["X"].append(X_enhanced[i])
                        # Store relative page within bucket
                        bucket_data[bucket]["y"].append(page - (bucket * self.bucket_size))
                        bucket_data[bucket]["pages"].add(page)
                    
                    # Train coarse-grained model for bucket prediction
                    y_bucket = [page_to_bucket[page] for page in balanced_y]
                    
                    if use_lightgbm:
                        self.coarse_model = lgb.LGBMRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            random_state=42,
                            early_stopping_rounds=5 if early_stopping else None,  # Early stopping
                            verbosity=-1,  # Silence warnings
                            verbose=-1     # Turn off LightGBM stdout
                        )
                        # Split for validation if early stopping
                        X_train_bucket, X_val_bucket, y_train_bucket, y_val_bucket = train_test_split(
                            X_enhanced, y_bucket, test_size=0.2, random_state=42
                        )
                        
                        # Train with early stopping if enabled
                        eval_set = [(np.array(X_val_bucket), np.array(y_val_bucket))] if early_stopping else None
                        self.coarse_model.fit(np.array(X_train_bucket), np.array(y_train_bucket), 
                                             eval_set=eval_set)
                    else:
                        self.coarse_model = GradientBoostingRegressor(
                            n_estimators=100, 
                            max_depth=6, 
                            learning_rate=0.1,
                            random_state=42
                        )
                        self.coarse_model.fit(X_enhanced, y_bucket)
                    
                    # Train fine-grained models for each bucket that has enough data
                    self.fine_models = {}
                    for bucket, data in bucket_data.items():
                        if len(data["X"]) >= 50:  # Only train if we have enough data
                            # Split data for training and validation
                            X_fine_train, X_fine_val, y_fine_train, y_fine_val = train_test_split(
                                data["X"], data["y"], test_size=0.2, random_state=42
                            )
                            
                            # Train fine-grained model
                            if use_lightgbm:
                                fine_model = lgb.LGBMRegressor(
                                    n_estimators=100,
                                    max_depth=5,  # Smaller trees for fine models
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    random_state=42,
                                    early_stopping_rounds=5 if early_stopping else None,
                                    verbosity=-1,  # Silence warnings
                                    verbose=-1     # Turn off LightGBM stdout
                                )
                                
                                # Set up validation data for early stopping
                                eval_set = [(np.array(X_fine_val), np.array(y_fine_val))] if early_stopping else None
                                fine_model.fit(np.array(X_fine_train), np.array(y_fine_train), 
                                             eval_set=eval_set)
                            else:
                                fine_model = GradientBoostingRegressor(
                                    n_estimators=100, 
                                    max_depth=5, 
                                    learning_rate=0.1,
                                    random_state=42
                                )
                                fine_model.fit(X_fine_train, y_fine_train)
                            
                            self.fine_models[bucket] = fine_model
                    
                    print(f"Trained {len(self.fine_models)} fine-grained models for {bucket_count} buckets")
            
            # Mark as trained
            self.trained = True
            
            # Capture predicted vs actual for evaluation
            y_pred = []
            for i in range(len(X_val)):
                if use_classification and use_lightgbm:
                    # For classification with LightGBM
                    features = X_val[i]
                    pred_page_encoded = self.model.predict([features])[0]
                    pred_page = self.label_encoder.inverse_transform([pred_page_encoded])[0]
                    y_pred.append(pred_page)
                elif use_classification:
                    # For classification without LightGBM
                    features = X_val[i]
                    pred_page = self.model.predict([features])[0]
                    y_pred.append(pred_page)
                else:
                    # Standard prediction for regression
                    key = raw_val[i]
                    features = self._create_features_for_key(key, 0)
                    pred_page = self.predict(key, 0)
                    y_pred.append(pred_page)
            
            # Calculate accuracy (exact hits and near hits)
            exact_hits = sum(1 for i in range(len(y_val)) if y_pred[i] == y_val[i])
            near_hits = sum(1 for i in range(len(y_val)) 
                           if abs(y_pred[i] - y_val[i]) <= 1)
            
            self.exact_hits = exact_hits
            self.near_hits = near_hits
            self.total_predictions = len(y_val)
            
            # Calculate accuracy metrics
            exact_accuracy = exact_hits / len(y_val) if len(y_val) > 0 else 0
            near_accuracy = near_hits / len(y_val) if len(y_val) > 0 else 0
            
            print(f"Fence pointer exact accuracy: {exact_accuracy:.2%}")
            print(f"Fence pointer ±1 page accuracy: {near_accuracy:.2%}")
            
            # Store the best accuracy metric
            self.accuracy = near_accuracy
            
            # Log completion
            if progress_callback:
                progress_callback(1.0, f"Fence pointer model training complete, accuracy: {self.accuracy:.2%}")
                
            return True
            
        except Exception as e:
            import traceback
            print(f"Error training fence pointer model: {e}")
            traceback.print_exc()
            
            # Set default accuracy
            self.accuracy = 0.25
            self.trained = True  # Mark as trained anyway to avoid crashes
            
            if progress_callback:
                progress_callback(1.0, f"Fence pointer model training failed: {e}")
                
            return False

    def predict(self, key, level):
        """Predict which page in a run likely contains the key."""
        if not self.trained or self.model is None:
            # Default to page 0 if not trained
            return 0
            
        # Convert key to features
        try:
            # Create features for the key
            features = self._create_features_for_key(key, level)
            
            # Check if this is a classification model
            if hasattr(self, 'label_encoder'):
                # Use classification prediction
                pred_page_encoded = self.model.predict([features])[0]
                pred_page = self.label_encoder.inverse_transform([pred_page_encoded])[0]
                return int(pred_page)
            
            if self.use_hierarchical and level in self.page_counts and self.page_counts[level] > self.bucket_size:
                # Use hierarchical approach for levels with many pages
                
                # Step 1: Predict coarse bucket
                if self.coarse_model is not None:
                    bucket = int(self.coarse_model.predict([features])[0])
                    bucket = max(0, min(bucket, (self.page_counts[level] // self.bucket_size) - 1))
                    
                    # Step 2: Predict page within bucket using fine-grained model
                    if bucket in self.fine_models and self.fine_models[bucket] is not None:
                        # Predict within bucket
                        relative_page = self.fine_models[bucket].predict([features])[0]
                        # Convert to absolute page
                        predicted_page = bucket * self.bucket_size + int(relative_page)
                        predicted_page = max(0, min(predicted_page, self.page_counts[level] - 1))
                    else:
                        # Fallback to middle of bucket if no fine model
                        predicted_page = bucket * self.bucket_size + (self.bucket_size // 2)
                        predicted_page = max(0, min(predicted_page, self.page_counts[level] - 1))
                else:
                    # Fallback to main model
                    predicted_page = int(self.model.predict([features])[0])
            else:
                # Use direct regression for smaller page counts
                predicted_page = int(self.model.predict([features])[0])
                
                # Ensure valid page number
                if level in self.page_counts:
                    predicted_page = max(0, min(predicted_page, self.page_counts[level] - 1))
            
            # Track prediction stats
            self.total_predictions += 1
            
            # Return the predicted page
            return predicted_page
            
        except Exception as e:
            # Fallback to middle page if prediction fails
            if level in self.page_counts:
                max_page = self.page_counts[level] - 1
                if max_page > 0:
                    return max(0, min(int(max_page / 2), max_page))  # Middle page as default
            
            return 0  # First page as ultimate fallback

    def _extract_page_boundaries(self, X, y, level=0):
        """Extract page boundaries from training data.
        
        Safely handles feature vectors by extracting the raw key from each input.
        
        Parameters:
        -----------
        X : List or array-like
            Feature vectors or raw keys
        y : List or array-like
            Target page numbers
        level : int, optional
            The level to store boundaries for (default: 0)
        """
        # 1) First pull out the raw float keys from X
        raw_keys = []
        for item in X:
            if isinstance(item, (list, tuple)) and not isinstance(item, str):
                raw_keys.append(item[0])
            else:
                raw_keys.append(item)

        # 2) Build per-page lists
        pages = {}
        for key, page in zip(raw_keys, y):
            pages.setdefault(page, []).append(key)

        # 3) Compute level-wide min/max
        level_min_key = min(raw_keys) if raw_keys else 0
        level_max_key = max(raw_keys) if raw_keys else 1
        
        # Store level min/max if we have the attribute (optional)
        if hasattr(self, 'level_min_key'):
            self.level_min_key[level] = level_min_key
        if hasattr(self, 'level_max_key'):
            self.level_max_key[level] = level_max_key

        # 4) Now build the page_boundaries list
        boundaries = [
            (min(keys), max(keys), page)
            for page, keys in sorted(pages.items())
            if keys  # Skip empty lists
        ]
        
        # Store boundaries for the specified level
        self.page_boundaries[level] = boundaries
        
        # Store page count for the specified level
        self.page_counts[level] = len(pages)

class LSMMLModels:
    """Manager class for LSM tree ML models."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Number of levels in your tree
        self.MAX_LEVEL = 7
        
        # Use dictionaries of models by level
        # — Bloom filters: one binary classifier per level
        self.bloom_models = {
            lvl: FastBloomFilter()
            for lvl in range(self.MAX_LEVEL)
        }
        
        # — Fence pointers: one regressor/classifier per level
        self.fence_models = {
            lvl: FastFencePointer(
                min_samples_per_page=50,
                max_samples_per_page=1000,
                max_total_samples=30000
            )
            for lvl in range(self.MAX_LEVEL)
        }
        
        # Prediction cache - avoid repeated calculations
        self.bloom_cache = {}  # (key, level) -> prediction
        self.fence_cache = {}  # (key, level) -> prediction
        self.max_cache_size = 100000  # Increased cache size
        
        # Training data - now per level
        self.bloom_data = {lvl: [] for lvl in range(self.MAX_LEVEL)}  # key→True/False per level
        self.fence_data = {lvl: [] for lvl in range(self.MAX_LEVEL)}  # key→page per level
        
        # Accuracy tracking - these default values are used if training fails
        self.bloom_accuracy = {lvl: 0.9 for lvl in range(self.MAX_LEVEL)}  # Initialize with optimistic value
        self.fence_accuracy = {lvl: 0.25 for lvl in range(self.MAX_LEVEL)}  # Initialize with more conservative value
        
        # Timing metrics
        self.bloom_prediction_time = 0.0
        self.fence_prediction_time = 0.0
        self.bloom_prediction_count = 0
        self.fence_prediction_count = 0
        
        # Configuration
        self.ensemble_prediction = True  # Use ensemble prediction by default
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load existing models if available."""
        # Load all bloom filter models
        for lvl in range(self.MAX_LEVEL):
            bloom_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
            
            if os.path.exists(bloom_path):
                try:
                    with open(bloom_path, 'rb') as f:
                        self.bloom_models[lvl] = pickle.load(f)
                    # Ensure ensemble prediction is enabled
                    self.bloom_models[lvl].use_ensemble = self.ensemble_prediction
                    
                    # Ensure hash attribute is set correctly
                    if not hasattr(self.bloom_models[lvl], 'has_mmh3'):
                        try:
                            import mmh3
                            self.bloom_models[lvl].has_mmh3 = True
                            self.bloom_models[lvl].hash_function = mmh3.hash
                        except ImportError:
                            self.bloom_models[lvl].has_mmh3 = False
                            import hashlib
                            self.bloom_models[lvl].hash_function = lambda x: int.from_bytes(hashlib.md5(str(x).encode('utf-8')).digest()[:4], 'little')
                except Exception as e:
                    print(f"Error loading bloom model for level {lvl}: {e}")
        
        # Load all fence pointer models
        for lvl in range(self.MAX_LEVEL):
            fence_path = os.path.join(self.models_dir, f"fence_model_lvl_{lvl}.pkl")
            
            if os.path.exists(fence_path):
                try:
                    with open(fence_path, 'rb') as f:
                        self.fence_models[lvl] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading fence model for level {lvl}: {e}")
                
        # Load training statistics if available
        data_path = os.path.join(self.models_dir, "training_data.pkl")
        if os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data.get('bloom_accuracy'), dict):
                        self.bloom_accuracy = data.get('bloom_accuracy', self.bloom_accuracy)
                    if isinstance(data.get('fence_accuracy'), dict):
                        self.fence_accuracy = data.get('fence_accuracy', self.fence_accuracy)
                    
                    # Load training data if available
                    if isinstance(data.get('bloom_data'), dict):
                        self.bloom_data = data.get('bloom_data', self.bloom_data)
                    if isinstance(data.get('fence_data'), dict):
                        self.fence_data = data.get('fence_data', self.fence_data)
            except Exception as e:
                print(f"Error loading training data: {e}")
                
        print(f"Models loaded from {self.models_dir}")
        
    def _save_models(self):
        """Save models to disk."""
        # Save each bloom model per level
        for lvl in range(self.MAX_LEVEL):
            bloom_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
            
            try:
                with open(bloom_path, 'wb') as f:
                    pickle.dump(self.bloom_models[lvl], f)
            except Exception as e:
                print(f"Error saving bloom model for level {lvl}: {e}")
        
        # Save each fence model per level
        for lvl in range(self.MAX_LEVEL):
            fence_path = os.path.join(self.models_dir, f"fence_model_lvl_{lvl}.pkl")
            
            try:
                with open(fence_path, 'wb') as f:
                    pickle.dump(self.fence_models[lvl], f)
            except Exception as e:
                print(f"Error saving fence model for level {lvl}: {e}")
                
    def add_bloom_training_data(self, key: float, level_or_exists, exists=None):
        """Add training data for Bloom filter model per level.
        
        This method supports two calling conventions:
        1. New: add_bloom_training_data(key, level, exists)
        2. Legacy: add_bloom_training_data(key, exists) - will use level 0
        
        Parameters:
        -----------
        key : float
            The key to add to training data
        level_or_exists : int or bool
            Either the level number (0-6) or existence flag (backwards compatibility)
        exists : bool, optional
            True if key exists, False if it doesn't. If None, level_or_exists is
            treated as the exists flag and level is set to 0
        """
        try:
            # Handle legacy calling convention
            if exists is None:
                # Old style: add_bloom_training_data(key, exists)
                exists_bool = bool(level_or_exists)
                level_int = 0  # Default to level 0 for backward compatibility
            else:
                # New style: add_bloom_training_data(key, level, exists)
                level_int = int(level_or_exists)
                exists_bool = bool(exists)
            
            # Validate key
            key_float = float(key)
            
            # Ensure level is valid
            if level_int < 0 or level_int >= self.MAX_LEVEL:
                print(f"Skipping invalid level {level_int} (must be 0-{self.MAX_LEVEL-1})")
                return
                
            # Add to appropriate level list
            self.bloom_data[level_int].append((key_float, exists_bool))
            # Clear cache when new data is added
            self.bloom_cache = {}
        except (ValueError, TypeError) as e:
            print(f"Skipping invalid bloom filter training sample: {e}")
        
    def add_fence_training_data(self, key: float, level_or_page, page=None):
        """Add training data for fence pointer model per level.
        
        This method supports two calling conventions:
        1. New: add_fence_training_data(key, level, page)
        2. Legacy: add_fence_training_data(key, page) - will use level 0

        Parameters:
        -----------
        key : float
            The key to add training data for
        level_or_page : int
            Either the level number (0-6) or page number (backwards compatibility)
        page : int, optional
            The page number. If None, level_or_page is used as the page
            and level is set to 0
        """
        try:
            # Handle legacy calling convention
            if page is None:
                # Old style: add_fence_training_data(key, page)
                page_int = int(level_or_page)
                level_int = 0  # Default to level 0 for backward compatibility
            else:
                # New style: add_fence_training_data(key, level, page)
                level_int = int(level_or_page)
                page_int = int(page)
            
            # Validate key
            key_float = float(key)
            
            # Ensure level is valid
            if level_int < 0 or level_int >= self.MAX_LEVEL:
                print(f"Skipping invalid level {level_int} (must be 0-{self.MAX_LEVEL-1})")
                return
            
            # Add to training data for this level
            self.fence_data[level_int].append((key_float, page_int))
            # Clear cache when new data is added
            self.fence_cache = {}
        except (ValueError, TypeError) as e:
            print(f"Skipping invalid fence pointer training sample: {e}")
        
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
        
        # Ensure level is valid
        if level < 0 or level >= self.MAX_LEVEL:
            # Default to level 0 if invalid
            level = 0
            
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys)
        
        # Time the batch prediction
        start_time = time.perf_counter()
        
        # Create results array
        results = np.empty(len(keys), dtype=np.int32)
        
        # Simply predict each key individually using the level-specific model
        for i, key in enumerate(keys):
            results[i] = self.fence_models[level].predict(key, level)
            
            # Update cache for future single-key lookups
            cache_key = (key, level)
            if len(self.fence_cache) < self.max_cache_size:
                self.fence_cache[cache_key] = results[i]
        
        # Record timing
        end_time = time.perf_counter()
        prediction_time = (end_time - start_time)
        self.fence_prediction_time += prediction_time
        self.fence_prediction_count += len(keys)
        
        return results
        
    def get_bloom_accuracy(self) -> float:
        """Get average Bloom filter model accuracy across all levels."""
        # Return average accuracy across all trained levels
        trained_levels = [lvl for lvl, acc in self.bloom_accuracy.items() if acc > 0]
        if trained_levels:
            return sum(self.bloom_accuracy[lvl] for lvl in trained_levels) / len(trained_levels)
        return 0.0
        
    def get_fence_accuracy(self) -> float:
        """Get average fence pointer model accuracy across all levels."""
        # Return average accuracy across all trained levels
        trained_levels = [lvl for lvl, acc in self.fence_accuracy.items() if acc > 0]
        if trained_levels:
            return sum(self.fence_accuracy[lvl] for lvl in trained_levels) / len(trained_levels)
        return 0.0
        
    def get_bloom_accuracy_by_level(self) -> dict:
        """Get Bloom filter accuracy for each level."""
        return self.bloom_accuracy
        
    def get_fence_accuracy_by_level(self) -> dict:
        """Get fence pointer accuracy for each level."""
        return self.fence_accuracy
        
    def get_prediction_stats(self) -> dict:
        """Get prediction timing statistics."""
        # Collect cache hit statistics across all fence models
        fence_cache_hits = 0
        fence_cache_misses = 0
        for lvl, model in self.fence_models.items():
            fence_cache_hits += getattr(model, 'cache_hits', 0)
            fence_cache_misses += getattr(model, 'cache_misses', 0)
            
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
            'fence_direct_cache_hits': fence_cache_hits,
            'fence_direct_cache_misses': fence_cache_misses,
            'bloom_accuracy_by_level': self.bloom_accuracy,
            'fence_accuracy_by_level': self.fence_accuracy
        }
        return stats
        
    def save_models(self):
        """Save trained models to disk."""
        import pickle
        import os
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            
        # Save all bloom filter models
        for lvl, model in self.bloom_models.items():
            bloom_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
            try:
                with open(bloom_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Error saving bloom model for level {lvl}: {e}")
                
        # Save all fence pointer models
        for lvl, model in self.fence_models.items():
            fence_path = os.path.join(self.models_dir, f"fence_model_lvl_{lvl}.pkl")
            try:
                with open(fence_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Error saving fence model for level {lvl}: {e}")
            
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
        
        # Load all bloom filter models
        for lvl in range(self.MAX_LEVEL):
            bloom_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
            
            if os.path.exists(bloom_path):
                try:
                    with open(bloom_path, 'rb') as f:
                        self.bloom_models[lvl] = pickle.load(f)
                    # Ensure ensemble prediction is enabled
                    self.bloom_models[lvl].use_ensemble = self.ensemble_prediction
                    
                    # Ensure hash attribute is set correctly
                    if not hasattr(self.bloom_models[lvl], 'has_mmh3'):
                        try:
                            import mmh3
                            self.bloom_models[lvl].has_mmh3 = True
                            self.bloom_models[lvl].hash_function = mmh3.hash
                        except ImportError:
                            self.bloom_models[lvl].has_mmh3 = False
                            import hashlib
                            self.bloom_models[lvl].hash_function = lambda x: int.from_bytes(hashlib.md5(str(x).encode('utf-8')).digest()[:4], 'little')
                except Exception as e:
                    print(f"Error loading bloom model for level {lvl}: {e}")
        
        # Load all fence pointer models
        for lvl in range(self.MAX_LEVEL):
            fence_path = os.path.join(self.models_dir, f"fence_model_lvl_{lvl}.pkl")
            
            if os.path.exists(fence_path):
                try:
                    with open(fence_path, 'rb') as f:
                        self.fence_models[lvl] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading fence model for level {lvl}: {e}")
                
        # Load training statistics if available
        data_path = os.path.join(self.models_dir, "training_data.pkl")
        if os.path.exists(data_path):
            try:
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data.get('bloom_accuracy'), dict):
                        self.bloom_accuracy = data.get('bloom_accuracy', self.bloom_accuracy)
                    if isinstance(data.get('fence_accuracy'), dict):
                        self.fence_accuracy = data.get('fence_accuracy', self.fence_accuracy)
                    
                    # Load training data if available
                    if isinstance(data.get('bloom_data'), dict):
                        self.bloom_data = data.get('bloom_data', self.bloom_data)
                    if isinstance(data.get('fence_data'), dict):
                        self.fence_data = data.get('fence_data', self.fence_data)
            except Exception as e:
                print(f"Error loading training data: {e}")
                
        print(f"Models loaded from {self.models_dir}") 
        
    @property
    def fence_data_length(self):
        """Get total number of fence pointer training samples across all levels.
        
        This is used for backward compatibility where code checks len(fence_data).
        """
        total = 0
        for level_data in self.fence_data.values():
            total += len(level_data)
        return total
        
    # For backward compatibility - make bloom_data and fence_data appear as lists when accessed directly
    def __getattribute__(self, name):
        """Custom attribute getter that handles backward compatibility for data attributes."""
        # Get the actual attribute
        attr = super().__getattribute__(name)
        
        # For backward compatibility: if code tries to iterate over bloom_data or fence_data directly
        # or check their length, we provide a flattened version
        if name == 'bloom_data' and isinstance(attr, dict):
            # When bloom_data is accessed directly, combine all level data for backward compatibility
            # But only if the calling method is not one of our own methods that already handles the dict structure
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_function = caller_frame.f_code.co_name if caller_frame else ""
            
            # If caller is train_bloom_models or any other method that knows about dict structure
            if caller_function in ('train_bloom_models', 'train_bloom_model', 'add_bloom_training_data', 
                                  'load_models', 'save_models'):
                return attr
            
            # Otherwise, return a flattened list for backward compatibility
            flat_data = []
            for level_data in attr.values():
                flat_data.extend(level_data)
            return flat_data
            
        elif name == 'fence_data' and isinstance(attr, dict):
            # Similar handling for fence_data
            import inspect
            caller_frame = inspect.currentframe().f_back
            caller_function = caller_frame.f_code.co_name if caller_frame else ""
            
            # If caller is train_fence_models or any other method that knows about dict structure
            if caller_function in ('train_fence_models', 'train_fence_model', 'add_fence_training_data',
                                  'load_models', 'save_models'):
                return attr
                
            # Otherwise, return a flattened list for backward compatibility
            flat_data = []
            for level_data in attr.values():
                flat_data.extend(level_data)
            return flat_data
                
        # Return the normal attribute for all other cases
        return attr

    def train_bloom_model(self):
        """Train the bloom filter model for level 0 only (for backward compatibility)."""
        print("Warning: Calling train_bloom_model which only trains level 0. Consider using train_bloom_models() instead.")
        
        if not self.bloom_data[0]:
            print("No training data for bloom filter at level 0")
            return
            
        # Extract keys and existence flags
        keys = []
        exists_flags = []
        
        # Process and validate training data
        for key, exists in self.bloom_data[0]:
            try:
                # Ensure key is a valid float
                key_float = float(key)
                # Ensure exists is a valid boolean (0 or 1)
                exists_bool = bool(exists)
                
                keys.append(key_float)
                exists_flags.append(exists_bool)
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid training sample: {e}")
                continue
        
        # Skip if all data was invalid
        if not keys:
            print("No valid training data after filtering")
            return
        
        # Train the model
        print(f"Training bloom filter model with {len(keys)} samples")
        try:
            self.bloom_models[0].train(keys, exists_flags)
            
            # Get accuracy and recall from the model
            self.bloom_accuracy[0] = self.bloom_models[0].accuracy
            print(f"Initial bloom accuracy from model: {self.bloom_accuracy[0]:.2%}")
        except Exception as e:
            print(f"Error training bloom filter model: {e}")
            # Provide more detailed error information for debugging
            import traceback
            traceback.print_exc()
            
            # If training failed, we keep the default accuracy value
            print(f"Using default bloom accuracy due to training failure: {self.bloom_accuracy[0]:.2%}")
        
        # Clear cache
        self.bloom_cache = {}
        
        self._save_models()

    def train_bloom_models(self):
        """Train all bloom filter models (one per level)."""
        import logging
        
        for lvl, model in self.bloom_models.items():
            data = self.bloom_data.get(lvl, [])
            if not data:
                logging.warning(f"[Bloom] No training data for level {lvl}, skipping.")
                print(f"Level {lvl} has no bloom filter training data, skipping...")
                continue
                
            print(f"Training bloom filter model for level {lvl} with {len(data)} samples")
            
            # Extract keys and existence flags
            try:
                X, y = zip(*data)
                X, y = list(X), list(y)
            except ValueError:
                # Handle empty data list
                logging.warning(f"[Bloom] Empty or invalid data for level {lvl}, skipping.")
                print(f"Empty or invalid bloom filter data for level {lvl}, skipping.")
                continue
            
            # Train the model for this level
            try:
                model.train(X, y)
                
                # Get accuracy from the model and store it
                self.bloom_accuracy[lvl] = model.accuracy
                print(f"Bloom filter accuracy for level {lvl}: {self.bloom_accuracy[lvl]:.2%}")
                
                # Persist the trained model
                save_path = os.path.join(self.models_dir, f"bloom_model_lvl_{lvl}.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump(model, f)
                logging.info(f"[Bloom] Trained and saved level {lvl} to {save_path}")
                print(f"Saved bloom model for level {lvl} to {save_path}")
                
            except Exception as e:
                logging.error(f"[Bloom] Error training level {lvl}: {e}")
                print(f"Error training bloom filter model for level {lvl}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clear cache
        self.bloom_cache = {}
        
        # Save all models to disk (this includes metadata like accuracies)
        self._save_models()

    def train_fence_models(self, X_by_level, y_by_level, progress_callback=None, **kwargs):
        """Train fence pointer models for each level."""
        if not hasattr(self, 'use_ml'):
            setattr(self, 'use_ml', True)
        
        for lvl, training_data in X_by_level.items():
            if not training_data or lvl not in y_by_level:
                continue
                
            X, y = training_data, y_by_level[lvl]
                
            if X and y and len(X) > 0 and len(y) > 0:
                model = self.fence_models.get(lvl)
                if not model:
                    model = FastFencePointer(min_samples_per_page=getattr(self, 'min_samples_per_page', 50),
                                           max_samples_per_page=getattr(self, 'max_samples_per_page', 1000),
                                           max_total_samples=getattr(self, 'max_total_samples', 30000))
                    self.fence_models[lvl] = model
                
                # Extract page boundaries with the correct level before training
                model._extract_page_boundaries(X, y, lvl)
                
                # Train the model
                model.train(X, y, progress_callback=progress_callback, **kwargs)
                
                # ... existing code ...

    def train_fence_model(self, cv_folds=3, n_iter=10, early_stopping=False, precompute_features=False):
        """Train the fence pointer model for level 0 only (for backward compatibility)."""
        print("Warning: Calling train_fence_model which only trains level 0. Consider using train_fence_models() instead.")
        
        if not self.fence_data[0]:
            print("No fence pointer training data for level 0")
            return
        
        # Try to import tqdm for progress bars
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
            print("Note: Install tqdm package for progress bars (pip install tqdm)")
            
        # Extract keys and pages
        X = []
        y = []
        
        for key, page in self.fence_data[0]:
            X.append(key)
            y.append(page)
            
        if not X:
            print("No valid fence pointer training data after filtering")
            return
            
        # Train the model
        start_time = time.perf_counter()
        
        print(f"Training fence pointer model with {len(X)} samples")
        
        try:
            # Use precomputed features if requested to speed up training
            if precompute_features and hasattr(self.fence_models[0], 'precompute_fence_features'):
                print("OPTIMIZATION: Precomputing features once to avoid redundant calculations...")
                feature_start = time.perf_counter()
                X_features, y_processed = self.fence_models[0].precompute_fence_features(X, y)
                feature_time = time.perf_counter() - feature_start
                print(f"Feature precomputation completed in {feature_time:.2f} seconds")
                
                # Train with precomputed features
                if has_tqdm:
                    with tqdm(total=100, desc="Training fence pointer model") as pbar:
                        def progress_update(progress):
                            pbar.update(int(progress * 100) - pbar.n)
                        # Train the model with cached features and progress updates
                        self.fence_models[0].train(
                            X_features, 
                            y_processed, 
                            progress_callback=progress_update,
                            cv_folds=cv_folds,
                            n_iter=n_iter,
                            early_stopping=early_stopping
                        )
                        pbar.update(100 - pbar.n)
                else:
                    # Train without progress updates but with cached features
                    self.fence_models[0].train(
                        X_features, 
                        y_processed,
                        cv_folds=cv_folds,
                        n_iter=n_iter,
                        early_stopping=early_stopping
                    )
            else:
                # Standard training without feature precomputation
                if has_tqdm:
                    with tqdm(total=100, desc="Training fence pointer model") as pbar:
                        def progress_update(progress):
                            pbar.update(int(progress * 100) - pbar.n)
                        self.fence_models[0].train(X, y, progress_callback=progress_update)
                        pbar.update(100 - pbar.n)
                else:
                    self.fence_models[0].train(X, y)
        except Exception as e:
            print(f"Error training fence pointer model: {e}")
            # Fall back to default accuracy
            print(f"Using default fence accuracy due to training failure: {self.fence_accuracy[0]:.2%}")
            
        # Record training time
        elapsed = time.perf_counter() - start_time
        print(f"Training completed in {elapsed:.2f} seconds")
        
        # Get accuracy from the model
        self.fence_accuracy[0] = self.fence_models[0].accuracy
        print(f"Fence pointer accuracy set from model: {self.fence_accuracy[0]:.2%}")
        
        # Clear cache
        self.fence_cache = {}
        
        # Save models
        self._save_models()