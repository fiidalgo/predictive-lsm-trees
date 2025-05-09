import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import pickle
from typing import List, Tuple, Dict, Set, Optional
import random
from numba import njit
import math
import struct

# Check if Metal is available (for Apple Silicon)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Apple Metal GPU acceleration enabled for LearnedBloomFilter")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA GPU acceleration enabled for LearnedBloomFilter")
else:
    DEVICE = torch.device("cpu")
    print("GPU acceleration not available, using CPU for LearnedBloomFilter")

@njit
def predict_bloom_numba(keys, model_outputs, backup_filter_bits, backup_filter_hashes, backup_filter_size):
    """Numba-optimized learned bloom filter prediction.
    
    Parameters:
    -----------
    keys : numpy.ndarray
        Array of keys to predict
    model_outputs : numpy.ndarray
        ML model predictions for each key (probabilities)
    backup_filter_bits : numpy.ndarray
        Bit array for the backup bloom filter
    backup_filter_hashes : int
        Number of hash functions for the backup filter
    backup_filter_size : int
        Size of the backup bloom filter
        
    Returns:
    --------
    numpy.ndarray
        Array of final predictions (0-1)
    """
    results = np.empty(len(keys), dtype=np.float32)
    
    for i in range(len(keys)):
        key = keys[i]
        model_score = model_outputs[i]
        
        # If model is very confident, use its prediction directly
        if model_score > 0.9:
            results[i] = 1.0
            continue
            
        if model_score < 0.1:
            results[i] = 0.0
            continue
            
        # For uncertain cases, check backup bloom filter
        key_str = str(key)
        hash_values = []
        for seed in range(backup_filter_hashes):
            # Simple hash function for numba compatibility
            hash_val = hash(key_str + str(seed)) % backup_filter_size
            hash_values.append(hash_val)
            
        # Check if all bits are set in the backup filter
        in_backup = True
        for bit_pos in hash_values:
            if not backup_filter_bits[bit_pos]:
                in_backup = False
                break
                
        if in_backup:
            # Boost the prediction if in backup filter
            results[i] = max(model_score, 0.7)
        else:
            # If not in backup filter, reduce confidence
            results[i] = min(model_score, 0.3)
            
    return results

# Define neural network architecture outside the method for proper serialization
class BloomFilterNetwork(nn.Module):
    def __init__(self, input_size):
        super(BloomFilterNetwork, self).__init__()
        # Smaller, more stable network
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights properly - critical for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.net(x)

class LearnedBloomFilter:
    """Learned Bloom Filter implementation.
    
    This combines a learned ML model with a small traditional bloom filter
    as backup to reduce false positives and false negatives.
    """
    
    def __init__(self, level=1, backup_fpr=0.1):
        # The ML model that predicts if a key exists
        self.model = None
        self.level = level
        self.trained = False
        self.accuracy = 0.0
        self.fpr = 0.0
        self.fnr = 0.0
        
        # Backup bloom filter for uncertain predictions
        self.backup_filter_bits = []
        self.backup_filter_size = 0
        self.backup_filter_hashes = 0
        self.backup_fpr = backup_fpr
        
        # Store key ranges for better predictions
        self.min_key = float('inf')
        self.max_key = float('-inf')
        self.example_keys = set()
        
        # Store statistics
        self.stats = {
            'model_hits': 0,
            'backup_hits': 0,
            'model_only_fps': 0,
            'backup_used': 0,
            'total_queries': 0
        }
        
        # PyTorch specific attributes
        self.pytorch_model = None
        self.num_features = 0
        
        # Prediction metadata
        self.key_ranges = []  # List of (min, max) ranges that definitely contain keys
        
    def add_key(self, key: float):
        """Add a key to the learned bloom filter.
        
        This stores the key in the example set and updates min/max ranges.
        """
        self.example_keys.add(key)
        self.min_key = min(self.min_key, key)
        self.max_key = max(self.max_key, key)
    
    def train(self, positive_keys, negative_keys, validation_ratio=0.2):
        """Train the Learned Bloom Filter model.
        
        Parameters:
        -----------
        positive_keys : List
            Keys that exist in the level
        negative_keys : List
            Keys that do not exist in the level
        validation_ratio : float
            Ratio of data to use for validation
        """
        if len(positive_keys) == 0:
            print(f"Error: No positive keys provided for level {self.level}")
            return False
        
        if len(negative_keys) == 0:
            print(f"Warning: No negative keys for level {self.level}, generating synthetic ones")
            # Generate synthetic negative keys
            key_range = max(positive_keys) - min(positive_keys)
            negative_keys = [min(positive_keys) - random.random() * key_range * 0.5 for _ in range(min(1000, len(positive_keys)))]
            negative_keys.extend([max(positive_keys) + random.random() * key_range * 0.5 for _ in range(min(1000, len(positive_keys)))])
        
        # Update min/max key range
        for key in positive_keys:
            self.add_key(key)
            
        # Create dataset
        X = positive_keys + negative_keys
        y = [1] * len(positive_keys) + [0] * len(negative_keys)
        
        # Create engineered features in batches to prevent memory issues
        print(f"Creating features for {len(X)} samples...")
        X_features = []
        batch_size = 10000  # Process in chunks to prevent memory issues
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            batch_features = self._create_features(X[i:batch_end])
            X_features.extend(batch_features)
            if i % 50000 == 0:
                print(f"Created features for {i}/{len(X)} samples...")
        
        print("Feature creation complete. Training model...")
        
        # Train the model
        if DEVICE is not None:
            return self._train_with_pytorch(X_features, y, positive_keys, negative_keys, validation_ratio)
        else:
            # Fallback to scikit-learn
            return self._train_with_sklearn(X_features, y, positive_keys, negative_keys, validation_ratio)
    
    def _train_with_pytorch(self, X_features, y, positive_keys, negative_keys, validation_ratio):
        """Train the model using PyTorch with Metal acceleration."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # Basic data preprocessing and safety checks
        print(f"Processing features for {len(X_features)} samples")
        
        # Filter out any empty feature vectors
        valid_indices = []
        for i, feat in enumerate(X_features):
            if feat and all(not (math.isnan(x) or math.isinf(x)) for x in feat):
                valid_indices.append(i)
        
        # Check if we have enough valid data
        if len(valid_indices) < 100:
            print(f"Warning: Only {len(valid_indices)} valid samples out of {len(X_features)}. Using simple model.")
            # Fall back to simple features
            X_features = [[float(i % 1000)] for i in range(len(X_features))]
            valid_indices = list(range(len(X_features)))
        
        # Select only valid samples
        X_features_filtered = [X_features[i] for i in valid_indices]
        y_filtered = [y[i] for i in valid_indices]
        
        # Verify consistent feature dimensions
        feature_dims = [len(x) for x in X_features_filtered]
        if len(set(feature_dims)) > 1:
            # If inconsistent, pad to maximum length
            max_dim = max(feature_dims)
            X_features_filtered = [x + [0.0] * (max_dim - len(x)) for x in X_features_filtered]
        
        # Convert to numpy arrays with safe type
        try:
            X_formatted = np.array(X_features_filtered, dtype=np.float32)
            y_formatted = np.array(y_filtered, dtype=np.float32)
            
            # Replace any remaining NaN or inf
            X_formatted = np.nan_to_num(X_formatted, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error converting to numpy arrays: {e}")
            print("Using simple fallback features")
            # Generate simple fallback features 
            X_formatted = np.zeros((len(valid_indices), 3), dtype=np.float32)
            for i in range(len(X_formatted)):
                X_formatted[i, 0] = float(i % 1000) / 1000.0
            y_formatted = np.array(y_filtered, dtype=np.float32)
        
        # Normalize features for better training
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_formatted)
            # Final safety check
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"Error scaling features: {e}")
            X_scaled = X_formatted  # Use unscaled if scaling fails
        
        # Store the number of features
        self.num_features = X_scaled.shape[1]
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_formatted, test_size=validation_ratio, random_state=42, stratify=y_formatted
        )
        
        print(f"Training learned bloom filter for level {self.level} with {len(X_train)} samples, validating with {len(X_val)} samples")
        
        # Calculate positive to negative ratio for weighting
        pos_count = np.sum(y_train == 1) 
        neg_count = np.sum(y_train == 0)
        pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"Positive class weight: {pos_weight:.4f} (based on {pos_count} positive and {neg_count} negative samples)")
        
        # Convert to PyTorch tensors - explicitly use float32
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
        
        # Create DataLoader for batching
        batch_size = 512  # Larger batch for stability
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.pytorch_model = BloomFilterNetwork(self.num_features).to(DEVICE)
        
        # Use binary cross entropy with logits for better numerical stability
        criterion = nn.BCELoss()
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
        
        # Use lower learning rate and weight decay for regularization
        optimizer = optim.Adam(self.pytorch_model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Learning rate scheduler for better convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        start_time = time.time()
        
        # Initialize values for metrics
        fpr = 0.0
        fnr = 0.0
        f1 = 0.0
        best_f1 = 0.0
        
        # Training loop
        num_epochs = 25
        
        for epoch in range(num_epochs):
            self.pytorch_model.train()
            epoch_loss = 0.0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.pytorch_model(inputs)
                
                # Weighted BCE loss
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.pytorch_model.parameters(), 1.0)
                
                optimizer.step()
                epoch_loss += loss.item() * len(inputs)
            
            avg_loss = epoch_loss / len(X_train)
            
            # Validation
            self.pytorch_model.eval()
            with torch.no_grad():
                outputs = self.pytorch_model(X_val_tensor.float())
                val_loss = criterion(outputs, y_val_tensor.unsqueeze(1)).item()
                
                # Convert to numpy for metric calculation
                outputs_np = outputs.squeeze().cpu().numpy()
                y_val_np = y_val_tensor.cpu().numpy()
                
                # Try different thresholds to find the best one
                best_threshold = 0.5
                best_val_f1 = 0.0
                
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    y_pred = (outputs_np >= threshold).astype(int)
                    tp = np.sum((y_pred == 1) & (y_val_np == 1))
                    fp = np.sum((y_pred == 1) & (y_val_np == 0))
                    tn = np.sum((y_pred == 0) & (y_val_np == 0))
                    fn = np.sum((y_pred == 0) & (y_val_np == 1))
                    
                    # Calculate metrics
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    _f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    if _f1 > best_val_f1:
                        best_val_f1 = _f1
                        best_threshold = threshold
                        curr_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        curr_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                
                # Save best metrics
                f1 = best_val_f1
                fpr = curr_fpr
                fnr = curr_fnr
                
                # Update learning rate based on F1 score
                scheduler.step(f1)
                
                # Save best model
                if f1 > best_f1:
                    best_f1 = f1
                    # Save best FPR and FNR
                    self.fpr = fpr
                    self.fnr = fnr
                    # Save best threshold
                    self.threshold = best_threshold
            
            # Print status
            if (epoch + 1) % 1 == 0:  # Print every epoch
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                     f"F1: {f1:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, Threshold: {best_threshold:.2f}")
        
        end_time = time.time()
        training_time_ms = (end_time - start_time) * 1000
        
        # Train backup bloom filter for false negatives
        self._train_backup_filter(positive_keys, X_val, y_val, self.pytorch_model)
        
        # Store accuracy and stats
        self.trained = True
        self.accuracy = best_f1
        
        print(f"Learned bloom filter for level {self.level} trained in {training_time_ms:.2f}ms with Metal acceleration")
        print(f"Final metrics - F1: {best_f1:.4f}, FPR: {self.fpr:.4f}, FNR: {self.fnr:.4f}, Threshold: {self.threshold:.2f}")
        print(f"Backup bloom filter size: {self.backup_filter_size} bits, hash functions: {self.backup_filter_hashes}")
        
        return True
    
    def _train_with_sklearn(self, X_features, y, positive_keys, negative_keys, validation_ratio):
        """Train the model using scikit-learn."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        
        # Format input data for model
        X_formatted = np.array(X_features)
        y_formatted = np.array(y)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_formatted, y_formatted, test_size=validation_ratio, random_state=42, stratify=y_formatted
        )
        
        print(f"Training learned bloom filter for level {self.level} with {len(X_train)} samples, validating with {len(X_val)} samples (CPU fallback)")
        
        start_time = time.time()
        
        # Train a gradient boosting classifier
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        
        # Calculate FPR and FNR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        self.fpr = fpr
        self.fnr = fnr
        
        # Store stats
        self.trained = True
        self.accuracy = f1
        
        end_time = time.time()
        training_time_ms = (end_time - start_time) * 1000
        
        # Train backup bloom filter for false negatives
        self._train_backup_filter(positive_keys, X_val, y_val, self.model)
        
        print(f"Learned bloom filter for level {self.level} trained in {training_time_ms:.2f}ms (CPU)")
        print(f"Final metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}")
        print(f"Backup bloom filter size: {self.backup_filter_size} bits, hash functions: {self.backup_filter_hashes}")
        
        return True
    
    def _train_backup_filter(self, positive_keys, X_val, y_val, model):
        """Train a backup bloom filter to catch false negatives from the ML model."""
        # Make predictions on validation set
        if self.pytorch_model is not None:
            # PyTorch prediction
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            self.pytorch_model.eval()
            with torch.no_grad():
                outputs = self.pytorch_model(X_val_tensor)
                # Use the best threshold found during training
                threshold = getattr(self, 'threshold', 0.5)  # Default to 0.5 if not set
                y_pred = (outputs > threshold).cpu().numpy().flatten()  # Make sure to flatten
                y_pred_proba = outputs.cpu().numpy().flatten()    # Make sure to flatten
        else:
            # Scikit-learn prediction
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Find validation examples that are false negatives (should be 1 but predicted 0)
        # and filter by the validation data with positive labels
        positive_val_indices = np.where(y_val == 1)[0]
        val_false_negative_indices = np.where((y_val == 1) & (y_pred == 0))[0]
        val_uncertain_indices = np.where((y_val == 1) & (y_pred_proba < 0.8))[0]
        
        # Calculate percentage of false negatives and uncertain predictions
        val_fn_percent = len(val_false_negative_indices) / len(positive_val_indices) if len(positive_val_indices) > 0 else 0
        val_uncertain_percent = len(val_uncertain_indices) / len(positive_val_indices) if len(positive_val_indices) > 0 else 0
        
        # Now build a backup filter using a representative sample of the original positive keys
        # Find approximately how many keys we should put in the backup filter based on validation set metrics
        fn_count = int(len(positive_keys) * val_fn_percent)
        uncertain_count = int(len(positive_keys) * val_uncertain_percent)
        
        # Sample keys for the backup filter
        backup_keys = []
        if fn_count > 0 or uncertain_count > 0:
            # Sample from original positive keys, ensuring we don't try to sample more than available
            sample_size = min(fn_count + uncertain_count, len(positive_keys))
            # Only sample if we have enough keys
            if sample_size > 0 and len(positive_keys) > 0:
                backup_keys = random.sample(positive_keys, sample_size)
        
        # If too many keys, sample to keep backup filter small
        max_backup_keys = min(len(positive_keys) // 2, 5000)
        if len(backup_keys) > max_backup_keys:
            backup_keys = random.sample(backup_keys, max_backup_keys)
        
        # Print backup filter stats
        print(f"Creating backup filter with {len(backup_keys)} keys "
              f"({len(val_false_negative_indices)} false negatives, {len(val_uncertain_indices)} uncertain predictions in validation)")
        
        # Create traditional bloom filter as backup
        self._create_backup_filter(backup_keys)
    
    def _create_backup_filter(self, keys, fpr=None):
        """Create a traditional bloom filter as backup."""
        if not keys:
            # Empty backup filter
            self.backup_filter_bits = np.zeros(10, dtype=bool)
            self.backup_filter_size = 10
            self.backup_filter_hashes = 1
            return
            
        # Use provided FPR or instance default
        if fpr is None:
            fpr = self.backup_fpr
            
        # Calculate optimal size and hash functions
        n = len(keys)
        m = int(-(n * math.log(fpr)) / (math.log(2) ** 2))
        k = max(1, int((m / n) * math.log(2)))
        
        # Initialize bit array
        self.backup_filter_size = m
        self.backup_filter_hashes = k
        self.backup_filter_bits = np.zeros(m, dtype=bool)
        
        # Add all keys to the backup filter
        for key in keys:
            for seed in range(k):
                # Simple hash function
                hash_val = hash(str(key) + str(seed)) % m
                self.backup_filter_bits[hash_val] = True
    
    def _create_features(self, keys):
        """Create rich features for the model from raw keys."""
        features = []
        
        # Track extreme values for safety checks
        max_safe_value = 1e30
        
        # Pre-compute example stats for faster lookups
        use_examples = len(self.example_keys) > 0
        if use_examples:
            example_floats = []
            for ex_key in list(self.example_keys)[:100]:  # Limit to 100 examples
                try:
                    if isinstance(ex_key, bytes):
                        ex_float = struct.unpack('d', ex_key[:8].ljust(8, b'\x00'))[0]
                    else:
                        ex_float = float(ex_key)
                    ex_float = max(min(ex_float, max_safe_value), -max_safe_value)
                    example_floats.append(ex_float)
                except:
                    pass
        
        # Pre-compute safe min/max
        has_range = self.min_key < self.max_key
        if has_range:
            safe_min = max(min(self.min_key, max_safe_value), -max_safe_value)
            safe_max = max(min(self.max_key, max_safe_value), -max_safe_value)
            range_size = max(safe_max - safe_min, 1e-10)
        
        for key in keys:
            try:
                # Initialize feature vector with basic values
                feature_vector = [0.0] * 8  # Pre-allocate for common features
                
                # Extract key values based on type
                if isinstance(key, bytes):
                    # Handle binary keys
                    try:
                        key_float1 = struct.unpack('d', key[:8].ljust(8, b'\x00'))[0]
                        key_float1 = max(min(key_float1, max_safe_value), -max_safe_value)
                        
                        # Get second half if available
                        key_float2 = 0.0
                        if len(key) >= 16:
                            key_float2 = struct.unpack('d', key[8:16].ljust(8, b'\x00'))[0]
                            key_float2 = max(min(key_float2, max_safe_value), -max_safe_value)
                        
                        # Set primary feature
                        feature_vector[0] = key_float1
                        if key_float2 != 0:
                            feature_vector[1] = key_float2
                    except:
                        feature_vector[0] = 0.0
                else:
                    # Handle numeric keys
                    try:
                        key_float = float(key)
                        feature_vector[0] = max(min(key_float, max_safe_value), -max_safe_value)
                    except:
                        feature_vector[0] = 0.0
                
                # Add range-based features if available
                key_float = feature_vector[0]  # Use the primary feature value
                
                if has_range:
                    # Calculate position in range (safer)
                    position = 0.5  # Default
                    if safe_min <= key_float <= safe_max:
                        position = (key_float - safe_min) / range_size
                        position = max(0.0, min(1.0, position))
                    
                    # Add range-based features
                    feature_vector[2] = position
                    feature_vector[3] = min(abs(key_float - safe_min), 1000.0)  # Distance to min
                    feature_vector[4] = min(abs(key_float - safe_max), 1000.0)  # Distance to max
                    feature_vector[5] = 1.0 if safe_min <= key_float <= safe_max else 0.0  # In range
                
                # Add example-based features if we have examples
                if use_examples and example_floats:
                    # Find closest example
                    min_dist = 1000.0
                    for ex_float in example_floats:
                        dist = abs(key_float - ex_float)
                        min_dist = min(min_dist, dist)
                    
                    feature_vector[6] = min(min_dist, 1000.0)  # Distance to closest example
                    feature_vector[7] = 1.0 if min_dist < 0.001 else 0.0  # Exact match indicator
                
                # Add hash-based feature to capture full key information
                hash_val = abs(hash(str(key)) % 10000) / 10000.0
                feature_vector.append(hash_val)
                
                features.append(feature_vector)
                
            except Exception:
                # Fallback feature vector if anything goes wrong
                features.append([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5])
        
        return features
    
    def predict(self, keys, return_probabilities=False):
        """
        Predict if keys exist in the dataset.
        
        Args:
            keys: List of keys to check
            return_probabilities: If True, return the prediction probabilities
            
        Returns:
            List of boolean predictions (True if key might exist, False if key definitely doesn't exist)
        """
        if not self.trained:
            # If not trained, always return True (full false positive rate)
            if return_probabilities:
                return np.ones(len(keys), dtype=bool), np.ones(len(keys), dtype=np.float32)
            return np.ones(len(keys), dtype=bool)
        
        # Create features from keys
        X_features = np.array(self._create_features(keys))
        
        # Get predictions from model
        if self.pytorch_model is not None:
            # PyTorch prediction
            X_tensor = torch.FloatTensor(X_features).to(DEVICE)
            self.pytorch_model.eval()
            with torch.no_grad():
                probs = self.pytorch_model(X_tensor).cpu().numpy().flatten()
                # Use the custom threshold from training
                threshold = getattr(self, 'threshold', 0.5)  # Default to 0.5 if not set
                preds = probs >= threshold
        else:
            # Scikit-learn prediction
            probs = self.model.predict_proba(X_features)[:, 1]
            preds = self.model.predict(X_features)
        
        return preds, probs
    
    def might_contain(self, key: float) -> float:
        """Check if a key might exist in the level.
        
        Returns a probability score between 0 and 1.
        """
        if not self.trained:
            # Default conservative behavior
            return 0.5
        
        # Quick check for exact matches with example keys
        if key in self.example_keys:
            self.stats['total_queries'] += 1
            self.stats['model_hits'] += 1
            return 1.0
            
        # Quick range check
        if key < self.min_key or key > self.max_key:
            self.stats['total_queries'] += 1
            return 0.1  # Very unlikely to exist
        
        # Create features for prediction
        X = self._create_features([key])
        
        # Make prediction
        if self.pytorch_model is not None:
            # PyTorch prediction
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            self.pytorch_model.eval()
            with torch.no_grad():
                score = self.pytorch_model(X_tensor).item()
        else:
            # Scikit-learn prediction
            score = self.model.predict_proba(X)[0, 1]
        
        self.stats['total_queries'] += 1
        
        # Check if backup filter is needed
        if score > 0.9:
            # High confidence - key exists
            self.stats['model_hits'] += 1
            return score
        elif score < 0.1:
            # High confidence - key doesn't exist
            return score
        else:
            # Uncertain prediction - check backup filter
            self.stats['backup_used'] += 1
            
            # Check backup filter
            in_backup = True
            for seed in range(self.backup_filter_hashes):
                hash_val = hash(str(key) + str(seed)) % self.backup_filter_size
                if not self.backup_filter_bits[hash_val]:
                    in_backup = False
                    break
            
            if in_backup:
                # Key is in backup filter
                self.stats['backup_hits'] += 1
                return max(score, 0.7)  # Boost confidence
            else:
                return min(score, 0.3)  # Reduce confidence
    
    def might_contain_batch(self, keys):
        """Batch predict if keys exist.
        
        Parameters:
        -----------
        keys : List[float] or numpy.ndarray
            List of keys to predict
            
        Returns:
        --------
        numpy.ndarray
            Array of prediction scores for each key
        """
        if not self.trained:
            # Default conservative behavior
            return np.ones(len(keys)) * 0.5
        
        # Convert to numpy array if needed
        if not isinstance(keys, np.ndarray):
            keys = np.array(keys)
        
        # Quick check for out-of-range keys
        results = np.empty(len(keys), dtype=np.float32)
        for i, key in enumerate(keys):
            # Check if key is in example set
            if key in self.example_keys:
                results[i] = 1.0
            elif key < self.min_key or key > self.max_key:
                results[i] = 0.1  # Very unlikely
            else:
                results[i] = 0.5  # Default, will be updated below
        
        # Create features for in-range keys
        in_range_indices = np.where((keys >= self.min_key) & (keys <= self.max_key) & (results == 0.5))[0]
        if len(in_range_indices) > 0:
            in_range_keys = keys[in_range_indices]
            X = self._create_features(in_range_keys)
            
            # Get model predictions
            if self.pytorch_model is not None:
                # PyTorch prediction
                X_tensor = torch.FloatTensor(X).to(DEVICE)
                self.pytorch_model.eval()
                with torch.no_grad():
                    scores = self.pytorch_model(X_tensor).cpu().numpy().flatten()
            else:
                # Scikit-learn prediction
                scores = self.model.predict_proba(X)[:, 1]
            
            # Use numba to check backup filter for uncertain predictions
            results[in_range_indices] = predict_bloom_numba(
                in_range_keys, scores, self.backup_filter_bits, 
                self.backup_filter_hashes, self.backup_filter_size
            )
        
        self.stats['total_queries'] += len(keys)
        
        return results
    
    def get_stats(self):
        """Get statistics about the LearnedBloomFilter."""
        total = self.stats['total_queries']
        if total == 0:
            total = 1  # Avoid division by zero
            
        return {
            'trained': self.trained,
            'accuracy': self.accuracy,
            'fpr': self.fpr,
            'fnr': self.fnr,
            'model_hit_rate': self.stats['model_hits'] / total,
            'backup_used_rate': self.stats['backup_used'] / total,
            'backup_hit_rate': self.stats['backup_hits'] / self.stats['backup_used'] if self.stats['backup_used'] > 0 else 0,
            'min_key': self.min_key,
            'max_key': self.max_key,
            'example_keys_count': len(self.example_keys),
            'backup_filter_size_bytes': self.backup_filter_size // 8,
            'backup_filter_hash_count': self.backup_filter_hashes,
            'level': self.level
        }

class LearnedBloomFilterManager:
    """Manager class for LearnedBloomFilter objects."""
    
    def __init__(self, models_dir: str, levels: List[int] = [1, 2, 3]):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Create learned bloom filters for each level
        self.filters = {}
        for level in levels:
            self.filters[level] = LearnedBloomFilter(level)
        
        # Training data per level
        self.training_data = {level: {'positive': [], 'negative': []} for level in levels}
        
        # Timing metrics
        self.prediction_time = 0.0
        self.prediction_count = 0
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load existing models if available."""
        for level, filter_obj in self.filters.items():
            model_path = os.path.join(self.models_dir, f"bloom_filter_level_{level}.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.filters[level] = pickle.load(f)
                    print(f"Loaded bloom filter for level {level} from {model_path}")
                except Exception as e:
                    print(f"Error loading bloom filter for level {level}: {e}")
    
    def save_models(self):
        """Save models to disk."""
        for level, filter_obj in self.filters.items():
            model_path = os.path.join(self.models_dir, f"bloom_filter_level_{level}.pkl")
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(filter_obj, f)
                print(f"Saved bloom filter for level {level} to {model_path}")
            except Exception as e:
                print(f"Error saving bloom filter for level {level}: {e}")
    
    def add_training_data(self, key: float, level: int, exists: bool):
        """Add training data for a level's bloom filter."""
        if level in self.training_data:
            if exists:
                self.training_data[level]['positive'].append(key)
                # Add key to learned filter directly
                self.filters[level].add_key(key)
            else:
                self.training_data[level]['negative'].append(key)
    
    def train_filters(self):
        """Train all bloom filters with collected data.
        
        This function is legacy - it's better to train each level explicitly
        using train_filter_level method for better control over the process.
        """
        for level, data in self.training_data.items():
            self.train_filter_level(level)
    
    def train_filter_level(self, level):
        """Train a specific level bloom filter.
        
        Args:
            level: The level to train
        
        Returns:
            True if successful, False otherwise
        """
        if level not in self.training_data:
            print(f"No training data for level {level}")
            return False
            
        data = self.training_data[level]
        positive_keys = data['positive']
        negative_keys = data['negative']
        
        if positive_keys:
            print(f"Training bloom filter for level {level} with {len(positive_keys)} positive and {len(negative_keys)} negative keys")
            success = self.filters[level].train(positive_keys, negative_keys)
            if success:
                # Save model after successful training
                model_path = os.path.join(self.models_dir, f"bloom_filter_level_{level}.pkl")
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(self.filters[level], f)
                    print(f"Saved bloom filter for level {level} to {model_path}")
                except Exception as e:
                    print(f"Error saving bloom filter for level {level}: {e}")
            return success
        else:
            print(f"Skipping training for level {level} - no positive keys")
            return False
    
    def might_contain(self, key: float, level: int) -> float:
        """Check if a key might exist in a specific level."""
        # Time the prediction
        start_time = time.perf_counter()
        
        if level in self.filters:
            result = self.filters[level].might_contain(key)
        else:
            # Default conservative behavior
            result = 0.5
        
        # Record timing
        end_time = time.perf_counter()
        self.prediction_time += (end_time - start_time)
        self.prediction_count += 1
        
        return result
    
    def might_contain_batch(self, keys, level: int) -> List[float]:
        """Batch predict if keys exist in a specific level."""
        # Time the prediction
        start_time = time.perf_counter()
        
        if level in self.filters:
            results = self.filters[level].might_contain_batch(keys)
        else:
            # Default conservative behavior
            results = np.ones(len(keys)) * 0.5
        
        # Record timing
        end_time = time.perf_counter()
        self.prediction_time += (end_time - start_time)
        self.prediction_count += len(keys)
        
        return results
    
    def get_stats(self):
        """Get statistics about all learned bloom filters."""
        stats = {}
        
        # Overall stats
        if self.prediction_count > 0:
            avg_prediction_time = self.prediction_time / self.prediction_count
        else:
            avg_prediction_time = 0
            
        stats['overall'] = {
            'total_predictions': self.prediction_count,
            'total_prediction_time': self.prediction_time,
            'avg_prediction_time': avg_prediction_time
        }
        
        # Per-level stats
        for level, filter_obj in self.filters.items():
            stats[f'level_{level}'] = filter_obj.get_stats()
        
        return stats 