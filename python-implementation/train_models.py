import os
import time
import random
import logging
import pickle
import struct
import numpy as np
from lsm import TraditionalLSMTree, LearnedLSMTree, Constants

# Configure logging to show info
logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    """Handles the training of ML models for the LSM tree."""
    
    def __init__(self, data_dir="data", model_dir="learned_test_data"):
        print("Initializing model trainer...")
        
        # Data directories
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Constants for buffer and level sizes
        self.BUFFER_SIZE_BYTES = 1 * 1024 * 1024  # 1MB
        self.LEVEL_SIZE_RATIO = 10  # Each level is 10x larger than previous
        
        # Initialize trees
        self.traditional_tree = TraditionalLSMTree(
            data_dir=self.data_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01
        )
        
        self.learned_tree = LearnedLSMTree(
            data_dir=self.model_dir,
            buffer_size=self.BUFFER_SIZE_BYTES,
            size_ratio=self.LEVEL_SIZE_RATIO,
            base_fpr=0.01,
            train_every=4,
            validation_rate=0.05
        )
        
        # Copy structure if needed
        if not os.path.exists(os.path.join(self.model_dir, "ml_models")):
            os.makedirs(os.path.join(self.model_dir, "ml_models"), exist_ok=True)
            
    def train_models(self):
        """Train ML models on the loaded data."""
        start_time = time.time()
        
        # Get all runs from the traditional tree
        trad_stats = self.traditional_tree.get_stats()
        
        # Get total number of runs across all levels
        run_count = 0
        for level_stats in trad_stats.get('levels', []):
            run_count += level_stats.get('run_count', 0)
        
        print(f"Training on data from {run_count} runs across all levels...")
        
        # Collect training data from traditional tree runs
        total_keys_added = 0
        total_negative_samples = 0
        
        for level_idx, level in enumerate(self.traditional_tree.levels):
            if not level.runs:
                print(f"Level {level_idx} has no runs, skipping")
                continue
                
            print(f"Training models for Level {level_idx} with {len(level.runs)} runs...")
            
            # Train models for each run
            for run in level.runs:
                try:
                    # Load run data
                    with open(run.file_path, 'rb') as f:
                        data = pickle.load(f)
                        pairs = data.get('pairs', [])
                        
                        if not pairs:
                            print(f"  Run {run.file_path} has no data, skipping")
                            continue
                            
                        print(f"  Processing run with {len(pairs)} entries")
                        
                        # Sample data if needed to speed up training
                        max_samples = 100000  # Increased from 10,000 to 100,000 (10x more data)
                        if len(pairs) > max_samples:
                            print(f"  Sampling {max_samples} records from {len(pairs)} for faster training")
                            pairs = random.sample(pairs, max_samples)
                        else:
                            # For smaller runs, duplicate data to have more training examples
                            if len(pairs) < 1000:
                                print(f"  Small run detected, duplicating data to increase training samples")
                                orig_pairs = pairs.copy()
                                while len(pairs) < 1000:
                                    # Add slight noise to keys to avoid exact duplicates
                                    noise_pairs = []
                                    for key, value in orig_pairs:
                                        if isinstance(key, bytes):
                                            key_float = struct.unpack('!d', key[:8])[0]
                                            # Add tiny noise
                                            key_float += random.uniform(-0.001, 0.001)
                                            key = struct.pack('!d', key_float).ljust(len(key), b'\x00')
                                        else:
                                            key += random.uniform(-0.001, 0.001)
                                        noise_pairs.append((key, value))
                                    pairs.extend(noise_pairs)
                        
                        # Get key set for generating negative samples
                        key_set = set()
                        
                        # Calculate page boundaries
                        records_per_page = run.page_size // Constants.ENTRY_SIZE_BYTES
                        
                        # Get min/max key range
                        min_key = run.min_key
                        max_key = run.max_key
                        
                        if isinstance(min_key, bytes):
                            min_key = struct.unpack('!d', min_key[:8])[0]
                        if isinstance(max_key, bytes):
                            max_key = struct.unpack('!d', max_key[:8])[0]
                        
                        # Process each key-value pair
                        keys_added = 0
                        positive_examples = []
                        
                        for i, (key, _) in enumerate(pairs):
                            # Convert key from bytes to float if needed
                            if isinstance(key, bytes):
                                key = struct.unpack('!d', key[:8])[0]
                                
                            # Add to key set and positive examples
                            key_set.add(key)
                            positive_examples.append(key)
                            
                            # Calculate page id
                            page_id = i // records_per_page
                            
                            # Add to fence pointer training immediately
                            self.learned_tree.ml_models.add_fence_training_data(key, level_idx, page_id)
                            keys_added += 1
                            
                        total_keys_added += keys_added
                        print(f"  Added {keys_added} positive training samples")
                        
                        # Generate negative samples for bloom filter training (equal to positive)
                        negative_samples = 0
                        if min_key is not None and max_key is not None and min_key < max_key:
                            key_range = max_key - min_key
                            num_negative_samples = len(positive_examples)  # Same as positive examples
                            
                            print(f"  Adding {num_negative_samples} negative samples for class balance")
                            
                            # Now add all positive examples for bloom filter - with their LEVEL information
                            # Instead of yes/no labels, we're assigning level numbers
                            for key in positive_examples:
                                # Add this key with its level_idx as the target
                                # This trains the model to predict WHICH LEVEL contains the key
                                self.learned_tree.ml_models.add_bloom_training_data(key, level_idx)
                            
                            # For negative examples (keys that don't exist), use level -1
                            for _ in range(num_negative_samples):
                                while True:
                                    # Generate a key outside the range or within gaps
                                    fake_key = random.uniform(min_key - key_range, max_key + key_range)
                                    # Ensure it's not an actual key
                                    if fake_key not in key_set:
                                        break
                                        
                                # Add negative bloom filter sample with level -1 (meaning "not in any level")
                                self.learned_tree.ml_models.add_bloom_training_data(fake_key, -1)
                                negative_samples += 1
                            
                            total_negative_samples += negative_samples
                            print(f"  Successfully added {negative_samples} negative samples")
                            print(f"  Total bloom filter samples: {len(positive_examples)} positive, {negative_samples} negative")
                    
                except Exception as e:
                    print(f"  Error processing run: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\nTotal training data collected:")
        print(f"  Positive samples: {total_keys_added}")
        print(f"  Negative samples: {total_negative_samples}")
        print(f"  Bloom filter training data ratio: {total_negative_samples/total_keys_added:.2f}:1 (negative:positive)")
        
        if total_keys_added == 0:
            print("ERROR: No training data collected, models cannot be trained!")
            return
            
        # Train the models with all the collected data
        print("\nTraining ML models with collected data...")
        
        # Check if there's any training data
        bloom_data_size = len(self.learned_tree.ml_models.bloom_data) if hasattr(self.learned_tree.ml_models, 'bloom_data') else 0
        fence_data_size = len(self.learned_tree.ml_models.fence_data) if hasattr(self.learned_tree.ml_models, 'fence_data') else 0
        print(f"Bloom filter training data: {bloom_data_size} samples")
        print(f"Fence pointer training data: {fence_data_size} samples")
        
        try:
            # Time the bloom filter training
            print("\nTraining Bloom Filter model...")
            bloom_start = time.time()
            self.learned_tree.ml_models.train_bloom_model()
            bloom_time = time.time() - bloom_start
            print(f"Bloom filter model training completed in {bloom_time:.2f} seconds")
            
            # Time the fence pointer training
            print("\nTraining Fence Pointer model...")
            fence_start = time.time()
            self.learned_tree.ml_models.train_fence_model()
            fence_time = time.time() - fence_start
            print(f"Fence pointer model training completed in {fence_time:.2f} seconds")
            
            # Update statistics
            new_bloom_acc = self.learned_tree.ml_models.get_bloom_accuracy()
            new_fence_acc = self.learned_tree.ml_models.get_fence_accuracy()
            
            # Save final accuracy values
            self.learned_tree.training_stats['bloom']['last_accuracy'] = new_bloom_acc
            self.learned_tree.training_stats['fence']['last_accuracy'] = new_fence_acc
            
            print(f"\nTraining results:")
            print(f"  Bloom Filter Accuracy: {new_bloom_acc:.2%}")
            print(f"  Fence Pointer Accuracy: {new_fence_acc:.2%}")
            
            # Save models explicitly
            self.learned_tree.ml_models.save_models()
            print(f"\nModels saved to {os.path.join(self.model_dir, 'ml_models')}")
            
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        print(f"\nML model training complete in {elapsed:.2f} seconds")
        print(f"Models are ready for use in performance testing!")

def main():
    """Main function to run model training."""
    print("Starting LSM tree model training...")
    
    # Create trainer and train models
    trainer = ModelTrainer()
    trainer.train_models()
    
    print("Training complete. You can now run test_performance.py without retraining.")
    
if __name__ == "__main__":
    main() 