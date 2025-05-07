import os
import sys
import time
import numpy as np
from shared_memory import SharedMemoryInterface
from bloom_filter_model import BloomFilterModel
from fence_pointer_model import FencePointerModel

def train_models():
    # Initialize shared memory interface
    shm = SharedMemoryInterface("lsm_ml_shm", 1024 * 1024)  # 1MB buffer
    
    # Initialize models
    bloom_model = BloomFilterModel()
    fence_model = FencePointerModel()
    
    print("ML training service started")
    
    while True:
        try:
            # Read data from shared memory
            data_type, data = shm.read_data()
            
            if data_type == 0:  # Training data
                keys, levels, page_ids = data
                
                # Skip if no data
                if not keys or len(keys) == 0:
                    print("Received empty training data, skipping")
                    continue
                
                # Convert to numpy arrays
                keys = np.array(keys)
                levels = np.array(levels)
                page_ids = np.array(page_ids)
                
                print(f"Training models with {len(keys)} samples")
                
                # Train models
                try:
                    bloom_model.train(keys, levels, page_ids)
                    fence_model.train(keys, levels, page_ids)
                    
                    # Signal that models are ready
                    shm.write_response(3, None)  # Type 3 indicates models are ready
                    print("Models trained and ready")
                except Exception as e:
                    print(f"Error during model training: {e}")
                    continue
                
            elif data_type == 1:  # Bloom filter prediction request
                key = data
                if bloom_model.is_trained:
                    prob = bloom_model.predict(key)
                    shm.write_response(1, prob)
                else:
                    shm.write_response(1, 0.5)  # Default probability if not trained
                
            elif data_type == 2:  # Fence pointer prediction request
                key, level = data
                if fence_model.is_trained:
                    page_id = fence_model.predict(key, level)
                    shm.write_response(2, page_id)
                else:
                    shm.write_response(2, -1)  # Invalid page ID if not trained
                    
        except Exception as e:
            print(f"Error in training service: {e}")
            time.sleep(1)  # Prevent tight loop on error

if __name__ == "__main__":
    train_models() 