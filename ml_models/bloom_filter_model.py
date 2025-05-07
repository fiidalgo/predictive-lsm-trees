import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from shared_memory import SharedMemoryInterface
import threading
import time

class BloomFilterModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super(BloomFilterModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.scaler = MinMaxScaler()
        self.batch_size = 1024  # Process data in batches for efficiency
        self.is_trained = False
        
    def forward(self, x):
        return self.network(x)
    
    def predict(self, key):
        # Normalize the key
        key_normalized = self.scaler.transform(np.array([[key]]))
        key_tensor = torch.FloatTensor(key_normalized)
        
        # Get prediction
        with torch.no_grad():
            prediction = self(key_tensor)
        return float(prediction[0])
    
    def train(self, keys, levels, page_ids):
        """Train the model with the given data"""
        # Check for empty data
        if not keys or len(keys) == 0:
            print("Warning: Received empty data for training")
            return 0.0
            
        # Convert to tensors
        keys_normalized = self.scaler.fit_transform(np.array(keys).reshape(-1, 1))
        keys_tensor = torch.FloatTensor(keys_normalized)
        labels_tensor = torch.ones(len(keys), 1)  # All keys are positive examples for bloom filter
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Process in batches
        total_loss = 0
        n_batches = (len(keys) + self.batch_size - 1) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(keys))
            
            batch_keys = keys_tensor[start_idx:end_idx]
            batch_labels = labels_tensor[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = self(batch_keys)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        self.is_trained = True
        return total_loss / n_batches
    
    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, 'bloom_filter_model.pt'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'bloom_filter_scaler.joblib'))
    
    def load(self, model_dir):
        self.load_state_dict(torch.load(os.path.join(model_dir, 'bloom_filter_model.pt')))
        self.scaler = joblib.load(os.path.join(model_dir, 'bloom_filter_scaler.joblib'))
        self.is_trained = True

class BloomFilterPredictor:
    def __init__(self, model_dir='models', shm_name='bloom_filter_shm', shm_size=1024*1024):
        self.model = BloomFilterModel()
        self.model_dir = model_dir
        self.shm = SharedMemoryInterface(shm_name, shm_size)
        self.load_model()
        
        # Start prediction thread
        self.running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.start()
    
    def load_model(self):
        model_path = os.path.join(self.model_dir, 'bloom_filter_model.pt')
        scaler_path = os.path.join(self.model_dir, 'bloom_filter_scaler.joblib')
        
        if os.path.exists(model_path):
            self.model.load(self.model_dir)
    
    def _prediction_loop(self):
        while self.running:
            try:
                data_type, data = self.shm.read_data()
                if data_type == 1:  # Bloom filter prediction request
                    key = data
                    prob = self.model.predict(key)
                    self.shm.write_response(1, prob)
            except Exception as e:
                print(f"Error in prediction loop: {e}")
            time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def train(self, keys, levels, page_ids):
        # For bloom filter, we use all keys as positive examples
        loss = self.model.train(
            keys=keys,
            levels=levels,
            page_ids=page_ids
        )
        self.model.save(self.model_dir)
        return loss
    
    def close(self):
        self.running = False
        self.prediction_thread.join()
        self.shm.close()

def main():
    predictor = BloomFilterPredictor()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        predictor.close()

if __name__ == '__main__':
    main() 