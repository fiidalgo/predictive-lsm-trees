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

class FencePointerModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_pages=100):
        super(FencePointerModel, self).__init__()
        self.num_pages = num_pages
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pages),
            nn.Softmax(dim=1)
        )
        self.scaler = MinMaxScaler()
        self.batch_size = 1024  # Process data in batches for efficiency
        self.is_trained = False
        
    def forward(self, x):
        return self.network(x)
    
    def predict(self, key, level):
        # Normalize the input
        input_normalized = self.scaler.transform(np.array([[key, level]]))
        input_tensor = torch.FloatTensor(input_normalized)
        
        # Get prediction
        with torch.no_grad():
            prediction = self(input_tensor)
            # Return the page with highest probability
            return int(torch.argmax(prediction[0]).item())
    
    def train(self, keys, levels, page_ids):
        """Train the model with the given data"""
        # Check for empty data
        if not keys or len(keys) == 0:
            print("Warning: Received empty data for training")
            return 0.0
            
        # Convert to tensors
        inputs = np.column_stack((keys, levels))
        inputs_normalized = self.scaler.fit_transform(inputs)
        inputs_tensor = torch.FloatTensor(inputs_normalized)
        labels_tensor = torch.LongTensor(page_ids)
        
        # Setup optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Process in batches
        total_loss = 0
        n_batches = (len(keys) + self.batch_size - 1) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(keys))
            
            batch_inputs = inputs_tensor[start_idx:end_idx]
            batch_labels = labels_tensor[start_idx:end_idx]
            
            optimizer.zero_grad()
            outputs = self(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        self.is_trained = True
        return total_loss / n_batches
    
    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, 'fence_pointer_model.pt'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'fence_pointer_scaler.joblib'))
    
    def load(self, model_dir):
        self.load_state_dict(torch.load(os.path.join(model_dir, 'fence_pointer_model.pt')))
        self.scaler = joblib.load(os.path.join(model_dir, 'fence_pointer_scaler.joblib'))
        self.is_trained = True

class FencePointerPredictor:
    def __init__(self, model_dir='models', shm_name='fence_pointer_shm', shm_size=1024*1024):
        self.model = FencePointerModel()
        self.model_dir = model_dir
        self.shm = SharedMemoryInterface(shm_name, shm_size)
        self.load_model()
        
        # Start prediction thread
        self.running = True
        self.prediction_thread = threading.Thread(target=self._prediction_loop)
        self.prediction_thread.start()
    
    def load_model(self):
        model_path = os.path.join(self.model_dir, 'fence_pointer_model.pt')
        scaler_path = os.path.join(self.model_dir, 'fence_pointer_scaler.joblib')
        
        if os.path.exists(model_path):
            self.model.load(self.model_dir)
    
    def _prediction_loop(self):
        while self.running:
            try:
                data_type, data = self.shm.read_data()
                if data_type == 2:  # Fence pointer prediction request
                    key, level = data
                    page_id = self.model.predict(key, level)
                    self.shm.write_response(2, page_id)
            except Exception as e:
                print(f"Error in prediction loop: {e}")
            time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def train(self, keys, levels, page_ids):
        loss = self.model.train(
            keys=keys,
            levels=levels,
            page_ids=page_ids,
            learning_rate=0.001
        )
        self.model.save(self.model_dir)
        return loss
    
    def close(self):
        self.running = False
        self.prediction_thread.join()
        self.shm.close()

def main():
    predictor = FencePointerPredictor()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        predictor.close()

if __name__ == '__main__':
    main() 