import os
import json
import sys
from bloom_filter_model import BloomFilterModel
from fence_pointer_model import FencePointerModel

class ModelPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        
        # Initialize models
        self.bloom_model = BloomFilterModel()
        self.fence_model = FencePointerModel()
        
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        self.bloom_model.load(self.model_dir)
        self.fence_model.load(self.model_dir)
    
    def predict_bloom_filter(self, key):
        """
        Predict if a key might be in the bloom filter
        Returns: float between 0 and 1 (probability)
        """
        return self.bloom_model.predict(key)
    
    def predict_fence_pointer(self, key, level):
        """
        Predict which page a key might be in
        Returns: int (page id)
        """
        return self.fence_model.predict(key, level)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    predictor = ModelPredictor()
    
    if command == "bloom":
        if len(sys.argv) != 3:
            print("Usage: python predict.py bloom <key>")
            sys.exit(1)
        key = float(sys.argv[2])
        prob = predictor.predict_bloom_filter(key)
        print(json.dumps({"probability": prob}))
    
    elif command == "fence":
        if len(sys.argv) != 4:
            print("Usage: python predict.py fence <key> <level>")
            sys.exit(1)
        key = float(sys.argv[2])
        level = int(sys.argv[3])
        page_id = predictor.predict_fence_pointer(key, level)
        print(json.dumps({"page_id": page_id}))
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main() 