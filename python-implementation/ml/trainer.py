import os
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from .models import LSMMLModels

class ModelTrainer:
    """Utility class for training and managing ML models for LSM trees."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = LSMMLModels(models_dir)
        
    def train_models(self, 
                    bloom_data: List[Tuple[float, bool]],
                    fence_data: List[Tuple[float, int, int]]) -> Dict[str, float]:
        """Train both Bloom filter and fence pointer models."""
        # Split data into training and validation sets
        bloom_train, bloom_val = train_test_split(bloom_data, test_size=0.2, random_state=42)
        fence_train, fence_val = train_test_split(fence_data, test_size=0.2, random_state=42)
        
        # Train models
        bloom_metrics = self._train_bloom_model(bloom_train, bloom_val)
        fence_metrics = self._train_fence_model(fence_train, fence_val)
        
        # Save models
        self.models.save_models()
        
        return {
            'bloom_accuracy': bloom_metrics['accuracy'],
            'bloom_fpr': bloom_metrics['false_positive_rate'],
            'fence_accuracy': fence_metrics['accuracy'],
            'fence_mae': fence_metrics['mean_absolute_error']
        }
        
    def _train_bloom_model(self, 
                          train_data: List[Tuple[float, bool]],
                          val_data: List[Tuple[float, bool]]) -> Dict[str, float]:
        """Train the Bloom filter model."""
        # Add training data
        for key, exists in train_data:
            self.models.add_bloom_training_data(key, exists)
            
        # Train model
        self.models.train_bloom_model()
        
        # Evaluate on validation set
        correct = 0
        false_positives = 0
        total_negatives = 0
        
        for key, exists in val_data:
            pred = self.models.predict_bloom(key) > 0.5
            if pred == exists:
                correct += 1
            if not exists and pred:
                false_positives += 1
            if not exists:
                total_negatives += 1
                
        return {
            'accuracy': correct / len(val_data),
            'false_positive_rate': false_positives / total_negatives if total_negatives > 0 else 0
        }
        
    def _train_fence_model(self,
                          train_data: List[Tuple[float, int, int]],
                          val_data: List[Tuple[float, int, int]]) -> Dict[str, float]:
        """Train the fence pointer model."""
        # Add training data
        for key, level, page in train_data:
            self.models.add_fence_training_data(key, level, page)
            
        # Train model
        self.models.train_fence_model()
        
        # Evaluate on validation set
        total_error = 0
        correct = 0
        
        for key, level, true_page in val_data:
            pred_page = self.models.predict_fence(key, level)
            total_error += abs(pred_page - true_page)
            if abs(pred_page - true_page) <= 1:  # Consider prediction correct if within 1 page
                correct += 1
                
        return {
            'accuracy': correct / len(val_data),
            'mean_absolute_error': total_error / len(val_data)
        }
        
    def evaluate_models(self,
                       bloom_data: List[Tuple[float, bool]],
                       fence_data: List[Tuple[float, int, int]]) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        bloom_metrics = self._evaluate_bloom_model(bloom_data)
        fence_metrics = self._evaluate_fence_model(fence_data)
        
        return {
            'bloom_accuracy': bloom_metrics['accuracy'],
            'bloom_fpr': bloom_metrics['false_positive_rate'],
            'fence_accuracy': fence_metrics['accuracy'],
            'fence_mae': fence_metrics['mean_absolute_error']
        }
        
    def _evaluate_bloom_model(self, test_data: List[Tuple[float, bool]]) -> Dict[str, float]:
        """Evaluate Bloom filter model performance."""
        correct = 0
        false_positives = 0
        total_negatives = 0
        
        for key, exists in test_data:
            pred = self.models.predict_bloom(key) > 0.5
            if pred == exists:
                correct += 1
            if not exists and pred:
                false_positives += 1
            if not exists:
                total_negatives += 1
                
        return {
            'accuracy': correct / len(test_data),
            'false_positive_rate': false_positives / total_negatives if total_negatives > 0 else 0
        }
        
    def _evaluate_fence_model(self, test_data: List[Tuple[float, int, int]]) -> Dict[str, float]:
        """Evaluate fence pointer model performance."""
        total_error = 0
        correct = 0
        
        for key, level, true_page in test_data:
            pred_page = self.models.predict_fence(key, level)
            total_error += abs(pred_page - true_page)
            if abs(pred_page - true_page) <= 1:  # Consider prediction correct if within 1 page
                correct += 1
                
        return {
            'accuracy': correct / len(test_data),
            'mean_absolute_error': total_error / len(test_data)
        } 