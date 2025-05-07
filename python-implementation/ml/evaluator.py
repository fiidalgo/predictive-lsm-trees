import os
import json
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime
from .models import LSMMLModels

class ModelEvaluator:
    """Utility class for evaluating ML model performance in LSM trees."""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models = LSMMLModels(models_dir)
        self.results_dir = os.path.join(models_dir, "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
    def evaluate_performance(self,
                           bloom_data: List[Tuple[float, bool]],
                           fence_data: List[Tuple[float, int, int]],
                           workload_name: str = "default") -> Dict[str, float]:
        """Evaluate model performance and save results."""
        # Evaluate Bloom filter model
        bloom_metrics = self._evaluate_bloom_model(bloom_data)
        
        # Evaluate fence pointer model
        fence_metrics = self._evaluate_fence_model(fence_data)
        
        # Combine metrics
        metrics = {
            'workload': workload_name,
            'timestamp': datetime.now().isoformat(),
            'bloom_filter': bloom_metrics,
            'fence_pointer': fence_metrics
        }
        
        # Save results
        self._save_results(metrics)
        
        return metrics
        
    def _evaluate_bloom_model(self, test_data: List[Tuple[float, bool]]) -> Dict[str, float]:
        """Evaluate Bloom filter model performance."""
        correct = 0
        false_positives = 0
        total_negatives = 0
        total_positives = 0
        
        for key, exists in test_data:
            pred = self.models.predict_bloom(key) > 0.5
            if pred == exists:
                correct += 1
            if not exists and pred:
                false_positives += 1
            if not exists:
                total_negatives += 1
            if exists:
                total_positives += 1
                
        return {
            'accuracy': correct / len(test_data),
            'false_positive_rate': false_positives / total_negatives if total_negatives > 0 else 0,
            'true_positive_rate': (total_positives - false_positives) / total_positives if total_positives > 0 else 0,
            'precision': (total_positives - false_positives) / (total_positives - false_positives + false_positives) if (total_positives - false_positives + false_positives) > 0 else 0
        }
        
    def _evaluate_fence_model(self, test_data: List[Tuple[float, int, int]]) -> Dict[str, float]:
        """Evaluate fence pointer model performance."""
        total_error = 0
        correct = 0
        max_error = 0
        
        for key, level, true_page in test_data:
            pred_page = self.models.predict_fence(key, level)
            error = abs(pred_page - true_page)
            total_error += error
            max_error = max(max_error, error)
            if error <= 1:  # Consider prediction correct if within 1 page
                correct += 1
                
        return {
            'accuracy': correct / len(test_data),
            'mean_absolute_error': total_error / len(test_data),
            'max_error': max_error,
            'error_std': np.std([abs(self.models.predict_fence(key, level) - true_page) 
                               for key, level, true_page in test_data])
        }
        
    def _save_results(self, metrics: Dict[str, float]) -> None:
        """Save evaluation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{metrics['workload']}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def load_results(self, workload_name: str = None) -> List[Dict[str, float]]:
        """Load evaluation results for a specific workload or all workloads."""
        results = []
        
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
                
            if workload_name and workload_name not in filename:
                continue
                
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'r') as f:
                results.append(json.load(f))
                
        return sorted(results, key=lambda x: x['timestamp'])
        
    def compare_workloads(self, workload_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare model performance across different workloads."""
        comparison = {}
        
        for workload in workload_names:
            results = self.load_results(workload)
            if not results:
                continue
                
            # Use the most recent result for each workload
            latest = results[-1]
            comparison[workload] = {
                'bloom_accuracy': latest['bloom_filter']['accuracy'],
                'bloom_fpr': latest['bloom_filter']['false_positive_rate'],
                'fence_accuracy': latest['fence_pointer']['accuracy'],
                'fence_mae': latest['fence_pointer']['mean_absolute_error']
            }
            
        return comparison 