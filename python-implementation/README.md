# Predictive LSM Trees: ML-Enhanced LSM Tree Implementation

This implementation provides a comparison between traditional LSM trees and ML-enhanced "learned" LSM trees for key-value storage.

## Overview

The learned LSM tree implementation uses machine learning models to replace traditional Bloom filters and fence pointers, enabling more efficient lookups by reducing unnecessary disk I/O. This implementation includes:

- Batch prediction APIs for improved ML inference performance
- Optional Numba JIT compilation for ultra-fast predictions
- Realistic disk I/O simulation with caching
- Performance comparison framework

## Project Structure

```
python-implementation/
├── lsm/
│   ├── __init__.py
│   ├── traditional.py    # Traditional LSM tree implementation
│   ├── learned.py        # Learned LSM tree with ML optimizations
│   ├── buffer.py         # Memory buffer implementation
│   ├── run.py            # SSTable/Run implementation
│   └── utils.py          # Utility functions
├── ml/
│   ├── __init__.py
│   └── models.py         # ML models (FastBloomFilter, FastFencePointer)
├── test_performance.py   # Performance comparison framework
└── interactive_lsm.py    # Interactive testing utility
```

## Key Features

- **ML-Enhanced Bloom Filters**: Uses ML models to predict if a key exists in a run, reducing unnecessary disk I/O
- **ML-Enhanced Fence Pointers**: Predicts the exact page where a key might be found
- **Batched Predictions**: Processes multiple keys at once to reduce Python function call overhead
- **Numba Optimization**: Optional JIT compilation for prediction functions
- **I/O Simulation**: Realistic disk access patterns with configurable latency and cache
- **Performance Metrics**: Comprehensive comparison between traditional and learned implementations

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from lsm import TraditionalLSMTree, LearnedLSMTree

# Create instances
trad_lsm = TraditionalLSMTree(data_dir="traditional_data")
learned_lsm = LearnedLSMTree(
    data_dir="learned_data",
    buffer_size=1024*1024,    # 1MB buffer
    size_ratio=10,            # Each level is 10x larger than previous
    base_fpr=0.01,            # 1% base false positive rate
    train_every=4,            # Train models every 4 flushes/compactions
    validation_rate=0.0       # Disable validation for maximum performance
)

# Basic operations
learned_lsm.put(key=42.0, value="value")
value = learned_lsm.get(key=42.0)

# Batch operations (for improved performance)
keys = [1.0, 2.0, 3.0, 4.0, 5.0]
values = learned_lsm.batch_get(keys)
```

## Performance Testing

Run the performance comparison test:
```bash
python test_performance.py
```

This will generate metrics comparing:
- Put/insert operations
- Get/lookup operations (with different access patterns)
- Range query performance
- ML prediction overhead
- Disk I/O statistics
- Cache performance

## Performance Optimizations

The learned LSM tree implementation includes several optimizations:
1. **Batched ML predictions**: Process multiple keys at once to amortize Python overhead
2. **Numba JIT compilation**: Compile prediction functions to machine code
3. **Cache-aware design**: Optimized for high cache hit rates
4. **ML-guided I/O**: Sequential reads for ML-predicted pages
5. **Aggressive filtering**: Lower bloom_thresh (0.3) to skip more unnecessary runs 