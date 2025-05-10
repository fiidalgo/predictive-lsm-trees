# Predictive LSM Trees

This project implements **ML-enhanced LSM trees** for key-value storage that use machine learning models to optimize Bloom filters and fence pointers, significantly improving lookup performance.

## Overview

Log-Structured Merge (LSM) trees are widely used in NoSQL databases like Apache Cassandra, LevelDB, and RocksDB. This implementation enhances traditional LSM trees with machine learning models to predict:

1. Which levels might contain a key (replacing Bloom filters)
2. Which specific data page contains a key (replacing fence pointers)

These ML-based optimizations can lead to significant performance improvements, especially for lookup operations, by reducing unnecessary disk I/O.

## Features

- **Traditional LSM Tree**: Complete implementation with configurable buffer size, level size ratio, and false positive rates
- **ML-Enhanced LSM Tree**:
  - ML-based Bloom Filter replacements with improved false positive rates
  - ML-based Fence Pointer replacements with accurate page prediction
  - Training data collection during normal operation
  - Periodic retraining to adapt to changing workloads
- **Batch Operations**: Support for efficient batch gets/lookups using vectorized predictions
- **Performance Testing**: Comprehensive framework for benchmarking different implementations
- **Interactive Shell**: Command-line interface for experimenting with different implementations

## Project Structure

```
/
├── lsm/                    # Core LSM tree implementation
│   ├── __init__.py         # Package exports
│   ├── traditional.py      # Traditional LSM tree implementation
│   ├── learned.py          # Learned LSM tree with ML enhancements
│   ├── buffer.py           # Memory buffer implementation
│   ├── bloom.py            # Traditional Bloom filter implementation
│   ├── learned_bloom.py    # ML-enhanced Bloom filter implementation
│   ├── run.py              # Run/SSTable implementation
│   └── utils.py            # Utility functions
├── ml/                     # Machine learning models
│   ├── __init__.py         # Package exports
│   ├── models.py           # ML model implementations
│   └── learned_bloom.py    # Learned Bloom filter manager
├── data/                   # Data storage directory (created by load_data.py)
├── load_data.py            # Script to load initial data
├── train_classifier.py     # Script to train classifier models
├── train_learned.py        # Script to train learned Bloom filter models
├── test_performance.py     # Performance testing framework
├── interactive_lsm.py      # Interactive LSM tree shell
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/predictive-lsm-trees.git
   cd predictive-lsm-trees
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Load initial data:**
   ```bash
   python load_data.py
   ```

5. **Train the ML models:**
   ```bash
   python train_classifier.py
   python train_learned.py
   ```

## Usage

### Performance Testing

Run the performance test suite to compare the three implementations:

```bash
python test_performance.py
```

This will generate detailed performance metrics and visualization plots in the `plots/` directory.

### Interactive Shell

For interactive testing and experimentation:

```bash
python interactive_lsm.py
```

Available commands in the shell:
- `put key value` - Insert a key-value pair
- `get key` - Retrieve a value by key
- `range start_key end_key` - Get range of values
- `remove key` - Remove a key
- `flush` - Force flush buffer to disk
- `stats` - Show tree statistics
- `switch [traditional|classifier|learned]` - Switch implementations
- `reset` - Reset the current tree
- `exit` - Exit the shell

### Programmatic Usage

```python
from lsm import TraditionalLSMTree, LearnedLSMTree

# Create a traditional LSM tree
trad_tree = TraditionalLSMTree(
    data_dir="traditional_data",
    buffer_size=1*1024*1024,  # 1MB buffer
    size_ratio=10,            # Each level is 10x larger than previous
    base_fpr=0.01             # 1% base false positive rate for Bloom filters
)

# Create a learned LSM tree
learned_tree = LearnedLSMTree(
    data_dir="learned_data",
    buffer_size=1*1024*1024,  # 1MB buffer
    size_ratio=10,            # Each level is 10x larger than previous
    base_fpr=0.01,            # 1% base false positive rate as fallback
    train_every=4,            # Train models every 4 flushes
    validation_rate=0.05      # Use 5% of data for validation
)

# Basic operations
key = 42.0
value = "example_value"

# Put operation
learned_tree.put(key, value)

# Get operation
result = learned_tree.get(key)

# Range operation
range_results = learned_tree.range(40.0, 45.0)

# Remove operation
learned_tree.remove(key)
```

## Performance Results

The ML-enhanced LSM tree implementation shows significant improvements over the traditional implementation, particularly for lookup operations:

- **Level 0 Lookups**: Up to 2.3x speedup with classifier-based implementation
- **Sequential Lookups**: Up to 1.7x speedup across all levels
- **Random Lookups**: Up to 1.6x speedup across all levels

The performance gains come from reducing unnecessary disk I/O through more accurate predictions about where keys might be located.

## Implementation Details

### Traditional LSM Tree

- In-memory buffer (MemTable) with sorted key-value pairs
- On-disk immutable runs (SSTables) organized in levels
- Each level is ~10x larger than the previous level
- Bloom filters to avoid unnecessary disk reads
- Fence pointers for binary search within runs

### ML-Enhanced LSM Tree

- Trained ML models to replace Bloom filters
- Regression models to predict exact page location
- Batch prediction API for efficient vector operations
- Training data collection during normal operation
- Adaptive retraining based on workload changes

## Acknowledgments

This implementation is inspired by research on learned indexes and LSM tree optimizations, including:
- "The Case for Learned Index Structures" (Kraska et al., 2018)
- "SageDB: A Learned Database System" (Kraska et al., 2019)
- "Learning-based Frequency Estimation Algorithms" (Hsu et al., 2019)

## License

[MIT License](LICENSE) 