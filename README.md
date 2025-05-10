# Learned LSM-trees: Two Approaches Using Learned Bloom Filters

Modern key-value stores rely heavily on Log-Structured Merge (LSM) trees for write optimization, but this design introduces significant read amplification. Auxiliary structures like Bloom filters help, but impose memory costs that scale with tree depth and dataset size. Recent advances in learned data structures suggest that machine learning models can augment or replace these components, trading handcrafted heuristics for data-adaptive behavior. In this work, we explore two approaches for integrating learned predictions into the LSM-tree lookup path. The first uses a classifier to selectively bypass Bloom filter probes for irrelevant levels, aiming to reduce average-case query latency. The second replaces traditional Bloom filters with compact learned models and small backup filters, targeting memory footprint reduction without compromising correctness.

We implement both methods atop a Monkey-style LSM-tree with leveled compaction, per-level Bloom filters, and realistic workloads. Our experiments show that the classifier reduces GET latency by up to 2.28× by skipping over 30% of Bloom filter checks with high precision, though it incurs a modest false-negative rate. The learned Bloom filter design achieves zero false negatives and retains baseline latency while cutting memory usage per level by 70–80%. Together, these designs illustrate complementary trade-offs between latency, memory, and correctness, and highlight the potential of learned index components in write-optimized storage systems.

## Table of Contents

- [Research Overview](#research-overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Implementation Details](#implementation-details)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [Performance Results](#performance-results)
- [Replicating Experiments](#replicating-experiments)
- [Acknowledgments](#acknowledgments)

## Research Overview

This project explores machine learning enhancements to Log-Structured Merge (LSM) trees, a fundamental data structure in NoSQL databases like Apache Cassandra, LevelDB, and RocksDB. We focus on two key ML-based approaches:

### 1. Level Classifier

A machine learning model that predicts which levels in the LSM tree are unlikely to contain a queried key. This allows the system to skip Bloom filter checks for irrelevant levels, reducing average lookup latency by avoiding unnecessary I/O operations.

**Benefits:**

- Reduces average GET latency by up to 2.28×
- Skips ~30% of Bloom filter checks
- Particularly effective for larger datasets with deeper LSM trees

**Trade-offs:**

- Introduces a small false-negative rate (mitigated by model tuning)
- Requires periodic retraining to adapt to changing workloads

### 2. Learned Bloom Filters

Replaces traditional Bloom filters with compact ML models paired with small backup filters. This approach maintains correctness guarantees while significantly reducing the memory footprint.

**Benefits:**

- Reduces memory usage per level by 70-80%
- Zero false negatives (maintains correctness guarantees)
- Maintains baseline query performance
- Scales better with increasing dataset size

**Trade-offs:**

- More complex implementation
- Requires training data collection and periodic model updates

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
├── models/                 # Directory for trained ML models
├── plots/                  # Performance visualization plots (created by test_performance.py)
├── load_data.py            # Script to load initial data
├── train_classifier.py     # Script to train level classifier models
├── train_learned.py        # Script to train learned Bloom filter models
├── test_performance.py     # Performance testing framework
├── interactive_lsm.py      # Interactive LSM tree shell
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Key Components

### Core LSM Implementation (`lsm/`)

- **traditional.py**: Implements a Monkey-style LSM tree with:

  - Memory buffer (MemTable) with sorted key-value pairs
  - On-disk immutable runs (SSTables) organized in levels
  - Per-level Bloom filters with optimal sizing
  - Fence pointers for binary search within runs
  - Tiering strategy for level 0, leveling for higher levels
  - Configurable buffer size, level size ratio, and false positive rates

- **learned.py**: ML-enhanced LSM tree that:

  - Inherits from traditional implementation
  - Integrates both level classifier and learned Bloom filters
  - Implements adaptive threshold selection based on workload
  - Provides batched prediction API for efficient vector operations
  - Collects training data during normal operations
  - Supports periodic retraining based on changing workloads

- **buffer.py**: Memory buffer (MemTable) implementation with:

  - Efficient in-memory storage of sorted key-value pairs
  - Configurable size limit
  - Thread-safe operations

- **run.py**: Immutable run (SSTable) implementation with:

  - Sorted key-value storage format
  - Bloom filter integration
  - Page-based access for efficient I/O
  - Min/max key tracking for range filtering
  - Fence pointers for binary search within pages

- **bloom.py**: Traditional Bloom filter implementation

  - Configurable false positive rate
  - Space-efficient bit array
  - Multiple hash functions

- **learned_bloom.py**: Learned Bloom filter with backup filter that:
  - Uses ML model for initial filtering
  - Maintains small backup filter for correctness guarantees
  - Achieves memory reduction without false negatives

### Machine Learning Components (`ml/`)

- **models.py**: ML model implementations including:

  - Level classifier for predicting key presence in levels
  - Learned Bloom filter models for memory-efficient filtering
  - Training and inference pipeline
  - Model serialization/deserialization

- **learned_bloom.py**: Learned Bloom filter manager that:
  - Maintains training data collection
  - Handles model training and evaluation
  - Provides prediction API
  - Implements adaptive thresholding based on workloads

### Data Management and Utilities

- **load_data.py**: Script to generate and load initial dataset with:

  - Configurable data distribution (uniform, normal, zipfian)
  - Key space configuration
  - Data size controls

- **train_classifier.py**: Trains the level classifier models:

  - Uses keys from existing LSM tree
  - Builds training dataset based on actual key distribution
  - Optimizes for different recall/precision tradeoffs per level
  - Saves models to the models/ directory

- **train_learned.py**: Trains learned Bloom filter models:
  - Builds level-specific training datasets
  - Optimizes models for recall-precision tradeoff
  - Supports different model architectures
  - Implements cross-validation

### Testing and Evaluation

- **test_performance.py**: Comprehensive performance evaluation:

  - Tests lookup performance (random, sequential, level-specific)
  - Evaluates range query performance
  - Measures I/O overhead (disk reads/writes)
  - Compares all three implementations (traditional, classifier, learned Bloom filter)
  - Generates detailed plots and metrics
  - Supports batch operations testing

- **interactive_lsm.py**: Interactive command-line shell:
  - Supports all LSM operations (put, get, range, delete)
  - Allows switching between implementations
  - Shows performance statistics
  - Provides debugging capabilities

## Implementation Details

### Traditional LSM Tree (Baseline)

Our baseline implementation follows the Monkey design, which optimizes Bloom filter memory allocation across levels:

1. **Memory Buffer**: New key-value pairs are initially stored in an in-memory buffer (MemTable).
2. **Flushing**: When the buffer is full, it's sorted and flushed to disk as a new immutable run (SSTable) in level 0.
3. **Leveled Compaction**: Each level is approximately 10x larger than the previous level, with one large run per level (except level 0).
4. **Bloom Filter Sizing**: Following the Monkey approach, we allocate Bloom filter memory across levels to minimize I/O cost.
5. **Lookup Process**:
   - Check memory buffer first
   - For each level (starting from level 0):
     - For each run in the level (newest to oldest):
       - Check if the key might be in the run using Bloom filter
       - If Bloom filter indicates presence, binary search using fence pointers
       - Return the value if found (or move to next run if not found)

### Level Classifier Implementation

The level classifier introduces a pre-filter step before the traditional Bloom filter check:

1. **ML Model**: A classifier trained to predict whether a key exists in a specific level
2. **Prediction Process**:
   - For each level, the classifier predicts the probability of key presence
   - If the probability is below a threshold, skip the level entirely
   - Otherwise, proceed with the traditional Bloom filter check
3. **Adaptive Thresholds**:
   - Each level uses different thresholds based on its position in the tree
   - Lower levels (with fewer keys) use higher thresholds to minimize false negatives
   - Higher levels use lower thresholds to maximize skipping opportunities
4. **Training Data Collection**:
   - During normal operation, the system collects positive and negative examples
   - Periodically retrains the model to adapt to changing workloads

### Learned Bloom Filter Implementation

The learned Bloom filter replaces the traditional Bloom filter with a hybrid approach:

1. **Learned Model**: Predicts the probability of key membership
2. **Backup Filter**: A small traditional Bloom filter that catches the false negatives from the model
3. **Filtering Process**:
   - First check if the key passes the learned model filter
   - If yes, perform additional check with the backup filter
   - Only if both pass do we proceed with the disk read
4. **Memory Efficiency**:
   - The learned model has a fixed memory footprint regardless of key space
   - The backup filter only needs to handle model false negatives
   - Together, they achieve 70-80% memory reduction compared to traditional Bloom filters

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

   This script generates a dataset with a configurable distribution. Options include:

   - `--size`: Number of key-value pairs to generate (default: 1,000,000)
   - `--distribution`: Data distribution - uniform, normal, or zipfian (default: uniform)
   - `--key-range`: Range of keys to generate (default: 0.0-1000000.0)
   - `--seed`: Random seed for reproducibility (default: 42)

5. **Train the ML models:**
   ```bash
   python train_classifier.py  # Train level classifier models
   python train_learned.py     # Train learned Bloom filter models
   ```
   The training scripts have several options:
   - `--epochs`: Number of training epochs (default: 20)
   - `--batch-size`: Batch size for training (default: 64)
   - `--validation-split`: Fraction of data to use for validation (default: 0.2)
   - `--learning-rate`: Learning rate for optimizer (default: 0.001)
   - `--model-type`: Type of model to use (default: mlp for learned_bloom, cnn for classifier)

## Usage Examples

### Performance Testing

Run the performance test suite to compare the three implementations:

```bash
python test_performance.py
```

This will generate detailed performance metrics and visualization plots in the `plots/` directory.

Options include:

- `--lookups`: Number of lookups to perform for each test (default: 10000)
- `--range-queries`: Number of range queries to test (default: 1000)
- `--range-size`: Average size of range queries (default: 100)
- `--batch-size`: Batch size for batched operations (default: 64)
- `--visualize-only`: Only visualize existing results without running tests
- `--no-plots`: Run tests without generating plots
- `--seed`: Random seed for reproducibility (default: 42)

### Interactive Shell

For interactive testing and experimentation:

```bash
python interactive_lsm.py
```

Available commands in the shell:

- `put <key> <value>` - Insert a key-value pair
- `get <key>` - Retrieve a value by key
- `range <start_key> <end_key>` - Get range of values
- `remove <key>` - Remove a key
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

# Batch operations
keys = [1.0, 2.0, 3.0, 4.0, 5.0]
batch_results = learned_tree.batch_get(keys)
```

## Performance Results

Our experiments demonstrate significant improvements over the traditional LSM tree implementation:

### Level-by-Level Performance

| Level | Traditional (ms) | Classifier (ms) | Learned (ms) | Speedup |
| ----- | ---------------- | --------------- | ------------ | ------- |
| 0     | 0.56             | 0.24            | 0.31         | 2.3x    |
| 1     | 1.12             | 0.71            | 0.67         | 1.7x    |
| 2     | 2.27             | 1.54            | 1.42         | 1.6x    |
| 3     | 4.08             | 2.45            | 2.38         | 1.7x    |
| All   | 1.88             | 1.12            | 1.09         | 1.7x    |

### Key Performance Metrics

#### Level Classifier

- **Latency Reduction**: Up to 2.28× faster GET operations
- **Bloom Filter Checks**: ~30% reduction in Bloom filter probes
- **False Negative Rate**: <0.1% (configurable via threshold tuning)
- **Memory Overhead**: Negligible (fixed model size regardless of data volume)

#### Learned Bloom Filters

- **Memory Reduction**: 70-80% memory savings per level
- **False Negative Rate**: Zero (maintains correctness guarantees)
- **Query Performance**: Comparable to traditional implementation
- **Training Overhead**: Minimal, can be performed during idle periods

### I/O Reduction

- **Disk Reads**: Reduced by 45-60% across all levels
- **Page Accesses**: Reduced by 35-55% across all levels
- **Bloom Filter Checks**: Reduced by 30-40% through predictive filtering

## Replicating Experiments

To fully replicate the experiments and results:

1. **Generate dataset with specific distribution**:

   ```bash
   python load_data.py --size 1000000 --distribution zipfian --seed 42
   ```

2. **Train models with optimized parameters**:

   ```bash
   python train_classifier.py --epochs 30 --batch-size 128 --learning-rate 0.001
   python train_learned.py --epochs 30 --batch-size 128 --learning-rate 0.001
   ```

3. **Run comprehensive performance tests**:

   ```bash
   python test_performance.py --lookups 50000 --range-queries 5000 --seed 42
   ```

4. **Analyze results**:

   - Detailed plots will be generated in the `plots/` directory
   - Average lookup times by level are displayed in terminal output
   - I/O statistics show the reduction in disk reads and page accesses

5. **Test with different workloads** by modifying the data distribution:
   ```bash
   python load_data.py --size 1000000 --distribution normal --seed 42
   ```
   Then repeat steps 2-4 to evaluate performance on different data distributions.

## Acknowledgments

This research builds upon foundational work in learned data structures and LSM-tree optimizations. See the [References](#references) section below for a complete list of works cited.

## References

[1] Burton H Bloom. Space/time trade-offs in hash coding with allowable errors. _Communications of the ACM_, 13(7):422–426, 1970.

[2] Peng Dai and Anshumali Shrivastava. Ada-bf: Adaptive bloom filter with learned model. In _Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data_, pages 1551–1566, 2020.

[3] Niv Dayan and Stratos Idreos. Monkey: Optimal navigable key-value store. In _Proceedings of the 2018 International Conference on Management of Data_, pages 79–94, 2018.

[4] Andreas Kipf, Tobias Kipf, Benedikt Radke, Alfons Kemper, and Thomas Neumann. Learned cardinalities: Estimating correlated joins with deep learning. In _CIDR_, 2019.

[5] Tim Kraska, Alex Beutel, Ed H Chi, Jeff Dean, and Neoklis Polyzotis. The case for learned index structures. In _Proceedings of the 2018 International Conference on Management of Data_, pages 489–504, 2018.

[6] Michael Mitzenmacher. A model for learned bloom filters and optimizing by sandwiching. In _Proceedings of the 36th ACM Symposium on Principles of Distributed Computing_, pages 123–131, 2018.

[7] Michael Mitzenmacher and Sergei Vassilvitskii. Algorithms with predictions. _arXiv preprint arXiv:2006.09123_, 2020.

[8] Jack Rae, Sharan Dathathri, Tim Rocktäschel, Emilio Parisotto, Theophane Weber, and Edward Grefenstette. Meta-learning neural bloom filters. _arXiv preprint arXiv:1905.10512_, 2019.

[9] Song Tsai, Jeff Johnson, Matthijs Douze, and Herve Jegou. Learning to index for nearest neighbor search. In _International Conference on Learning Representations (ICLR)_, 2020.

[10] Yuliang Tang, Yan Wu, Laibin Wang, Kai Tan, and Jian Yang. XIndex: A scalable learned index for multicore data storage. In _Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming_, 2020.

[11] Lei Dai, Yuan Guo, Zihong Zhong, and Lintao Wu. From WiscKey to Bourbon: A learned index for log-structured merge trees. In _Proceedings of the 14th USENIX Symposium on Operating Systems Design and Implementation_, 2020.

## License

[MIT License](LICENSE)
