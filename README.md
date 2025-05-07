# Predictive LSM Trees

This project implements a Log-Structured Merge-tree (LSM tree) key-value store based on the Monkey design (2017). The implementation follows the design guidelines for an optimal LSM tree, with plans to later add machine learning-based predictions for bloom filters and fence pointers.

## Design

The LSM tree implementation follows the Monkey paper design with these key components:

1. **Level 0 (Memory Buffer)**: 1MB in-memory buffer using a skip list for efficient insertions and lookups.
2. **Tiered Disk Levels**: Each level is 10x larger than the previous level (size ratio T=10).
   - Level 1: First disk level with capacity = 10 × buffer capacity
   - Level 2: Second disk level with capacity = 10 × Level 1 capacity
   - And so on...
3. **Bloom Filters**: Non-uniform false positive rates that increase exponentially with level depth.
   - Base FPR at Level 1 (typically 0.01 or 1%)
   - Each deeper level has 10x higher FPR (following the size ratio)
4. **Fence Pointers**: Efficient disk page lookup within runs to avoid reading entire files.
5. **Leveling Compaction**: Each level contains only one run, with overflow data pushed to deeper levels.

## Building

Requirements:

- C++17 compatible compiler
- GNU Make

Build the project:

```bash
make clean
make
```

This will create two executables in the `bin/` directory:

- `lsm_tree`: The main LSM tree executable with an interactive CLI
- `data_generator`: A utility to generate test data

## Usage

### Data Generator

The data generator creates test datasets of key-value pairs for the LSM tree:

```bash
./bin/data_generator [options]
```

Options:

- `--count N`: Generate N key-value pairs (default: 2,000,000)
- `--key-range N`: Generate keys in range 1-N (default: 20,000,000)
- `--output FILE`: Output to FILE (default: data/test_data.bin)
- `--help`: Display help and exit

Example to generate 10 million entries:

```bash
./bin/data_generator --count 10000000
```

The generator will output a binary file of sorted key-value pairs that can be loaded into the LSM tree.

### Running the LSM Tree

Run the LSM tree with:

```bash
./bin/lsm_tree
```

This launches an interactive command-line interface. On startup, the LSM tree automatically loads any existing run files from the `data/` directory.

If you need to start fresh, clear the data directory first:

```bash
rm -rf data/*
```

### Command Line Interface

The LSM tree supports the following commands:

- `p <key> <value>` - Put a key-value pair
- `g <key>` - Get the value for a key
- `d <key>` - Delete a key
- `r <start_key> <end_key>` - Range query for keys in [start_key, end_key]
- `l <file_path>` - Load key-value pairs from a binary file
- `f` - Force buffer flush to disk
- `s` - Print statistics about the tree
- `h` - Print help message
- `q` - Quit

Example session:

```
> l data/test_data.bin
Load successful. (84357639 μs)
> s
LSM Tree Statistics (Monkey Design):
Level 0 (Memory Buffer):
  Entries: 0 (32 bytes)
  Capacity: 131072 entries (1048576 bytes)
  Flush Policy: Writes to Level 1 when full
Disk Levels:
  Level 1: 1310720 entries (10485760 bytes) - Capacity: 1310720 entries - FPR: 0.01
  Level 2: 7712859 entries (61702872 bytes) - Capacity: 13107200 entries - FPR: 0.1
Bloom Filter Policy:
  Strategy: Exponentially increasing FPR by level (Monkey)
  Base FPR: 0.01
  FPR increases by factor of 10 per level
> g 123
Value: 456 (1250 μs)
> p 1 100
Put successful. (139 μs)
> q
```

## Workflow Example

Here's a typical workflow to test the LSM tree with large datasets:

1. Clear the data directory (optional, for a fresh start)

   ```bash
   rm -rf data/*
   ```

2. Build the project

   ```bash
   make clean
   make
   ```

3. Generate test data

   ```bash
   ./bin/data_generator --count 10000000
   ```

4. Run the LSM tree

   ```bash
   ./bin/lsm_tree
   ```

5. Load the generated data

   ```
   > l data/test_data.bin
   ```

6. Check the tree structure

   ```
   > s
   ```

7. Test operations
   ```
   > g 123
   > r 100 200
   > p 1 100
   ```

## Potential Improvements

The current implementation could be enhanced in several ways:

1. **Metadata Persistence**

   - Add a persistent metadata file to store information about runs
   - Avoid re-reading all run files and rebuilding bloom filters on startup

2. **More Efficient Compaction**

   - Implement a more sophisticated compaction strategy
   - Avoid unnecessary merges and splits during compaction

3. **Bloom Filter Persistence**

   - Save bloom filters to disk to avoid rebuilding them on startup

4. **Lazy Loading**

   - Only load metadata at startup and load actual run data on demand

5. **Machine Learning Integration**
   - Use ML models to predict optimal bloom filter configurations
   - Predict access patterns to optimize fence pointer placement

To implement these changes, focus on modifying:

- `src/lsm_tree.cpp`: The `loadExistingRuns()` method for improved startup performance
- `src/lsm_tree.cpp`: The compaction logic in `insertRunIntoLevel()` for more efficient merging
- Add new files for ML model integration once the basic functionality is stable