#ifndef LSM_TREE_H
#define LSM_TREE_H

#include "memory_buffer.h"
#include "bloom_filter.h"
#include "fence_pointers.h"
#include "utils.h"
#include <vector>
#include <memory>
#include <string>
#include <mutex>
#include <filesystem>
#include "level.h"
#include "fence_pointer.h"
#include "ml_interface.h"

namespace lsm
{
    // Forward declaration for Run
    class Run;

    // Level in the LSM tree
    class Level
    {
    private:
        size_t levelNumber;
        size_t capacity;
        std::shared_ptr<Run> run;

    public:
        Level(size_t levelNumber, size_t capacity);

        bool isEmpty() const;
        bool isFull() const;
        size_t getCapacity() const;
        size_t getLevelNumber() const;

        // Set the run for this level
        void setRun(std::shared_ptr<Run> newRun);

        // Get the run for this level
        std::shared_ptr<Run> getRun() const;

        // Check if a key exists in this level
        bool containsKey(const KeyType &key) const;

        // Get a value for a key from this level
        bool get(const KeyType &key, ValueType &value) const;

        // Get all key-value pairs in a range from this level
        std::vector<KeyValuePair> range(const KeyType &startKey, const KeyType &endKey) const;
    };

    // Run (sorted file of key-value pairs)
    class Run
    {
    private:
        std::string filePath;
        size_t entryCount;
        FencePointers fencePointers;
        std::shared_ptr<BloomFilter> bloomFilter;

    public:
        // Create a new empty run
        Run(const std::string &filePath, size_t level);

        // Create a run from a vector of key-value pairs
        Run(const std::string &filePath, size_t level, const std::vector<KeyValuePair> &pairs, double falsePositiveRate);

        // Get file path
        std::string getFilePath() const;

        // Get entry count
        size_t getEntryCount() const;

        // Get the size in bytes
        size_t getSizeBytes() const;

        // Check if a key might exist in this run using bloom filter
        bool mightContain(const KeyType &key) const;

        // Get a value for a key from this run
        bool get(const KeyType &key, ValueType &value) const;

        // Get all key-value pairs in a range from this run
        std::vector<KeyValuePair> range(const KeyType &startKey, const KeyType &endKey) const;

        // Split a run if it's too large
        std::shared_ptr<Run> split(size_t maxEntries);
    };

    // Main LSM Tree class
    class LSMTree
    {
    private:
        std::string dataDir;
        MemoryBuffer buffer;
        std::vector<std::unique_ptr<Level>> levels_;
        size_t sizeRatio;
        double baseFpr;
        size_t pageSize;
        size_t entriesPerPage;
        std::mutex treeMutex;
        std::unique_ptr<BloomFilter> bloom_filter_;
        std::unique_ptr<FencePointer> fence_pointer_;
        std::unique_ptr<MLInterface> ml_interface_;
        size_t max_level_size_;
        size_t current_level_size_;
        bool use_ml_predictions_;

        // Create a filename for a run at a specific level
        std::string createRunFilename(size_t level, size_t timestamp) const;

        // Merge two runs
        std::shared_ptr<Run> merge(const std::shared_ptr<Run> &run1, const std::shared_ptr<Run> &run2, size_t level);

        // Insert a run into a level
        void insertRunIntoLevel(std::shared_ptr<Run> run, size_t level);

        // Calculate the optimal false positive rate for a level
        double calculateFalsePositiveRate(size_t level) const;

        // Load existing runs from data directory
        void loadExistingRuns();

        // Helper methods
        void flush_to_next_level(Level* current_level);
        bool try_ml_bloom_prediction(const std::string& key);
        int try_ml_fence_prediction(const std::string& key, int level);

    public:
        // Constructor
        LSMTree(const std::string &dataDir,
                size_t bufferSizeBytes = Constants::DEFAULT_BUFFER_SIZE,
                size_t sizeRatio = Constants::DEFAULT_LEVEL_SIZE_RATIO,
                double baseFpr = 0.01);

        // Destructor
        ~LSMTree();

        // Put a key-value pair
        bool put(const KeyType &key, const ValueType &value);

        // Get a value for a key
        bool get(const KeyType &key, ValueType &value) const;

        // Get all key-value pairs in a range
        std::vector<KeyValuePair> range(const KeyType &startKey, const KeyType &endKey) const;

        // Delete a key
        bool remove(const KeyType &key);

        // Load key-value pairs from a binary file
        bool loadFromFile(const std::string &filePath);

        // Flush buffer if necessary
        void flushIfNecessary();

        // Forcibly flush data to disk, returns success/failure
        bool flushData();

        // Get statistics about the tree
        std::string getStats() const;

        // ML-related methods
        void enable_ml_predictions(bool enable) { use_ml_predictions_ = enable; }
        bool is_ml_ready() const { return ml_interface_ && ml_interface_->are_models_ready(); }
        
        // Training data collection
        void collect_training_data(const std::vector<std::string>& keys,
                                 const std::vector<int>& levels,
                                 const std::vector<int>& page_ids);
    };

} // namespace lsm

#endif // LSM_TREE_H