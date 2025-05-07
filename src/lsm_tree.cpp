#include "lsm_tree.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <regex>

namespace fs = std::filesystem;

namespace lsm
{
    // Level implementation
    Level::Level(size_t levelNumber, size_t capacity)
        : levelNumber(levelNumber), capacity(capacity), run(nullptr) {}

    bool Level::isEmpty() const
    {
        return run == nullptr;
    }

    bool Level::isFull() const
    {
        if (isEmpty())
            return false;
        return run->getEntryCount() >= capacity;
    }

    size_t Level::getCapacity() const
    {
        return capacity;
    }

    size_t Level::getLevelNumber() const
    {
        return levelNumber;
    }

    void Level::setRun(std::shared_ptr<Run> newRun)
    {
        run = newRun;
    }

    std::shared_ptr<Run> Level::getRun() const
    {
        return run;
    }

    bool Level::containsKey(const KeyType &key) const
    {
        if (isEmpty())
            return false;

        // Check bloom filter first (if available)
        if (!run->mightContain(key))
            return false;

        // If bloom filter says maybe, check the actual data
        ValueType value;
        return run->get(key, value);
    }

    bool Level::get(const KeyType &key, ValueType &value) const
    {
        if (isEmpty())
        {
            logDebug("Level " + std::to_string(levelNumber) + " is empty");
            return false;
        }

        // Check bloom filter first (if available)
        if (!run->mightContain(key))
        {
            logDebug("Level " + std::to_string(levelNumber) + ": bloom filter indicates key is definitely not present");
            return false;
        }

        logDebug("Level " + std::to_string(levelNumber) + ": bloom filter says key might be present, checking run");

        // If bloom filter says maybe, check the actual data
        return run->get(key, value);
    }

    std::vector<KeyValuePair> Level::range(const KeyType &startKey, const KeyType &endKey) const
    {
        if (isEmpty())
            return {};

        return run->range(startKey, endKey);
    }

    // Run implementation
    Run::Run(const std::string &filePath, size_t level)
        : filePath(filePath), entryCount(0), fencePointers(Constants::DEFAULT_FENCE_POINTER_INTERVAL), bloomFilter(nullptr)
    {
        // Suppress unused parameter warning
        (void)level;
    }

    Run::Run(const std::string &filePath, size_t level, const std::vector<KeyValuePair> &pairs, double falsePositiveRate)
        : filePath(filePath), entryCount(pairs.size()), fencePointers(Constants::DEFAULT_FENCE_POINTER_INTERVAL)
    {
        // Suppress unused parameter warning
        (void)level;

        // Create directory if it doesn't exist
        fs::path dir = fs::path(filePath).parent_path();
        if (!dir.empty() && !fs::exists(dir))
        {
            fs::create_directories(dir);
        }

        // Write sorted pairs to file
        if (!writeKeyValuePairsToFile(filePath, pairs))
        {
            throw LSMException("Failed to write run to file: " + filePath);
        }

        // Build fence pointers
        fencePointers.build(pairs);

        // Create bloom filter (if falsePositiveRate < 1.0)
        if (falsePositiveRate < 1.0)
        {
            bloomFilter = std::make_shared<BloomFilter>(pairs.size(), falsePositiveRate);
            for (const auto &pair : pairs)
            {
                bloomFilter->add(pair.key);
            }
        }
    }

    std::string Run::getFilePath() const
    {
        return filePath;
    }

    size_t Run::getEntryCount() const
    {
        return entryCount;
    }

    size_t Run::getSizeBytes() const
    {
        if (fs::exists(filePath))
        {
            return fs::file_size(filePath);
        }
        return 0;
    }

    bool Run::mightContain(const KeyType &key) const
    {
        // If no bloom filter, assume it might contain the key
        if (!bloomFilter)
        {
            return true;
        }

        return bloomFilter->mightContain(key);
    }

    bool Run::get(const KeyType &key, ValueType &value) const
    {
        logDebug("Checking run at: " + filePath + " for key: " + std::to_string(key));

        // If the bloom filter says it's not here, return false immediately
        if (bloomFilter)
        {
            bool mightContainResult = bloomFilter->mightContain(key);
            logDebug("Bloom filter result: " + std::string(mightContainResult ? "MIGHT CONTAIN" : "DEFINITELY NOT PRESENT"));

            if (!mightContainResult)
            {
                return false;
            }
        }
        else
        {
            logDebug("No bloom filter present for this run, must check disk");
        }

        // Use fence pointers to find the search bounds
        auto bounds = fencePointers.findSearchBounds(key);
        size_t offset = bounds.first;
        size_t count = bounds.second;

        logDebug("Fence pointers narrowed search to offset: " + std::to_string(offset) +
                 ", entries to check: " + std::to_string(count));

        // Read the entries in the bounded range
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file: " + filePath);
            return false;
        }

        // Seek to the starting position
        file.seekg(offset * sizeof(KeyValuePair));
        logDebug("Seeking to position: " + std::to_string(offset * sizeof(KeyValuePair)) + " bytes");

        // Read up to 'count' entries
        size_t entriesChecked = 0;
        for (size_t i = 0; i < count && file.good(); ++i)
        {
            KeyValuePair pair = deserializeKeyValuePair(file);
            entriesChecked++;

            logDebug("Checking entry with key: " + std::to_string(pair.key));

            if (pair.key == key)
            {
                value = pair.value;
                // If it's a tombstone, consider it not found
                if (value == TOMBSTONE_VALUE)
                {
                    logDebug("Found key but it's a tombstone (deleted)");
                    return false;
                }
                logDebug("Found key with value: " + std::to_string(value) +
                         " after checking " + std::to_string(entriesChecked) + " entries");
                return true;
            }
            if (pair.key > key)
            {
                // We've gone past the key, so it's not here
                logDebug("Passed target key position, key not found");
                break;
            }
        }

        logDebug("Key not found in run after checking " + std::to_string(entriesChecked) + " entries");
        return false;
    }

    std::vector<KeyValuePair> Run::range(const KeyType &startKey, const KeyType &endKey) const
    {
        std::vector<KeyValuePair> result;

        // Open the file
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            return result;
        }

        // Use fence pointers to find the start search bounds
        auto bounds = fencePointers.findSearchBounds(startKey);
        size_t offset = bounds.first;

        // Seek to the starting position
        file.seekg(offset * sizeof(KeyValuePair));

        // Read entries until we hit the end key
        while (file.good())
        {
            KeyValuePair pair = deserializeKeyValuePair(file);

            // If we've gone past the end key, we're done
            if (pair.key > endKey)
            {
                break;
            }

            // If the key is in range and not a tombstone, add it to the result
            if (pair.key >= startKey && pair.value != TOMBSTONE_VALUE)
            {
                result.push_back(pair);
            }
        }

        return result;
    }

    std::shared_ptr<Run> Run::split(size_t maxEntries)
    {
        // If the run is small enough, no need to split
        if (entryCount <= maxEntries)
        {
            return nullptr;
        }

        // Read all entries
        std::vector<KeyValuePair> allEntries;
        if (!readKeyValuePairsFromFile(filePath, allEntries))
        {
            throw LSMException("Failed to read run for splitting: " + filePath);
        }

        // Split point
        size_t splitPoint = maxEntries;

        // Create new runs with the split data
        std::vector<KeyValuePair> firstPart(allEntries.begin(), allEntries.begin() + splitPoint);
        std::vector<KeyValuePair> secondPart(allEntries.begin() + splitPoint, allEntries.end());

        // Create a new file path for the second part
        std::string newFilePath = filePath + ".split";

        // Create the second run
        double fpr = bloomFilter ? bloomFilter->getFalsePositiveRate() : 1.0;
        auto secondRun = std::make_shared<Run>(newFilePath, 0, secondPart, fpr);

        // Update this run with the first part
        if (!writeKeyValuePairsToFile(filePath, firstPart))
        {
            throw LSMException("Failed to write first part of split run to file: " + filePath);
        }

        entryCount = firstPart.size();
        fencePointers.build(firstPart);

        if (bloomFilter)
        {
            bloomFilter = std::make_shared<BloomFilter>(firstPart.size(), bloomFilter->getFalsePositiveRate());
            for (const auto &pair : firstPart)
            {
                bloomFilter->add(pair.key);
            }
        }

        return secondRun;
    }

    // LSMTree implementation
    LSMTree::LSMTree(const std::string &dataDir, size_t bufferSizeBytes, size_t sizeRatio, double baseFpr)
        : dataDir(dataDir), buffer(bufferSizeBytes), sizeRatio(sizeRatio), baseFpr(baseFpr),
          pageSize(Constants::DEFAULT_BLOCK_SIZE), entriesPerPage(pageSize / sizeof(KeyValuePair)),
          max_level_size_(0), current_level_size_(0), use_ml_predictions_(false)
    {
        // Create data directory if it doesn't exist
        if (!fs::exists(dataDir))
        {
            fs::create_directories(dataDir);
        }

        // Calculate buffer (Level 0) capacity in entries
        size_t bufferEntriesCapacity = bufferSizeBytes / sizeof(KeyValuePair);

        // As per Monkey paper:
        // - Level 0 is the buffer (in memory)
        // - Level 1 is the first disk level with capacity = buffer capacity * size ratio
        size_t level1Capacity = bufferEntriesCapacity * sizeRatio;

        // Initialize with Level 1 (first disk level)
        levels.push_back(Level(1, level1Capacity));

        logDebug("Initialized LSM Tree with:");
        logDebug("  Level 0 (Buffer): " + std::to_string(bufferEntriesCapacity) + " entries capacity");
        logDebug("  Level 1 (Disk): " + std::to_string(level1Capacity) + " entries capacity");
        logDebug("  Size ratio (T): " + std::to_string(sizeRatio));

        // Scan for existing run files and load them
        loadExistingRuns();

        // Initialize ML interface
        try {
            ml_interface_ = std::make_unique<MLInterface>("lsm_ml_shm", 1024 * 1024);
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize ML interface: " << e.what() << std::endl;
        }
    }

    LSMTree::~LSMTree()
    {
        // Flush buffer if not empty
        if (!buffer.isEmpty())
        {
            flushIfNecessary();
        }
    }

    std::string LSMTree::createRunFilename(size_t level, size_t timestamp) const
    {
        std::stringstream ss;
        ss << dataDir << "/level_" << level << "_" << timestamp << ".bin";
        return ss.str();
    }

    double LSMTree::calculateFalsePositiveRate(size_t level) const
    {
        // Implement Monkey's strategy: FPR increases exponentially with level
        return baseFpr * std::pow(sizeRatio, level);
    }

    std::shared_ptr<Run> LSMTree::merge(const std::shared_ptr<Run> &run1, const std::shared_ptr<Run> &run2, size_t level)
    {
        logDebug("Merging runs at level " + std::to_string(level) + ": " +
                 run1->getFilePath() + " and " + run2->getFilePath());

        // Read both runs
        std::vector<KeyValuePair> entries1, entries2;

        try
        {
            if (!readKeyValuePairsFromFile(run1->getFilePath(), entries1))
            {
                logError("Failed to read first run for merging: " + run1->getFilePath() +
                         " - Creating new run with second run only");
                return run2;
            }

            if (!readKeyValuePairsFromFile(run2->getFilePath(), entries2))
            {
                logError("Failed to read second run for merging: " + run2->getFilePath() +
                         " - Creating new run with first run only");
                return run1;
            }

            logDebug("Successfully read " + std::to_string(entries1.size()) +
                     " entries from run1 and " + std::to_string(entries2.size()) + " entries from run2");
        }
        catch (const std::exception &e)
        {
            logError("Exception while reading runs for merging: " + std::string(e.what()));

            // If we have at least one valid run, return it
            if (!entries1.empty())
            {
                logDebug("Using run1 only for merge result");
                return run1;
            }
            if (!entries2.empty())
            {
                logDebug("Using run2 only for merge result");
                return run2;
            }

            // Create an empty run if both failed
            std::string newFilePath = createRunFilename(level, std::chrono::system_clock::now().time_since_epoch().count());
            logDebug("Creating empty run as fallback: " + newFilePath);
            return std::make_shared<Run>(newFilePath, level);
        }

        // Merge the two sorted runs
        std::vector<KeyValuePair> mergedEntries;
        mergedEntries.reserve(entries1.size() + entries2.size());

        size_t i = 0, j = 0;
        while (i < entries1.size() && j < entries2.size())
        {
            if (entries1[i].key < entries2[j].key)
            {
                mergedEntries.push_back(entries1[i++]);
            }
            else if (entries1[i].key > entries2[j].key)
            {
                mergedEntries.push_back(entries2[j++]);
            }
            else
            {
                // If keys are equal, take the newer value (assumed to be from entries2)
                mergedEntries.push_back(entries2[j++]);
                i++;
            }
        }

        // Add remaining entries
        while (i < entries1.size())
        {
            mergedEntries.push_back(entries1[i++]);
        }
        while (j < entries2.size())
        {
            mergedEntries.push_back(entries2[j++]);
        }

        // Create a new run for the merged entries
        std::string newFilePath = createRunFilename(level, std::chrono::system_clock::now().time_since_epoch().count());
        double fpr = calculateFalsePositiveRate(level);

        logDebug("Creating merged run with " + std::to_string(mergedEntries.size()) + " entries");
        return std::make_shared<Run>(newFilePath, level, mergedEntries, fpr);
    }

    void LSMTree::insertRunIntoLevel(std::shared_ptr<Run> run, size_t level)
    {
        logDebug("Inserting run into level " + std::to_string(level) + ": " + run->getFilePath() +
                 " (" + std::to_string(run->getEntryCount()) + " entries)");

        // Get buffer capacity in entries
        size_t bufferEntriesCapacity = buffer.getCapacity() / sizeof(KeyValuePair);

        // Ensure we have enough levels with proper capacity based on Monkey paper
        while (levels.size() < level)
        {
            size_t newLevelNumber = levels.size() + 1; // Level numbers start from 1

            // Calculate the capacity following the Monkey paper's formula:
            // For Level L, capacity = buffer_capacity * T^L
            size_t newCapacity = bufferEntriesCapacity * std::pow(sizeRatio, newLevelNumber);

            levels.push_back(Level(newLevelNumber, newCapacity));
            logDebug("Created new level " + std::to_string(newLevelNumber) +
                     " with capacity " + std::to_string(newCapacity) + " entries");
        }

        // Adjust level index for zero-based array access
        size_t levelIdx = level - 1;

        // Check if the level is empty
        if (levels[levelIdx].isEmpty())
        {
            logDebug("Level " + std::to_string(level) + " is empty, setting run directly");
            levels[levelIdx].setRun(run);
            return;
        }

        try
        {
            // Merge with existing run at this level
            std::shared_ptr<Run> existingRun = levels[levelIdx].getRun();
            std::shared_ptr<Run> mergedRun = merge(existingRun, run, level);

            // If the merged run fits in this level, update the level
            if (mergedRun->getEntryCount() <= levels[levelIdx].getCapacity())
            {
                logDebug("Merged run fits in level " + std::to_string(level) +
                         " (" + std::to_string(mergedRun->getEntryCount()) + "/" +
                         std::to_string(levels[levelIdx].getCapacity()) + " entries)");

                levels[levelIdx].setRun(mergedRun);

                // Delete the old run files - but handle errors gracefully
                try
                {
                    if (fs::exists(existingRun->getFilePath()))
                    {
                        fs::remove(existingRun->getFilePath());
                        logDebug("Deleted old run file: " + existingRun->getFilePath());
                    }
                    if (fs::exists(run->getFilePath()))
                    {
                        fs::remove(run->getFilePath());
                        logDebug("Deleted old run file: " + run->getFilePath());
                    }
                }
                catch (const std::exception &e)
                {
                    logError("Error deleting old run files: " + std::string(e.what()));
                    // Continue anyway, not critical if old files remain
                }
            }
            else
            {
                logDebug("Merged run too large for level " + std::to_string(level) +
                         " (" + std::to_string(mergedRun->getEntryCount()) + " > " +
                         std::to_string(levels[levelIdx].getCapacity()) + " entries), splitting");

                // If it doesn't fit, split it
                levels[levelIdx].setRun(mergedRun);
                std::shared_ptr<Run> overflowRun = mergedRun->split(levels[levelIdx].getCapacity());

                if (overflowRun)
                {
                    logDebug("Created overflow run with " + std::to_string(overflowRun->getEntryCount()) +
                             " entries, inserting into level " + std::to_string(level + 1));

                    // Insert overflow into the next level
                    insertRunIntoLevel(overflowRun, level + 1);
                }
            }
        }
        catch (const std::exception &e)
        {
            logError("Error during run insertion: " + std::string(e.what()));

            // In case of error, just set the run directly to avoid data loss
            logDebug("Setting run directly due to error");
            levels[levelIdx].setRun(run);
        }
    }

    bool LSMTree::put(const KeyType &key, const ValueType &value)
    {
        std::lock_guard<std::mutex> lock(treeMutex);

        bool success = buffer.put(key, value);
        flushIfNecessary();
        return success;
    }

    bool LSMTree::get(const KeyType &key, ValueType &value) const
    {
        logDebug("Looking up key: " + std::to_string(key));

        // First check the buffer
        logDebug("Checking memory buffer");
        if (buffer.get(key, value))
        {
            logDebug("Key found in memory buffer with value: " + std::to_string(value));
            return true;
        }

        logDebug("Key not found in memory buffer, checking disk levels");

        // Then check each level
        for (size_t i = 0; i < levels.size(); ++i)
        {
            const auto &level = levels[i];
            logDebug("Checking level " + std::to_string(i) +
                     " (FPR: " + std::to_string(calculateFalsePositiveRate(i)) + ")");

            if (level.isEmpty())
            {
                logDebug("Level " + std::to_string(i) + " is empty, skipping");
                continue;
            }

            if (level.get(key, value))
            {
                logDebug("Key found in level " + std::to_string(i) + " with value: " + std::to_string(value));
                return true;
            }
        }

        logDebug("Key not found in any level");
        return false;
    }

    std::vector<KeyValuePair> LSMTree::range(const KeyType &startKey, const KeyType &endKey) const
    {
        // Get results from buffer
        std::vector<KeyValuePair> result = buffer.range(startKey, endKey);

        // Get results from each level
        for (const auto &level : levels)
        {
            if (level.isEmpty())
            {
                continue;
            }

            std::vector<KeyValuePair> levelResults = level.range(startKey, endKey);
            result.insert(result.end(), levelResults.begin(), levelResults.end());
        }

        // Sort and remove duplicates (keeping the latest value for each key)
        std::sort(result.begin(), result.end(), [](const KeyValuePair &a, const KeyValuePair &b)
                  { return a.key < b.key; });

        auto it = std::unique(result.begin(), result.end(), [](const KeyValuePair &a, const KeyValuePair &b)
                              { return a.key == b.key; });

        result.erase(it, result.end());

        // Remove tombstones
        auto newEnd = std::remove_if(result.begin(), result.end(), [](const KeyValuePair &pair)
                                     { return pair.value == TOMBSTONE_VALUE; });

        result.erase(newEnd, result.end());

        return result;
    }

    bool LSMTree::remove(const KeyType &key)
    {
        // Delete is implemented as a put with the tombstone value
        return put(key, TOMBSTONE_VALUE);
    }

    bool LSMTree::loadFromFile(const std::string &filePath)
    {
        std::lock_guard<std::mutex> lock(treeMutex);

        // Load key-value pairs from the file
        std::vector<KeyValuePair> pairs;
        if (!loadKeyValuePairsFromBinaryFile(filePath, pairs))
        {
            return false;
        }

        logDebug("Loaded " + std::to_string(pairs.size()) + " key-value pairs from file");

        // Sort the pairs by key
        std::sort(pairs.begin(), pairs.end());

        // Insert the pairs in small batches to ensure buffer flushes
        // Each entry in the SkipList has significant overhead, so use a conservative batch size
        size_t batchSize = 1000; // Conservative batch size to ensure flushing

        for (size_t i = 0; i < pairs.size(); i += batchSize)
        {
            size_t end = std::min(i + batchSize, pairs.size());
            logDebug("Processing batch " + std::to_string(i / batchSize + 1) +
                     " of " + std::to_string((pairs.size() + batchSize - 1) / batchSize));

            // Check and flush before inserting the batch
            if (buffer.getSize() > 0 && buffer.getMemoryUsage() >= buffer.getCapacity() * 0.9)
            {
                logDebug("Buffer is approaching capacity (" + std::to_string(buffer.getMemoryUsage()) +
                         "/" + std::to_string(buffer.getCapacity()) + " bytes), forcing flush before batch");
                if (!flushData())
                {
                    logError("Failed to flush data during batch loading");
                    return false;
                }
            }

            for (size_t j = i; j < end; ++j)
            {
                // Try to insert
                if (!buffer.put(pairs[j].key, pairs[j].value))
                {
                    logDebug("Buffer can't fit next entry, flushing");

                    // Flush and try again
                    if (!flushData())
                    {
                        logError("Failed to flush data during entry insertion");
                        return false;
                    }

                    // Retry insertion
                    if (!buffer.put(pairs[j].key, pairs[j].value))
                    {
                        logError("Failed to insert key-value pair even after flushing. Key: " +
                                 std::to_string(pairs[j].key));
                        return false;
                    }
                }
            }
        }

        // Make sure any remaining data is flushed
        if (buffer.getSize() > 0)
        {
            flushData();
        }

        logDebug("Finished loading file with " + std::to_string(pairs.size()) + " key-value pairs");
        return true;
    }

    // Helper function to flush data to disk
    bool LSMTree::flushData()
    {
        if (buffer.getSize() == 0)
        {
            return true; // Nothing to flush
        }

        logDebug("Flushing " + std::to_string(buffer.getSize()) + " entries from buffer");

        // Flush buffer to a new run
        std::vector<KeyValuePair> flushData = buffer.flush();

        if (flushData.empty())
        {
            logError("Flush produced empty data despite buffer having entries");
            return false;
        }

        // Create a new run at level 1
        std::string runFilePath = createRunFilename(1, std::chrono::system_clock::now().time_since_epoch().count());
        logDebug("Creating new run at level 1: " + runFilePath);

        auto run = std::make_shared<Run>(runFilePath, 1, flushData, calculateFalsePositiveRate(0));

        // Insert the run into level 1
        insertRunIntoLevel(run, 1);
        logDebug("Run inserted into level 1");

        buffer.clear();
        logDebug("Buffer cleared");

        return true;
    }

    void LSMTree::flushIfNecessary()
    {
        if (buffer.isFull())
        {
            logDebug("Buffer is full (" + std::to_string(buffer.getMemoryUsage()) +
                     "/" + std::to_string(buffer.getCapacity()) + " bytes), flushing to disk");

            // Flush buffer to a new run
            std::vector<KeyValuePair> flushData = buffer.flush();

            logDebug("Flushed " + std::to_string(flushData.size()) + " entries from buffer");

            if (!flushData.empty())
            {
                // Create a new run at level 1
                std::string runFilePath = createRunFilename(1, std::chrono::system_clock::now().time_since_epoch().count());
                logDebug("Creating new run at level 1: " + runFilePath);

                auto run = std::make_shared<Run>(runFilePath, 1, flushData, calculateFalsePositiveRate(0));

                // Insert the run into level 1
                insertRunIntoLevel(run, 1);
                logDebug("Run inserted into level 1");
            }

            buffer.clear();
            logDebug("Buffer cleared");
        }
        else
        {
            logDebug("Buffer not full (" + std::to_string(buffer.getMemoryUsage()) +
                     "/" + std::to_string(buffer.getCapacity()) + " bytes), no flush needed");
        }
    }

    std::string LSMTree::getStats() const
    {
        std::stringstream ss;

        ss << "LSM Tree Statistics (Monkey Design):" << std::endl;

        // Buffer is Level 0 in the Monkey paper
        size_t bufferCapacityEntries = buffer.getCapacity() / sizeof(KeyValuePair);
        ss << "Level 0 (Memory Buffer):" << std::endl;
        ss << "  Entries: " << buffer.getSize() << " (" << buffer.getMemoryUsage() << " bytes)" << std::endl;
        ss << "  Capacity: " << bufferCapacityEntries << " entries (" << buffer.getCapacity() << " bytes)" << std::endl;
        ss << "  Flush Policy: Writes to Level 1 when full" << std::endl;

        ss << "Disk Levels:" << std::endl;

        for (size_t i = 0; i < levels.size(); ++i)
        {
            const auto &level = levels[i];
            // Actual level numbers in the Monkey design (level 1 is first disk level)
            size_t levelNumber = i + 1;
            ss << "  Level " << levelNumber << ": ";

            if (level.isEmpty())
            {
                ss << "empty";
            }
            else
            {
                auto run = level.getRun();
                ss << run->getEntryCount() << " entries (" << run->getSizeBytes() << " bytes)";
                ss << " - Capacity: " << level.getCapacity() << " entries";
                ss << " - FPR: " << calculateFalsePositiveRate(i);
            }

            ss << std::endl;
        }

        // Add info about Bloom filters (Monkey's key optimization)
        ss << "Bloom Filter Policy:" << std::endl;
        ss << "  Strategy: Exponentially increasing FPR by level (Monkey)" << std::endl;
        ss << "  Base FPR: " << baseFpr << std::endl;
        ss << "  FPR increases by factor of " << sizeRatio << " per level" << std::endl;

        return ss.str();
    }

    void LSMTree::loadExistingRuns()
    {
        logDebug("Scanning for existing run files in: " + dataDir);

        // Map to store level -> list of run files for that level
        std::map<size_t, std::vector<std::string>> levelFiles;

        // Regex to match level file pattern: level_X_timestamp.bin
        std::regex filePattern("level_(\\d+)_(\\d+)\\.bin");

        // Scan the data directory for run files
        try
        {
            for (const auto &entry : fs::directory_iterator(dataDir))
            {
                if (entry.is_regular_file())
                {
                    std::string filename = entry.path().filename().string();
                    std::smatch matches;

                    if (std::regex_match(filename, matches, filePattern) && matches.size() == 3)
                    {
                        size_t level = std::stoul(matches[1].str());
                        levelFiles[level].push_back(entry.path().string());
                        logDebug("Found run file: " + filename + " for level " + std::to_string(level));
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            logError("Error scanning data directory: " + std::string(e.what()));
            return;
        }

        // Calculate buffer capacity in entries
        size_t bufferEntriesCapacity = buffer.getCapacity() / sizeof(KeyValuePair);

        // Make sure we have enough levels with proper capacities
        for (const auto &levelPair : levelFiles)
        {
            size_t level = levelPair.first;

            // Ensure we have enough levels, with proper capacity based on the Monkey paper's size ratio
            while (levels.size() < level)
            {
                size_t newLevelNumber = levels.size() + 1; // Level numbers start from 1 for disk levels

                // Calculate the capacity following the Monkey paper's formula:
                // For Level L, capacity = buffer_capacity * T^L
                // Where T is the size ratio (typically 10)
                size_t newCapacity = bufferEntriesCapacity * std::pow(sizeRatio, newLevelNumber);

                logDebug("Creating Level " + std::to_string(newLevelNumber) +
                         " with capacity " + std::to_string(newCapacity) + " entries");

                levels.push_back(Level(newLevelNumber, newCapacity));
            }
        }

        // Load each level's runs, starting from the highest level and working down
        // This allows us to properly handle overflow during loading
        std::vector<size_t> levelNumbers;
        for (const auto &levelPair : levelFiles)
        {
            levelNumbers.push_back(levelPair.first);
        }

        // Sort levels in descending order (highest level first)
        std::sort(levelNumbers.begin(), levelNumbers.end(), std::greater<size_t>());

        for (size_t level : levelNumbers)
        {
            const auto &files = levelFiles[level];

            if (files.empty())
            {
                continue;
            }

            // Process files one by one, merging them into the level
            for (const auto &filePath : files)
            {
                // Try to read the run
                std::vector<KeyValuePair> entries;
                if (readKeyValuePairsFromFile(filePath, entries))
                {
                    logDebug("Loading " + std::to_string(entries.size()) + " entries from " + filePath);

                    // Create a new Run with the existing data
                    double fpr = calculateFalsePositiveRate(level - 1); // Adjust index for 0-based array
                    auto run = std::make_shared<Run>(filePath, level, entries, fpr);

                    // If the level is empty, just set the run
                    if (levels[level - 1].isEmpty()) // Adjust index for 0-based array
                    {
                        // Check if we need to split this run (it might be too large for this level)
                        if (run->getEntryCount() > levels[level - 1].getCapacity())
                        {
                            logDebug("Run is too large for level " + std::to_string(level) +
                                     " (" + std::to_string(run->getEntryCount()) + " > " +
                                     std::to_string(levels[level - 1].getCapacity()) + " entries), splitting");

                            levels[level - 1].setRun(run);
                            std::shared_ptr<Run> overflowRun = run->split(levels[level - 1].getCapacity());

                            if (overflowRun && level < levels.size())
                            {
                                // Insert overflow into the next level
                                insertRunIntoLevel(overflowRun, level);
                            }
                        }
                        else
                        {
                            levels[level - 1].setRun(run);
                            logDebug("Run loaded into level " + std::to_string(level));
                        }
                    }
                    else
                    {
                        // Merge with existing run
                        std::shared_ptr<Run> existingRun = levels[level - 1].getRun();
                        std::shared_ptr<Run> mergedRun = merge(existingRun, run, level);

                        // Check if merged run fits in this level
                        if (mergedRun->getEntryCount() <= levels[level - 1].getCapacity())
                        {
                            logDebug("Merged run fits in level " + std::to_string(level) +
                                     " (" + std::to_string(mergedRun->getEntryCount()) + "/" +
                                     std::to_string(levels[level - 1].getCapacity()) + " entries)");

                            levels[level - 1].setRun(mergedRun);

                            // Clean up old files if they're different from the merged run
                            try
                            {
                                if (fs::exists(existingRun->getFilePath()) &&
                                    existingRun->getFilePath() != mergedRun->getFilePath())
                                {
                                    fs::remove(existingRun->getFilePath());
                                }
                                if (fs::exists(run->getFilePath()) &&
                                    run->getFilePath() != mergedRun->getFilePath())
                                {
                                    fs::remove(run->getFilePath());
                                }
                            }
                            catch (const std::exception &e)
                            {
                                logError("Error cleaning up old run files: " + std::string(e.what()));
                            }
                        }
                        else
                        {
                            logDebug("Merged run too large for level " + std::to_string(level) +
                                     " (" + std::to_string(mergedRun->getEntryCount()) + " > " +
                                     std::to_string(levels[level - 1].getCapacity()) + " entries), splitting");

                            // If it doesn't fit, split it
                            levels[level - 1].setRun(mergedRun);
                            std::shared_ptr<Run> overflowRun = mergedRun->split(levels[level - 1].getCapacity());

                            if (overflowRun && level < levels.size())
                            {
                                // Insert overflow into the next level
                                insertRunIntoLevel(overflowRun, level);
                            }
                        }
                    }
                }
                else
                {
                    logError("Failed to read run file: " + filePath);
                }
            }
        }

        logDebug("Finished loading existing runs");
    }

    void LSMTree::put(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Add to current level
        levels_[0]->put(key, value);
        current_level_size_++;
        
        // Check if we need to flush
        if (current_level_size_ >= max_level_size_) {
            flush_to_next_level(levels_[0].get());
            current_level_size_ = 0;
        }
    }

    std::string LSMTree::get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try ML-based bloom filter prediction if enabled
        if (use_ml_predictions_ && is_ml_ready()) {
            if (!try_ml_bloom_prediction(key)) {
                return "";  // Key definitely not present
            }
        } else {
            // Use traditional bloom filter
            if (!bloom_filter_->might_contain(key)) {
                return "";
            }
        }
        
        // Search through levels
        for (size_t i = 0; i < levels_.size(); i++) {
            // Try ML-based fence pointer prediction if enabled
            int page_id = -1;
            if (use_ml_predictions_ && is_ml_ready()) {
                page_id = try_ml_fence_prediction(key, i);
            }
            
            // If ML prediction failed, use traditional fence pointer
            if (page_id == -1) {
                page_id = fence_pointer_->get_page_id(key, i);
            }
            
            if (page_id >= 0) {
                std::string value = levels_[i]->get_from_page(key, page_id);
                if (!value.empty()) {
                    return value;
                }
            }
        }
        
        return "";
    }

    void LSMTree::remove(const std::string& key) {
        put(key, "");  // Tombstone
    }

    void LSMTree::flush_to_next_level(Level* current_level) {
        // Collect training data
        std::vector<std::string> keys;
        std::vector<int> levels;
        std::vector<int> page_ids;
        
        current_level->collect_training_data(keys, levels, page_ids);
        
        // Send training data to ML models
        if (ml_interface_) {
            std::vector<double> key_doubles;
            for (const auto& key : keys) {
                // Convert string key to double (simple hash)
                double key_double = 0;
                for (char c : key) {
                    key_double = key_double * 31 + c;
                }
                key_doubles.push_back(key_double);
            }
            
            ml_interface_->write_training_data(key_doubles, levels, page_ids);
        }
        
        // Perform the actual flush
        if (levels_.size() <= current_level->get_level() + 1) {
            levels_.push_back(std::make_unique<Level>(current_level->get_level() + 1));
        }
        
        Level* next_level = levels_[current_level->get_level() + 1].get();
        current_level->flush_to(next_level);
    }

    bool LSMTree::try_ml_bloom_prediction(const std::string& key) {
        if (!ml_interface_) return true;
        
        try {
            // Convert key to double
            double key_double = 0;
            for (char c : key) {
                key_double = key_double * 31 + c;
            }
            
            // Get prediction
            ml_interface_->write_bloom_request(key_double);
            float prob = ml_interface_->read_bloom_response();
            
            return prob > 0.5;  // Threshold for positive prediction
        } catch (const std::exception& e) {
            std::cerr << "ML bloom prediction failed: " << e.what() << std::endl;
            return true;  // Fall back to traditional bloom filter
        }
    }

    int LSMTree::try_ml_fence_prediction(const std::string& key, int level) {
        if (!ml_interface_) return -1;
        
        try {
            // Convert key to double
            double key_double = 0;
            for (char c : key) {
                key_double = key_double * 31 + c;
            }
            
            // Get prediction
            ml_interface_->write_fence_request(key_double, level);
            return ml_interface_->read_fence_response();
        } catch (const std::exception& e) {
            std::cerr << "ML fence prediction failed: " << e.what() << std::endl;
            return -1;  // Fall back to traditional fence pointer
        }
    }

    void LSMTree::collect_training_data(const std::vector<std::string>& keys,
                                      const std::vector<int>& levels,
                                      const std::vector<int>& page_ids) {
        if (!ml_interface_) return;
        
        try {
            std::vector<double> key_doubles;
            for (const auto& key : keys) {
                double key_double = 0;
                for (char c : key) {
                    key_double = key_double * 31 + c;
                }
                key_doubles.push_back(key_double);
            }
            
            ml_interface_->write_training_data(key_doubles, levels, page_ids);
        } catch (const std::exception& e) {
            std::cerr << "Failed to collect training data: " << e.what() << std::endl;
        }
    }

} // namespace lsm
