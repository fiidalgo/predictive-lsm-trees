#ifndef LSM_FENCE_POINTERS_H
#define LSM_FENCE_POINTERS_H

#include "utils.h"
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <fstream>

namespace lsm
{

    class FencePointers
    {
    private:
        std::vector<KeyType> keys;   // Min keys for each segment
        std::vector<size_t> offsets; // Offsets (in entries) to the start of each segment
        size_t entryCount;           // Total number of entries in the run
        size_t interval;             // Number of entries between fence pointers

    public:
        // Constructor with interval
        FencePointers(size_t interval = Constants::DEFAULT_FENCE_POINTER_INTERVAL)
            : entryCount(0), interval(interval) {}

        // Build fence pointers from a sorted vector of key-value pairs
        void build(const std::vector<KeyValuePair> &sortedPairs);

        // Find the segment that may contain a key
        // Returns the offset to start searching from and the number of entries to search
        std::pair<size_t, size_t> findSearchBounds(const KeyType &key) const;

        // Get the total number of entries in the run
        size_t getEntryCount() const { return entryCount; }

        // Get the number of fence pointers
        size_t getFencePointerCount() const { return keys.size(); }

        // Get the interval between fence pointers
        size_t getInterval() const { return interval; }

        // Serialize fence pointers to a file
        bool serialize(const std::string &filePath) const;

        // Deserialize fence pointers from a file
        bool deserialize(const std::string &filePath);

        // Print fence pointers for debugging
        void print() const;
    };

} // namespace lsm

#endif // LSM_FENCE_POINTERS_H