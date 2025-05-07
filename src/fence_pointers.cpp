#include "fence_pointers.h"
#include <iostream>

namespace lsm
{

    // Build fence pointers from a sorted vector of key-value pairs
    void FencePointers::build(const std::vector<KeyValuePair> &sortedPairs)
    {
        keys.clear();
        offsets.clear();

        entryCount = sortedPairs.size();

        if (entryCount == 0)
        {
            return;
        }

        // Create a fence pointer for every 'interval' entries
        for (size_t i = 0; i < entryCount; i += interval)
        {
            keys.push_back(sortedPairs[i].key);
            offsets.push_back(i);
        }
    }

    // Find the segment that may contain a key
    std::pair<size_t, size_t> FencePointers::findSearchBounds(const KeyType &key) const
    {
        logDebug("Fence pointers: Searching for bounds for key " + std::to_string(key));
        logDebug("Fence pointers: Total keys: " + std::to_string(keys.size()) +
                 ", Total entry count: " + std::to_string(entryCount));

        // If there are no fence pointers, search the entire range
        if (keys.empty())
        {
            logDebug("Fence pointers: No fence pointers, searching entire range (0 to " +
                     std::to_string(entryCount) + ")");
            return std::make_pair(0, entryCount);
        }

        // Binary search to find the segment that may contain the key
        auto it = std::upper_bound(keys.begin(), keys.end(), key);
        size_t index = (it == keys.begin()) ? 0 : (it - keys.begin() - 1);

        // Calculate offset and count
        size_t offset = offsets[index];
        size_t nextOffset = (index + 1 < offsets.size()) ? offsets[index + 1] : entryCount;
        size_t count = nextOffset - offset;

        logDebug("Fence pointers: Found segment at index " + std::to_string(index) +
                 " with key range " + std::to_string(keys[index]) +
                 " to " + (index + 1 < keys.size() ? std::to_string(keys[index + 1]) : "end"));
        logDebug("Fence pointers: Search bounds - offset: " + std::to_string(offset) +
                 ", count: " + std::to_string(count));

        return std::make_pair(offset, count);
    }

    // Serialize fence pointers to a file
    bool FencePointers::serialize(const std::string &filePath) const
    {
        std::ofstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file for writing fence pointers: " + filePath);
            return false;
        }

        // Write metadata
        file.write(reinterpret_cast<const char *>(&interval), sizeof(interval));
        file.write(reinterpret_cast<const char *>(&entryCount), sizeof(entryCount));

        // Write number of fence pointers
        size_t numFencePointers = keys.size();
        file.write(reinterpret_cast<const char *>(&numFencePointers), sizeof(numFencePointers));

        // Write keys
        for (const auto &key : keys)
        {
            file.write(reinterpret_cast<const char *>(&key), sizeof(KeyType));
        }

        // Write offsets
        for (const auto &offset : offsets)
        {
            file.write(reinterpret_cast<const char *>(&offset), sizeof(size_t));
        }

        return !file.fail();
    }

    // Deserialize fence pointers from a file
    bool FencePointers::deserialize(const std::string &filePath)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file for reading fence pointers: " + filePath);
            return false;
        }

        // Read metadata
        file.read(reinterpret_cast<char *>(&interval), sizeof(interval));
        file.read(reinterpret_cast<char *>(&entryCount), sizeof(entryCount));

        // Read number of fence pointers
        size_t numFencePointers;
        file.read(reinterpret_cast<char *>(&numFencePointers), sizeof(numFencePointers));

        if (file.fail())
        {
            logError("Error reading fence pointer metadata from file: " + filePath);
            return false;
        }

        // Read keys
        keys.resize(numFencePointers);
        for (size_t i = 0; i < numFencePointers; i++)
        {
            file.read(reinterpret_cast<char *>(&keys[i]), sizeof(KeyType));
        }

        // Read offsets
        offsets.resize(numFencePointers);
        for (size_t i = 0; i < numFencePointers; i++)
        {
            file.read(reinterpret_cast<char *>(&offsets[i]), sizeof(size_t));
        }

        if (file.fail() && !file.eof())
        {
            logError("Error reading fence pointer data from file: " + filePath);
            return false;
        }

        return true;
    }

    // Print fence pointers for debugging
    void FencePointers::print() const
    {
        std::cout << "Fence Pointers (" << keys.size() << " pointers, "
                  << entryCount << " total entries, interval " << interval << "):" << std::endl;

        for (size_t i = 0; i < keys.size(); i++)
        {
            std::cout << "  [" << i << "] Key: " << keys[i] << ", Offset: " << offsets[i] << std::endl;
        }
    }

} // namespace lsm