#ifndef LSM_MEMORY_BUFFER_H
#define LSM_MEMORY_BUFFER_H

#include "utils.h"
#include <vector>
#include <map>
#include <random>
#include <functional>
#include <mutex>

namespace lsm
{

    // Skip list node for efficient in-memory operations
    template <typename K, typename V>
    class SkipListNode
    {
    public:
        K key;
        V value;
        std::vector<SkipListNode<K, V> *> forward;

        SkipListNode(K k, V v, int level)
            : key(k), value(v), forward(level + 1, nullptr) {}
    };

    // Memory buffer implemented as a skip list for efficient insertion and lookup
    class MemoryBuffer
    {
    private:
        static constexpr int MAX_LEVEL = 16; // Maximum level for skip list
        float probability;                   // Probability used to determine node level
        int currentLevel;                    // Current maximum level in the skip list
        size_t size;                         // Number of elements in the buffer
        size_t capacityBytes;                // Maximum capacity in bytes

        // Head node of the skip list
        SkipListNode<KeyType, ValueType> *head;

        // RNG for level generation
        std::mt19937 generator;
        std::uniform_real_distribution<float> distribution;

        // Thread safety
        mutable std::mutex bufferMutex;

        // Generate a random level for a new node
        int randomLevel();

        // Calculate memory usage of a key-value pair
        size_t getEntrySize(const KeyType &key, const ValueType &value) const;

    public:
        // Constructor with capacity in bytes
        MemoryBuffer(size_t capacityBytes = Constants::DEFAULT_BUFFER_SIZE);

        // Destructor
        ~MemoryBuffer();

        // Insert a key-value pair into the buffer
        // Returns true if insertion was successful, false if buffer is full
        bool put(const KeyType &key, const ValueType &value);

        // Get a value for a key
        // Returns true if key was found, false otherwise
        // Value is returned through the value reference parameter
        bool get(const KeyType &key, ValueType &value) const;

        // Get all key-value pairs within a range [startKey, endKey]
        std::vector<KeyValuePair> range(const KeyType &startKey, const KeyType &endKey) const;

        // Delete a key from the buffer
        // Returns true if key was found and deleted, false otherwise
        bool remove(const KeyType &key);

        // Check if buffer is empty
        bool isEmpty() const { return size == 0; }

        // Get the number of elements in the buffer
        size_t getSize() const { return size; }

        // Get the current memory usage in bytes
        size_t getMemoryUsage() const;

        // Get the capacity in bytes
        size_t getCapacity() const { return capacityBytes; }

        // Check if buffer is full (memory usage >= capacity)
        bool isFull() const
        {
            // Consider buffer full if it's above 98% capacity to avoid edge cases
            return getMemoryUsage() >= (capacityBytes * 0.98);
        }

        // Flush buffer to a vector of key-value pairs (sorted by key)
        std::vector<KeyValuePair> flush();

        // Clear the buffer
        void clear();
    };

} // namespace lsm

#endif // LSM_MEMORY_BUFFER_H