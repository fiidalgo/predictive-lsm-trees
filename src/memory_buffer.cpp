#include "memory_buffer.h"
#include <algorithm>
#include <chrono>

namespace lsm
{

    // Constructor
    MemoryBuffer::MemoryBuffer(size_t capacityBytes)
        : probability(0.5),
          currentLevel(0),
          size(0),
          capacityBytes(capacityBytes),
          generator(std::chrono::system_clock::now().time_since_epoch().count()),
          distribution(0.0, 1.0)
    {

        // Initialize head node with maximum level and dummy key/value
        head = new SkipListNode<KeyType, ValueType>(0, 0, MAX_LEVEL);
    }

    // Destructor
    MemoryBuffer::~MemoryBuffer()
    {
        clear();
        delete head;
    }

    // Generate a random level for a new node
    int MemoryBuffer::randomLevel()
    {
        int level = 0;
        while (distribution(generator) < probability && level < MAX_LEVEL)
        {
            level++;
        }
        return level;
    }

    // Calculate memory usage of a key-value pair
    size_t MemoryBuffer::getEntrySize(const KeyType &key, const ValueType &value) const
    {
        // Suppress unused parameter warnings
        (void)key;
        (void)value;

        // SkipListNode contains:
        // - key: sizeof(KeyType)
        // - value: sizeof(ValueType)
        // - forward array of pointers: sizeof(SkipListNode<KeyType, ValueType>*) * (average level + 1)
        // We estimate the average level to be 2 (since probability is 0.5, average level is roughly 1/(1-p) - 1 = 1)
        const int avgLevel = 2;

        return sizeof(KeyType) + sizeof(ValueType) + (sizeof(void *) * (avgLevel + 1));
    }

    // Insert a key-value pair into the buffer
    bool MemoryBuffer::put(const KeyType &key, const ValueType &value)
    {
        std::lock_guard<std::mutex> lock(bufferMutex);

        // Check if adding this entry would exceed capacity
        size_t entrySize = getEntrySize(key, value);
        if (getMemoryUsage() + entrySize > capacityBytes)
        {
            return false;
        }

        // Find the place to insert by traversing the skip list
        SkipListNode<KeyType, ValueType> *update[MAX_LEVEL + 1];
        SkipListNode<KeyType, ValueType> *current = head;

        // Start from the highest level and work down
        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] != nullptr && current->forward[i]->key < key)
            {
                current = current->forward[i];
            }
            update[i] = current;
        }

        current = current->forward[0];

        // If key already exists, update the value
        if (current != nullptr && current->key == key)
        {
            current->value = value;
            return true;
        }

        // Generate a random level for the new node
        int newLevel = randomLevel();

        // Update the current maximum level if needed
        if (newLevel > currentLevel)
        {
            for (int i = currentLevel + 1; i <= newLevel; i++)
            {
                update[i] = head;
            }
            currentLevel = newLevel;
        }

        // Create a new node
        SkipListNode<KeyType, ValueType> *newNode = new SkipListNode<KeyType, ValueType>(key, value, newLevel);

        // Insert the node into the skip list
        for (int i = 0; i <= newLevel; i++)
        {
            newNode->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = newNode;
        }

        size++;
        return true;
    }

    // Get a value for a key
    bool MemoryBuffer::get(const KeyType &key, ValueType &value) const
    {
        std::lock_guard<std::mutex> lock(bufferMutex);

        SkipListNode<KeyType, ValueType> *current = head;

        // Start from the highest level and work down
        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] != nullptr && current->forward[i]->key < key)
            {
                current = current->forward[i];
            }
        }

        current = current->forward[0];

        // Check if key exists
        if (current != nullptr && current->key == key)
        {
            value = current->value;
            return true;
        }

        return false;
    }

    // Get all key-value pairs within a range [startKey, endKey]
    std::vector<KeyValuePair> MemoryBuffer::range(const KeyType &startKey, const KeyType &endKey) const
    {
        std::lock_guard<std::mutex> lock(bufferMutex);

        std::vector<KeyValuePair> result;

        SkipListNode<KeyType, ValueType> *current = head;

        // Find the node with key >= startKey
        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] != nullptr && current->forward[i]->key < startKey)
            {
                current = current->forward[i];
            }
        }

        // Move to the first node in range
        current = current->forward[0];

        // Collect all nodes with key <= endKey
        while (current != nullptr && current->key <= endKey)
        {
            result.emplace_back(current->key, current->value);
            current = current->forward[0];
        }

        return result;
    }

    // Delete a key from the buffer
    bool MemoryBuffer::remove(const KeyType &key)
    {
        std::lock_guard<std::mutex> lock(bufferMutex);

        SkipListNode<KeyType, ValueType> *update[MAX_LEVEL + 1];
        SkipListNode<KeyType, ValueType> *current = head;

        // Find the node to delete
        for (int i = currentLevel; i >= 0; i--)
        {
            while (current->forward[i] != nullptr && current->forward[i]->key < key)
            {
                current = current->forward[i];
            }
            update[i] = current;
        }

        current = current->forward[0];

        // Check if key exists
        if (current == nullptr || current->key != key)
        {
            return false;
        }

        // Remove the node from all levels
        for (int i = 0; i <= currentLevel; i++)
        {
            if (update[i]->forward[i] != current)
            {
                break;
            }
            update[i]->forward[i] = current->forward[i];
        }

        // Update the current level if necessary
        while (currentLevel > 0 && head->forward[currentLevel] == nullptr)
        {
            currentLevel--;
        }

        delete current;
        size--;

        return true;
    }

    // Get the current memory usage in bytes
    size_t MemoryBuffer::getMemoryUsage() const
    {
        // Simple approximation: size of head node + size of each key-value pair * number of elements
        return sizeof(SkipListNode<KeyType, ValueType>) +
               size * (sizeof(KeyType) + sizeof(ValueType) + sizeof(SkipListNode<KeyType, ValueType>));
    }

    // Flush buffer to a vector of key-value pairs (sorted by key)
    std::vector<KeyValuePair> MemoryBuffer::flush()
    {
        std::lock_guard<std::mutex> lock(bufferMutex);

        std::vector<KeyValuePair> result;
        result.reserve(size);

        SkipListNode<KeyType, ValueType> *current = head->forward[0];

        // Collect all key-value pairs
        while (current != nullptr)
        {
            result.emplace_back(current->key, current->value);
            current = current->forward[0];
        }

        return result;
    }

    // Clear the buffer
    void MemoryBuffer::clear()
    {
        std::lock_guard<std::mutex> lock(bufferMutex);

        SkipListNode<KeyType, ValueType> *current = head->forward[0];
        SkipListNode<KeyType, ValueType> *next;

        // Delete all nodes
        while (current != nullptr)
        {
            next = current->forward[0];
            delete current;
            current = next;
        }

        // Reset head's forward pointers
        for (int i = 0; i <= currentLevel; i++)
        {
            head->forward[i] = nullptr;
        }

        currentLevel = 0;
        size = 0;
    }

} // namespace lsm