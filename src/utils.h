#ifndef LSM_UTILS_H
#define LSM_UTILS_H

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <functional>

// Define DEBUG to enable debug logging
#define DEBUG

namespace lsm
{

    // Type definitions for keys and values
    using KeyType = int32_t;   // Signed 32-bit integers for keys
    using ValueType = int32_t; // Signed 32-bit integers for values

    // Key-Value pair structure
    struct KeyValuePair
    {
        KeyType key;
        ValueType value;

        KeyValuePair() : key(0), value(0) {}
        KeyValuePair(KeyType k, ValueType v) : key(k), value(v) {}

        // Comparison operators
        bool operator<(const KeyValuePair &other) const { return key < other.key; }
        bool operator==(const KeyValuePair &other) const { return key == other.key; }
        bool operator!=(const KeyValuePair &other) const { return key != other.key; }
    };

    // Tombstone marker for deleted entries
    const ValueType TOMBSTONE_VALUE = INT32_MIN;

    // System constants
    struct Constants
    {
        static const size_t DEFAULT_BUFFER_SIZE = 1024 * 1024;        // 1MB buffer (Level 0)
        static const size_t DEFAULT_BLOCK_SIZE = 4096;                // 4KB block size for disk I/O
        static const size_t DEFAULT_FENCE_POINTER_INTERVAL = 16;      // Store fence pointer every 16 entries
        static const size_t DEFAULT_BLOOM_FILTER_BITS_PER_ENTRY = 10; // 10 bits per entry for bloom filter
        static const size_t DEFAULT_BLOOM_FILTER_HASH_COUNT = 7;      // Number of hash functions for bloom filter
        static const size_t DEFAULT_LEVEL_SIZE_RATIO = 10;            // Each level is 10x larger than previous (Monkey paper)
        static const int DEFAULT_SERVER_PORT = 9876;                  // Default server port
    };

    // File path utilities
    std::string joinPath(const std::string &base, const std::string &path);
    bool fileExists(const std::string &path);
    bool createDirectory(const std::string &path);
    std::vector<std::string> listFiles(const std::string &directory, const std::string &extension = "");

    // Binary I/O utilities
    bool writeKeyValuePairsToFile(const std::string &filePath, const std::vector<KeyValuePair> &pairs);
    bool readKeyValuePairsFromFile(const std::string &filePath, std::vector<KeyValuePair> &pairs);
    bool loadKeyValuePairsFromBinaryFile(const std::string &filePath, std::vector<KeyValuePair> &pairs);

    // Serialization utilities
    void serializeKeyValuePair(const KeyValuePair &pair, std::ostream &out);
    KeyValuePair deserializeKeyValuePair(std::istream &in);

    // Error handling
    class LSMException : public std::exception
    {
    private:
        std::string message;

    public:
        explicit LSMException(const std::string &msg) : message(msg) {}

        const char *what() const noexcept override
        {
            return message.c_str();
        }
    };

    // Debug utilities
    void logDebug(const std::string &message);
    void logError(const std::string &message);

} // namespace lsm

#endif // LSM_UTILS_H