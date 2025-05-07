#include "bloom_filter.h"
#include <cmath>
#include <random>
#include <fstream>

namespace lsm
{

    // Constructor with optimal parameters
    BloomFilter::BloomFilter(size_t expectedNumElements, double falsePositiveRate)
        : numElements(0)
    {
        numBits = calculateOptimalNumBits(expectedNumElements, falsePositiveRate);
        numHashFunctions = calculateOptimalNumHashFunctions(numBits, expectedNumElements);
        bits.resize(numBits, false);
    }

    // Constructor with explicit parameters
    BloomFilter::BloomFilter(size_t numBits, size_t numHashFunctions)
        : numBits(numBits), numHashFunctions(numHashFunctions), numElements(0)
    {
        bits.resize(numBits, false);
    }

    // Add a key to the bloom filter
    void BloomFilter::add(const KeyType &key)
    {
        for (size_t i = 0; i < numHashFunctions; i++)
        {
            size_t pos = hash(key, i) % numBits;
            bits[pos] = true;
        }
        numElements++;
    }

    // Check if a key might be in the bloom filter
    bool BloomFilter::mightContain(const KeyType &key) const
    {
        if (numBits == 0 || numHashFunctions == 0)
        {
            logDebug("Bloom filter has 0 bits or 0 hash functions, returning true");
            return true;
        }

        // Check each hash function
        for (size_t i = 0; i < numHashFunctions; i++)
        {
            size_t hashValue = hash(key, i) % numBits;
            if (!bits[hashValue])
            {
                logDebug("Bloom filter: Key definitely not present (bit " + std::to_string(hashValue) + " not set)");
                return false;
            }
        }

        logDebug("Bloom filter: Key might be present (all bits set, but could be false positive)");
        return true;
    }

    // Clear the bloom filter
    void BloomFilter::clear()
    {
        std::fill(bits.begin(), bits.end(), false);
        numElements = 0;
    }

    // Get the approximate false positive rate
    double BloomFilter::getFalsePositiveRate() const
    {
        if (numElements == 0 || numBits == 0)
        {
            return 0.0;
        }

        // False positive probability = (1 - e^(-k*n/m))^k
        // where k = number of hash functions, n = number of elements, m = number of bits
        double exponent = -1.0 * (double)numHashFunctions * (double)numElements / (double)numBits;
        return std::pow(1.0 - std::exp(exponent), numHashFunctions);
    }

    // Hash function using simple multiplication and modulo with different seeds
    size_t BloomFilter::hash(const KeyType &key, size_t seed) const
    {
        // Using MurmurHash-inspired mixing
        size_t h = static_cast<size_t>(key);
        h ^= seed;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    // Serialize the bloom filter to a file
    bool BloomFilter::serialize(const std::string &filePath) const
    {
        std::ofstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file for writing bloom filter: " + filePath);
            return false;
        }

        // Write metadata
        file.write(reinterpret_cast<const char *>(&numBits), sizeof(numBits));
        file.write(reinterpret_cast<const char *>(&numHashFunctions), sizeof(numHashFunctions));
        file.write(reinterpret_cast<const char *>(&numElements), sizeof(numElements));

        // Write bit array (compacting 8 bits into a byte)
        std::vector<uint8_t> bytes((numBits + 7) / 8);
        for (size_t i = 0; i < numBits; i++)
        {
            if (bits[i])
            {
                bytes[i / 8] |= (1 << (i % 8));
            }
        }

        file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());

        return !file.fail();
    }

    // Deserialize the bloom filter from a file
    bool BloomFilter::deserialize(const std::string &filePath)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file for reading bloom filter: " + filePath);
            return false;
        }

        // Read metadata
        file.read(reinterpret_cast<char *>(&numBits), sizeof(numBits));
        file.read(reinterpret_cast<char *>(&numHashFunctions), sizeof(numHashFunctions));
        file.read(reinterpret_cast<char *>(&numElements), sizeof(numElements));

        if (file.fail())
        {
            logError("Error reading bloom filter metadata from file: " + filePath);
            return false;
        }

        // Read bit array
        bits.resize(numBits);
        std::vector<uint8_t> bytes((numBits + 7) / 8);
        file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());

        if (file.fail() && !file.eof())
        {
            logError("Error reading bloom filter data from file: " + filePath);
            return false;
        }

        // Expand bytes back to bits
        for (size_t i = 0; i < numBits; i++)
        {
            bits[i] = (bytes[i / 8] & (1 << (i % 8))) != 0;
        }

        return true;
    }

    // Calculate optimal number of bits for a given false positive rate
    size_t BloomFilter::calculateOptimalNumBits(size_t expectedNumElements, double falsePositiveRate)
    {
        // Formula: m = -n*ln(p) / (ln(2)^2)
        // where m = number of bits, n = number of elements, p = false positive rate
        if (expectedNumElements == 0 || falsePositiveRate <= 0 || falsePositiveRate >= 1)
        {
            return 0;
        }

        double ln2squared = std::log(2) * std::log(2);
        return static_cast<size_t>(-1.0 * expectedNumElements * std::log(falsePositiveRate) / ln2squared);
    }

    // Calculate optimal number of hash functions
    size_t BloomFilter::calculateOptimalNumHashFunctions(size_t numBits, size_t expectedNumElements)
    {
        // Formula: k = (m/n) * ln(2)
        // where k = number of hash functions, m = number of bits, n = number of elements
        if (numBits == 0 || expectedNumElements == 0)
        {
            return 0;
        }

        double optimalK = (static_cast<double>(numBits) / static_cast<double>(expectedNumElements)) * std::log(2);
        return static_cast<size_t>(std::max(1.0, std::round(optimalK)));
    }

} // namespace lsm