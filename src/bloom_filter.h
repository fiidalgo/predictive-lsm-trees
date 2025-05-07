#ifndef LSM_BLOOM_FILTER_H
#define LSM_BLOOM_FILTER_H

#include "utils.h"
#include <vector>
#include <cmath>
#include <bitset>
#include <fstream>
#include <functional>

namespace lsm
{

    class BloomFilter
    {
    private:
        std::vector<bool> bits;
        size_t numBits;
        size_t numHashFunctions;
        size_t numElements;

        // Hash function using simple multiplication and modulo
        size_t hash(const KeyType &key, size_t seed) const;

    public:
        // Constructor with optimal bits per entry
        BloomFilter(size_t expectedNumElements, double falsePositiveRate = 0.01);

        // Constructor with explicit parameters
        BloomFilter(size_t numBits, size_t numHashFunctions);

        // Default constructor for empty bloom filter
        BloomFilter() : numBits(0), numHashFunctions(0), numElements(0) {}

        // Add a key to the bloom filter
        void add(const KeyType &key);

        // Check if a key might be in the bloom filter
        // Returns false if key is definitely not in the set
        // Returns true if key might be in the set (could be false positive)
        bool mightContain(const KeyType &key) const;

        // Clear the bloom filter
        void clear();

        // Get the approximate false positive rate
        double getFalsePositiveRate() const;

        // Get the number of bits used
        size_t getNumBits() const { return numBits; }

        // Get the number of hash functions used
        size_t getNumHashFunctions() const { return numHashFunctions; }

        // Get the number of elements added
        size_t getNumElements() const { return numElements; }

        // Serialize the bloom filter to a file
        bool serialize(const std::string &filePath) const;

        // Deserialize the bloom filter from a file
        bool deserialize(const std::string &filePath);

        // Static utility function to calculate optimal number of bits
        static size_t calculateOptimalNumBits(size_t expectedNumElements, double falsePositiveRate);

        // Static utility function to calculate optimal number of hash functions
        static size_t calculateOptimalNumHashFunctions(size_t numBits, size_t expectedNumElements);
    };

} // namespace lsm

#endif // LSM_BLOOM_FILTER_H