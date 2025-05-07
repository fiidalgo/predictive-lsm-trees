#include "utils.h"
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

using namespace lsm;

void printUsage(const char *programName)
{
    std::cout << "Usage: " << programName << " <output_file> <num_pairs> [min_key] [max_key]\n"
              << "  output_file - Path to the output binary file\n"
              << "  num_pairs   - Number of key-value pairs to generate\n"
              << "  min_key     - Minimum key value (default: 1)\n"
              << "  max_key     - Maximum key value (default: num_pairs*2)\n";
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printUsage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string outputFile = argv[1];
    int numPairs = std::stoi(argv[2]);
    int minKey = (argc > 3) ? std::stoi(argv[3]) : 1;
    int maxKey = (argc > 4) ? std::stoi(argv[4]) : numPairs * 2;

    if (numPairs <= 0)
    {
        std::cerr << "Number of pairs must be positive\n";
        return 1;
    }

    if (minKey >= maxKey)
    {
        std::cerr << "Min key must be less than max key\n";
        return 1;
    }

    // Create output directory if it doesn't exist
    fs::path outputPath(outputFile);
    fs::path outputDir = outputPath.parent_path();
    if (!outputDir.empty() && !fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }

    // Generate random key-value pairs
    std::vector<KeyValuePair> pairs;
    pairs.reserve(numPairs);

    // Use seed based on current time
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> keyDist(minKey, maxKey);
    std::uniform_int_distribution<int> valDist(1, 1000000);

    // Generate unique keys
    std::vector<int> keys;
    keys.reserve(maxKey - minKey + 1);
    for (int i = minKey; i <= maxKey; ++i)
    {
        keys.push_back(i);
    }

    // Shuffle the keys
    std::shuffle(keys.begin(), keys.end(), gen);

    // Take the first numPairs keys
    for (int i = 0; i < numPairs; ++i)
    {
        int key = keys[i];
        int value = valDist(gen);
        pairs.push_back(KeyValuePair(key, value));
    }

    // Sort pairs by key (optional)
    std::sort(pairs.begin(), pairs.end());

    // Write pairs to binary file
    bool success = writeKeyValuePairsToFile(outputFile, pairs);
    if (!success)
    {
        std::cerr << "Failed to write pairs to file\n";
        return 1;
    }

    std::cout << "Successfully generated " << numPairs << " key-value pairs and wrote them to " << outputFile << "\n";
    return 0;
}