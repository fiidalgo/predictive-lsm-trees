#include "src/utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace lsm;

// Helper function to format large numbers with commas
std::string format_with_commas(size_t value)
{
    std::stringstream ss;
    ss << std::fixed << value;
    std::string str = ss.str();

    int insertPosition = str.length() - 3;
    while (insertPosition > 0)
    {
        str.insert(insertPosition, ",");
        insertPosition -= 3;
    }
    return str;
}

// Calculate memory usage for the generated entries
size_t calculate_memory_usage(size_t num_entries)
{
    // Each entry is 8 bytes (4 for key, 4 for value)
    size_t bytes = num_entries * 8;

    // Convert to MB for easier reading
    double mb = bytes / (1024.0 * 1024.0);

    std::cout << "Memory usage: ~" << mb << " MB ("
              << format_with_commas(bytes) << " bytes)" << std::endl;

    return bytes;
}

int main(int argc, char *argv[])
{
    // Default values - much larger to demonstrate multiple levels
    size_t num_entries = 2000000; // Generate 2M entries by default
    std::string output_file = "data/test_data.bin";
    int key_range = 20000000; // Wider key range to reduce duplicates

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--count" && i + 1 < argc)
        {
            num_entries = std::stoul(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (arg == "--key-range" && i + 1 < argc)
        {
            key_range = std::stoi(argv[++i]);
        }
        else if (arg == "--help")
        {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --count N       Generate N key-value pairs (default: 2,000,000)\n"
                      << "  --key-range N   Generate keys in range 1-N (default: 20,000,000)\n"
                      << "  --output FILE   Output to FILE (default: data/test_data.bin)\n"
                      << "  --help          Display this help and exit\n";
            return 0;
        }
    }

    // Calculate and display memory usage
    calculate_memory_usage(num_entries);

    // Generate random key-value pairs
    std::cout << "Generating " << format_with_commas(num_entries) << " key-value pairs...\n";

    // Use a time-based seed for random number generation
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    // Distribution for keys (1 to key_range)
    std::uniform_int_distribution<KeyType> key_dist(1, key_range);

    // Distribution for values (1 to 1,000)
    std::uniform_int_distribution<ValueType> value_dist(1, 1000);

    // Generate pairs in batches to avoid memory issues
    std::vector<KeyValuePair> pairs;
    size_t batch_size = 100000; // Process 100K at a time
    size_t total_unique = 0;
    size_t total_batches = (num_entries + batch_size - 1) / batch_size;

    std::ofstream out_file(output_file, std::ios::binary);
    if (!out_file)
    {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
    }

    for (size_t batch = 0; batch < total_batches; batch++)
    {
        std::cout << "Processing batch " << (batch + 1) << " of " << total_batches << "...\n";

        size_t current_batch_size = std::min(batch_size, num_entries - batch * batch_size);
        pairs.clear();
        pairs.reserve(current_batch_size);

        // Generate batch of pairs
        for (size_t i = 0; i < current_batch_size; ++i)
        {
            KeyType key = key_dist(generator);
            ValueType value = value_dist(generator);
            pairs.push_back(KeyValuePair(key, value));
        }

        // Sort pairs by key
        std::sort(pairs.begin(), pairs.end());

        // Remove duplicates (keep the last occurrence of each key)
        auto last = std::unique(pairs.begin(), pairs.end(),
                                [](const KeyValuePair &a, const KeyValuePair &b)
                                {
                                    return a.key == b.key;
                                });

        pairs.erase(last, pairs.end());
        total_unique += pairs.size();

        // Write batch to file
        for (const auto &pair : pairs)
        {
            serializeKeyValuePair(pair, out_file);
        }
    }

    out_file.close();

    std::cout << "Successfully wrote " << format_with_commas(total_unique)
              << " unique key-value pairs to " << output_file << "\n";
    std::cout << "You can load this data into the LSM tree using the 'l " << output_file << "' command.\n";

    return 0;
}