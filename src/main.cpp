#include "lsm_tree.h"
#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>

using namespace lsm;
namespace fs = std::filesystem;

void printUsage()
{
    std::cout << "Usage:\n"
              << "  p <key> <value>     - Put a key-value pair\n"
              << "  g <key>             - Get the value for a key\n"
              << "  d <key>             - Delete the value for a key\n"
              << "  r <start_key> <end_key> - Range query for keys in [start_key, end_key]\n"
              << "  l <file_path>       - Load key-value pairs from a binary file\n"
              << "  f                   - Force buffer flush\n"
              << "  s                   - Print statistics\n"
              << "  h                   - Print this help message\n"
              << "  q                   - Quit\n";
}

int main(int argc, char *argv[])
{
    // Suppress unused parameter warnings
    (void)argc;
    (void)argv;

    // Create data directory if it doesn't exist
    std::string dataDir = "data";
    if (!fs::exists(dataDir))
    {
        fs::create_directories(dataDir);
    }

    // Create LSM tree
    LSMTree lsmTree(dataDir);

    std::cout << "LSM Tree initialized. Type 'h' for help.\n";

    std::string line;
    while (true)
    {
        std::cout << "> ";
        std::getline(std::cin, line);

        if (line.empty())
        {
            continue;
        }

        std::istringstream iss(line);
        std::string command;
        iss >> command;

        if (command == "p" || command == "put")
        {
            // Put command
            KeyType key;
            ValueType value;
            if (iss >> key >> value)
            {
                auto start = std::chrono::high_resolution_clock::now();
                bool success = lsmTree.put(key, value);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (success)
                {
                    std::cout << "Put successful. (" << duration << " μs)" << std::endl;
                }
                else
                {
                    std::cout << "Put failed." << std::endl;
                }
            }
            else
            {
                std::cout << "Invalid put command. Usage: p <key> <value>" << std::endl;
            }
        }
        else if (command == "g" || command == "get")
        {
            // Get command
            KeyType key;
            if (iss >> key)
            {
                ValueType value;
                auto start = std::chrono::high_resolution_clock::now();
                bool found = lsmTree.get(key, value);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (found)
                {
                    std::cout << "Value: " << value << " (" << duration << " μs)" << std::endl;
                }
                else
                {
                    std::cout << "Key not found. (" << duration << " μs)" << std::endl;
                }
            }
            else
            {
                std::cout << "Invalid get command. Usage: g <key>" << std::endl;
            }
        }
        else if (command == "d" || command == "delete")
        {
            // Delete command
            KeyType key;
            if (iss >> key)
            {
                auto start = std::chrono::high_resolution_clock::now();
                bool success = lsmTree.remove(key);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (success)
                {
                    std::cout << "Delete successful. (" << duration << " μs)" << std::endl;
                }
                else
                {
                    std::cout << "Delete failed." << std::endl;
                }
            }
            else
            {
                std::cout << "Invalid delete command. Usage: d <key>" << std::endl;
            }
        }
        else if (command == "r" || command == "range")
        {
            // Range command
            KeyType startKey, endKey;
            if (iss >> startKey >> endKey)
            {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<KeyValuePair> results = lsmTree.range(startKey, endKey);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                std::cout << "Range query results (" << results.size() << " entries, " << duration << " μs):" << std::endl;
                for (const auto &pair : results)
                {
                    std::cout << pair.key << ": " << pair.value << std::endl;
                }
            }
            else
            {
                std::cout << "Invalid range command. Usage: r <start_key> <end_key>" << std::endl;
            }
        }
        else if (command == "l" || command == "load")
        {
            // Load command
            std::string filePath;
            if (iss >> filePath)
            {
                auto start = std::chrono::high_resolution_clock::now();
                bool success = lsmTree.loadFromFile(filePath);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (success)
                {
                    std::cout << "Load successful. (" << duration << " μs)" << std::endl;
                }
                else
                {
                    std::cout << "Load failed. Make sure the file exists and is in the correct format." << std::endl;
                }
            }
            else
            {
                std::cout << "Invalid load command. Usage: l <file_path>" << std::endl;
            }
        }
        else if (command == "f" || command == "flush")
        {
            // Force flush command
            auto start = std::chrono::high_resolution_clock::now();
            lsmTree.flushIfNecessary();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "Forced buffer flush. (" << duration << " μs)" << std::endl;
        }
        else if (command == "s" || command == "stats")
        {
            // Statistics command
            std::cout << lsmTree.getStats() << std::endl;
        }
        else if (command == "h" || command == "help")
        {
            // Help command
            printUsage();
        }
        else if (command == "q" || command == "quit" || command == "exit")
        {
            // Quit command
            break;
        }
        else
        {
            std::cout << "Unknown command. Type 'h' for help." << std::endl;
        }
    }

    return 0;
}