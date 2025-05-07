#include "utils.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <chrono>
#include <ctime>

namespace fs = std::filesystem;

namespace lsm
{

    // File path utilities
    std::string joinPath(const std::string &base, const std::string &path)
    {
        fs::path basePath(base);
        fs::path subPath(path);
        return (basePath / subPath).string();
    }

    bool fileExists(const std::string &path)
    {
        return fs::exists(path);
    }

    bool createDirectory(const std::string &path)
    {
        if (fileExists(path))
        {
            return true;
        }
        try
        {
            return fs::create_directories(path);
        }
        catch (const std::exception &e)
        {
            logError("Failed to create directory: " + std::string(e.what()));
            return false;
        }
    }

    std::vector<std::string> listFiles(const std::string &directory, const std::string &extension)
    {
        std::vector<std::string> files;

        if (!fileExists(directory))
        {
            return files;
        }

        for (const auto &entry : fs::directory_iterator(directory))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            if (extension.empty() || entry.path().extension() == extension)
            {
                files.push_back(entry.path().string());
            }
        }

        return files;
    }

    // Binary I/O utilities
    bool writeKeyValuePairsToFile(const std::string &filePath, const std::vector<KeyValuePair> &pairs)
    {
        std::ofstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file for writing: " + filePath);
            return false;
        }

        for (const auto &pair : pairs)
        {
            serializeKeyValuePair(pair, file);
        }

        return true;
    }

    bool readKeyValuePairsFromFile(const std::string &filePath, std::vector<KeyValuePair> &pairs)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open file for reading: " + filePath);
            return false;
        }

        pairs.clear();

        while (file.peek() != EOF)
        {
            pairs.push_back(deserializeKeyValuePair(file));

            if (file.fail() && !file.eof())
            {
                logError("Error reading from file: " + filePath);
                return false;
            }
        }

        return true;
    }

    bool loadKeyValuePairsFromBinaryFile(const std::string &filePath, std::vector<KeyValuePair> &pairs)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            logError("Failed to open binary file for loading: " + filePath);
            return false;
        }

        pairs.clear();

        // Read until EOF in chunks of key-value pairs
        KeyType key;
        ValueType value;

        while (file.read(reinterpret_cast<char *>(&key), sizeof(KeyType)))
        {
            if (!file.read(reinterpret_cast<char *>(&value), sizeof(ValueType)))
            {
                logError("Binary file has odd number of integers: " + filePath);
                return false;
            }

            pairs.emplace_back(key, value);
        }

        return true;
    }

    // Serialization utilities
    void serializeKeyValuePair(const KeyValuePair &pair, std::ostream &out)
    {
        out.write(reinterpret_cast<const char *>(&pair.key), sizeof(KeyType));
        out.write(reinterpret_cast<const char *>(&pair.value), sizeof(ValueType));
    }

    KeyValuePair deserializeKeyValuePair(std::istream &in)
    {
        KeyValuePair pair;
        in.read(reinterpret_cast<char *>(&pair.key), sizeof(KeyType));
        in.read(reinterpret_cast<char *>(&pair.value), sizeof(ValueType));
        return pair;
    }

    // Debug utilities
    void logDebug(const std::string &message)
    {
#ifdef DEBUG
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time);
        timestamp.pop_back(); // Remove trailing newline

        std::cout << "\033[34m[" << timestamp << "] DEBUG: " << message << "\033[0m" << std::endl;
#else
        // Suppress unused parameter warning
        (void)message;
#endif
    }

    void logError(const std::string &message)
    {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::string timestamp = std::ctime(&time);
        timestamp.pop_back(); // Remove trailing newline

        std::cerr << "[" << timestamp << "] ERROR: " << message << std::endl;
    }

} // namespace lsm