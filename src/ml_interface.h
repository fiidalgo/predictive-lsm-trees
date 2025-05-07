#ifndef LSM_ML_INTERFACE_H
#define LSM_ML_INTERFACE_H

#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <vector>

namespace lsm {

class MLInterface {
private:
    int fd;
    void* shm_ptr;
    size_t shm_size;
    std::string shm_path;

    // Helper function to get shared memory path
    std::string get_shm_path(const std::string& name) {
        #ifdef _WIN32
            return std::filesystem::temp_directory_path().string() + "/" + name + ".shm";
        #else
            return "/dev/shm/" + name;
        #endif
    }

    // Helper function to open shared memory
    void open_shm(const std::string& name, size_t size) {
        shm_path = get_shm_path(name);
        shm_size = size;

        // Create shared memory object
        #ifdef _WIN32
            fd = open(shm_path.c_str(), O_CREAT | O_RDWR, 0666);
        #else
            fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        #endif

        if (fd == -1) {
            throw std::runtime_error("Failed to create shared memory: " + std::string(strerror(errno)));
        }

        // Set size
        if (ftruncate(fd, size) == -1) {
            close(fd);
            throw std::runtime_error("Failed to set shared memory size: " + std::string(strerror(errno)));
        }

        // Map shared memory
        shm_ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (shm_ptr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map shared memory: " + std::string(strerror(errno)));
        }
    }

public:
    MLInterface(const std::string& name, size_t size) {
        open_shm(name, size);
    }

    ~MLInterface() {
        if (shm_ptr != nullptr) {
            munmap(shm_ptr, shm_size);
        }
        if (fd != -1) {
            close(fd);
        }
        #ifndef _WIN32
            shm_unlink(shm_path.c_str());
        #endif
    }

    // Write training data
    void write_training_data(const std::vector<double>& keys,
                           const std::vector<int>& levels,
                           const std::vector<int>& page_ids) {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        
        // Write type (0 for training data)
        *ptr = 0;
        ptr += 1;

        // Write number of entries
        uint32_t n_entries = keys.size();
        memcpy(ptr, &n_entries, sizeof(n_entries));
        ptr += sizeof(n_entries);

        // Write data
        for (size_t i = 0; i < n_entries; i++) {
            memcpy(ptr, &keys[i], sizeof(double));
            ptr += sizeof(double);
            memcpy(ptr, &levels[i], sizeof(int));
            ptr += sizeof(int);
            memcpy(ptr, &page_ids[i], sizeof(int));
            ptr += sizeof(int);
        }
    }

    // Write bloom filter prediction request
    void write_bloom_request(double key) {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        
        // Write type (1 for bloom filter request)
        *ptr = 1;
        ptr += 1;

        // Write key
        memcpy(ptr, &key, sizeof(double));
    }

    // Write fence pointer prediction request
    void write_fence_request(double key, int level) {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        
        // Write type (2 for fence pointer request)
        *ptr = 2;
        ptr += 1;

        // Write key and level
        memcpy(ptr, &key, sizeof(double));
        ptr += sizeof(double);
        memcpy(ptr, &level, sizeof(int));
    }

    // Read bloom filter prediction response
    float read_bloom_response() {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        
        // Skip type
        ptr += 1;

        // Read probability
        float prob;
        memcpy(&prob, ptr, sizeof(float));
        return prob;
    }

    // Read fence pointer prediction response
    int read_fence_response() {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        
        // Skip type
        ptr += 1;

        // Read page id
        int page_id;
        memcpy(&page_id, ptr, sizeof(int));
        return page_id;
    }

    // Check if models are ready for prediction
    bool are_models_ready() {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        // Check if models are initialized (type 3)
        return *ptr == 3;
    }

    // Write model ready signal
    void write_models_ready() {
        uint8_t* ptr = static_cast<uint8_t*>(shm_ptr);
        *ptr = 3;  // Type 3 indicates models are ready
    }
};

} // namespace lsm

#endif // LSM_ML_INTERFACE_H 