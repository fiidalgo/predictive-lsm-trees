import mmap
import struct
import numpy as np
import os
import threading
import time
import platform
import tempfile

class SharedMemoryInterface:
    def __init__(self, shm_name, size):
        self.shm_name = shm_name
        self.size = size
        self.lock = threading.Lock()
        
        # Create shared memory file in temp directory
        if platform.system() == 'Windows':
            self.shm_path = os.path.join(tempfile.gettempdir(), f"{shm_name}.shm")
        else:
            self.shm_path = f"/dev/shm/{shm_name}"
        
        # Create shared memory file if it doesn't exist
        if not os.path.exists(self.shm_path):
            with open(self.shm_path, "wb") as f:
                f.write(b"\0" * size)
        
        # Open shared memory
        self.fd = os.open(self.shm_path, os.O_RDWR)
        self.mmap = mmap.mmap(self.fd, size)
    
    def write_data(self, data_type, data):
        """Write data to shared memory with type header"""
        with self.lock:
            self.mmap.seek(0)
            # Write type header (1 byte)
            self.mmap.write(struct.pack("B", data_type))
            
            if data_type == 0:  # Training data
                keys, levels, page_ids = data
                # Write number of entries
                self.mmap.write(struct.pack("I", len(keys)))
                # Write data
                for k, l, p in zip(keys, levels, page_ids):
                    self.mmap.write(struct.pack("dii", k, l, p))
            
            elif data_type == 1:  # Bloom filter prediction request
                self.mmap.write(struct.pack("d", data))
            
            elif data_type == 2:  # Fence pointer prediction request
                key, level = data
                self.mmap.write(struct.pack("di", key, level))
    
    def read_data(self):
        """Read data from shared memory"""
        with self.lock:
            self.mmap.seek(0)
            data_type = struct.unpack("B", self.mmap.read(1))[0]
            
            if data_type == 0:  # Training data
                n_entries = struct.unpack("I", self.mmap.read(4))[0]
                keys = []
                levels = []
                page_ids = []
                for _ in range(n_entries):
                    k, l, p = struct.unpack("dii", self.mmap.read(16))
                    keys.append(k)
                    levels.append(l)
                    page_ids.append(p)
                return data_type, (keys, levels, page_ids)
            
            elif data_type == 1:  # Bloom filter prediction request
                key = struct.unpack("d", self.mmap.read(8))[0]
                return data_type, key
            
            elif data_type == 2:  # Fence pointer prediction request
                key, level = struct.unpack("di", self.mmap.read(12))
                return data_type, (key, level)
    
    def write_response(self, data_type, response):
        """Write prediction response to shared memory"""
        with self.lock:
            self.mmap.seek(0)
            self.mmap.write(struct.pack("B", data_type))
            
            if data_type == 1:  # Bloom filter prediction
                self.mmap.write(struct.pack("f", response))
            
            elif data_type == 2:  # Fence pointer prediction
                self.mmap.write(struct.pack("i", response))
            
            elif data_type == 3:  # Model ready signal
                pass  # No additional data needed
    
    def close(self):
        """Clean up shared memory"""
        self.mmap.close()
        os.close(self.fd)
        try:
            os.unlink(self.shm_path)
        except:
            pass 