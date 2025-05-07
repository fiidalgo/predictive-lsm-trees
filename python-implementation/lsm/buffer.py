from typing import Dict, List, Tuple, Optional
import bisect
import logging
from .utils import Constants

logger = logging.getLogger(__name__)

class MemoryBuffer:
    """In-memory buffer for LSM tree that maintains sorted key-value pairs."""
    
    def __init__(self, size_bytes: int = Constants.DEFAULT_BUFFER_SIZE):
        self.size_bytes = size_bytes
        self.current_size = 0
        self.buffer: Dict[bytes, bytes] = {}  # Using bytes for fixed-size keys and values
        
    def put(self, key: bytes, value: bytes) -> bool:
        """Insert a key-value pair into the buffer."""
        # Ensure key and value sizes match the expected fixed sizes
        if len(key) != Constants.KEY_SIZE_BYTES:
            logger.warning(f"Key size mismatch: expected {Constants.KEY_SIZE_BYTES}, got {len(key)}")
            return False
        
        if len(value) != Constants.VALUE_SIZE_BYTES:
            logger.warning(f"Value size mismatch: expected {Constants.VALUE_SIZE_BYTES}, got {len(value)}")
            return False
        
        # Check if we have space (each entry is a fixed size)
        entry_size = Constants.ENTRY_SIZE_BYTES
        if key not in self.buffer and self.current_size + entry_size > self.size_bytes:
            logger.debug(f"Buffer full: {self.current_size}/{self.size_bytes} bytes")
            return False
            
        # If key exists, we're just replacing the value (no size change)
        # If key is new, increment size by the full entry size
        if key not in self.buffer:
            self.current_size += entry_size
            
        self.buffer[key] = value
        logger.debug(f"Added key-value pair. New size: {self.current_size}/{self.size_bytes} bytes")
        return True
        
    def get(self, key: bytes) -> Optional[bytes]:
        """Retrieve a value by key."""
        return self.buffer.get(key)
        
    def range(self, start_key: bytes, end_key: bytes) -> List[Tuple[bytes, bytes]]:
        """Get all key-value pairs within the given range."""
        result = []
        for k in sorted(self.buffer.keys()):
            if start_key <= k <= end_key:
                result.append((k, self.buffer[k]))
        return result
        
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
        
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.current_size >= self.size_bytes
        
    def get_size(self) -> int:
        """Get number of entries in buffer."""
        return len(self.buffer)
        
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return self.current_size
        
    def get_capacity(self) -> int:
        """Get buffer capacity in bytes."""
        return self.size_bytes
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.current_size = 0
        logger.debug("Buffer cleared")
        
    def get_sorted_pairs(self) -> List[Tuple[bytes, bytes]]:
        """Get all key-value pairs in sorted order."""
        return [(k, self.buffer[k]) for k in sorted(self.buffer.keys())]
        
    def remove(self, key: bytes) -> bool:
        """Remove a key-value pair from the buffer."""
        if key in self.buffer:
            self.current_size -= Constants.ENTRY_SIZE_BYTES
            del self.buffer[key]
            return True
        return False 