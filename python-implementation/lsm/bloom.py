import math
import mmh3  # MurmurHash3 for better hash distribution
from typing import List

class BloomFilter:
    """Bloom filter implementation for probabilistic membership testing."""
    
    def __init__(self, expected_elements: int, false_positive_rate: float):
        """Initialize Bloom filter with desired false positive rate."""
        self.false_positive_rate = false_positive_rate
        self.expected_elements = expected_elements
        
        # Calculate optimal number of bits and hash functions
        self.size = self.get_size(expected_elements, false_positive_rate)
        self.hash_count = self.get_hash_count(self.size, expected_elements)
        
        # Initialize bit array
        self.bit_array = [False] * self.size
        
    @staticmethod
    def get_size(n: int, p: float) -> int:
        """Calculate optimal size of bit array."""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return math.ceil(m)
        
    @staticmethod
    def get_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        k = (m / n) * math.log(2)
        return math.ceil(k)
        
    def _get_hash_values(self, item: float) -> List[int]:
        """Generate hash values for an item."""
        hash_values = []
        for seed in range(self.hash_count):
            hash_val = mmh3.hash(str(item).encode(), seed) % self.size
            hash_values.append(hash_val)
        return hash_values
        
    def add(self, item: float) -> None:
        """Add an item to the Bloom filter."""
        for bit_pos in self._get_hash_values(item):
            self.bit_array[bit_pos] = True
            
    def might_contain(self, item: float) -> bool:
        """Check if an item might be in the set."""
        for bit_pos in self._get_hash_values(item):
            if not self.bit_array[bit_pos]:
                return False
        return True
        
    def get_size_bytes(self) -> int:
        """Get size of Bloom filter in bytes."""
        return math.ceil(self.size / 8)  # Convert bits to bytes
        
    def get_false_positive_rate(self) -> float:
        """Get the current false positive rate."""
        return self.false_positive_rate 