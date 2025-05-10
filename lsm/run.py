import os
import pickle
import struct
import logging
import math
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from .bloom import BloomFilter
from .utils import Constants, ensure_directory, merge_sorted_pairs, is_tombstone

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Change from INFO to WARNING
logger = logging.getLogger(__name__)

# Helper function for key conversion
def bytes_to_float(key: bytes) -> Union[float, bytes]:
    """Convert bytes to float if possible."""
    if not isinstance(key, bytes):
        return key
    try:
        # Make sure we have at least 8 bytes, pad if needed
        bs_padded = key[:8].ljust(8, b'\x00')
        return struct.unpack('!d', bs_padded)[0]
    except (struct.error, ValueError, TypeError):
        # Fall back to original value if unpacking fails
        logger.warning(f"Failed to unpack bytes to float: {key}")
        return key

class Run:
    """Represents a sorted string table (SSTable) in the LSM tree."""
    
    def __init__(self, file_path: str, level: int, bloom_fpr: float = None):
        logger.info(f"Initializing Run at {file_path} for level {level}")
        self.file_path = file_path
        self.level = level
        self.entry_count = 0
        self.page_size = Constants.DEFAULT_PAGE_SIZE  # bytes
        self.num_pages = 0  # Initialize num_pages
        self.min_key = None
        self.max_key = None
        
        # Use Monkey paper FPR scaling if no specific FPR is provided
        if bloom_fpr is None:
            self.bloom_fpr = Constants.get_level_fpr(level)
        else:
            self.bloom_fpr = bloom_fpr
            
        logger.info(f"Using Bloom filter FPR of {self.bloom_fpr} for level {level}")
        
        self.bloom_filter = None  # Will be initialized in create_from_pairs
        self.page_min_keys = []  # Min key in each page (fence pointers)
        self.page_max_keys = []  # Max key in each page (fence pointers)
        logger.info("Run initialization complete")
        
    def create_from_pairs(self, pairs: List[Tuple[bytes, bytes]]) -> None:
        """Create a new run from sorted key-value pairs."""
        logger.info(f"Creating new run from {len(pairs)} pairs")
        self.entry_count = len(pairs)
        
        # Calculate number of pages based on fixed-size entries
        entries_per_page = self.page_size // Constants.ENTRY_SIZE_BYTES
        self.num_pages = (self.entry_count + entries_per_page - 1) // entries_per_page
        
        # Store min and max keys
        if pairs:
            self.min_key = pairs[0][0]
            self.max_key = pairs[-1][0]
            
            # Create bloom filter with level-appropriate FPR
            self.bloom_filter = BloomFilter(expected_elements=len(pairs), false_positive_rate=self.bloom_fpr)
            for key, _ in pairs:
                self.bloom_filter.add(key)
            
            # Calculate fence pointers (min/max keys per page)
            self.page_min_keys = []
            self.page_max_keys = []
            
            for page_id in range(self.num_pages):
                start_idx = page_id * entries_per_page
                end_idx = min(len(pairs), start_idx + entries_per_page)
                
                if start_idx < len(pairs):
                    self.page_min_keys.append(pairs[start_idx][0])
                else:
                    self.page_min_keys.append(b'\xff' * Constants.KEY_SIZE_BYTES)  # Max key value
                    
                if end_idx > 0 and end_idx - 1 < len(pairs):
                    self.page_max_keys.append(pairs[end_idx - 1][0])
                else:
                    self.page_max_keys.append(b'\x00' * Constants.KEY_SIZE_BYTES)  # Min key value
            
        # Write pairs to disk
        with open(self.file_path, 'wb') as f:
            pickle.dump({
                'pairs': pairs,
                'level': self.level,
                'num_pages': self.num_pages,
                'min_key': self.min_key,
                'max_key': self.max_key,
                'bloom_filter': self.bloom_filter,
                'page_min_keys': self.page_min_keys,
                'page_max_keys': self.page_max_keys,
                'bloom_fpr': self.bloom_fpr  # Store FPR for reference
            }, f)
        logger.info(f"Run creation complete with {self.num_pages} pages and Bloom filter FPR: {self.bloom_fpr}")
            
    def load(self) -> None:
        """Load run metadata from disk."""
        logger.info(f"Loading run from {self.file_path}")
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self.level = data['level']
                self.entry_count = len(data['pairs'])
                self.num_pages = data.get('num_pages', 0)
                self.min_key = data.get('min_key', None)
                self.max_key = data.get('max_key', None)
                self.bloom_filter = data.get('bloom_filter', None)
                self.page_min_keys = data.get('page_min_keys', [])
                self.page_max_keys = data.get('page_max_keys', [])
                self.bloom_fpr = data.get('bloom_fpr', Constants.get_level_fpr(self.level))
                
                # If min/max keys aren't stored, compute them
                if self.min_key is None and data['pairs']:
                    self.min_key = data['pairs'][0][0]
                if self.max_key is None and data['pairs']:
                    self.max_key = data['pairs'][-1][0]
                
                # Convert stored bytes to floats
                # Convert min_key and max_key to floats if they're bytes
                self.min_key = bytes_to_float(self.min_key)
                self.max_key = bytes_to_float(self.max_key)

                # Convert fence pointer arrays
                self.page_min_keys = [bytes_to_float(k) for k in self.page_min_keys]
                self.page_max_keys = [bytes_to_float(k) for k in self.page_max_keys]
                
            logger.info(f"Run loaded successfully with {self.entry_count} entries, {self.num_pages} pages, and Bloom filter FPR: {self.bloom_fpr}")
        except Exception as e:
            logger.error(f"Error loading run: {e}")
            raise
        
    def might_contain(self, key, ml_models=None, bloom_thresh=0.5) -> bool:
        """Check if the run might contain the key using bloom filter.
        
        This is a probabilistic check that allows us to skip runs when the model
        is confident the key doesn't exist in this run.
        
        Parameters:
        -----------
        key : float
            The key to check
        ml_models : LSMMLModels, optional
            ML models for prediction
        bloom_thresh : float, optional
            Threshold for bloom filter predictions (default: 0.5)
            
        Returns:
        --------
        bool
            True if the run might contain the key, False if definitely does not
        """
        # Convert key to float if needed
        key_float = bytes_to_float(key) if isinstance(key, bytes) else key
        
        # First, do a basic range check
        if self.min_key is None or self.max_key is None:
            # If we don't have min/max keys, assume it might contain the key
            return True
            
        if key_float < self.min_key or key_float > self.max_key:
            return False
        
        # If ML models are provided, use the learned model
        if ml_models is not None:
            try:
                # Use adaptive thresholds based on level - deeper levels can be more aggressive
                # since false negatives at deep levels have less impact (fewer total keys will match)
                adaptive_thresh = bloom_thresh
                if hasattr(self, 'level'):
                    # Lower threshold for deeper levels (more aggressive filtering)
                    # Level 0 has highest threshold (least aggressive)
                    adaptive_thresh = max(0.1, bloom_thresh - (0.05 * self.level))
                
                # Get prediction from model (0.0 to 1.0)
                prediction = ml_models.predict_bloom(key_float)
                
                # Use the adaptive threshold - lower values will be more aggressive
                # about filtering out runs (but risk more false negatives)
                return prediction >= adaptive_thresh
            except Exception as e:
                # On any error, default to True for safety
                logger.warning(f"Bloom filter prediction failed: {e}")
                return True
        
        # Otherwise, use traditional bloom filter if available
        if self.bloom_filter is not None:
            return self.bloom_filter.might_contain(key)
            
        # If no bloom filter available, assume it might contain the key
        return True
        
    def get(self, key) -> Optional[str]:
        """Get value for a key."""
        # Make sure key is in the right format
        original_key = key
        key_float = bytes_to_float(key) if isinstance(key, bytes) else key
        
        # Check bloom filter first (optimization)
        if self.bloom_filter is not None and not self.bloom_filter.might_contain(original_key):
            return None
            
        # If we have fence pointers, use them to find the right page
        if self.page_min_keys and hasattr(self, 'num_pages') and self.num_pages > 1:
            page_id = self._find_page_with_fence_pointers(key_float)
            if page_id is not None:
                return self.get_from_page(key_float, page_id)
            
        # Fallback to traditional binary search
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            pairs = data['pairs']
            
            # Binary search for the key
            left, right = 0, len(pairs) - 1
            while left <= right:
                mid = (left + right) // 2
                mid_key, mid_value = pairs[mid]
                
                # Convert mid_key to float if needed
                mid_key_float = bytes_to_float(mid_key) if isinstance(mid_key, bytes) else mid_key
                
                # Handle the case of string or incomparable types
                if isinstance(mid_key_float, bytes) or isinstance(key_float, bytes):
                    # If at least one is bytes, do bytes comparison if possible
                    if isinstance(mid_key, bytes) and isinstance(original_key, bytes):
                        if mid_key == original_key:
                            return mid_value
                        elif mid_key < original_key:
                            left = mid + 1
                        else:
                            right = mid - 1
                        continue
                    else:
                        # Can't compare different types, move through the list
                        left = mid + 1
                        continue
                
                # Both are comparable types (floats)
                if mid_key_float == key_float:
                    return mid_value
                elif mid_key_float < key_float:
                    left = mid + 1
                else:
                    right = mid - 1
                    
        return None
        
    def range(self, start_key, end_key) -> List[Tuple[float, str]]:
        """Get all key-value pairs in the given range."""
        result = []
        
        # Convert keys to float if needed
        start_key_float = bytes_to_float(start_key) if isinstance(start_key, bytes) else start_key
        end_key_float = bytes_to_float(end_key) if isinstance(end_key, bytes) else end_key
        
        # Use fence pointers to find the start and end pages
        start_page = None
        end_page = None
        
        if self.page_min_keys and self.page_max_keys:
            # Find start page - fence pointers are already converted to float in load()
            for i, min_key in enumerate(self.page_min_keys):
                max_key = self.page_max_keys[i]
                
                # Skip if we can't compare
                if (isinstance(min_key, bytes) and not isinstance(start_key_float, bytes)) or \
                   (not isinstance(min_key, bytes) and isinstance(start_key_float, bytes)):
                    continue
                
                if min_key <= start_key_float and max_key >= start_key_float:
                    start_page = i
                    break
                if min_key > start_key_float:
                    start_page = i
                    break
            
            # Find end page
            for i, max_key in enumerate(self.page_max_keys):
                # Skip if we can't compare
                if (isinstance(max_key, bytes) and not isinstance(end_key_float, bytes)) or \
                   (not isinstance(max_key, bytes) and isinstance(end_key_float, bytes)):
                    continue
                        
                if max_key >= end_key_float:
                    end_page = i
                    break
            
            # If we found valid pages, use them
            if start_page is not None and end_page is not None:
                for page_id in range(start_page, end_page + 1):
                    page_results = self.get_range_from_page(start_key_float, end_key_float, page_id)
                    result.extend(page_results)
                return result
        
        # Fallback to scanning the whole file
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            pairs = data['pairs']
            
            # Scan through relevant pairs
            for pair in pairs:
                key, value = pair
                
                # Convert key to float if needed for comparison
                key_float = bytes_to_float(key) if isinstance(key, bytes) else key
                
                # Handle the case of incomparable types
                if isinstance(key_float, bytes) or isinstance(start_key_float, bytes) or isinstance(end_key_float, bytes):
                    # Do bytes comparison if possible
                    if isinstance(key, bytes) and isinstance(start_key, bytes) and isinstance(end_key, bytes):
                        if start_key <= key <= end_key:
                            result.append((key, value))
                    continue
                
                # Both are comparable types (floats)
                if start_key_float <= key_float <= end_key_float:
                    result.append((key, value))
                elif key_float > end_key_float:
                    break
                    
        return result
        
    def get_size_bytes(self) -> int:
        """Get size of run in bytes."""
        return os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0 

    def get_from_page(self, key, page_id: int) -> Optional[str]:
        """Get value for a key from a specific page. Optimized for minimal disk I/O."""
        # Input validation
        if page_id < 0 or (hasattr(self, 'num_pages') and page_id >= self.num_pages):
            return None
            
        # Use fence pointers if available to quickly reject keys outside page range
        if self.page_min_keys and self.page_max_keys and len(self.page_min_keys) > page_id and len(self.page_max_keys) > page_id:
            min_key = self.page_min_keys[page_id]
            max_key = self.page_max_keys[page_id]
            
            # Handle incomparable types
            if (isinstance(min_key, bytes) and not isinstance(key, bytes)) or \
               (not isinstance(min_key, bytes) and isinstance(key, bytes)) or \
               (isinstance(max_key, bytes) and not isinstance(key, bytes)) or \
               (not isinstance(max_key, bytes) and isinstance(key, bytes)):
                # Can't determine if key is in range, so continue
                pass
            else:
                # Compare when types match
                if key < min_key or key > max_key:
                    return None
        
        # Try to access page from cache first if available
        pairs = None
        if hasattr(self, 'page_cache'):
            pairs = self.page_cache.get(self.file_path, page_id)
        
        # If not in cache, load from disk
        if pairs is None:
            try:
                with open(self.file_path, 'rb') as f:
                    data = pickle.load(f)
                    all_pairs = data['pairs']
                    
                    # Calculate page boundaries
                    entries_per_page = self.page_size // Constants.ENTRY_SIZE_BYTES
                    start_idx = page_id * entries_per_page
                    end_idx = min(len(all_pairs), start_idx + entries_per_page)
                    
                    # Get just this page's data
                    pairs = all_pairs[start_idx:end_idx]
                    
                    # Cache the page for future access
                    if hasattr(self, 'page_cache'):
                        self.page_cache.put(self.file_path, page_id, pairs)
            except Exception as e:
                # Handle file access errors
                logger.error(f"Error loading page: {e}")
                return None
        
        # Binary search within the page
        if not pairs:
            return None
        
        # Fast path for small pages (linear scan)
        if len(pairs) <= 8:
            for k, v in pairs:
                # Convert stored key to float if needed
                k_float = bytes_to_float(k) if isinstance(k, bytes) else k
                
                # If they're both bytes, compare as bytes
                if isinstance(k, bytes) and isinstance(key, bytes):
                    if k == key:
                        return v
                    continue
                
                # If they're both floats (or float-like), compare as floats
                if not isinstance(k_float, bytes) and not isinstance(key, bytes):
                    if k_float == key:
                        return v
                # Skip if types don't match
            return None
        
        # Binary search for larger pages
        left, right = 0, len(pairs) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_key, mid_value = pairs[mid]
            
            # Convert mid_key to float if needed
            mid_key_float = bytes_to_float(mid_key) if isinstance(mid_key, bytes) else mid_key
            
            # Handle the case of incomparable types
            if isinstance(mid_key_float, bytes) or isinstance(key, bytes):
                if isinstance(mid_key, bytes) and isinstance(key, bytes):
                    # Both are bytes, compare directly
                    if mid_key == key:
                        return mid_value
                    elif mid_key < key:
                        left = mid + 1
                    else:
                        right = mid - 1
                else:
                    # Can't compare different types, approximate
                    left = mid + 1
                continue
            
            # Both are floats
            if mid_key_float == key:
                return mid_value
            elif mid_key_float < key:
                left = mid + 1
            else:
                right = mid - 1
            
        return None
        
    def get_range_from_page(self, start_key, end_key, page_id: int) -> List[Tuple[float, str]]:
        """Get all key-value pairs in the given range from a specific page."""
        result = []
        
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            pairs = data['pairs']
            
            # Calculate page boundaries
            entries_per_page = self.page_size // Constants.ENTRY_SIZE_BYTES
            start_idx = page_id * entries_per_page
            end_idx = min(len(pairs), start_idx + entries_per_page)
            
            # Scan through the page
            for i in range(start_idx, end_idx):
                if i >= len(pairs):
                    break
                key, value = pairs[i]
                key_float = bytes_to_float(key) if isinstance(key, bytes) else key
                
                # Handle incomparable types
                if isinstance(key_float, bytes) or isinstance(start_key, bytes) or isinstance(end_key, bytes):
                    # If all are bytes, compare as bytes
                    if isinstance(key, bytes) and isinstance(start_key, bytes) and isinstance(end_key, bytes):
                        if start_key <= key <= end_key:
                            result.append((key, value))
                    continue
                
                # All are comparable types
                if start_key <= key_float <= end_key:
                    result.append((key, value))
                elif key_float > end_key:
                    break
                    
        return result 

    def _find_page_with_fence_pointers(self, key) -> Optional[int]:
        """Find the page that contains the key using fence pointers.
        
        This uses a binary search on the fence pointers to find which page
        might contain the key, avoiding the need to check all pages.
        
        Parameters:
        -----------
        key : float
            The key to search for
            
        Returns:
        --------
        int or None
            Page ID if found, None if no pages match or fence pointers aren't available
        """
        # Basic checks for valid fence pointers
        if not self.page_min_keys or not self.page_max_keys:
            return None
            
        if len(self.page_min_keys) != len(self.page_max_keys):
            logger.warning(f"Mismatched fence pointer arrays: {len(self.page_min_keys)} vs {len(self.page_max_keys)}")
            return None
            
        # First check if key is outside the run's range
        if self.min_key is not None and isinstance(self.min_key, type(key)) and key < self.min_key:
            return None
        if self.max_key is not None and isinstance(self.max_key, type(key)) and key > self.max_key:
            return None
            
        # Binary search on fence pointers to find potential page
        left, right = 0, len(self.page_min_keys) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            # Get min and max keys for this range - they should already be converted to float
            min_key = self.page_min_keys[mid]
            max_key = self.page_max_keys[mid]
            
            # Handle incomparable types
            if (isinstance(min_key, bytes) and not isinstance(key, bytes)) or \
               (not isinstance(min_key, bytes) and isinstance(key, bytes)) or \
               (isinstance(max_key, bytes) and not isinstance(key, bytes)) or \
               (not isinstance(max_key, bytes) and isinstance(key, bytes)):
                # Can't use binary search with incomparable types
                return self._find_page(key)
            
            # Check if key is in this page's range
            if min_key <= key <= max_key:
                return mid
            
            # If not, adjust search bounds
            if key < min_key:
                right = mid - 1
            else:
                left = mid + 1
                
        # Check the closest page
        if left < len(self.page_min_keys):
            return left
        elif right >= 0:
            return right
            
        return None
            
    def _find_page(self, key) -> int:
        """Find the page that might contain the key."""
        # Try finding page with fence pointers first
        page = self._find_page_with_fence_pointers(key)
        if page is not None:
            return page
            
        # Fallback to simple page calculation method
        # This is a fallback method when fence pointers aren't available
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            pairs = data['pairs']
            
            # Calculate records per page
            entries_per_page = self.page_size // Constants.ENTRY_SIZE_BYTES
            
            # Binary search to find the page
            left = 0
            right = len(pairs)
            
            while left < right:
                mid = (left + right) // 2
                if mid >= len(pairs):
                    break
                
                mid_key = pairs[mid][0]
                mid_key_float = bytes_to_float(mid_key) if isinstance(mid_key, bytes) else mid_key
                
                # Handle incomparable types
                if isinstance(mid_key_float, bytes) or isinstance(key, bytes):
                    if isinstance(mid_key, bytes) and isinstance(key, bytes):
                        # Both are bytes, compare directly
                        if mid_key == key:
                            return mid // entries_per_page
                        elif mid_key < key:
                            left = mid + 1
                        else:
                            right = mid
                    else:
                        # Can't compare different types, move forward
                        left = mid + 1
                    continue
                
                # Both are comparable types
                if mid_key_float == key:
                    return mid // entries_per_page
                elif mid_key_float < key:
                    left = mid + 1
                else:
                    right = mid
                    
            # Return the page where the key would be inserted
            return left // entries_per_page 