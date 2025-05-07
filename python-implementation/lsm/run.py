import os
import pickle
import struct
import logging
import math
import time
from typing import List, Tuple, Optional, Dict, Any
from .bloom import BloomFilter
from .utils import Constants, ensure_directory, merge_sorted_pairs, is_tombstone

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Change from INFO to WARNING
logger = logging.getLogger(__name__)

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
        # First, do a basic range check
        if self.min_key is None or self.max_key is None:
            # If we don't have min/max keys, assume it might contain the key
            return True
            
        if key < self.min_key or key > self.max_key:
            return False
        
        # If ML models are provided, use the learned model
        if ml_models is not None:
            try:
                # Get prediction from model (0.0 to 1.0)
                prediction = ml_models.predict_bloom(key)
                
                # Use the provided threshold - lower values will be more aggressive
                # about filtering out runs (but risk more false negatives)
                return prediction >= bloom_thresh
            except Exception as e:
                # On any error, default to True for safety
                logger.warning(f"Bloom filter prediction failed: {e}")
                return True
        
        # Otherwise, use traditional bloom filter if available
        if self.bloom_filter is not None:
            return self.bloom_filter.might_contain(key)
            
        # If no bloom filter available, assume it might contain the key
        return True
        
    def get(self, key: float) -> Optional[str]:
        """Get value for a key."""
        # Check bloom filter first (optimization)
        if self.bloom_filter is not None and not self.bloom_filter.might_contain(key):
            return None
            
        # If we have fence pointers, use them to find the right page
        if self.page_min_keys and hasattr(self, 'num_pages') and self.num_pages > 1:
            page_id = self._find_page_with_fence_pointers(key)
            if page_id is not None:
                return self.get_from_page(key, page_id)
            
        # Fallback to traditional binary search
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            pairs = data['pairs']
            
            # Binary search for the key
            left, right = 0, len(pairs) - 1
            while left <= right:
                mid = (left + right) // 2
                if pairs[mid][0] == key:
                    return pairs[mid][1]
                elif pairs[mid][0] < key:
                    left = mid + 1
                else:
                    right = mid - 1
                    
        return None
        
    def range(self, start_key: float, end_key: float) -> List[Tuple[float, str]]:
        """Get all key-value pairs in the given range."""
        result = []
        
        # Use fence pointers to find the start and end pages
        start_page = None
        end_page = None
        
        if self.page_min_keys and self.page_max_keys:
            # Find start page
            for i, min_key in enumerate(self.page_min_keys):
                if min_key <= start_key and i < len(self.page_max_keys) and self.page_max_keys[i] >= start_key:
                    start_page = i
                    break
                if min_key > start_key:
                    start_page = i
                    break
            
            # Find end page
            for i, max_key in enumerate(self.page_max_keys):
                if max_key >= end_key:
                    end_page = i
                    break
            
            # If we found valid pages, use them
            if start_page is not None and end_page is not None:
                for page_id in range(start_page, end_page + 1):
                    page_results = self.get_range_from_page(start_key, end_key, page_id)
                    result.extend(page_results)
                return result
        
        # Fallback to scanning the whole file
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            pairs = data['pairs']
            
            # Scan through relevant pages
            for pair in pairs:
                key, value = pair
                if start_key <= key <= end_key:
                    result.append((key, value))
                elif key > end_key:
                    break
                    
        return result
        
    def get_size_bytes(self) -> int:
        """Get size of run in bytes."""
        return os.path.getsize(self.file_path) if os.path.exists(self.file_path) else 0 

    def get_from_page(self, key: float, page_id: int) -> Optional[str]:
        """Get value for a key from a specific page. Optimized for minimal disk I/O."""
        # Input validation
        if page_id < 0 or (hasattr(self, 'num_pages') and page_id >= self.num_pages):
            return None
            
        # Use fence pointers if available to quickly reject keys outside page range
        if self.page_min_keys and self.page_max_keys and len(self.page_min_keys) > page_id and len(self.page_max_keys) > page_id:
            if key < self.page_min_keys[page_id] or key > self.page_max_keys[page_id]:
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
                return None
        
        # Binary search within the page
        if not pairs:
            return None
        
        # Fast path for small pages (linear scan)
        if len(pairs) <= 8:
            for k, v in pairs:
                if k == key:
                    return v
            return None
        
        # Binary search for larger pages
        left, right = 0, len(pairs) - 1
        while left <= right:
            mid = (left + right) // 2
            if pairs[mid][0] == key:
                return pairs[mid][1]
            elif pairs[mid][0] < key:
                left = mid + 1
            else:
                right = mid - 1
            
        return None
        
    def get_range_from_page(self, start_key: float, end_key: float, page_id: int) -> List[Tuple[float, str]]:
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
                if start_key <= key <= end_key:
                    result.append((key, value))
                elif key > end_key:
                    break
                    
        return result 

    def _find_page_with_fence_pointers(self, key: float) -> Optional[int]:
        """Use fence pointers to quickly find the page that might contain the key."""
        if not self.page_min_keys or not self.page_max_keys:
            return None
            
        # Linear search for small number of pages
        if len(self.page_min_keys) <= 8:
            for i, (min_key, max_key) in enumerate(zip(self.page_min_keys, self.page_max_keys)):
                if min_key <= key <= max_key:
                    return i
            return None
            
        # Binary search for larger number of pages
        left = 0
        right = len(self.page_min_keys) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if mid >= len(self.page_min_keys):
                break
                
            min_key = self.page_min_keys[mid]
            max_key = self.page_max_keys[mid]
            
            if min_key <= key <= max_key:
                return mid
            elif max_key < key:
                left = mid + 1
            else:
                right = mid - 1
                
        return None
            
    def _find_page(self, key: float) -> int:
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
                if pairs[mid][0] == key:
                    return mid // entries_per_page
                elif pairs[mid][0] < key:
                    left = mid + 1
                else:
                    right = mid
                    
            # Return the page where the key would be inserted
            return left // entries_per_page 