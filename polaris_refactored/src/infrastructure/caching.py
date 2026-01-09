"""
Multi-level caching strategies for POLARIS system.

This module implements L1 (in-memory) and L2 (persistent) caching with
invalidation policies and consistency management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, List, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
from pathlib import Path

from infrastructure.observability.factory import get_infrastructure_logger


T = TypeVar('T')


class CacheLevel(Enum):
    """Cache level enumeration."""
    L1 = "L1"  # In-memory cache
    L2 = "L2"  # Persistent cache


class InvalidationPolicy(Enum):
    """Cache invalidation policy types."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    MANUAL = "manual"  # Manual invalidation only


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: Set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    @abstractmethod
    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Determine which entries should be evicted."""
        pass


class TTLStrategy(CacheStrategy):
    """Time-to-live cache strategy."""

    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Evict expired entries."""
        return [key for key, entry in entries.items() if entry.is_expired()]


class LRUStrategy(CacheStrategy):
    """Least recently used cache strategy."""

    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Evict least recently used entries when over capacity."""
        if len(entries) <= max_size:
            return []

        sorted_entries = sorted(
            entries.items(),
            key=lambda x: x[1].last_accessed
        )
        return [key for key, _ in sorted_entries[:len(entries) - max_size]]


class LFUStrategy(CacheStrategy):
    """Least frequently used cache strategy."""

    def should_evict(self, entries: Dict[str, CacheEntry], max_size: int) -> List[str]:
        """Evict least frequently used entries when over capacity."""
        if len(entries) <= max_size:
            return []

        sorted_entries = sorted(
            entries.items(),
            key=lambda x: (x[1].access_count, x[1].last_accessed)
        )
        return [key for key, _ in sorted_entries[:len(entries) - max_size]]


@dataclass
class CacheConfiguration:
    """Cache configuration settings."""
    max_size: int = 1000
    default_ttl_seconds: Optional[int] = 3600  # 1 hour
    invalidation_policy: InvalidationPolicy = InvalidationPolicy.LRU
    enable_l2_cache: bool = True
    l2_cache_path: Optional[Path] = None
    cleanup_interval_seconds: int = 300  # 5 minutes
    enable_compression: bool = False


class L1Cache:
    """In-memory L1 cache implementation."""

    def __init__(self, config: CacheConfiguration):
        self.config = config
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._strategy = self._create_strategy()
        self._logger = get_infrastructure_logger("cache.l1")
        self._stop_event = threading.Event()  # Thread-safe shutdown signal
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0
        }

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="L1Cache-Cleanup"
        )
        self._cleanup_thread.start()
        self._logger.info("L1 cache initialized", extra={
            "max_size": config.max_size,
            "policy": config.invalidation_policy.value,
            "ttl_seconds": config.default_ttl_seconds
        })

    def _create_strategy(self) -> CacheStrategy:
        """Create cache strategy based on configuration."""
        if self.config.invalidation_policy == InvalidationPolicy.TTL:
            return TTLStrategy()
        elif self.config.invalidation_policy == InvalidationPolicy.LRU:
            return LRUStrategy()
        elif self.config.invalidation_policy == InvalidationPolicy.LFU:
            return LFUStrategy()
        else:
            return LRUStrategy()  # Default fallback

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                self._logger.debug("Cache miss", extra={"key": key})
                return None
            
            if entry.is_expired():
                del self._entries[key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                self._logger.debug("Cache entry expired", extra={"key": key})
                return None
            
            entry.touch()
            self._stats["hits"] += 1
            self._logger.debug("Cache hit", extra={"key": key, "access_count": entry.access_count})
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: Optional[Set[str]] = None) -> None:
        """Set value in cache."""
        with self._lock:
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                tags=tags or set()
            )
            self._entries[key] = entry
            self._stats["sets"] += 1
            
            # Check if eviction is needed
            keys_to_evict = self._strategy.should_evict(self._entries, self.config.max_size)
            for evict_key in keys_to_evict:
                del self._entries[evict_key]
                self._stats["evictions"] += 1
            
            if keys_to_evict:
                self._logger.debug("Evicted entries", extra={
                    "evicted_count": len(keys_to_evict),
                    "current_size": len(self._entries)
                })

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                self._logger.debug("Cache entry deleted", extra={"key": key})
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._logger.info("Cache cleared", extra={"cleared_count": count})

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag."""
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._entries.items()
                if tag in entry.tags
            ]
            for key in keys_to_delete:
                del self._entries[key]
            
            if keys_to_delete:
                self._logger.info("Invalidated entries by tag", extra={
                    "tag": tag,
                    "invalidated_count": len(keys_to_delete)
                })
            return len(keys_to_delete)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0.0
            return {
                **self._stats,
                "current_size": len(self._entries),
                "max_size": self.config.max_size,
                "hit_rate": hit_rate
            }

    def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired entries."""
        while not self._stop_event.wait(self.config.cleanup_interval_seconds):
            with self._lock:
                expired_keys = [
                    key for key, entry in self._entries.items()
                    if entry.is_expired()
                ]
                for key in expired_keys:
                    del self._entries[key]
                    self._stats["evictions"] += 1
                
                if expired_keys:
                    self._logger.debug("Cleanup removed expired entries", extra={
                        "removed_count": len(expired_keys),
                        "remaining_size": len(self._entries)
                    })

    def shutdown(self) -> None:
        """Shutdown the cache and cleanup thread."""
        self._stop_event.set()
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        self._logger.info("L1 cache shutdown initiated")


class CacheManager:
    """
    Centralized cache manager for POLARIS system.
    
    Manages L1 (in-memory) and optionally L2 (persistent) caches
    with unified access patterns and statistics.
    """

    def __init__(self, config: Optional[CacheConfiguration] = None):
        self.config = config or CacheConfiguration()
        self._logger = get_infrastructure_logger("cache.manager")
        self._l1_cache = L1Cache(self.config)
        self._l2_cache = None  # L2 cache implementation placeholder
        
        self._logger.info("Cache manager initialized", extra={
            "l1_enabled": True,
            "l2_enabled": self.config.enable_l2_cache
        })

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)."""
        # Try L1 first
        value = self._l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 if enabled (placeholder for future implementation)
        if self.config.enable_l2_cache and self._l2_cache:
            value = self._l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self._l1_cache.set(key, value)
                return value
        
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, tags: Optional[Set[str]] = None) -> None:
        """Set value in cache (both L1 and L2 if enabled)."""
        self._l1_cache.set(key, value, ttl_seconds, tags)
        
        # Also set in L2 if enabled (placeholder)
        if self.config.enable_l2_cache and self._l2_cache:
            self._l2_cache.set(key, value, ttl_seconds, tags)

    def delete(self, key: str) -> bool:
        """Delete entry from all cache levels."""
        l1_deleted = self._l1_cache.delete(key)
        l2_deleted = False
        
        if self.config.enable_l2_cache and self._l2_cache:
            l2_deleted = self._l2_cache.delete(key)
        
        return l1_deleted or l2_deleted

    def clear(self) -> None:
        """Clear all cache levels."""
        self._l1_cache.clear()
        if self.config.enable_l2_cache and self._l2_cache:
            self._l2_cache.clear()

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate entries by tag across all levels."""
        count = self._l1_cache.invalidate_by_tag(tag)
        if self.config.enable_l2_cache and self._l2_cache:
            count += self._l2_cache.invalidate_by_tag(tag)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        stats = {
            "l1": self._l1_cache.get_stats()
        }
        if self.config.enable_l2_cache and self._l2_cache:
            stats["l2"] = self._l2_cache.get_stats()
        return stats

    def shutdown(self) -> None:
        """Shutdown all cache levels."""
        self._l1_cache.shutdown()
        if self._l2_cache:
            self._l2_cache.shutdown()
        self._logger.info("Cache manager shutdown complete")


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(config: Optional[CacheConfiguration] = None) -> CacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
    return _cache_manager
