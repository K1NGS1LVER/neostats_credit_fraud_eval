"""
Debounce and LRU Cache Module for Talk-to-Data Engine.

Provides:
- Debouncer: Thread-safe execution lock and 400ms event suppression to prevent duplicate clicks
  and rapid API spamming.
- QueryLRUCache: Thread-safe in-memory LRU cache storing the last 50 generated SQL queries
  and executive summaries.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class DebounceSuppressionError(Exception):
    """Raised when an operation is suppressed due to active debounce cooldown."""
    pass


class Debouncer:
    """
    Execution lock and event suppressor with a default 400ms cooldown window.
    Guards against rapid duplicate clicks and API spamming.
    """

    def __init__(
        self,
        suppression_ms: float = 400.0,
        clock_fn: Any = time.monotonic,
    ) -> None:
        """
        Initialize the Debouncer.

        Args:
            suppression_ms: Cooldown window in milliseconds (default: 400ms).
            clock_fn: Clock provider (default: time.monotonic).
        """
        self.cooldown_sec = max(0.0, suppression_ms / 1000.0)
        self.clock_fn = clock_fn
        self._lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._last_call_timestamps: Dict[str, float] = {}

    def is_suppressed(self, key: str = "default") -> bool:
        """Check if an event for the given key is currently within the debounce window."""
        with self._lock:
            last_time = self._last_call_timestamps.get(key, 0.0)
            return (self.clock_fn() - last_time) < self.cooldown_sec

    def acquire(self, key: str = "default", raise_on_suppress: bool = True) -> bool:
        """
        Attempt to pass the debounce gate.

        Args:
            key: Identifier for the debounced event (e.g. user action or query).
            raise_on_suppress: If True, raises DebounceSuppressionError when within cooldown.

        Returns:
            True if allowed through the debounce gate.

        Raises:
            DebounceSuppressionError: If suppressed and raise_on_suppress is True.
        """
        with self._lock:
            now = self.clock_fn()
            last_time = self._last_call_timestamps.get(key, 0.0)
            elapsed = now - last_time

            if elapsed < self.cooldown_sec:
                remaining_ms = (self.cooldown_sec - elapsed) * 1000.0
                msg = (
                    f"Event suppressed by 400ms debounce guard. "
                    f"Please wait {remaining_ms:.0f}ms before retrying."
                )
                if raise_on_suppress:
                    raise DebounceSuppressionError(msg)
                return False

            self._last_call_timestamps[key] = now
            return True

    @contextmanager
    def execution_guard(
        self,
        key: str = "default",
        blocking: bool = True,
        timeout: float = -1.0,
    ) -> Generator[None, None, None]:
        """
        Context manager combining 400ms event suppression and thread execution serialization.

        Args:
            key: Identification key for debounce check.
            blocking: Whether execution lock should block.
            timeout: Timeout for execution lock acquisition.

        Raises:
            DebounceSuppressionError: If event is suppressed by debounce cooldown.
            TimeoutError: If execution lock acquisition times out.
        """
        self.acquire(key=key, raise_on_suppress=True)

        acquired = self._execution_lock.acquire(blocking=blocking, timeout=timeout)
        if not acquired:
            raise TimeoutError("Concurrent query execution lock could not be acquired.")
        try:
            yield
        finally:
            self._execution_lock.release()

    def reset(self, key: Optional[str] = None) -> None:
        """Reset the debounce cooldown for a key or all keys."""
        with self._lock:
            if key is not None:
                self._last_call_timestamps.pop(key, None)
            else:
                self._last_call_timestamps.clear()


class QueryLRUCache:
    """
    Thread-safe in-memory LRU cache storing the last 50 generated SQL queries
    and executive summaries.
    """

    DEFAULT_CAPACITY = 50

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        """
        Initialize the LRU cache.

        Args:
            capacity: Maximum number of query entries to retain (default: 50).
        """
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive.")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._cache: collections.OrderedDict[str, Dict[str, Any]] = collections.OrderedDict()

    @staticmethod
    def _normalize_key(query: str) -> str:
        """Normalize query text for cache key matching."""
        return " ".join(query.strip().lower().split())

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached entry for a query and update MRU order.

        Args:
            query: The natural language user query.

        Returns:
            Dictionary containing sql, summary, timestamp, and metadata, or None if not found.
        """
        norm_key = self._normalize_key(query)
        with self._lock:
            if norm_key in self._cache:
                self._cache.move_to_end(norm_key)
                return self._cache[norm_key].copy()
            return None

    def put(
        self,
        query: str,
        sql: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store or update a query entry, evicting the least recently used entry if at capacity.

        Args:
            query: The natural language user query.
            sql: Generated DuckDB SQL string.
            summary: Executive summary text.
            metadata: Optional dictionary with performance, tier, or execution details.
        """
        norm_key = self._normalize_key(query)
        entry = {
            "query": query,
            "sql": sql,
            "summary": summary,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        with self._lock:
            if norm_key in self._cache:
                self._cache.move_to_end(norm_key)
            self._cache[norm_key] = entry

            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def contains(self, query: str) -> bool:
        """Check if a query is present in the cache without altering LRU order."""
        norm_key = self._normalize_key(query)
        with self._lock:
            return norm_key in self._cache

    def peek(self, query: str) -> Optional[Dict[str, Any]]:
        """Inspect cached entry without updating LRU order."""
        norm_key = self._normalize_key(query)
        with self._lock:
            if norm_key in self._cache:
                return self._cache[norm_key].copy()
            return None

    def size(self) -> int:
        """Return the current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def list_entries(self) -> List[Dict[str, Any]]:
        """Return a snapshot list of cached items from most recent to oldest."""
        with self._lock:
            return [entry.copy() for entry in reversed(self._cache.values())]
