"""Tiny thread-safe LRU caches for responses and embeddings."""
from __future__ import annotations
import hashlib
import json
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUCache:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(1, int(capacity))
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._data),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
            }


def make_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
