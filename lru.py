import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    expirations: int
    size: int
    capacity: int

    @property
    def hit_rate(self) -> float:
        lookups = self.hits + self.misses
        return self.hits / lookups if lookups else 0.0


class _Node:
    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key=None, value=None, expires_at=None):
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int, ttl_seconds: float | None = None, time_fn=time.monotonic):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")

        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._time = time_fn
        self._map: dict = {}
        self._lock = threading.Lock()

        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

    def _push_front(self, node: _Node) -> None:
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _drop(self, node: _Node) -> None:
        self._unlink(node)
        del self._map[node.key]

    def _is_expired(self, node: _Node, now: float) -> bool:
        return node.expires_at is not None and now >= node.expires_at

    def get(self, key, default=None):
        now = self._time()

        with self._lock:
            node = self._map.get(key)

            if node is None:
                self._misses += 1
                return default

            if self._is_expired(node, now):
                self._drop(node)
                self._expirations += 1
                self._misses += 1
                return default

            self._unlink(node)
            self._push_front(node)
            self._hits += 1
            return node.value

    def put(self, key, value) -> None:
        now = self._time()
        expires_at = now + self.ttl_seconds if self.ttl_seconds is not None else None

        with self._lock:
            node = self._map.get(key)

            if node is not None:
                node.value = value
                node.expires_at = expires_at
                self._unlink(node)
                self._push_front(node)
                return

            node = _Node(key, value, expires_at)
            self._map[key] = node
            self._push_front(node)

            if len(self._map) > self.capacity:
                victim = self._tail.prev
                self._drop(victim)
                self._evictions += 1

    def peek(self, key, default=None):
        now = self._time()

        with self._lock:
            node = self._map.get(key)
            if node is None or self._is_expired(node, now):
                return default
            return node.value

    def invalidate(self, key) -> bool:
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return False
            self._drop(node)
            return True

    def clear(self) -> None:
        with self._lock:
            self._map.clear()
            self._head.next = self._tail
            self._tail.prev = self._head

    def keys_most_recent_first(self) -> list:
        with self._lock:
            keys = []
            node = self._head.next
            while node is not self._tail:
                keys.append(node.key)
                node = node.next
            return keys

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._map),
                capacity=self.capacity,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)
