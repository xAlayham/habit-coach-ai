import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    allowed: bool
    limit: int
    remaining: int
    reset_after: float
    retry_after: float


class _Bucket:
    __slots__ = ("tokens", "updated_at")

    def __init__(self, tokens: float, updated_at: float):
        self.tokens = tokens
        self.updated_at = updated_at


class TokenBucketRateLimiter:
    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        time_fn=time.monotonic,
        max_buckets: int = 10_000,
    ):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be positive")
        if max_buckets < 1:
            raise ValueError("max_buckets must be at least 1")

        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.max_buckets = max_buckets
        self._time = time_fn
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str, cost: int = 1) -> Decision:
        if cost < 1:
            raise ValueError("cost must be at least 1")

        now = self._time()

        with self._lock:
            bucket = self._buckets.get(key)

            if bucket is None:
                self._evict_full_buckets(now)
                bucket = _Bucket(float(self.capacity), now)
                self._buckets[key] = bucket
            else:
                elapsed = now - bucket.updated_at
                bucket.tokens = min(
                    float(self.capacity),
                    bucket.tokens + elapsed * self.refill_per_second,
                )
                bucket.updated_at = now

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                allowed = True
                retry_after = 0.0
            else:
                allowed = False
                retry_after = (cost - bucket.tokens) / self.refill_per_second

            reset_after = (self.capacity - bucket.tokens) / self.refill_per_second

            return Decision(
                allowed=allowed,
                limit=self.capacity,
                remaining=int(bucket.tokens),
                reset_after=reset_after,
                retry_after=retry_after,
            )

    def _evict_full_buckets(self, now: float) -> int:
        if len(self._buckets) < self.max_buckets:
            return 0

        evicted = 0
        for key, bucket in list(self._buckets.items()):
            refilled = bucket.tokens + (now - bucket.updated_at) * self.refill_per_second
            if refilled >= self.capacity:
                del self._buckets[key]
                evicted += 1
        return evicted

    def __len__(self) -> int:
        with self._lock:
            return len(self._buckets)
