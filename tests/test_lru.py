from concurrent.futures import ThreadPoolExecutor

import pytest

from lru import LRUCache


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


def build(clock=None, capacity=3, ttl_seconds=None):
    return LRUCache(
        capacity=capacity,
        ttl_seconds=ttl_seconds,
        time_fn=clock if clock is not None else FakeClock(),
    )


def test_put_then_get():
    cache = build()
    cache.put("a", 1)

    assert cache.get("a") == 1


def test_missing_key_returns_default():
    cache = build()

    assert cache.get("nope") is None
    assert cache.get("nope", "fallback") == "fallback"


def test_put_overwrites_without_growing():
    cache = build()
    cache.put("a", 1)
    cache.put("a", 2)

    assert cache.get("a") == 2
    assert len(cache) == 1


def test_eviction_drops_the_least_recently_used():
    cache = build(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    cache.put("d", 4)

    assert cache.get("a") is None
    assert cache.keys_most_recent_first() == ["d", "c", "b"]


def test_a_get_protects_a_key_from_eviction():
    cache = build(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    cache.get("a")
    cache.put("d", 4)

    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_a_repeated_put_also_refreshes_recency():
    cache = build(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)

    cache.put("a", 99)
    cache.put("d", 4)

    assert cache.get("a") == 99
    assert cache.get("b") is None


def test_capacity_is_never_exceeded():
    cache = build(capacity=3)

    for index in range(100):
        cache.put(index, index)

    assert len(cache) == 3
    assert cache.keys_most_recent_first() == [99, 98, 97]


def test_capacity_of_one():
    cache = build(capacity=1)
    cache.put("a", 1)
    cache.put("b", 2)

    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_order_survives_a_full_cycle():
    cache = build(capacity=3)
    for key in ("a", "b", "c"):
        cache.put(key, key)

    cache.get("a")
    cache.get("b")
    cache.get("c")

    assert cache.keys_most_recent_first() == ["c", "b", "a"]


def test_entries_expire_after_the_ttl(clock):
    cache = build(clock, ttl_seconds=60)
    cache.put("a", 1)

    clock.advance(59)
    assert cache.get("a") == 1

    clock.advance(2)
    assert cache.get("a") is None


def test_expiry_is_counted_and_frees_space(clock):
    cache = build(clock, capacity=3, ttl_seconds=10)
    cache.put("a", 1)

    clock.advance(11)
    cache.get("a")

    assert len(cache) == 0
    assert cache.stats().expirations == 1


def test_a_put_refreshes_the_ttl(clock):
    cache = build(clock, ttl_seconds=10)
    cache.put("a", 1)

    clock.advance(9)
    cache.put("a", 2)
    clock.advance(9)

    assert cache.get("a") == 2


def test_a_get_does_not_refresh_the_ttl(clock):
    cache = build(clock, ttl_seconds=10)
    cache.put("a", 1)

    clock.advance(9)
    cache.get("a")
    clock.advance(2)

    assert cache.get("a") is None


def test_no_ttl_means_entries_never_expire(clock):
    cache = build(clock, ttl_seconds=None)
    cache.put("a", 1)

    clock.advance(10_000_000)

    assert cache.get("a") == 1


def test_peek_does_not_change_recency():
    cache = build(capacity=3)
    for key in ("a", "b", "c"):
        cache.put(key, key)

    assert cache.peek("a") == "a"
    cache.put("d", "d")

    assert cache.get("a") is None


def test_peek_does_not_count_as_a_hit():
    cache = build()
    cache.put("a", 1)

    cache.peek("a")
    cache.peek("missing")

    assert cache.stats().hits == 0
    assert cache.stats().misses == 0


def test_invalidate_removes_an_entry():
    cache = build()
    cache.put("a", 1)

    assert cache.invalidate("a") is True
    assert cache.invalidate("a") is False
    assert cache.get("a") is None
    assert len(cache) == 0


def test_clear_empties_the_cache():
    cache = build(capacity=3)
    for key in ("a", "b", "c"):
        cache.put(key, key)

    cache.clear()

    assert len(cache) == 0
    assert cache.keys_most_recent_first() == []
    cache.put("d", 4)
    assert cache.get("d") == 4


def test_stats_track_hits_misses_and_evictions():
    cache = build(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.get("missing")
    cache.put("c", 3)

    stats = cache.stats()

    assert (stats.hits, stats.misses, stats.evictions) == (1, 1, 1)
    assert stats.size == 2
    assert stats.capacity == 2
    assert stats.hit_rate == 0.5


def test_hit_rate_is_zero_before_any_lookup():
    assert build().stats().hit_rate == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"capacity": 0}, {"capacity": -1}, {"capacity": 2, "ttl_seconds": 0}, {"capacity": 2, "ttl_seconds": -5}],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LRUCache(**kwargs)


def test_values_may_be_any_object():
    cache = build()
    payload = {"answer": "hi", "sources": [1, 2]}
    cache.put("a", payload)

    assert cache.get("a") is payload


def test_concurrent_access_keeps_the_structure_consistent():
    cache = LRUCache(capacity=50, time_fn=FakeClock())

    def hammer(index):
        cache.put(index % 200, index)
        cache.get(index % 200)
        return len(cache)

    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = list(pool.map(hammer, range(4000)))

    assert max(sizes) <= 50
    assert len(cache) == 50
    assert len(cache.keys_most_recent_first()) == 50
    assert len(set(cache.keys_most_recent_first())) == 50
