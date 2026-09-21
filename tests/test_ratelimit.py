from concurrent.futures import ThreadPoolExecutor

import pytest

from ratelimit import TokenBucketRateLimiter


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


def build(clock, capacity=3, refill_per_second=1.0, max_buckets=10_000):
    return TokenBucketRateLimiter(
        capacity=capacity,
        refill_per_second=refill_per_second,
        time_fn=clock,
        max_buckets=max_buckets,
    )


def test_a_fresh_key_starts_full(clock):
    limiter = build(clock, capacity=3)

    decision = limiter.acquire("alice")

    assert decision.allowed is True
    assert decision.limit == 3
    assert decision.remaining == 2


def test_burst_is_allowed_up_to_capacity_then_denied(clock):
    limiter = build(clock, capacity=3)

    results = [limiter.acquire("alice").allowed for _ in range(5)]

    assert results == [True, True, True, False, False]


def test_tokens_refill_over_time(clock):
    limiter = build(clock, capacity=3, refill_per_second=1.0)
    for _ in range(3):
        limiter.acquire("alice")

    assert limiter.acquire("alice").allowed is False

    clock.advance(1.0)
    assert limiter.acquire("alice").allowed is True

    assert limiter.acquire("alice").allowed is False
    clock.advance(2.0)
    assert limiter.acquire("alice").allowed is True
    assert limiter.acquire("alice").allowed is True


def test_refill_never_exceeds_capacity(clock):
    limiter = build(clock, capacity=3, refill_per_second=1.0)
    limiter.acquire("alice")

    clock.advance(10_000)

    results = [limiter.acquire("alice").allowed for _ in range(4)]
    assert results == [True, True, True, False]


def test_keys_are_isolated(clock):
    limiter = build(clock, capacity=2)

    assert limiter.acquire("alice").allowed is True
    assert limiter.acquire("alice").allowed is True
    assert limiter.acquire("alice").allowed is False

    assert limiter.acquire("bob").allowed is True
    assert limiter.acquire("bob").allowed is True


def test_retry_after_is_zero_while_allowed(clock):
    limiter = build(clock, capacity=2)

    assert limiter.acquire("alice").retry_after == 0.0


def test_retry_after_counts_down_to_the_next_token(clock):
    limiter = build(clock, capacity=2, refill_per_second=0.5)
    limiter.acquire("alice")
    limiter.acquire("alice")

    denied = limiter.acquire("alice")
    assert denied.allowed is False
    assert denied.retry_after == pytest.approx(2.0)

    clock.advance(1.0)
    still_denied = limiter.acquire("alice")
    assert still_denied.allowed is False
    assert still_denied.retry_after == pytest.approx(1.0)


def test_reset_after_is_time_to_a_full_bucket(clock):
    limiter = build(clock, capacity=4, refill_per_second=2.0)

    decision = limiter.acquire("alice")

    assert decision.remaining == 3
    assert decision.reset_after == pytest.approx(0.5)


def test_reset_after_is_zero_when_untouched(clock):
    limiter = build(clock, capacity=4, refill_per_second=2.0)
    limiter.acquire("alice")
    clock.advance(100)

    decision = limiter.acquire("alice", cost=4)

    assert decision.allowed is True
    assert decision.reset_after == pytest.approx(2.0)


def test_cost_greater_than_one(clock):
    limiter = build(clock, capacity=5)

    assert limiter.acquire("alice", cost=3).remaining == 2
    assert limiter.acquire("alice", cost=3).allowed is False
    assert limiter.acquire("alice", cost=2).allowed is True


def test_cost_must_be_positive(clock):
    limiter = build(clock)

    with pytest.raises(ValueError):
        limiter.acquire("alice", cost=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": 0, "refill_per_second": 1.0},
        {"capacity": 3, "refill_per_second": 0},
        {"capacity": 3, "refill_per_second": -1},
        {"capacity": 3, "refill_per_second": 1.0, "max_buckets": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(**kwargs)


def test_idle_buckets_are_evicted_once_the_cap_is_reached(clock):
    limiter = build(clock, capacity=2, refill_per_second=1.0, max_buckets=3)

    for name in ("a", "b", "c"):
        limiter.acquire(name)
    assert len(limiter) == 3

    clock.advance(100)
    limiter.acquire("d")

    assert len(limiter) == 1


def test_eviction_keeps_buckets_that_are_still_spending(clock):
    limiter = build(clock, capacity=2, refill_per_second=1.0, max_buckets=3)

    limiter.acquire("heavy")
    limiter.acquire("heavy")
    limiter.acquire("idle_one")
    limiter.acquire("idle_two")

    limiter.acquire("newcomer")

    assert limiter.acquire("heavy").allowed is False


def test_no_eviction_below_the_cap(clock):
    limiter = build(clock, capacity=2, max_buckets=100)

    clock.advance(100)
    for name in ("a", "b", "c"):
        limiter.acquire(name)

    assert len(limiter) == 3


def test_concurrent_callers_never_exceed_capacity(clock):
    limiter = build(clock, capacity=50, refill_per_second=1.0)

    with ThreadPoolExecutor(max_workers=16) as pool:
        decisions = list(pool.map(lambda _: limiter.acquire("shared"), range(500)))

    allowed = [decision for decision in decisions if decision.allowed]
    assert len(allowed) == 50


def test_concurrent_distinct_keys_each_get_their_own_budget(clock):
    limiter = build(clock, capacity=2, refill_per_second=1.0)
    keys = [f"user{i}" for i in range(40)] * 3

    with ThreadPoolExecutor(max_workers=16) as pool:
        decisions = list(pool.map(limiter.acquire, keys))

    assert sum(1 for decision in decisions if decision.allowed) == 40 * 2
