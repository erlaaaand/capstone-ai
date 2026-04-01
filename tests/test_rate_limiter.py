"""
Tests untuk core/rate_limiter.py — SlidingWindowRateLimiter.
"""
import asyncio
import time

import pytest

from core.rate_limiter import (
    BURST_LIMIT,
    WINDOW_SECONDS,
    SlidingWindowRateLimiter,
    build_rate_limit_headers,
    get_rate_limiter,
    RateLimitResult,
)


def run(coro):
    """Helper untuk menjalankan coroutine dalam test sinkron."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSlidingWindowRateLimiter:

    def _fresh(self) -> SlidingWindowRateLimiter:
        return SlidingWindowRateLimiter()

    # ── Allow cases ──────────────────────────────────────────────────────────

    def test_first_request_allowed(self):
        limiter = self._fresh()
        result  = run(limiter.check("user:1", limit=10))
        assert result.allowed is True

    def test_requests_within_limit_allowed(self):
        limiter = self._fresh()
        for i in range(5):
            result = run(limiter.check("user:2", limit=10))
            assert result.allowed, f"Request #{i+1} harus diizinkan"

    def test_remaining_decreases_each_request(self):
        limiter = self._fresh()
        r1 = run(limiter.check("user:3", limit=10))
        r2 = run(limiter.check("user:3", limit=10))
        assert r2.remaining < r1.remaining

    def test_remaining_not_negative(self):
        limiter = self._fresh()
        for _ in range(10):
            run(limiter.check("user:4", limit=10))
        result = run(limiter.check("user:4", limit=10))
        # Either blocked or remaining >= 0
        assert result.remaining >= 0

    def test_result_contains_limit(self):
        limiter = self._fresh()
        result  = run(limiter.check("user:5", limit=42))
        assert result.limit == 42

    def test_result_contains_reset_at(self):
        limiter = self._fresh()
        result  = run(limiter.check("user:6", limit=10))
        assert result.reset_at > time.time()

    # ── Deny cases ───────────────────────────────────────────────────────────

    def test_exceeding_limit_denied(self):
        limiter = self._fresh()
        limit   = 5
        identifier = "user:rate_test"
        for _ in range(limit):
            run(limiter.check(identifier, limit=limit))
        result = run(limiter.check(identifier, limit=limit))
        assert result.allowed is False

    def test_denied_result_has_retry_after(self):
        limiter    = self._fresh()
        identifier = "user:retry"
        for _ in range(3):
            run(limiter.check(identifier, limit=3))
        result = run(limiter.check(identifier, limit=3))
        if not result.allowed:
            assert result.retry_after > 0

    def test_denied_result_has_reason(self):
        limiter    = self._fresh()
        identifier = "user:reason"
        for _ in range(3):
            run(limiter.check(identifier, limit=3))
        result = run(limiter.check(identifier, limit=3))
        if not result.allowed:
            assert result.reason != ""

    def test_burst_limit_enforced(self):
        """Lebih dari BURST_LIMIT request dalam 1 detik harus ditolak."""
        limiter    = self._fresh()
        identifier = "user:burst"
        denied     = False
        # Kirim lebih dari BURST_LIMIT request sekaligus
        for _ in range(BURST_LIMIT + 5):
            result = run(limiter.check(identifier, limit=9999))
            if not result.allowed:
                denied = True
                assert "burst" in result.reason.lower()
                break
        assert denied, "Burst limit tidak ter-enforce"

    # ── Isolation ────────────────────────────────────────────────────────────

    def test_different_identifiers_isolated(self):
        limiter = self._fresh()
        limit   = 3
        for _ in range(limit):
            run(limiter.check("user:A", limit=limit))
        # user:A habis, tapi user:B masih bisa
        result_b = run(limiter.check("user:B", limit=limit))
        assert result_b.allowed is True

    # ── Stats ────────────────────────────────────────────────────────────────

    def test_get_stats_returns_dict(self):
        limiter = self._fresh()
        stats   = limiter.get_stats()
        assert isinstance(stats, dict)
        assert "tracked_identifiers" in stats

    def test_stats_tracks_new_identifiers(self):
        limiter = self._fresh()
        assert limiter.get_stats()["tracked_identifiers"] == 0
        run(limiter.check("new:user", limit=10))
        assert limiter.get_stats()["tracked_identifiers"] == 1


class TestBuildRateLimitHeaders:
    def test_allowed_result_returns_standard_headers(self):
        result  = RateLimitResult(allowed=True, limit=100, remaining=99, reset_at=time.time() + 60)
        headers = build_rate_limit_headers(result)
        assert "X-RateLimit-Limit"     in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset"     in headers
        assert "X-RateLimit-Policy"    in headers

    def test_denied_result_includes_retry_after(self):
        result  = RateLimitResult(
            allowed=False, limit=60, remaining=0,
            reset_at=time.time() + 30, retry_after=30.0,
            reason="Rate limit terlampaui",
        )
        headers = build_rate_limit_headers(result)
        assert "Retry-After" in headers

    def test_allowed_result_no_retry_after(self):
        result  = RateLimitResult(allowed=True, limit=60, remaining=50, reset_at=time.time() + 60)
        headers = build_rate_limit_headers(result)
        assert "Retry-After" not in headers

    def test_header_values_are_strings(self):
        result  = RateLimitResult(allowed=True, limit=100, remaining=99, reset_at=time.time() + 60)
        headers = build_rate_limit_headers(result)
        for k, v in headers.items():
            assert isinstance(v, str), f"Header {k} bukan string: {type(v)}"

    def test_limit_value_matches_result(self):
        result  = RateLimitResult(allowed=True, limit=300, remaining=299, reset_at=time.time() + 60)
        headers = build_rate_limit_headers(result)
        assert headers["X-RateLimit-Limit"] == "300"


class TestGetRateLimiter:
    def test_returns_instance(self):
        limiter = get_rate_limiter()
        assert isinstance(limiter, SlidingWindowRateLimiter)

    def test_returns_same_instance(self):
        l1 = get_rate_limiter()
        l2 = get_rate_limiter()
        assert l1 is l2
