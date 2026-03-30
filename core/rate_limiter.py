import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)

WINDOW_SECONDS = 60

BURST_LIMIT = 20

CLEANUP_INTERVAL_SECONDS = 300


@dataclass
class RateLimitState:
    requests:       Deque[float] = field(default_factory=deque)
    burst_requests: Deque[float] = field(default_factory=deque)
    last_seen:      float        = field(default_factory=time.time)


@dataclass
class RateLimitResult:
    allowed:    bool
    limit:      int
    remaining:  int
    reset_at:   float
    retry_after: float        = 0.0
    reason:     str           = ""


class SlidingWindowRateLimiter:

    def __init__(self) -> None:
        self._states:      Dict[str, RateLimitState] = {}
        self._lock:        asyncio.Lock               = asyncio.Lock()
        self._last_cleanup: float                     = time.time()

    async def check(
        self,
        identifier: str,
        limit:      int,
        burst_limit: int = BURST_LIMIT,
    ) -> RateLimitResult:
        async with self._lock:
            now   = time.time()
            state = self._states.get(identifier)

            if state is None:
                state = RateLimitState()
                self._states[identifier] = state

            state.last_seen = now

            window_start = now - WINDOW_SECONDS
            while state.requests and state.requests[0] < window_start:
                state.requests.popleft()

            burst_start = now - 1.0
            while state.burst_requests and state.burst_requests[0] < burst_start:
                state.burst_requests.popleft()

            if len(state.burst_requests) >= burst_limit:
                retry_after = round(1.0 - (now - state.burst_requests[0]), 2)
                return RateLimitResult(
                    allowed     = False,
                    limit       = limit,
                    remaining   = 0,
                    reset_at    = now + retry_after,
                    retry_after = max(retry_after, 0.1),
                    reason      = f"Burst limit terlampaui ({burst_limit} req/detik).",
                )

            current_count = len(state.requests)
            if current_count >= limit:
                oldest       = state.requests[0]
                reset_at     = oldest + WINDOW_SECONDS
                retry_after  = max(reset_at - now, 0.1)
                return RateLimitResult(
                    allowed     = False,
                    limit       = limit,
                    remaining   = 0,
                    reset_at    = reset_at,
                    retry_after = round(retry_after, 2),
                    reason      = f"Rate limit terlampaui ({limit} req/menit).",
                )

            state.requests.append(now)
            state.burst_requests.append(now)

            reset_at  = (state.requests[0] + WINDOW_SECONDS) if state.requests else (now + WINDOW_SECONDS)
            remaining = max(limit - len(state.requests), 0)

            if now - self._last_cleanup > CLEANUP_INTERVAL_SECONDS:
                await self._cleanup(now)

            return RateLimitResult(
                allowed   = True,
                limit     = limit,
                remaining = remaining,
                reset_at  = reset_at,
            )

    async def _cleanup(self, now: float) -> None:
        cutoff      = now - 600
        stale_keys  = [k for k, s in self._states.items() if s.last_seen < cutoff]
        for k in stale_keys:
            del self._states[k]
        if stale_keys:
            logger.debug(f"Rate limiter cleanup: hapus {len(stale_keys)} entry stale.")
        self._last_cleanup = now

    def get_stats(self) -> Dict[str, int]:
        return {
            "tracked_identifiers": len(self._states),
        }


_rate_limiter: Optional[SlidingWindowRateLimiter] = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SlidingWindowRateLimiter()
    return _rate_limiter


def build_rate_limit_headers(result: RateLimitResult) -> Dict[str, str]:
    headers = {
        "X-RateLimit-Limit":     str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset":     str(int(result.reset_at)),
        "X-RateLimit-Policy":    f"{result.limit};w={WINDOW_SECONDS}",
    }
    if not result.allowed:
        headers["Retry-After"] = str(int(result.retry_after) + 1)
    return headers