import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from core.logger import get_logger

logger = get_logger(__name__)

WINDOW_SECONDS            = 60
BURST_LIMIT               = 20
CLEANUP_INTERVAL_SECONDS  = 300
STALE_THRESHOLD_SECONDS   = 600

@dataclass
class RateLimitState:
    requests:       Deque[float] = field(default_factory=deque)
    burst_requests: Deque[float] = field(default_factory=deque)
    last_seen:      float        = field(default_factory=time.time)


@dataclass
class RateLimitResult:
    allowed:     bool
    limit:       int
    remaining:   int
    reset_at:    float
    retry_after: float = 0.0
    reason:      str   = ""


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._states:        Dict[str, RateLimitState] = {}
        self._lock:          asyncio.Lock               = asyncio.Lock()
        self._last_cleanup:  float                      = time.time()
        self._cleanup_task:  Optional[asyncio.Task]     = None

    async def start_cleanup_task(self) -> None:

        if self._cleanup_task is not None and not self._cleanup_task.done():
            logger.debug("Cleanup task sudah berjalan, skip.")
            return

        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="rate_limiter_cleanup",
        )
        logger.info(
            f"[RateLimiter] Background cleanup task dimulai "
            f"(interval={CLEANUP_INTERVAL_SECONDS}s, "
            f"stale_threshold={STALE_THRESHOLD_SECONDS}s)."
        )

    async def stop_cleanup_task(self) -> None:

        if self._cleanup_task is None or self._cleanup_task.done():
            return

        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("[RateLimiter] Background cleanup task dihentikan.")

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                await self._cleanup(time.time())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[RateLimiter] Error di cleanup loop: {e}", exc_info=True)
                
                await asyncio.sleep(10)

    async def check(
        self,
        identifier:  str,
        limit:       int,
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
                oldest      = state.requests[0]
                reset_at    = oldest + WINDOW_SECONDS
                retry_after = max(reset_at - now, 0.1)
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

            if (
                self._cleanup_task is None or self._cleanup_task.done()
            ) and (now - self._last_cleanup > CLEANUP_INTERVAL_SECONDS):
                asyncio.ensure_future(self._cleanup_safe(now))

            return RateLimitResult(
                allowed   = True,
                limit     = limit,
                remaining = remaining,
                reset_at  = reset_at,
            )

    async def _cleanup_safe(self, now: float) -> None:

        try:
            await self._cleanup(now)
        except Exception as e:
            logger.error(f"[RateLimiter] Error saat cleanup: {e}", exc_info=True)

    async def _cleanup(self, now: float) -> None:

        async with self._lock:
            cutoff     = now - STALE_THRESHOLD_SECONDS
            stale_keys = [
                k for k, s in self._states.items()
                if s.last_seen < cutoff
            ]
            for k in stale_keys:
                del self._states[k]

            if stale_keys:
                logger.info(
                    f"[RateLimiter] Cleanup: hapus {len(stale_keys)} stale entries. "
                    f"Sisa: {len(self._states)} entries."
                )
            else:
                logger.debug(
                    f"[RateLimiter] Cleanup: tidak ada stale entries. "
                    f"Total: {len(self._states)} entries."
                )

            self._last_cleanup = now

    def get_stats(self) -> Dict[str, int]:
        return {
            "tracked_identifiers": len(self._states),
            "cleanup_task_active": int(
                self._cleanup_task is not None and not self._cleanup_task.done()
            ),
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
