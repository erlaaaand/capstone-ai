# core/brute_force.py
# Single source of truth untuk PBKDF2 brute-force guard.
# Digunakan oleh core/security.py — jangan mendefinisikan ulang di tempat lain.

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

from core.logger import get_logger

logger = get_logger(__name__)

_PBKDF2_FAIL_WINDOW = 60   # detik — window pengamatan kegagalan
_PBKDF2_FAIL_MAX    = 10   # maks kegagalan sebelum lockout
_PBKDF2_LOCKOUT_SEC = 30   # durasi lockout dalam detik


@dataclass
class _FailRecord:
    timestamps:   List[float] = field(default_factory=list)
    locked_until: float       = 0.0


class PBKDF2GuardState:
    """Thread-safe guard terhadap serangan brute-force pada endpoint autentikasi PBKDF2."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._records: Dict[str, _FailRecord] = {}

    def is_locked(self, prefix: str) -> bool:
        with self._lock:
            rec = self._records.get(prefix)
            return bool(rec and rec.locked_until and time.time() < rec.locked_until)

    def record_failure(self, prefix: str) -> None:
        now = time.time()
        with self._lock:
            rec    = self._records.setdefault(prefix, _FailRecord())
            cutoff = now - _PBKDF2_FAIL_WINDOW
            rec.timestamps = [t for t in rec.timestamps if t >= cutoff]
            rec.timestamps.append(now)
            if len(rec.timestamps) >= _PBKDF2_FAIL_MAX:
                rec.locked_until = now + _PBKDF2_LOCKOUT_SEC
                logger.warning(
                    f"[BruteForceGuard] PBKDF2 brute-force guard activated for "
                    f"prefix={prefix!r} — locked {_PBKDF2_LOCKOUT_SEC}s."
                )

    def record_success(self, prefix: str) -> None:
        with self._lock:
            self._records.pop(prefix, None)


# Singleton instance — import dan gunakan langsung.
pbkdf2_guard = PBKDF2GuardState()