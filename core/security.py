# core/security.py

import asyncio
import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from core.logger import get_logger

logger = get_logger(__name__)


class KeyScope(str, Enum):
    PREDICT  = "predict"
    HEALTH   = "health"
    ADMIN    = "admin"
    READONLY = "readonly"


class RateLimitTier(str, Enum):
    FREE      = "free"
    STANDARD  = "standard"
    PREMIUM   = "premium"
    UNLIMITED = "unlimited"


TIER_LIMITS: Dict[RateLimitTier, int] = {
    RateLimitTier.FREE:      60,
    RateLimitTier.STANDARD:  300,
    RateLimitTier.PREMIUM:   1_000,
    RateLimitTier.UNLIMITED: 999_999,
}

KEY_PREFIX_LIVE = "dk_live_"
KEY_PREFIX_TEST = "dk_test_"

_PBKDF2_FAIL_WINDOW = 60
_PBKDF2_FAIL_MAX    = 10
_PBKDF2_LOCKOUT_SEC = 30


@dataclass
class _FailRecord:
    timestamps:   List[float] = field(default_factory=list)
    locked_until: float       = 0.0


class _PBKDF2GuardState:
    """Melindungi endpoint auth dari serangan brute-force PBKDF2."""

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
                    f"[Security] PBKDF2 brute-force guard activated for "
                    f"prefix={prefix!r} — locked {_PBKDF2_LOCKOUT_SEC}s."
                )

    def record_success(self, prefix: str) -> None:
        with self._lock:
            self._records.pop(prefix, None)


_pbkdf2_guard = _PBKDF2GuardState()


@dataclass
class APIKeyRecord:
    key_hash:   str
    key_prefix: str
    name:       str
    scopes:     Set[KeyScope]
    tier:       RateLimitTier
    active:     bool            = True
    deprecated: bool            = False
    created_at: float           = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class AuthResult:
    valid:      bool
    key_prefix: str           = ""
    key_name:   str           = ""
    scopes:     Set[KeyScope] = field(default_factory=set)
    tier:       RateLimitTier = RateLimitTier.FREE
    deprecated: bool          = False
    error:      str           = ""


def _get_pbkdf2_iterations() -> int:
    """
    Baca iterasi PBKDF2 dari settings.
    Fallback ke nilai OWASP-safe (600k) selama fase bootstrap.
    """
    try:
        from core.config import settings
        return settings.PBKDF2_ITERATIONS
    except Exception:
        return 600_000


def _hash_key(raw_key: str, salt: Optional[str] = None) -> tuple[str, str]:
    iterations = _get_pbkdf2_iterations()

    if salt is None:
        salt_bytes = os.urandom(16)
        salt_hex   = salt_bytes.hex()
    else:
        salt_bytes = bytes.fromhex(salt)
        salt_hex   = salt

    dk = hashlib.pbkdf2_hmac(
        hash_name  = "sha256",
        password   = raw_key.encode("utf-8"),
        salt       = salt_bytes,
        iterations = iterations,
        dklen      = 32,
    )
    return dk.hex(), salt_hex


def _verify_key(raw_key: str, stored_hash: str) -> bool:
    try:
        salt_hex, expected_hash = stored_hash.split(":", 1)
        computed_hash, _        = _hash_key(raw_key, salt=salt_hex)
        return hmac.compare_digest(
            computed_hash.encode("ascii"),
            expected_hash.encode("ascii"),
        )
    except Exception:
        return False


# ── Public key-management utilities ───────────────────────────────────────────

def hash_api_key(raw_key: str) -> str:
    """Hash API key untuk disimpan di env/database. Gunakan saat provisioning key baru."""
    dk, salt = _hash_key(raw_key)
    return f"{salt}:{dk}"


def generate_api_key(live: bool = True) -> str:
    """Generate API key baru. Gunakan saat provisioning key baru."""
    prefix = KEY_PREFIX_LIVE if live else KEY_PREFIX_TEST
    return f"{prefix}{secrets.token_urlsafe(24)}"


def get_key_prefix(raw_key: str) -> str:
    """Ambil 12 karakter pertama key sebagai identifier aman untuk logging."""
    if len(raw_key) <= 12:
        return raw_key[:4] + "..."
    return raw_key[:12] + "..."


# ── APIKeyManager ──────────────────────────────────────────────────────────────

class APIKeyManager:
    _singleton_lock: threading.Lock            = threading.Lock()
    _instance:       Optional["APIKeyManager"] = None

    def __new__(cls) -> "APIKeyManager":
        if cls._instance is not None:
            return cls._instance
        with cls._singleton_lock:
            if cls._instance is None:
                inst            = super().__new__(cls)
                inst._keys      = {}
                inst._loaded    = False
                inst._load_lock = threading.RLock()
                cls._instance   = inst
        return cls._instance

    def load_keys(self) -> None:
        load_dotenv(override=True)

        with self._load_lock:
            self._keys   = {}
            loaded_count = 0

            for i in range(1, 20):
                raw_key = os.getenv(f"API_KEY_{i}")
                if not raw_key:
                    continue

                self._register_key(
                    raw_key    = raw_key,
                    name       = os.getenv(f"API_KEY_{i}_NAME",   f"Key #{i}"),
                    scopes     = self._parse_scopes(os.getenv(f"API_KEY_{i}_SCOPES", "predict,health")),
                    tier       = self._parse_tier(os.getenv(f"API_KEY_{i}_TIER", "standard")),
                    expires_at = self._parse_expiry(os.getenv(f"API_KEY_{i}_EXPIRES_AT")),
                    deprecated = os.getenv(f"API_KEY_{i}_DEPRECATED", "false").lower() == "true",
                )
                loaded_count += 1

            if loaded_count == 0:
                legacy_key = os.getenv("API_KEY")
                if legacy_key:
                    self._register_key(
                        raw_key = legacy_key,
                        name    = "Default Key (Legacy)",
                        scopes  = {KeyScope.PREDICT, KeyScope.HEALTH},
                        tier    = RateLimitTier.STANDARD,
                    )
                    loaded_count += 1
                    logger.warning(
                        "Menggunakan legacy API_KEY. Upgrade ke API_KEY_1/API_KEY_2 "
                        "untuk fitur enterprise penuh."
                    )

            if loaded_count == 0:
                logger.critical(
                    "TIDAK ADA API KEY yang dikonfigurasi! "
                    "Set API_KEY_1 di environment variables. "
                    "Semua request akan ditolak."
                )

            self._loaded = True

    def _register_key(
        self,
        raw_key:    str,
        name:       str,
        scopes:     Set[KeyScope],
        tier:       RateLimitTier,
        expires_at: Optional[float] = None,
        deprecated: bool = False,
    ) -> None:
        key_hash   = hash_api_key(raw_key)
        key_prefix = get_key_prefix(raw_key)
        record     = APIKeyRecord(
            key_hash   = key_hash,
            key_prefix = key_prefix,
            name       = name,
            scopes     = scopes,
            tier       = tier,
            expires_at = expires_at,
            deprecated = deprecated,
        )

        bucket = self._keys.setdefault(key_prefix, [])
        bucket.append(record)

        if len(bucket) > 1:
            logger.warning(
                f"  Prefix collision terdeteksi: prefix={key_prefix!r} "
                f"sekarang memiliki {len(bucket)} key. "
                "Pertimbangkan key dengan prefix unik untuk performa optimal."
            )

        logger.info(
            f"  API Key ter-load: prefix={key_prefix} name='{name}' "
            f"scopes={[s.value for s in scopes]} tier={tier.value}"
        )

    def loaded_key_count(self) -> int:
        """Jumlah total API key yang ter-load (semua bucket)."""
        with self._load_lock:
            return sum(len(bucket) for bucket in self._keys.values())

    def validate(self, raw_key: str) -> AuthResult:
        if not self._loaded:
            self.load_keys()

        if not raw_key or not raw_key.strip():
            return AuthResult(valid=False, error="API key tidak ada.")

        key_prefix = get_key_prefix(raw_key)

        if _pbkdf2_guard.is_locked(key_prefix):
            logger.warning(
                f"[Security] Request ditolak karena prefix terkunci (brute-force): "
                f"prefix={key_prefix!r}"
            )
            return AuthResult(
                valid      = False,
                error      = "Terlalu banyak percobaan autentikasi gagal. Coba lagi nanti.",
                key_prefix = key_prefix,
            )

        with self._load_lock:
            candidates = list(self._keys.get(key_prefix, []))

        return self._verify_candidates(raw_key, key_prefix, candidates)

    async def validate_async(self, raw_key: str) -> AuthResult:
        """
        Verifikasi PBKDF2 (CPU-bound) dijalankan di thread pool agar tidak
        memblokir event loop.
        """
        if not self._loaded:
            await asyncio.to_thread(self.load_keys)

        if not raw_key or not raw_key.strip():
            return AuthResult(valid=False, error="API key tidak ada.")

        key_prefix = get_key_prefix(raw_key)

        if _pbkdf2_guard.is_locked(key_prefix):
            logger.warning(
                f"[Security] Request ditolak karena prefix terkunci (brute-force): "
                f"prefix={key_prefix!r}"
            )
            return AuthResult(
                valid      = False,
                error      = "Terlalu banyak percobaan autentikasi gagal. Coba lagi nanti.",
                key_prefix = key_prefix,
            )

        with self._load_lock:
            candidates = list(self._keys.get(key_prefix, []))

        if not candidates:
            return AuthResult(valid=False, error="API key tidak valid.", key_prefix=key_prefix)

        return await asyncio.to_thread(
            self._verify_candidates, raw_key, key_prefix, candidates
        )

    def _verify_candidates(
        self,
        raw_key:    str,
        key_prefix: str,
        candidates: List[APIKeyRecord],
    ) -> AuthResult:
        for record in candidates:
            if _verify_key(raw_key, record.key_hash):
                _pbkdf2_guard.record_success(key_prefix)
                return self._build_auth_result(record)

        _pbkdf2_guard.record_failure(key_prefix)
        return AuthResult(valid=False, error="API key tidak valid.", key_prefix=key_prefix)

    def _build_auth_result(self, record: APIKeyRecord) -> AuthResult:
        if not record.active:
            logger.warning(f"Key nonaktif digunakan: {record.key_prefix}")
            return AuthResult(
                valid      = False,
                error      = "API key tidak aktif.",
                key_prefix = record.key_prefix,
            )

        if record.expires_at and time.time() > record.expires_at:
            logger.warning(f"Key kadaluarsa digunakan: {record.key_prefix}")
            return AuthResult(
                valid      = False,
                error      = "API key sudah kadaluarsa.",
                key_prefix = record.key_prefix,
            )

        if record.deprecated:
            logger.warning(
                f"Key deprecated digunakan: {record.key_prefix} ('{record.name}'). "
                "Segera ganti dengan key baru."
            )

        return AuthResult(
            valid      = True,
            key_prefix = record.key_prefix,
            key_name   = record.name,
            scopes     = record.scopes,
            tier       = record.tier,
            deprecated = record.deprecated,
        )

    def get_tier_limit(self, tier: RateLimitTier) -> int:
        return TIER_LIMITS.get(tier, TIER_LIMITS[RateLimitTier.FREE])

    @staticmethod
    def _parse_scopes(scope_str: str) -> Set[KeyScope]:
        scopes = set()
        for s in scope_str.split(","):
            try:
                scopes.add(KeyScope(s.strip().lower()))
            except ValueError:
                logger.warning(f"Scope tidak dikenal: '{s}', diabaikan.")
        return scopes or {KeyScope.PREDICT}

    @staticmethod
    def _parse_tier(tier_str: str) -> RateLimitTier:
        try:
            return RateLimitTier(tier_str.strip().lower())
        except ValueError:
            logger.warning(f"Tier tidak dikenal: '{tier_str}', menggunakan 'standard'.")
            return RateLimitTier.STANDARD

    @staticmethod
    def _parse_expiry(expiry_str: Optional[str]) -> Optional[float]:
        if not expiry_str:
            return None
        try:
            return float(expiry_str)
        except (ValueError, TypeError):
            return None


_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager