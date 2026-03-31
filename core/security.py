import hashlib
import hmac
import os
import secrets
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
    FREE       = "free"
    STANDARD   = "standard"
    PREMIUM    = "premium"
    UNLIMITED  = "unlimited"


TIER_LIMITS: Dict[RateLimitTier, int] = {
    RateLimitTier.FREE:      60,
    RateLimitTier.STANDARD:  300,
    RateLimitTier.PREMIUM:   1000,
    RateLimitTier.UNLIMITED: 999_999,
}

KEY_PREFIX_LIVE = "dk_live_"
KEY_PREFIX_TEST = "dk_test_"
KEY_PREFIX_LEN  = 10


@dataclass
class APIKeyRecord:
    key_hash:    str
    key_prefix:  str
    name:        str
    scopes:      Set[KeyScope]
    tier:        RateLimitTier
    active:      bool       = True
    deprecated:  bool       = False
    created_at:  float      = field(default_factory=time.time)
    expires_at:  Optional[float] = None


@dataclass
class AuthResult:
    valid:       bool
    key_prefix:  str         = ""
    key_name:    str         = ""
    scopes:      Set[KeyScope] = field(default_factory=set)
    tier:        RateLimitTier = RateLimitTier.FREE
    deprecated:  bool        = False
    error:       str         = ""


def _hash_key(raw_key: str, salt: Optional[str] = None) -> tuple[str, str]:
    if salt is None:
        salt_bytes = os.urandom(16)
        salt_hex   = salt_bytes.hex()
    else:
        salt_bytes = bytes.fromhex(salt)
        salt_hex   = salt

    dk = hashlib.pbkdf2_hmac(
        hash_name   = "sha256",
        password    = raw_key.encode("utf-8"),
        salt        = salt_bytes,
        iterations  = 100_000,
        dklen       = 32,
    )
    return dk.hex(), salt_hex


def _verify_key(raw_key: str, stored_hash: str) -> bool:
    try:
        salt_hex, expected_hash = stored_hash.split(":", 1)
        computed_hash, _ = _hash_key(raw_key, salt=salt_hex)
        return hmac.compare_digest(
            computed_hash.encode("ascii"),
            expected_hash.encode("ascii"),
        )
    except (ValueError, Exception):
        return False


def hash_api_key(raw_key: str) -> str:
    dk, salt = _hash_key(raw_key)
    return f"{salt}:{dk}"


def generate_api_key(live: bool = True) -> str:
    prefix    = KEY_PREFIX_LIVE if live else KEY_PREFIX_TEST
    random_part = secrets.token_urlsafe(24)
    return f"{prefix}{random_part}"


def get_key_prefix(raw_key: str) -> str:
    if len(raw_key) <= 12:
        return raw_key[:4] + "..."
    return raw_key[:12] + "..."


class APIKeyManager:

    _instance: Optional["APIKeyManager"] = None
    _keys: Dict[str, APIKeyRecord] = {}

    def __new__(cls) -> "APIKeyManager":
        if cls._instance is None:
            inst          = super().__new__(cls)
            inst._keys    = {}
            inst._loaded  = False
            cls._instance = inst
        return cls._instance

    def load_keys(self) -> None:
        load_dotenv()
        self._keys = {}
        loaded_count = 0

        for i in range(1, 20):
            raw_key = os.getenv(f"API_KEY_{i}")
            if not raw_key:
                continue

            name   = os.getenv(f"API_KEY_{i}_NAME",   f"Key #{i}")
            scopes = self._parse_scopes(os.getenv(f"API_KEY_{i}_SCOPES", "predict,health"))
            tier   = self._parse_tier(os.getenv(f"API_KEY_{i}_TIER", "standard"))
            expiry = self._parse_expiry(os.getenv(f"API_KEY_{i}_EXPIRES_AT"))
            deprecated = os.getenv(f"API_KEY_{i}_DEPRECATED", "false").lower() == "true"

            self._register_key(raw_key, name, scopes, tier, expiry, deprecated)
            loaded_count += 1

        if loaded_count == 0:
            legacy_key = os.getenv("API_KEY")
            if legacy_key:
                self._register_key(
                    raw_key    = legacy_key,
                    name       = "Default Key (Legacy)",
                    scopes     = {KeyScope.PREDICT, KeyScope.HEALTH},
                    tier       = RateLimitTier.STANDARD,
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
        else:
            logger.info(f"API Key Manager: {loaded_count} key(s) ter-load.")

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
        self._keys[key_prefix] = record
        logger.info(
            f"  API Key ter-load: prefix={key_prefix} name='{name}' "
            f"scopes={[s.value for s in scopes]} tier={tier.value}"
        )

    def validate(self, raw_key: str) -> AuthResult:
        if not self._loaded:
            self.load_keys()

        if not raw_key or not raw_key.strip():
            return AuthResult(valid=False, error="API key tidak ada.")

        key_prefix = get_key_prefix(raw_key)

        for record in self._keys.values():
            if not _verify_key(raw_key, record.key_hash):
                continue

            if not record.active:
                logger.warning(f"Key nonaktif digunakan: {record.key_prefix}")
                return AuthResult(valid=False, error="API key tidak aktif.", key_prefix=record.key_prefix)

            if record.expires_at and time.time() > record.expires_at:
                logger.warning(f"Key kadaluarsa digunakan: {record.key_prefix}")
                return AuthResult(valid=False, error="API key sudah kadaluarsa.", key_prefix=record.key_prefix)

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

        return AuthResult(valid=False, error="API key tidak valid.", key_prefix=key_prefix)

    def get_tier_limit(self, tier: RateLimitTier) -> int:
        return TIER_LIMITS.get(tier, TIER_LIMITS[RateLimitTier.FREE])

    @staticmethod
    def _parse_scopes(scope_str: str) -> Set[KeyScope]:
        scopes = set()
        for s in scope_str.split(","):
            s = s.strip().lower()
            try:
                scopes.add(KeyScope(s))
            except ValueError:
                logger.warning(f"Scope tidak dikenal: '{s}', diabaikan.")
        return scopes if scopes else {KeyScope.PREDICT}

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