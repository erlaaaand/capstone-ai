"""
Tests untuk core/security.py — APIKeyManager, hashing, AuthResult, scopes.
"""
import os
import time
from unittest.mock import patch

import pytest

from core.security import (
    APIKeyManager,
    AuthResult,
    KeyScope,
    RateLimitTier,
    TIER_LIMITS,
    generate_api_key,
    get_key_prefix,
    hash_api_key,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper
# ─────────────────────────────────────────────────────────────────────────────

def fresh_manager() -> APIKeyManager:
    """Buat instance APIKeyManager baru (reset singleton untuk isolasi test)."""
    APIKeyManager._instance = None
    mgr = APIKeyManager()
    mgr._loaded = False
    mgr._keys = {}
    return mgr


# ─────────────────────────────────────────────────────────────────────────────
#  generate_api_key
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateApiKey:
    def test_live_key_starts_with_dk_live(self):
        key = generate_api_key(live=True)
        assert key.startswith("dk_live_")

    def test_test_key_starts_with_dk_test(self):
        key = generate_api_key(live=False)
        assert key.startswith("dk_test_")

    def test_key_minimum_length(self):
        key = generate_api_key()
        assert len(key) > 20

    def test_two_keys_are_unique(self):
        k1 = generate_api_key()
        k2 = generate_api_key()
        assert k1 != k2


# ─────────────────────────────────────────────────────────────────────────────
#  hash_api_key & verify
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyHashing:
    def test_hash_returns_salt_colon_hash(self):
        result = hash_api_key("dk_live_testkey123")
        assert ":" in result
        parts = result.split(":", 1)
        assert len(parts) == 2
        assert all(parts)

    def test_same_key_different_hash_each_time(self):
        """Karena salt acak, hash berbeda tiap pemanggilan."""
        h1 = hash_api_key("dk_live_testkey123")
        h2 = hash_api_key("dk_live_testkey123")
        assert h1 != h2

    def test_get_key_prefix_short_key(self):
        prefix = get_key_prefix("short")
        assert "..." in prefix

    def test_get_key_prefix_long_key(self):
        key    = "dk_live_" + "x" * 32
        prefix = get_key_prefix(key)
        assert prefix.endswith("...")
        assert len(prefix) <= 16


# ─────────────────────────────────────────────────────────────────────────────
#  KeyScope & RateLimitTier
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyScope:
    def test_all_scopes_exist(self):
        assert KeyScope.PREDICT  == "predict"
        assert KeyScope.HEALTH   == "health"
        assert KeyScope.ADMIN    == "admin"
        assert KeyScope.READONLY == "readonly"

    def test_scope_is_string_enum(self):
        assert isinstance(KeyScope.PREDICT.value, str)


class TestRateLimitTier:
    def test_tier_limits_defined(self):
        assert TIER_LIMITS[RateLimitTier.FREE]      == 60
        assert TIER_LIMITS[RateLimitTier.STANDARD]  == 300
        assert TIER_LIMITS[RateLimitTier.PREMIUM]   == 1000
        assert TIER_LIMITS[RateLimitTier.UNLIMITED] >= 999_000

    def test_tiers_ascending_order(self):
        assert (
            TIER_LIMITS[RateLimitTier.FREE]
            < TIER_LIMITS[RateLimitTier.STANDARD]
            < TIER_LIMITS[RateLimitTier.PREMIUM]
            < TIER_LIMITS[RateLimitTier.UNLIMITED]
        )


# ─────────────────────────────────────────────────────────────────────────────
#  APIKeyManager
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIKeyManager:
    RAW_KEY = "dk_live_UnitTestKeyForSecurityTests1234"

    def _manager_with_key(
        self,
        raw_key: str = None,
        scopes: str = "predict,health",
        tier: str = "standard",
        deprecated: bool = False,
        expires_at: str = None,
    ) -> APIKeyManager:
        raw_key = raw_key or self.RAW_KEY
        env = {
            "API_KEY_1":          raw_key,
            "API_KEY_1_NAME":     "Test Key",
            "API_KEY_1_SCOPES":   scopes,
            "API_KEY_1_TIER":     tier,
            "API_KEY_1_DEPRECATED": "true" if deprecated else "false",
        }
        if expires_at:
            env["API_KEY_1_EXPIRES_AT"] = expires_at

        mgr = fresh_manager()
        with patch.dict(os.environ, env, clear=False):
            mgr.load_keys()
        return mgr

    # ── Load Keys ────────────────────────────────────────────────────────────

    def test_load_keys_registers_one_key(self):
        mgr = self._manager_with_key()
        assert len(mgr._keys) == 1

    def test_load_multiple_keys(self):
        key2 = "dk_live_SecondKeyForMultipleKeyTest1234"
        mgr  = fresh_manager()
        with patch.dict(os.environ, {
            "API_KEY_1": self.RAW_KEY, "API_KEY_1_NAME": "K1",
            "API_KEY_1_SCOPES": "predict", "API_KEY_1_TIER": "standard",
            "API_KEY_2": key2, "API_KEY_2_NAME": "K2",
            "API_KEY_2_SCOPES": "predict,admin", "API_KEY_2_TIER": "premium",
        }, clear=False):
            mgr.load_keys()
        assert len(mgr._keys) == 2

    # ── Validate — success ───────────────────────────────────────────────────

    def test_valid_key_returns_valid_auth_result(self):
        mgr    = self._manager_with_key()
        result = mgr.validate(self.RAW_KEY)
        assert result.valid is True

    def test_valid_key_has_correct_name(self):
        mgr    = self._manager_with_key()
        result = mgr.validate(self.RAW_KEY)
        assert result.key_name == "Test Key"

    def test_valid_key_has_correct_scopes(self):
        mgr    = self._manager_with_key(scopes="predict,health,admin")
        result = mgr.validate(self.RAW_KEY)
        assert KeyScope.PREDICT in result.scopes
        assert KeyScope.HEALTH  in result.scopes
        assert KeyScope.ADMIN   in result.scopes

    def test_valid_key_has_correct_tier(self):
        mgr    = self._manager_with_key(tier="premium")
        result = mgr.validate(self.RAW_KEY)
        assert result.tier == RateLimitTier.PREMIUM

    def test_not_deprecated_by_default(self):
        mgr    = self._manager_with_key()
        result = mgr.validate(self.RAW_KEY)
        assert result.deprecated is False

    def test_deprecated_key_still_valid_but_flagged(self):
        mgr    = self._manager_with_key(deprecated=True)
        result = mgr.validate(self.RAW_KEY)
        assert result.valid      is True
        assert result.deprecated is True

    # ── Validate — failures ──────────────────────────────────────────────────

    def test_wrong_key_returns_invalid(self):
        mgr    = self._manager_with_key()
        result = mgr.validate("dk_live_WrongKeyThatDoesNotMatch12345")
        assert result.valid is False

    def test_empty_key_returns_invalid(self):
        mgr    = self._manager_with_key()
        result = mgr.validate("")
        assert result.valid is False

    def test_whitespace_only_key_returns_invalid(self):
        mgr    = self._manager_with_key()
        result = mgr.validate("   ")
        assert result.valid is False

    def test_expired_key_returns_invalid(self):
        past_ts = str(int(time.time()) - 3600)  # 1 jam lalu
        mgr     = self._manager_with_key(expires_at=past_ts)
        result  = mgr.validate(self.RAW_KEY)
        assert result.valid is False
        assert "kadaluarsa" in result.error.lower()

    def test_future_expiry_still_valid(self):
        future_ts = str(int(time.time()) + 3600)  # 1 jam mendatang
        mgr       = self._manager_with_key(expires_at=future_ts)
        result    = mgr.validate(self.RAW_KEY)
        assert result.valid is True

    # ── Tier limits ──────────────────────────────────────────────────────────

    def test_get_tier_limit_free(self):
        mgr = fresh_manager()
        assert mgr.get_tier_limit(RateLimitTier.FREE) == 60

    def test_get_tier_limit_standard(self):
        mgr = fresh_manager()
        assert mgr.get_tier_limit(RateLimitTier.STANDARD) == 300

    def test_get_tier_limit_premium(self):
        mgr = fresh_manager()
        assert mgr.get_tier_limit(RateLimitTier.PREMIUM) == 1000

    def test_get_tier_limit_unknown_falls_back_to_free(self):
        mgr = fresh_manager()
        # passing string bukan RateLimitTier
        result = mgr.get_tier_limit("nonexistent_tier")  # type: ignore
        assert result == TIER_LIMITS[RateLimitTier.FREE]

    # ── _parse_scopes ────────────────────────────────────────────────────────

    def test_parse_scopes_single(self):
        scopes = APIKeyManager._parse_scopes("predict")
        assert KeyScope.PREDICT in scopes

    def test_parse_scopes_multiple(self):
        scopes = APIKeyManager._parse_scopes("predict,health,admin")
        assert KeyScope.PREDICT in scopes
        assert KeyScope.HEALTH  in scopes
        assert KeyScope.ADMIN   in scopes

    def test_parse_scopes_unknown_ignored(self):
        scopes = APIKeyManager._parse_scopes("predict,unknown_scope")
        assert KeyScope.PREDICT in scopes
        assert len(scopes) == 1

    def test_parse_scopes_empty_falls_back_to_predict(self):
        scopes = APIKeyManager._parse_scopes("")
        assert KeyScope.PREDICT in scopes

    # ── _parse_tier ──────────────────────────────────────────────────────────

    def test_parse_tier_valid(self):
        assert APIKeyManager._parse_tier("premium") == RateLimitTier.PREMIUM

    def test_parse_tier_case_insensitive(self):
        assert APIKeyManager._parse_tier("PREMIUM") == RateLimitTier.PREMIUM
        assert APIKeyManager._parse_tier("Standard") == RateLimitTier.STANDARD

    def test_parse_tier_unknown_falls_back_to_standard(self):
        tier = APIKeyManager._parse_tier("ultra_super")
        assert tier == RateLimitTier.STANDARD

    # ── _parse_expiry ────────────────────────────────────────────────────────

    def test_parse_expiry_valid_timestamp(self):
        ts = str(int(time.time()) + 3600)
        assert APIKeyManager._parse_expiry(ts) == float(ts)

    def test_parse_expiry_none_returns_none(self):
        assert APIKeyManager._parse_expiry(None) is None

    def test_parse_expiry_empty_returns_none(self):
        assert APIKeyManager._parse_expiry("") is None

    def test_parse_expiry_invalid_returns_none(self):
        assert APIKeyManager._parse_expiry("not-a-number") is None


# ─────────────────────────────────────────────────────────────────────────────
#  AuthResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthResult:
    def test_default_invalid_result(self):
        result = AuthResult(valid=False, error="test error")
        assert result.valid      is False
        assert result.key_prefix == ""
        assert result.scopes     == set()
        assert result.deprecated is False

    def test_valid_result_fields(self):
        result = AuthResult(
            valid=True,
            key_prefix="dk_live_...",
            key_name="My App",
            scopes={KeyScope.PREDICT, KeyScope.HEALTH},
            tier=RateLimitTier.STANDARD,
        )
        assert result.valid
        assert KeyScope.PREDICT in result.scopes
        assert result.tier == RateLimitTier.STANDARD
