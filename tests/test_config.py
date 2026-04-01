"""
Tests untuk core/config.py — Settings, VARIETY_MAP, helper functions.
"""
import pytest
from unittest.mock import patch
import os


class TestVarietyMap:
    """Test VARIETY_MAP dan helper functions."""

    def test_variety_map_has_8_entries(self):
        from core.config import VARIETY_MAP
        assert len(VARIETY_MAP) == 8

    def test_all_expected_codes_present(self):
        from core.config import VARIETY_MAP
        expected = {"D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"}
        assert set(VARIETY_MAP.keys()) == expected

    def test_each_variety_has_required_fields(self):
        from core.config import VARIETY_MAP
        for code, info in VARIETY_MAP.items():
            assert info.display_name, f"{code} tidak punya display_name"
            assert info.local_name,   f"{code} tidak punya local_name"
            assert info.origin,       f"{code} tidak punya origin"
            assert info.description,  f"{code} tidak punya description"

    def test_musang_king_is_d200(self):
        from core.config import VARIETY_MAP
        assert VARIETY_MAP["D200"].display_name == "Musang King"

    def test_golden_phoenix_is_d197(self):
        from core.config import VARIETY_MAP
        assert VARIETY_MAP["D197"].display_name == "Golden Phoenix"

    def test_red_prawn_is_d198(self):
        from core.config import VARIETY_MAP
        assert VARIETY_MAP["D198"].display_name == "Red Prawn"

    def test_sultan_is_d24(self):
        from core.config import VARIETY_MAP
        assert VARIETY_MAP["D24"].display_name == "Sultan"

    def test_chanee_is_d2(self):
        from core.config import VARIETY_MAP
        assert VARIETY_MAP["D2"].display_name == "Chanee"


class TestGetVarietyInfo:
    """Test fungsi get_variety_info()."""

    def test_known_code_returns_correct_info(self):
        from core.config import get_variety_info
        info = get_variety_info("D200")
        assert info.display_name == "Musang King"

    def test_unknown_code_returns_unknown_variety(self):
        from core.config import get_variety_info
        info = get_variety_info("D999")
        assert "Tidak Dikenal" in info.display_name or info.display_name == "Varietas Tidak Dikenal"

    def test_strips_whitespace_from_code(self):
        from core.config import get_variety_info
        info = get_variety_info("  D200  ")
        assert info.display_name == "Musang King"

    def test_empty_string_returns_unknown(self):
        from core.config import get_variety_info
        info = get_variety_info("")
        assert info is not None  # tidak raise exception


class TestGetDisplayName:
    """Test fungsi get_display_name()."""

    def test_known_code_returns_display_name(self):
        from core.config import get_display_name
        assert get_display_name("D200") == "Musang King"

    def test_unknown_code_returns_code_itself(self):
        from core.config import get_display_name
        result = get_display_name("D999")
        assert result == "D999"

    def test_all_8_classes_have_display_names(self):
        from core.config import get_display_name
        codes = ["D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"]
        for code in codes:
            name = get_display_name(code)
            assert name != code, f"{code} seharusnya punya display name berbeda"


class TestSettings:
    """Test Settings class dan properties."""

    def test_default_image_size_is_224(self):
        from core.config import settings
        assert settings.IMAGE_SIZE == 224

    def test_default_num_classes_is_8(self):
        from core.config import settings
        assert settings.num_classes == 8

    def test_class_names_list_sorted_alphabetically(self):
        from core.config import settings
        names = settings.class_names_list
        assert names == sorted(names), "CLASS_NAMES harus urutan alfabetikal"

    def test_class_names_list_has_8_items(self):
        from core.config import settings
        assert len(settings.class_names_list) == 8

    def test_image_size_tuple_is_square(self):
        from core.config import settings
        h, w = settings.image_size_tuple
        assert h == w == settings.IMAGE_SIZE

    def test_allowed_extensions_set_contains_jpg(self):
        from core.config import settings
        assert "jpg" in settings.allowed_extensions_set
        assert "jpeg" in settings.allowed_extensions_set
        assert "png" in settings.allowed_extensions_set
        assert "webp" in settings.allowed_extensions_set

    def test_max_file_size_bytes_correct(self):
        from core.config import settings
        expected = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        assert settings.max_file_size_bytes == expected

    def test_model_abs_path_is_absolute(self):
        from core.config import settings
        assert settings.model_abs_path.is_absolute()

    def test_cors_origins_is_list(self):
        from core.config import settings
        assert isinstance(settings.CORS_ORIGINS, list)

    def test_allowed_hosts_is_list(self):
        from core.config import settings
        assert isinstance(settings.ALLOWED_HOSTS, list)

    def test_variety_map_property_returns_dict(self):
        from core.config import settings
        assert isinstance(settings.variety_map, dict)
        assert len(settings.variety_map) == 8

    def test_invalid_log_level_raises_error(self):
        from pydantic import ValidationError
        from core.config import Settings
        with pytest.raises((ValidationError, Exception)):
            Settings(LOG_LEVEL="INVALID_LEVEL")

    def test_image_size_too_small_raises_error(self):
        from pydantic import ValidationError
        from core.config import Settings
        with pytest.raises((ValidationError, Exception)):
            Settings(IMAGE_SIZE=10)

    def test_image_size_too_large_raises_error(self):
        from pydantic import ValidationError
        from core.config import Settings
        with pytest.raises((ValidationError, Exception)):
            Settings(IMAGE_SIZE=9999)

    def test_max_file_size_zero_raises_error(self):
        from pydantic import ValidationError
        from core.config import Settings
        with pytest.raises((ValidationError, Exception)):
            Settings(MAX_FILE_SIZE_MB=0)

    def test_get_settings_returns_singleton(self):
        from core.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
