"""
tests/test_export_pipeline.py — BARU (FIX #8 coverage):

  Test untuk memverifikasi bahwa _sanitize_model_config() bekerja
  secara struktural (aman) dan tidak corrupt nama layer atau field lain
  yang kebetulan mengandung substring target.
"""

import json
import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Import fungsi yang di-fix
# ─────────────────────────────────────────────────────────────────────────────

# Guard: jika tensorflow tidak tersedia (CI environment), skip semua test ini
tf_available = True
try:
    import tensorflow as tf
    from pipelines.export_to_onnx import (
        _replace_activation_in_config,
        _replace_dtype_in_config,
        _sanitize_model_config,
    )
except ImportError:
    tf_available = False

pytestmark = pytest.mark.skipif(
    not tf_available,
    reason="TensorFlow tidak tersedia di environment ini."
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _layer_config(name: str, activation: str) -> dict:
    """Buat config layer Dense sederhana."""
    return {
        "class_name": "Dense",
        "name": name,
        "config": {
            "name": name,
            "units": 512,
            "activation": activation,
            "dtype": "float32",
        },
    }


def _model_config_with_layer(layer_name: str, activation: str) -> str:
    """Buat JSON model config minimal dengan satu layer."""
    config = {
        "class_name": "Functional",
        "config": {
            "name": "TestModel",
            "layers": [
                {
                    "module": "keras.layers",
                    "class_name": "Dense",
                    "config": _layer_config(layer_name, activation)["config"],
                    "name": layer_name,
                }
            ],
        },
    }
    return json.dumps(config)


# ─────────────────────────────────────────────────────────────────────────────
#  _replace_activation_in_config
# ─────────────────────────────────────────────────────────────────────────────

class TestReplaceActivationInConfig:

    def test_replaces_gelu_string_activation(self):
        """Field 'activation' dengan nilai string 'gelu' harus diganti."""
        config = {"activation": "gelu", "units": 512}
        _replace_activation_in_config(config, "gelu", "swish")
        assert config["activation"] == "swish"

    def test_does_not_replace_layer_name_containing_gelu(self):
        """
        KASUS KRITIS: Nama layer yang mengandung 'gelu' TIDAK boleh diubah.
        Bug sebelumnya: str.replace('"gelu"', '"swish"') akan mengubah
        nama layer "dense_gelu_output" menjadi "dense_swish_output".
        """
        config = {
            "name": "dense_gelu_output",   # nama layer — JANGAN diubah
            "activation": "gelu",           # activation — ini yang diubah
            "units": 256,
        }
        _replace_activation_in_config(config, "gelu", "swish")
        assert config["name"] == "dense_gelu_output", (
            "Nama layer tidak boleh berubah meskipun mengandung 'gelu'"
        )
        assert config["activation"] == "swish"

    def test_does_not_replace_description_field_containing_gelu(self):
        """Field 'description' yang mengandung 'gelu' tidak boleh diubah."""
        config = {
            "description": "This layer uses gelu activation for better gradients",
            "activation": "relu",   # bukan gelu, tidak diubah
        }
        _replace_activation_in_config(config, "gelu", "swish")
        assert "gelu" in config["description"], (
            "Field description tidak boleh dimodifikasi"
        )

    def test_replaces_nested_activation(self):
        """Activation di dalam nested config harus ter-replace."""
        config = {
            "class_name": "Dense",
            "config": {
                "activation": "gelu",
                "units": 128,
            },
        }
        _replace_activation_in_config(config, "gelu", "swish")
        assert config["config"]["activation"] == "swish"

    def test_replaces_activation_in_list_of_layers(self):
        """Activation di dalam list layer harus ter-replace."""
        config = [
            {"name": "layer1", "activation": "gelu"},
            {"name": "layer2", "activation": "relu"},
        ]
        _replace_activation_in_config(config, "gelu", "swish")
        assert config[0]["activation"] == "swish"
        assert config[1]["activation"] == "relu"   # tidak berubah

    def test_non_target_activation_unchanged(self):
        """Activation selain target ('relu', 'tanh', dll) tidak boleh berubah."""
        config = {"activation": "relu", "units": 64}
        _replace_activation_in_config(config, "gelu", "swish")
        assert config["activation"] == "relu"

    def test_empty_config_no_crash(self):
        """Config kosong tidak boleh crash."""
        _replace_activation_in_config({}, "gelu", "swish")
        _replace_activation_in_config([], "gelu", "swish")

    def test_deeply_nested_activation(self):
        """Activation yang sangat dalam (3+ level) harus ter-replace."""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "activation": "gelu"
                    }
                }
            }
        }
        _replace_activation_in_config(config, "gelu", "swish")
        assert config["level1"]["level2"]["level3"]["activation"] == "swish"


# ─────────────────────────────────────────────────────────────────────────────
#  _replace_dtype_in_config
# ─────────────────────────────────────────────────────────────────────────────

class TestReplaceDtypeInConfig:

    def test_replaces_dtype_field(self):
        config = {"dtype": "mixed_float16", "units": 512}
        _replace_dtype_in_config(config, "mixed_float16", "float32")
        assert config["dtype"] == "float32"

    def test_replaces_compute_dtype(self):
        config = {"compute_dtype": "float16"}
        _replace_dtype_in_config(config, "float16", "float32")
        assert config["compute_dtype"] == "float32"

    def test_does_not_replace_layer_name_containing_dtype_word(self):
        """
        Nama layer yang mengandung 'float16' sebagai bagian string
        tidak boleh diubah.
        """
        config = {
            "name": "dense_float16_custom",   # nama layer — JANGAN diubah
            "dtype": "mixed_float16",          # dtype — ini yang diubah
        }
        _replace_dtype_in_config(config, "mixed_float16", "float32")
        assert config["name"] == "dense_float16_custom"
        assert config["dtype"] == "float32"

    def test_replaces_policy_config(self):
        """Policy object dengan format Keras harus ter-replace dengan benar."""
        config = {
            "class_name": "Policy",
            "config": {
                "name": "mixed_float16",
            },
        }
        _replace_dtype_in_config(config, "mixed_float16", "float32")
        assert config["config"]["name"] == "float32"

    def test_non_target_dtype_unchanged(self):
        config = {"dtype": "float32"}
        _replace_dtype_in_config(config, "mixed_float16", "float32")
        assert config["dtype"] == "float32"   # sudah benar, tidak berubah


# ─────────────────────────────────────────────────────────────────────────────
#  _sanitize_model_config — integrasi
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeModelConfig:

    def test_output_is_valid_json(self):
        """Hasil sanitasi harus JSON yang valid."""
        model_json = _model_config_with_layer("dense_1", "gelu")
        result     = _sanitize_model_config(model_json)
        # Tidak boleh raise JSONDecodeError
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_gelu_activation_replaced_with_swish(self):
        """activation 'gelu' di dalam JSON harus diganti dengan 'swish'."""
        config = {
            "class_name": "Functional",
            "config": {
                "layers": [{"config": {"activation": "gelu", "name": "dense_1"}}]
            }
        }
        result = _sanitize_model_config(json.dumps(config))
        parsed = json.loads(result)
        assert parsed["config"]["layers"][0]["config"]["activation"] == "swish"

    def test_layer_name_with_gelu_preserved(self):
        """
        KASUS KRITIS: Layer bernama 'dense_gelu_output' tidak boleh corrupt.
        Ini adalah bug utama yang diperbaiki di fix #8.
        """
        config = {
            "class_name": "Functional",
            "config": {
                "layers": [{
                    "name": "dense_gelu_output",   # nama layer mengandung 'gelu'
                    "config": {
                        "name": "dense_gelu_output",
                        "activation": "gelu",      # ini yang diganti
                    },
                }]
            }
        }
        result = _sanitize_model_config(json.dumps(config))
        parsed = json.loads(result)

        layer = parsed["config"]["layers"][0]
        assert layer["name"] == "dense_gelu_output", (
            "BUG #8: Nama layer tidak boleh berubah!"
        )
        assert layer["config"]["name"] == "dense_gelu_output", (
            "BUG #8: config.name tidak boleh berubah!"
        )
        assert layer["config"]["activation"] == "swish", (
            "Activation harus diganti ke swish"
        )

    def test_mixed_float16_replaced(self):
        """mixed_float16 di dtype harus diganti float32."""
        config = {
            "class_name": "Policy",
            "config": {"name": "mixed_float16"},
        }
        outer = {"dtype": json.dumps(config), "layers": [{"dtype": "mixed_float16"}]}
        result = _sanitize_model_config(json.dumps(outer))
        parsed = json.loads(result)
        assert parsed["layers"][0]["dtype"] == "float32"

    def test_invalid_json_raises_error(self):
        """JSON tidak valid harus raise exception, bukan silent fail."""
        with pytest.raises(json.JSONDecodeError):
            _sanitize_model_config("{ this is not valid json }")

    def test_result_round_trips_correctly(self):
        """
        JSON yang di-sanitasi harus bisa di-parse dan hasilnya konsisten
        jika di-sanitasi lagi (idempotent untuk operasi yang sudah benar).
        """
        config = {
            "config": {
                "layers": [
                    {"config": {"activation": "gelu", "name": "layer1"}},
                    {"config": {"activation": "relu", "name": "layer2"}},
                ]
            }
        }
        result1 = _sanitize_model_config(json.dumps(config))
        result2 = _sanitize_model_config(result1)   # sanitasi kedua kali

        parsed1 = json.loads(result1)
        parsed2 = json.loads(result2)

        # Hasil harus sama (idempotent setelah first pass)
        assert parsed1 == parsed2
