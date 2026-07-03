"""
Unit tests for src/config_manager.py

Covers:
- Happy-path loading from the real config/config.yaml
- Nested attribute access (_Namespace)
- Every validation rule (one bad value → ConfigurationError)
- CLI apply_overrides() — valid and invalid inputs
- Environment variable overrides — valid, invalid, and out-of-range
- Helper methods: get_model_path, get_output_dir, is_visualization_enabled
- to_dict() round-trip
- __repr__
"""

import os
import sys
import textwrap
import pytest

# ---------------------------------------------------------------------------
# Make repo root importable regardless of how pytest is invoked
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config_manager import ConfigManager, _Namespace
from exceptions import ConfigurationError


# ===========================================================================
# Helpers
# ===========================================================================

def _write_config(tmp_path, content: str) -> str:
    """Write a YAML string to a temp file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


VALID_YAML = """\
    model:
      type: "yolov8n"
      confidence_threshold: 0.25
      iou_threshold: 0.45
      weights_dir: "models"
      warmup_frames: 3
    device:
      preferred: "cpu"
      fallback: "cpu"
    tracker:
      config_file: "config/botsort.yaml"
      persist: true
    video:
      output_codec: "mp4v"
      skip_frames: 0
      max_resolution: null
    visualization:
      show_boxes: true
      show_ids: true
      show_unique_count: true
      show_fps: true
      trail_length: 30
      bbox_thickness: 2
      hud_alpha: 0.6
    export:
      output_dir: "outputs"
      save_video: true
      save_json: true
      save_csv: true
    analytics:
      ema_alpha: 0.1
    logging:
      level: "INFO"
      format: "%(asctime)s | %(message)s"
      date_format: "%Y-%m-%d %H:%M:%S"
      log_file: "visiontrack.log"
"""


# ===========================================================================
# _Namespace
# ===========================================================================

class TestNamespace:
    def test_flat_access(self):
        ns = _Namespace({"a": 1, "b": "hello"})
        assert ns.a == 1
        assert ns.b == "hello"

    def test_nested_access(self):
        ns = _Namespace({"outer": {"inner": 42}})
        assert isinstance(ns.outer, _Namespace)
        assert ns.outer.inner == 42

    def test_none_value_preserved(self):
        ns = _Namespace({"key": None})
        assert ns.key is None

    def test_list_value_preserved(self):
        ns = _Namespace({"dims": [1280, 720]})
        assert ns.dims == [1280, 720]

    def test_to_dict_flat(self):
        ns = _Namespace({"x": 10, "y": 20})
        assert ns.to_dict() == {"x": 10, "y": 20}

    def test_to_dict_nested(self):
        ns = _Namespace({"model": {"type": "yolov8n", "conf": 0.25}})
        d = ns.to_dict()
        assert d == {"model": {"type": "yolov8n", "conf": 0.25}}

    def test_repr_contains_keys(self):
        ns = _Namespace({"alpha": 1, "beta": 2})
        r = repr(ns)
        assert "alpha" in r
        assert "beta" in r


# ===========================================================================
# Happy-path loading
# ===========================================================================

class TestConfigManagerLoad:
    def test_loads_real_config(self):
        """Loads the actual project config/config.yaml without errors."""
        cfg = ConfigManager("config/config.yaml")
        assert cfg.model.type == "yolov8n"

    def test_loads_temp_valid_config(self, tmp_path):
        path = _write_config(tmp_path, VALID_YAML)
        cfg = ConfigManager(path)
        assert cfg.model.type == "yolov8n"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not found"):
            ConfigManager(str(tmp_path / "nonexistent.yaml"))

    def test_invalid_yaml_raises(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="parse"):
            ConfigManager(str(p))

    def test_non_mapping_yaml_raises(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            ConfigManager(str(p))

    def test_repr(self, tmp_path):
        path = _write_config(tmp_path, VALID_YAML)
        cfg = ConfigManager(path)
        r = repr(cfg)
        assert "ConfigManager" in r
        assert "yolov8n" in r
        assert "cpu" in r


# ===========================================================================
# Nested attribute access
# ===========================================================================

class TestNestedAccess:
    @pytest.fixture
    def cfg(self, tmp_path):
        return ConfigManager(_write_config(tmp_path, VALID_YAML))

    def test_model_section(self, cfg):
        assert cfg.model.type == "yolov8n"
        assert cfg.model.confidence_threshold == 0.25
        assert cfg.model.iou_threshold == 0.45
        assert cfg.model.weights_dir == "models"
        assert cfg.model.warmup_frames == 3

    def test_device_section(self, cfg):
        assert cfg.device.preferred == "cpu"
        assert cfg.device.fallback == "cpu"

    def test_tracker_section(self, cfg):
        assert cfg.tracker.config_file == "config/botsort.yaml"
        assert cfg.tracker.persist is True

    def test_video_section(self, cfg):
        assert cfg.video.output_codec == "mp4v"
        assert cfg.video.skip_frames == 0
        assert cfg.video.max_resolution is None

    def test_visualization_section(self, cfg):
        assert cfg.visualization.show_boxes is True
        assert cfg.visualization.show_ids is True
        assert cfg.visualization.trail_length == 30
        assert cfg.visualization.hud_alpha == 0.6

    def test_export_section(self, cfg):
        assert cfg.export.output_dir == "outputs"
        assert cfg.export.save_video is True
        assert cfg.export.save_json is True
        assert cfg.export.save_csv is True

    def test_analytics_section(self, cfg):
        assert cfg.analytics.ema_alpha == 0.1

    def test_logging_section(self, cfg):
        assert cfg.logging.level == "INFO"


# ===========================================================================
# Validation — model section
# ===========================================================================

class TestValidationModel:
    def _make(self, tmp_path, overrides: dict) -> str:
        import yaml, copy
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["model"].update(overrides)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        return str(p)

    def test_invalid_model_type(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not supported"):
            ConfigManager(self._make(tmp_path, {"type": "resnet50"}))

    def test_confidence_too_high(self, tmp_path):
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            ConfigManager(self._make(tmp_path, {"confidence_threshold": 1.5}))

    def test_confidence_negative(self, tmp_path):
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            ConfigManager(self._make(tmp_path, {"confidence_threshold": -0.1}))

    def test_iou_too_high(self, tmp_path):
        with pytest.raises(ConfigurationError, match="iou_threshold"):
            ConfigManager(self._make(tmp_path, {"iou_threshold": 2.0}))

    def test_warmup_negative(self, tmp_path):
        with pytest.raises(ConfigurationError, match="warmup_frames"):
            ConfigManager(self._make(tmp_path, {"warmup_frames": -1}))

    def test_warmup_non_integer(self, tmp_path):
        with pytest.raises(ConfigurationError, match="warmup_frames"):
            ConfigManager(self._make(tmp_path, {"warmup_frames": "three"}))

    def test_boundary_confidence_zero(self, tmp_path):
        """0.0 is a valid confidence threshold boundary."""
        cfg = ConfigManager(self._make(tmp_path, {"confidence_threshold": 0.0}))
        assert cfg.model.confidence_threshold == 0.0

    def test_boundary_confidence_one(self, tmp_path):
        """1.0 is a valid confidence threshold boundary."""
        cfg = ConfigManager(self._make(tmp_path, {"confidence_threshold": 1.0}))
        assert cfg.model.confidence_threshold == 1.0

    @pytest.mark.parametrize("model", ["yolov8n", "yolov8m", "yolo11x", "yolov9c"])
    def test_all_supported_model_types(self, tmp_path, model):
        cfg = ConfigManager(self._make(tmp_path, {"type": model}))
        assert cfg.model.type == model


# ===========================================================================
# Validation — device section
# ===========================================================================

class TestValidationDevice:
    def _make(self, tmp_path, overrides: dict) -> str:
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["device"].update(overrides)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        return str(p)

    def test_invalid_preferred(self, tmp_path):
        with pytest.raises(ConfigurationError, match="device.preferred"):
            ConfigManager(self._make(tmp_path, {"preferred": "tpu"}))

    def test_invalid_fallback(self, tmp_path):
        with pytest.raises(ConfigurationError, match="device.fallback"):
            ConfigManager(self._make(tmp_path, {"fallback": "gpu"}))

    @pytest.mark.parametrize("device", ["cuda", "mps", "cpu"])
    def test_all_valid_devices(self, tmp_path, device):
        cfg = ConfigManager(self._make(tmp_path, {"preferred": device}))
        assert cfg.device.preferred == device


# ===========================================================================
# Validation — video section
# ===========================================================================

class TestValidationVideo:
    def _make(self, tmp_path, overrides: dict) -> str:
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["video"].update(overrides)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        return str(p)

    def test_negative_skip_frames(self, tmp_path):
        with pytest.raises(ConfigurationError, match="skip_frames"):
            ConfigManager(self._make(tmp_path, {"skip_frames": -1}))

    def test_invalid_max_resolution_single_value(self, tmp_path):
        with pytest.raises(ConfigurationError, match="max_resolution"):
            ConfigManager(self._make(tmp_path, {"max_resolution": [1280]}))

    def test_invalid_max_resolution_negative(self, tmp_path):
        with pytest.raises(ConfigurationError, match="max_resolution"):
            ConfigManager(self._make(tmp_path, {"max_resolution": [-1, 720]}))

    def test_valid_max_resolution(self, tmp_path):
        cfg = ConfigManager(self._make(tmp_path, {"max_resolution": [1280, 720]}))
        assert cfg.video.max_resolution == [1280, 720]

    def test_null_max_resolution(self, tmp_path):
        cfg = ConfigManager(self._make(tmp_path, {"max_resolution": None}))
        assert cfg.video.max_resolution is None


# ===========================================================================
# Validation — visualization section
# ===========================================================================

class TestValidationVisualization:
    def _make(self, tmp_path, overrides: dict) -> str:
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["visualization"].update(overrides)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        return str(p)

    def test_invalid_show_boxes_type(self, tmp_path):
        with pytest.raises(ConfigurationError, match="show_boxes"):
            ConfigManager(self._make(tmp_path, {"show_boxes": "yes"}))

    def test_negative_trail_length(self, tmp_path):
        with pytest.raises(ConfigurationError, match="trail_length"):
            ConfigManager(self._make(tmp_path, {"trail_length": -5}))

    def test_hud_alpha_out_of_range(self, tmp_path):
        with pytest.raises(ConfigurationError, match="hud_alpha"):
            ConfigManager(self._make(tmp_path, {"hud_alpha": 1.5}))

    def test_trail_length_zero_is_valid(self, tmp_path):
        """trail_length=0 means no trails — still valid."""
        cfg = ConfigManager(self._make(tmp_path, {"trail_length": 0}))
        assert cfg.visualization.trail_length == 0


# ===========================================================================
# Validation — analytics & logging sections
# ===========================================================================

class TestValidationAnalyticsLogging:
    def _make_analytics(self, tmp_path, overrides: dict) -> str:
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["analytics"].update(overrides)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        return str(p)

    def _make_logging(self, tmp_path, overrides: dict) -> str:
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["logging"].update(overrides)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        return str(p)

    def test_ema_alpha_zero_raises(self, tmp_path):
        """ema_alpha must be > 0.0 (exclusive lower bound)."""
        with pytest.raises(ConfigurationError, match="ema_alpha"):
            ConfigManager(self._make_analytics(tmp_path, {"ema_alpha": 0.0}))

    def test_ema_alpha_too_large_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match="ema_alpha"):
            ConfigManager(self._make_analytics(tmp_path, {"ema_alpha": 1.5}))

    def test_ema_alpha_one_is_valid(self, tmp_path):
        """1.0 is the inclusive upper bound for ema_alpha."""
        cfg = ConfigManager(self._make_analytics(tmp_path, {"ema_alpha": 1.0}))
        assert cfg.analytics.ema_alpha == 1.0

    def test_invalid_log_level(self, tmp_path):
        with pytest.raises(ConfigurationError, match="level"):
            ConfigManager(self._make_logging(tmp_path, {"level": "VERBOSE"}))

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_all_valid_log_levels(self, tmp_path, level):
        cfg = ConfigManager(self._make_logging(tmp_path, {"level": level}))
        assert cfg.logging.level == level


# ===========================================================================
# Missing required sections
# ===========================================================================

class TestMissingSection:
    @pytest.mark.parametrize("section", [
        "model", "device", "tracker", "video",
        "visualization", "export", "analytics", "logging",
    ])
    def test_missing_section_raises(self, tmp_path, section):
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        del base[section]
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Missing"):
            ConfigManager(str(p))


# ===========================================================================
# apply_overrides (CLI)
# ===========================================================================

class TestApplyOverrides:
    @pytest.fixture
    def cfg(self, tmp_path):
        return ConfigManager(_write_config(tmp_path, VALID_YAML))

    def test_override_model(self, cfg):
        cfg.apply_overrides(model="yolov8m")
        assert cfg.model.type == "yolov8m"

    def test_override_confidence(self, cfg):
        cfg.apply_overrides(confidence=0.6)
        assert cfg.model.confidence_threshold == 0.6

    def test_override_device(self, cfg):
        cfg.apply_overrides(device="cpu")
        assert cfg.device.preferred == "cpu"

    def test_override_skip_frames(self, cfg):
        cfg.apply_overrides(skip_frames=2)
        assert cfg.video.skip_frames == 2

    def test_override_output_dir(self, cfg):
        cfg.apply_overrides(output_dir="custom_outputs")
        assert cfg.export.output_dir == "custom_outputs"

    def test_override_verbose_sets_debug(self, cfg):
        cfg.apply_overrides(verbose=True)
        assert cfg.logging.level == "DEBUG"

    def test_none_overrides_are_noop(self, cfg):
        """Passing None for every arg must not change any value."""
        original_model = cfg.model.type
        original_conf = cfg.model.confidence_threshold
        cfg.apply_overrides(model=None, confidence=None, device=None)
        assert cfg.model.type == original_model
        assert cfg.model.confidence_threshold == original_conf

    def test_invalid_model_raises(self, cfg):
        with pytest.raises(ConfigurationError, match="not supported"):
            cfg.apply_overrides(model="resnet50")

    def test_invalid_confidence_high(self, cfg):
        with pytest.raises(ConfigurationError, match="confidence"):
            cfg.apply_overrides(confidence=1.5)

    def test_invalid_confidence_negative(self, cfg):
        with pytest.raises(ConfigurationError, match="confidence"):
            cfg.apply_overrides(confidence=-0.1)

    def test_invalid_device_raises(self, cfg):
        with pytest.raises(ConfigurationError, match="not supported"):
            cfg.apply_overrides(device="tpu")

    def test_invalid_skip_frames_raises(self, cfg):
        with pytest.raises(ConfigurationError, match="skip-frames"):
            cfg.apply_overrides(skip_frames=-1)

    def test_multiple_overrides_applied_together(self, cfg):
        cfg.apply_overrides(model="yolov8l", confidence=0.5, device="cpu", skip_frames=1)
        assert cfg.model.type == "yolov8l"
        assert cfg.model.confidence_threshold == 0.5
        assert cfg.device.preferred == "cpu"
        assert cfg.video.skip_frames == 1


# ===========================================================================
# Environment variable overrides
# ===========================================================================

class TestEnvOverrides:
    @pytest.fixture
    def cfg_path(self, tmp_path):
        return _write_config(tmp_path, VALID_YAML)

    def test_device_env_override(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_DEVICE", "cpu")
        cfg = ConfigManager(cfg_path)
        assert cfg.device.preferred == "cpu"

    def test_model_env_override(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_MODEL", "yolov8m")
        cfg = ConfigManager(cfg_path)
        assert cfg.model.type == "yolov8m"

    def test_confidence_env_override(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_CONFIDENCE", "0.5")
        cfg = ConfigManager(cfg_path)
        assert cfg.model.confidence_threshold == 0.5

    def test_log_level_env_override(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_LOG_LEVEL", "DEBUG")
        cfg = ConfigManager(cfg_path)
        assert cfg.logging.level == "DEBUG"

    def test_invalid_device_env_ignored(self, cfg_path, monkeypatch):
        """An unsupported env device value is ignored — YAML default wins."""
        monkeypatch.setenv("VISIONTRACK_DEVICE", "tpu")
        cfg = ConfigManager(cfg_path)
        assert cfg.device.preferred == "cpu"  # YAML default

    def test_invalid_model_env_ignored(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_MODEL", "resnet50")
        cfg = ConfigManager(cfg_path)
        assert cfg.model.type == "yolov8n"  # YAML default

    def test_invalid_confidence_string_env_ignored(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_CONFIDENCE", "high")
        cfg = ConfigManager(cfg_path)
        assert cfg.model.confidence_threshold == 0.25  # YAML default

    def test_out_of_range_confidence_env_ignored(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_CONFIDENCE", "1.5")
        cfg = ConfigManager(cfg_path)
        assert cfg.model.confidence_threshold == 0.25  # YAML default

    def test_invalid_log_level_env_ignored(self, cfg_path, monkeypatch):
        monkeypatch.setenv("VISIONTRACK_LOG_LEVEL", "VERBOSE")
        cfg = ConfigManager(cfg_path)
        assert cfg.logging.level == "INFO"  # YAML default


# ===========================================================================
# Helper methods
# ===========================================================================

class TestHelperMethods:
    @pytest.fixture
    def cfg(self, tmp_path):
        return ConfigManager(_write_config(tmp_path, VALID_YAML))

    def test_get_model_path(self, cfg):
        path = cfg.get_model_path()
        assert path == "models/yolov8n.pt"

    def test_get_model_path_after_override(self, cfg):
        cfg.apply_overrides(model="yolov8m")
        assert cfg.get_model_path() == "models/yolov8m.pt"

    def test_get_output_dir(self, cfg):
        result = cfg.get_output_dir("run_20240115_143022")
        assert result == "outputs/run_20240115_143022"

    def test_get_output_dir_custom_base(self, cfg):
        cfg.apply_overrides(output_dir="my_results")
        assert cfg.get_output_dir("run_xyz") == "my_results/run_xyz"

    def test_is_visualization_enabled_all_true(self, cfg):
        assert cfg.is_visualization_enabled() is True

    def test_is_visualization_disabled_when_all_false(self, tmp_path):
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        for flag in ["show_boxes", "show_ids", "show_unique_count", "show_fps"]:
            base["visualization"][flag] = False
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        cfg = ConfigManager(str(p))
        assert cfg.is_visualization_enabled() is False

    def test_is_visualization_enabled_partial(self, tmp_path):
        """At least one flag True → enabled."""
        import yaml
        base = yaml.safe_load(textwrap.dedent(VALID_YAML))
        base["visualization"]["show_boxes"] = False
        base["visualization"]["show_ids"] = False
        base["visualization"]["show_unique_count"] = False
        base["visualization"]["show_fps"] = True
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(base), encoding="utf-8")
        cfg = ConfigManager(str(p))
        assert cfg.is_visualization_enabled() is True


# ===========================================================================
# to_dict round-trip
# ===========================================================================

class TestToDict:
    @pytest.fixture
    def cfg(self, tmp_path):
        return ConfigManager(_write_config(tmp_path, VALID_YAML))

    def test_to_dict_returns_dict(self, cfg):
        d = cfg.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_all_sections(self, cfg):
        d = cfg.to_dict()
        for section in ["model", "device", "tracker", "video",
                        "visualization", "export", "analytics", "logging"]:
            assert section in d, f"Missing section: {section}"

    def test_to_dict_values_match_attributes(self, cfg):
        d = cfg.to_dict()
        assert d["model"]["type"] == cfg.model.type
        assert d["model"]["confidence_threshold"] == cfg.model.confidence_threshold
        assert d["device"]["preferred"] == cfg.device.preferred
        assert d["analytics"]["ema_alpha"] == cfg.analytics.ema_alpha

    def test_to_dict_is_serialisable(self, cfg):
        """to_dict() result must be JSON-serialisable for export metadata."""
        import json
        d = cfg.to_dict()
        # Should not raise
        json.dumps(d)

    def test_to_dict_reflects_overrides(self, cfg):
        cfg.apply_overrides(model="yolov8l", confidence=0.6)
        d = cfg.to_dict()
        assert d["model"]["type"] == "yolov8l"
        assert d["model"]["confidence_threshold"] == 0.6
