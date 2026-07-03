"""
Configuration Manager for VisionTrack.

Loads config/config.yaml, validates every field against expected types and
value ranges, and exposes settings through nested attribute access so the
rest of the pipeline never touches raw dicts or YAML strings.

CLI arguments and environment variables are merged in at runtime, giving
a clean single source of truth that flows through every module.

Usage:
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")

    # Nested attribute access
    print(cfg.model.type)               # "yolov8n"
    print(cfg.tracker.config_file)      # "config/botsort.yaml"
    print(cfg.device.preferred)         # "cuda"

    # Helper methods
    path = cfg.get_model_path()         # "models/yolov8n.pt"
    out  = cfg.get_output_dir("run_x")  # "outputs/run_x"

    # CLI override (after argparse)
    cfg.apply_overrides(model="yolov8m", device="cpu", confidence=0.4)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from exceptions import ConfigurationError
from loggers import get_logger
from src.constants import (
    DEFAULT_CODEC,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRACKER_CONFIG,
    SUPPORTED_DEVICES,
    SUPPORTED_MODELS,
)

log = get_logger(__name__)


# ==============================================================================
# Namespace — turns a plain dict into attribute-accessible object
# ==============================================================================

class _Namespace:
    """Recursively converts a dictionary into dot-accessible attributes.

    Nested dicts become nested _Namespace instances, so you can write
    ``cfg.model.type`` instead of ``cfg["model"]["type"]``.

    Args:
        data: Dictionary to wrap.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _Namespace(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain dictionary."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, _Namespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        keys = ", ".join(self.__dict__.keys())
        return f"_Namespace({keys})"


# ==============================================================================
# ConfigManager
# ==============================================================================

class ConfigManager:
    """Loads, validates and exposes the VisionTrack configuration.

    Attribute sections (all are _Namespace instances):
        model        — YOLO model type, confidence/IoU thresholds, weights dir
        device       — preferred and fallback compute device
        tracker      — BoT-SORT config file path and persist flag
        video        — output codec, skip_frames, max_resolution
        visualization — overlay toggles, trail length, bbox thickness
        export       — output directory, save_video/json/csv flags
        analytics    — EMA alpha for FPS smoothing
        logging      — log level, format strings, log file path

    Args:
        config_path: Path to config.yaml (absolute or relative to CWD).

    Raises:
        ConfigurationError: If the file is missing, unparseable, or any
                            field fails validation.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = Path(config_path)
        self._raw: dict[str, Any] = self._load()
        self._validate(self._raw)
        self._apply_env_overrides(self._raw)
        self._populate(self._raw)
        log.info("Configuration loaded from '%s'.", self._config_path)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_model_path(self) -> str:
        """Return the expected local path for the current model weights.

        Example:
            cfg.get_model_path()  →  "models/yolov8n.pt"
        """
        return str(Path(self.model.weights_dir) / f"{self.model.type}.pt")

    def get_output_dir(self, run_name: str) -> str:
        """Return the full path for a named pipeline run directory.

        Args:
            run_name: Timestamped directory name, e.g. ``"run_20240115_143022"``.

        Returns:
            Path string, e.g. ``"outputs/run_20240115_143022"``.
        """
        return str(Path(self.export.output_dir) / run_name)

    def is_visualization_enabled(self) -> bool:
        """Return True if any visualisation overlay is active."""
        v = self.visualization
        return any([v.show_boxes, v.show_ids, v.show_unique_count, v.show_fps])

    def apply_overrides(
        self,
        *,
        model: Optional[str] = None,
        confidence: Optional[float] = None,
        device: Optional[str] = None,
        skip_frames: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Merge CLI argument values into the loaded configuration.

        Only non-None arguments are applied, preserving YAML defaults for
        everything the user did not explicitly specify on the command line.

        Args:
            model:       YOLO model variant (e.g. ``"yolov8m"``).
            confidence:  Detection confidence threshold [0.0, 1.0].
            device:      Compute device (``"cuda"``, ``"mps"``, ``"cpu"``).
            skip_frames: Frames to skip between processed frames (>= 0).
            output_dir:  Override the export output root directory.
            verbose:     When True, sets logging level to ``"DEBUG"``.
        """
        changed: list[str] = []

        if model is not None:
            if model not in SUPPORTED_MODELS:
                raise ConfigurationError(
                    f"--model '{model}' is not supported. "
                    f"Choose from: {SUPPORTED_MODELS}",
                    details={"provided": model, "supported": SUPPORTED_MODELS},
                )
            self.model.type = model
            changed.append(f"model.type={model}")

        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                raise ConfigurationError(
                    f"--confidence {confidence} is out of range [0.0, 1.0].",
                    details={"provided": confidence},
                )
            self.model.confidence_threshold = confidence
            changed.append(f"model.confidence_threshold={confidence}")

        if device is not None:
            if device not in SUPPORTED_DEVICES:
                raise ConfigurationError(
                    f"--device '{device}' is not supported. "
                    f"Choose from: {SUPPORTED_DEVICES}",
                    details={"provided": device, "supported": SUPPORTED_DEVICES},
                )
            self.device.preferred = device
            changed.append(f"device.preferred={device}")

        if skip_frames is not None:
            if skip_frames < 0:
                raise ConfigurationError(
                    f"--skip-frames {skip_frames} must be >= 0.",
                    details={"provided": skip_frames},
                )
            self.video.skip_frames = skip_frames
            changed.append(f"video.skip_frames={skip_frames}")

        if output_dir is not None:
            self.export.output_dir = output_dir
            changed.append(f"export.output_dir={output_dir}")

        if verbose:
            self.logging.level = "DEBUG"
            changed.append("logging.level=DEBUG")

        if changed:
            log.debug("CLI overrides applied: %s", ", ".join(changed))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full resolved configuration back to a plain dict.

        Useful for embedding a config snapshot in export metadata.
        """
        return {
            "model": self.model.to_dict(),
            "device": self.device.to_dict(),
            "tracker": self.tracker.to_dict(),
            "video": self.video.to_dict(),
            "visualization": self.visualization.to_dict(),
            "export": self.export.to_dict(),
            "analytics": self.analytics.to_dict(),
            "logging": self.logging.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal: load
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        """Read and parse the YAML file."""
        if not self._config_path.is_file():
            raise ConfigurationError(
                f"Config file not found: '{self._config_path}'",
                details={"path": str(self._config_path)},
            )
        try:
            with self._config_path.open(encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Failed to parse YAML '{self._config_path}': {exc}",
                details={"path": str(self._config_path)},
            ) from exc

        if not isinstance(data, dict):
            raise ConfigurationError(
                f"Config file '{self._config_path}' must contain a YAML mapping.",
                details={"type": type(data).__name__},
            )
        return data

    # ------------------------------------------------------------------
    # Internal: validate
    # ------------------------------------------------------------------

    def _validate(self, cfg: dict[str, Any]) -> None:
        """Validate every required section and field."""
        self._check_required_sections(cfg)
        self._validate_model(cfg["model"])
        self._validate_device(cfg["device"])
        self._validate_tracker(cfg["tracker"])
        self._validate_video(cfg["video"])
        self._validate_visualization(cfg["visualization"])
        self._validate_export(cfg["export"])
        self._validate_analytics(cfg["analytics"])
        self._validate_logging(cfg["logging"])
        log.debug("Configuration validation passed.")

    def _check_required_sections(self, cfg: dict[str, Any]) -> None:
        required = ["model", "device", "tracker", "video", "visualization",
                    "export", "analytics", "logging"]
        missing = [s for s in required if s not in cfg]
        if missing:
            raise ConfigurationError(
                f"Missing required config sections: {missing}",
                details={"missing": missing},
            )

    def _validate_model(self, m: dict[str, Any]) -> None:
        model_type = m.get("type", DEFAULT_MODEL)
        if model_type not in SUPPORTED_MODELS:
            raise ConfigurationError(
                f"model.type '{model_type}' is not supported. "
                f"Supported: {SUPPORTED_MODELS}",
                details={"provided": model_type},
            )

        conf = m.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ConfigurationError(
                f"model.confidence_threshold must be in [0.0, 1.0], got '{conf}'.",
                details={"provided": conf},
            )

        iou = m.get("iou_threshold", DEFAULT_IOU_THRESHOLD)
        if not isinstance(iou, (int, float)) or not (0.0 <= float(iou) <= 1.0):
            raise ConfigurationError(
                f"model.iou_threshold must be in [0.0, 1.0], got '{iou}'.",
                details={"provided": iou},
            )

        warmup = m.get("warmup_frames", 3)
        if not isinstance(warmup, int) or warmup < 0:
            raise ConfigurationError(
                f"model.warmup_frames must be a non-negative integer, got '{warmup}'.",
                details={"provided": warmup},
            )

    def _validate_device(self, d: dict[str, Any]) -> None:
        preferred = d.get("preferred", "cuda")
        if preferred not in SUPPORTED_DEVICES:
            raise ConfigurationError(
                f"device.preferred '{preferred}' is not supported. "
                f"Choose from: {SUPPORTED_DEVICES}",
                details={"provided": preferred},
            )
        fallback = d.get("fallback", "cpu")
        if fallback not in SUPPORTED_DEVICES:
            raise ConfigurationError(
                f"device.fallback '{fallback}' is not supported. "
                f"Choose from: {SUPPORTED_DEVICES}",
                details={"provided": fallback},
            )

    def _validate_tracker(self, t: dict[str, Any]) -> None:
        config_file = t.get("config_file", DEFAULT_TRACKER_CONFIG)
        if not isinstance(config_file, str) or not config_file.strip():
            raise ConfigurationError(
                "tracker.config_file must be a non-empty string.",
                details={"provided": config_file},
            )
        if not isinstance(t.get("persist", True), bool):
            raise ConfigurationError(
                "tracker.persist must be a boolean.",
                details={"provided": t.get("persist")},
            )

    def _validate_video(self, v: dict[str, Any]) -> None:
        codec = v.get("output_codec", DEFAULT_CODEC)
        if not isinstance(codec, str) or len(codec) == 0:
            raise ConfigurationError(
                "video.output_codec must be a non-empty string.",
                details={"provided": codec},
            )
        skip = v.get("skip_frames", 0)
        if not isinstance(skip, int) or skip < 0:
            raise ConfigurationError(
                f"video.skip_frames must be a non-negative integer, got '{skip}'.",
                details={"provided": skip},
            )
        max_res = v.get("max_resolution")
        if max_res is not None:
            if (
                not isinstance(max_res, (list, tuple))
                or len(max_res) != 2
                or not all(isinstance(x, int) and x > 0 for x in max_res)
            ):
                raise ConfigurationError(
                    "video.max_resolution must be null or a list of two positive "
                    f"integers [width, height], got '{max_res}'.",
                    details={"provided": max_res},
                )

    def _validate_visualization(self, v: dict[str, Any]) -> None:
        for flag in ["show_boxes", "show_ids", "show_unique_count", "show_fps"]:
            val = v.get(flag, True)
            if not isinstance(val, bool):
                raise ConfigurationError(
                    f"visualization.{flag} must be a boolean, got '{val}'.",
                    details={"field": flag, "provided": val},
                )
        trail = v.get("trail_length", 30)
        if not isinstance(trail, int) or trail < 0:
            raise ConfigurationError(
                f"visualization.trail_length must be a non-negative integer, got '{trail}'.",
                details={"provided": trail},
            )
        alpha = v.get("hud_alpha", 0.6)
        if not isinstance(alpha, (int, float)) or not (0.0 <= float(alpha) <= 1.0):
            raise ConfigurationError(
                f"visualization.hud_alpha must be in [0.0, 1.0], got '{alpha}'.",
                details={"provided": alpha},
            )

    def _validate_export(self, e: dict[str, Any]) -> None:
        output_dir = e.get("output_dir", DEFAULT_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ConfigurationError(
                "export.output_dir must be a non-empty string.",
                details={"provided": output_dir},
            )
        for flag in ["save_video", "save_json", "save_csv"]:
            val = e.get(flag, True)
            if not isinstance(val, bool):
                raise ConfigurationError(
                    f"export.{flag} must be a boolean, got '{val}'.",
                    details={"field": flag, "provided": val},
                )

    def _validate_analytics(self, a: dict[str, Any]) -> None:
        alpha = a.get("ema_alpha", 0.1)
        if not isinstance(alpha, (int, float)) or not (0.0 < float(alpha) <= 1.0):
            raise ConfigurationError(
                f"analytics.ema_alpha must be in (0.0, 1.0], got '{alpha}'.",
                details={"provided": alpha},
            )

    def _validate_logging(self, lg: dict[str, Any]) -> None:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = lg.get("level", "INFO")
        if level not in valid_levels:
            raise ConfigurationError(
                f"logging.level '{level}' is not valid. "
                f"Choose from: {sorted(valid_levels)}",
                details={"provided": level},
            )

    # ------------------------------------------------------------------
    # Internal: environment variable overrides
    # ------------------------------------------------------------------

    def _apply_env_overrides(self, cfg: dict[str, Any]) -> None:
        """Apply environment variable overrides before populating namespaces.

        Supported environment variables:
            VISIONTRACK_DEVICE      — overrides device.preferred
            VISIONTRACK_MODEL       — overrides model.type
            VISIONTRACK_CONFIDENCE  — overrides model.confidence_threshold
            VISIONTRACK_LOG_LEVEL   — overrides logging.level
        """
        env_device = os.getenv("VISIONTRACK_DEVICE")
        if env_device:
            env_device = env_device.lower().strip()
            if env_device in SUPPORTED_DEVICES:
                cfg["device"]["preferred"] = env_device
                log.debug("Env override: device.preferred='%s'.", env_device)
            else:
                log.warning(
                    "VISIONTRACK_DEVICE='%s' is not a supported device. "
                    "Ignoring. Supported: %s",
                    env_device,
                    SUPPORTED_DEVICES,
                )

        env_model = os.getenv("VISIONTRACK_MODEL")
        if env_model:
            if env_model in SUPPORTED_MODELS:
                cfg["model"]["type"] = env_model
                log.debug("Env override: model.type='%s'.", env_model)
            else:
                log.warning(
                    "VISIONTRACK_MODEL='%s' is not a supported model. Ignoring.",
                    env_model,
                )

        env_conf = os.getenv("VISIONTRACK_CONFIDENCE")
        if env_conf:
            try:
                conf_val = float(env_conf)
                if 0.0 <= conf_val <= 1.0:
                    cfg["model"]["confidence_threshold"] = conf_val
                    log.debug(
                        "Env override: model.confidence_threshold=%.2f.", conf_val
                    )
                else:
                    log.warning(
                        "VISIONTRACK_CONFIDENCE='%s' is out of [0, 1]. Ignoring.",
                        env_conf,
                    )
            except ValueError:
                log.warning(
                    "VISIONTRACK_CONFIDENCE='%s' is not a valid float. Ignoring.",
                    env_conf,
                )

        env_log = os.getenv("VISIONTRACK_LOG_LEVEL")
        if env_log:
            env_log = env_log.upper().strip()
            valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
            if env_log in valid:
                cfg["logging"]["level"] = env_log
                log.debug("Env override: logging.level='%s'.", env_log)
            else:
                log.warning(
                    "VISIONTRACK_LOG_LEVEL='%s' is not valid. Ignoring. "
                    "Valid: %s",
                    env_log,
                    sorted(valid),
                )

    # ------------------------------------------------------------------
    # Internal: populate namespaces
    # ------------------------------------------------------------------

    def _populate(self, cfg: dict[str, Any]) -> None:
        """Convert raw dict sections into _Namespace attributes."""
        self.model         = _Namespace(cfg["model"])
        self.device        = _Namespace(cfg["device"])
        self.tracker       = _Namespace(cfg["tracker"])
        self.video         = _Namespace(cfg["video"])
        self.visualization = _Namespace(cfg["visualization"])
        self.export        = _Namespace(cfg["export"])
        self.analytics     = _Namespace(cfg["analytics"])
        self.logging       = _Namespace(cfg["logging"])

    def __repr__(self) -> str:
        return (
            f"ConfigManager(path='{self._config_path}', "
            f"model='{self.model.type}', "
            f"device='{self.device.preferred}')"
        )
