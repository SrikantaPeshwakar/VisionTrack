"""
YOLO Detector Module for VisionTrack.

Wraps the Ultralytics YOLO model and is responsible for exactly two things:
  1. Running inference on a single BGR frame.
  2. Returning only person detections above the configured confidence threshold.

Everything else — tracking, analytics, visualisation — is handled downstream.

Usage:
    from src.detector import Detector
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    detector = Detector(cfg, device="mps")

    detections = detector.detect(frame)   # List[Detection]
    print(f"Found {len(detections)} persons")
    print(f"Last inference: {detector.last_inference_ms:.1f} ms")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from exceptions import DetectionError, ModelLoadError
from loggers import get_logger
from src.constants import PERSON_CLASS_ID
from src.data_models import BoundingBox, Detection

if TYPE_CHECKING:
    from src.config_manager import ConfigManager

log = get_logger(__name__)


class Detector:
    """YOLO-based person detector.

    Loads a YOLO model (auto-downloading weights on first run if not cached),
    runs inference on individual frames, and returns only person detections
    above the configured confidence threshold.

    Attributes:
        model_type:          YOLO variant name (e.g. ``"yolov8n"``).
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold:        NMS IoU threshold.
        device:               Resolved compute device string.
        last_inference_ms:    Wall-clock time of the most recent detect() call.

    Args:
        config:  Loaded ConfigManager instance.
        device:  Resolved device string from DeviceManager
                 (e.g. ``"mps"``, ``"cuda:0"``, ``"cpu"``).
                 Overrides config.device.preferred when provided.
    """

    def __init__(self, config: ConfigManager, device: str | None = None) -> None:
        self.model_type: str = config.model.type
        self.confidence_threshold: float = config.model.confidence_threshold
        self.iou_threshold: float = config.model.iou_threshold
        self.device: str = device or config.device.preferred
        self._weights_dir: Path = Path(config.model.weights_dir)
        self._warmup_frames: int = config.model.warmup_frames

        self.last_inference_ms: float = 0.0
        self._model = None  # loaded lazily in _load_model()

        self._load_model()
        self._warmup()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single frame and return person detections.

        Filters results to class 0 (person) and applies the configured
        confidence threshold before returning.

        Args:
            frame: BGR image as a numpy array with shape (H, W, 3).
                   This is the native format produced by OpenCV's
                   ``VideoCapture.read()``.

        Returns:
            List of :class:`Detection` instances, one per detected person.
            Empty list when no persons are found or the frame is invalid.

        Raises:
            DetectionError: If the model raises an unexpected exception
                            during inference (e.g. CUDA OOM).
        """
        if not self._is_valid_frame(frame):
            log.warning("detect() received an invalid frame — skipping.")
            return []

        t0 = time.perf_counter()
        try:
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=[PERSON_CLASS_ID],  # only request person class
                verbose=False,  # suppress Ultralytics console output
                device=self.device,
            )
        except Exception as exc:
            raise DetectionError(
                frame_id=-1,
                reason=str(exc),
                details={"model": self.model_type, "device": self.device},
            ) from exc
        finally:
            self.last_inference_ms = (time.perf_counter() - t0) * 1000

        detections = self._parse_results(results)
        log.debug(
            "detect() → %d persons | %.1f ms",
            len(detections),
            self.last_inference_ms,
        )
        return detections

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load YOLO weights, downloading them on first run if not cached."""
        from ultralytics import YOLO  # deferred import — keeps startup fast

        # Ensure the weights directory exists before any download attempt
        self._weights_dir.mkdir(parents=True, exist_ok=True)

        weights_path = self._weights_dir / f"{self.model_type}.pt"

        if weights_path.is_file():
            log.info("Loading model weights from '%s'.", weights_path)
            source = str(weights_path)
        else:
            log.info(
                "Weights not found at '%s'. "
                "Ultralytics will download '%s' to models/ automatically.",
                weights_path,
                self.model_type,
            )
            # Pass the full target path so Ultralytics downloads directly
            # into models/ rather than the CWD (which is the project root).
            source = str(weights_path)

        try:
            self._model = YOLO(source)
            log.info(
                "Model '%s' loaded successfully on device '%s'.",
                self.model_type,
                self.device,
            )
        except Exception as exc:
            raise ModelLoadError(
                model_name=self.model_type,
                reason=str(exc),
                details={"weights_path": str(weights_path), "device": self.device},
            ) from exc

    # ------------------------------------------------------------------
    # Internal: warmup
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        """Run inference on dummy frames to initialise the GPU/CPU pipeline.

        This ensures that the first real detect() call measures true
        steady-state latency rather than JIT compilation or memory
        allocation overhead.
        """
        if self._warmup_frames <= 0:
            return

        log.debug(
            "Running %d warmup frame(s) on device '%s' …",
            self._warmup_frames,
            self.device,
        )
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(self._warmup_frames):
            try:
                self._model(
                    dummy,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    classes=[PERSON_CLASS_ID],
                    verbose=False,
                    device=self.device,
                )
            except Exception as exc:
                # Warmup failures are non-fatal — log and continue
                log.warning("Warmup frame failed (non-fatal): %s", exc)
                break

        log.debug("Warmup complete.")

    # ------------------------------------------------------------------
    # Internal: result parsing
    # ------------------------------------------------------------------

    def _parse_results(self, results: list) -> list[Detection]:
        """Convert raw Ultralytics results into a list of Detection objects.

        Ultralytics returns a list of Results objects, one per image.
        For single-frame inference this list always has exactly one element.

        Args:
            results: Raw output from ``model(frame, ...)``.

        Returns:
            List of Detection instances for detected persons.
        """
        detections: list[Detection] = []

        if not results:
            return detections

        result = results[0]  # single frame → single Results object

        # result.boxes is None when the model found nothing
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        # boxes.data has shape (N, 6): x1 y1 x2 y2 conf class_id
        boxes_data = result.boxes.data.cpu().numpy()

        for row in boxes_data:
            x1, y1, x2, y2, conf, class_id = (
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                # Clamp to [0, 1] — fused scores can slightly exceed 1.0
                min(1.0, max(0.0, float(row[4]))),
                int(row[5]),
            )

            # Guard: only person class (should already be filtered by
            # classes=[PERSON_CLASS_ID] in the model call, but be explicit)
            if class_id != PERSON_CLASS_ID:
                continue

            # Guard: confidence threshold (Ultralytics may still return
            # borderline boxes depending on NMS implementation)
            if conf < self.confidence_threshold:
                continue

            # Guard: valid bounding box (non-zero area)
            if x2 <= x1 or y2 <= y1:
                log.debug(
                    "Skipping degenerate bbox [%.1f, %.1f, %.1f, %.1f]",
                    x1,
                    y1,
                    x2,
                    y2,
                )
                continue

            detections.append(
                Detection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Internal: frame validation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_frame(frame: np.ndarray) -> bool:
        """Return True if frame is a non-empty 3-channel numpy array."""
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.size > 0
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Detector("
            f"model='{self.model_type}', "
            f"conf={self.confidence_threshold}, "
            f"iou={self.iou_threshold}, "
            f"device='{self.device}')"
        )
