"""
Unit tests for src/detector.py

All tests mock the Ultralytics YOLO model so no GPU, no internet connection,
and no model weights are required to run the suite.

Covers:
- Detector construction and __repr__
- _is_valid_frame: all valid and invalid frame shapes
- _parse_results: persons only, confidence filtering, degenerate boxes,
                  multi-detection, empty results
- detect(): happy path, empty frame guard, invalid frame guard,
            DetectionError on model exception
- Warmup: called the right number of times, non-fatal on failure
- Model loading: from local cache path, bare model name fallback,
                 ModelLoadError on YOLO constructor failure
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_models import BoundingBox, Detection
from src.constants import PERSON_CLASS_ID
from exceptions import DetectionError, ModelLoadError


# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

def _make_config(
    model_type: str = "yolov8n",
    conf: float = 0.25,
    iou: float = 0.45,
    warmup: int = 0,          # 0 by default so tests run fast
    device: str = "cpu",
    weights_dir: str = "models",
) -> MagicMock:
    """Build a minimal ConfigManager mock."""
    cfg = MagicMock()
    cfg.model.type = model_type
    cfg.model.confidence_threshold = conf
    cfg.model.iou_threshold = iou
    cfg.model.warmup_frames = warmup
    cfg.model.weights_dir = weights_dir
    cfg.device.preferred = device
    return cfg


def _make_boxes_data(rows: list[list[float]]) -> MagicMock:
    """Build a mock result.boxes.data from a list of [x1,y1,x2,y2,conf,cls] rows."""
    tensor_mock = MagicMock()
    tensor_mock.cpu.return_value.numpy.return_value = np.array(rows, dtype=np.float32)
    return tensor_mock


def _make_results(rows: list[list[float]] | None) -> list[MagicMock]:
    """Build a mock Ultralytics results list."""
    result = MagicMock()
    if rows is None:
        result.boxes = None
    elif len(rows) == 0:
        result.boxes = MagicMock()
        result.boxes.__len__ = lambda self: 0
        result.boxes.data = _make_boxes_data([])
    else:
        result.boxes = MagicMock()
        result.boxes.__len__ = lambda self: len(rows)
        result.boxes.data = _make_boxes_data(rows)
    return [result]


def _make_detector(cfg=None, device="cpu", mock_yolo=None):
    """Construct a Detector with a patched YOLO class."""
    from src.detector import Detector

    if cfg is None:
        cfg = _make_config(device=device)

    yolo_instance = mock_yolo or MagicMock()
    yolo_instance.return_value = _make_results([])  # warmup default

    with patch("src.detector.YOLO", return_value=yolo_instance, create=True):
        # Patch the import inside _load_model
        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=MagicMock(return_value=yolo_instance))}):
            detector = Detector(cfg, device=device)
            detector._model = yolo_instance
    return detector


# ===========================================================================
# _is_valid_frame
# ===========================================================================

class TestIsValidFrame:
    """Static method — test directly without constructing a full Detector."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.detector import Detector
        self.fn = Detector._is_valid_frame

    def test_valid_bgr_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert self.fn(frame) is True

    def test_valid_small_frame(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        assert self.fn(frame) is True

    def test_none_is_invalid(self):
        assert self.fn(None) is False

    def test_grayscale_is_invalid(self):
        frame = np.zeros((480, 640), dtype=np.uint8)
        assert self.fn(frame) is False

    def test_4channel_is_invalid(self):
        frame = np.zeros((480, 640, 4), dtype=np.uint8)
        assert self.fn(frame) is False

    def test_empty_array_is_invalid(self):
        assert self.fn(np.array([])) is False

    def test_zero_size_frame_is_invalid(self):
        frame = np.zeros((0, 640, 3), dtype=np.uint8)
        assert self.fn(frame) is False

    def test_list_is_invalid(self):
        assert self.fn([[1, 2, 3]]) is False

    def test_float32_frame_is_valid(self):
        frame = np.zeros((480, 640, 3), dtype=np.float32)
        assert self.fn(frame) is True


# ===========================================================================
# _parse_results
# ===========================================================================

class TestParseResults:
    @pytest.fixture(autouse=True)
    def _detector(self):
        self.detector = _make_detector()

    def test_empty_results_list(self):
        assert self.detector._parse_results([]) == []

    def test_boxes_is_none(self):
        assert self.detector._parse_results(_make_results(None)) == []

    def test_no_boxes(self):
        assert self.detector._parse_results(_make_results([])) == []

    def test_single_person_detected(self):
        rows = [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert len(dets) == 1
        d = dets[0]
        assert isinstance(d, Detection)
        assert d.bbox.x1 == pytest.approx(10.0)
        assert d.bbox.y1 == pytest.approx(20.0)
        assert d.bbox.x2 == pytest.approx(110.0)
        assert d.bbox.y2 == pytest.approx(120.0)
        assert d.confidence == pytest.approx(0.9)
        assert d.class_id == PERSON_CLASS_ID

    def test_multiple_persons(self):
        rows = [
            [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
            [200.0, 50.0, 300.0, 200.0, 0.75, PERSON_CLASS_ID],
            [400.0, 100.0, 500.0, 300.0, 0.6, PERSON_CLASS_ID],
        ]
        dets = self.detector._parse_results(_make_results(rows))
        assert len(dets) == 3

    def test_non_person_class_filtered(self):
        """Car (class 2) and dog (class 16) must be excluded."""
        rows = [
            [10.0, 20.0, 110.0, 120.0, 0.9, 2],    # car
            [200.0, 50.0, 300.0, 200.0, 0.8, 16],   # dog
        ]
        dets = self.detector._parse_results(_make_results(rows))
        assert dets == []

    def test_mixed_classes_only_persons_returned(self):
        rows = [
            [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
            [200.0, 50.0, 300.0, 200.0, 0.85, 2],  # car — excluded
        ]
        dets = self.detector._parse_results(_make_results(rows))
        assert len(dets) == 1
        assert dets[0].class_id == PERSON_CLASS_ID

    def test_below_confidence_threshold_filtered(self):
        """Detections below threshold must be excluded even if class is correct."""
        self.detector.confidence_threshold = 0.5
        rows = [[10.0, 20.0, 110.0, 120.0, 0.3, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert dets == []

    def test_exactly_at_confidence_threshold_kept(self):
        self.detector.confidence_threshold = 0.5
        rows = [[10.0, 20.0, 110.0, 120.0, 0.5, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert len(dets) == 1

    def test_degenerate_box_zero_width_filtered(self):
        """x2 == x1 means zero width — must be excluded."""
        rows = [[100.0, 50.0, 100.0, 200.0, 0.9, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert dets == []

    def test_degenerate_box_zero_height_filtered(self):
        rows = [[100.0, 50.0, 200.0, 50.0, 0.9, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert dets == []

    def test_degenerate_box_inverted_coords_filtered(self):
        """x2 < x1 is an inverted box — must be excluded."""
        rows = [[200.0, 50.0, 100.0, 200.0, 0.9, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert dets == []

    def test_detection_bbox_dimensions(self):
        rows = [[50.0, 100.0, 150.0, 300.0, 0.8, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        assert len(dets) == 1
        assert dets[0].bbox.width == pytest.approx(100.0)
        assert dets[0].bbox.height == pytest.approx(200.0)

    def test_output_types(self):
        rows = [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]]
        dets = self.detector._parse_results(_make_results(rows))
        d = dets[0]
        assert isinstance(d.bbox, BoundingBox)
        assert isinstance(d.confidence, float)
        assert isinstance(d.class_id, int)


# ===========================================================================
# detect()
# ===========================================================================

class TestDetect:
    @pytest.fixture(autouse=True)
    def _detector(self):
        self.detector = _make_detector()

    def _set_model_return(self, rows):
        self.detector._model.return_value = _make_results(rows)
        self.detector._model.side_effect = None

    def test_returns_list(self):
        self._set_model_return([])
        result = self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(result, list)

    def test_detects_single_person(self):
        self._set_model_return([[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]])
        dets = self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert dets[0].class_id == PERSON_CLASS_ID

    def test_detects_multiple_persons(self):
        rows = [
            [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
            [200.0, 50.0, 300.0, 200.0, 0.7, PERSON_CLASS_ID],
        ]
        self._set_model_return(rows)
        dets = self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 2

    def test_empty_frame_returns_empty_list(self):
        """A None frame must return [] without calling the model."""
        result = self.detector.detect(None)
        assert result == []
        self.detector._model.assert_not_called()

    def test_invalid_frame_returns_empty_list(self):
        """A grayscale (2-channel) frame must return [] without calling the model."""
        bad_frame = np.zeros((480, 640), dtype=np.uint8)
        result = self.detector.detect(bad_frame)
        assert result == []
        self.detector._model.assert_not_called()

    def test_model_called_with_correct_args(self):
        self._set_model_return([])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.detect(frame)
        call_kwargs = self.detector._model.call_args
        assert call_kwargs is not None
        # Verify key kwargs passed to model
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert kwargs.get("conf") == self.detector.confidence_threshold
        assert kwargs.get("iou") == self.detector.iou_threshold
        assert kwargs.get("classes") == [PERSON_CLASS_ID]
        assert kwargs.get("verbose") is False

    def test_model_exception_raises_detection_error(self):
        self.detector._model.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(DetectionError):
            self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_last_inference_ms_updated_after_detect(self):
        self._set_model_return([])
        self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert self.detector.last_inference_ms >= 0.0

    def test_last_inference_ms_updated_on_model_exception(self):
        """last_inference_ms must be set even when the model raises."""
        self.detector._model.side_effect = RuntimeError("boom")
        with pytest.raises(DetectionError):
            self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert self.detector.last_inference_ms >= 0.0

    def test_no_detections_returns_empty_list(self):
        self._set_model_return(None)  # boxes=None case
        result = self.detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert result == []


# ===========================================================================
# Detector construction & repr
# ===========================================================================

class TestDetectorConstruction:
    def test_attributes_set_from_config(self):
        det = _make_detector(_make_config(
            model_type="yolov8m",
            conf=0.4,
            iou=0.5,
            device="cpu",
        ))
        assert det.model_type == "yolov8m"
        assert det.confidence_threshold == pytest.approx(0.4)
        assert det.iou_threshold == pytest.approx(0.5)
        assert det.device == "cpu"

    def test_device_arg_overrides_config(self):
        cfg = _make_config(device="cuda")
        det = _make_detector(cfg, device="cpu")
        assert det.device == "cpu"

    def test_last_inference_ms_initialised_to_zero(self):
        det = _make_detector()
        assert det.last_inference_ms == 0.0

    def test_repr_contains_key_info(self):
        det = _make_detector(_make_config(model_type="yolov8n", conf=0.25))
        r = repr(det)
        assert "Detector" in r
        assert "yolov8n" in r
        assert "0.25" in r
        assert "cpu" in r


# ===========================================================================
# Warmup
# ===========================================================================

class TestWarmup:
    def test_warmup_called_n_times(self):
        """When warmup_frames=3, model should be called 3 times during init."""
        cfg = _make_config(warmup=3)
        yolo_instance = MagicMock()
        yolo_instance.return_value = _make_results([])
        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(YOLO=MagicMock(return_value=yolo_instance))
        }):
            from src.detector import Detector
            det = Detector(cfg, device="cpu")
            det._model = yolo_instance  # not counted, already set

        # model was called warmup_frames times during _warmup
        assert yolo_instance.call_count == 3

    def test_warmup_zero_skips_calls(self):
        """When warmup_frames=0, model should NOT be called during init."""
        cfg = _make_config(warmup=0)
        yolo_instance = MagicMock()
        yolo_instance.return_value = _make_results([])
        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(YOLO=MagicMock(return_value=yolo_instance))
        }):
            from src.detector import Detector
            det = Detector(cfg, device="cpu")

        assert yolo_instance.call_count == 0

    def test_warmup_failure_is_nonfatal(self):
        """A RuntimeError during warmup must not crash the Detector."""
        cfg = _make_config(warmup=2)
        yolo_instance = MagicMock()
        yolo_instance.side_effect = RuntimeError("warmup fail")
        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(YOLO=MagicMock(return_value=yolo_instance))
        }):
            from src.detector import Detector
            # Should not raise
            det = Detector(cfg, device="cpu")
        assert det is not None


# ===========================================================================
# Model loading
# ===========================================================================

class TestModelLoading:
    def test_model_load_error_raised_on_yolo_failure(self, tmp_path):
        """If YOLO() raises, Detector must re-raise as ModelLoadError."""
        cfg = _make_config(weights_dir=str(tmp_path))

        def _bad_yolo(*args, **kwargs):
            raise RuntimeError("weights corrupted")

        with patch.dict("sys.modules", {
            "ultralytics": MagicMock(YOLO=_bad_yolo)
        }):
            from src.detector import Detector
            with pytest.raises(ModelLoadError, match="yolov8n"):
                Detector(cfg, device="cpu")

    def test_loads_from_local_weights_when_file_exists(self, tmp_path):
        """When models/yolov8n.pt exists, it must be passed to YOLO()."""
        weights_file = tmp_path / "yolov8n.pt"
        weights_file.touch()  # create empty placeholder

        cfg = _make_config(weights_dir=str(tmp_path), warmup=0)
        yolo_instance = MagicMock()
        yolo_instance.return_value = _make_results([])
        yolo_cls = MagicMock(return_value=yolo_instance)

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=yolo_cls)}):
            from src.detector import Detector
            Detector(cfg, device="cpu")

        # YOLO constructor must have been called with the local file path
        call_arg = yolo_cls.call_args[0][0]
        assert str(tmp_path) in call_arg
        assert "yolov8n.pt" in call_arg

    def test_loads_from_model_name_when_no_local_file(self, tmp_path):
        """When weights file is absent, bare model name is passed to YOLO()."""
        cfg = _make_config(weights_dir=str(tmp_path), warmup=0)
        yolo_instance = MagicMock()
        yolo_instance.return_value = _make_results([])
        yolo_cls = MagicMock(return_value=yolo_instance)

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=yolo_cls)}):
            from src.detector import Detector
            Detector(cfg, device="cpu")

        call_arg = yolo_cls.call_args[0][0]
        assert call_arg == "yolov8n"
