"""
Unit tests for src/pipeline.py

Every external dependency (OpenCV, Detector, Tracker, Analytics,
Visualizer, Exporter) is mocked so the suite runs without any video
files, GPU, or model weights.

Covers:
- Construction and __repr__
- _open_video: file not found, OpenCV fails to open, success
- _build_summary: zero frames, multiple frames, peak concurrent, avg dwell
- _handle_interrupt: sets _interrupted flag
- run(): full happy-path wiring, skip_frames logic, empty detections,
         interrupted mid-loop, VideoIOError propagation,
         component call order verification
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_models import (
    BoundingBox, Detection, FrameResult, PipelineSummary, Track, TrackSummary,
)
from src.constants import PERSON_CLASS_ID
from exceptions import VideoIOError


# ===========================================================================
# Helpers
# ===========================================================================

def _make_config(skip_frames: int = 0, model_type: str = "yolov8n") -> MagicMock:
    cfg = MagicMock()
    cfg.model.type = model_type
    cfg.video.skip_frames = skip_frames
    cfg.get_output_dir.return_value = "/tmp/vt_test/run_xyz"
    cfg.to_dict.return_value = {"model": {"type": model_type}}
    return cfg


def _make_track(track_id: int = 1, frame_id: int = 0, ts: float = 0.0) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(10, 20, 110, 120),
        confidence=0.9,
        class_id=PERSON_CLASS_ID,
        frame_id=frame_id,
        timestamp=ts,
    )


def _make_frame_result(
    frame_id: int = 0,
    tracks: list | None = None,
    inference_ms: float = 30.0,
    fps: float = 25.0,
) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        tracks=tracks or [],
        detection_count=len(tracks or []),
        inference_time_ms=inference_ms,
        fps=fps,
    )


def _make_pipeline(
    skip_frames: int = 0,
    model_type: str = "yolov8n",
    n_frames: int = 3,
    tracks_per_frame: list[list[Track]] | None = None,
):
    """Build a VideoPipeline with all dependencies fully mocked.

    Returns (pipeline, mocks_dict) so tests can assert on individual mocks.
    """
    from src.pipeline import VideoPipeline

    cfg       = _make_config(skip_frames=skip_frames, model_type=model_type)
    detector  = MagicMock()
    tracker   = MagicMock()
    analytics = MagicMock()
    visualizer = MagicMock()
    exporter  = MagicMock()

    # detector always returns one detection per frame
    detector.detect.return_value = [
        Detection.from_raw(10, 20, 110, 120, 0.9, PERSON_CLASS_ID)
    ]

    # tracker returns configurable tracks per frame
    if tracks_per_frame is None:
        tracks_per_frame = [[_make_track(1, i, i / 30.0)] for i in range(n_frames)]
    tracker.track.side_effect = (
        lambda frame, dets, frame_id, timestamp:
        tracks_per_frame[frame_id] if frame_id < len(tracks_per_frame) else []
    )
    tracker.all_track_ids = {1}
    tracker.get_all_summaries.return_value = [
        TrackSummary(1, 0, n_frames - 1, 0.0, (n_frames - 1) / 30.0, n_frames)
    ]

    # analytics returns a FrameResult per call
    def _analytics_update(frame_id, timestamp, tracks, detection_count, inference_time_ms):
        return _make_frame_result(frame_id, tracks, inference_time_ms)
    analytics.update.side_effect = _analytics_update
    analytics.get_summary.return_value = {"total_frames": n_frames}

    # visualizer returns the frame unchanged (annotated frame)
    visualizer.annotate.side_effect = lambda frame, *a, **kw: frame

    # exporter is a no-op for frame writes; finalise returns updated summary
    def _finalise(summary, *args, **kwargs):
        summary.output_video_path = "/tmp/vt_test/run_xyz/result.mp4"
        summary.output_json_path  = "/tmp/vt_test/run_xyz/analytics.json"
        summary.output_csv_path   = "/tmp/vt_test/run_xyz/tracks.csv"
        return summary
    exporter.finalise.side_effect = _finalise

    pipeline = VideoPipeline(cfg, detector, tracker, analytics, visualizer, exporter)

    return pipeline, {
        "cfg": cfg,
        "detector": detector,
        "tracker": tracker,
        "analytics": analytics,
        "visualizer": visualizer,
        "exporter": exporter,
    }


def _make_cap(n_frames: int = 3, fps: float = 30.0, w: int = 640, h: int = 480):
    """Build a mock cv2.VideoCapture that yields n_frames of zeros."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        0:  fps,    # CAP_PROP_FPS
        7:  n_frames,  # CAP_PROP_FRAME_COUNT
        3:  w,      # CAP_PROP_FRAME_WIDTH
        4:  h,      # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # read() returns (True, frame) for the first n_frames calls, then (False, None)
    read_returns = [(True, frame.copy()) for _ in range(n_frames)] + [(False, None)]
    cap.read.side_effect = read_returns
    return cap


# ===========================================================================
# Construction & repr
# ===========================================================================

class TestPipelineConstruction:
    def test_attributes_stored(self):
        pipeline, mocks = _make_pipeline()
        assert pipeline.config    is mocks["cfg"]
        assert pipeline.detector  is mocks["detector"]
        assert pipeline.tracker   is mocks["tracker"]
        assert pipeline.analytics is mocks["analytics"]
        assert pipeline.visualizer is mocks["visualizer"]
        assert pipeline.exporter  is mocks["exporter"]

    def test_skip_frames_from_config(self):
        pipeline, _ = _make_pipeline(skip_frames=2)
        assert pipeline._skip_frames == 2

    def test_interrupted_flag_initialised_false(self):
        pipeline, _ = _make_pipeline()
        assert pipeline._interrupted is False

    def test_repr_contains_model_and_skip(self):
        pipeline, _ = _make_pipeline(skip_frames=1, model_type="yolov8m")
        r = repr(pipeline)
        assert "VideoPipeline" in r
        assert "yolov8m" in r
        assert "skip_frames=1" in r


# ===========================================================================
# _open_video
# ===========================================================================

class TestOpenVideo:
    def test_missing_file_raises_video_io_error(self, tmp_path):
        pipeline, _ = _make_pipeline()
        with pytest.raises(VideoIOError, match="not found"):
            pipeline._open_video(str(tmp_path / "nonexistent.mp4"))

    def test_opencv_fails_to_open_raises_video_io_error(self, tmp_path):
        # Create a real (but empty) file so path check passes
        p = tmp_path / "bad.mp4"
        p.write_bytes(b"not a video")
        pipeline, _ = _make_pipeline()
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap_mock):
            with pytest.raises(VideoIOError, match="OpenCV"):
                pipeline._open_video(str(p))

    def test_valid_file_returns_cap(self, tmp_path):
        p = tmp_path / "video.mp4"
        p.write_bytes(b"placeholder")
        pipeline, _ = _make_pipeline()
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap_mock):
            result = pipeline._open_video(str(p))
        assert result is cap_mock


# ===========================================================================
# _handle_interrupt
# ===========================================================================

class TestHandleInterrupt:
    def test_sets_interrupted_flag(self):
        pipeline, _ = _make_pipeline()
        assert pipeline._interrupted is False
        pipeline._handle_interrupt(2, None)
        assert pipeline._interrupted is True


# ===========================================================================
# _build_summary
# ===========================================================================

class TestBuildSummary:
    def test_zero_frames_returns_zero_fps(self):
        pipeline, _ = _make_pipeline()
        s = pipeline._build_summary(10.0, "/tmp", [])
        assert s.avg_fps == 0.0
        assert s.total_frames == 0

    def test_fps_calculation(self):
        pipeline, _ = _make_pipeline()
        results = [_make_frame_result(i) for i in range(30)]
        s = pipeline._build_summary(1.0, "/tmp", results)
        assert s.avg_fps == pytest.approx(30.0)

    def test_total_frames(self):
        pipeline, _ = _make_pipeline()
        results = [_make_frame_result(i) for i in range(5)]
        s = pipeline._build_summary(1.0, "/tmp", results)
        assert s.total_frames == 5

    def test_avg_inference_time(self):
        pipeline, _ = _make_pipeline()
        results = [_make_frame_result(i, inference_ms=40.0) for i in range(4)]
        s = pipeline._build_summary(1.0, "/tmp", results)
        assert s.avg_inference_time_ms == pytest.approx(40.0)

    def test_peak_concurrent_tracks(self):
        pipeline, _ = _make_pipeline()
        r0 = _make_frame_result(0, tracks=[_make_track(1)])
        r1 = _make_frame_result(1, tracks=[_make_track(1), _make_track(2)])
        r2 = _make_frame_result(2, tracks=[_make_track(1)])
        s = pipeline._build_summary(1.0, "/tmp", [r0, r1, r2])
        assert s.peak_concurrent_tracks == 2

    def test_unique_visitors_from_tracker(self):
        pipeline, mocks = _make_pipeline()
        mocks["tracker"].all_track_ids = {1, 2, 3}
        s = pipeline._build_summary(1.0, "/tmp", [_make_frame_result(0)])
        assert s.unique_visitors == 3

    def test_avg_dwell_time(self):
        pipeline, mocks = _make_pipeline()
        mocks["tracker"].get_all_summaries.return_value = [
            TrackSummary(1, 0, 30, 0.0, 1.0, 30),   # dwell = 1.0s
            TrackSummary(2, 0, 60, 0.0, 2.0, 60),   # dwell = 2.0s
        ]
        s = pipeline._build_summary(1.0, "/tmp", [_make_frame_result(0)])
        assert s.avg_dwell_time == pytest.approx(1.5)

    def test_zero_dwell_when_no_summaries(self):
        pipeline, mocks = _make_pipeline()
        mocks["tracker"].get_all_summaries.return_value = []
        s = pipeline._build_summary(1.0, "/tmp", [_make_frame_result(0)])
        assert s.avg_dwell_time == pytest.approx(0.0)

    def test_returns_pipeline_summary_instance(self):
        pipeline, _ = _make_pipeline()
        s = pipeline._build_summary(1.0, "/tmp", [])
        assert isinstance(s, PipelineSummary)


# ===========================================================================
# run() — full happy path
# ===========================================================================

class TestRunHappyPath:
    def _run(self, n_frames: int = 3, skip_frames: int = 0):
        pipeline, mocks = _make_pipeline(n_frames=n_frames, skip_frames=skip_frames)
        cap = _make_cap(n_frames=n_frames)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    summary = pipeline.run("fake_video.mp4")
        return summary, mocks, cap

    def test_returns_pipeline_summary(self):
        summary, _, _ = self._run()
        assert isinstance(summary, PipelineSummary)

    def test_output_paths_set_by_exporter(self):
        summary, _, _ = self._run()
        assert summary.output_video_path is not None
        assert summary.output_json_path is not None
        assert summary.output_csv_path is not None

    def test_detector_called_once_per_processed_frame(self):
        _, mocks, _ = self._run(n_frames=3)
        assert mocks["detector"].detect.call_count == 3

    def test_tracker_called_once_per_processed_frame(self):
        _, mocks, _ = self._run(n_frames=3)
        assert mocks["tracker"].track.call_count == 3

    def test_analytics_update_called_once_per_processed_frame(self):
        _, mocks, _ = self._run(n_frames=3)
        assert mocks["analytics"].update.call_count == 3

    def test_visualizer_annotate_called_once_per_processed_frame(self):
        _, mocks, _ = self._run(n_frames=3)
        assert mocks["visualizer"].annotate.call_count == 3

    def test_exporter_write_frame_called_once_per_processed_frame(self):
        _, mocks, _ = self._run(n_frames=3)
        assert mocks["exporter"].write_frame.call_count == 3

    def test_exporter_prepare_called_once(self):
        _, mocks, _ = self._run()
        mocks["exporter"].prepare.assert_called_once()

    def test_exporter_finalise_called_once(self):
        _, mocks, _ = self._run()
        mocks["exporter"].finalise.assert_called_once()

    def test_cap_released_after_run(self):
        _, _, cap = self._run()
        cap.release.assert_called_once()

    def test_unique_visitors_in_summary(self):
        summary, mocks, _ = self._run(n_frames=3)
        assert summary.unique_visitors == len(mocks["tracker"].all_track_ids)

    def test_total_frames_in_summary(self):
        summary, _, _ = self._run(n_frames=4)
        assert summary.total_frames == 4


# ===========================================================================
# run() — skip_frames logic
# ===========================================================================

class TestSkipFrames:
    def test_skip_frames_reduces_detector_calls(self):
        """With skip_frames=1, every other frame is skipped → half the calls."""
        n_frames = 6
        # frames 0,2,4 are processed (3 out of 6)
        tracks_per_frame = {0: [_make_track(1, 0)], 2: [_make_track(1, 2)], 4: [_make_track(1, 4)]}

        pipeline, mocks = _make_pipeline(skip_frames=1, n_frames=n_frames)
        # tracker.track needs to handle only the processed frame_ids
        mocks["tracker"].track.side_effect = (
            lambda frame, dets, frame_id, timestamp:
            tracks_per_frame.get(frame_id, [])
        )
        mocks["analytics"].update.side_effect = (
            lambda frame_id, timestamp, tracks, detection_count, inference_time_ms:
            _make_frame_result(frame_id, tracks)
        )

        cap = _make_cap(n_frames=n_frames)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    pipeline.run("fake.mp4")

        # With skip_frames=1: process frames 0, 2, 4 → 3 calls
        assert mocks["detector"].detect.call_count == 3

    def test_skip_frames_zero_processes_all(self):
        n_frames = 4
        pipeline, mocks = _make_pipeline(skip_frames=0, n_frames=n_frames)
        cap = _make_cap(n_frames=n_frames)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    pipeline.run("fake.mp4")
        assert mocks["detector"].detect.call_count == n_frames


# ===========================================================================
# run() — empty detections
# ===========================================================================

class TestEmptyDetections:
    def test_tracker_called_even_with_no_detections(self):
        """Pipeline always delegates to tracker.track(); the tracker itself
        short-circuits when detections is empty. Pipeline must not skip it."""
        pipeline, mocks = _make_pipeline(n_frames=2)
        mocks["detector"].detect.return_value = []
        mocks["tracker"].track.return_value = []

        cap = _make_cap(n_frames=2)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    pipeline.run("fake.mp4")

        # Pipeline always calls tracker — the short-circuit is inside Tracker
        assert mocks["tracker"].track.call_count == 2

    def test_analytics_still_called_when_no_detections(self):
        """Analytics must be updated even for frames with zero detections."""
        pipeline, mocks = _make_pipeline(n_frames=2)
        mocks["detector"].detect.return_value = []
        mocks["tracker"].track.return_value = []

        cap = _make_cap(n_frames=2)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    pipeline.run("fake.mp4")

        assert mocks["analytics"].update.call_count == 2


# ===========================================================================
# run() — VideoIOError
# ===========================================================================

class TestRunVideoIOError:
    def test_missing_video_raises_video_io_error(self):
        pipeline, _ = _make_pipeline()
        with pytest.raises(VideoIOError):
            pipeline.run("/absolutely/nonexistent/video.mp4")

    def test_unopenable_video_raises_video_io_error(self, tmp_path):
        p = tmp_path / "corrupt.mp4"
        p.write_bytes(b"garbage")
        pipeline, _ = _make_pipeline()
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap_mock):
            with pytest.raises(VideoIOError):
                pipeline.run(str(p))


# ===========================================================================
# run() — interrupt handling
# ===========================================================================

class TestRunInterrupt:
    def test_interrupt_stops_loop_early(self):
        """Simulates Ctrl+C after the first frame by setting _interrupted=True."""
        n_frames = 5
        pipeline, mocks = _make_pipeline(n_frames=n_frames)

        original_detect = mocks["detector"].detect

        call_count = {"n": 0}

        def detect_and_interrupt(frame):
            call_count["n"] += 1
            if call_count["n"] == 1:
                pipeline._interrupted = True   # simulate SIGINT after frame 0
            return [Detection.from_raw(10, 20, 110, 120, 0.9, PERSON_CLASS_ID)]

        mocks["detector"].detect.side_effect = detect_and_interrupt

        cap = _make_cap(n_frames=n_frames)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    summary = pipeline.run("fake.mp4")

        # Only 1 frame was processed before interrupt
        assert mocks["detector"].detect.call_count == 1
        assert isinstance(summary, PipelineSummary)

    def test_cap_released_even_after_interrupt(self):
        pipeline, mocks = _make_pipeline(n_frames=3)
        pipeline._interrupted = True   # interrupt before any frame

        cap = _make_cap(n_frames=3)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    pipeline.run("fake.mp4")

        cap.release.assert_called_once()

    def test_partial_summary_returned_after_interrupt(self):
        """A partial run must still return a valid PipelineSummary."""
        pipeline, mocks = _make_pipeline(n_frames=5)
        call_count = {"n": 0}

        def detect_and_interrupt(frame):
            call_count["n"] += 1
            if call_count["n"] == 2:
                pipeline._interrupted = True
            return [Detection.from_raw(10, 20, 110, 120, 0.9, PERSON_CLASS_ID)]

        mocks["detector"].detect.side_effect = detect_and_interrupt

        cap = _make_cap(n_frames=5)
        with patch("src.pipeline.cv2.VideoCapture", return_value=cap):
            with patch("src.pipeline.Path.mkdir"):
                with patch("src.pipeline.Path.is_file", return_value=True):
                    summary = pipeline.run("fake.mp4")

        assert isinstance(summary, PipelineSummary)
        assert summary.total_frames <= 5
