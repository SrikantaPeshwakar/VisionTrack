"""
Unit tests for src/analytics.py

No external dependencies — everything is driven by plain Python objects.

Covers:
- Construction and __repr__
- update(): FrameResult returned, unique visitor set, detection count,
            peak concurrent, inference time, tracks defensive copy,
            frame_results accumulation
- EMA FPS: first-frame returns 0.0, seeds on second frame, blends correctly
- Properties: unique_visitor_count, avg_inference_ms, current_fps, total_frames
- get_current_stats(): empty and non-empty state
- get_summary(): all keys present, frames list correct, values accurate
- reset(): clears all state, re-usable after reset
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_models import BoundingBox, FrameResult, Track
from src.constants import PERSON_CLASS_ID


# ===========================================================================
# Helpers
# ===========================================================================

def _make_config(ema_alpha: float = 0.1) -> MagicMock:
    cfg = MagicMock()
    cfg.analytics.ema_alpha = ema_alpha
    return cfg


def _make_track(track_id: int, frame_id: int = 0, ts: float = 0.0) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(10, 20, 110, 120),
        confidence=0.9,
        class_id=PERSON_CLASS_ID,
        frame_id=frame_id,
        timestamp=ts,
    )


def _make_analytics(ema_alpha: float = 0.1):
    from src.analytics import Analytics
    return Analytics(_make_config(ema_alpha))


def _call_update(
    analytics,
    frame_id: int = 0,
    timestamp: float = 0.0,
    tracks: list | None = None,
    detection_count: int = 1,
    inference_ms: float = 30.0,
) -> FrameResult:
    return analytics.update(
        frame_id=frame_id,
        timestamp=timestamp,
        tracks=tracks if tracks is not None else [],
        detection_count=detection_count,
        inference_time_ms=inference_ms,
    )


# ===========================================================================
# Construction & repr
# ===========================================================================

class TestAnalyticsConstruction:
    def test_initial_unique_visitor_count_zero(self):
        a = _make_analytics()
        assert a.unique_visitor_count == 0

    def test_initial_total_frames_zero(self):
        a = _make_analytics()
        assert a.total_frames == 0

    def test_initial_peak_concurrent_zero(self):
        a = _make_analytics()
        assert a.peak_concurrent_tracks == 0

    def test_initial_avg_inference_ms_zero(self):
        a = _make_analytics()
        assert a.avg_inference_ms == pytest.approx(0.0)

    def test_initial_current_fps_zero(self):
        a = _make_analytics()
        assert a.current_fps == pytest.approx(0.0)

    def test_ema_alpha_from_config(self):
        a = _make_analytics(ema_alpha=0.3)
        assert a.ema_alpha == pytest.approx(0.3)

    def test_repr_contains_key_info(self):
        a = _make_analytics()
        r = repr(a)
        assert "Analytics" in r
        assert "frames=0" in r
        assert "unique=0" in r

    def test_repr_updates_after_update(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1)])
        r = repr(a)
        assert "frames=1" in r
        assert "unique=1" in r


# ===========================================================================
# update() — FrameResult
# ===========================================================================

class TestUpdateFrameResult:
    def test_returns_frame_result_instance(self):
        a = _make_analytics()
        result = _call_update(a)
        assert isinstance(result, FrameResult)

    def test_frame_id_in_result(self):
        a = _make_analytics()
        result = _call_update(a, frame_id=42)
        assert result.frame_id == 42

    def test_timestamp_in_result(self):
        a = _make_analytics()
        result = _call_update(a, timestamp=1.5)
        assert result.timestamp == pytest.approx(1.5)

    def test_detection_count_in_result(self):
        a = _make_analytics()
        result = _call_update(a, detection_count=5)
        assert result.detection_count == 5

    def test_inference_ms_in_result(self):
        a = _make_analytics()
        result = _call_update(a, inference_ms=42.5)
        assert result.inference_time_ms == pytest.approx(42.5)

    def test_tracks_copied_into_result(self):
        a = _make_analytics()
        tracks = [_make_track(1), _make_track(2)]
        result = _call_update(a, tracks=tracks)
        assert len(result.tracks) == 2

    def test_tracks_defensive_copy(self):
        """Mutating the original list must not affect the stored FrameResult."""
        a = _make_analytics()
        tracks = [_make_track(1)]
        result = _call_update(a, tracks=tracks)
        tracks.append(_make_track(2))   # mutate original
        assert len(result.tracks) == 1  # result unchanged

    def test_empty_tracks_result(self):
        a = _make_analytics()
        result = _call_update(a, tracks=[])
        assert result.active_track_count == 0

    def test_frame_results_accumulated(self):
        a = _make_analytics()
        for i in range(5):
            _call_update(a, frame_id=i)
        assert len(a.frame_results) == 5

    def test_frame_results_in_order(self):
        a = _make_analytics()
        for i in range(3):
            _call_update(a, frame_id=i)
        assert [r.frame_id for r in a.frame_results] == [0, 1, 2]


# ===========================================================================
# update() — unique visitors
# ===========================================================================

class TestUniqueVisitors:
    def test_single_track_increments_count(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1)])
        assert a.unique_visitor_count == 1

    def test_same_id_across_frames_not_double_counted(self):
        a = _make_analytics()
        for i in range(5):
            _call_update(a, frame_id=i, tracks=[_make_track(1, i)])
        assert a.unique_visitor_count == 1

    def test_different_ids_all_counted(self):
        a = _make_analytics()
        _call_update(a, frame_id=0, tracks=[_make_track(1), _make_track(2)])
        _call_update(a, frame_id=1, tracks=[_make_track(3)])
        assert a.unique_visitor_count == 3

    def test_reappearing_id_not_double_counted(self):
        a = _make_analytics()
        _call_update(a, frame_id=0, tracks=[_make_track(1)])
        _call_update(a, frame_id=1, tracks=[])          # ID 1 disappears
        _call_update(a, frame_id=2, tracks=[_make_track(1)])  # reappears
        assert a.unique_visitor_count == 1

    def test_zero_tracks_zero_visitors(self):
        a = _make_analytics()
        _call_update(a, tracks=[])
        assert a.unique_visitor_count == 0

    def test_unique_visitor_ids_contains_correct_ids(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(7), _make_track(42)])
        assert a.unique_visitor_ids == {7, 42}


# ===========================================================================
# update() — aggregate counters
# ===========================================================================

class TestAggregateCounters:
    def test_total_detection_count_accumulates(self):
        a = _make_analytics()
        _call_update(a, detection_count=3)
        _call_update(a, detection_count=5)
        assert a.total_detection_count == 8

    def test_peak_concurrent_tracks_updated(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1)])                        # 1 track
        _call_update(a, tracks=[_make_track(1), _make_track(2)])        # 2 tracks — new peak
        _call_update(a, tracks=[_make_track(1)])                        # back to 1
        assert a.peak_concurrent_tracks == 2

    def test_peak_concurrent_never_decreases(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1), _make_track(2), _make_track(3)])
        _call_update(a, tracks=[])
        assert a.peak_concurrent_tracks == 3

    def test_peak_concurrent_zero_when_no_tracks(self):
        a = _make_analytics()
        _call_update(a, tracks=[])
        assert a.peak_concurrent_tracks == 0

    def test_avg_inference_ms_single_frame(self):
        a = _make_analytics()
        _call_update(a, inference_ms=40.0)
        assert a.avg_inference_ms == pytest.approx(40.0)

    def test_avg_inference_ms_multiple_frames(self):
        a = _make_analytics()
        _call_update(a, inference_ms=20.0)
        _call_update(a, inference_ms=40.0)
        assert a.avg_inference_ms == pytest.approx(30.0)


# ===========================================================================
# EMA FPS
# ===========================================================================

class TestEmaFps:
    def test_first_frame_fps_is_zero(self):
        """First call cannot measure an interval — must return 0.0."""
        a = _make_analytics()
        result = _call_update(a)
        assert result.fps == pytest.approx(0.0)

    def test_second_frame_seeds_ema(self):
        """Second call should produce a non-zero FPS."""
        a = _make_analytics()
        _call_update(a, frame_id=0)
        result = _call_update(a, frame_id=1)
        assert result.fps > 0.0

    def test_ema_alpha_one_equals_instant_fps(self):
        """With α=1.0 the EMA equals the instantaneous FPS exactly."""
        a = _make_analytics(ema_alpha=1.0)
        t_calls = [0.0, 0.1, 0.1]  # intervals: 0.1s → 10 FPS, 0.1s → 10 FPS

        with patch("src.analytics.time") as mock_time:
            mock_time.perf_counter.side_effect = t_calls
            _call_update(a, frame_id=0)   # seeds _last_frame_time = 0.0
            result = _call_update(a, frame_id=1)  # Δt=0.1 → 10 FPS

        assert result.fps == pytest.approx(10.0, rel=0.01)

    def test_ema_smoothing_blends_values(self):
        """With α=0.5 and two different intervals, the EMA should be between them."""
        a = _make_analytics(ema_alpha=0.5)
        # Frame 0: seed at t=0.0
        # Frame 1: Δt=0.1 → 10 FPS, seeds EMA=10
        # Frame 2: Δt=0.05 → 20 FPS, EMA = 0.5*20 + 0.5*10 = 15
        t_calls = [0.0, 0.1, 0.15]

        with patch("src.analytics.time") as mock_time:
            mock_time.perf_counter.side_effect = t_calls
            _call_update(a, frame_id=0)
            _call_update(a, frame_id=1)
            result = _call_update(a, frame_id=2)

        assert result.fps == pytest.approx(15.0, rel=0.01)

    def test_fps_stored_in_frame_result(self):
        """The fps field of FrameResult must match analytics.current_fps."""
        a = _make_analytics()
        _call_update(a, frame_id=0)
        result = _call_update(a, frame_id=1)
        assert result.fps == pytest.approx(a.current_fps)


# ===========================================================================
# get_current_stats()
# ===========================================================================

class TestGetCurrentStats:
    def test_empty_state_returns_zeros(self):
        a = _make_analytics()
        stats = a.get_current_stats()
        assert stats["frame_id"] == 0
        assert stats["unique_visitors"] == 0
        assert stats["active_tracks"] == 0

    def test_after_update_reflects_latest_frame(self):
        a = _make_analytics()
        _call_update(a, frame_id=10, tracks=[_make_track(1), _make_track(2)])
        stats = a.get_current_stats()
        assert stats["frame_id"] == 10
        assert stats["unique_visitors"] == 2
        assert stats["active_tracks"] == 2

    def test_returns_all_expected_keys(self):
        a = _make_analytics()
        stats = a.get_current_stats()
        assert "frame_id" in stats
        assert "unique_visitors" in stats
        assert "active_tracks" in stats
        assert "current_fps" in stats
        assert "avg_inference_ms" in stats


# ===========================================================================
# get_summary()
# ===========================================================================

class TestGetSummary:
    def test_returns_dict(self):
        a = _make_analytics()
        assert isinstance(a.get_summary(), dict)

    def test_all_top_level_keys_present(self):
        a = _make_analytics()
        s = a.get_summary()
        for key in [
            "total_frames", "unique_visitors", "peak_concurrent_tracks",
            "total_detections", "avg_fps", "avg_inference_ms", "frames",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_total_frames_correct(self):
        a = _make_analytics()
        for i in range(4):
            _call_update(a, frame_id=i)
        assert a.get_summary()["total_frames"] == 4

    def test_unique_visitors_correct(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1), _make_track(2)])
        _call_update(a, tracks=[_make_track(3)])
        assert a.get_summary()["unique_visitors"] == 3

    def test_peak_concurrent_correct(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1)])
        _call_update(a, tracks=[_make_track(1), _make_track(2), _make_track(3)])
        assert a.get_summary()["peak_concurrent_tracks"] == 3

    def test_total_detections_correct(self):
        a = _make_analytics()
        _call_update(a, detection_count=4)
        _call_update(a, detection_count=6)
        assert a.get_summary()["total_detections"] == 10

    def test_frames_list_length_matches_total_frames(self):
        a = _make_analytics()
        for i in range(3):
            _call_update(a, frame_id=i)
        s = a.get_summary()
        assert len(s["frames"]) == s["total_frames"]

    def test_frames_list_contains_frame_ids(self):
        a = _make_analytics()
        for i in range(3):
            _call_update(a, frame_id=i)
        ids = [f["frame_id"] for f in a.get_summary()["frames"]]
        assert ids == [0, 1, 2]

    def test_frame_entry_has_required_keys(self):
        a = _make_analytics()
        _call_update(a, frame_id=0, tracks=[_make_track(1)], detection_count=1, inference_ms=25.0)
        frame_entry = a.get_summary()["frames"][0]
        for key in ["frame_id", "timestamp", "active_tracks", "track_ids",
                    "detection_count", "inference_ms", "fps"]:
            assert key in frame_entry, f"Missing key: {key}"

    def test_frame_entry_track_ids(self):
        a = _make_analytics()
        _call_update(a, frame_id=0, tracks=[_make_track(3), _make_track(7)])
        entry = a.get_summary()["frames"][0]
        assert set(entry["track_ids"]) == {3, 7}

    def test_summary_is_json_serialisable(self):
        """get_summary() must be JSON-serialisable for Exporter use."""
        import json
        a = _make_analytics()
        _call_update(a, frame_id=0, tracks=[_make_track(1)])
        json.dumps(a.get_summary())   # must not raise


# ===========================================================================
# reset()
# ===========================================================================

class TestReset:
    def test_reset_clears_unique_visitors(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1)])
        a.reset()
        assert a.unique_visitor_count == 0

    def test_reset_clears_frame_results(self):
        a = _make_analytics()
        _call_update(a)
        a.reset()
        assert len(a.frame_results) == 0

    def test_reset_clears_total_detection_count(self):
        a = _make_analytics()
        _call_update(a, detection_count=5)
        a.reset()
        assert a.total_detection_count == 0

    def test_reset_clears_peak_concurrent(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1), _make_track(2)])
        a.reset()
        assert a.peak_concurrent_tracks == 0

    def test_reset_clears_fps(self):
        a = _make_analytics()
        _call_update(a, frame_id=0)
        _call_update(a, frame_id=1)
        a.reset()
        assert a.current_fps == pytest.approx(0.0)

    def test_reset_clears_inference_tracking(self):
        a = _make_analytics()
        _call_update(a, inference_ms=50.0)
        a.reset()
        assert a.avg_inference_ms == pytest.approx(0.0)

    def test_usable_after_reset(self):
        a = _make_analytics()
        _call_update(a, tracks=[_make_track(1)])
        a.reset()
        _call_update(a, frame_id=0, tracks=[_make_track(99)])
        assert a.unique_visitor_count == 1
        assert a.unique_visitor_ids == {99}

    def test_total_frames_zero_after_reset(self):
        a = _make_analytics()
        for i in range(5):
            _call_update(a, frame_id=i)
        a.reset()
        assert a.total_frames == 0
