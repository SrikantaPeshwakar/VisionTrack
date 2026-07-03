"""
Unit tests for src/tracker.py

All tests mock the Ultralytics YOLO model so no GPU, no internet connection,
and no model weights are required.

Covers:
- Tracker construction and __repr__
- _parse_results: happy path, no boxes, no IDs, multi-track,
                  non-person filtered, degenerate bbox filtered,
                  negative track ID filtered, output types
- track(): happy path, empty detections short-circuit, TrackingError on
           model exception, model called with correct args
- History management: first_seen, last_seen, all_track_ids, track_history
                      updated correctly across multiple frames
- get_track_summary(): known ID, unknown ID, trajectory length
- get_all_summaries(): ordering, dwell_time calculation
- reset(): clears all state
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exceptions import TrackingError
from src.constants import PERSON_CLASS_ID
from src.data_models import BoundingBox, Detection, Track, TrackSummary

# ===========================================================================
# Helpers
# ===========================================================================


def _make_config(
    tracker_config: str = "config/botsort.yaml",
    persist: bool = True,
) -> MagicMock:
    cfg = MagicMock()
    cfg.tracker.config_file = tracker_config
    cfg.tracker.persist = persist
    return cfg


def _make_detection(x1=10.0, y1=20.0, x2=110.0, y2=120.0, conf=0.9) -> Detection:
    return Detection.from_raw(x1, y1, x2, y2, conf, PERSON_CLASS_ID)


def _make_tracking_result(
    rows: list[list[float]] | None,
    ids: list[int] | None,
) -> list[MagicMock]:
    """Build a mock Ultralytics tracking result.

    Args:
        rows: List of [x1, y1, x2, y2, conf, class_id] rows. None = no boxes.
        ids:  Corresponding track IDs. None = boxes.id is None.
    """
    result = MagicMock()

    if rows is None:
        result.boxes = None
        return [result]

    result.boxes = MagicMock()

    if ids is None:
        result.boxes.id = None
    else:
        id_tensor = MagicMock()
        id_tensor.cpu.return_value.numpy.return_value = np.array(ids, dtype=np.float32)
        result.boxes.id = id_tensor

    data_tensor = MagicMock()
    data_tensor.cpu.return_value.numpy.return_value = (
        np.array(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)
    )
    result.boxes.data = data_tensor

    return [result]


def _make_tracker(cfg=None, model=None):
    """Construct a Tracker with a mock model."""
    from src.tracker import Tracker

    if cfg is None:
        cfg = _make_config()
    if model is None:
        model = MagicMock()
    return Tracker(cfg, model=model)


# ===========================================================================
# Construction & repr
# ===========================================================================


class TestTrackerConstruction:
    def test_attributes_set_from_config(self):
        tracker = _make_tracker(
            _make_config(
                tracker_config="config/botsort.yaml",
                persist=True,
            )
        )
        assert tracker.tracker_config == "config/botsort.yaml"
        assert tracker.persist is True

    def test_initial_state_empty(self):
        tracker = _make_tracker()
        assert len(tracker.all_track_ids) == 0
        assert len(tracker.track_history) == 0
        assert len(tracker.first_seen) == 0
        assert len(tracker.last_seen) == 0

    def test_repr_contains_key_info(self):
        tracker = _make_tracker()
        r = repr(tracker)
        assert "Tracker" in r
        assert "botsort.yaml" in r
        assert "persist=True" in r
        assert "unique_tracks=0" in r

    def test_repr_updates_after_tracking(self):
        model = MagicMock()
        model.track.return_value = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]],
            [1],
        )
        tracker = _make_tracker(model=model)
        tracker.track(
            np.zeros((480, 640, 3), dtype=np.uint8),
            [_make_detection()],
            frame_id=0,
            timestamp=0.0,
        )
        assert "unique_tracks=1" in repr(tracker)


# ===========================================================================
# _parse_results
# ===========================================================================


class TestParseResults:
    @pytest.fixture(autouse=True)
    def _tracker(self):
        self.tracker = _make_tracker()

    def test_empty_results_list(self):
        assert self.tracker._parse_results([], 0, 0.0) == []

    def test_boxes_is_none(self):
        result = _make_tracking_result(None, None)
        assert self.tracker._parse_results(result, 0, 0.0) == []

    def test_boxes_id_is_none(self):
        """When tracker hasn't associated yet, boxes.id is None → empty."""
        result = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]],
            None,
        )
        assert self.tracker._parse_results(result, 0, 0.0) == []

    def test_single_track_parsed(self):
        result = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]],
            [1],
        )
        tracks = self.tracker._parse_results(result, 5, 0.5)
        assert len(tracks) == 1
        t = tracks[0]
        assert isinstance(t, Track)
        assert t.track_id == 1
        assert t.frame_id == 5
        assert t.timestamp == pytest.approx(0.5)
        assert t.bbox.x1 == pytest.approx(10.0)
        assert t.bbox.y2 == pytest.approx(120.0)
        assert t.confidence == pytest.approx(0.9)
        assert t.class_id == PERSON_CLASS_ID

    def test_multiple_tracks_parsed(self):
        result = _make_tracking_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
                [200.0, 50.0, 300.0, 200.0, 0.7, PERSON_CLASS_ID],
                [400.0, 10.0, 500.0, 300.0, 0.6, PERSON_CLASS_ID],
            ],
            [1, 2, 3],
        )
        tracks = self.tracker._parse_results(result, 0, 0.0)
        assert len(tracks) == 3
        assert [t.track_id for t in tracks] == [1, 2, 3]

    def test_non_person_class_filtered(self):
        result = _make_tracking_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, 2],  # car
                [200.0, 50.0, 300.0, 200.0, 0.8, 16],  # dog
            ],
            [1, 2],
        )
        tracks = self.tracker._parse_results(result, 0, 0.0)
        assert tracks == []

    def test_mixed_classes_only_persons_returned(self):
        result = _make_tracking_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
                [200.0, 50.0, 300.0, 200.0, 0.8, 2],  # car
            ],
            [1, 2],
        )
        tracks = self.tracker._parse_results(result, 0, 0.0)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_degenerate_bbox_zero_width_filtered(self):
        result = _make_tracking_result(
            [[100.0, 50.0, 100.0, 200.0, 0.9, PERSON_CLASS_ID]],
            [1],
        )
        assert self.tracker._parse_results(result, 0, 0.0) == []

    def test_degenerate_bbox_zero_height_filtered(self):
        result = _make_tracking_result(
            [[100.0, 50.0, 200.0, 50.0, 0.9, PERSON_CLASS_ID]],
            [1],
        )
        assert self.tracker._parse_results(result, 0, 0.0) == []

    def test_negative_track_id_filtered(self):
        result = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]],
            [-1],
        )
        assert self.tracker._parse_results(result, 0, 0.0) == []

    def test_output_types(self):
        result = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]],
            [42],
        )
        tracks = self.tracker._parse_results(result, 0, 0.0)
        t = tracks[0]
        assert isinstance(t.track_id, int)
        assert isinstance(t.bbox, BoundingBox)
        assert isinstance(t.confidence, float)
        assert isinstance(t.class_id, int)
        assert isinstance(t.frame_id, int)
        assert isinstance(t.timestamp, float)

    def test_frame_id_and_timestamp_embedded(self):
        result = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]],
            [7],
        )
        tracks = self.tracker._parse_results(result, frame_id=99, timestamp=3.14)
        assert tracks[0].frame_id == 99
        assert tracks[0].timestamp == pytest.approx(3.14)


# ===========================================================================
# track()
# ===========================================================================


class TestTrack:
    def _make_model_with_result(self, rows, ids):
        model = MagicMock()
        model.track.return_value = _make_tracking_result(rows, ids)
        return model

    def test_returns_list(self):
        model = self._make_model_with_result([], None)
        tracker = _make_tracker(model=model)
        result = tracker.track(
            np.zeros((480, 640, 3), dtype=np.uint8),
            [_make_detection()],
            frame_id=0,
            timestamp=0.0,
        )
        assert isinstance(result, list)

    def test_single_track_returned(self):
        model = self._make_model_with_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]], [1]
        )
        tracker = _make_tracker(model=model)
        tracks = tracker.track(
            np.zeros((480, 640, 3), dtype=np.uint8),
            [_make_detection()],
            frame_id=0,
            timestamp=0.0,
        )
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_empty_detections_short_circuits(self):
        """No detections → model.track must NOT be called."""
        model = MagicMock()
        tracker = _make_tracker(model=model)
        result = tracker.track(
            np.zeros((480, 640, 3), dtype=np.uint8),
            [],  # empty detections
            frame_id=0,
            timestamp=0.0,
        )
        assert result == []
        model.track.assert_not_called()

    def test_model_called_with_correct_kwargs(self):
        model = self._make_model_with_result([], None)
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)

        model.track.assert_called_once()
        _, kwargs = model.track.call_args
        assert kwargs.get("persist") is True
        assert kwargs.get("tracker") == "config/botsort.yaml"
        assert kwargs.get("verbose") is False

    def test_model_exception_raises_tracking_error(self):
        model = MagicMock()
        model.track.side_effect = RuntimeError("tracker crashed")
        tracker = _make_tracker(model=model)
        with pytest.raises(TrackingError, match="tracker crashed"):
            tracker.track(
                np.zeros((480, 640, 3), dtype=np.uint8),
                [_make_detection()],
                frame_id=0,
                timestamp=0.0,
            )

    def test_multiple_tracks_returned(self):
        model = self._make_model_with_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
                [200.0, 50.0, 300.0, 200.0, 0.7, PERSON_CLASS_ID],
            ],
            [1, 2],
        )
        tracker = _make_tracker(model=model)
        tracks = tracker.track(
            np.zeros((480, 640, 3), dtype=np.uint8),
            [_make_detection(), _make_detection()],
            frame_id=5,
            timestamp=0.5,
        )
        assert len(tracks) == 2
        assert {t.track_id for t in tracks} == {1, 2}


# ===========================================================================
# History management
# ===========================================================================


class TestHistoryManagement:
    def _tracker_with_two_frames(self):
        """Return a Tracker that has processed two frames:
        Frame 0: tracks [1, 2]
        Frame 1: tracks [1, 2, 3]  (ID 3 is new; IDs 1 & 2 persist)
        """
        model = MagicMock()
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        model.track.return_value = _make_tracking_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
                [200.0, 50.0, 300.0, 200.0, 0.8, PERSON_CLASS_ID],
            ],
            [1, 2],
        )
        tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)

        model.track.return_value = _make_tracking_result(
            [
                [11.0, 21.0, 111.0, 121.0, 0.88, PERSON_CLASS_ID],
                [201.0, 51.0, 301.0, 201.0, 0.77, PERSON_CLASS_ID],
                [400.0, 100.0, 500.0, 300.0, 0.65, PERSON_CLASS_ID],
            ],
            [1, 2, 3],
        )
        tracker.track(frame, [_make_detection()], frame_id=1, timestamp=1.0 / 30)

        return tracker

    def test_all_track_ids_collected(self):
        tracker = self._tracker_with_two_frames()
        assert tracker.all_track_ids == {1, 2, 3}

    def test_history_length_per_track(self):
        tracker = self._tracker_with_two_frames()
        assert len(tracker.track_history[1]) == 2  # appeared in both frames
        assert len(tracker.track_history[2]) == 2
        assert len(tracker.track_history[3]) == 1  # only frame 1

    def test_first_seen_correct(self):
        tracker = self._tracker_with_two_frames()
        assert tracker.first_seen[1] == (0, pytest.approx(0.0))
        assert tracker.first_seen[3][0] == 1  # frame 1

    def test_last_seen_updated(self):
        tracker = self._tracker_with_two_frames()
        assert tracker.last_seen[1][0] == 1  # last seen in frame 1
        assert tracker.last_seen[3][0] == 1  # only seen in frame 1

    def test_bbox_updated_per_frame(self):
        tracker = self._tracker_with_two_frames()
        # Frame 0: x1=10, Frame 1: x1=11
        assert tracker.track_history[1][0].bbox.x1 == pytest.approx(10.0)
        assert tracker.track_history[1][1].bbox.x1 == pytest.approx(11.0)

    def test_new_id_not_in_first_seen_before_appearance(self):
        tracker = _make_tracker()
        assert 99 not in tracker.first_seen

    def test_id_registered_exactly_once(self):
        """A track seen in 5 frames should have first_seen set only once."""
        model = MagicMock()
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(5):
            model.track.return_value = _make_tracking_result(
                [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]], [1]
            )
            tracker.track(frame, [_make_detection()], frame_id=i, timestamp=float(i))

        # first_seen should still be frame 0
        assert tracker.first_seen[1] == (0, pytest.approx(0.0))
        assert len(tracker.track_history[1]) == 5


# ===========================================================================
# get_track_summary()
# ===========================================================================


class TestGetTrackSummary:
    @pytest.fixture
    def tracker(self):
        model = MagicMock()
        t = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        model.track.return_value = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]], [1]
        )
        t.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)

        model.track.return_value = _make_tracking_result(
            [[11.0, 21.0, 111.0, 121.0, 0.88, PERSON_CLASS_ID]], [1]
        )
        t.track(frame, [_make_detection()], frame_id=1, timestamp=1.0)
        return t

    def test_unknown_id_returns_none(self, tracker):
        assert tracker.get_track_summary(999) is None

    def test_returns_track_summary_instance(self, tracker):
        summary = tracker.get_track_summary(1)
        assert isinstance(summary, TrackSummary)

    def test_summary_track_id(self, tracker):
        assert tracker.get_track_summary(1).track_id == 1

    def test_summary_first_seen_frame(self, tracker):
        assert tracker.get_track_summary(1).first_seen_frame == 0

    def test_summary_last_seen_frame(self, tracker):
        assert tracker.get_track_summary(1).last_seen_frame == 1

    def test_summary_first_seen_time(self, tracker):
        assert tracker.get_track_summary(1).first_seen_time == pytest.approx(0.0)

    def test_summary_last_seen_time(self, tracker):
        assert tracker.get_track_summary(1).last_seen_time == pytest.approx(1.0)

    def test_summary_total_appearances(self, tracker):
        assert tracker.get_track_summary(1).total_appearances == 2

    def test_summary_dwell_time(self, tracker):
        assert tracker.get_track_summary(1).dwell_time == pytest.approx(1.0)

    def test_summary_trajectory_length(self, tracker):
        summary = tracker.get_track_summary(1)
        assert len(summary.trajectory) == 2

    def test_summary_trajectory_types(self, tracker):
        summary = tracker.get_track_summary(1)
        assert all(isinstance(b, BoundingBox) for b in summary.trajectory)


# ===========================================================================
# get_all_summaries()
# ===========================================================================


class TestGetAllSummaries:
    def test_returns_list(self):
        tracker = _make_tracker()
        assert isinstance(tracker.get_all_summaries(), list)

    def test_empty_when_no_tracks(self):
        tracker = _make_tracker()
        assert tracker.get_all_summaries() == []

    def test_sorted_by_track_id(self):
        model = MagicMock()
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        model.track.return_value = _make_tracking_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
                [200.0, 50.0, 300.0, 200.0, 0.7, PERSON_CLASS_ID],
                [400.0, 10.0, 500.0, 300.0, 0.6, PERSON_CLASS_ID],
            ],
            [5, 2, 9],  # deliberately unsorted
        )
        tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)

        summaries = tracker.get_all_summaries()
        ids = [s.track_id for s in summaries]
        assert ids == sorted(ids)

    def test_count_matches_unique_tracks(self):
        model = MagicMock()
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        model.track.return_value = _make_tracking_result(
            [
                [10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID],
                [200.0, 50.0, 300.0, 200.0, 0.7, PERSON_CLASS_ID],
            ],
            [1, 2],
        )
        tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)
        assert len(tracker.get_all_summaries()) == 2


# ===========================================================================
# reset()
# ===========================================================================


class TestReset:
    def test_reset_clears_all_state(self):
        model = MagicMock()
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        model.track.return_value = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]], [1]
        )
        tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)
        assert len(tracker.all_track_ids) == 1

        tracker.reset()

        assert len(tracker.all_track_ids) == 0
        assert len(tracker.track_history) == 0
        assert len(tracker.first_seen) == 0
        assert len(tracker.last_seen) == 0

    def test_tracking_works_after_reset(self):
        model = MagicMock()
        tracker = _make_tracker(model=model)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        model.track.return_value = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]], [1]
        )
        tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)
        tracker.reset()

        model.track.return_value = _make_tracking_result(
            [[10.0, 20.0, 110.0, 120.0, 0.9, PERSON_CLASS_ID]], [1]
        )
        tracks = tracker.track(frame, [_make_detection()], frame_id=0, timestamp=0.0)

        assert len(tracks) == 1
        assert len(tracker.all_track_ids) == 1
