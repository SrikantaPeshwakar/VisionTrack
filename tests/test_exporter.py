"""
Unit tests for src/exporter.py

Uses pytest's tmp_path fixture so all file I/O is self-contained and
cleaned up automatically.  VideoWriter is mocked to avoid codec
dependencies; JSON and CSV tests use real disk I/O.

Covers:
- Construction and __repr__
- prepare(): creates output directory, opens VideoWriter when save_video=True,
             skips VideoWriter when save_video=False, ExportError on bad codec
- write_frame(): calls VideoWriter.write(), no-op when save_video=False,
                 no-op on empty/None frame, increments frames_written
- finalise(): releases VideoWriter, sets summary paths, JSON created,
              CSV created, skips outputs based on flags
- _save_json(): file exists, valid JSON, all top-level keys, tracks list,
                metadata fields, frames list, JSON serialisable
- _save_csv(): file exists, correct headers, one row per track per frame,
               empty frame_results produces header-only file
- ExportError propagation from VideoWriter.write() failure
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_models import (
    BoundingBox, FrameResult, PipelineSummary, Track, TrackSummary,
)
from src.constants import (
    DEFAULT_CSV_FILENAME,
    DEFAULT_JSON_FILENAME,
    DEFAULT_VIDEO_FILENAME,
    PERSON_CLASS_ID,
)
from exceptions import ExportError


# ===========================================================================
# Helpers
# ===========================================================================

def _make_config(
    save_video: bool = True,
    save_json: bool = True,
    save_csv: bool = True,
    codec: str = "mp4v",
) -> MagicMock:
    cfg = MagicMock()
    cfg.export.save_video = save_video
    cfg.export.save_json  = save_json
    cfg.export.save_csv   = save_csv
    cfg.video.output_codec = codec
    return cfg


def _make_exporter(**kwargs):
    from src.exporter import Exporter
    return Exporter(_make_config(**kwargs))


def _make_summary() -> PipelineSummary:
    return PipelineSummary(
        total_frames=100,
        total_processing_time=3.5,
        avg_fps=28.6,
        avg_inference_time_ms=35.0,
        unique_visitors=12,
        peak_concurrent_tracks=5,
        avg_dwell_time=4.2,
    )


def _make_track(track_id: int = 1, frame_id: int = 0) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(10.0, 20.0, 110.0, 120.0),
        confidence=0.9,
        class_id=PERSON_CLASS_ID,
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
    )


def _make_frame_result(
    frame_id: int = 0,
    tracks: list | None = None,
    fps: float = 28.0,
) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        timestamp=frame_id / 30.0,
        tracks=tracks or [],
        detection_count=len(tracks or []),
        inference_time_ms=35.0,
        fps=fps,
    )


def _make_track_summary(track_id: int = 1) -> TrackSummary:
    return TrackSummary(
        track_id=track_id,
        first_seen_frame=0,
        last_seen_frame=30,
        first_seen_time=0.0,
        last_seen_time=1.0,
        total_appearances=30,
        trajectory=[BoundingBox(10, 20, 110, 120)],
    )


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ===========================================================================
# Construction & __repr__
# ===========================================================================

class TestExporterConstruction:
    def test_attributes_from_config(self):
        e = _make_exporter(save_video=True, save_json=False, save_csv=True, codec="XVID")
        assert e.save_video is True
        assert e.save_json is False
        assert e.save_csv is True
        assert e._codec == "XVID"

    def test_initial_frames_written_zero(self):
        e = _make_exporter()
        assert e._frames_written == 0

    def test_initial_video_writer_none(self):
        e = _make_exporter()
        assert e._video_writer is None

    def test_repr_contains_key_info(self):
        e = _make_exporter(save_video=True, save_json=False, save_csv=True, codec="mp4v")
        r = repr(e)
        assert "Exporter" in r
        assert "video=True" in r
        assert "json=False" in r
        assert "csv=True" in r
        assert "mp4v" in r


# ===========================================================================
# prepare()
# ===========================================================================

class TestPrepare:
    def test_creates_output_directory(self, tmp_path):
        e = _make_exporter(save_video=False)
        out_dir = str(tmp_path / "run_001")
        e.prepare(out_dir, fps=30.0, width=640, height=480)
        assert Path(out_dir).is_dir()

    def test_sets_output_dir_attribute(self, tmp_path):
        e = _make_exporter(save_video=False)
        out_dir = str(tmp_path / "run_002")
        e.prepare(out_dir, fps=30.0, width=640, height=480)
        assert e.output_dir == out_dir

    def test_resets_frames_written(self, tmp_path):
        e = _make_exporter(save_video=False)
        e._frames_written = 99
        e.prepare(str(tmp_path / "run"), fps=30.0, width=640, height=480)
        assert e._frames_written == 0

    def test_save_video_false_no_video_writer(self, tmp_path):
        e = _make_exporter(save_video=False)
        e.prepare(str(tmp_path / "run"), fps=30.0, width=640, height=480)
        assert e._video_writer is None

    def test_save_video_true_opens_writer(self, tmp_path):
        e = _make_exporter(save_video=True)
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        with patch("src.exporter.cv2.VideoWriter", return_value=mock_writer):
            e.prepare(str(tmp_path / "run"), fps=30.0, width=640, height=480)
        assert e._video_writer is mock_writer

    def test_bad_codec_raises_export_error(self, tmp_path):
        e = _make_exporter(save_video=True)
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = False  # codec failed to open
        with patch("src.exporter.cv2.VideoWriter", return_value=mock_writer):
            with pytest.raises(ExportError):
                e.prepare(str(tmp_path / "run"), fps=30.0, width=640, height=480)

    def test_video_path_uses_default_filename(self, tmp_path):
        e = _make_exporter(save_video=True)
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        out_dir = str(tmp_path / "run")
        with patch("src.exporter.cv2.VideoWriter", return_value=mock_writer):
            e.prepare(out_dir, fps=30.0, width=640, height=480)
        assert DEFAULT_VIDEO_FILENAME in e._video_path


# ===========================================================================
# write_frame()
# ===========================================================================

class TestWriteFrame:
    def _prepared_exporter(self, tmp_path):
        e = _make_exporter(save_video=True)
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        with patch("src.exporter.cv2.VideoWriter", return_value=mock_writer):
            e.prepare(str(tmp_path / "run"), fps=30.0, width=640, height=480)
        e._video_writer = mock_writer
        return e, mock_writer

    def test_write_frame_calls_writer_write(self, tmp_path):
        e, writer = self._prepared_exporter(tmp_path)
        e.write_frame(_blank_frame())
        writer.write.assert_called_once()

    def test_write_frame_increments_counter(self, tmp_path):
        e, _ = self._prepared_exporter(tmp_path)
        e.write_frame(_blank_frame())
        e.write_frame(_blank_frame())
        assert e._frames_written == 2

    def test_write_frame_noop_when_save_video_false(self, tmp_path):
        e = _make_exporter(save_video=False)
        e.prepare(str(tmp_path / "run"), fps=30.0, width=640, height=480)
        # No writer — should not raise
        e.write_frame(_blank_frame())
        assert e._frames_written == 0

    def test_write_frame_noop_on_none_frame(self, tmp_path):
        e, writer = self._prepared_exporter(tmp_path)
        e.write_frame(None)
        writer.write.assert_not_called()

    def test_write_frame_noop_on_empty_frame(self, tmp_path):
        e, writer = self._prepared_exporter(tmp_path)
        e.write_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        writer.write.assert_not_called()

    def test_write_frame_raises_export_error_on_writer_failure(self, tmp_path):
        e, writer = self._prepared_exporter(tmp_path)
        writer.write.side_effect = RuntimeError("disk full")
        with pytest.raises(ExportError):
            e.write_frame(_blank_frame())


# ===========================================================================
# finalise()
# ===========================================================================

class TestFinalise:
    def _run_finalise(self, tmp_path, **exporter_kwargs):
        e = _make_exporter(**exporter_kwargs)
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True
        out_dir = str(tmp_path / "run")
        with patch("src.exporter.cv2.VideoWriter", return_value=mock_writer):
            e.prepare(out_dir, fps=30.0, width=640, height=480)
        e._video_writer = mock_writer if exporter_kwargs.get("save_video", True) else None

        summary = _make_summary()
        frame_results = [
            _make_frame_result(0, [_make_track(1, 0)]),
            _make_frame_result(1, [_make_track(1, 1), _make_track(2, 1)]),
        ]
        track_summaries = [_make_track_summary(1), _make_track_summary(2)]
        analytics_summary = {"total_frames": 2, "unique_visitors": 2, "frames": []}
        config_dict = {"model": {"type": "yolov8n"}}

        return e.finalise(
            summary, frame_results, track_summaries,
            analytics_summary, config_dict, "metro.mp4"
        ), e, out_dir

    def test_returns_pipeline_summary(self, tmp_path):
        summary, _, _ = self._run_finalise(tmp_path)
        assert isinstance(summary, PipelineSummary)

    def test_video_writer_released(self, tmp_path):
        _, e, _ = self._run_finalise(tmp_path)
        # writer is released — set to None
        assert e._video_writer is None

    def test_output_video_path_set(self, tmp_path):
        summary, _, out_dir = self._run_finalise(tmp_path)
        assert summary.output_video_path is not None
        assert DEFAULT_VIDEO_FILENAME in summary.output_video_path

    def test_output_json_path_set(self, tmp_path):
        summary, _, out_dir = self._run_finalise(tmp_path)
        assert summary.output_json_path is not None
        assert DEFAULT_JSON_FILENAME in summary.output_json_path

    def test_output_csv_path_set(self, tmp_path):
        summary, _, out_dir = self._run_finalise(tmp_path)
        assert summary.output_csv_path is not None
        assert DEFAULT_CSV_FILENAME in summary.output_csv_path

    def test_save_video_false_no_video_path(self, tmp_path):
        summary, _, _ = self._run_finalise(tmp_path, save_video=False)
        assert summary.output_video_path is None

    def test_save_json_false_no_json_path(self, tmp_path):
        summary, _, _ = self._run_finalise(
            tmp_path, save_video=False, save_json=False
        )
        assert summary.output_json_path is None

    def test_save_csv_false_no_csv_path(self, tmp_path):
        summary, _, _ = self._run_finalise(
            tmp_path, save_video=False, save_csv=False
        )
        assert summary.output_csv_path is None


# ===========================================================================
# _save_json()
# ===========================================================================

class TestSaveJson:
    @pytest.fixture
    def json_data(self, tmp_path):
        from src.exporter import Exporter
        e = Exporter(_make_config(save_json=True, save_video=False, save_csv=False))
        path = str(tmp_path / DEFAULT_JSON_FILENAME)
        frame_results = [
            _make_frame_result(0, [_make_track(1, 0)]),
            _make_frame_result(1, [_make_track(2, 1)]),
        ]
        track_summaries = [_make_track_summary(1), _make_track_summary(2)]
        analytics_summary = {
            "total_frames": 2, "unique_visitors": 2, "frames": [],
            "peak_concurrent_tracks": 1, "total_detections": 2,
            "avg_fps": 28.0, "avg_inference_ms": 35.0,
        }
        e._save_json(
            path,
            _make_summary(),
            frame_results,
            track_summaries,
            analytics_summary,
            {"model": {"type": "yolov8n"}},
            "metro.mp4",
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data, path

    def test_file_created(self, json_data):
        _, path = json_data
        assert Path(path).is_file()

    def test_top_level_keys_present(self, json_data):
        data, _ = json_data
        for key in ["metadata", "summary", "analytics", "tracks"]:
            assert key in data, f"Missing key: {key}"

    def test_metadata_input_video(self, json_data):
        data, _ = json_data
        assert data["metadata"]["input_video"] == "metro.mp4"

    def test_metadata_generated_at_present(self, json_data):
        data, _ = json_data
        assert "generated_at" in data["metadata"]

    def test_metadata_config_present(self, json_data):
        data, _ = json_data
        assert "config" in data["metadata"]
        assert data["metadata"]["config"]["model"]["type"] == "yolov8n"

    def test_summary_fields(self, json_data):
        data, _ = json_data
        s = data["summary"]
        assert s["total_frames"] == 100
        assert s["unique_visitors"] == 12
        assert s["peak_concurrent_tracks"] == 5

    def test_tracks_list_length(self, json_data):
        data, _ = json_data
        assert len(data["tracks"]) == 2

    def test_track_entry_fields(self, json_data):
        data, _ = json_data
        t = data["tracks"][0]
        for key in ["track_id", "first_seen_frame", "last_seen_frame",
                    "first_seen_time", "last_seen_time", "dwell_time",
                    "total_appearances", "trajectory"]:
            assert key in t, f"Missing track key: {key}"

    def test_trajectory_bbox_fields(self, json_data):
        data, _ = json_data
        bbox = data["tracks"][0]["trajectory"][0]
        assert "x1" in bbox and "y1" in bbox and "x2" in bbox and "y2" in bbox

    def test_json_is_valid(self, json_data):
        """Re-parse the file to confirm it is well-formed JSON."""
        _, path = json_data
        with open(path, encoding="utf-8") as f:
            parsed = json.load(f)
        assert isinstance(parsed, dict)


# ===========================================================================
# _save_csv()
# ===========================================================================

class TestSaveCsv:
    @pytest.fixture
    def csv_path(self, tmp_path):
        from src.exporter import Exporter
        e = Exporter(_make_config(save_csv=True, save_video=False, save_json=False))
        path = str(tmp_path / DEFAULT_CSV_FILENAME)
        frame_results = [
            _make_frame_result(0, [_make_track(1, 0), _make_track(2, 0)]),
            _make_frame_result(1, [_make_track(1, 1)]),
            _make_frame_result(2, []),  # no tracks — no rows
        ]
        e._save_csv(path, frame_results)
        return path

    def test_file_created(self, csv_path):
        assert Path(csv_path).is_file()

    def test_header_row_correct(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "frame_id", "timestamp", "track_id",
                "x1", "y1", "x2", "y2", "confidence", "fps",
            }

    def test_row_count_matches_total_tracks_across_frames(self, csv_path):
        """frame 0: 2 tracks, frame 1: 1 track, frame 2: 0 tracks → 3 rows."""
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_frame_id_values(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        frame_ids = [int(r["frame_id"]) for r in rows]
        assert frame_ids == [0, 0, 1]

    def test_track_id_values(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        track_ids = [int(r["track_id"]) for r in rows]
        assert track_ids == [1, 2, 1]

    def test_bbox_columns_numeric(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for col in ["x1", "y1", "x2", "y2"]:
                float(row[col])  # must not raise

    def test_confidence_column_numeric(self, csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert 0.0 <= float(row["confidence"]) <= 1.0

    def test_empty_frame_results_produces_header_only(self, tmp_path):
        from src.exporter import Exporter
        e = Exporter(_make_config())
        path = str(tmp_path / "empty.csv")
        e._save_csv(path, [])
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == []

    def test_frames_with_no_tracks_produce_no_rows(self, tmp_path):
        from src.exporter import Exporter
        e = Exporter(_make_config())
        path = str(tmp_path / "notrack.csv")
        e._save_csv(path, [_make_frame_result(0, []), _make_frame_result(1, [])])
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == []
