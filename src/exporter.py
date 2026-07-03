"""
Exporter Module for VisionTrack.

Responsible for all file output produced by a pipeline run:
  - Annotated output video (MP4 via OpenCV VideoWriter)
  - Structured analytics log (JSON) — per-frame + per-track + summary
  - Flat tracking CSV — one row per detection, ready for spreadsheet analysis

The Exporter follows a three-phase lifecycle that matches the pipeline loop:

    1. prepare(output_dir, fps, width, height)
       Opens the VideoWriter and records run metadata.

    2. write_frame(annotated_frame)
       Called once per processed frame during the pipeline loop.

    3. finalise(summary, frame_results, track_summaries, analytics_summary,
                config_dict, video_path)
       Flushes and closes the VideoWriter, writes JSON and CSV,
       updates the PipelineSummary with output paths, and returns it.

Usage:
    from src.exporter import Exporter
    from src.config_manager import ConfigManager

    cfg      = ConfigManager("config/config.yaml")
    exporter = Exporter(cfg)

    exporter.prepare("outputs/run_20240115_143022", fps=30.0, width=1920, height=1080)
    for annotated_frame in frames:
        exporter.write_frame(annotated_frame)
    summary = exporter.finalise(summary, frame_results, track_summaries,
                                analytics_summary, config_dict, video_path)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from exceptions import ExportError
from loggers import get_logger
from src.constants import (
    DEFAULT_CSV_FILENAME,
    DEFAULT_JSON_FILENAME,
    DEFAULT_VIDEO_FILENAME,
)
from src.data_models import FrameResult, PipelineSummary, TrackSummary

if TYPE_CHECKING:
    from src.config_manager import ConfigManager

log = get_logger(__name__)


class Exporter:
    """Writes pipeline outputs to disk: video, JSON analytics, CSV tracks.

    Attributes:
        save_video: Whether to write the annotated output video.
        save_json:  Whether to write the JSON analytics log.
        save_csv:   Whether to write the flat CSV track file.
        output_dir: Path to the current run's output directory.

    Args:
        config: Loaded ConfigManager instance.
    """

    def __init__(self, config: ConfigManager) -> None:
        self.save_video: bool = config.export.save_video
        self.save_json: bool = config.export.save_json
        self.save_csv: bool = config.export.save_csv
        self._codec: str = config.video.output_codec

        # State set by prepare()
        self.output_dir: str = ""
        self._video_writer: cv2.VideoWriter | None = None
        self._video_path: str = ""
        self._frames_written: int = 0

        log.info(
            "Exporter ready — video=%s, json=%s, csv=%s, codec=%s.",
            self.save_video,
            self.save_json,
            self.save_csv,
            self._codec,
        )

    # ------------------------------------------------------------------
    # Phase 1: prepare
    # ------------------------------------------------------------------

    def prepare(
        self,
        output_dir: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Initialise the exporter for a new run.

        Creates the output directory (if needed) and opens the VideoWriter.
        Must be called before the first write_frame() call.

        Args:
            output_dir: Path to the run's output directory.
            fps:        Frame rate for the output video.
            width:      Frame width in pixels.
            height:     Frame height in pixels.

        Raises:
            ExportError: If the VideoWriter cannot be initialised.
        """
        self.output_dir = str(output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._frames_written = 0

        if self.save_video:
            self._video_path = str(Path(self.output_dir) / DEFAULT_VIDEO_FILENAME)
            self._video_writer = self._open_video_writer(self._video_path, fps, width, height)

        log.info(
            "Exporter prepared — output_dir='%s', %dx%d @ %.1f FPS.",
            self.output_dir,
            width,
            height,
            fps,
        )

    # ------------------------------------------------------------------
    # Phase 2: write_frame
    # ------------------------------------------------------------------

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single annotated frame to the output video.

        No-op when save_video=False or prepare() has not been called.

        Args:
            frame: Annotated BGR frame as a numpy array.
        """
        if not self.save_video or self._video_writer is None:
            return
        if frame is None or frame.size == 0:
            return

        try:
            self._video_writer.write(frame)
            self._frames_written += 1
        except Exception as exc:
            raise ExportError(
                output_path=self._video_path,
                reason=f"VideoWriter.write() failed: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Phase 3: finalise
    # ------------------------------------------------------------------

    def finalise(
        self,
        summary: PipelineSummary,
        frame_results: list[FrameResult],
        track_summaries: list[TrackSummary],
        analytics_summary: dict[str, Any],
        config_dict: dict[str, Any],
        video_path: str,
    ) -> PipelineSummary:
        """Flush all outputs, close the VideoWriter, and return updated summary.

        Args:
            summary:           PipelineSummary from the pipeline.
            frame_results:     Per-frame FrameResult list from Analytics.
            track_summaries:   Per-track TrackSummary list from Tracker.
            analytics_summary: Aggregate analytics dict from Analytics.
            config_dict:       Serialised config snapshot from ConfigManager.
            video_path:        Path to the original input video.

        Returns:
            Updated PipelineSummary with output_video_path, output_json_path,
            and output_csv_path populated.
        """
        # ── Close VideoWriter ────────────────────────────────────────────
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            log.info(
                "Video saved: '%s' (%d frames).",
                self._video_path,
                self._frames_written,
            )
            summary.output_video_path = self._video_path

        # ── JSON export ──────────────────────────────────────────────────
        if self.save_json:
            json_path = str(Path(self.output_dir) / DEFAULT_JSON_FILENAME)
            self._save_json(
                json_path,
                summary,
                frame_results,
                track_summaries,
                analytics_summary,
                config_dict,
                video_path,
            )
            summary.output_json_path = json_path

        # ── CSV export ───────────────────────────────────────────────────
        if self.save_csv:
            csv_path = str(Path(self.output_dir) / DEFAULT_CSV_FILENAME)
            self._save_csv(csv_path, frame_results)
            summary.output_csv_path = csv_path

        log.info("Exporter finalised — outputs in '%s'.", self.output_dir)
        return summary

    # ------------------------------------------------------------------
    # Internal: video writer
    # ------------------------------------------------------------------

    def _open_video_writer(
        self,
        path: str,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        """Create and return an opened cv2.VideoWriter.

        Args:
            path:   Output file path.
            fps:    Output frame rate.
            width:  Frame width.
            height: Frame height.

        Returns:
            Opened VideoWriter instance.

        Raises:
            ExportError: If the writer cannot be opened.
        """
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        if not writer.isOpened():
            raise ExportError(
                output_path=path,
                reason=(
                    f"cv2.VideoWriter could not open '{path}' "
                    f"with codec '{self._codec}'. "
                    "Try codec 'XVID' for .avi output."
                ),
            )
        return writer

    # ------------------------------------------------------------------
    # Internal: JSON
    # ------------------------------------------------------------------

    def _save_json(
        self,
        path: str,
        summary: PipelineSummary,
        frame_results: list[FrameResult],
        track_summaries: list[TrackSummary],
        analytics_summary: dict[str, Any],
        config_dict: dict[str, Any],
        video_path: str,
    ) -> None:
        """Serialise all analytics data to a structured JSON file.

        JSON structure:
            metadata:  Run provenance (date, video, config snapshot).
            summary:   Aggregate statistics from PipelineSummary.
            analytics: Detailed analytics dict from Analytics.get_summary().
            tracks:    Per-track lifetime summaries from Tracker.
            frames:    Per-frame data (from analytics_summary["frames"]).

        Args:
            path:              Output file path.
            summary:           PipelineSummary instance.
            frame_results:     Per-frame FrameResult list.
            track_summaries:   Per-track TrackSummary list.
            analytics_summary: Aggregate analytics dict.
            config_dict:       Serialised config snapshot.
            video_path:        Input video path for provenance.
        """
        doc = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_video": video_path,
                "config": config_dict,
            },
            "summary": {
                "total_frames": summary.total_frames,
                "total_processing_time": round(summary.total_processing_time, 3),
                "avg_fps": round(summary.avg_fps, 2),
                "avg_inference_time_ms": round(summary.avg_inference_time_ms, 2),
                "unique_visitors": summary.unique_visitors,
                "peak_concurrent_tracks": summary.peak_concurrent_tracks,
                "avg_dwell_time": round(summary.avg_dwell_time, 3),
            },
            "analytics": analytics_summary,
            "tracks": [
                {
                    "track_id": ts.track_id,
                    "first_seen_frame": ts.first_seen_frame,
                    "last_seen_frame": ts.last_seen_frame,
                    "first_seen_time": round(ts.first_seen_time, 4),
                    "last_seen_time": round(ts.last_seen_time, 4),
                    "dwell_time": round(ts.dwell_time, 4),
                    "total_appearances": ts.total_appearances,
                    "trajectory": [
                        {
                            "x1": round(b.x1, 1),
                            "y1": round(b.y1, 1),
                            "x2": round(b.x2, 1),
                            "y2": round(b.y2, 1),
                        }
                        for b in ts.trajectory
                    ],
                }
                for ts in track_summaries
            ],
        }

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2, ensure_ascii=False)
            log.info("JSON analytics saved: '%s'.", path)
        except OSError as exc:
            raise ExportError(
                output_path=path,
                reason=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Internal: CSV
    # ------------------------------------------------------------------

    def _save_csv(
        self,
        path: str,
        frame_results: list[FrameResult],
    ) -> None:
        """Write per-detection data to a flat CSV file.

        Columns:
            frame_id, track_id, x1, y1, x2, y2, confidence, timestamp, fps

        One row per active track per frame.  Frames with no active tracks
        still produce a header row but no data rows for that frame.

        Args:
            path:          Output file path.
            frame_results: Per-frame FrameResult list from Analytics.
        """
        fieldnames = [
            "frame_id",
            "timestamp",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "fps",
        ]

        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()

                for fr in frame_results:
                    for track in fr.tracks:
                        writer.writerow(
                            {
                                "frame_id": fr.frame_id,
                                "timestamp": round(fr.timestamp, 4),
                                "track_id": track.track_id,
                                "x1": round(track.bbox.x1, 1),
                                "y1": round(track.bbox.y1, 1),
                                "x2": round(track.bbox.x2, 1),
                                "y2": round(track.bbox.y2, 1),
                                "confidence": round(track.confidence, 4),
                                "fps": round(fr.fps, 2),
                            }
                        )

            log.info("CSV tracks saved: '%s'.", path)
        except OSError as exc:
            raise ExportError(
                output_path=path,
                reason=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Exporter("
            f"video={self.save_video}, "
            f"json={self.save_json}, "
            f"csv={self.save_csv}, "
            f"codec='{self._codec}')"
        )
