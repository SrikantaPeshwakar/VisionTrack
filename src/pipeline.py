"""
Video Pipeline Orchestration for VisionTrack.

VideoPipeline wires every component together in the correct order:

    VideoCapture → Detector → Tracker → Analytics → Visualizer → Exporter

All dependencies are injected via the constructor so each component is
independently testable and swappable without touching pipeline logic.

Usage:
    from src.pipeline import VideoPipeline
    from src.config_manager import ConfigManager
    from src.detector import Detector
    from src.tracker import Tracker
    from src.analytics import Analytics
    from src.visualizer import Visualizer
    from src.exporter import Exporter
    from src.device_manager import DeviceManager

    cfg     = ConfigManager("config/config.yaml")
    device  = DeviceManager(cfg.device.preferred).device
    det     = Detector(cfg, device=device)
    tracker = Tracker(cfg, model=det._model)
    ana     = Analytics(cfg)
    vis     = Visualizer(cfg)
    exp     = Exporter(cfg)

    pipeline = VideoPipeline(cfg, det, tracker, ana, vis, exp)
    summary  = pipeline.run("metro.mp4")
    print(summary)
"""

from __future__ import annotations

import signal
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from tqdm import tqdm

from exceptions import VideoIOError
from loggers import get_logger
from src.constants import DEFAULT_VIDEO_FILENAME
from src.data_models import FrameResult, PipelineSummary

if TYPE_CHECKING:
    from src.analytics import Analytics
    from src.config_manager import ConfigManager
    from src.detector import Detector
    from src.exporter import Exporter
    from src.tracker import Tracker
    from src.visualizer import Visualizer

log = get_logger(__name__)


class VideoPipeline:
    """Main pipeline orchestrator for VisionTrack.

    Reads a video file frame-by-frame, runs detection and tracking on each
    frame, feeds results to analytics, annotates the frame, then hands off
    to the exporter.  A tqdm progress bar gives live feedback on throughput.

    All heavy components are injected so the pipeline itself stays thin and
    focused on orchestration only.

    Attributes:
        config:     Loaded ConfigManager instance.
        detector:   Configured Detector instance.
        tracker:    Configured Tracker instance.
        analytics:  Configured Analytics instance.
        visualizer: Configured Visualizer instance.
        exporter:   Configured Exporter instance.

    Args:
        config:     Loaded ConfigManager.
        detector:   Detector wrapping a YOLO model.
        tracker:    Tracker wrapping BoT-SORT.
        analytics:  Analytics accumulates metrics frame-by-frame.
        visualizer: Visualizer annotates each frame.
        exporter:   Exporter writes video / JSON / CSV outputs.
        logger:     Optional custom logger (defaults to module logger).
    """

    def __init__(
        self,
        config: "ConfigManager",
        detector: "Detector",
        tracker: "Tracker",
        analytics: "Analytics",
        visualizer: "Visualizer",
        exporter: "Exporter",
    ) -> None:
        self.config     = config
        self.detector   = detector
        self.tracker    = tracker
        self.analytics  = analytics
        self.visualizer = visualizer
        self.exporter   = exporter

        self._skip_frames: int = config.video.skip_frames
        self._interrupted: bool = False

        log.info(
            "VideoPipeline ready — model=%s, skip_frames=%d.",
            config.model.type,
            self._skip_frames,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> PipelineSummary:
        """Process a video file end-to-end.

        Opens the video, iterates over frames, runs the full pipeline on
        each non-skipped frame, and writes all outputs.  A tqdm progress
        bar shows frame index, live FPS, ETA, and unique visitor count.

        Handles SIGINT (Ctrl+C) gracefully: saves whatever has been
        processed so far, then returns a partial summary.

        Args:
            video_path: Path to the input video file.

        Returns:
            :class:`PipelineSummary` with aggregate statistics and output
            file paths.

        Raises:
            VideoIOError: If the video file cannot be opened.
        """
        video_path = str(video_path)
        self._interrupted = False

        # ── Open video ──────────────────────────────────────────────────
        cap = self._open_video(video_path)
        fps_in      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        log.info(
            "Video opened: %s | %dx%d | %.1f FPS | %d frames.",
            video_path,
            width,
            height,
            fps_in,
            total_frames,
        )

        # ── Prepare output directory ────────────────────────────────────
        run_name   = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.config.get_output_dir(run_name)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # ── Prepare exporter for frame-by-frame accumulation ────────────
        self.exporter.prepare(output_dir, fps_in, width, height)

        # ── Install SIGINT handler ───────────────────────────────────────
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # ── Main processing loop ─────────────────────────────────────────
        frame_results: list[FrameResult] = []
        t_pipeline_start = time.perf_counter()

        try:
            pbar = tqdm(
                total=total_frames if total_frames > 0 else None,
                desc="Processing",
                unit="frame",
                dynamic_ncols=True,
            )

            frame_idx   = 0   # absolute frame counter (includes skipped)
            processed   = 0   # frames actually sent through the pipeline

            while True:
                if self._interrupted:
                    log.warning("Pipeline interrupted by user — saving partial results.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # ── Skip logic ───────────────────────────────────────────
                if self._skip_frames > 0 and frame_idx % (self._skip_frames + 1) != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                timestamp = frame_idx / fps_in

                # ── Detection ────────────────────────────────────────────
                t0 = time.perf_counter()
                detections = self.detector.detect(frame)

                # ── Tracking ─────────────────────────────────────────────
                tracks = self.tracker.track(
                    frame,
                    detections,
                    frame_id=frame_idx,
                    timestamp=timestamp,
                )

                inference_ms = (time.perf_counter() - t0) * 1000

                # ── Analytics ────────────────────────────────────────────
                frame_result = self.analytics.update(
                    frame_id=frame_idx,
                    timestamp=timestamp,
                    tracks=tracks,
                    detection_count=len(detections),
                    inference_time_ms=inference_ms,
                )
                frame_results.append(frame_result)

                # ── Visualisation ─────────────────────────────────────────
                annotated = self.visualizer.annotate(
                    frame,
                    tracks,
                    unique_count=len(self.tracker.all_track_ids),
                    fps=frame_result.fps,
                    inference_ms=inference_ms,
                )

                # ── Export frame ──────────────────────────────────────────
                self.exporter.write_frame(annotated)

                processed += 1
                frame_idx += 1

                # ── Update progress bar ───────────────────────────────────
                pbar.set_postfix(
                    FPS=f"{frame_result.fps:.1f}",
                    inf=f"{inference_ms:.0f}ms",
                    unique=len(self.tracker.all_track_ids),
                )
                pbar.update(1)

            pbar.close()

        finally:
            cap.release()
            signal.signal(signal.SIGINT, original_handler)

        # ── Compute summary ──────────────────────────────────────────────
        total_time = time.perf_counter() - t_pipeline_start
        summary    = self._build_summary(total_time, output_dir, frame_results)

        # ── Finalise exports ─────────────────────────────────────────────
        summary = self.exporter.finalise(
            summary,
            frame_results,
            self.tracker.get_all_summaries(),
            self.analytics.get_summary(),
            self.config.to_dict(),
            video_path,
        )

        log.info(
            "Pipeline complete — %d frames | %.1f avg FPS | %d unique visitors.",
            processed,
            summary.avg_fps,
            summary.unique_visitors,
        )
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_video(self, video_path: str) -> cv2.VideoCapture:
        """Open a VideoCapture and validate the file is readable.

        Args:
            video_path: Path to the video file.

        Returns:
            Opened cv2.VideoCapture instance.

        Raises:
            VideoIOError: If the file does not exist or OpenCV cannot open it.
        """
        if not Path(video_path).is_file():
            raise VideoIOError(
                video_path=video_path,
                reason="file not found",
                details={"path": video_path},
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoIOError(
                video_path=video_path,
                reason="OpenCV could not open the file — unsupported format or codec",
                details={"path": video_path},
            )
        return cap

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Set the interrupt flag on SIGINT so the loop exits cleanly."""
        log.warning("SIGINT received — finishing current frame and saving outputs …")
        self._interrupted = True

    def _build_summary(
        self,
        total_time: float,
        output_dir: str,
        frame_results: list[FrameResult],
    ) -> PipelineSummary:
        """Build a PipelineSummary from accumulated frame results.

        Args:
            total_time:    Wall-clock seconds for the full run.
            output_dir:    Directory where outputs will be written.
            frame_results: All FrameResult objects produced during the run.

        Returns:
            PipelineSummary with aggregate statistics.
        """
        n = len(frame_results)
        avg_fps = (n / total_time) if total_time > 0 else 0.0
        avg_inf = (
            sum(r.inference_time_ms for r in frame_results) / n if n > 0 else 0.0
        )
        peak_concurrent = (
            max((r.active_track_count for r in frame_results), default=0)
        )

        # Dwell time from tracker summaries
        summaries = self.tracker.get_all_summaries()
        avg_dwell = (
            sum(s.dwell_time for s in summaries) / len(summaries)
            if summaries else 0.0
        )

        return PipelineSummary(
            total_frames=n,
            total_processing_time=total_time,
            avg_fps=avg_fps,
            avg_inference_time_ms=avg_inf,
            unique_visitors=len(self.tracker.all_track_ids),
            peak_concurrent_tracks=peak_concurrent,
            avg_dwell_time=avg_dwell,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VideoPipeline("
            f"model='{self.config.model.type}', "
            f"skip_frames={self._skip_frames})"
        )
