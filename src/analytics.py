"""
Analytics Module for VisionTrack.

Accumulates all per-frame and per-track metrics across a video run and
exposes them to downstream consumers (Visualizer, Exporter, Pipeline).

Responsibilities:
  - Unique visitor counter (set of all track IDs ever seen)
  - Exponential moving average (EMA) FPS for smooth HUD display
  - Per-frame data storage (FrameResult list)
  - Inference time tracking (running average)
  - Peak concurrent track count

Analytics is intentionally separated from Tracker: the Tracker owns
identity persistence and trajectory; Analytics owns aggregate statistics
and system performance metrics.

Usage:
    from src.analytics import Analytics
    from src.config_manager import ConfigManager

    cfg       = ConfigManager("config/config.yaml")
    analytics = Analytics(cfg)

    # Called once per processed frame by VideoPipeline
    result = analytics.update(
        frame_id=0,
        timestamp=0.0,
        tracks=tracks,
        detection_count=len(detections),
        inference_time_ms=33.5,
    )

    # Called at end of run by Exporter
    summary = analytics.get_summary()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loggers import get_logger
from src.constants import EMA_ALPHA
from src.data_models import FrameResult, Track

if TYPE_CHECKING:
    from src.config_manager import ConfigManager

log = get_logger(__name__)


class Analytics:
    """Accumulates per-frame and aggregate analytics for a pipeline run.

    Attributes:
        ema_alpha:              Smoothing factor for EMA FPS [0 < α ≤ 1].
        unique_visitor_ids:     Set of all track IDs observed so far.
        frame_results:          Ordered list of FrameResult objects.
        total_detection_count:  Cumulative number of raw detections.
        peak_concurrent_tracks: Maximum simultaneous tracks in any frame.

    Args:
        config: Loaded ConfigManager instance.
    """

    def __init__(self, config: "ConfigManager") -> None:
        self.ema_alpha: float = config.analytics.ema_alpha

        # Unique visitor tracking
        self.unique_visitor_ids: set[int] = set()

        # Per-frame storage
        self.frame_results: list[FrameResult] = []

        # Aggregate counters
        self.total_detection_count: int = 0
        self.peak_concurrent_tracks: int = 0

        # EMA FPS state
        self._ema_fps: float = 0.0
        self._last_frame_time: float | None = None

        # Inference time tracking
        self._total_inference_ms: float = 0.0
        self._inference_count: int = 0

        log.info("Analytics initialised — EMA alpha=%.2f.", self.ema_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_id: int,
        timestamp: float,
        tracks: list[Track],
        detection_count: int,
        inference_time_ms: float,
    ) -> FrameResult:
        """Process a single frame's tracking results and update all metrics.

        Called once per processed frame by :class:`~src.pipeline.VideoPipeline`.

        Args:
            frame_id:         Sequential frame index (0-based).
            timestamp:        Elapsed time in seconds from video start.
            tracks:           Active tracks returned by the Tracker.
            detection_count:  Number of raw detections before tracking.
            inference_time_ms: Combined detection + tracking time (ms).

        Returns:
            A :class:`FrameResult` snapshot for this frame, including the
            current EMA FPS value.
        """
        # ── EMA FPS ──────────────────────────────────────────────────────
        fps = self._compute_ema_fps()

        # ── Unique visitor set ────────────────────────────────────────────
        for track in tracks:
            self.unique_visitor_ids.add(track.track_id)

        # ── Aggregate counters ────────────────────────────────────────────
        self.total_detection_count += detection_count

        active_count = len(tracks)
        if active_count > self.peak_concurrent_tracks:
            self.peak_concurrent_tracks = active_count

        # ── Inference time running average ────────────────────────────────
        self._total_inference_ms += inference_time_ms
        self._inference_count += 1

        # ── Build FrameResult ─────────────────────────────────────────────
        result = FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            tracks=list(tracks),          # defensive copy
            detection_count=detection_count,
            inference_time_ms=inference_time_ms,
            fps=fps,
        )
        self.frame_results.append(result)

        log.debug(
            "frame %d | active=%d | unique=%d | FPS=%.1f | inf=%.1fms",
            frame_id,
            active_count,
            len(self.unique_visitor_ids),
            fps,
            inference_time_ms,
        )
        return result

    @property
    def unique_visitor_count(self) -> int:
        """Total number of unique persons observed across all frames."""
        return len(self.unique_visitor_ids)

    @property
    def avg_inference_ms(self) -> float:
        """Average inference time per frame in milliseconds."""
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_ms / self._inference_count

    @property
    def current_fps(self) -> float:
        """Most recent EMA FPS value."""
        return self._ema_fps

    @property
    def total_frames(self) -> int:
        """Number of frames that have been processed."""
        return len(self.frame_results)

    def get_current_stats(self) -> dict[str, Any]:
        """Return a lightweight dict of live metrics for HUD display.

        Suitable for logging mid-run status without serialising the full
        frame_results list.

        Returns:
            Dict with keys: frame_id, unique_visitors, active_tracks,
            current_fps, avg_inference_ms.
        """
        last = self.frame_results[-1] if self.frame_results else None
        return {
            "frame_id":        last.frame_id if last else 0,
            "unique_visitors": self.unique_visitor_count,
            "active_tracks":   last.active_track_count if last else 0,
            "current_fps":     round(self._ema_fps, 1),
            "avg_inference_ms": round(self.avg_inference_ms, 1),
        }

    def get_summary(self) -> dict[str, Any]:
        """Return a complete analytics summary for export metadata.

        Called once at end of a pipeline run by the Exporter.

        Returns:
            Dict with aggregate statistics: total_frames, unique_visitors,
            peak_concurrent_tracks, total_detections, avg_fps,
            avg_inference_ms, and a per-frame data list.
        """
        n = self.total_frames
        avg_fps = self._ema_fps  # last EMA value ≈ steady-state FPS

        # True average FPS from frame_results timing is computed by the
        # pipeline's _build_summary; here we expose the EMA value which is
        # already available without recomputing.

        return {
            "total_frames":           n,
            "unique_visitors":        self.unique_visitor_count,
            "peak_concurrent_tracks": self.peak_concurrent_tracks,
            "total_detections":       self.total_detection_count,
            "avg_fps":                round(avg_fps, 2),
            "avg_inference_ms":       round(self.avg_inference_ms, 2),
            "frames": [
                {
                    "frame_id":        r.frame_id,
                    "timestamp":       round(r.timestamp, 4),
                    "active_tracks":   r.active_track_count,
                    "track_ids":       r.active_track_ids,
                    "detection_count": r.detection_count,
                    "inference_ms":    round(r.inference_time_ms, 2),
                    "fps":             round(r.fps, 2),
                }
                for r in self.frame_results
            ],
        }

    def reset(self) -> None:
        """Clear all accumulated state — useful between independent video runs."""
        self.unique_visitor_ids.clear()
        self.frame_results.clear()
        self.total_detection_count = 0
        self.peak_concurrent_tracks = 0
        self._ema_fps = 0.0
        self._last_frame_time = None
        self._total_inference_ms = 0.0
        self._inference_count = 0
        log.debug("Analytics state reset.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_ema_fps(self) -> float:
        """Compute EMA FPS from wall-clock time between update() calls.

        On the first call there is no previous timestamp, so we return 0.0
        and record the start time.  From the second call onward we compute
        the instantaneous FPS (1 / Δt) and blend it with the running EMA.

        Returns:
            Updated EMA FPS value (0.0 on the first frame).
        """
        now = time.perf_counter()

        if self._last_frame_time is None:
            # First frame — no interval to measure yet
            self._last_frame_time = now
            return 0.0

        delta = now - self._last_frame_time
        self._last_frame_time = now

        if delta <= 0:
            return self._ema_fps

        instant_fps = 1.0 / delta

        # EMA: new_ema = α * instant + (1 - α) * old_ema
        if self._ema_fps == 0.0:
            # Seed the EMA with the first real measurement
            self._ema_fps = instant_fps
        else:
            self._ema_fps = (
                self.ema_alpha * instant_fps
                + (1.0 - self.ema_alpha) * self._ema_fps
            )

        return self._ema_fps

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Analytics("
            f"frames={self.total_frames}, "
            f"unique={self.unique_visitor_count}, "
            f"fps={self._ema_fps:.1f})"
        )
