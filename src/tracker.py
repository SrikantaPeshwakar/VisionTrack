"""
BoT-SORT Tracker Module for VisionTrack.

Wraps Ultralytics' built-in BoT-SORT tracker and is responsible for:
  1. Accepting a raw BGR frame + the detector's List[Detection] for that frame.
  2. Running model.track() with persist=True to maintain ID continuity.
  3. Returning a List[Track] with stable IDs, bounding boxes and timestamps.
  4. Maintaining per-track history (trajectory, lifecycle timestamps) for
     downstream analytics and export.

Design note:
    We pass detections back through the YOLO model's .track() method rather
    than implementing our own Kalman filter. This is intentional — the
    assignment requires BoT-SORT, not a re-implementation of it.

Usage:
    from src.tracker import Tracker
    from src.config_manager import ConfigManager

    cfg = ConfigManager("config/config.yaml")
    tracker = Tracker(cfg, model=detector._model)

    tracks = tracker.track(frame, detections, frame_id=0, timestamp=0.0)
    summary = tracker.get_track_summary(track_id=1)
    history = tracker.track_history   # Dict[int, List[Track]]
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from exceptions import TrackingError
from loggers import get_logger
from src.constants import PERSON_CLASS_ID
from src.data_models import BoundingBox, Track, TrackSummary

if TYPE_CHECKING:
    from src.config_manager import ConfigManager
    from src.data_models import Detection

log = get_logger(__name__)


class Tracker:
    """BoT-SORT multi-object tracker wrapping Ultralytics' built-in tracker.

    Accepts detections produced by :class:`~src.detector.Detector` on each
    frame and associates them with persistent track IDs using BoT-SORT's
    Kalman filter + ReID appearance embeddings.

    Track history is maintained internally so the Analytics and Exporter
    modules can access full trajectory data without re-processing.

    Attributes:
        tracker_config:   Path to botsort.yaml used by Ultralytics.
        persist:          Whether to persist track state across frames.
        track_history:    Dict mapping track_id → ordered list of Track objects.
        first_seen:       Dict mapping track_id → (frame_id, timestamp).
        last_seen:        Dict mapping track_id → (frame_id, timestamp).
        all_track_ids:    Set of every track_id ever observed.

    Args:
        config: Loaded ConfigManager instance.
        model:  The Ultralytics YOLO model instance from Detector._model.
                The tracker is attached to this model so that
                ``model.track(..., persist=True)`` maintains state.
    """

    def __init__(self, config: ConfigManager, model: Any) -> None:
        self.tracker_config: str = config.tracker.config_file
        self.persist: bool = config.tracker.persist

        # Track state
        self.track_history: dict[int, list[Track]] = defaultdict(list)
        self.first_seen: dict[int, tuple[int, float]] = {}  # id → (frame_id, ts)
        self.last_seen: dict[int, tuple[int, float]] = {}  # id → (frame_id, ts)
        self.all_track_ids: set[int] = set()

        self._model = model
        log.info(
            "Tracker initialised — config='%s', persist=%s.",
            self.tracker_config,
            self.persist,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_id: int,
        timestamp: float,
    ) -> list[Track]:
        """Run BoT-SORT tracking on a single frame.

        Passes the raw frame to ``model.track()`` so that BoT-SORT can use
        both the bounding boxes from the detector and the appearance features
        extracted directly from the image pixels.

        Args:
            frame:      BGR image (H, W, 3) numpy array — same frame the
                        detector just processed.
            detections: Person detections from Detector.detect() for this frame.
            frame_id:   Sequential frame index (0-based).
            timestamp:  Elapsed time in seconds from the video start.

        Returns:
            List of :class:`Track` instances — one per active track in this
            frame.  Empty list when no persons are being tracked.

        Raises:
            TrackingError: If the Ultralytics tracker raises an unexpected
                           exception.
        """
        if not detections:
            log.debug("frame %d: no detections — skipping tracker.", frame_id)
            return []

        try:
            results = self._model.track(
                frame,
                persist=self.persist,
                tracker=self.tracker_config,
                verbose=False,
            )
        except Exception as exc:
            raise TrackingError(
                reason=str(exc),
                details={
                    "frame_id": frame_id,
                    "tracker_config": self.tracker_config,
                },
            ) from exc

        tracks = self._parse_results(results, frame_id, timestamp)
        self._update_history(tracks)

        log.debug(
            "frame %d | active=%d | total_unique=%d",
            frame_id,
            len(tracks),
            len(self.all_track_ids),
        )
        return tracks

    def get_track_summary(self, track_id: int) -> TrackSummary | None:
        """Return a lifetime summary for a single track.

        Args:
            track_id: The BoT-SORT assigned track identifier.

        Returns:
            :class:`TrackSummary` if the track_id has been observed, else None.
        """
        if track_id not in self.all_track_ids:
            return None

        history = self.track_history[track_id]
        first_frame, first_ts = self.first_seen[track_id]
        last_frame, last_ts = self.last_seen[track_id]

        return TrackSummary(
            track_id=track_id,
            first_seen_frame=first_frame,
            last_seen_frame=last_frame,
            first_seen_time=first_ts,
            last_seen_time=last_ts,
            total_appearances=len(history),
            trajectory=[t.bbox for t in history],
        )

    def get_all_summaries(self) -> list[TrackSummary]:
        """Return lifetime summaries for every observed track.

        Returns:
            List of TrackSummary objects sorted ascending by track_id.
        """
        return [
            self.get_track_summary(tid)
            for tid in sorted(self.all_track_ids)
            if self.get_track_summary(tid) is not None
        ]

    def reset(self) -> None:
        """Clear all track state — useful between independent video runs."""
        self.track_history.clear()
        self.first_seen.clear()
        self.last_seen.clear()
        self.all_track_ids.clear()
        log.debug("Tracker state reset.")

    # ------------------------------------------------------------------
    # Internal: result parsing
    # ------------------------------------------------------------------

    def _parse_results(
        self,
        results: list,
        frame_id: int,
        timestamp: float,
    ) -> list[Track]:
        """Convert raw Ultralytics tracking results into Track objects.

        Ultralytics .track() returns a list of Results objects.  Each result
        has a ``.boxes`` attribute with an optional ``.id`` tensor containing
        the BoT-SORT track IDs.

        Args:
            results:   Raw output from ``model.track(frame, ...)``.
            frame_id:  Frame index to embed in each Track.
            timestamp: Video timestamp to embed in each Track.

        Returns:
            List of Track instances for this frame.
        """
        tracks: list[Track] = []

        if not results:
            return tracks

        result = results[0]

        # No boxes or no track IDs means the tracker hasn't associated yet
        if result.boxes is None or result.boxes.id is None:
            return tracks

        # boxes.data shape: (N, 6) — x1 y1 x2 y2 conf class_id
        # boxes.id   shape: (N,)   — integer track IDs
        boxes_data = result.boxes.data.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for i, row in enumerate(boxes_data):
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            # Clamp to [0, 1] — BoT-SORT with fuse_score=True can produce
            # values slightly above 1.0 (conf × IoU fusion).
            conf = min(1.0, max(0.0, float(row[4])))
            class_id = int(row[5])
            track_id = int(track_ids[i])

            # Guard: only person class
            if class_id != PERSON_CLASS_ID:
                continue

            # Guard: valid bounding box
            if x2 <= x1 or y2 <= y1:
                log.debug(
                    "Skipping degenerate track bbox [%.1f, %.1f, %.1f, %.1f]",
                    x1,
                    y1,
                    x2,
                    y2,
                )
                continue

            # Guard: valid track ID
            if track_id < 0:
                continue

            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    confidence=conf,
                    class_id=class_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
            )

        return tracks

    # ------------------------------------------------------------------
    # Internal: history management
    # ------------------------------------------------------------------

    def _update_history(self, tracks: list[Track]) -> None:
        """Update internal track state from the current frame's tracks.

        For each track:
        - Appends to track_history[track_id].
        - Records first_seen on first appearance.
        - Updates last_seen on every appearance.
        - Adds to all_track_ids set.

        Args:
            tracks: Active tracks returned by _parse_results for this frame.
        """
        for track in tracks:
            tid = track.track_id

            # Register new track
            if tid not in self.all_track_ids:
                self.all_track_ids.add(tid)
                self.first_seen[tid] = (track.frame_id, track.timestamp)
                log.debug(
                    "New track ID=%d appeared at frame %d (t=%.2fs).",
                    tid,
                    track.frame_id,
                    track.timestamp,
                )

            # Always update last_seen and append to history
            self.last_seen[tid] = (track.frame_id, track.timestamp)
            self.track_history[tid].append(track)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tracker("
            f"config='{self.tracker_config}', "
            f"persist={self.persist}, "
            f"unique_tracks={len(self.all_track_ids)})"
        )
