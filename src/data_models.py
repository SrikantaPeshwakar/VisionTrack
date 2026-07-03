"""
Data models for VisionTrack.

Defines typed dataclasses used throughout the pipeline to avoid passing
raw tuples and ensure consistent data shapes across all modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge (pixels).
        y1: Top edge (pixels).
        x2: Right edge (pixels).
        y2: Bottom edge (pixels).
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """(cx, cy) center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_list(self) -> list[float]:
        """Return bounding box as [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_int_list(self) -> list[int]:
        """Return bounding box as [x1, y1, x2, y2] with integer coordinates."""
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    @classmethod
    def from_list(cls, coords: list[float]) -> BoundingBox:
        """Create a BoundingBox from a [x1, y1, x2, y2] list."""
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


@dataclass
class Detection:
    """A single object detection produced by the detector.

    Attributes:
        bbox: Bounding box in pixel coordinates.
        confidence: Detection confidence score in [0, 1].
        class_id: COCO class index (0 = person).
    """

    bbox: BoundingBox
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.class_id < 0:
            raise ValueError(f"class_id must be >= 0, got {self.class_id}")

    @classmethod
    def from_raw(
        cls, x1: float, y1: float, x2: float, y2: float, conf: float, class_id: int
    ) -> Detection:
        """Convenience constructor from raw coordinate values."""
        return cls(
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=conf,
            class_id=class_id,
        )


@dataclass
class Track:
    """A tracked object with a persistent identity across frames.

    Attributes:
        track_id: Unique integer ID assigned by BoT-SORT, persists across frames.
        bbox: Bounding box in the current frame.
        confidence: Detection confidence for the current frame.
        class_id: COCO class index (0 = person).
        frame_id: Frame number where this track was observed.
        timestamp: Elapsed time in seconds from video start.
    """

    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: int
    frame_id: int
    timestamp: float

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError(f"track_id must be >= 0, got {self.track_id}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {self.frame_id}")
        if self.timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0.0, got {self.timestamp}")

    @classmethod
    def from_raw(
        cls,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        conf: float,
        class_id: int,
        frame_id: int,
        timestamp: float,
    ) -> Track:
        """Convenience constructor from raw coordinate values."""
        return cls(
            track_id=track_id,
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=conf,
            class_id=class_id,
            frame_id=frame_id,
            timestamp=timestamp,
        )


@dataclass
class FrameResult:
    """All tracking data produced for a single video frame.

    Attributes:
        frame_id: Sequential frame index (0-based).
        timestamp: Elapsed time in seconds from video start.
        tracks: List of active tracks in this frame.
        detection_count: Number of raw detections before tracking.
        inference_time_ms: Time taken for detection + tracking (milliseconds).
        fps: Instantaneous frames-per-second at this frame.
    """

    frame_id: int
    timestamp: float
    tracks: list[Track] = field(default_factory=list)
    detection_count: int = 0
    inference_time_ms: float = 0.0
    fps: float = 0.0

    @property
    def active_track_ids(self) -> list[int]:
        """List of track IDs active in this frame."""
        return [t.track_id for t in self.tracks]

    @property
    def active_track_count(self) -> int:
        """Number of active tracks in this frame."""
        return len(self.tracks)


@dataclass
class TrackSummary:
    """Lifetime summary for a single tracked person.

    Attributes:
        track_id: Unique integer ID assigned by BoT-SORT.
        first_seen_frame: Frame index when this track first appeared.
        last_seen_frame: Frame index when this track was last observed.
        first_seen_time: Timestamp (seconds) of first appearance.
        last_seen_time: Timestamp (seconds) of last observation.
        total_appearances: Number of frames this track was active.
        trajectory: Ordered list of bounding boxes across all frames.
    """

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: float
    last_seen_time: float
    total_appearances: int
    trajectory: list[BoundingBox] = field(default_factory=list)

    @property
    def dwell_time(self) -> float:
        """Total time (seconds) this person was visible in the video."""
        return self.last_seen_time - self.first_seen_time


@dataclass
class PipelineSummary:
    """Aggregated statistics produced at the end of a pipeline run.

    Attributes:
        total_frames: Total number of frames processed.
        total_processing_time: Wall-clock time for the entire run (seconds).
        avg_fps: Average frames per second over the full run.
        avg_inference_time_ms: Average detection + tracking time per frame (ms).
        unique_visitors: Total unique person IDs seen across the video.
        peak_concurrent_tracks: Maximum number of simultaneous tracks in any frame.
        avg_dwell_time: Average dwell time across all tracks (seconds).
        output_video_path: Path to the saved annotated video (if exported).
        output_json_path: Path to the saved JSON analytics log (if exported).
        output_csv_path: Path to the saved CSV tracks file (if exported).
    """

    total_frames: int
    total_processing_time: float
    avg_fps: float
    avg_inference_time_ms: float
    unique_visitors: int
    peak_concurrent_tracks: int
    avg_dwell_time: float
    output_video_path: str | None = None
    output_json_path: str | None = None
    output_csv_path: str | None = None

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "  VisionTrack Pipeline Summary",
            "=" * 50,
            f"  Frames Processed   : {self.total_frames}",
            f"  Processing Time    : {self.total_processing_time:.2f}s",
            f"  Average FPS        : {self.avg_fps:.1f}",
            f"  Avg Inference Time : {self.avg_inference_time_ms:.1f}ms",
            f"  Unique Visitors    : {self.unique_visitors}",
            f"  Peak Concurrent    : {self.peak_concurrent_tracks}",
            f"  Avg Dwell Time     : {self.avg_dwell_time:.1f}s",
            "-" * 50,
        ]
        if self.output_video_path:
            lines.append(f"  Video Output       : {self.output_video_path}")
        if self.output_json_path:
            lines.append(f"  JSON Output        : {self.output_json_path}")
        if self.output_csv_path:
            lines.append(f"  CSV Output         : {self.output_csv_path}")
        lines.append("=" * 50)
        return "\n".join(lines)
