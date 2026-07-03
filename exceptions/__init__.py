"""
VisionTrack Exception Module

Provides a standardised exception hierarchy for consistent error handling
across the entire pipeline.

All application exceptions inherit from VisionTrackException so callers
can catch either a specific error or the entire family with one clause.

Usage:
    from exceptions import (
        VisionTrackException,
        ConfigurationError,
        ModelLoadError,
        VideoIOError,
        TrackingError,
        DeviceError,
        ExportError,
    )
"""

from typing import Any

__all__ = [
    "error_message_detail",
    "VisionTrackException",
    "ConfigurationError",
    "ModelLoadError",
    "VideoIOError",
    "DetectionError",
    "TrackingError",
    "DeviceError",
    "ExportError",
    "AnalyticsError",
]


# ------------------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------------------


def error_message_detail(error: Exception, error_detail: Any | None = None) -> str:
    """Build a detailed error message with filename, line number, and description.

    Args:
        error: The Exception object.
        error_detail: The ``sys`` module (or any object exposing ``exc_info``).
                      Pass ``sys`` from the call site to capture live traceback
                      context; omit (or pass ``None``) for a plain message.

    Returns:
        Formatted error string including source location when available.
    """
    if error_detail is not None and hasattr(error_detail, "exc_info"):
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown file"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown line"
        return f"Error in [{file_name}] at line [{line_number}] — {str(error)}"
    return f"Error occurred — {str(error)}"


# ------------------------------------------------------------------------------
# Base Exception
# ------------------------------------------------------------------------------


class VisionTrackException(Exception):  # noqa: N818
    """Base exception for all VisionTrack errors.

    Carries both a detailed internal message (safe for logs) and a concise
    user-facing message (safe to surface in CLI output).

    Attributes:
        message: Full technical description for logs and debugging.
        user_message: Short, human-readable message for CLI/UI output.
        details: Optional dict of extra context (paths, values, etc.).
    """

    def __init__(
        self,
        message: str,
        user_message: str = "An unexpected error occurred in VisionTrack.",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


# ------------------------------------------------------------------------------
# Specific Exceptions
# ------------------------------------------------------------------------------


class ConfigurationError(VisionTrackException):
    """Raised when config.yaml is missing, malformed, or contains invalid values.

    Example:
        raise ConfigurationError("confidence_threshold must be in [0, 1]")
    """

    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        message = f"Configuration error: {reason}"
        user_message = "Invalid configuration. Check config/config.yaml and try again."
        super().__init__(message, user_message=user_message, details=details)


class ModelLoadError(VisionTrackException):
    """Raised when a YOLO model fails to load or download.

    Example:
        raise ModelLoadError("yolov8x", "weights file corrupted")
    """

    def __init__(
        self,
        model_name: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Failed to load model '{model_name}': {reason}"
        user_message = (
            f"Could not load model '{model_name}'. "
            "Ensure the model name is valid and you have an internet connection "
            "for the first-run download."
        )
        super().__init__(message, user_message=user_message, details=details)


class VideoIOError(VisionTrackException):
    """Raised when a video file cannot be opened, read, or written.

    Example:
        raise VideoIOError("metro.mp4", "file not found")
    """

    def __init__(
        self,
        video_path: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Video I/O error for '{video_path}': {reason}"
        user_message = (
            f"Cannot access video file '{video_path}'. "
            "Check the path and ensure the file is a valid video format."
        )
        super().__init__(message, user_message=user_message, details=details)


class DetectionError(VisionTrackException):
    """Raised when the detection step fails for a frame.

    Example:
        raise DetectionError(42, "CUDA out of memory")
    """

    def __init__(
        self,
        frame_id: int,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Detection failed at frame {frame_id}: {reason}"
        user_message = (
            "Detection error encountered. "
            "Try reducing the model size or switching to CPU with --device cpu."
        )
        super().__init__(message, user_message=user_message, details=details)


class TrackingError(VisionTrackException):
    """Raised when the BoT-SORT tracker fails to initialise or update.

    Example:
        raise TrackingError("botsort.yaml not found")
    """

    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        message = f"Tracking error: {reason}"
        user_message = (
            "Tracker encountered an error. "
            "Verify config/botsort.yaml exists and parameters are valid."
        )
        super().__init__(message, user_message=user_message, details=details)


class DeviceError(VisionTrackException):
    """Raised when no suitable compute device is available.

    Example:
        raise DeviceError("cuda", "CUDA toolkit not installed")
    """

    def __init__(
        self,
        requested_device: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Device error for '{requested_device}': {reason}"
        user_message = (
            f"Requested device '{requested_device}' is unavailable. "
            "The pipeline will fall back to CPU automatically, "
            "or specify --device cpu explicitly."
        )
        super().__init__(message, user_message=user_message, details=details)


class ExportError(VisionTrackException):
    """Raised when writing output files (video, JSON, CSV) fails.

    Example:
        raise ExportError("outputs/run_xyz/result.mp4", "disk full")
    """

    def __init__(
        self,
        output_path: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Export failed for '{output_path}': {reason}"
        user_message = (
            f"Could not write output to '{output_path}'. "
            "Check disk space and directory permissions."
        )
        super().__init__(message, user_message=user_message, details=details)


class AnalyticsError(VisionTrackException):
    """Raised when analytics calculations encounter unexpected state.

    Example:
        raise AnalyticsError("track history is empty")
    """

    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        message = f"Analytics error: {reason}"
        user_message = "Analytics computation failed. Check pipeline output for details."
        super().__init__(message, user_message=user_message, details=details)
