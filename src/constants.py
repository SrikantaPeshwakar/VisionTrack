"""
Project-wide constants for VisionTrack.

Centralises all magic numbers, string literals, and configuration
defaults so they never appear scattered across the codebase.
"""

import cv2

# ==============================================================================
# COCO Class IDs
# ==============================================================================

# Person class index in the COCO dataset (used by all YOLO models)
PERSON_CLASS_ID: int = 0

# Human-readable label for the person class
PERSON_CLASS_LABEL: str = "person"

# ==============================================================================
# Supported Models
# ==============================================================================

SUPPORTED_MODELS: list[str] = [
    # YOLOv8 family — speed (n) → accuracy (x)
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    # YOLOv9 family
    "yolov9c",
    "yolov9e",
    # YOLOv10 family
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10l",
    "yolov10x",
    # YOLOv11 family
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
]

# Default model used when none is specified
DEFAULT_MODEL: str = "yolov8n"

# ==============================================================================
# Device Options
# ==============================================================================

SUPPORTED_DEVICES: list[str] = ["cuda", "mps", "cpu"]
DEFAULT_DEVICE: str = "cpu"

# ==============================================================================
# Video / Export
# ==============================================================================

# Default codec for output video files (FourCC)
DEFAULT_CODEC: str = "mp4v"

# Default output file names inside a timestamped run directory
DEFAULT_VIDEO_FILENAME: str = "result.mp4"
DEFAULT_JSON_FILENAME: str = "analytics.json"
DEFAULT_CSV_FILENAME: str = "tracks.csv"
DEFAULT_EVAL_REPORT_FILENAME: str = "evaluation_report.json"

# Default output root directory
DEFAULT_OUTPUT_DIR: str = "outputs"

# Default models cache directory
DEFAULT_MODELS_DIR: str = "models"

# ==============================================================================
# Confidence / IoU Defaults
# ==============================================================================

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.45

# ==============================================================================
# Tracking Defaults
# ==============================================================================

DEFAULT_TRACKER_CONFIG: str = "config/botsort.yaml"

# ==============================================================================
# Visualisation
# ==============================================================================

# OpenCV font used for all on-frame text
FONT: int = cv2.FONT_HERSHEY_SIMPLEX

# Font scale and thickness for track ID labels
FONT_SCALE_LABEL: float = 0.6
FONT_THICKNESS_LABEL: int = 2

# Font scale and thickness for HUD overlays (FPS, unique count)
FONT_SCALE_HUD: float = 0.7
FONT_THICKNESS_HUD: int = 2

# Bounding box border thickness (pixels)
BBOX_THICKNESS: int = 2

# Padding (pixels) around text inside label backgrounds
LABEL_PADDING: int = 4

# Opacity of the semi-transparent HUD background panels (0.0–1.0)
HUD_ALPHA: float = 0.6

# Colour palette for track ID bounding boxes.
# Each entry is (B, G, R) — OpenCV uses BGR.
# 20 visually distinct colours that work well on dark and light backgrounds.
COLOR_PALETTE: list[tuple[int, int, int]] = [
    (255, 56, 56),  # 0  red
    (56, 255, 56),  # 1  green
    (56, 56, 255),  # 2  blue
    (255, 157, 56),  # 3  orange
    (255, 56, 157),  # 4  pink
    (56, 255, 255),  # 5  cyan
    (255, 255, 56),  # 6  yellow
    (157, 56, 255),  # 7  purple
    (56, 157, 255),  # 8  sky blue
    (157, 255, 56),  # 9  lime
    (255, 100, 100),  # 10 light red
    (100, 255, 100),  # 11 light green
    (100, 100, 255),  # 12 light blue
    (255, 200, 100),  # 13 peach
    (200, 100, 255),  # 14 lavender
    (100, 255, 200),  # 15 mint
    (255, 100, 200),  # 16 rose
    (200, 255, 100),  # 17 yellow-green
    (100, 200, 255),  # 18 light sky
    (255, 180, 50),  # 19 amber
]

# Fallback colour when track_id is somehow out of palette range
DEFAULT_TRACK_COLOR: tuple[int, int, int] = (200, 200, 200)  # light grey

# HUD overlay text colour (white)
HUD_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)

# HUD overlay background colour (dark panel)
HUD_BG_COLOR: tuple[int, int, int] = (30, 30, 30)

# ==============================================================================
# Analytics
# ==============================================================================

# Smoothing factor for exponential moving average FPS calculation (0 < α ≤ 1).
# Lower values = smoother but slower to react; higher = more reactive.
EMA_ALPHA: float = 0.1

# ==============================================================================
# Logging
# ==============================================================================

DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILENAME: str = "visiontrack.log"
