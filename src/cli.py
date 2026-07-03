"""
Command-line interface for VisionTrack.

Provides the ``visiontrack`` entry point defined in pyproject.toml.
Parses arguments, wires every pipeline component together via dependency
injection, runs the pipeline, and prints a formatted summary.

Usage:
    visiontrack --input metro.mp4 --output results/
    visiontrack --input metro.mp4 --model yolov8m --device cuda --verbose
    visiontrack --input metro.mp4 --confidence 0.4 --skip-frames 1
    python main.py --input metro.mp4 --output results/
"""

from __future__ import annotations

import argparse
import sys
import traceback

from src import __version__
from src.constants import DEFAULT_MODEL, SUPPORTED_DEVICES, SUPPORTED_MODELS
from loggers import get_logger

log = get_logger(__name__)


# ==============================================================================
# Argument parser
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build and return the fully configured argument parser."""
    parser = argparse.ArgumentParser(
        prog="visiontrack",
        description=(
            "VisionTrack — Real-Time Multi-Object Tracking & Crowd Analytics\n"
            "Detects and tracks people in video using YOLOv8+ and BoT-SORT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  visiontrack --input metro.mp4 --output results/\n"
            "  visiontrack --input metro.mp4 --model yolov8m --device cuda\n"
            "  visiontrack --input metro.mp4 --confidence 0.4 --skip-frames 1\n"
            "  visiontrack --input metro.mp4 --verbose\n"
        ),
    )

    # --- Required ---
    parser.add_argument(
        "--input", "-i",
        metavar="VIDEO",
        required=True,
        help="Path to the input video file (e.g. metro.mp4).",
    )

    # --- Output & config ---
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default="outputs",
        help=(
            "Root directory for output files. A timestamped sub-directory is "
            "created per run. Default: outputs/"
        ),
    )
    parser.add_argument(
        "--config", "-c",
        metavar="YAML",
        default="config/config.yaml",
        help="Path to the configuration YAML file. Default: config/config.yaml",
    )

    # --- Model / detection overrides ---
    parser.add_argument(
        "--model",
        metavar="NAME",
        choices=SUPPORTED_MODELS,
        default=None,
        help=(
            f"YOLO model variant. Overrides config.yaml. "
            f"Default (from config): {DEFAULT_MODEL}. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}"
        ),
    )
    parser.add_argument(
        "--confidence",
        metavar="FLOAT",
        type=float,
        default=None,
        help="Detection confidence threshold [0.0–1.0]. Overrides config.yaml.",
    )

    # --- Device ---
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        choices=SUPPORTED_DEVICES,
        default=None,
        help=(
            f"Compute device to use. Overrides config.yaml. "
            f"Choices: {', '.join(SUPPORTED_DEVICES)}"
        ),
    )

    # --- Video processing ---
    parser.add_argument(
        "--skip-frames",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Skip N frames between each processed frame. "
            "0 = every frame (highest quality). "
            "1 = every other frame (2× speed). "
            "Overrides config.yaml."
        ),
    )

    # --- Flags ---
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging for detailed pipeline output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"VisionTrack {__version__}",
    )

    return parser


# ==============================================================================
# Argument validation
# ==============================================================================

def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate parsed arguments before touching the pipeline.

    Calls ``parser.error()`` on the first bad value found, which prints a
    clean message and exits with code 2 (standard argparse convention).

    Args:
        args:   Parsed namespace from ``parser.parse_args()``.
        parser: The ArgumentParser (used to call ``.error()``).
    """
    import os

    # Input file must exist
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: '{args.input}'")

    # Config file must exist
    if not os.path.isfile(args.config):
        parser.error(f"Config file not found: '{args.config}'")

    # Confidence must be in [0, 1]
    if args.confidence is not None and not (0.0 <= args.confidence <= 1.0):
        parser.error(
            f"--confidence must be in [0.0, 1.0], got {args.confidence}"
        )

    # skip_frames must be non-negative
    if args.skip_frames is not None and args.skip_frames < 0:
        parser.error(
            f"--skip-frames must be >= 0, got {args.skip_frames}"
        )


# ==============================================================================
# Pipeline assembly
# ==============================================================================

def build_pipeline(args: argparse.Namespace):
    """Construct and return a fully wired VideoPipeline.

    Wiring order (each component depends on the previous):
        ConfigManager → apply CLI overrides
        DeviceManager → resolve CUDA / MPS / CPU
        Detector      → load YOLO model, perform warmup
        Tracker       → attach BoT-SORT to the YOLO model
        Analytics     → initialise metrics accumulators
        Visualizer    → prepare rendering overlays
        Exporter      → prepare file output handlers
        VideoPipeline → inject all components

    Args:
        args: Validated parsed arguments.

    Returns:
        Tuple of (VideoPipeline, ConfigManager).
    """
    from src.analytics import Analytics
    from src.config_manager import ConfigManager
    from src.device_manager import DeviceManager
    from src.detector import Detector
    from src.exporter import Exporter
    from src.pipeline import VideoPipeline
    from src.tracker import Tracker
    from src.visualizer import Visualizer

    # ── Config ──────────────────────────────────────────────────────────
    log.info("Loading configuration from '%s' …", args.config)
    cfg = ConfigManager(args.config)

    # Apply CLI overrides (only non-None values are merged)
    cfg.apply_overrides(
        model=args.model,
        confidence=args.confidence,
        device=args.device,
        skip_frames=args.skip_frames,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # ── Device ───────────────────────────────────────────────────────────
    log.info("Detecting compute device …")
    device_mgr = DeviceManager(
        preferred=cfg.device.preferred,
        fallback=cfg.device.fallback,
    )
    log.info(device_mgr.summary())

    # ── Detector ────────────────────────────────────────────────────────
    log.info("Loading YOLO model '%s' …", cfg.model.type)
    detector = Detector(cfg, device=device_mgr.device)

    # ── Tracker ─────────────────────────────────────────────────────────
    log.info("Initialising BoT-SORT tracker …")
    tracker = Tracker(cfg, model=detector._model)

    # ── Analytics ────────────────────────────────────────────────────────
    analytics = Analytics(cfg)

    # ── Visualizer ───────────────────────────────────────────────────────
    visualizer = Visualizer(cfg)

    # ── Exporter ────────────────────────────────────────────────────────
    exporter = Exporter(cfg)

    # ── Pipeline ─────────────────────────────────────────────────────────
    pipeline = VideoPipeline(
        config=cfg,
        detector=detector,
        tracker=tracker,
        analytics=analytics,
        visualizer=visualizer,
        exporter=exporter,
    )

    return pipeline, cfg


# ==============================================================================
# Summary display
# ==============================================================================

def print_summary(summary, verbose: bool = False) -> None:
    """Print the pipeline run summary to stdout.

    Args:
        summary: PipelineSummary returned by pipeline.run().
        verbose: When True, include output file paths.
    """
    print()
    print(str(summary))

    if verbose:
        print()
        print("Output files:")
        if summary.output_video_path:
            print(f"  Video : {summary.output_video_path}")
        if summary.output_json_path:
            print(f"  JSON  : {summary.output_json_path}")
        if summary.output_csv_path:
            print(f"  CSV   : {summary.output_csv_path}")


# ==============================================================================
# Main entry point
# ==============================================================================

def main() -> None:
    """Main CLI entry point registered in pyproject.toml scripts.

    Exit codes:
        0 — successful run
        1 — pipeline error (VideoIOError, ModelLoadError, etc.)
        2 — argument / configuration error (argparse convention)
    """
    parser = build_parser()
    args   = parser.parse_args()

    # ── Validate arguments ───────────────────────────────────────────────
    validate_args(args, parser)

    # ── Print run header ─────────────────────────────────────────────────
    print(f"VisionTrack {__version__}")
    print(f"  Input    : {args.input}")
    print(f"  Output   : {args.output}")
    print(f"  Config   : {args.config}")
    print(f"  Model    : {args.model or '(from config)'}")
    print(f"  Device   : {args.device or '(from config)'}")
    print(f"  Verbose  : {args.verbose}")
    print()

    # ── Build pipeline ───────────────────────────────────────────────────
    try:
        pipeline, cfg = build_pipeline(args)
    except Exception as exc:
        log.error("Failed to initialise pipeline: %s", exc)
        if args.verbose:
            traceback.print_exc()
        print(f"\nError: {exc}", file=sys.stderr)
        print(
            "Tip: check your config file, model name, and device settings.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        summary = pipeline.run(args.input)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        print(f"\nError: {exc}", file=sys.stderr)
        print(
            "Tip: ensure the input video is a valid file and the output "
            "directory is writable.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────
    print_summary(summary, verbose=args.verbose)
    sys.exit(0)


if __name__ == "__main__":
    main()
