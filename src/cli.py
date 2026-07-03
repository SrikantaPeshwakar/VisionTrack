"""
Command-line interface for VisionTrack.

Full implementation: Task 7.
This stub allows main.py to import and --help to work from Task 1 onwards.
"""

import argparse
import sys

from src import __version__
from src.constants import SUPPORTED_MODELS, DEFAULT_MODEL, SUPPORTED_DEVICES


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
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
        ),
    )

    # --- Required ---
    parser.add_argument(
        "--input", "-i",
        metavar="VIDEO",
        required=True,
        help="Path to the input video file (e.g. metro.mp4).",
    )

    # --- Output ---
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default="outputs",
        help="Root directory for output files. A timestamped sub-directory is "
             "created per run. Default: outputs/",
    )
    parser.add_argument(
        "--config", "-c",
        metavar="YAML",
        default="config/config.yaml",
        help="Path to the configuration YAML file. Default: config/config.yaml",
    )

    # --- Model overrides ---
    parser.add_argument(
        "--model",
        metavar="NAME",
        choices=SUPPORTED_MODELS,
        default=None,
        help=(
            f"YOLO model variant to use. Overrides config.yaml. "
            f"Default (from config): {DEFAULT_MODEL}. "
            f"Choices: {', '.join(SUPPORTED_MODELS)}"
        ),
    )
    parser.add_argument(
        "--confidence",
        metavar="FLOAT",
        type=float,
        default=None,
        help="Detection confidence threshold [0.0–1.0]. Overrides config.yaml.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        choices=SUPPORTED_DEVICES,
        default=None,
        help=f"Compute device to use. Overrides config.yaml. Choices: {', '.join(SUPPORTED_DEVICES)}",
    )
    parser.add_argument(
        "--skip-frames",
        metavar="N",
        type=int,
        default=None,
        help="Process every (N+1)th frame. 0 = every frame. Overrides config.yaml.",
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


def main() -> None:
    """Main entry point — parses arguments and launches the pipeline.

    Full pipeline wiring is implemented in Task 7. This stub validates
    arguments and prints a clear message until the pipeline is ready.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Basic input validation (full validation added in Task 7)
    import os
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: '{args.input}'")

    if args.confidence is not None and not (0.0 <= args.confidence <= 1.0):
        parser.error(f"--confidence must be in [0.0, 1.0], got {args.confidence}")

    if args.skip_frames is not None and args.skip_frames < 0:
        parser.error(f"--skip-frames must be >= 0, got {args.skip_frames}")

    # Pipeline not yet wired — implemented in Task 7
    print(f"VisionTrack {__version__}")
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Config  : {args.config}")
    print(f"  Model   : {args.model or '(from config)'}")
    print(f"  Device  : {args.device or '(from config)'}")
    print(f"  Verbose : {args.verbose}")
    print()
    print("Pipeline not yet initialised — full implementation in Task 7.")
    sys.exit(0)


if __name__ == "__main__":
    main()
