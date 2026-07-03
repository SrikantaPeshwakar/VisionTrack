#!/usr/bin/env python3
"""
VisionTrack Evaluation Script

Benchmarks multiple YOLO model variants on the same input video and
prints a formatted comparison table with per-model performance metrics.
Results are also saved to an evaluation_report.json file.

Metrics reported per model:
  - Avg FPS           — smoothed frames per second
  - Avg inference ms  — mean detection + tracking time per frame
  - Total time (s)    — wall-clock time for the full run
  - Unique visitors   — total unique person IDs across the video
  - Peak concurrent   — max simultaneous tracks in any single frame
  - Avg dwell time    — average time each person was visible (seconds)

Usage:
    # Compare three model sizes
    visiontrack-eval --video sample_videos/sample.mp4 \\
                     --models yolov8n yolov8s yolov8m

    # Quick single-model benchmark
    visiontrack-eval --video sample_videos/sample.mp4

    # Custom config + device
    visiontrack-eval --video metro.mp4 --models yolov8n yolov8m \\
                     --config config/config.yaml --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable when run as `python evaluate.py`
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loggers import get_logger
from src import __version__
from src.constants import (
    DEFAULT_EVAL_REPORT_FILENAME,
    SUPPORTED_DEVICES,
    SUPPORTED_MODELS,
)

log = get_logger(__name__)


# ===========================================================================
# Argument parser
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visiontrack-eval",
        description=(
            "VisionTrack Evaluation — benchmark multiple YOLO models on a video\n"
            "and print a side-by-side comparison table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  visiontrack-eval --video sample_videos/sample.mp4\n"
            "  visiontrack-eval --video metro.mp4 --models yolov8n yolov8s yolov8m\n"
            "  visiontrack-eval --video metro.mp4 --models yolov8n yolov8m --device mps\n"
        ),
    )

    parser.add_argument(
        "--video",
        "-i",
        metavar="VIDEO",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--models",
        metavar="NAME",
        nargs="+",
        default=["yolov8n"],
        choices=SUPPORTED_MODELS,
        help=(
            f"One or more YOLO model variants to benchmark. "
            f"Default: yolov8n. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="YAML",
        default="config/config.yaml",
        help="Path to config.yaml. Default: config/config.yaml",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        choices=SUPPORTED_DEVICES,
        default=None,
        help=f"Compute device. Overrides config.yaml. Choices: {', '.join(SUPPORTED_DEVICES)}",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        default="outputs/eval",
        help=(
            "Directory for evaluation outputs (annotated videos + report). " "Default: outputs/eval"
        ),
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help="Save annotated output video for each model (disabled by default for speed).",
    )
    parser.add_argument(
        "--skip-frames",
        metavar="N",
        type=int,
        default=None,
        help="Skip N frames between processed frames. Applied to all models.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"VisionTrack {__version__}",
    )

    return parser


# ===========================================================================
# Validation
# ===========================================================================


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not os.path.isfile(args.video):
        parser.error(f"Video file not found: '{args.video}'")
    if not os.path.isfile(args.config):
        parser.error(f"Config file not found: '{args.config}'")
    if args.skip_frames is not None and args.skip_frames < 0:
        parser.error(f"--skip-frames must be >= 0, got {args.skip_frames}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for m in args.models:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    args.models = deduped


# ===========================================================================
# Single-model benchmark
# ===========================================================================


def _run_model(
    model_name: str,
    video_path: str,
    config_path: str,
    device: str | None,
    output_dir: str,
    save_video: bool,
    skip_frames: int | None,
) -> dict:
    """Run the full pipeline for one model and return a metrics dict.

    Args:
        model_name:  YOLO variant name.
        video_path:  Input video path.
        config_path: Path to config.yaml.
        device:      Compute device override.
        output_dir:  Root output directory for this model run.
        save_video:  Whether to write the annotated output video.
        skip_frames: Frame-skipping override.

    Returns:
        Dict with keys: model, status, error, avg_fps, avg_inference_ms,
        total_time, unique_visitors, peak_concurrent, avg_dwell_time,
        output_dir.
    """
    from src.analytics import Analytics
    from src.config_manager import ConfigManager
    from src.detector import Detector
    from src.device_manager import DeviceManager
    from src.exporter import Exporter
    from src.pipeline import VideoPipeline
    from src.tracker import Tracker
    from src.visualizer import Visualizer

    result: dict = {
        "model": model_name,
        "status": "error",
        "error": None,
        "avg_fps": 0.0,
        "avg_inference_ms": 0.0,
        "total_time": 0.0,
        "unique_visitors": 0,
        "peak_concurrent": 0,
        "avg_dwell_time": 0.0,
        "output_dir": "",
    }

    try:
        # ── Config ──────────────────────────────────────────────────────
        cfg = ConfigManager(config_path)
        cfg.apply_overrides(
            model=model_name,
            device=device,
            skip_frames=skip_frames,
            output_dir=output_dir,
            # Always save JSON for metrics; video is optional
        )
        # Force export flags for eval context
        cfg.export.save_video = save_video
        cfg.export.save_json = True
        cfg.export.save_csv = False  # not needed for benchmarking

        # ── Device ──────────────────────────────────────────────────────
        device_mgr = DeviceManager(
            preferred=cfg.device.preferred,
            fallback=cfg.device.fallback,
        )

        # ── Components ──────────────────────────────────────────────────
        detector = Detector(cfg, device=device_mgr.device)
        tracker = Tracker(cfg, model=detector._model)
        analytics = Analytics(cfg)
        visualizer = Visualizer(cfg)
        exporter = Exporter(cfg)

        pipeline = VideoPipeline(
            config=cfg,
            detector=detector,
            tracker=tracker,
            analytics=analytics,
            visualizer=visualizer,
            exporter=exporter,
        )

        # ── Run ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        summary = pipeline.run(video_path)
        elapsed = time.perf_counter() - t0

        result.update(
            {
                "status": "ok",
                "avg_fps": round(summary.avg_fps, 2),
                "avg_inference_ms": round(summary.avg_inference_time_ms, 2),
                "total_time": round(elapsed, 2),
                "unique_visitors": summary.unique_visitors,
                "peak_concurrent": summary.peak_concurrent_tracks,
                "avg_dwell_time": round(summary.avg_dwell_time, 2),
                "output_dir": summary.output_video_path or cfg.export.output_dir,
            }
        )

    except Exception as exc:
        result["error"] = str(exc)
        log.error("Evaluation failed for model '%s': %s", model_name, exc)

    return result


# ===========================================================================
# Display helpers
# ===========================================================================


def _print_table(results: list[dict]) -> None:
    """Print a formatted ASCII comparison table to stdout."""
    cols = [
        ("Model", "model", 12),
        ("Avg FPS", "avg_fps", 9),
        ("Inf (ms)", "avg_inference_ms", 10),
        ("Time (s)", "total_time", 9),
        ("Unique", "unique_visitors", 8),
        ("Peak", "peak_concurrent", 6),
        ("Dwell (s)", "avg_dwell_time", 10),
        ("Status", "status", 8),
    ]

    # Header
    sep = "+" + "+".join("-" * (w + 2) for _, _, w in cols) + "+"
    hdr = "|" + "|".join(f" {label:<{w}} " for label, _, w in cols) + "|"
    print()
    print(sep)
    print(hdr)
    print(sep)

    for r in results:
        row = "|"
        for label, key, w in cols:
            val = r.get(key, "—")
            if key in ("avg_fps", "avg_inference_ms", "total_time", "avg_dwell_time"):
                cell = f"{val:.2f}" if isinstance(val, float) else str(val)
            else:
                cell = str(val)
            row += f" {cell:<{w}} |"
        print(row)

    print(sep)
    print()


def _print_header(video_path: str, models: list[str], device: str | None) -> None:
    print()
    print(f"VisionTrack {__version__} — Evaluation")
    print(f"  Video   : {video_path}")
    print(f"  Models  : {', '.join(models)}")
    print(f"  Device  : {device or '(from config)'}")
    print()


# ===========================================================================
# System info
# ===========================================================================


def _collect_system_info() -> dict:
    """Collect platform and library versions for the report metadata."""
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "N/A"
        info["mps"] = str(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except ImportError:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except ImportError:
        pass
    return info


# ===========================================================================
# Report saving
# ===========================================================================


def _save_report(
    results: list[dict],
    video_path: str,
    output_dir: str,
) -> str:
    """Write the evaluation report to JSON and return its path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = str(Path(output_dir) / DEFAULT_EVAL_REPORT_FILENAME)

    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "visiontrack": __version__,
        "input_video": video_path,
        "system": _collect_system_info(),
        "results": results,
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)

    return report_path


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Entry point registered in pyproject.toml as ``visiontrack-eval``."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    _print_header(args.video, args.models, args.device)

    results: list[dict] = []

    for i, model_name in enumerate(args.models, 1):
        model_output_dir = str(Path(args.output) / model_name)
        print(
            f"[{i}/{len(args.models)}] Running {model_name} …",
            flush=True,
        )

        result = _run_model(
            model_name=model_name,
            video_path=args.video,
            config_path=args.config,
            device=args.device,
            output_dir=model_output_dir,
            save_video=args.save_video,
            skip_frames=args.skip_frames,
        )
        results.append(result)

        if result["status"] == "ok":
            print(
                f"    Done — {result['avg_fps']:.1f} FPS | "
                f"{result['avg_inference_ms']:.0f}ms | "
                f"{result['unique_visitors']} unique"
            )
        else:
            print(f"    FAILED — {result['error']}")

    # ── Comparison table ─────────────────────────────────────────────────
    _print_table(results)

    # ── Save report ───────────────────────────────────────────────────────
    report_path = _save_report(results, args.video, args.output)
    print(f"Evaluation report saved: {report_path}")
    print()

    # Exit non-zero if any model failed
    failed = [r for r in results if r["status"] != "ok"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
