#!/usr/bin/env python3
"""
MOT17 Dataset Download & Conversion Helper

Downloads selected MOT17 sequences from the official source and converts
the image sequences to MP4 video files suitable for VisionTrack evaluation.

MOT17 sequences used for VisionTrack testing:
  - MOT17-04  : Static camera, crowded train station platform
  - MOT17-09  : Static camera, busy pedestrian crossing
  - MOT17-11  : Moving camera, crowded outdoor scene

These sequences represent the hardest tracking conditions:
  - Dense crowds with heavy occlusions
  - Crisscrossing pedestrian paths
  - Variable lighting

Usage:
    python scripts/download_mot.py
    python scripts/download_mot.py --sequences MOT17-04 MOT17-09
    python scripts/download_mot.py --output sample_videos/ --fps 30

Requirements:
    pip install requests tqdm opencv-python

Notes:
    - Full MOT17 dataset is ~5GB. This script downloads selected sequences only.
    - Images are converted to MP4 using OpenCV and then the raw frames are removed.
    - The official MOT17 download requires registration at https://motchallenge.net
      Use --mot-root to point to a locally extracted MOT17 directory instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# MOT17 sequence metadata
# ---------------------------------------------------------------------------

# Each entry: (sequence_name, description, approx_frames, fps)
MOT17_SEQUENCES = {
    "MOT17-04": (
        "MOT17-04-DPM",
        "Static camera — crowded train station (1050 frames)",
        1050,
        30,
    ),
    "MOT17-09": (
        "MOT17-09-DPM",
        "Static camera — busy pedestrian crossing (525 frames)",
        525,
        30,
    ),
    "MOT17-11": (
        "MOT17-11-DPM",
        "Moving camera — crowded outdoor scene (900 frames)",
        900,
        30,
    ),
}

DEFAULT_SEQUENCES = ["MOT17-04", "MOT17-09"]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_mot",
        description="Download and convert MOT17 sequences to MP4 for VisionTrack.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Convert from a locally extracted MOT17 directory\n"
            "  python scripts/download_mot.py --mot-root ~/datasets/MOT17\n\n"
            "  # Specify sequences and output directory\n"
            "  python scripts/download_mot.py --mot-root ~/datasets/MOT17 \\\n"
            "      --sequences MOT17-04 MOT17-09 --output sample_videos/\n"
        ),
    )
    parser.add_argument(
        "--mot-root",
        metavar="DIR",
        default=None,
        help=(
            "Path to a locally extracted MOT17 dataset directory "
            "(contains MOT17-04-DPM, MOT17-09-DPM, etc.). "
            "If not provided, download instructions are printed."
        ),
    )
    parser.add_argument(
        "--sequences",
        metavar="SEQ",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        choices=list(MOT17_SEQUENCES.keys()),
        help=(
            f"Sequences to convert. Default: {' '.join(DEFAULT_SEQUENCES)}. "
            f"Available: {', '.join(MOT17_SEQUENCES.keys())}"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default="sample_videos",
        help="Output directory for MP4 files. Default: sample_videos/",
    )
    parser.add_argument(
        "--fps",
        metavar="N",
        type=float,
        default=30.0,
        help="Frame rate for output videos. Default: 30",
    )
    return parser


# ---------------------------------------------------------------------------
# Conversion: image sequence → MP4
# ---------------------------------------------------------------------------

def convert_sequence_to_mp4(
    seq_dir: Path,
    output_path: Path,
    fps: float,
    seq_name: str,
) -> bool:
    """Convert a MOT17 image sequence directory to an MP4 file.

    MOT17 sequences store frames as JPEG images in:
        <seq_dir>/img1/000001.jpg
        <seq_dir>/img1/000002.jpg
        ...

    Args:
        seq_dir:     Path to the sequence directory (e.g. MOT17-04-DPM/).
        output_path: Path to write the output MP4 file.
        fps:         Frame rate for the output video.
        seq_name:    Display name for progress output.

    Returns:
        True on success, False on failure.
    """
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required. Run: pip install opencv-python")
        return False

    img_dir = seq_dir / "img1"
    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        return False

    frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not frames:
        print(f"ERROR: No image files found in {img_dir}")
        return False

    # Read first frame to get dimensions
    first = cv2.imread(str(frames[0]))
    if first is None:
        print(f"ERROR: Could not read first frame: {frames[0]}")
        return False

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"ERROR: Could not open VideoWriter for: {output_path}")
        return False

    print(f"  Converting {seq_name}: {len(frames)} frames @ {fps} FPS → {output_path.name}")

    try:
        from tqdm import tqdm
        frame_iter = tqdm(frames, desc=f"  {seq_name}", unit="frame", leave=False)
    except ImportError:
        frame_iter = frames

    written = 0
    for frame_path in frame_iter:
        img = cv2.imread(str(frame_path))
        if img is not None:
            writer.write(img)
            written += 1

    writer.release()
    print(f"  ✓ {seq_name}: {written} frames written → {output_path}")
    return True


# ---------------------------------------------------------------------------
# Download instructions
# ---------------------------------------------------------------------------

def print_download_instructions() -> None:
    """Print manual download instructions when --mot-root is not provided."""
    print()
    print("=" * 65)
    print("  MOT17 Dataset — Manual Download Instructions")
    print("=" * 65)
    print()
    print("  MOT17 requires registration at:")
    print("  https://motchallenge.net/data/MOT17/")
    print()
    print("  Steps:")
    print("  1. Register at https://motchallenge.net (free)")
    print("  2. Download MOT17.zip (~5GB) or individual sequences")
    print("  3. Extract to a local directory, e.g. ~/datasets/MOT17/")
    print("  4. Run this script with --mot-root pointing to that directory:")
    print()
    print("     python scripts/download_mot.py \\")
    print("         --mot-root ~/datasets/MOT17 \\")
    print("         --sequences MOT17-04 MOT17-09")
    print()
    print("  Recommended sequences for VisionTrack testing:")
    for key, (name, desc, frames, fps) in MOT17_SEQUENCES.items():
        print(f"    {key:12s}  {desc}")
    print()
    print("  Alternative: use any crowd video with the standard CLI:")
    print("    python main.py --input your_crowd_video.mp4 --model yolov8m")
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.mot_root is None:
        print_download_instructions()
        sys.exit(0)

    mot_root = Path(args.mot_root)
    if not mot_root.is_dir():
        print(f"ERROR: MOT root directory not found: {mot_root}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisionTrack — MOT17 Sequence Converter")
    print(f"  MOT root : {mot_root}")
    print(f"  Output   : {output_dir}")
    print(f"  Sequences: {', '.join(args.sequences)}")
    print()

    success_count = 0
    for seq_key in args.sequences:
        seq_folder, desc, _, _ = MOT17_SEQUENCES[seq_key]
        seq_dir = mot_root / seq_folder

        if not seq_dir.is_dir():
            # Some MOT17 distributions use different detector suffixes
            # try other common variants
            for suffix in ["DPM", "FRCNN", "SDP"]:
                alt = mot_root / f"{seq_key}-{suffix}"
                if alt.is_dir():
                    seq_dir = alt
                    break
            else:
                print(f"  SKIP {seq_key}: directory not found in {mot_root}")
                continue

        output_path = output_dir / f"{seq_key}.mp4"
        if output_path.is_file():
            print(f"  SKIP {seq_key}: already exists at {output_path}")
            success_count += 1
            continue

        ok = convert_sequence_to_mp4(seq_dir, output_path, args.fps, seq_key)
        if ok:
            success_count += 1

    print()
    print(f"Done: {success_count}/{len(args.sequences)} sequences converted.")
    print()
    if success_count > 0:
        print("Run evaluation with:")
        videos = " ".join(
            str(output_dir / f"{s}.mp4")
            for s in args.sequences
        )
        print(f"  python evaluate.py --video {list(output_dir.glob('MOT17*.mp4'))[0]} \\")
        print(f"      --models yolov8n yolov8s yolov8m --device mps")
    print()


if __name__ == "__main__":
    main()
