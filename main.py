#!/usr/bin/env python3
"""
VisionTrack - Real-Time Multi-Object Tracking & Crowd Analytics.

Entry point for both direct execution and the installed CLI command.

Usage:
    # Direct execution
    python main.py --input video.mp4 --output results/

    # After `pip install -e .`
    visiontrack --input video.mp4 --output results/
"""

from src.cli import main

if __name__ == "__main__":
    main()
