"""
Visualizer Module for VisionTrack.

Responsible for annotating BGR frames with tracking overlays:
  - Coloured bounding boxes with track ID labels
  - Semi-transparent HUD panels (unique visitor count, FPS, inference time)
  - Optional trajectory trails (last N bounding box centres)

All rendering is done in-place on a copy of the input frame so the
original is never mutated.  Every overlay is independently toggleable
via config flags so the same Visualizer works for both debug and clean
output modes.

Usage:
    from src.visualizer import Visualizer
    from src.config_manager import ConfigManager

    cfg        = ConfigManager("config/config.yaml")
    visualizer = Visualizer(cfg)

    annotated = visualizer.annotate(
        frame,
        tracks,
        unique_count=12,
        fps=28.5,
        inference_ms=35.0,
    )
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from loggers import get_logger
from src.constants import (
    BBOX_THICKNESS,
    COLOR_PALETTE,
    DEFAULT_TRACK_COLOR,
    FONT,
    FONT_SCALE_HUD,
    FONT_SCALE_LABEL,
    FONT_THICKNESS_HUD,
    FONT_THICKNESS_LABEL,
    HUD_ALPHA,
    HUD_BG_COLOR,
    HUD_TEXT_COLOR,
    LABEL_PADDING,
)
from src.data_models import Track

if TYPE_CHECKING:
    from src.config_manager import ConfigManager

log = get_logger(__name__)


class Visualizer:
    """Annotates video frames with tracking and analytics overlays.

    All drawing is performed on a copy of the input frame — the original
    is never modified.  Config flags control which overlays are active.

    Attributes:
        show_boxes:       Draw bounding boxes around tracked persons.
        show_ids:         Overlay track ID labels on bounding boxes.
        show_unique_count: Render unique visitor counter (top-left HUD).
        show_fps:         Render FPS / inference time (top-right HUD).
        trail_length:     Number of past centres kept per trail (0 = off).
        bbox_thickness:   Bounding box border thickness in pixels.
        hud_alpha:        Opacity of HUD background panels [0.0–1.0].

    Args:
        config: Loaded ConfigManager instance.
    """

    def __init__(self, config: "ConfigManager") -> None:
        v = config.visualization
        self.show_boxes:        bool  = v.show_boxes
        self.show_ids:          bool  = v.show_ids
        self.show_unique_count: bool  = v.show_unique_count
        self.show_fps:          bool  = v.show_fps
        self.trail_length:      int   = v.trail_length
        self.bbox_thickness:    int   = v.bbox_thickness
        self.hud_alpha:         float = v.hud_alpha

        # Trail history: track_id → deque of (cx, cy) centre points
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length if self.trail_length > 0 else 1)
        )

        log.info(
            "Visualizer ready — boxes=%s, ids=%s, trails=%d, alpha=%.1f.",
            self.show_boxes,
            self.show_ids,
            self.trail_length,
            self.hud_alpha,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        unique_count: int,
        fps: float,
        inference_ms: float,
    ) -> np.ndarray:
        """Annotate a frame with all enabled overlays.

        Creates a copy of the input frame so the original is preserved.

        Args:
            frame:        BGR image (H, W, 3) numpy array.
            tracks:       Active tracks for this frame.
            unique_count: Total unique persons seen so far.
            fps:          Current EMA frames-per-second.
            inference_ms: Most recent inference time in milliseconds.

        Returns:
            Annotated BGR frame as a new numpy array.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()

        if self.trail_length > 0:
            out = self._draw_trails(out, tracks)

        if self.show_boxes or self.show_ids:
            out = self._draw_tracks(out, tracks)

        if self.show_unique_count:
            out = self._draw_unique_count_hud(out, unique_count)

        if self.show_fps:
            out = self._draw_fps_hud(out, fps, inference_ms)

        return out

    def reset_trails(self) -> None:
        """Clear all trail history — call between independent video runs."""
        self._trails.clear()
        log.debug("Visualizer trail history cleared.")

    # ------------------------------------------------------------------
    # Internal: colour assignment
    # ------------------------------------------------------------------

    @staticmethod
    def get_track_color(track_id: int) -> tuple[int, int, int]:
        """Return a deterministic BGR colour for a given track ID.

        Uses modulo indexing into the palette so the same track always
        gets the same colour regardless of processing order.

        Args:
            track_id: BoT-SORT assigned integer track ID.

        Returns:
            (B, G, R) colour tuple.
        """
        if not COLOR_PALETTE:
            return DEFAULT_TRACK_COLOR
        return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]

    # ------------------------------------------------------------------
    # Internal: bounding boxes + labels
    # ------------------------------------------------------------------

    def _draw_tracks(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """Draw bounding boxes and/or ID labels for every active track.

        Args:
            frame:  BGR frame to annotate (modified in-place).
            tracks: Active tracks for this frame.

        Returns:
            The annotated frame.
        """
        for track in tracks:
            color = self.get_track_color(track.track_id)
            x1, y1, x2, y2 = track.bbox.to_int_list()

            if self.show_boxes:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    self.bbox_thickness,
                )

            if self.show_ids:
                label = f"#{track.track_id}"
                self._draw_label(frame, label, x1, y1, color)

        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        bg_color: tuple[int, int, int],
    ) -> None:
        """Draw a text label with a solid background rectangle.

        The background is drawn above the bounding box top edge.  If the
        label would render above the frame, it is shifted below y instead.

        Args:
            frame:    BGR frame (modified in-place).
            text:     Label string.
            x:        Left edge of the bounding box.
            y:        Top edge of the bounding box.
            bg_color: Background colour (same as bounding box colour).
        """
        (text_w, text_h), baseline = cv2.getTextSize(
            text, FONT, FONT_SCALE_LABEL, FONT_THICKNESS_LABEL
        )
        pad = LABEL_PADDING

        # Position the label above the box; fall back to below if clipped
        label_y1 = y - text_h - 2 * pad
        label_y2 = y
        text_y   = y - pad

        if label_y1 < 0:
            label_y1 = y
            label_y2 = y + text_h + 2 * pad
            text_y   = y + text_h + pad

        label_x2 = x + text_w + 2 * pad

        cv2.rectangle(frame, (x, label_y1), (label_x2, label_y2), bg_color, cv2.FILLED)
        cv2.putText(
            frame,
            text,
            (x + pad, text_y),
            FONT,
            FONT_SCALE_LABEL,
            HUD_TEXT_COLOR,
            FONT_THICKNESS_LABEL,
            cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Internal: trajectory trails
    # ------------------------------------------------------------------

    def _draw_trails(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """Draw trajectory trails for each active track.

        Updates the internal trail deque with the current bbox centre,
        then draws connected line segments through stored positions.

        Args:
            frame:  BGR frame to annotate (modified in-place).
            tracks: Active tracks for this frame.

        Returns:
            The annotated frame.
        """
        for track in tracks:
            cx = int((track.bbox.x1 + track.bbox.x2) / 2)
            cy = int((track.bbox.y1 + track.bbox.y2) / 2)
            self._trails[track.track_id].append((cx, cy))

            pts = list(self._trails[track.track_id])
            if len(pts) < 2:
                continue

            color = self.get_track_color(track.track_id)
            n     = len(pts)

            for i in range(1, n):
                # Fade older segments: opacity proportional to position in trail
                alpha  = i / n
                faded  = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(self.bbox_thickness * alpha))
                cv2.line(frame, pts[i - 1], pts[i], faded, thickness)

        return frame

    # ------------------------------------------------------------------
    # Internal: HUD overlays
    # ------------------------------------------------------------------

    def _draw_unique_count_hud(
        self, frame: np.ndarray, unique_count: int
    ) -> np.ndarray:
        """Render the unique visitor counter in the top-left corner.

        Uses a semi-transparent dark background panel for legibility on
        any video content.

        Args:
            frame:        BGR frame to annotate (modified in-place).
            unique_count: Total unique persons counted so far.

        Returns:
            The annotated frame.
        """
        text = f"Unique Visitors: {unique_count}"
        return self._draw_hud_panel(frame, text, position="top-left")

    def _draw_fps_hud(
        self, frame: np.ndarray, fps: float, inference_ms: float
    ) -> np.ndarray:
        """Render FPS and inference time in the top-right corner.

        Args:
            frame:        BGR frame to annotate (modified in-place).
            fps:          Current EMA FPS.
            inference_ms: Most recent inference latency in milliseconds.

        Returns:
            The annotated frame.
        """
        text = f"FPS: {fps:.1f} | {inference_ms:.0f}ms"
        return self._draw_hud_panel(frame, text, position="top-right")

    def _draw_hud_panel(
        self,
        frame: np.ndarray,
        text: str,
        position: str = "top-left",
    ) -> np.ndarray:
        """Draw a single HUD text panel with a semi-transparent background.

        The transparency is achieved with an additive blend rather than
        full alpha compositing to avoid per-pixel copies on every frame.

        Args:
            frame:    BGR frame (modified in-place).
            text:     Text to display.
            position: ``"top-left"`` or ``"top-right"``.

        Returns:
            The annotated frame.
        """
        h, w = frame.shape[:2]
        margin = 10

        (text_w, text_h), _ = cv2.getTextSize(
            text, FONT, FONT_SCALE_HUD, FONT_THICKNESS_HUD
        )
        pad = LABEL_PADDING * 2

        panel_w = text_w + pad * 2
        panel_h = text_h + pad * 2

        if position == "top-left":
            px1, py1 = margin, margin
        else:  # top-right
            px1 = w - panel_w - margin
            py1 = margin

        px2 = px1 + panel_w
        py2 = py1 + panel_h

        # Clamp to frame boundaries
        px1 = max(0, px1)
        py1 = max(0, py1)
        px2 = min(w, px2)
        py2 = min(h, py2)

        # Semi-transparent background via weighted blend
        roi = frame[py1:py2, px1:px2]
        if roi.size > 0:
            bg = np.full_like(roi, HUD_BG_COLOR, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 1.0 - self.hud_alpha, bg, self.hud_alpha, 0)
            frame[py1:py2, px1:px2] = blended

        text_x = px1 + pad
        text_y = py1 + pad + text_h

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            FONT,
            FONT_SCALE_HUD,
            HUD_TEXT_COLOR,
            FONT_THICKNESS_HUD,
            cv2.LINE_AA,
        )

        return frame

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Visualizer("
            f"boxes={self.show_boxes}, "
            f"ids={self.show_ids}, "
            f"trails={self.trail_length}, "
            f"alpha={self.hud_alpha})"
        )
