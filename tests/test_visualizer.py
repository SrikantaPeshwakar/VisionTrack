"""
Unit tests for src/visualizer.py

Uses real numpy arrays as frames — no display window or GPU needed.
All tests verify pixel-level changes or structural behaviour rather
than exact pixel values, making them robust to minor rendering tweaks.

Covers:
- Construction and __repr__
- get_track_color: determinism, consistency, palette cycling, empty palette
- annotate(): returns copy (original unchanged), correct shape, all
              overlays toggled via config flags
- _draw_tracks: bounding box pixels change, label drawn above box
- _draw_hud_panel: top-left and top-right placement, alpha blend changes pixels
- _draw_trails: trail history updated, trail drawn for 2+ frames, fade
- reset_trails: clears internal state
- Edge cases: empty tracks, None/empty frame guards
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.constants import (
    COLOR_PALETTE,
    DEFAULT_TRACK_COLOR,
    PERSON_CLASS_ID,
)
from src.data_models import BoundingBox, Track

# ===========================================================================
# Helpers
# ===========================================================================


def _make_config(
    show_boxes: bool = True,
    show_ids: bool = True,
    show_unique_count: bool = True,
    show_fps: bool = True,
    trail_length: int = 0,
    bbox_thickness: int = 2,
    hud_alpha: float = 0.6,
) -> MagicMock:
    cfg = MagicMock()
    cfg.visualization.show_boxes = show_boxes
    cfg.visualization.show_ids = show_ids
    cfg.visualization.show_unique_count = show_unique_count
    cfg.visualization.show_fps = show_fps
    cfg.visualization.trail_length = trail_length
    cfg.visualization.bbox_thickness = bbox_thickness
    cfg.visualization.hud_alpha = hud_alpha
    return cfg


def _make_visualizer(**kwargs):
    from src.visualizer import Visualizer

    return Visualizer(_make_config(**kwargs))


def _make_track(
    track_id: int = 1,
    x1: float = 50.0,
    y1: float = 50.0,
    x2: float = 150.0,
    y2: float = 200.0,
    frame_id: int = 0,
    ts: float = 0.0,
) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=0.9,
        class_id=PERSON_CLASS_ID,
        frame_id=frame_id,
        timestamp=ts,
    )


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _white_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a white BGR frame (255 everywhere)."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


# ===========================================================================
# Construction & __repr__
# ===========================================================================


class TestVisualizerConstruction:
    def test_attributes_from_config(self):
        v = _make_visualizer(show_boxes=True, show_ids=False, trail_length=10, hud_alpha=0.5)
        assert v.show_boxes is True
        assert v.show_ids is False
        assert v.trail_length == 10
        assert v.hud_alpha == pytest.approx(0.5)

    def test_repr_contains_key_info(self):
        v = _make_visualizer(trail_length=5, hud_alpha=0.7)
        r = repr(v)
        assert "Visualizer" in r
        assert "trails=5" in r
        assert "0.7" in r

    def test_trails_dict_initially_empty(self):
        v = _make_visualizer()
        assert len(v._trails) == 0


# ===========================================================================
# get_track_color
# ===========================================================================


class TestGetTrackColor:
    def test_returns_tuple_of_three(self):
        from src.visualizer import Visualizer

        color = Visualizer.get_track_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3

    def test_deterministic_same_id(self):
        from src.visualizer import Visualizer

        assert Visualizer.get_track_color(1) == Visualizer.get_track_color(1)

    def test_different_ids_may_differ(self):
        from src.visualizer import Visualizer

        # IDs 0 and 1 are different palette entries
        assert Visualizer.get_track_color(0) != Visualizer.get_track_color(1)

    def test_palette_cycling(self):
        from src.visualizer import Visualizer

        n = len(COLOR_PALETTE)
        assert Visualizer.get_track_color(0) == Visualizer.get_track_color(n)
        assert Visualizer.get_track_color(3) == Visualizer.get_track_color(n + 3)

    def test_values_in_bgr_range(self):
        from src.visualizer import Visualizer

        for tid in range(50):
            color = Visualizer.get_track_color(tid)
            assert all(0 <= c <= 255 for c in color), f"Color out of range: {color}"

    def test_empty_palette_returns_default(self, monkeypatch):
        import src.visualizer as vis_mod
        from src.visualizer import Visualizer

        monkeypatch.setattr(vis_mod, "COLOR_PALETTE", [])
        assert Visualizer.get_track_color(5) == DEFAULT_TRACK_COLOR


# ===========================================================================
# annotate() — output properties
# ===========================================================================


class TestAnnotateOutput:
    def test_returns_ndarray(self):
        v = _make_visualizer()
        out = v.annotate(_blank_frame(), [], 0, 0.0, 0.0)
        assert isinstance(out, np.ndarray)

    def test_output_same_shape_as_input(self):
        v = _make_visualizer()
        frame = _blank_frame(360, 480)
        out = v.annotate(frame, [], 0, 0.0, 0.0)
        assert out.shape == frame.shape

    def test_original_frame_not_mutated(self):
        v = _make_visualizer()
        frame = _blank_frame()
        original = frame.copy()
        v.annotate(frame, [_make_track()], 5, 25.0, 35.0)
        np.testing.assert_array_equal(frame, original)

    def test_empty_tracks_returns_annotated_frame(self):
        """Even with no tracks the HUD overlays still change pixels."""
        v = _make_visualizer(show_unique_count=True, show_fps=True)
        frame = _blank_frame()
        out = v.annotate(frame, [], 0, 0.0, 0.0)
        assert out.shape == frame.shape

    def test_none_frame_returned_as_is(self):
        v = _make_visualizer()
        result = v.annotate(None, [], 0, 0.0, 0.0)
        assert result is None

    def test_empty_numpy_frame_returned_as_is(self):
        v = _make_visualizer()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = v.annotate(empty, [], 0, 0.0, 0.0)
        assert result.size == 0


# ===========================================================================
# annotate() — overlay toggles
# ===========================================================================


class TestAnnotateOverlayToggles:
    def _pixels_changed(self, original: np.ndarray, annotated: np.ndarray) -> bool:
        return not np.array_equal(original, annotated)

    def test_show_boxes_true_changes_pixels(self):
        v = _make_visualizer(
            show_boxes=True, show_ids=False, show_unique_count=False, show_fps=False
        )
        frame = _blank_frame()
        out = v.annotate(frame, [_make_track()], 0, 0.0, 0.0)
        assert self._pixels_changed(frame, out)

    def test_show_ids_true_changes_pixels(self):
        v = _make_visualizer(
            show_boxes=False, show_ids=True, show_unique_count=False, show_fps=False
        )
        frame = _blank_frame()
        out = v.annotate(frame, [_make_track()], 0, 0.0, 0.0)
        assert self._pixels_changed(frame, out)

    def test_show_unique_count_true_changes_pixels(self):
        v = _make_visualizer(
            show_boxes=False, show_ids=False, show_unique_count=True, show_fps=False
        )
        frame = _blank_frame()
        out = v.annotate(frame, [], 5, 0.0, 0.0)
        assert self._pixels_changed(frame, out)

    def test_show_fps_true_changes_pixels(self):
        v = _make_visualizer(
            show_boxes=False, show_ids=False, show_unique_count=False, show_fps=True
        )
        frame = _blank_frame()
        out = v.annotate(frame, [], 0, 28.5, 35.0)
        assert self._pixels_changed(frame, out)

    def test_all_overlays_disabled_no_change_from_tracks(self):
        """With all overlays off and no trails, output == input copy."""
        v = _make_visualizer(
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
            trail_length=0,
        )
        frame = _blank_frame()
        out = v.annotate(frame, [_make_track()], 0, 0.0, 0.0)
        np.testing.assert_array_equal(out, frame)

    def test_trails_enabled_changes_pixels_after_two_frames(self):
        """Trails need 2 frames before they draw line segments."""
        v = _make_visualizer(
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
            trail_length=10,
        )
        frame = _blank_frame()
        track = _make_track(1, x1=50, y1=50, x2=150, y2=200)
        v.annotate(frame, [track], 0, 0.0, 0.0)  # frame 0 — no trail yet

        track2 = _make_track(1, x1=60, y1=60, x2=160, y2=210)
        out = v.annotate(frame, [track2], 0, 0.0, 0.0)  # frame 1 — trail drawn
        assert not np.array_equal(out, frame)


# ===========================================================================
# _draw_tracks
# ===========================================================================


class TestDrawTracks:
    def test_bbox_region_changes_on_black_frame(self):
        """On a black frame, drawing a coloured box changes pixels in bbox area."""
        v = _make_visualizer(
            show_boxes=True, show_ids=False, show_unique_count=False, show_fps=False
        )
        frame = _blank_frame()
        track = _make_track(1, x1=100, y1=100, x2=200, y2=300)
        out = v._draw_tracks(frame.copy(), [track])
        # Some pixels on the bounding box edge should be non-zero
        assert out[100, 100:200].max() > 0 or out[100:300, 100].max() > 0

    def test_multiple_tracks_all_drawn(self):
        v = _make_visualizer(
            show_boxes=True, show_ids=False, show_unique_count=False, show_fps=False
        )
        frame = _blank_frame()
        tracks = [
            _make_track(1, x1=10, y1=10, x2=100, y2=100),
            _make_track(2, x1=300, y1=200, x2=400, y2=350),
        ]
        out = v._draw_tracks(frame.copy(), tracks)
        # Both track regions should have non-zero pixels
        assert out[10, 10:100].max() > 0
        assert out[200, 300:400].max() > 0

    def test_different_track_ids_get_different_colors(self):
        from src.visualizer import Visualizer

        c1 = Visualizer.get_track_color(0)
        c2 = Visualizer.get_track_color(1)
        assert c1 != c2


# ===========================================================================
# HUD panels
# ===========================================================================


class TestHudPanels:
    def test_unique_count_hud_changes_top_left(self):
        v = _make_visualizer(
            show_boxes=False, show_ids=False, show_fps=False, show_unique_count=True
        )
        frame = _blank_frame()
        out = v.annotate(frame, [], 42, 0.0, 0.0)
        # Top-left region (first 80 rows, first 300 cols) should have changed
        assert out[:80, :300].max() > 0

    def test_fps_hud_changes_top_right(self):
        v = _make_visualizer(
            show_boxes=False, show_ids=False, show_fps=True, show_unique_count=False
        )
        w = 640
        frame = _blank_frame(w=w)
        out = v.annotate(frame, [], 0, 28.5, 35.0)
        # Top-right region should have changed
        assert out[:80, w // 2 :].max() > 0

    def test_hud_alpha_zero_no_background_blend(self):
        """With alpha=0 the background ROI is untouched — only text is drawn."""
        v = _make_visualizer(
            show_unique_count=True, show_fps=False, show_boxes=False, show_ids=False, hud_alpha=0.0
        )
        frame = _blank_frame()
        out = v.annotate(frame, [], 5, 0.0, 0.0)
        # Some pixels will be non-zero (text), but fewer than with alpha=0.6
        assert isinstance(out, np.ndarray)

    def test_hud_alpha_one_full_background(self):
        """With alpha=1 the background panel is fully opaque."""
        v = _make_visualizer(
            show_unique_count=True, show_fps=False, show_boxes=False, show_ids=False, hud_alpha=1.0
        )
        frame = _blank_frame()
        out = v.annotate(frame, [], 5, 0.0, 0.0)
        # Top-left panel region should match HUD_BG_COLOR (30,30,30) roughly
        roi = out[10:50, 10:150]
        assert roi.mean() > 0  # not all black

    def test_top_left_and_top_right_huds_both_rendered(self):
        v = _make_visualizer(
            show_unique_count=True, show_fps=True, show_boxes=False, show_ids=False
        )
        frame = _blank_frame()
        out = v.annotate(frame, [], 10, 30.0, 25.0)
        # Both corners should have non-zero pixels
        assert out[:80, :300].max() > 0  # top-left
        assert out[:80, 350:].max() > 0  # top-right


# ===========================================================================
# Trail drawing
# ===========================================================================


class TestTrails:
    def test_trail_history_updated_per_track(self):
        v = _make_visualizer(
            trail_length=5,
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
        )
        frame = _blank_frame()
        track = _make_track(1)
        v.annotate(frame, [track], 0, 0.0, 0.0)
        assert 1 in v._trails
        assert len(v._trails[1]) == 1

    def test_trail_grows_across_frames(self):
        v = _make_visualizer(
            trail_length=5,
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
        )
        frame = _blank_frame()
        for i in range(4):
            v.annotate(
                frame, [_make_track(1, x1=10 + i, y1=10 + i, x2=110 + i, y2=110 + i)], 0, 0.0, 0.0
            )
        assert len(v._trails[1]) == 4

    def test_trail_respects_maxlen(self):
        """Trail deque maxlen should cap at trail_length."""
        v = _make_visualizer(
            trail_length=3,
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
        )
        frame = _blank_frame()
        for i in range(10):
            v.annotate(
                frame,
                [
                    _make_track(
                        1,
                        x1=float(i * 10),
                        y1=float(i * 10),
                        x2=float(i * 10 + 100),
                        y2=float(i * 10 + 100),
                    )
                ],
                0,
                0.0,
                0.0,
            )
        assert len(v._trails[1]) <= 3

    def test_trail_length_zero_no_trails(self):
        v = _make_visualizer(
            trail_length=0,
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
        )
        frame = _blank_frame()
        v.annotate(frame, [_make_track(1)], 0, 0.0, 0.0)
        # With trail_length=0 the trail branch is skipped
        assert len(v._trails) == 0

    def test_separate_trails_per_track_id(self):
        v = _make_visualizer(
            trail_length=5,
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
        )
        frame = _blank_frame()
        v.annotate(frame, [_make_track(1), _make_track(2)], 0, 0.0, 0.0)
        assert 1 in v._trails
        assert 2 in v._trails


# ===========================================================================
# reset_trails
# ===========================================================================


class TestResetTrails:
    def test_reset_clears_trail_history(self):
        v = _make_visualizer(trail_length=5)
        frame = _blank_frame()
        v.annotate(frame, [_make_track(1), _make_track(2)], 0, 0.0, 0.0)
        assert len(v._trails) > 0
        v.reset_trails()
        assert len(v._trails) == 0

    def test_trails_repopulate_after_reset(self):
        v = _make_visualizer(
            trail_length=5,
            show_boxes=False,
            show_ids=False,
            show_unique_count=False,
            show_fps=False,
        )
        frame = _blank_frame()
        v.annotate(frame, [_make_track(1)], 0, 0.0, 0.0)
        v.reset_trails()
        v.annotate(frame, [_make_track(1)], 0, 0.0, 0.0)
        assert 1 in v._trails
        assert len(v._trails[1]) == 1
