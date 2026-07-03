"""
Unit tests for src/device_manager.py

Uses monkeypatching to control torch.cuda.is_available() and
torch.backends.mps availability so tests run identically on any
machine regardless of actual hardware.

Covers:
- Construction: preferred=None triggers auto, preferred=cpu resolves immediately
- Properties: is_cuda, is_mps, is_cpu — mutually exclusive
- CPU explicit: always resolves to "cpu" without touching torch
- Unknown preference: warns and falls back to auto-detect
- CUDA path: available → succeeds; not available → fallback; init error → fallback
- MPS path: available → succeeds; not available → fallback; init error → fallback
- Auto-detect: CUDA first, MPS second, CPU last
- _safe_fallback: cpu fallback; non-cpu fallback resolved recursively
- summary(): contains device name; CPU note; MPS note; CUDA info
- _mps_available(): returns bool based on torch.backends.mps attributes
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.device_manager import DeviceManager

# ===========================================================================
# Helpers
# ===========================================================================


def _no_cuda_no_mps(monkeypatch) -> None:
    """Disable both CUDA and MPS so CPU is the only option."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: False)
    )


def _only_mps(monkeypatch) -> None:
    """Disable CUDA, enable MPS."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: True)
    )
    # Make torch.zeros(1, device="mps") succeed

    monkeypatch.setattr("torch.zeros", lambda *a, **kw: MagicMock())


def _only_cuda(monkeypatch) -> None:
    """Enable CUDA, disable MPS."""

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.zeros", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda idx: "Test GPU")
    props = MagicMock()
    props.total_memory = 8 * 1024**3
    monkeypatch.setattr("torch.cuda.get_device_properties", lambda idx: props)
    monkeypatch.setattr(
        "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: False)
    )


# ===========================================================================
# CPU explicit
# ===========================================================================


class TestCpuExplicit:
    def test_preferred_cpu_resolves_to_cpu(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="cpu")
        assert dm.device == "cpu"

    def test_is_cpu_true(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="cpu")
        assert dm.is_cpu is True
        assert dm.is_cuda is False
        assert dm.is_mps is False

    def test_none_preferred_auto_detects_cpu_when_no_gpu(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred=None)
        assert dm.device == "cpu"

    def test_auto_preferred_also_detects_cpu(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="auto")
        assert dm.device == "cpu"


# ===========================================================================
# MPS path
# ===========================================================================


class TestMpsPath:
    def test_preferred_mps_resolves_to_mps(self, monkeypatch):
        _only_mps(monkeypatch)
        dm = DeviceManager(preferred="mps")
        assert dm.device == "mps"

    def test_is_mps_true(self, monkeypatch):
        _only_mps(monkeypatch)
        dm = DeviceManager(preferred="mps")
        assert dm.is_mps is True
        assert dm.is_cpu is False
        assert dm.is_cuda is False

    def test_mps_unavailable_falls_back_to_cpu(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)  # MPS disabled
        dm = DeviceManager(preferred="mps", fallback="cpu")
        assert dm.device == "cpu"

    def test_mps_init_error_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr(
            "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: True)
        )

        monkeypatch.setattr(
            "torch.zeros", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("mps init failed"))
        )
        dm = DeviceManager(preferred="mps", fallback="cpu")
        assert dm.device == "cpu"

    def test_auto_detect_picks_mps_when_no_cuda(self, monkeypatch):
        _only_mps(monkeypatch)
        dm = DeviceManager(preferred=None)
        assert dm.device == "mps"


# ===========================================================================
# CUDA path
# ===========================================================================


class TestCudaPath:
    def test_preferred_cuda_resolves_to_cuda(self, monkeypatch):
        _only_cuda(monkeypatch)
        dm = DeviceManager(preferred="cuda")
        assert dm.device.startswith("cuda")

    def test_is_cuda_true(self, monkeypatch):
        _only_cuda(monkeypatch)
        dm = DeviceManager(preferred="cuda")
        assert dm.is_cuda is True
        assert dm.is_cpu is False
        assert dm.is_mps is False

    def test_cuda_device_string_format(self, monkeypatch):
        _only_cuda(monkeypatch)
        dm = DeviceManager(preferred="cuda")
        assert dm.device == "cuda:0"

    def test_cuda_unavailable_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        monkeypatch.setattr(
            "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: False)
        )
        dm = DeviceManager(preferred="cuda", fallback="cpu")
        assert dm.device == "cpu"

    def test_cuda_init_error_falls_back_to_cpu(self, monkeypatch):

        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            "torch.zeros", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("cuda init failed"))
        )
        monkeypatch.setattr(
            "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: False)
        )
        dm = DeviceManager(preferred="cuda", fallback="cpu")
        assert dm.device == "cpu"

    def test_auto_detect_picks_cuda_first(self, monkeypatch):
        _only_cuda(monkeypatch)
        # Also make MPS available — CUDA should win
        monkeypatch.setattr(
            "src.device_manager.DeviceManager._mps_available", staticmethod(lambda: True)
        )
        dm = DeviceManager(preferred=None)
        assert dm.device.startswith("cuda")


# ===========================================================================
# Unknown preference
# ===========================================================================


class TestUnknownPreference:
    def test_unknown_preference_falls_back_to_auto(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="tpu")
        # Unknown → auto-detect → CPU (no GPU available)
        assert dm.device == "cpu"

    def test_unknown_preference_auto_detects_mps(self, monkeypatch):
        _only_mps(monkeypatch)
        dm = DeviceManager(preferred="tpu")
        assert dm.device == "mps"


# ===========================================================================
# summary()
# ===========================================================================


class TestSummary:
    def test_cpu_summary_contains_device_and_note(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="cpu")
        s = dm.summary()
        assert "cpu" in s
        assert "Compute device" in s
        assert "CPU mode" in s

    def test_mps_summary_contains_apple_silicon(self, monkeypatch):
        _only_mps(monkeypatch)
        dm = DeviceManager(preferred="mps")
        s = dm.summary()
        assert "mps" in s
        assert "Apple Silicon" in s

    def test_cuda_summary_contains_gpu_name(self, monkeypatch):
        _only_cuda(monkeypatch)
        dm = DeviceManager(preferred="cuda")
        s = dm.summary()
        assert "cuda" in s
        assert "Test GPU" in s
        assert "VRAM" in s

    def test_summary_returns_string(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="cpu")
        assert isinstance(dm.summary(), str)

    def test_summary_multiline_for_cuda(self, monkeypatch):
        _only_cuda(monkeypatch)
        dm = DeviceManager(preferred="cuda")
        lines = dm.summary().splitlines()
        assert len(lines) >= 3


# ===========================================================================
# _mps_available static method
# ===========================================================================


class TestMpsAvailable:
    def test_returns_false_when_torch_has_no_mps(self, monkeypatch):
        import torch

        # Simulate a PyTorch build without MPS support
        monkeypatch.setattr(torch, "backends", MagicMock(spec=[]))
        assert DeviceManager._mps_available() is False

    def test_returns_false_when_mps_not_built(self, monkeypatch):
        import torch

        mock_backends = MagicMock()
        mock_backends.mps.is_available.return_value = True
        mock_backends.mps.is_built.return_value = False
        monkeypatch.setattr(torch, "backends", mock_backends)
        assert DeviceManager._mps_available() is False

    def test_returns_false_when_mps_not_available(self, monkeypatch):
        import torch

        mock_backends = MagicMock()
        mock_backends.mps.is_available.return_value = False
        mock_backends.mps.is_built.return_value = True
        monkeypatch.setattr(torch, "backends", mock_backends)
        assert DeviceManager._mps_available() is False

    def test_returns_true_when_fully_available(self, monkeypatch):
        import torch

        mock_backends = MagicMock()
        mock_backends.mps.is_available.return_value = True
        mock_backends.mps.is_built.return_value = True
        monkeypatch.setattr(torch, "backends", mock_backends)
        assert DeviceManager._mps_available() is True


# ===========================================================================
# Properties are mutually exclusive
# ===========================================================================


class TestProperties:
    @pytest.mark.parametrize(
        "preferred,expected_is_cpu,expected_is_cuda,expected_is_mps",
        [
            ("cpu", True, False, False),
        ],
    )
    def test_cpu_properties(
        self, monkeypatch, preferred, expected_is_cpu, expected_is_cuda, expected_is_mps
    ):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred=preferred)
        assert dm.is_cpu == expected_is_cpu
        assert dm.is_cuda == expected_is_cuda
        assert dm.is_mps == expected_is_mps

    def test_exactly_one_property_true_on_cpu(self, monkeypatch):
        _no_cuda_no_mps(monkeypatch)
        dm = DeviceManager(preferred="cpu")
        assert sum([dm.is_cpu, dm.is_cuda, dm.is_mps]) == 1

    def test_exactly_one_property_true_on_mps(self, monkeypatch):
        _only_mps(monkeypatch)
        dm = DeviceManager(preferred="mps")
        assert sum([dm.is_cpu, dm.is_cuda, dm.is_mps]) == 1

    def test_exactly_one_property_true_on_cuda(self, monkeypatch):
        _only_cuda(monkeypatch)
        dm = DeviceManager(preferred="cuda")
        assert sum([dm.is_cpu, dm.is_cuda, dm.is_mps]) == 1
