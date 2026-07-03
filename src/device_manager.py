"""
Device Manager for VisionTrack.

Responsible for detecting and selecting the best available compute device
(CUDA → MPS → CPU) with a graceful fallback chain.

The selected device string is passed directly to Ultralytics YOLO and PyTorch
so all model inference runs on the appropriate hardware.

Usage:
    from src.device_manager import DeviceManager

    dm = DeviceManager(preferred="cuda", fallback="cpu")
    device = dm.device          # e.g. "cuda:0"
    print(dm.summary())
"""

import torch

from loggers import get_logger

log = get_logger(__name__)


class DeviceManager:
    """Detects and selects the best available compute device.

    Priority order: CUDA → MPS → CPU.

    The ``preferred`` parameter lets callers (or the CLI) request a specific
    device; if unavailable, the manager falls back to the next in the chain
    rather than raising an error, unless the fallback is also unavailable.

    Attributes:
        device: The resolved device string ready for use with PyTorch /
                Ultralytics (e.g. ``"cuda:0"``, ``"mps"``, ``"cpu"``).

    Args:
        preferred: Desired device — ``"cuda"``, ``"mps"``, or ``"cpu"``.
                   ``None`` triggers full auto-detection.
        fallback:  Device to use when ``preferred`` is unavailable.
                   Defaults to ``"cpu"`` which is always available.
    """

    def __init__(
        self,
        preferred: str | None = None,
        fallback: str = "cpu",
    ) -> None:
        self._preferred = (preferred or "auto").lower().strip()
        self._fallback = fallback.lower().strip()
        self.device: str = self._resolve()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def is_cuda(self) -> bool:
        """True when the resolved device is a CUDA GPU."""
        return self.device.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        """True when the resolved device is Apple MPS."""
        return self.device == "mps"

    @property
    def is_cpu(self) -> bool:
        """True when the resolved device is CPU."""
        return self.device == "cpu"

    def summary(self) -> str:
        """Return a human-readable summary of the selected device.

        Returns:
            Multi-line string with device name and, for CUDA, GPU details.
        """
        lines = [f"Compute device : {self.device}"]

        if self.is_cuda:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total_mem = torch.cuda.get_device_properties(idx).total_memory
            lines.append(f"GPU name       : {name}")
            lines.append(f"VRAM           : {total_mem / 1024 ** 3:.1f} GB")
        elif self.is_mps:
            lines.append("GPU name       : Apple Silicon (MPS)")
        else:
            lines.append("Note           : CPU mode — inference will be slower")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal resolution logic
    # ------------------------------------------------------------------

    def _resolve(self) -> str:
        """Walk the preference chain and return the first available device."""
        if self._preferred == "auto":
            return self._auto_detect()

        if self._preferred == "cuda":
            return self._try_cuda()

        if self._preferred == "mps":
            return self._try_mps()

        if self._preferred == "cpu":
            log.info("Device set to CPU (explicit config).")
            return "cpu"

        # Unknown preference — warn and auto-detect
        log.warning(
            "Unknown device preference '%s'. Running auto-detection.",
            self._preferred,
        )
        return self._auto_detect()

    def _auto_detect(self) -> str:
        """Return the best available device without a preference hint."""
        log.debug("Auto-detecting compute device …")

        if torch.cuda.is_available():
            device = self._try_cuda()
            if not device.startswith("cuda"):
                pass  # CUDA reported available but initialisation failed
            else:
                return device

        if self._mps_available():
            return self._try_mps()

        log.info("No GPU detected. Falling back to CPU.")
        return "cpu"

    def _try_cuda(self) -> str:
        """Attempt to initialise CUDA; return device string or fallback."""
        if not torch.cuda.is_available():
            log.warning(
                "CUDA requested but not available. Falling back to '%s'.",
                self._fallback,
            )
            return self._safe_fallback()

        try:
            # Force CUDA context initialisation to surface driver errors early
            _ = torch.zeros(1, device="cuda")
            device_idx = torch.cuda.current_device()
            device_str = f"cuda:{device_idx}"
            gpu_name = torch.cuda.get_device_name(device_idx)
            vram = torch.cuda.get_device_properties(device_idx).total_memory
            log.info(
                "CUDA device selected: %s — %s (%.1f GB VRAM)",
                device_str,
                gpu_name,
                vram / 1024**3,
            )
            return device_str
        except Exception as exc:
            log.warning(
                "CUDA initialisation failed (%s). Falling back to '%s'.",
                exc,
                self._fallback,
            )
            return self._safe_fallback()

    def _try_mps(self) -> str:
        """Attempt to initialise Apple MPS; return device string or fallback."""
        if not self._mps_available():
            log.warning(
                "MPS requested but not available. Falling back to '%s'.",
                self._fallback,
            )
            return self._safe_fallback()

        try:
            _ = torch.zeros(1, device="mps")
            log.info("MPS device selected: Apple Silicon GPU.")
            return "mps"
        except Exception as exc:
            log.warning(
                "MPS initialisation failed (%s). Falling back to '%s'.",
                exc,
                self._fallback,
            )
            return self._safe_fallback()

    def _safe_fallback(self) -> str:
        """Return the fallback device, ensuring it is actually usable."""
        if self._fallback == "cpu":
            log.info("Using CPU as fallback device.")
            return "cpu"

        # Recursively resolve the fallback (e.g. fallback="mps")
        original_preferred = self._preferred
        self._preferred = self._fallback
        self._fallback = "cpu"
        result = self._resolve()
        # Restore state
        self._preferred = original_preferred
        return result

    @staticmethod
    def _mps_available() -> bool:
        """Return True when Apple MPS is available and built into this PyTorch."""
        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
