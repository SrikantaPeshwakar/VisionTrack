"""VisionTrack - Real-Time Multi-Object Tracking & Crowd Analytics."""

__version__ = "1.0.0"
__author__ = "VisionTrack"

# Configure third-party library verbosity and set up file+console handlers
# as soon as the src package is imported.
from loggers import configure_logging  # noqa: E402

configure_logging()
