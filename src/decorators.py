"""
Performance and resilience decorators for VisionTrack.

Provides reusable decorators that can be applied to any pipeline method:

    @measure_time
    def detect(self, frame): ...

    @measure_time(name="tracker.update")
    def update(self, frame, detections): ...

    @retry(max_attempts=3, delay=0.5)
    def load_model(self): ...

All timing results are logged via the 'visiontrack.decorators' logger so
they appear in the standard log file without extra setup.
"""

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar, overload

from loggers import get_logger

log = get_logger(__name__)

# Generic callable type for type-checker compatibility
F = TypeVar("F", bound=Callable[..., Any])


# ==============================================================================
# @measure_time
# ==============================================================================


@overload
def measure_time(func: F) -> F: ...


@overload
def measure_time(*, name: str | None = None, log_level: str = "DEBUG") -> Callable[[F], F]: ...


def measure_time(
    func: F | None = None,
    *,
    name: str | None = None,
    log_level: str = "DEBUG",
) -> Any:
    """Decorator that measures and logs a function's wall-clock execution time.

    Can be used with or without arguments:

        @measure_time
        def detect(self, frame): ...

        @measure_time(name="YOLO.detect", log_level="INFO")
        def detect(self, frame): ...

    The elapsed time (in milliseconds) is stored on the wrapper as
    ``wrapper.last_elapsed_ms`` so callers can read it without parsing logs:

        elapsed = detect.last_elapsed_ms  # float, milliseconds

    Args:
        func:      The function being decorated (positional, no-argument form).
        name:      Display name used in the log message.
                   Defaults to ``ClassName.method_name`` or ``function_name``.
        log_level: Python log level string for timing messages.
                   ``"DEBUG"`` (default) keeps hot-path logs quiet;
                   use ``"INFO"`` for one-off operations like model loading.

    Returns:
        Decorated function with identical signature.
    """

    def decorator(fn: F) -> F:
        _label = name or fn.__qualname__
        _level = getattr(log, log_level.lower(), log.debug)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                wrapper.last_elapsed_ms = elapsed_ms  # type: ignore[attr-defined]
                _level("%s completed in %.2f ms", _label, elapsed_ms)
            return result

        wrapper.last_elapsed_ms = 0.0  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    # Handle both @measure_time and @measure_time(...) usage
    if func is not None:
        return decorator(func)
    return decorator


# ==============================================================================
# @retry
# ==============================================================================


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    log_level: str = "WARNING",
) -> Callable[[F], F]:
    """Decorator that retries a function on failure with exponential back-off.

    Useful for operations that may transiently fail, such as model weight
    downloads or file I/O on network drives.

        @retry(max_attempts=3, delay=0.5, exceptions=(IOError, TimeoutError))
        def download_weights(url): ...

    Args:
        max_attempts: Total number of attempts before re-raising the exception.
                      Must be >= 1.
        delay:        Initial wait (seconds) between attempts.
        backoff:      Multiplier applied to ``delay`` after each failure.
                      ``backoff=2.0`` doubles the wait on each retry.
        exceptions:   Tuple of exception types that trigger a retry.
                      Any exception NOT in this tuple propagates immediately.
        log_level:    Python log level string for retry warning messages.

    Returns:
        Decorated function with identical signature.

    Raises:
        The last caught exception if all attempts are exhausted.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    def decorator(fn: F) -> F:
        _label = fn.__qualname__
        _level = getattr(log, log_level.lower(), log.warning)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        _level(
                            "%s failed (attempt %d/%d): %s — retrying in %.1fs …",
                            _label,
                            attempt,
                            max_attempts,
                            exc,
                            current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        _level(
                            "%s failed after %d attempts: %s",
                            _label,
                            max_attempts,
                            exc,
                        )

            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
