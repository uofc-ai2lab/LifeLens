from __future__ import annotations

import threading
from contextlib import contextmanager


_GPU_LOCK = threading.Lock()


@contextmanager
def gpu_exclusive(owner: str, logger=None):
    """Serialize GPU-heavy sections across audio/video threads."""
    if logger is not None:
        logger.debug(f"Waiting for GPU lock ({owner})")

    _GPU_LOCK.acquire()

    if logger is not None:
        logger.debug(f"GPU lock acquired ({owner})")

    try:
        yield
    finally:
        _GPU_LOCK.release()
        if logger is not None:
            logger.debug(f"GPU lock released ({owner})")
