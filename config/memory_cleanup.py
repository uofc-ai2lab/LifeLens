import gc
import ctypes

try:
    libc = ctypes.CDLL("libc.so.6")
except OSError:
    libc = None


def cleanup_memory(*objs):
    """Best-effort heap cleanup using GC + malloc_trim on Linux.

    Optionally pass objects you want to drop references to; they will be
    deleted before triggering collection and allocator trim.
    """
    for obj in objs:
        try:
            del obj
        except Exception:
            # Ignore deletion issues; GC will handle remaining references.
            pass

    gc.collect()

    if libc is None:
        return

    try:
        libc.malloc_trim(0)
    except Exception:
        # Some libc builds may not support malloc_trim; ignore gracefully.
        pass
