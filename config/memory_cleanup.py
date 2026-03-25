from jtop import jtop
import gc
import torch
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



def aggressive_cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    try:
        with jtop() as jetson:
            if jetson.ok():
                jetson.memory.clear_cache()
    except Exception as e:
        print(f"jtop cache clear failed: {e}")