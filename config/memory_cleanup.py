import gc
import ctypes
import os
import subprocess
import threading
from typing import Any, Dict
try:
    from jtop import jtop
except Exception:  # pragma: no cover - runtime dependency may be absent
    jtop = None

try:
    libc = ctypes.CDLL("libc.so.6")
except OSError:
    libc = None


def _mem_available_mb() -> float | None:
    """Read MemAvailable from /proc/meminfo when available."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None


def _drop_linux_page_cache() -> tuple[bool, str]:
    """Attempt to drop Linux filesystem page caches.

    Requires root privileges. Returns (success, message).
    """
    path = "/proc/sys/vm/drop_caches"
    if not os.path.exists(path):
        return False, "drop_caches interface unavailable on this system"

    try:
        os.sync()
    except Exception:
        # Continue even if sync fails; cache drop may still work.
        pass

    try:
        with open(path, "w", encoding="utf-8") as fh:
            # 3 => pagecache + dentries + inodes
            fh.write("3\n")
        return True, "linux page cache dropped"
    except PermissionError:
        return False, "insufficient privileges to write /proc/sys/vm/drop_caches"
    except Exception as e:
        return False, f"failed to drop linux page cache: {e}"


def _drop_linux_page_cache_with_sudo() -> tuple[bool, str]:
    """Attempt page-cache drop via non-interactive sudo.

    This requires sudoers to permit the command without a password.
    """
    cmd = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, "sudo not found on system"
    except Exception as e:
        return False, f"sudo cache drop failed to run: {e}"

    if completed.returncode == 0:
        return True, "linux page cache dropped via sudo"

    err = (completed.stderr or completed.stdout or "").strip()
    if not err:
        err = f"exit code {completed.returncode}"
    return False, f"sudo cache drop failed: {err}"


def _drop_linux_page_cache_best_effort() -> tuple[bool, str]:
    """Drop page cache directly, then optionally retry via sudo.

    Set LIFELENS_DROP_OS_PAGE_CACHE_WITH_SUDO=1 to allow sudo fallback.
    """
    ok, msg = _drop_linux_page_cache()
    if ok:
        return ok, msg

    with_sudo = os.getenv("LIFELENS_DROP_OS_PAGE_CACHE_WITH_SUDO", "0").strip().lower() in {
        "1", "true", "t", "yes", "y", "on"
    }
    if not with_sudo:
        return ok, msg

    sudo_ok, sudo_msg = _drop_linux_page_cache_with_sudo()
    if sudo_ok:
        return True, sudo_msg
    return False, f"{msg}; {sudo_msg}"


def _clear_cuda_cache_if_available() -> bool:
    """Best-effort release of CUDA caching allocator."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except Exception:
        return False
    return False


def cleanup_memory(
    *objs,
    clear_cuda_cache: bool = True,
    clear_jtop: bool = False,
    drop_linux_page_cache: bool = False,
) -> Dict[str, Any]:
    """Best-effort memory cleanup for process + optional system-level caches.

    Optionally pass objects you want to drop references to; they will be
    deleted before triggering collection and allocator trim.

    Args:
        clear_cuda_cache: If True, release CUDA allocator cache when torch+CUDA exist.
        clear_jtop: If True, attempt jtop cache clear on Jetson.
        drop_linux_page_cache: If True, attempt /proc/sys/vm/drop_caches write.

    Returns:
        Dict with status and before/after MemAvailable when readable.
    """
    before_mb = _mem_available_mb()

    for obj in objs:
        try:
            del obj
        except Exception:
            # Ignore deletion issues; GC will handle remaining references.
            pass

    collected = gc.collect()

    cuda_cleared = False
    if clear_cuda_cache:
        cuda_cleared = _clear_cuda_cache_if_available()

    malloc_trimmed = False
    if libc is not None:
        try:
            libc.malloc_trim(0)
            malloc_trimmed = True
        except Exception:
            # Some libc builds may not support malloc_trim; ignore gracefully.
            pass

    jtop_cleared = False
    if clear_jtop:
        jtop_cleared = bool(clear_jtop_cache())

    os_cache_dropped = False
    os_cache_message = "linux page cache drop not requested"
    if drop_linux_page_cache:
        os_cache_dropped, os_cache_message = _drop_linux_page_cache_best_effort()

    after_mb = _mem_available_mb()
    reclaimed_mb = None
    if before_mb is not None and after_mb is not None:
        reclaimed_mb = after_mb - before_mb

    return {
        "gc_collected": collected,
        "cuda_cache_cleared": cuda_cleared,
        "malloc_trimmed": malloc_trimmed,
        "jtop_cache_cleared": jtop_cleared,
        "linux_page_cache_dropped": os_cache_dropped,
        "linux_page_cache_message": os_cache_message,
        "mem_available_before_mb": before_mb,
        "mem_available_after_mb": after_mb,
        "mem_reclaimed_mb": reclaimed_mb,
    }

def clear_jtop_cache():
    """Best-effort Jetson cache clear with Linux fallback.

    Behavior:
      1) Try known jtop cache-clear APIs.
      2) If unavailable/failed and LIFELENS_JTOP_OS_CACHE_FALLBACK=1 (default),
         try dropping Linux page cache.
    """
    try:
        jtop_timeout_s = float(os.getenv("LIFELENS_JTOP_CLEAR_TIMEOUT_S", "1.5"))
    except Exception:
        jtop_timeout_s = 1.5

    fallback_to_os_cache = os.getenv("LIFELENS_JTOP_OS_CACHE_FALLBACK", "1").strip().lower() in {
        "1", "true", "t", "yes", "y", "on"
    }

    jtop_success = False

    if jtop is None:
        print("jtop not available.")
    else:
        done = threading.Event()
        worker_error: list[str] = []

        def _run_jtop_clear() -> None:
            nonlocal jtop_success
            try:
                with jtop() as jetson:
                    if jetson.ok():
                        clear_candidates = [
                            getattr(jetson, "clear_cache", None),
                            getattr(getattr(jetson, "memory", None), "clear_cache", None),
                        ]
                        for method in clear_candidates:
                            if callable(method):
                                method()
                                jtop_success = True
                                print("jtop cache cleared successfully.")
                                break

                        if not jtop_success:
                            print(
                                "jtop cache-clear API not available in this jtop version."
                            )
                    else:
                        print("Could not connect to jtop.")
            except Exception as e:
                worker_error.append(str(e))
            finally:
                done.set()

        t = threading.Thread(target=_run_jtop_clear, daemon=True)
        t.start()
        if not done.wait(timeout=max(0.1, jtop_timeout_s)):
            print(
                f"jtop cache clear timed out after {jtop_timeout_s:.1f}s; "
                "skipping direct jtop call."
            )
        elif worker_error:
            print(f"jtop cache clear failed: {worker_error[0]}")

    if jtop_success:
        return True

    if not fallback_to_os_cache:
        return False

    dropped, message = _drop_linux_page_cache_best_effort()
    print(f"OS cache fallback: {message}")
    return dropped
