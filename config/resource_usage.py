from __future__ import annotations

import csv
import os
import re
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import psutil
from config.audio_settings import USAGE_FILE_PATH

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_log_lock = threading.Lock()

_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
_POM_IN_RE = re.compile(r"POM_5V_IN\s+(\d+)(?:/(\d+))?")  # current/avg in mW (often)

def _parse_gpu_percent(line: str) -> Optional[float]:
    m = _GR3D_RE.search(line)
    return float(m.group(1)) if m else None

def _parse_power_in_mw(line: str) -> Optional[float]:
    m = _POM_IN_RE.search(line)
    if not m:
        return None
    current_mw = float(m.group(1))
    return current_mw

def _run_tegrastats(timeout_s: float = 1.5) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (gpu_percent, power_watts). Either may be None if not found.
    """
    try:
        proc = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        try:
            start = time.time()
            line = None
            while time.time() - start < timeout_s:
                if proc.stdout is None:
                    break
                line = proc.stdout.readline()
                if line:
                    break

            if not line:
                return None, None

            gpu_percent = _parse_gpu_percent(line)
            power_mw = _parse_power_in_mw(line)
            power_watts = (power_mw / 1000.0) if power_mw is not None else None

            return gpu_percent, power_watts

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()

    except FileNotFoundError:
        # Not a Jetson / tegrastats missing
        return None, None
    except Exception:
        return None, None

def _ensure_csv_header(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Always overwrite file at start
    with _log_lock, log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_iso",
            "cpu_percent",
            "mem_percent",
            "gpu_percent",
            "power_watts",
        ])
        
        
def _append_csv(log_path: Path, cpu: float, mem: float, gpu: Optional[float], power_w: Optional[float]) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with _log_lock, log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts,
            f"{cpu:.2f}",
            f"{mem:.2f}",
            "" if gpu is None else f"{gpu:.2f}",
            "" if power_w is None else f"{power_w:.3f}",
        ])

def _format_status(cpu: float, mem: float, gpu: Optional[float], power_w: Optional[float]) -> str:
    gpu_str = "  N/A " if gpu is None else f"{gpu:6.2f}%"
    pwr_str = "  N/A " if power_w is None else f"{power_w:6.3f}W"
    return f"CPU {cpu:6.2f}% | MEM {mem:6.2f}% | GPU {gpu_str} | PWR {pwr_str}"

def _monitor_loop(stop_event: threading.Event, interval: float, log_path: Path, show_stderr_line: bool) -> None:
    _ensure_csv_header(log_path)

    psutil.cpu_percent(interval=None)  # prime

    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        gpu, power_w = _run_tegrastats()
        _append_csv(log_path, cpu, mem, gpu, power_w)

        if show_stderr_line:
            msg = _format_status(cpu, mem, gpu, power_w)
            print("\r" + msg + " " * 10, end="\n", file=os.sys.stderr, flush=True)

        end_t = time.time() + interval
        while time.time() < end_t:
            if stop_event.is_set():
                break
            time.sleep(min(0.1, interval))

def start_monitoring(interval: float = 1.0, log_file: str | Path = USAGE_FILE_PATH, show_stderr_line: bool = False) -> None:
    global _monitor_thread, _stop_event

    if _monitor_thread is not None and _monitor_thread.is_alive():
        return

    _stop_event = threading.Event()
    log_path = Path(log_file)

    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(_stop_event, interval, log_path, show_stderr_line),
        daemon=True,
        name="ResourceMonitor",
    )
    _monitor_thread.start()

def stop_monitoring() -> None:
    global _monitor_thread, _stop_event

    if _stop_event is not None:
        _stop_event.set()

    if _monitor_thread is not None and _monitor_thread.is_alive():
        _monitor_thread.join(timeout=2.0)

    _monitor_thread = None
    _stop_event = None

if __name__ == "__main__":
    start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_monitoring()
        print("\nMonitoring stopped.", file=os.sys.stderr)
