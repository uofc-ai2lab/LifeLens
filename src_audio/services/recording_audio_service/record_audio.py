import subprocess, time, signal
from pathlib import Path
from src_audio.domain.constants import ARECORD_DEVICE, CHUNK_SECONDS

def record_one_chunk(output_dir: str | Path, stop_event) -> bool:
    """
    Records ONE chunk into AUDIO_CHUNKS_DIR (output_dir).
    Returns True if a chunk file was written.
    Does NOT move into processed. Worker will do that.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"recording_{ts}.wav"

    print(f"[audio] Recording chunk -> {out_path}")

    proc = subprocess.Popen(
        ["arecord", "-D", ARECORD_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "6", str(out_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    start = time.time()
    try:
        # Poll quickly so ENTER stops "immediately"
        while True:
            if stop_event.is_set():
                break
            if (time.time() - start) >= CHUNK_SECONDS:
                break
            time.sleep(0.1)

        proc.send_signal(signal.SIGINT)

        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise

    ok = out_path.exists() and out_path.stat().st_size > 0
    if ok:
        print(f"[audio] Chunk written -> {out_path}")
    else:
        print("[audio] Chunk missing/empty; skipping")

    return ok
