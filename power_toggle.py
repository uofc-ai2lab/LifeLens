import os, subprocess, sys, time
import threading
from evdev import InputDevice, ecodes, list_devices
import Jetson.GPIO as GPIO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CMD = [sys.executable, "-m", "main"]
LOG_DIR = os.path.join(PROJECT_ROOT, "data")
SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"lifelens_power_toggle_{SESSION_TS}.log")
LED_PIN = 7  # Jetson Board Pin 7

os.makedirs(LOG_DIR, exist_ok=True)
proc_log = open(LOG_FILE, "a", buffering=1)
EVENT_DEV = "/dev/input/event0"

def find_power_button():
    for path in list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_POWER in caps:
            return dev
    raise RuntimeError("No input device with KEY_POWER found")

dev = InputDevice(EVENT_DEV)
print(f"Using input device: {dev.path} ({dev.name})")

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

# ---------------- LED control (OFF / ON / BLINK) ----------------
def set_led(is_on: bool):
    GPIO.output(LED_PIN, GPIO.HIGH if is_on else GPIO.LOW)

def blink_led(times: int = 3, on_s: float = 0.15, off_s: float = 0.15):
    for _ in range(times):
        set_led(False)
        time.sleep(off_s)
        set_led(True)
        time.sleep(on_s)
        
# ---------------- process control ----------------
proc = None
proc_lock = threading.Lock()
last = 0.0


def _watch_process(p: subprocess.Popen):
    """Turn LED off if the managed process exits on its own."""
    try:
        exit_code = p.wait()
    except Exception as e:
        print(f"PROCESS_WATCHER_FAILED: {e}")
        return

    global proc
    with proc_lock:
        if proc is p:
            proc = None
            set_led(False)
            print(f"STOPPED (exit={exit_code})")

def toggle():
    global proc, proc_log

    with proc_lock:
        # Process is considered running only if the child exists and hasn't exited
        is_running = proc is not None and proc.poll() is None
    
    if not is_running:
        set_led(True)
        try:
            new_proc = subprocess.Popen(
                CMD,
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=proc_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            set_led(False)
            print(f"FAILED_TO_START: {e}")
            return

        with proc_lock:
            proc = new_proc

        time.sleep(0.3)
        exit_code = new_proc.poll()
        if exit_code is not None:
            set_led(False)
            print(f"FAILED_TO_STAY_RUNNING (exit={exit_code})")
            print(f"See logs: {LOG_FILE}")
            with proc_lock:
                if proc is new_proc:
                    proc = None
            return

        threading.Thread(target=_watch_process, args=(new_proc,), daemon=True).start()
        set_led(True)
        print(f"STARTED pid={new_proc.pid} (logs: {LOG_FILE})")
    else:
        # --- STOP: quick blink animation to acknowledge button press ---
        # LED ends ON (still running) while main gracefully shuts down.
        blink_led(times=3, on_s=0.12, off_s=0.12)
        set_led(True)
        with proc_lock:
            current_proc = proc
        
        try:
            if current_proc is not None and current_proc.stdin is not None:
                current_proc.stdin.write("STOP\n")
                current_proc.stdin.flush()
        except Exception as e:
            print(f"STOP_COMMAND_FAILED: {e}")

        try:
            # Block until the program has fully exited so a new
            # start cannot occur while it is still shutting down.
            if current_proc is not None:
                current_proc.wait()
        except Exception as e:
            print(f"WAIT_FAILED: {e}")

        with proc_lock:
            if proc is current_proc:
                proc = None
        set_led(False)
        print("STOPPED")

print("Listening for power button...")
print(f"Logging to: {LOG_FILE}")
try:
    for e in dev.read_loop():
        if e.type == ecodes.EV_KEY and e.code == ecodes.KEY_POWER and e.value == 1:  # key down
            now = time.time()
            if now - last > 0.5:  # debounce
                toggle()
                last = now
finally:
    if proc_log is not None:
        proc_log.close()
    set_led(False)
    GPIO.cleanup()

