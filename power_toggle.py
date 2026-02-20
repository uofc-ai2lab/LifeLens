import os
import signal
import subprocess
import sys
import time
from evdev import InputDevice, ecodes
import Jetson.GPIO as GPIO

EVENT_DEV = "/dev/input/event0"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CMD = [sys.executable, "-m", "main"]
LOG_DIR = os.path.join(PROJECT_ROOT, "data")
SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"lifelens_power_toggle_{SESSION_TS}.log")
LED_PIN = 7  # Jetson Board Pin 7

os.makedirs(LOG_DIR, exist_ok=True)
proc_log = open(LOG_FILE, "a", buffering=1)

dev = InputDevice(EVENT_DEV)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(LED_PIN, GPIO.OUT, initial=GPIO.LOW)

proc = None
running = False
last = 0.0


def set_led(is_on: bool):
    GPIO.output(LED_PIN, GPIO.HIGH if is_on else GPIO.LOW)

def toggle():
    global proc, proc_log, running
    if not running:
        try:
            proc = subprocess.Popen(
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

        time.sleep(0.3)
        exit_code = proc.poll()
        if exit_code is not None:
            set_led(False)
            print(f"FAILED_TO_STAY_RUNNING (exit={exit_code})")
            print(f"See logs: {LOG_FILE}")
            proc = None
            running = False
            return

        running = True
        set_led(True)
        print(f"STARTED pid={proc.pid} (logs: {LOG_FILE})")
    else:
        if proc is None or proc.poll() is not None:
            running = False
            set_led(False)
            proc = None
            print("ALREADY_STOPPED")
            return

        try:
            if proc.stdin is not None:
                proc.stdin.write("STOP\n")
                proc.stdin.flush()
        except Exception as e:
            print(f"STOP_COMMAND_FAILED: {e}")

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

        proc = None
        running = False
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

