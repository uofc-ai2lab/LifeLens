import subprocess, signal, time
from evdev import InputDevice, ecodes

EVENT_DEV = "/dev/input/event0" 
CMD = ["python3", "-m", "main"]

dev = InputDevice(EVENT_DEV)

proc = None
running = False
last = 0.0

def toggle():
    global proc, running
    if not running:
        proc = subprocess.Popen(CMD)
        running = True
        print("STARTED")
    else:
        # "ESC-like" clean stop: send SIGTERM (your app should handle it)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        running = False
        print("STOPPED")

print("Listening for power button...")
for e in dev.read_loop():
    if e.type == ecodes.EV_KEY and e.code == ecodes.KEY_POWER and e.value == 1:  # key down
        now = time.time()
        if now - last > 0.5:  # debounce
            toggle()
            last = now