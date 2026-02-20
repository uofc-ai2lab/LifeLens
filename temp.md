You can. The “power button” is just a **KEY_POWER** event. Ubuntu (systemd-logind) is currently grabbing it and showing the shutdown prompt. We’ll (1) tell Ubuntu to ignore it, then (2) run our own listener that toggles your program.

## 1) Tell Ubuntu to ignore the power button

Edit logind config:

```bash
sudo nano /etc/systemd/logind.conf
```

Find/uncomment (or add) these lines:

```ini
HandlePowerKey=ignore
HandlePowerKeyLongPress=poweroff
```

Apply:

```bash
sudo systemctl restart systemd-logind
```

(Short press won’t pop the shutdown dialog anymore. Long press can still force poweroff.)

---

## 2) Find which input device is the power key

Install tools:

```bash
sudo apt update
sudo apt install -y evtest
```

List devices:

```bash
sudo evtest
```

Pick the one named like **“gpio-keys”** (or anything that sounds like power button), note the `/dev/input/eventX`.

Quick confirm:

```bash
sudo evtest /dev/input/eventX
```

Press the power button once—you should see `KEY_POWER`.

---

## 3) Python listener to toggle your program

Install python lib:

```bash
pip3 install evdev
```

Create `power_toggle.py`:

```python
import subprocess, signal, time
from evdev import InputDevice, ecodes

EVENT_DEV = "/dev/input/eventX"  # <-- change this
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
```

Run it (needs root to read input devices):

```bash
sudo python3 power_toggle.py
```

---

## 4) Make it run on boot (optional but recommended)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/power-toggle.service
```

Paste:

```ini
[Unit]
Description=Toggle program with power button
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/sudo /usr/bin/python3 /home/$USER/power_toggle.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now power-toggle.service
```

---

### One critical note (so “stop” works how you want)

Your `main` program must exit cleanly when it receives **SIGTERM** (or you can implement your existing “Esc behavior” on SIGTERM). If you tell me what “Esc” currently does in your app, I’ll show you the 3–5 lines to make SIGTERM trigger the same shutdown path.
