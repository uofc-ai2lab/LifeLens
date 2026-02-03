#!/usr/bin/env python3
"""
tune_mic_array.py — Get / set DSP parameters on the Seeed ReSpeaker
                    USB Mic Array v2 / v3.

Uses the official USB control protocol documented at:
  https://github.com/respeaker/usb_4_mic_array/wiki/USB-Control-Protocol

Protocol recap
--------------
ctrl_transfer(CTRL_OUT|VENDOR|DEVICE, bRequest=0, wValue=command, wIndex=id, data)

  command byte layout:
      bit 7   — 1 = read,  0 = write
      bit 6   — 1 = int,   0 = float
      bits 5-0 — cmd  (the "offset" in the parameter table)

  id   — the parameter group id (18, 19, 21 …)
  data — 4 bytes, little-endian int32 or float32

Requirements
------------
    sudo pip install pyusb --break-system-packages
    sudo apt install libusb-1-0                        # usually already there

Usage
-----
    sudo python3 tune_mic_array.py                    # apply SETTINGS below
    sudo python3 tune_mic_array.py --read             # dump every parameter
    sudo python3 tune_mic_array.py --list             # list all known params
    sudo python3 tune_mic_array.py GET  ECHOONOFF     # read one param
    sudo python3 tune_mic_array.py SET  ECHOONOFF  1  # write one param
"""

import sys
import struct
import usb.core
import usb.util

# ---------------------------------------------------------------------------
# ← EDIT THESE VALUES to taste before running in batch mode (no args).
# ---------------------------------------------------------------------------
SETTINGS = {
    # --- Echo cancellation --------------------------------------------------
    "AECFREEZEONOFF":    0,      # 0 = keep AEC adapting
    "ECHOONOFF":         1,      # 1 = enable echo suppression
    "GAMMA_E":           2.3,    # echo over-subtraction        (0–3)
    "GAMMA_ETAIL":       2.3,    # echo-tail over-subtraction   (0–3)
    "GAMMA_ENL":         3.0,    # non-linear echo              (0–5)
    "NLATTENONOFF":      1,      # 1 = enable NL echo atten.
    "NLAEC_MODE":        2,      # 0=off  1=phase1  2=phase2
    "TRANSIENTONOFF":    1,      # 1 = transient echo suppress.

    # --- Noise suppression --------------------------------------------------
    "STATNOISEONOFF":    1,      # 1 = stationary noise suppress.
    "GAMMA_NS":          2.0,    # stationary over-subtraction  (0–3)
    "MIN_NS":            0.15,   # stationary gain-floor   (linear; ≈ -16 dB)
    "NONSTATNOISEONOFF": 1,      # 1 = non-stationary noise suppress.
    "GAMMA_NN":          2.0,    # non-stat. over-subtraction   (0–3)
    "MIN_NN":            0.3,    # non-stat. gain-floor    (linear; ≈ -10 dB)

    # --- High-pass filter ---------------------------------------------------
    "HPFONOFF":          1,      # 0=off  1=70 Hz  2=125 Hz  3=180 Hz

    # --- Automatic Gain Control ---------------------------------------------
    "AGCONOFF":          1,      # 1 = enable AGC
    "AGCDESIREDLEVEL":   0.005,  # target power (linear; ≈ -23 dBov)
}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parameter table — matches the official wiki exactly.
# Layout:  name → (id, cmd, type, max, min, rw)
# ---------------------------------------------------------------------------
PARAMETERS = {
    "AECFREEZEONOFF":       (18,  7, "int",    1,      0,    "rw"),
    "AECNORM":              (18, 19, "float",  16,     0.25, "rw"),
    "AECPATHCHANGE":        (18, 25, "int",    1,      0,    "ro"),
    "RT60":                 (18, 26, "float",  0.9,    0.25, "ro"),
    "HPFONOFF":             (18, 27, "int",    3,      0,    "rw"),
    "RT60ONOFF":            (18, 28, "int",    1,      0,    "rw"),
    "AECSILENCELEVEL":      (18, 30, "float",  1,      1e-9, "rw"),
    "AECSILENCEMODE":       (18, 31, "int",    1,      0,    "ro"),
    "AGCONOFF":             (19,  0, "int",    1,      0,    "rw"),
    "AGCMAXGAIN":           (19,  1, "float",  1000,   1,    "rw"),
    "AGCDESIREDLEVEL":      (19,  2, "float",  0.99,   1e-8, "rw"),
    "AGCGAIN":              (19,  3, "float",  1000,   1,    "ro"),
    "AGCTIME":              (19,  4, "float",  1,      0.1,  "rw"),
    "CNIONOFF":             (19,  5, "int",    1,      0,    "rw"),
    "FREEZEONOFF":          (19,  6, "int",    1,      0,    "rw"),
    "STATNOISEONOFF":       (19,  8, "int",    1,      0,    "rw"),
    "GAMMA_NS":             (19,  9, "float",  3,      0,    "rw"),
    "MIN_NS":               (19, 10, "float",  1,      0,    "rw"),
    "NONSTATNOISEONOFF":    (19, 11, "int",    1,      0,    "rw"),
    "GAMMA_NN":             (19, 12, "float",  3,      0,    "rw"),
    "MIN_NN":               (19, 13, "float",  1,      0,    "rw"),
    "ECHOONOFF":            (19, 14, "int",    1,      0,    "rw"),
    "GAMMA_E":              (19, 15, "float",  3,      0,    "rw"),
    "GAMMA_ETAIL":          (19, 16, "float",  3,      0,    "rw"),
    "GAMMA_ENL":            (19, 17, "float",  5,      0,    "rw"),
    "NLATTENONOFF":         (19, 18, "int",    1,      0,    "rw"),
    "NLAEC_MODE":           (19, 20, "int",    2,      0,    "rw"),
    "SPEECHDETECTED":       (19, 22, "int",    1,      0,    "ro"),
    "FSBUPDATED":           (19, 23, "int",    1,      0,    "ro"),
    "FSBPATHCHANGE":        (19, 24, "int",    1,      0,    "ro"),
    "TRANSIENTONOFF":       (19, 29, "int",    1,      0,    "rw"),
    "VOICEACTIVITY":        (19, 32, "int",    1,      0,    "ro"),
    "STATNOISEONOFF_SR":    (19, 33, "int",    1,      0,    "rw"),
    "NONSTATNOISEONOFF_SR": (19, 34, "int",    1,      0,    "rw"),
    "GAMMA_NS_SR":          (19, 35, "float",  3,      0,    "rw"),
    "GAMMA_NN_SR":          (19, 36, "float",  3,      0,    "rw"),
    "MIN_NS_SR":            (19, 37, "float",  1,      0,    "rw"),
    "MIN_NN_SR":            (19, 38, "float",  1,      0,    "rw"),
    "GAMMAVAD_SR":          (19, 39, "float",  1000,   0,    "rw"),
    "DOAANGLE":             (21,  0, "int",    359,    0,    "ro"),
}

# Seeed vendor ID shared across mic-array products
VENDOR_ID   = 0x2886
PRODUCT_IDS = [0x0018, 0x0019]   # covers v2 and v3

TIMEOUT = 2000  # ms


# ---------------------------------------------------------------------------
# USB helpers
# ---------------------------------------------------------------------------
def find_device():
    """Return the first ReSpeaker mic array found, or exit helpfully."""
    for pid in PRODUCT_IDS:
        dev = usb.core.find(idVendor=VENDOR_ID, idProduct=pid)
        if dev:
            return dev
    print("ERROR — no ReSpeaker mic array found on USB.")
    print("  • Is it plugged in?")
    print("  • Are you running with sudo?")
    sys.exit(1)


def _build_command(param_type, read=False):
    """
    Build the 'command' byte.
        bit 7 — 1 = read, 0 = write
        bit 6 — 1 = int,  0 = float
        bits 5-0 — cmd (filled in by caller via OR)
    """
    cmd = 0
    if read:
        cmd |= (1 << 7)
    if param_type == "int":
        cmd |= (1 << 6)
    return cmd


# ---------------------------------------------------------------------------
# Public get / set
# ---------------------------------------------------------------------------
def get_param(dev, name):
    """Read a single parameter from the device. Returns (value, type_str)."""
    if name not in PARAMETERS:
        print(f"  ERROR  {name} — unknown parameter")
        return None
    pid, cmd_offset, ptype, _, _, _ = PARAMETERS[name]

    command = _build_command(ptype, read=True) | (cmd_offset & 0x3F)

    # Read transfer — send command, device replies with 4 bytes
    raw = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0,           # bRequest
        command,     # wValue  ← command byte
        pid,         # wIndex  ← id
        4,           # length
        TIMEOUT,
    )

    if ptype == "int":
        value = struct.unpack("<I", bytes(raw))[0]
    else:
        value = struct.unpack("<f", bytes(raw))[0]

    return value, ptype


def set_param(dev, name, value):
    """Write a single parameter to the device."""
    if name not in PARAMETERS:
        print(f"  SKIP  {name} — unknown parameter")
        return False
    pid, cmd_offset, ptype, pmax, pmin, rw = PARAMETERS[name]

    if rw == "ro":
        print(f"  SKIP  {name} — read-only")
        return False

    # Clamp
    value = max(pmin, min(pmax, value))

    command = _build_command(ptype, read=False) | (cmd_offset & 0x3F)

    if ptype == "int":
        data = struct.pack("<I", int(value))
    else:
        data = struct.pack("<f", float(value))

    dev.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0,           # bRequest
        command,     # wValue  ← command byte
        pid,         # wIndex  ← id
        data,        # payload
        TIMEOUT,
    )

    print(f"  SET   {name:30s} = {value}")
    return True


# ---------------------------------------------------------------------------
# CLI actions
# ---------------------------------------------------------------------------
def action_list():
    """Print every known parameter with its metadata (no USB needed)."""
    print(f"\n{'NAME':<28} {'ID':>3} {'CMD':>4} {'TYPE':<6} {'MAX':>8} {'MIN':>10} {'R/W'}")
    print("-" * 72)
    for name in sorted(PARAMETERS):
        pid, cmd, ptype, pmax, pmin, rw = PARAMETERS[name]
        print(f"{name:<28} {pid:>3} {cmd:>4} {ptype:<6} {pmax:>8} {pmin:>10} {rw}")


def action_read_all(dev):
    """Read and print every parameter from the device."""
    print(f"\n{'NAME':<28} {'TYPE':<6} {'VALUE':>12}   {'R/W'}")
    print("-" * 58)
    for name in sorted(PARAMETERS):
        try:
            value, ptype = get_param(dev, name)
            if ptype == "int":
                print(f"{name:<28} {ptype:<6} {value:>12d}   {PARAMETERS[name][5]}")
            else:
                print(f"{name:<28} {ptype:<6} {value:>12.4f}   {PARAMETERS[name][5]}")
        except Exception as e:
            print(f"{name:<28} {'?':<6} {'ERROR':>12}   {PARAMETERS[name][5]}   ({e})")


def action_apply(dev):
    """Write every entry in SETTINGS to the device."""
    print("\nApplying echo & noise-reduction settings…\n")
    for name, value in SETTINGS.items():
        set_param(dev, name, value)
    print("\nDone!  Run with --read to verify.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]

    # --list needs no device
    if "--list" in args:
        action_list()
        return

    dev = find_device()
    print(f"Found device: {dev}\n")

    # --read  →  dump everything
    if "--read" in args:
        action_read_all(dev)
        return

    # GET <name>
    if len(args) >= 2 and args[0].upper() == "GET":
        name = args[1].upper()
        result = get_param(dev, name)
        if result:
            value, ptype = result
            fmt = f"{value}" if ptype == "int" else f"{value:.4f}"
            print(f"{name}: {fmt}")
        return

    # SET <name> <value>
    if len(args) >= 3 and args[0].upper() == "SET":
        name  = args[1].upper()
        value = float(args[2])
        set_param(dev, name, value)
        return

    # No args  →  batch apply SETTINGS
    if not args:
        action_apply(dev)
        return

    # Fallback — unrecognised
    print(__doc__)


if __name__ == "__main__":
    main()