#!/usr/bin/env bash
set -euo pipefail

# Jetson startup tasks: reset camera + start utilization map (tegrastats)

echo "[startup] Resetting camera: sudo systemctl restart nvargus-daemon"
if sudo systemctl restart nvargus-daemon; then
  echo "[startup] Camera reset complete"
else
  echo "[startup] WARNING: Camera reset failed. Restarting daemon..."
fi

# Give nvargus-daemon time to fully initialize before camera pipeline tries to connect
# This prevents "Failed to create CaptureSession" errors
echo "[startup] Waiting for nvargus-daemon to stabilize (5 seconds)..."
sleep 5
echo "[startup] nvargus-daemon ready for connections"

tegrastats_bin="$(command -v tegrastats || true)"
if [[ -z "$tegrastats_bin" ]]; then
  echo "[startup] WARNING: tegrastats not found; skipping utilization map"
  exit 0
fi

pid_file="/tmp/tegrastats.pid"

# If tegrastats already running (pid file exists and process alive), don't start another.
if [[ -f "$pid_file" ]]; then
  existing_pid="$(cat "$pid_file" 2>/dev/null || echo "")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[startup] tegrastats already running (pid=$existing_pid)"
    exit 0
  fi
fi

echo "[startup] Starting tegrastats (1s interval)"
"$tegrastats_bin" --interval 1000 > /tmp/tegrastats.log 2>&1 &
tegrastats_pid=$!

echo "$tegrastats_pid" > "$pid_file"
echo "[startup] tegrastats started (pid=$tegrastats_pid)"
