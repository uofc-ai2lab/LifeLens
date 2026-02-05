#!/usr/bin/env bash
set -euo pipefail

echo "[startup] Resetting camera: sudo systemctl restart nvargus-daemon"
if sudo systemctl restart nvargus-daemon; then
  echo "[startup] Camera reset complete"
else
  echo "[startup] WARNING: Camera reset failed"
  exit 1
fi

# Give nvargus-daemon time to fully initialize before camera pipeline tries to connect
# This prevents "Failed to create CaptureSession" errors
echo "[startup] Waiting for nvargus-daemon to stabilize (5 seconds)..."
sleep 5
echo "[startup] nvargus-daemon ready for connections"
