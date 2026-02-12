#!/usr/bin/env bash
set -euo pipefail

echo "[startup] Resetting camera: sudo systemctl restart nvargus-daemon"
if sudo systemctl restart nvargus-daemon; then
  echo "[startup] Camera reset complete"
else
  echo "[startup] WARNING: Camera reset failed"
  exit 1
fi

# Wait for nvargus-daemon to actually be running before proceeding
# This prevents "Failed to create CaptureSession" errors
echo "[startup] Waiting for nvargus-daemon to be active..."
until systemctl is-active --quiet nvargus-daemon; do
  sleep 1
done
echo "[startup] nvargus-daemon is active"

# Check if camera device exists and is accessible
echo "[startup] Checking camera device..."
if [ ! -e /dev/video0 ]; then
  echo "[startup] ERROR: /dev/video0 not found - check camera connection"
  exit 1
fi

if [ ! -r /dev/video0 ]; then
  echo "[startup] ERROR: /dev/video0 not readable - check permissions"
  exit 1
fi
echo "[startup] Camera device: ✓ /dev/video0 accessible"

# Check if camera is already in use
echo "[startup] Checking if camera is in use..."
if lsof /dev/video0 >/dev/null 2>&1; then
  echo "[startup] ERROR: Camera is in use by another process:"
  lsof /dev/video0
  exit 1
fi
echo "[startup] Camera: ✓ Available"

# Check critical GStreamer plugins
echo "[startup] Checking GStreamer plugins..."
PLUGINS=("nvarguscamerasrc" "nvvidconv" "videoconvert" "appsink")
for plugin in "${PLUGINS[@]}"; do
  if ! gst-inspect-1.0 "$plugin" >/dev/null 2>&1; then
    echo "[startup] ERROR: GStreamer plugin '$plugin' not found"
    exit 1
  fi
done
echo "[startup] GStreamer plugins: ✓ All available"

echo "[startup] All checks passed - camera ready"
