"""
GStreamer Camera Diagnostics Module

Provides diagnostic utilities for troubleshooting NVIDIA Jetson camera and GStreamer issues.
Helps identify buffer memory problems, daemon issues, and configuration problems.
"""

import subprocess
import os
import sys
from config.logger import Logger

log = Logger("[video][diag]")


def check_nvargus_daemon() -> bool:
    """
    Check if nvargus-daemon is running.
    
    Returns:
        bool: True if daemon is running, False otherwise
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nvargus-daemon"],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_running = result.returncode == 0
        status = "✓ Running" if is_running else "✗ Not running"
        log.info(f"nvargus-daemon: {status}")
        return is_running
    except Exception as e:
        log.error(f"Error checking nvargus-daemon: {e}")
        return False


def check_camera_device() -> bool:
    """
    Check if camera device is accessible.
    
    Returns:
        bool: True if device exists and is accessible, False otherwise
    """
    try:
        camera_exists = os.path.exists("/dev/video0")
        status = "✓ Exists" if camera_exists else "✗ Not found"
        log.info(f"/dev/video0: {status}")
        
        if camera_exists:
            camera_readable = os.access("/dev/video0", os.R_OK)
            readable_status = "✓ Readable" if camera_readable else "✗ Not readable"
            log.info(f"/dev/video0 permission: {readable_status}")
            return camera_readable
        return False
    except Exception as e:
        log.error(f"Error checking camera device: {e}")
        return False


def check_camera_in_use() -> bool:
    """
    Check if another process is using the camera.
    
    Returns:
        bool: True if camera is in use, False if available
    """
    try:
        result = subprocess.run(
            ["lsof", "/dev/video0"],
            capture_output=True,
            text=True,
            timeout=5
        )
        in_use = result.returncode == 0
        if in_use:
            log.warning("Camera in use by:")
            log.info(result.stdout.strip())
            return True
        else:
            log.info("Camera: ✓ Available (not in use)")
            return False
    except subprocess.CalledProcessError:
        log.info("Camera: ✓ Available (not in use)")
        return False
    except Exception as e:
        log.error(f"Error checking camera usage: {e}")
        return False


def check_gstreamer_plugin(plugin_name: str) -> bool:
    """
    Check if a specific GStreamer plugin is available.
    
    Args:
        plugin_name: Name of the plugin (e.g., 'nvarguscamerasrc', 'nvvidconv')
    
    Returns:
        bool: True if plugin is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", plugin_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        available = result.returncode == 0
        status = "✓ Available" if available else "✗ Not found"
        log.info(f"GStreamer plugin '{plugin_name}': {status}")
        return available
    except Exception as e:
        log.error(f"Error checking GStreamer plugin '{plugin_name}': {e}")
        return False


def check_gstreamer_plugins() -> dict:
    """
    Check availability of critical GStreamer plugins for camera pipeline.
    
    Returns:
        dict: Dictionary with plugin names as keys and availability as values
    """
    plugins = {
        "nvarguscamerasrc": False,
        "nvvidconv": False,
        "videoconvert": False,
        "appsink": False,
    }
    
    log.info("Checking GStreamer plugins...")
    for plugin in plugins.keys():
        plugins[plugin] = check_gstreamer_plugin(plugin)
    
    return plugins


def check_system_memory() -> tuple:
    """
    Check available system memory.
    
    Returns:
        tuple: (total_mb, available_mb, percent_used)
    """
    try:
        result = subprocess.run(
            ["free", "-m"],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.strip().split('\n')
        mem_info = lines[1].split()
        total = int(mem_info[1])
        available = int(mem_info[6])
        percent_used = ((total - available) / total) * 100
        
        log.info(f"System Memory: {available}MB / {total}MB available ({percent_used:.1f}% used)")
        return total, available, percent_used
    except Exception as e:
        log.error(f"Error checking system memory: {e}")
        return 0, 0, 0


def check_nvargus_daemon_logs() -> None:
    """
    Display recent nvargus-daemon log messages using dmesg.
    """
    try:
        result = subprocess.run(
            ["dmesg"],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.strip().split('\n')
        # Find lines mentioning nvargus or video
        relevant_lines = [l for l in lines if 'nvargus' in l.lower() or 'video' in l.lower()]
        
        if relevant_lines:
            log.info("Recent dmesg entries related to nvargus/video:")
            for line in relevant_lines[-10:]:
                log.info(f"       {line}")
        else:
            log.info("No recent nvargus/video entries in dmesg")
    except Exception as e:
        log.error(f"Error checking dmesg: {e}")


def run_full_diagnostics() -> bool:
    """
    Run comprehensive diagnostics for camera and GStreamer setup.
    
    Returns:
        bool: True if all checks pass, False if any check fails
    """
    log.header("GStreamer Camera Diagnostics")
    
    # Basic checks
    daemon_ok = check_nvargus_daemon()
    camera_ok = check_camera_device()
    camera_in_use = check_camera_in_use()
    
    # GStreamer plugin checks
    plugins = check_gstreamer_plugins()
    plugins_ok = all(plugins.values())
    
    # System resources
    total_mem, available_mem, mem_used = check_system_memory()
    
    # Logs
    check_nvargus_daemon_logs()
    
    # Summary
    log.info("Summary:")
    all_ok = daemon_ok and camera_ok and plugins_ok and not camera_in_use
    
    if all_ok:
        log.success("All checks passed. Camera should be ready to use.")
    else:
        log.warning("Some checks failed. Troubleshooting required:")
        if not daemon_ok:
            log.info("  - Run: sudo systemctl restart nvargus-daemon")
        if not camera_ok:
            log.info("  - Check camera connection and ribbon cable")
        if camera_in_use:
            log.info("  - Stop other processes using the camera")
        if not plugins_ok:
            log.info("  - Install missing GStreamer plugins")
            log.info("    For NVIDIA Jetson, check CUDA/JetPack installation")
    return all_ok


if __name__ == "__main__":
    success = run_full_diagnostics()
    sys.exit(0 if success else 1)
