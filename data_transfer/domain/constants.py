import os
import socket
from config.logger import Logger

log = Logger("[data-transfer]")

def get_jetson_serial():
    path = '/proc/device-tree/serial-number'
    
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                # Stripping null terminators and whitespace
                serial = f.read().strip('\x00').strip()
                if serial:
                    return f"Jetson-{serial}"
        except Exception as e:
            log.error(f"Error reading serial number: {e}")
            pass 

    # Fallback for non-Jetson environments or permission issues
    return f"Local-{socket.gethostname()}"

DEVICE_ID = get_jetson_serial()

CONNECT_TIMEOUT = 10
PUBLISH_TIMEOUT = 5
QOS = 1
VALID_PIPELINES = {"audio", "video"}
