import threading
from typing import Optional
from data_transfer.mqtt_sender import MQTTSender

_sender: Optional[MQTTSender] = None
_lock = threading.Lock()


def init(device_id: str) -> MQTTSender:
    """
    Initialize the global MQTTSender singleton.

    Raises:
        RuntimeError: If already initialized.
    """
    global _sender
    with _lock:
        if _sender is not None:
            raise RuntimeError("MQTTSender already initialized")
        _sender = MQTTSender(device_id)
        return _sender


def get() -> MQTTSender:
    """
    Retrieve the global MQTTSender instance.

    Raises:
        RuntimeError: If not initialized.
    """
    if _sender is None:
        raise RuntimeError("MQTTSender not initialized. Call init() first.")
    return _sender

def reset():
    """
    Reset the global MQTTSender instance. For testing purposes only.
    """
    global _sender
    if _sender is not None:
        try:
            _sender.disconnect()
        except Exception:
            pass
    _sender = None