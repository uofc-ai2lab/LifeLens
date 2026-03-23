import base64
import json
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional
from config.data_transfer_settings import BROKER, PORT, USERNAME, PASSWORD
from data_transfer.domain.constants import (
    CONNECT_TIMEOUT,
    PUBLISH_TIMEOUT,
    QOS,
    VALID_PIPELINES,
)
from config.logger import Logger
import paho.mqtt.client as mqtt

log = Logger("[data-transfer]")


class MQTTSender:
    """MQTT client for sending batched pipeline data."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._client: Optional[mqtt.Client] = None
        self._connected_event = threading.Event()

    # MQTT connection management
    def connect(self) -> bool:
        """
        Establish connection to MQTT broker.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            self._client = mqtt.Client(client_id=self.device_id)
            self._client.username_pw_set(USERNAME, PASSWORD)

            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect

            self._client.connect(BROKER, PORT, keepalive=60)
            self._client.loop_start()

            if not self._connected_event.wait(CONNECT_TIMEOUT):
                log.error(f"Timed out connecting to broker at {BROKER}:{PORT}")
                return False

            log.info(f"Connected to broker at {BROKER}:{PORT}")
            return True

        except Exception:
            log.error("Connection failed")
            return False

    def disconnect(self):
        """Disconnect from MQTT broker and stop network loop."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected_event.clear()
            log.info("Disconnected from broker")

    # MQTT session management
    def start_session(self):
        """Notify server that a new session has started."""
        if not self._connected_event.is_set():
            log.warning("Cannot start session — not connected")
            return

        topic = f"lab/session/start/{self.device_id}"
        payload = json.dumps({"device_id": self.device_id})

        self._client.publish(topic, payload, qos=QOS)
        log.info(f"Session started → {topic}")

    def end_session(self):
        """Notify server that the current session has ended."""
        if not self._connected_event.is_set():
            log.warning("Cannot end session — not connected")
            return

        topic = f"lab/session/end/{self.device_id}"
        payload = json.dumps({"device_id": self.device_id})

        self._client.publish(topic, payload, qos=QOS)
        log.info(f"Session ended → {topic}")

    # Data sending functions
    def send_batch(self, pipeline: str, files: List[Tuple[str, str]]):
        """
        Send a batch of files from disk.

        Args:
            pipeline: "audio" or "video"
            files: List of (file_path, data_type)
        """
        if not self._connected_event.is_set():
            log.warning("Cannot send batch — not connected")
            return

        if pipeline not in VALID_PIPELINES:
            raise ValueError(f"Invalid pipeline: {pipeline}")

        if not files:
            log.warning("send_batch called with empty file list")
            return

        encoded_files = []
        for file_path, data_type in files:
            entry = self._encode_file(file_path, data_type)
            if entry:
                encoded_files.append(entry)

        if not encoded_files:
            log.warning("No files encoded — skipping publish")
            return

        self._publish_batch(pipeline, encoded_files)

    def send_image_bytes(
        self,
        pipeline: str,
        files: List[Tuple[bytes, str, str]],
    ):
        """
        Send a batch of in-memory byte data.

        Args:
            pipeline: "video"
            files: List of (bytes_data, filename, data_type)
        """
        if not self._connected_event.is_set():
            log.warning("Cannot send batch — not connected")
            return

        if pipeline not in VALID_PIPELINES:
            raise ValueError(f"Invalid pipeline: {pipeline}")

        if not files:
            log.warning("send_image_bytes called with empty file list")
            return

        encoded_files = [
            {
                "data_type": data_type,
                "filename": filename,
                "bytes_b64": base64.b64encode(data).decode("ascii"),
            }
            for data, filename, data_type in files
        ]

        self._publish_batch(pipeline, encoded_files)

    def _publish_batch(self, pipeline: str, encoded_files: List[dict]):
        """Internal helper to publish a batch payload."""
        topic = f"lab/ingest/{pipeline}/{self.device_id}"

        payload = json.dumps(
            {
                "device_id": self.device_id,
                "files": encoded_files,
            }
        )

        result = self._client.publish(topic, payload, qos=QOS)
        result.wait_for_publish(timeout=PUBLISH_TIMEOUT)

        log.info(
            f"Batch sent → {topic} ({len(encoded_files)} file(s): {[f['data_type'] for f in encoded_files]})",
        )

    # Heartbeat management (So Server knows we're alive/helps for terminating connection on server side if something goes wrong here.)
    def start_heartbeat(self, interval: int = 15):
        """Start sending heartbeat messages every `interval` seconds.
        
        Args:
            interval: Seconds between heartbeats (default: 15)

        Returns:
            None
        """
        def _heartbeat_loop():
            while self._connected_event.is_set():
                try:
                    topic = f"lab/heartbeat/{self.device_id}"
                    payload = json.dumps({"device_id": self.device_id})

                    self._client.publish(topic, payload, qos=QOS)
                    log.debug(f"Heartbeat → {topic}")

                except Exception:
                    log.error("Failed to send heartbeat")

                finally:
                    time.sleep(interval)

        threading.Thread(target=_heartbeat_loop, daemon=True).start()

    # Helper functions
    def _encode_file(self, file_path: str, data_type: str) -> Optional[dict]:
        """Read and base64-encode a file from disk."""
        path = Path(file_path)

        if not path.exists():
            log.error(f"File not found: {file_path}")
            return None

        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()

            return {
                "data_type": data_type,
                "filename": path.name,
                "bytes_b64": base64.b64encode(raw_bytes).decode("ascii"),
            }

        except Exception:
            log.error(f"Failed to encode file: {file_path}")
            return None

    # Custom MQTT callbacks to help with debugging
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected_event.set()
            log.info("Broker connection confirmed")
        else:
            self._connected_event.clear()
            log.error(f"Connection refused, rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected_event.clear()
        if rc != 0:
            log.warning(f"Unexpected disconnect (rc={rc}), retrying")
        else:
            log.info("Clean disconnect")