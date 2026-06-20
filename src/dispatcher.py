"""
dispatcher.py — Robot Dispatch Signal Module
=============================================
Sends a structured JSON signal to the robot team whenever a fall is detected
in the vision pipeline.

Backend is pluggable via config.DISPATCH_BACKEND:
  - "console"   → Print the payload to stdout (default, no dependencies)
  - "mqtt"      → Publish to an MQTT topic via paho-mqtt
  - "http"      → POST to a REST endpoint via requests

For the hackathon demo, "mqtt" connects to the free HiveMQ public broker and
the robot_dashboard.html file on any other laptop will receive the signal live.
"""

import json
import time
import threading
from typing import Optional

import src.config as config
from src.session_labels import worker_label


class RobotDispatcher:
    """
    Sends a dispatch signal to the robot team when a fall event is confirmed.
    
    Features:
      - Pluggable backends (console / mqtt / http)
      - Per-person debouncing: once a person's fall triggers a dispatch,
        the same person won't trigger another dispatch until the cooldown
        expires (default: DISPATCH_COOLDOWN_SECONDS). This prevents the
        robot from being spammed with duplicate commands for a single fall.
    """

    def __init__(self):
        self.backend = config.DISPATCH_BACKEND
        self._cooldown_map: dict[int, float] = {}  # person_id -> last dispatch timestamp
        self._lock = threading.Lock()

        # Backend-specific initialisation
        self._mqtt_client = None
        if self.backend == "mqtt":
            self._init_mqtt()

        print(f"[Dispatcher] Initialized. Backend='{self.backend}', "
              f"Team='{config.TEAM_ID}', Zone='{config.ZONE_ID}', "
              f"Topic='{config.DISPATCH_MQTT_TOPIC}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(
        self,
        alert_type: str = "FALL_DETECTED",
        person_id: int = -1,
        zone_id: Optional[str] = None,
    ) -> bool:
        """
        Send a dispatch signal for the given person, if not in cooldown.

        Args:
            alert_type: Type of alert (e.g. "FALL_DETECTED", "RESTRICTED_ENTRY").
            person_id:  The tracked person's ID from the pipeline.
            zone_id:    Camera-space zone id; falls back to config.ZONE_ID.

        Returns:
            True if the signal was sent, False if skipped (cooldown).
        """
        with self._lock:
            now = time.time()
            last_sent = self._cooldown_map.get(person_id, 0.0)

            # Debounce: skip if we dispatched for this person recently
            if now - last_sent < config.DISPATCH_COOLDOWN_SECONDS:
                return False

            self._cooldown_map[person_id] = now

        payload = {
            "team_id": config.TEAM_ID,
            "zone_id": zone_id or config.ZONE_ID,
            "alert_type": alert_type,
            "person_id": person_id,
            "person_label": worker_label(person_id) if person_id >= 0 else None,
            "timestamp": time.time(),
        }

        try:
            if self.backend == "console":
                self._send_console(payload)
            elif self.backend == "mqtt":
                self._send_mqtt(payload)
            elif self.backend == "http":
                self._send_http(payload)
            else:
                print(f"[Dispatcher] Unknown backend '{self.backend}'. Falling back to console.")
                self._send_console(payload)
            return True
        except Exception as e:
            print(f"[Dispatcher] ERROR sending signal: {e}")
            return False

    def shutdown(self):
        """Clean up backend connections on pipeline shutdown."""
        if self._mqtt_client is not None:
            try:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
                print("[Dispatcher] MQTT client disconnected.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Backend adapters
    # ------------------------------------------------------------------

    def _send_console(self, payload: dict):
        """Console adapter — just pretty-prints the JSON. No dependencies."""
        print(f"\n{'='*55}")
        print(f"  [ROBOT] ROBOT DISPATCH SIGNAL SENT")
        print(f"{'='*55}")
        print(json.dumps(payload, indent=2))
        print(f"{'='*55}\n")

    def _send_mqtt(self, payload: dict):
        """MQTT adapter — publishes JSON to the configured broker/topic."""
        if self._mqtt_client is None:
            print("[Dispatcher] MQTT client not initialised, falling back to console.")
            self._send_console(payload)
            return

        message = json.dumps(payload)
        result = self._mqtt_client.publish(
            config.DISPATCH_MQTT_TOPIC,
            payload=message,
            qos=1,
        )
        if result.rc == 0:
            print(f"[Dispatcher] [OK] MQTT signal sent -> topic='{config.DISPATCH_MQTT_TOPIC}' | "
                  f"zone={payload['zone_id']} person={payload.get('person_label', payload['person_id'])}")
        else:
            print(f"[Dispatcher] [WARNING] MQTT publish returned rc={result.rc}")

    def _send_http(self, payload: dict):
        """HTTP adapter — POSTs JSON to the configured REST endpoint."""
        try:
            import requests  # only needed if backend == "http"
        except ImportError:
            print("[Dispatcher] 'requests' not installed. Run: pip install requests")
            self._send_console(payload)
            return

        response = requests.post(
            config.DISPATCH_HTTP_URL,
            json=payload,
            timeout=5,
        )
        print(f"[Dispatcher] [OK] HTTP signal sent -> {config.DISPATCH_HTTP_URL} "
              f"[{response.status_code}]")

    # ------------------------------------------------------------------
    # MQTT initialisation (separate method for clarity)
    # ------------------------------------------------------------------

    def _init_mqtt(self):
        """Connect to the MQTT broker in background (non-blocking)."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("[Dispatcher] 'paho-mqtt' not installed. Run: pip install paho-mqtt")
            print("[Dispatcher] Falling back to console backend.")
            self.backend = "console"
            return

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"[Dispatcher] [OK] Connected to MQTT broker "
                      f"{config.DISPATCH_MQTT_BROKER}:{config.DISPATCH_MQTT_PORT}")
            else:
                print(f"[Dispatcher] [ERROR] MQTT connection failed with code {rc}")

        def on_disconnect(client, userdata, rc):
            if rc != 0:
                print(f"[Dispatcher] [WARNING] MQTT disconnected unexpectedly (rc={rc}). Will auto-reconnect.")

        client = mqtt.Client(
            client_id=f"hackatum-{config.TEAM_ID}-dispatcher",
            transport="websockets" if config.DISPATCH_MQTT_USE_WS else "tcp",
        )
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        if config.DISPATCH_MQTT_USE_TLS:
            client.tls_set()  # Uses default system CA certs

        client.connect_async(config.DISPATCH_MQTT_BROKER, config.DISPATCH_MQTT_PORT)
        client.loop_start()  # Background thread — non-blocking
        self._mqtt_client = client
