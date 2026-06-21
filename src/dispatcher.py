"""
Publish safety alerts to the robot team (MQTT by default).

Payload is small JSON: team_id, zone_id, alert_type, person label. HiveMQ public
broker + robot_dashboard.html means any laptop on the internet can show live
dispatches without us running a server.
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
      - Per-alert debouncing: the same person won't re-trigger the same
        alert_type until DISPATCH_COOLDOWN_SECONDS expires.
    """

    def __init__(self):
        self.backend = config.DISPATCH_BACKEND
        self._cooldown_map: dict[tuple[int, str], float] = {}  # (person_id, alert_type) -> last sent
        self._lock = threading.Lock()

        # Backend-specific initialisation
        self._mqtt_client = None
        if self.backend == "mqtt":
            self._init_mqtt()

        if getattr(config, "ENABLE_TTS_SIREN", False):
            try:
                import os
                os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
                import pygame
                pygame.mixer.init()
            except Exception as e:
                print(f"[Dispatcher] Failed to init pygame: {e}")

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
            cooldown_key = (person_id, alert_type)
            last_sent = self._cooldown_map.get(cooldown_key, 0.0)

            if now - last_sent < config.DISPATCH_COOLDOWN_SECONDS:
                return False

            self._cooldown_map[cooldown_key] = now

        # Fire off our "wow factor" hackathon notifications
        if alert_type in ["FALL_DETECTED", "RESTRICTED_ENTRY"]:
            msg = "Warning. Critical Fall Detected in Sector A." if alert_type == "FALL_DETECTED" else "Warning. Unauthorized access in restricted zone."
            self._trigger_wow_factor(msg)

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
        """HTTP adapter — POSTs JSON to the configured REST endpoint asynchronously."""
        try:
            import requests
        except ImportError:
            print("[Dispatcher] 'requests' not installed. Run: pip install requests")
            self._send_console(payload)
            return

        def _do_post():
            try:
                response = requests.post(
                    config.DISPATCH_HTTP_URL,
                    json=payload,
                    timeout=5,
                )
                print(f"[Dispatcher] [OK] HTTP signal sent -> {config.DISPATCH_HTTP_URL} [{response.status_code}]")
            except Exception as e:
                print(f"[Dispatcher] [ERROR] HTTP signal failed: {e}")

        import threading
        t = threading.Thread(target=_do_post)
        t.daemon = True
        t.start()

    # ------------------------------------------------------------------
    # Hackathon "Wow Factor" Notifications
    # ------------------------------------------------------------------

    def _trigger_wow_factor(self, message: str) -> None:
        if getattr(config, "ENABLE_TTS_SIREN", False):
            self._play_tts(message)
        if getattr(config, "ENABLE_TWILIO_SMS", False):
            self._send_twilio_sms(message)

    def _play_tts(self, message: str) -> None:
        def _play_mp3():
            try:
                import pygame
                import os
                # Dynamically get the absolute path to the project root
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                mp3_path = os.path.join(base_dir, "images", "spongebobalarm.mp3")
                
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"[Dispatcher] MP3 playback failed: {e}")
        t = threading.Thread(target=_play_mp3)
        t.daemon = True
        t.start()

    def _send_twilio_sms(self, message: str) -> None:
        def _send():
            try:
                from twilio.rest import Client
                account_sid = config.TWILIO_ACCOUNT_SID
                auth_token = config.TWILIO_AUTH_TOKEN
                if not account_sid or not auth_token:
                    print("[Dispatcher] Twilio keys missing in .env. Skipping SMS.")
                    return

                client = Client(account_sid, auth_token)
                
                msg = client.messages.create(
                    body=f"{message} Dispatching emergency rover immediately.",
                    to=config.TWILIO_TO_NUMBER,
                    from_=config.TWILIO_FROM_NUMBER
                )
                print(f"[Dispatcher] [OK] Twilio SMS sent: {msg.sid}")
            except Exception as e:
                print(f"[Dispatcher] Twilio SMS failed: {e}")
        t = threading.Thread(target=_send)
        t.daemon = True
        t.start()

    # ------------------------------------------------------------------
    # MQTT initialisation (separate method for clarity)
    # ------------------------------------------------------------------

    def _init_mqtt(self):
        """Connect to the MQTT broker in background (non-blocking)."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("[Dispatcher] MQTT disabled: missing dependency 'paho-mqtt'.")
            print("[Dispatcher] Install with: pip install paho-mqtt")
            print("[Dispatcher] Falling back to console dispatch backend.")
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
