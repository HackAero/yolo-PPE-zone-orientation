#!/usr/bin/env python3
"""
Stand-in for the physical rover during demos.

Listens on hackatum/robot/dispatch, fakes drive time + on-scene actions, publishes
status to hackatum/robot/status so robot_dashboard.html can show ETA / siren.
"""

import json
import os
import random
import sys
import threading
import time

import paho.mqtt.client as mqtt

# Allow `python robot_sim.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.config as config

DISPATCH_TOPIC = config.DISPATCH_MQTT_TOPIC
STATUS_TOPIC = config.DISPATCH_MQTT_STATUS_TOPIC

IDLE = "IDLE"
EN_ROUTE = "EN_ROUTE"
ARRIVED = "ARRIVED"

# PPE alerts: audio warning on scene, no siren during travel
PPE_ALERTS = frozenset({"NO_HELMET", "NO_GLASSES", "NO_VEST"})
# Critical alerts: siren + full dispatch
CRITICAL_ALERTS = frozenset({
    "FALL_DETECTED",
    "RESTRICTED_ENTRY",
    "FIRE_DETECTED",
    "SMOKE_DETECTED",
})


class RobotSimulator:
    def __init__(self):
        self.state = IDLE
        self.current_zone = None
        self.client = mqtt.Client(
            client_id="robot-sim-" + str(random.randint(1000, 9999)),
            transport="websockets" if config.DISPATCH_MQTT_USE_WS else "tcp",
        )
        if config.DISPATCH_MQTT_USE_TLS:
            self.client.tls_set()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(config.DISPATCH_MQTT_BROKER, config.DISPATCH_MQTT_PORT, 60)
        self.client.loop_start()
        print(f"[Robot] Started. Dispatch={DISPATCH_TOPIC}  Status={STATUS_TOPIC}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[Robot] Connected to MQTT broker")
            client.subscribe(DISPATCH_TOPIC, qos=1)
        else:
            print(f"[Robot] Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        if msg.topic != DISPATCH_TOPIC:
            return
        try:
            payload = json.loads(msg.payload.decode())
            alert_type = payload.get("alert_type", "UNKNOWN")
            zone = payload.get("zone_id", "UNKNOWN")
            person_id = payload.get("person_id", -1)
        except Exception as e:
            print(f"[Robot] Invalid payload: {e}")
            return

        if self.state != IDLE:
            print(f"[Robot] Busy ({self.state}), ignoring {alert_type}.")
            return

        if alert_type in ("FIRE_DETECTED", "SMOKE_DETECTED"):
            self.scream(alert_type, zone, person_id)

        threading.Thread(
            target=self.dispatch,
            args=(alert_type, zone, person_id),
            daemon=True,
        ).start()

    def dispatch(self, alert_type, zone, person_id):
        print(f"[Robot] Received {alert_type} for person {person_id} in zone {zone}")
        self.state = EN_ROUTE
        self.current_zone = zone

        travel_time = random.uniform(4, 7)
        status_msg = {
            "status": "EN_ROUTE",
            "zone": zone,
            "eta": int(travel_time),
            "alert_type": alert_type,
            "action": "Navigating to scene...",
            "siren": alert_type in CRITICAL_ALERTS,
            "timestamp": time.time(),
        }
        self.publish_status(status_msg)
        time.sleep(travel_time)

        self.state = ARRIVED
        status_msg["status"] = "ARRIVED"
        status_msg["eta"] = 0
        status_msg["siren"] = False
        status_msg["action"] = "Securing area..."
        self.publish_status(status_msg)
        time.sleep(2)

        if alert_type == "FALL_DETECTED":
            status_msg["status"] = "ASSESSING"
            status_msg["action"] = "Scanning worker vitals..."
            self.publish_status(status_msg)
            time.sleep(3)

            status_msg["status"] = "ALERTING"
            status_msg["action"] = "Calling Medical Rescue!"
            status_msg["siren"] = True
            self.publish_status(status_msg)
            time.sleep(3)

        elif alert_type == "RESTRICTED_ENTRY":
            status_msg["status"] = "ALERTING"
            status_msg["action"] = "Unauthorized entry — securing perimeter..."
            status_msg["siren"] = True
            self.publish_status(status_msg)
            time.sleep(3)

            status_msg["status"] = "INVESTIGATING"
            status_msg["action"] = "Recording video log..."
            self.publish_status(status_msg)
            time.sleep(3)

        elif alert_type in ("FIRE_DETECTED", "SMOKE_DETECTED"):
            status_msg["status"] = "ALERTING"
            status_msg["action"] = "Calling Fire Department!"
            status_msg["siren"] = True
            self.publish_status(status_msg)
            time.sleep(2)

            status_msg["status"] = "INVESTIGATING"
            status_msg["action"] = "Recording video log..."
            self.publish_status(status_msg)
            time.sleep(3)

            status_msg["status"] = "EVACUATING"
            status_msg["action"] = "Remove Robot from Fire/Smoke Zone"
            self.publish_status(status_msg)
            time.sleep(4)

        elif alert_type in PPE_ALERTS:
            label = {
                "NO_HELMET": "helmet",
                "NO_GLASSES": "safety glasses",
                "NO_VEST": "high-visibility vest",
            }.get(alert_type, "PPE")
            print(f"[Robot] Playing audio warning: worker missing {label}.")

            status_msg["status"] = "WARNING"
            status_msg["action"] = f"Audio warning: missing {label}..."
            self.publish_status(status_msg)
            time.sleep(4)

            status_msg["status"] = "RESOLVING"
            status_msg["action"] = "Awaiting compliance..."
            self.publish_status(status_msg)
            time.sleep(2)

        else:
            status_msg["status"] = "INVESTIGATING"
            status_msg["action"] = "Recording video log..."
            self.publish_status(status_msg)
            time.sleep(4)

        self.state = IDLE
        self.current_zone = None
        status_msg["status"] = "IDLE"
        status_msg["zone"] = None
        status_msg["action"] = "Awaiting orders"
        status_msg["siren"] = False
        self.publish_status(status_msg)
        print("[Robot] Task complete. Back to IDLE.")

    def publish_status(self, data):
        self.client.publish(STATUS_TOPIC, json.dumps(data), qos=1)
        print(f"[Robot] Status: {data['status']} (zone {data.get('zone', 'None')})")

    def scream(self, alert_type, zone, person_id):
        print(
            f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║  WARNING  {alert_type} DETECTED!                               ║
        ║  DANGER! EVACUATE                                            ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        )
        self.publish_status({
            "status": "DANGER",
            "zone": zone,
            "alert_type": alert_type,
            "person_id": person_id,
            "siren": True,
            "timestamp": time.time(),
        })

    def run(self):
        try:
            while True:
                time.sleep(3)
        except KeyboardInterrupt:
            print("[Robot] Shutting down...")
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    RobotSimulator().run()
