#!/usr/bin/env python3
"""
robot_sim.py – Dummy Robot Simulator
Subscribes to dispatch signals and simulates robot movement.
Publishes status updates for the dashboard.
"""

import json
import time
import random
import threading
import paho.mqtt.client as mqtt

# Configuration
BROKER = "broker.hivemq.com"
PORT = 8884
USE_TLS = True
USE_WS = True

DISPATCH_TOPIC = "hackatum/robot/dispatch"
STATUS_TOPIC = "hackatum/robot/status"

# Robot states
IDLE = "IDLE"
EN_ROUTE = "EN_ROUTE"
ARRIVED = "ARRIVED"

class RobotSimulator:
    def __init__(self):
        self.state = IDLE
        self.current_zone = None
        self.client = mqtt.Client(
            client_id="robot-sim-" + str(random.randint(1000, 9999)),
            transport="websockets" if USE_WS else "tcp"
        )
        if USE_TLS:
            self.client.tls_set()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(BROKER, PORT, 60)
        self.client.loop_start()
        print(f"[Robot] Started. Subscribed to {DISPATCH_TOPIC}")

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

        # Ignore if already en route or arrived (cooldown)
        if self.state != IDLE:
            print(f"[Robot] Busy ({self.state}), ignoring dispatch.")
            return
        
  
        if alert_type.upper() in ("NO_HELMET"):
            print(f"[Robot] Robot scolds Worker for not wearing a helmet.")
            return
        
        if alert_type.upper() in ("FIRE_DETECTED", "SMOKE_DETECTED"):
            self.scream(alert_type, zone, person_id)

        # Run dispatch in a background thread so it doesn't freeze the MQTT loop
        threading.Thread(target=self.dispatch, args=(alert_type, zone, person_id), daemon=True).start()

        # self.dispatch(alert_type, zone, person_id)

    def dispatch(self, alert_type, zone, person_id):
        print(f"[Robot] 📨 Received {alert_type} for Person {person_id} in Zone {zone}")
        self.state = EN_ROUTE
        self.current_zone = zone

        # 1. Travel Phase
        travel_time = random.uniform(4, 7)
        status_msg = {
            "status": "EN_ROUTE",
            "zone": zone,
            "eta": int(travel_time),
            "alert_type": alert_type,
            "action": "Navigating to scene...",
            "siren": alert_type in ("FALL_DETECTED", "RESTRICTED_ENTRY", "FIRE_DETECTED"), # Turn on siren for criticals
            "timestamp": time.time()
        }
        self.publish_status(status_msg)
        time.sleep(travel_time)

        # 2. Arrival Phase
        self.state = ARRIVED
        status_msg["status"] = "ARRIVED"
        status_msg["eta"] = 0
        status_msg["siren"] = False # Turn off travel siren
        status_msg["action"] = "Securing area..."
        self.publish_status(status_msg)
        time.sleep(2)

        # 3. On-Scene Action Phase (Based on alert type)
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
            
        elif alert_type in ("NO_HELMET", "NO_GLASSES"):
            status_msg["status"] = "WARNING"
            status_msg["action"] = "Playing audio warning..."
            self.publish_status(status_msg)
            time.sleep(4)
            
            status_msg["status"] = "RESOLVING"
            status_msg["action"] = "Worker is now compliant."
            self.publish_status(status_msg)
            time.sleep(2)
            
        else:
            status_msg["status"] = "INVESTIGATING"
            status_msg["action"] = "Recording video log..."
            self.publish_status(status_msg)
            time.sleep(4)

        # 4. Return to Base
        self.state = IDLE
        self.current_zone = None
        status_msg["status"] = "IDLE"
        status_msg["zone"] = None
        status_msg["action"] = "Awaiting orders"
        status_msg["siren"] = False
        self.publish_status(status_msg)
        print(f"[Robot] Task complete. Back to IDLE.")
        
        
    def publish_status(self, data):
        self.client.publish(STATUS_TOPIC, json.dumps(data), qos=1)
        # Also print for console visibility
        print(f"[Robot] Status: {data['status']} (Zone {data.get('zone', 'None')})")

    def scream(self, alert_type, zone, person_id):
        # Dramatic ASCII art
        scream_msg = f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║  🚨🚨🚨  WARNING  {alert_type} DETECTED!  🚨🚨🚨          ║
        ║  🔥  DANGER! EVACUATE 🔥                                    ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print(scream_msg)

        # Optionally, publish a scream flag in status (your dashboard could listen)
        scream_status = {
            "status": "DANGER",
            "zone": zone,
            "alert_type": alert_type,
            "person_id": person_id,
            "timestamp": time.time()
        }
        self.publish_status(scream_status)
        
        
    def run(self):
        try:
            while True:
                time.sleep(3)
        except KeyboardInterrupt:
            print("[Robot] Shutting down...")
            self.client.loop_stop()
            self.client.disconnect()

if __name__ == "__main__":
    sim = RobotSimulator()
    sim.run()