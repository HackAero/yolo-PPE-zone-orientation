"""
Zone map — safety rules tied to regions in the camera image.

ArUco markers switch the 2D zone layout (distance-based far/close modes per wall).
The active layout persists when the marker leaves the frame. Bbox center picks the
zone; rules differ per zone (PPE required or not, restricted entry alerts).
"""

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np

import src.config as config
from src.pipeline_types import FrameData, TrackedPerson
from src.session_labels import worker_label

RectNorm = Tuple[float, float, float, float]
FULL_FRAME: RectNorm = (0.0, 0.0, 1.0, 1.0)

@dataclass
class ZoneDefinition:
    id: str
    label: str
    color_bgr: Tuple[int, int, int]
    frame_rect_norm: RectNorm
    require_helmet: bool
    require_glasses: bool
    alert_on_entry: bool

    def contains(self, nx: float, ny: float) -> bool:
        x1, y1, x2, y2 = self.frame_rect_norm
        return x1 <= nx <= x2 and y1 <= ny <= y2

    def rect_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.frame_rect_norm
        return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))

class ZoneMonitor:
    def __init__(self, **kwargs):
        self._alert_last_sent: Dict[Tuple[int, str], float] = {}
        self._restricted_inside: Dict[int, bool] = {}
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        
        self.current_mode = "startup" # Default
        
        self.tpl_work = ZoneDefinition("work_floor", "Work Floor", (0, 255, 255), FULL_FRAME, True, True, False)
        self.tpl_danger = ZoneDefinition("restricted", "Danger Zone", (0, 0, 255), FULL_FRAME, False, False, True)
        self.tpl_safe = ZoneDefinition("safe", "Safe Zone", (0, 255, 0), FULL_FRAME, False, False, False)

        # Startup mode: Neutral safe zone waiting for instructions
        self.layout_startup = [replace(self.tpl_safe, frame_rect_norm=FULL_FRAME, label="SCANNING FOR ZONE SIGN...")]

        # The user's Pink Cross Path logic!
        # Robot is in the center of the cross, looking at 4 different walls.

        # --- DYNAMIC DISTANCE LAYOUTS ---
        
        # Marker 0 (North Wall)
        # FAR (> 2m): Robot is in Red/Green. Near is Red/Green, Far is Yellow.
        self.layout_0_far = [
            replace(self.tpl_work, frame_rect_norm=(0.0, 0.0, 1.0, 0.5)),   # Top (Far)
            replace(self.tpl_danger, frame_rect_norm=(0.0, 0.5, 0.5, 1.0)), # Bottom-Left (Near)
            replace(self.tpl_safe, frame_rect_norm=(0.5, 0.5, 1.0, 1.0))    # Bottom-Right (Near)
        ]
        # CLOSE (< 2m): Robot has entered Yellow zone. Everything in front of it is Yellow.
        self.layout_0_close = [replace(self.tpl_work, frame_rect_norm=FULL_FRAME)]
        
        # Marker 1 (South Wall)
        # FAR (> 2m): Robot is in Yellow. Near is Yellow, Far is Green/Red.
        self.layout_1_far = [
            replace(self.tpl_safe, frame_rect_norm=(0.0, 0.0, 0.5, 0.5)),   # Top-Left (Far)
            replace(self.tpl_danger, frame_rect_norm=(0.5, 0.0, 1.0, 0.5)), # Top-Right (Far)
            replace(self.tpl_work, frame_rect_norm=(0.0, 0.5, 1.0, 1.0))    # Bottom (Near)
        ]
        # CLOSE (< 2m): Robot has entered Red/Green zone. No Yellow is in view.
        self.layout_1_close = [
            replace(self.tpl_safe, frame_rect_norm=(0.0, 0.0, 0.5, 1.0)),   # Left (All)
            replace(self.tpl_danger, frame_rect_norm=(0.5, 0.0, 1.0, 1.0))  # Right (All)
        ]

        # Marker 2 (East Wall)
        # FAR (> 2m): Robot is on West side. Left=Yellow, Right-Near=Red, Right-Far=Green
        self.layout_2_far = [
            replace(self.tpl_work, frame_rect_norm=(0.0, 0.0, 0.5, 1.0)),   # Left side
            replace(self.tpl_safe, frame_rect_norm=(0.5, 0.0, 1.0, 0.5)),   # Right-Top (Far)
            replace(self.tpl_danger, frame_rect_norm=(0.5, 0.5, 1.0, 1.0))  # Right-Bottom (Near)
        ]
        # CLOSE (< 2m): Robot is on East side. Left=Yellow, Right=Green. (Red is behind)
        self.layout_2_close = [
            replace(self.tpl_work, frame_rect_norm=(0.0, 0.0, 0.5, 1.0)),   # Left side
            replace(self.tpl_safe, frame_rect_norm=(0.5, 0.0, 1.0, 1.0))    # Right side
        ]

        # Marker 3 (West Wall)
        # FAR (> 2m): Robot is on East side. Right=Yellow, Left-Near=Green, Left-Far=Red
        self.layout_3_far = [
            replace(self.tpl_danger, frame_rect_norm=(0.0, 0.0, 0.5, 0.5)), # Left-Top (Far)
            replace(self.tpl_safe, frame_rect_norm=(0.0, 0.5, 0.5, 1.0)),   # Left-Bottom (Near)
            replace(self.tpl_work, frame_rect_norm=(0.5, 0.0, 1.0, 1.0))    # Right side
        ]
        # CLOSE (< 2m): Robot is on West side. Right=Yellow, Left=Red. (Green is behind)
        self.layout_3_close = [
            replace(self.tpl_danger, frame_rect_norm=(0.0, 0.0, 0.5, 1.0)), # Left side
            replace(self.tpl_work, frame_rect_norm=(0.5, 0.0, 1.0, 1.0))    # Right side
        ]
        
        self._apply_layout()

    def cycle_layout(self) -> str:
        modes = [
            "startup", 
            "marker_0_far", "marker_0_close", 
            "marker_1_far", "marker_1_close",
            "marker_2_far", "marker_2_close",
            "marker_3_far", "marker_3_close"
        ]
        if self.current_mode in modes:
            idx = modes.index(self.current_mode)
            self.current_mode = modes[(idx + 1) % len(modes)]
        else:
            self.current_mode = modes[0]
        self._apply_layout()
        return f"Manual override -> {self.current_mode}"

    def _apply_layout(self):
        self._restricted_inside.clear()
        if self.current_mode == "startup": self.zones = self.layout_startup
        elif self.current_mode == "marker_0_far": self.zones = self.layout_0_far
        elif self.current_mode == "marker_0_close": self.zones = self.layout_0_close
        elif self.current_mode == "marker_1_far": self.zones = self.layout_1_far
        elif self.current_mode == "marker_1_close": self.zones = self.layout_1_close
        elif self.current_mode == "marker_2_far": self.zones = self.layout_2_far
        elif self.current_mode == "marker_2_close": self.zones = self.layout_2_close
        elif self.current_mode == "marker_3_far": self.zones = self.layout_3_far
        elif self.current_mode == "marker_3_close": self.zones = self.layout_3_close

    def _debounced(self, person_id: int, alert_key: str) -> bool:
        now = time.time()
        key = (person_id, alert_key)
        if now - self._alert_last_sent.get(key, 0.0) < config.ALERT_DEBOUNCE_SECONDS:
            return True
        self._alert_last_sent[key] = now
        return False

    def process(self, frame_data: FrameData, dispatcher=None) -> FrameData:
        if frame_data.raw_frame.size == 0:
            return frame_data

        h, w = frame_data.raw_frame.shape[:2]
        triggered_new_mode = False
        distance_msg = getattr(self, "_last_distance_msg", "")

        if getattr(frame_data, "frame_index", 0) % 10 == 0:
            gray = cv2.cvtColor(frame_data.raw_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            distance_msg = ""
            
            if ids is not None:
                # We need a synthetic camera matrix to estimate 3D distance
                focal_length = w
                camera_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                
                # Estimate pose (assuming marker is 15cm wide)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.15, camera_matrix, dist_coeffs)
                
                first_id = ids[0][0]
                new_mode = self.current_mode
                
                if tvecs is not None:
                    # Get the Z distance to the marker in meters
                    dist_m = tvecs[0][0][2]
                    distance_msg = f" (Dist: {dist_m:.1f}m)"
                    
                    # Threshold for "Close" vs "Far"
                    is_close = dist_m < 2.0
                    
                    if first_id == 0:
                        new_mode = "marker_0_close" if is_close else "marker_0_far"
                    elif first_id == 1:
                        new_mode = "marker_1_close" if is_close else "marker_1_far"
                    elif first_id == 2:
                        new_mode = "marker_2_close" if is_close else "marker_2_far"
                    elif first_id == 3:
                        new_mode = "marker_3_close" if is_close else "marker_3_far"
                
                if new_mode != self.current_mode:
                    self.current_mode = new_mode
                    self._apply_layout()
                    triggered_new_mode = True
            self._last_distance_msg = distance_msg

        frame_data.extra_metadata["zone_layout_label"] = f"MODE: {self.current_mode.upper()}{distance_msg}"
        if triggered_new_mode:
            frame_data.extra_metadata["zone_layout_label"] += " (NEW!)"

        active_ids = set()
        for person in frame_data.persons:
            active_ids.add(person.person_id)
            
            px, py = (person.bbox[0] + person.bbox[2])/2.0, person.bbox[3]
            nx, ny = px / w, py / h
            person.metadata["zone_point"] = (int(px), int(py))
            
            zone = self.zones[0]
            for z in self.zones:
                if z.contains(nx, ny):
                    zone = z
                    break
                    
            person.metadata["zone_id"] = zone.id
            person.metadata["zone_label"] = zone.label
            person.metadata["zone_forbidden"] = bool(zone.alert_on_entry)

            required_violations = []
            if zone.require_helmet and person.has_helmet is not True:
                required_violations.append("Helmet")
            if zone.require_glasses and person.has_glasses is not True:
                required_violations.append("Glasses")

            optional_missing = []
            if zone.id in ("work_floor", "restricted") and person.metadata.get("has_yellow_vest") is False:
                optional_missing.append("Vest")

            person.metadata["zone_violations"] = required_violations
            person.metadata["zone_optional_missing"] = optional_missing
            person.metadata["zone_required_ok"] = len(required_violations) == 0

            if zone.alert_on_entry:
                was_inside = self._restricted_inside.get(person.person_id, False)
                if not was_inside:
                    if not self._debounced(person.person_id, f"entry_{zone.id}"):
                        msg = (
                            f"{worker_label(person.person_id)} entered restricted area "
                            f"'{zone.label}'"
                        )
                        frame_data.alerts.append({
                            "type": "RESTRICTED_ENTRY", "severity": "Critical", "message": msg,
                            "person_id": person.person_id, "zone_id": zone.id, "timestamp": frame_data.timestamp
                        })
                        if dispatcher: dispatcher.send(alert_type="RESTRICTED_ENTRY", person_id=person.person_id, zone_id=zone.id)
                self._restricted_inside[person.person_id] = True
            else:
                self._restricted_inside.pop(person.person_id, None)

            for violation in required_violations:
                alert_key = f"zone_{zone.id}_{violation}"
                if self._debounced(person.person_id, alert_key):
                    continue

                frame_data.alerts.append({
                    "type": "ZONE_PPE_VIOLATION",
                    "severity": "Warning",
                    "message": (
                        f"{worker_label(person.person_id)} missing {violation.lower()} "
                        f"in zone '{zone.label}'"
                    ),
                    "person_id": person.person_id,
                    "zone_id": zone.id,
                    "timestamp": frame_data.timestamp,
                })
                if dispatcher is not None:
                    alert_type = {
                        "Helmet": "NO_HELMET",
                        "Glasses": "NO_GLASSES",
                        "Vest": "NO_VEST",
                    }.get(violation, "ZONE_PPE_VIOLATION")
                    dispatcher.send(
                        alert_type=alert_type,
                        person_id=person.person_id,
                        zone_id=zone.id,
                    )

        self._restricted_inside = {pid: v for pid, v in self._restricted_inside.items() if pid in active_ids}
        
        frame_data.extra_metadata["zones"] = [
            {"id": z.id, "label": z.label, "color_bgr": z.color_bgr, "rect": z.rect_pixels(w, h)}
            for z in self.zones
        ]
        return frame_data


def draw_zones_overlay(frame: np.ndarray, frame_data: FrameData) -> None:
    overlay = frame.copy()
    
    zones = frame_data.extra_metadata.get("zones")
    if zones:
        for zone in zones:
            x1, y1, x2, y2 = zone["rect"]
            color = tuple(zone["color_bgr"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    for person in frame_data.persons:
        pt = person.metadata.get("zone_point")
        if pt:
            cv2.circle(frame, pt, 2, (255, 255, 255), -1)

    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
