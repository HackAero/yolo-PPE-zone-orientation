"""
Zone map ("B-lite") — safety rules tied to regions in the camera image.

We draw vertical bands on the frame (see zones/*.json) instead of a factory floor
plan because we had no SLAM or robot pose at the hackathon. Bbox center picks
the zone; rules differ per zone (PPE required or not, restricted entry alerts).

Press w in main.py to cycle: 3 bands → full safe → full work → full restricted.
"""

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import src.config as config
from src.pipeline_types import FrameData, TrackedPerson
from src.session_labels import worker_label

RectNorm = Tuple[float, float, float, float]
FULL_FRAME: RectNorm = (0.0, 0.0, 1.0, 1.0)

LAYOUT_MODES = ("split", "all_safe", "all_work", "all_restricted")
LAYOUT_LABELS = {
    "split": "3 zones (safe | work | restricted)",
    "all_safe": "Full frame: safe",
    "all_work": "Full frame: work",
    "all_restricted": "Full frame: restricted",
}


@dataclass
class ZoneDefinition:
    id: str
    label: str
    color_bgr: Tuple[int, int, int]
    frame_rect_norm: RectNorm
    require_helmet: bool
    require_glasses: bool
    alert_on_entry: bool
    priority: int

    def contains(self, nx: float, ny: float) -> bool:
        x1, y1, x2, y2 = self.frame_rect_norm
        return x1 <= nx <= x2 and y1 <= ny <= y2

    def rect_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.frame_rect_norm
        return (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )


def person_frame_norm(person: TrackedPerson, width: int, height: int) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = person.bbox
    nx = (xmin + xmax) / 2.0 / max(1, width)
    ny = (ymin + ymax) / 2.0 / max(1, height)
    return nx, ny


def load_zones(path: Optional[str] = None) -> List[ZoneDefinition]:
    path = path or config.resolve_zones_path()
    if not os.path.exists(path):
        print(f"[ZoneMap] No zones file at {path}; zone monitoring disabled.")
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    profile = data.get("profile", "unknown")
    print(f"[ZoneMap] Loaded profile '{profile}' from {path}")

    zones = []
    for entry in data.get("zones", []):
        rect = (
            entry.get("frame_rect_norm")
            or entry.get("floor_rect_norm")
            or entry.get("rect_norm")
        )
        if not rect:
            continue
        zones.append(
            ZoneDefinition(
                id=entry["id"],
                label=entry.get("label", entry["id"]),
                color_bgr=tuple(entry.get("color_bgr", [180, 180, 180])),
                frame_rect_norm=tuple(rect),
                require_helmet=bool(entry.get("require_helmet", True)),
                require_glasses=bool(entry.get("require_glasses", True)),
                alert_on_entry=bool(entry.get("alert_on_entry", False)),
                priority=int(entry.get("priority", 0)),
            )
        )
    zones.sort(key=lambda z: z.priority, reverse=True)
    return zones


def assign_zone(
    person: TrackedPerson,
    width: int,
    height: int,
    zones: List[ZoneDefinition],
    split_mode: bool,
) -> ZoneDefinition:
    nx, ny = person_frame_norm(person, width, height)
    person.metadata["frame_x"] = round(nx, 3)
    person.metadata["frame_y"] = round(ny, 3)
    person.metadata["zone_point"] = (int(nx * width), int(ny * height))

    if len(zones) == 1:
        return zones[0]

    for zone in zones:
        if zone.contains(nx, ny):
            return zone

    if not split_mode:
        return zones[0]

    by_id = {z.id: z for z in zones}
    if nx < 0.2:
        return by_id.get("safe") or zones[-1]
    if nx < 0.75:
        return by_id.get("work_floor") or zones[-1]
    return by_id.get("restricted") or zones[0]


class ZoneMonitor:
    """Assigns persons to frame zones; press 'w' to cycle layout."""

    def __init__(
        self,
        zones_path: Optional[str] = None,
        camera_profile: Optional[str] = None,
    ):
        self.camera_profile = camera_profile or config.ZONES_PROFILE
        self._zones_path = zones_path or config.resolve_zones_path()
        self._base_zones = load_zones(self._zones_path) if config.ZONES_ENABLED else []
        self._layout_mode = "split"
        self.zones: List[ZoneDefinition] = []
        self._restricted_inside: Dict[int, bool] = {}
        self._alert_last_sent: Dict[Tuple[int, str], float] = {}
        self._apply_layout()

        if self._base_zones:
            print(f"[ZoneMap] Layout: {LAYOUT_LABELS[self._layout_mode]} (press w to cycle)")

    def _template(self, zone_id: str) -> Optional[ZoneDefinition]:
        for zone in self._base_zones:
            if zone.id == zone_id:
                return zone
        return None

    def _full_frame_zone(self, template: ZoneDefinition) -> ZoneDefinition:
        return replace(template, frame_rect_norm=FULL_FRAME)

    def _apply_layout(self) -> None:
        if not self._base_zones:
            self.zones = []
            return

        if self._layout_mode == "split":
            self.zones = list(self._base_zones)
            return

        template_id = {
            "all_safe": "safe",
            "all_work": "work_floor",
            "all_restricted": "restricted",
        }.get(self._layout_mode)

        template = self._template(template_id) if template_id else None
        if template is None:
            self.zones = list(self._base_zones)
            return

        self.zones = [self._full_frame_zone(template)]

    def cycle_layout(self) -> str:
        idx = LAYOUT_MODES.index(self._layout_mode)
        self._layout_mode = LAYOUT_MODES[(idx + 1) % len(LAYOUT_MODES)]
        self._restricted_inside.clear()
        self._apply_layout()
        label = LAYOUT_LABELS[self._layout_mode]
        print(f"[ZoneMap] Layout -> {label}")
        return label

    @property
    def layout_label(self) -> str:
        return LAYOUT_LABELS[self._layout_mode]

    def _debounced(self, person_id: int, alert_key: str) -> bool:
        now = time.time()
        key = (person_id, alert_key)
        last = self._alert_last_sent.get(key, 0.0)
        if now - last < config.ALERT_DEBOUNCE_SECONDS:
            return True
        self._alert_last_sent[key] = now
        return False

    def process(self, frame_data: FrameData, dispatcher=None) -> FrameData:
        if not self.zones or frame_data.raw_frame.size == 0:
            return frame_data

        h, w = frame_data.raw_frame.shape[:2]
        split_mode = self._layout_mode == "split"
        frame_data.extra_metadata["zones_ready"] = True
        frame_data.extra_metadata["zone_layout"] = self._layout_mode
        frame_data.extra_metadata["zone_layout_label"] = self.layout_label
        active_ids = set()

        for person in frame_data.persons:
            active_ids.add(person.person_id)
            zone = assign_zone(person, w, h, self.zones, split_mode)

            person.metadata["zone_id"] = zone.id
            person.metadata["zone_label"] = zone.label

            if zone.alert_on_entry:
                was_inside = self._restricted_inside.get(person.person_id, False)
                if not was_inside:
                    if not self._debounced(person.person_id, f"entry_{zone.id}"):
                        msg = (
                            f"{worker_label(person.person_id)} entered restricted area "
                            f"'{zone.label}'"
                        )
                        frame_data.alerts.append({
                            "type": "RESTRICTED_ENTRY",
                            "severity": "Critical",
                            "message": msg,
                            "person_id": person.person_id,
                            "zone_id": zone.id,
                            "timestamp": frame_data.timestamp,
                        })
                        if dispatcher is not None:
                            dispatcher.send(
                                alert_type="RESTRICTED_ENTRY",
                                person_id=person.person_id,
                                zone_id=zone.id,
                            )
                self._restricted_inside[person.person_id] = True
            elif person.person_id in self._restricted_inside:
                self._restricted_inside[person.person_id] = False

            zone_violations = []
            if zone.require_helmet and person.has_helmet is not True:
                zone_violations.append("Helmet")
            if zone.require_glasses and person.has_glasses is not True:
                zone_violations.append("Glasses")
            if zone.id in ("work_floor", "restricted") and person.metadata.get("has_yellow_vest") is False:
                zone_violations.append("Vest")

            person.metadata["zone_violations"] = zone_violations

            for violation in zone_violations:
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

        for pid in list(self._restricted_inside):
            if pid not in active_ids:
                del self._restricted_inside[pid]

        frame_data.extra_metadata["zones"] = [
            {
                "id": z.id,
                "label": z.label,
                "color_bgr": z.color_bgr,
                "rect": z.rect_pixels(w, h),
            }
            for z in self.zones
        ]
        frame_data.extra_metadata["zone_occupancy"] = len(frame_data.persons)

        return frame_data


def draw_zones_overlay(frame: np.ndarray, frame_data: FrameData) -> None:
    zones = frame_data.extra_metadata.get("zones")
    if not zones:
        return

    overlay = frame.copy()
    for zone in zones:
        x1, y1, x2, y2 = zone["rect"]
        color = tuple(zone["color_bgr"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            zone["label"],
            (x1 + 4, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for person in frame_data.persons:
        pt = person.metadata.get("zone_point")
        if pt:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, (0, 0, 0), 1)

    cv2.addWeighted(overlay, 0.12, frame, 0.85, 0, frame)

    layout = frame_data.extra_metadata.get("zone_layout_label")
    if layout:
        h, w = frame.shape[:2]
        hint = f"Zones: {layout}  |  w=cycle  q=quit"
        cv2.putText(
            frame,
            hint,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
