from typing import Dict, Any, List

import src.config as config
from src.pipeline_types import FrameData


class AlertVerificationFilter:
    """Converts noisy per-frame alerts into one-shot confirmed incidents."""

    def __init__(self):
        self._state: Dict[str, Dict[str, Any]] = {}

    def _required_frames(self, alert: Dict[str, Any]) -> int:
        severity = str(alert.get("severity", "Warning")).upper()
        alert_type = str(alert.get("type", ""))

        if alert_type in {"FALL_ALERT", "RESTRICTED_ENTRY", "ENVIRONMENT_ALERT"}:
            return max(1, config.ALERT_VERIFY_CRITICAL_FRAMES)
        if severity == "CRITICAL":
            return max(1, config.ALERT_VERIFY_CRITICAL_FRAMES)
        return max(1, config.ALERT_VERIFY_WARNING_FRAMES)

    def _key(self, alert: Dict[str, Any]) -> str:
        alert_type = str(alert.get("type", "ALERT"))
        person_id = str(alert.get("person_id", "none"))
        zone_id = str(alert.get("zone_id", "none"))
        message = str(alert.get("message", ""))[:64]
        return f"{alert_type}|{person_id}|{zone_id}|{message}"

    def _cleanup_stale(self, now_ts: float) -> None:
        stale_after = max(0.5, float(config.ALERT_VERIFY_STALE_SECONDS))
        stale_keys = [
            key
            for key, state in self._state.items()
            if (now_ts - float(state.get("last_ts", now_ts))) > stale_after
        ]
        for key in stale_keys:
            del self._state[key]

    def apply(self, frame_data: FrameData) -> FrameData:
        now_ts = float(frame_data.timestamp)
        verified_alerts: List[Dict[str, Any]] = []
        verifying_alerts: List[Dict[str, Any]] = []

        for alert in frame_data.alerts:
            key = self._key(alert)
            required = self._required_frames(alert)
            state = self._state.get(key, {"count": 0, "emitted": False})

            state["count"] = int(state.get("count", 0)) + 1
            state["last_ts"] = now_ts
            self._state[key] = state

            if state["count"] >= required:
                if not state.get("emitted", False):
                    confirmed = dict(alert)
                    confirmed["debounced"] = False
                    confirmed["verification_state"] = "CONFIRMED"
                    confirmed["verification_frames"] = state["count"]
                    verified_alerts.append(confirmed)
                    state["emitted"] = True
            else:
                verifying_alerts.append({
                    "type": str(alert.get("type", "ALERT")),
                    "severity": str(alert.get("severity", "Warning")),
                    "person_id": alert.get("person_id"),
                    "zone_id": alert.get("zone_id"),
                    "frames": state["count"],
                    "required": required,
                })

        frame_data.alerts = verified_alerts
        frame_data.extra_metadata["verifying_alerts"] = verifying_alerts[:4]
        frame_data.extra_metadata["verifying_count"] = len(verifying_alerts)

        self._cleanup_stale(now_ts)
        return frame_data
