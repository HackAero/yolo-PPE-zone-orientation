import cv2
import argparse
import time
import sys
import os
import json
from datetime import datetime
from collections import deque
import threading
from src.engine import SafetyPipelineEngine
from src.session_labels import worker_label
from src.zone_map import draw_zones_overlay
import src.config as config

SHOW_PPE_DEBUG = False


class SupervisorReplayRecorder:
    def __init__(self, output_dir="data/replays", pre_seconds=5.0, post_seconds=5.0, cooldown_seconds=15.0):
        self.output_dir = output_dir
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.cooldown_seconds = cooldown_seconds

        self._pre_buffer = deque(maxlen=1200)
        self._active_event = None
        self._last_trigger_ts_by_type = {}
        self._popup_until_ts = 0.0
        self._popup_lines = []
        self._saved_until_ts = 0.0
        self._saved_text = ""

        os.makedirs(self.output_dir, exist_ok=True)

    def add_frame(self, timestamp, frame):
        frame_copy = frame.copy()
        self._pre_buffer.append((timestamp, frame_copy))

        if self._active_event is None:
            return

        self._active_event["frames"].append(frame_copy)
        if timestamp >= self._active_event["end_ts"]:
            self._finalize_event(timestamp)

    def trigger(self, timestamp, event):
        if self._active_event is not None:
            return False
        event_key = str(event.get("type", "CRITICAL"))
        last_ts = float(self._last_trigger_ts_by_type.get(event_key, 0.0))
        if timestamp - last_ts < self.cooldown_seconds:
            return False

        pre_start = timestamp - self.pre_seconds
        pre_frames = [frame for ts, frame in self._pre_buffer if ts >= pre_start]

        self._active_event = {
            "event": event,
            "start_ts": timestamp,
            "end_ts": timestamp + self.post_seconds,
            "frames": pre_frames,
        }
        self._last_trigger_ts_by_type[event_key] = timestamp

        self._popup_until_ts = timestamp + 4.0
        self._popup_lines = [
            f"CRITICAL EVENT: {event['type']}",
            event["summary"],
            f"ACTION: {event['action']} | CONF: {event['confidence']}",
        ]
        return True

    def _finalize_event(self, timestamp):
        if self._active_event is None:
            return

        event = self._active_event["event"]
        frames = self._active_event["frames"]
        self._active_event = None

        if not frames:
            return

        # Start a thread to save the video so we don't block the main loop
        t = threading.Thread(target=self._save_video_task, args=(timestamp, event, frames))
        t.daemon = True
        t.start()

    def _save_video_task(self, timestamp, event, frames):
        if not frames:
            return

        base_h, base_w = frames[0].shape[:2]
        base_w = int(base_w) - (int(base_w) % 2)
        base_h = int(base_h) - (int(base_h) % 2)

        duration = max(0.1, self.pre_seconds + self.post_seconds)
        fps = max(8.0, min(24.0, len(frames) / duration))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_slug = event["type"].lower().replace("/", "_").replace(" ", "_")
        base_name = f"event_{stamp}_{event_slug}"
        avi_path = os.path.join(self.output_dir, f"{base_name}.avi")
        meta_path = os.path.join(self.output_dir, f"{base_name}.json")

        avi_writer = cv2.VideoWriter(
            avi_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (base_w, base_h),
        )

        valid_frames_count = 0
        if avi_writer.isOpened():
            for frame in frames:
                if frame is None or frame.size == 0:
                    continue
                out = frame
                if len(out.shape) == 2:
                    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
                elif out.shape[2] == 4:
                    out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
                if out.dtype != "uint8":
                    out = out.clip(0, 255).astype("uint8")
                if out.shape[1] != base_w or out.shape[0] != base_h:
                    out = cv2.resize(out, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
                else:
                    out = out[:base_h, :base_w]
                avi_writer.write(out)
                valid_frames_count += 1
            avi_writer.release()

        metadata = {
            "timestamp": timestamp,
            "event": event,
            "video_path": avi_path,
            "video_path_avi": avi_path,
            "frame_count": valid_frames_count,
            "fps": fps,
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._saved_until_ts = timestamp + 3.0
        self._saved_text = f"REPLAY SAVED: {os.path.basename(avi_path)}"

    def draw_overlay(self, frame, timestamp):
        w = frame.shape[1]
        font_scale = 0.4
        
        # Center x calculation helper
        def get_center_x(text, scale):
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            return (w - tw) // 2

        if timestamp <= self._saved_until_ts and self._saved_text:
            text = self._saved_text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cx = get_center_x(text, font_scale)
            cv2.rectangle(frame, (cx - 10, 10), (cx + tw + 10, 10 + th + 10), (0, 120, 0), -1)
            cv2.putText(frame, text, (cx, 10 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        if timestamp <= self._popup_until_ts and self._popup_lines:
            y_pos = 40 if (timestamp <= self._saved_until_ts and self._saved_text) else 10
            line_height = 20
            
            # Find max width for the pill
            max_w = 0
            for line in self._popup_lines:
                (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                max_w = max(max_w, tw)
                
            pill_h = 10 + line_height * len(self._popup_lines)
            cx_rect = (w - max_w) // 2
            
            cv2.rectangle(frame, (cx_rect - 15, y_pos), (cx_rect + max_w + 15, y_pos + pill_h), (0, 0, 180), -1)
            
            for i, line in enumerate(self._popup_lines):
                cx_text = get_center_x(line, font_scale)
                cv2.putText(frame, line, (cx_text, y_pos + 15 + i * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def build_critical_event(frame_data):
    event_time = datetime.fromtimestamp(frame_data.timestamp).strftime("%H:%M:%S")
    reliability_mode = str(frame_data.extra_metadata.get("reliability_mode", "NORMAL"))
    confidence = "MEDIUM" if reliability_mode == "LIMITED" else "HIGH"

    for alert in frame_data.alerts:
        if alert.get("debounced", False):
            continue

        alert_type = str(alert.get("type", ""))
        severity = str(alert.get("severity", "")).upper()

        if alert_type == "RESTRICTED_ENTRY":
            person_id = alert.get("person_id", "unknown")
            zone_id = alert.get("zone_id", "unknown")
            return {
                "type": "RESTRICTED ENTRY",
                "summary": f"Worker {person_id} entered restricted zone {zone_id} at {event_time}",
                "where": zone_id,
                "who": f"worker_{person_id}",
                "confidence": confidence,
                "action": "DISPATCH ROVER",
            }

        if alert_type == "ENVIRONMENT_ALERT":
            msg = str(alert.get("message", "")).upper()
            if "SMOKE" in msg or "FIRE" in msg:
                return {
                    "type": "FIRE/SMOKE",
                    "summary": f"Smoke/fire detected at {event_time} in camera view",
                    "where": "camera_view",
                    "who": "environment",
                    "confidence": confidence,
                    "action": "DISPATCH ROVER / EVACUATE",
                }
            continue

        if alert_type == "FALL_ALERT":
            person_id = alert.get("person_id", "unknown")
            zone = alert.get("zone_id", "unknown")
            return {
                "type": "FALL",
                "summary": f"Worker {person_id} fell in zone {zone} at {event_time}",
                "where": zone,
                "who": f"worker_{person_id}",
                "confidence": confidence,
                "action": "CHECK WORKER NOW",
            }

        if severity == "CRITICAL":
            person_id = alert.get("person_id", "unknown")
            zone_id = alert.get("zone_id", "unknown")
            return {
                "type": alert_type.replace("_", " ") if alert_type else "CRITICAL",
                "summary": f"Worker {person_id} critical alert in zone {zone_id} at {event_time}",
                "where": zone_id,
                "who": f"worker_{person_id}",
                "confidence": confidence,
                "action": "DISPATCH ROVER",
            }

    return None

def draw_corner_brackets(frame, xmin, ymin, xmax, ymax, color, thickness=2, length=15):
    # Top-left
    cv2.line(frame, (xmin, ymin), (xmin + length, ymin), color, thickness)
    cv2.line(frame, (xmin, ymin), (xmin, ymin + length), color, thickness)
    # Top-right
    cv2.line(frame, (xmax, ymin), (xmax - length, ymin), color, thickness)
    cv2.line(frame, (xmax, ymin), (xmax, ymin + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (xmin, ymax), (xmin + length, ymax), color, thickness)
    cv2.line(frame, (xmin, ymax), (xmin, ymax - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (xmax, ymax), (xmax - length, ymax), color, thickness)
    cv2.line(frame, (xmax, ymax), (xmax, ymax - length), color, thickness)


def render_annotations(frame_data):
    """Renders sleek, minimalist annotations grouped into a single HUD."""
    frame = frame_data.processed_frame
    h, w = frame.shape[:2]

    draw_zones_overlay(frame, frame_data)

    overlay = frame.copy()
    danger_area_count = 0

    for person in frame_data.persons:
        xmin, ymin, xmax, ymax = person.bbox
        zone_id = person.metadata.get("zone_id", "safe")
        has_yellow_vest = person.metadata.get("has_yellow_vest")
        zone_violations = list(person.metadata.get("zone_violations", []))
        optional_missing = list(person.metadata.get("zone_optional_missing", []))
        zone_forbidden = bool(person.metadata.get("zone_forbidden", False))

        is_safe = True
        color = (0, 200, 0)
        status = "SAFE"

        if person.is_fallen:
            is_safe = False
            color = (0, 0, 255)
            status = "FALL"
            danger_area_count += 1
        elif zone_forbidden or zone_id == "restricted":
            is_safe = False
            color = (0, 0, 255)
            status = "RESTRICTED: NO ENTRY"
            danger_area_count += 1
        elif zone_id == "work_floor":
            if zone_violations:
                is_safe = False
                color = (0, 140, 255)
                status = "WORK: MISSING " + "/".join(v.upper() for v in zone_violations)
                danger_area_count += 1
            else:
                status = "WORK SAFE (HELMET+GLASSES)"
                if "Vest" in optional_missing or has_yellow_vest is False:
                    status = "WORK SAFE (VEST OPTIONAL)"
        else:
            status = "SAFE ZONE"

        if is_safe:
            draw_corner_brackets(frame, xmin, ymin, xmax, ymax, color, thickness=1, length=10)
        else:
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)

        tag = f"{worker_label(person.person_id)} {status}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        text_y = max(th + 2, ymin - 2)
        cv2.rectangle(frame, (xmin, text_y - th - 4), (xmin + tw + 6, text_y + 2), color, -1)
        cv2.putText(
            frame,
            tag,
            (xmin + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    font_scale = 0.3
    y_top = 8
    x_left = 8

    fps = frame_data.extra_metadata.get("fps", 0)
    lat = frame_data.extra_metadata.get("latency_ms", 0)
    stats_text = f"LIVE: {frame_data.current_people_count} | FPS: {fps} | LAT: {lat}ms"
    (tw, th), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(frame, (x_left, y_top), (x_left + tw + 10, y_top + th + 10), (0, 0, 0), -1)
    cv2.putText(frame, stats_text, (x_left + 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

    x_left += tw + 15

    layout = frame_data.extra_metadata.get("zone_layout_label", "MODE: UNKNOWN")
    (tw, th), _ = cv2.getTextSize(layout, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(frame, (x_left, y_top), (x_left + tw + 10, y_top + th + 10), (0, 0, 0), -1)
    cv2.putText(frame, layout, (x_left + 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    x_right = w - 8

    verifying_count = int(frame_data.extra_metadata.get("verifying_count", 0) or 0)
    if verifying_count > 0:
        text = f"VERIFY {verifying_count}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x_right - tw - 10, y_top), (x_right, y_top + th + 10), (0, 200, 255), -1)
        cv2.putText(frame, text, (x_right - tw - 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        x_right -= (tw + 15)

    reliability_mode = str(frame_data.extra_metadata.get("reliability_mode", "NORMAL"))
    if reliability_mode == "LIMITED":
        text = "LIMITED"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x_right - tw - 10, y_top), (x_right, y_top + th + 10), (80, 80, 255), -1)
        cv2.putText(frame, text, (x_right - tw - 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        x_right -= (tw + 15)

    if frame_data.is_image_blurry:
        text = "BLUR/FOG"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x_right - tw - 10, y_top), (x_right, y_top + th + 10), (0, 140, 255), -1)
        cv2.putText(frame, text, (x_right - tw - 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        x_right -= (tw + 15)

    if frame_data.is_smoke_detected:
        text = "FIRE/SMOKE"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x_right - tw - 10, y_top), (x_right, y_top + th + 10), (0, 0, 255), -1)
        cv2.putText(frame, text, (x_right - tw - 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        x_right -= (tw + 15)

    if danger_area_count > 0:
        text = f"{danger_area_count} IN DANGER"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x_right - tw - 10, y_top), (x_right, y_top + th + 10), (0, 0, 255), -1)
        cv2.putText(frame, text, (x_right - tw - 5, y_top + th + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    visible_alerts = [a for a in frame_data.alerts if not a.get("debounced", False)]
    if visible_alerts:
        top = visible_alerts[0]
        alert_text = str(top.get("message", "ALERT"))
        if len(alert_text) > 70:
            alert_text = alert_text[:67] + "..."
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        y_alert = y_top + 22
        cv2.rectangle(frame, (8, y_alert), (18 + tw, y_alert + th + 10), (0, 0, 200), -1)
        cv2.putText(
            frame,
            alert_text,
            (12, y_alert + th + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    window_name = "MTU Room Monitor"
    parser = argparse.ArgumentParser(description="Hackatum safety monitoring computer vision pipeline.")
    parser.add_argument("--mock", action="store_true", help="Run with simulated warehouse video and inputs")
    parser.add_argument("--source", type=str, default=None, help="Video source (e.g. '0' for webcam, or path to file)")
    parser.add_argument(
        "--camera-profile",
        choices=["monitor", "rover"],
        default="monitor",
        help="Zone layout for fixed monitor cam vs rover-mounted cam (default: monitor)",
    )
    parser.add_argument(
        "--zones-file",
        type=str,
        default=None,
        help="Custom zones JSON path (overrides --camera-profile)",
    )
    args = parser.parse_args()

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    engine = SafetyPipelineEngine(
        use_mock=args.mock,
        video_source=source,
        camera_profile=args.camera_profile,
        zones_path=args.zones_file,
    )
    stream = engine.stream_frames()
    replay = SupervisorReplayRecorder()

    print("\n" + "=" * 50)
    print("MTU Pipeline Engine Active.")
    print("Press 'q' to quit.")
    print("=" * 50 + "\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1800, 1350)

    try:
        for frame_data in stream:
            render_annotations(frame_data)

            event = build_critical_event(frame_data)
            if event is not None:
                replay.trigger(frame_data.timestamp, event)

            replay.draw_overlay(frame_data.processed_frame, frame_data.timestamp)
            replay.add_frame(frame_data.timestamp, frame_data.processed_frame)

            if frame_data.alerts:
                for alert in frame_data.alerts:
                    if not alert.get("debounced", False):
                        print(
                            f"[{time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))}] "
                            f"[{alert['severity']}] {alert['message']}"
                        )

            cv2.imshow(window_name, frame_data.processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nShutdown command received. Closing stream.")
                break

    except KeyboardInterrupt:
        print("\nShutdown via keyboard interrupt.")
    finally:
        engine.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
