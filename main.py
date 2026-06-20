import cv2
import argparse
import time
import sys
import os
import json
from datetime import datetime
from collections import deque
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
        self._last_trigger_ts = 0.0
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
        if timestamp - self._last_trigger_ts < self.cooldown_seconds:
            return False

        pre_start = timestamp - self.pre_seconds
        pre_frames = [frame for ts, frame in self._pre_buffer if ts >= pre_start]

        self._active_event = {
            "event": event,
            "start_ts": timestamp,
            "end_ts": timestamp + self.post_seconds,
            "frames": pre_frames,
        }
        self._last_trigger_ts = timestamp

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

        # Normalize replay frames so codecs receive consistent 8-bit BGR images.
        norm_frames = []
        base_h, base_w = frames[0].shape[:2]
        base_w = int(base_w) - (int(base_w) % 2)
        base_h = int(base_h) - (int(base_h) % 2)
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
            norm_frames.append(out.copy())

        if not norm_frames:
            return

        height, width = norm_frames[0].shape[:2]
        duration = max(0.1, self.pre_seconds + self.post_seconds)
        fps = max(8.0, min(24.0, len(norm_frames) / duration))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_slug = event["type"].lower().replace("/", "_").replace(" ", "_")
        base_name = f"event_{stamp}_{event_slug}"
        avi_path = os.path.join(self.output_dir, f"{base_name}.avi")
        mp4_path = os.path.join(self.output_dir, f"{base_name}.mp4")
        meta_path = os.path.join(self.output_dir, f"{base_name}.json")

        # Primary export: MJPG AVI is broadly decodable on macOS and avoids green-frame artifacts.
        avi_writer = cv2.VideoWriter(
            avi_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (width, height),
        )

        if avi_writer.isOpened():
            for frame in norm_frames:
                avi_writer.write(frame)
            avi_writer.release()
            replay_path = avi_path
        else:
            replay_path = mp4_path

        # Secondary export: keep mp4 as optional compatibility artifact.
        mp4_writer = cv2.VideoWriter(
            mp4_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if mp4_writer.isOpened():
            for frame in norm_frames:
                mp4_writer.write(frame)
            mp4_writer.release()

        metadata = {
            "timestamp": timestamp,
            "event": event,
            "video_path": replay_path,
            "video_path_avi": avi_path,
            "video_path_mp4": mp4_path,
            "frame_count": len(norm_frames),
            "fps": fps,
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._saved_until_ts = timestamp + 3.0
        self._saved_text = f"REPLAY SAVED: {os.path.basename(replay_path)}"

    def draw_overlay(self, frame, timestamp):
        if timestamp <= self._popup_until_ts and self._popup_lines:
            line_height = 24
            panel_h = 18 + line_height * len(self._popup_lines)
            panel_w = max(420, int(frame.shape[1] * 0.55))
            cv2.rectangle(frame, (12, 90), (12 + panel_w, 90 + panel_h), (0, 0, 255), -1)
            for i, text in enumerate(self._popup_lines):
                y = 90 + 24 + i * line_height
                scale = 0.6 if i == 0 else 0.48
                thick = 2 if i == 0 else 1
                cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick)

        if timestamp <= self._saved_until_ts and self._saved_text:
            cv2.rectangle(frame, (12, 64), (620, 86), (0, 120, 0), -1)
            cv2.putText(frame, self._saved_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def build_critical_event(frame_data):
    event_time = datetime.fromtimestamp(frame_data.timestamp).strftime("%H:%M:%S")
    reliability_mode = str(frame_data.extra_metadata.get("reliability_mode", "NORMAL"))
    confidence = "MEDIUM" if reliability_mode == "LIMITED" else "HIGH"

    for alert in frame_data.alerts:
        alert_type = str(alert.get("type", ""))
        severity = str(alert.get("severity", "")).upper()

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

        if severity == "CRITICAL" or alert_type == "RESTRICTED_ENTRY":
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
    """Render only essential safety information with low visual noise."""
    frame = frame_data.processed_frame
    h, w = frame.shape[:2]

    overlay = frame.copy()
    danger_count = 0

    for person in frame_data.persons:
        xmin, ymin, xmax, ymax = person.bbox
        zone_id = person.metadata.get("zone_id", "safe")
        has_yellow_vest = person.metadata.get("has_yellow_vest")

        is_safe = True
        status = "SAFE"
        color = (40, 185, 40)

        if person.is_fallen:
            is_safe = False
            status = "FALL"
            color = (0, 0, 255)
        elif zone_id == "restricted":
            is_safe = False
            status = "RESTRICTED"
            color = (0, 0, 255)
        elif zone_id == "work_floor":
            violations = []
            if person.has_helmet is not True:
                violations.append("NO HELMET")
            if person.has_glasses is not True:
                violations.append("NO GLASSES")
            if has_yellow_vest is False:
                violations.append("NO VEST")
            if violations:
                is_safe = False
                status = "/".join(violations)
                color = (0, 140, 255)

        if not is_safe:
            danger_count += 1
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, -1)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
        label = f"{worker_label(person.person_id)} {status}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        ly = ymin - 6
        if ly - th - 6 < 0:
            ly = min(h - 4, ymax + th + 8)
        cv2.rectangle(frame, (xmin, ly - th - 5), (xmin + tw + 8, ly + 2), color, -1)
        cv2.putText(frame, label, (xmin + 4, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    stats = f"LIVE {frame_data.current_people_count} | TOTAL {frame_data.total_unique_people}"
    cv2.putText(frame, stats, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)

    if danger_count > 0:
        alert = f"ALERT {danger_count}"
        (aw, ah), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        cv2.rectangle(frame, (10, 24), (22 + aw, 26 + ah), (0, 0, 255), -1)
        cv2.putText(frame, alert, (16, 24 + ah - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    tags = []
    verifying_count = int(frame_data.extra_metadata.get("verifying_count", 0) or 0)
    if verifying_count > 0:
        tags.append((f"VERIFY {verifying_count}", (0, 200, 255)))
    reliability_mode = str(frame_data.extra_metadata.get("reliability_mode", "NORMAL"))
    if reliability_mode == "LIMITED":
        tags.append(("LIMITED MODE", (80, 80, 255)))
    if frame_data.is_smoke_detected:
        tags.append(("SMOKE", (0, 0, 255)))
    if frame_data.is_image_blurry:
        tags.append(("BLUR", (0, 140, 255)))
    x = w - 10
    for text, color in tags:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x - tw - 12, 8), (x, 12 + th), color, -1)
        cv2.putText(frame, text, (x - tw - 7, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x -= (tw + 18)

    fps = frame_data.extra_metadata.get("fps", 0)
    latency = frame_data.extra_metadata.get("latency_ms", 0)
    cv2.putText(frame, f"{fps} FPS | {latency} ms", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)


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

            display_frame = cv2.resize(
                frame_data.processed_frame,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LINEAR,
            )
            cv2.imshow(window_name, display_frame)

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
